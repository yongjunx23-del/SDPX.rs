//! Native entry point; all numerical work stays in the shared solver.
#[cfg(not(feature = "all-precisions"))]
use sdpx_arithmetic::{
    with_default_precisions as with_frontend_precisions,
    DEFAULT_FRONTEND_PRECISION_HELP as FRONTEND_PRECISION_HELP,
};
#[cfg(feature = "all-precisions")]
use sdpx_arithmetic::{with_frontend_precisions, FRONTEND_PRECISION_HELP};
use sdpx_solver::{algebra::FloatT, io::ConfigurablePrintTarget, solver::*, MpiContext};
use serde::{de::DeserializeOwned, Serialize};
use sha2::{Digest, Sha256};
use std::{
    env,
    error::Error,
    fs::File,
    io::{self, BufReader, BufWriter, Write},
    path::PathBuf,
    process::ExitCode,
    str::FromStr,
    time::Instant,
};

const USAGE: &str = "SDPX — native conic solver\n\
Usage: sdpx INPUT [--precision BITS] [--settings FILE] [--output FILE]\n\
                     [--threads N] [--partitions N|auto]\n\
                     [--cost-history-in FILE] [--cost-history-out FILE] [--quiet]\n\
BITS defaults to 53 (Float64); see the compiled precision list below.\n\
INPUT is conic JSON, an SDPB sampled JSON directory (requires sdp), or '-'\n\
for stdin. MPFR coefficients use decimal strings, never fractional\n\
JSON numbers. --settings replaces input settings; unspecified settings use core\n\
defaults. Progress goes to stderr, one JSON result to stdout or --output.\n\
With MPI, --partitions auto or a count equal to the world size enables one\n\
owner per rank; the default without --partitions keeps the ordinary MPI path.\n\
--cost-history-in and --cost-history-out are supported for partitioned MPI\n\
solves; export gathers owner timings to rank zero.\n\
Exit: 0 fully solved/infeasible; 2 incomplete/reduced accuracy; 1 input/I/O error.";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum PartitionMode {
    Auto,
    Count(usize),
}

#[derive(Default)]
struct Options {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    settings: Option<PathBuf>,
    precision: Option<usize>,
    threads: Option<u32>,
    partitions: Option<PartitionMode>,
    cost_history_in: Option<PathBuf>,
    cost_history_out: Option<PathBuf>,
    quiet: bool,
}

// Borrow large solution vectors so writing a point doesn't allocate a second
// decimal-string copy of every high-precision entry.
#[derive(Serialize)]
struct ResultRecord<'a, T: Serialize> {
    version: &'static str,
    precision_bits: usize,
    status: String,
    iterations: u32,
    objective: Option<T>,
    dual_objective: Option<T>,
    primal_residual: Option<T>,
    dual_residual: Option<T>,
    dual_componentwise_residual: Option<T>,
    x: &'a [T],
    s: &'a [T],
    z: &'a [T],
    native_seconds: f64,
    api_seconds: f64,
    load_seconds: f64,
    threads_requested: u32,
    cone_threads: usize,
    partitions: usize,
    linear_solver: &'a str,
    linear_solver_threads: usize,
    kkt_form: &'static str,
    settings: serde_json::Value,
    original_objective: Option<T>,
    original_dual_objective: Option<T>,
    sampled_y: Option<Vec<T>>,
    sampled: Option<serde_json::Value>,
    mpi_world_size: usize,
    output_rank: usize,
}

type CliResult<T> = Result<T, Box<dyn Error>>;

enum RuntimeSolver<T: FloatT> {
    Default(DefaultSolver<T>),
    #[cfg(feature = "sdp")]
    Partitioned(PartitionedSolver<T>),
}

impl<T: FloatT> RuntimeSolver<T> {
    fn solve(&mut self) {
        match self {
            Self::Default(solver) => solver.solve(),
            #[cfg(feature = "sdp")]
            Self::Partitioned(solver) => solver.solve(),
        }
    }

    fn print_to_stream(&mut self, stream: Box<dyn Write + Send + Sync>) {
        match self {
            Self::Default(solver) => solver.print_to_stream(stream),
            #[cfg(feature = "sdp")]
            Self::Partitioned(solver) => solver.print_to_stream(stream),
        }
    }

    fn print_to_sink(&mut self) {
        match self {
            Self::Default(solver) => solver.print_to_sink(),
            #[cfg(feature = "sdp")]
            Self::Partitioned(solver) => solver.print_to_sink(),
        }
    }

    fn solution(&self) -> &DefaultSolution<T> {
        match self {
            Self::Default(solver) => &solver.solution,
            #[cfg(feature = "sdp")]
            Self::Partitioned(solver) => solver.solution(),
        }
    }

    fn info(&self) -> &DefaultInfo<T> {
        match self {
            Self::Default(solver) => &solver.info,
            #[cfg(feature = "sdp")]
            Self::Partitioned(solver) => solver.info(),
        }
    }

    fn settings(&self) -> &DefaultSettings<T> {
        match self {
            Self::Default(solver) => solver.settings(),
            #[cfg(feature = "sdp")]
            Self::Partitioned(solver) => solver.settings(),
        }
    }

    fn cone_threads(&self) -> usize {
        match self {
            Self::Default(solver) => solver.cones.cone_threads(),
            #[cfg(feature = "sdp")]
            Self::Partitioned(solver) => solver.cone_threads(),
        }
    }

    fn partitions(&self) -> usize {
        match self {
            Self::Default(_) => 1,
            #[cfg(feature = "sdp")]
            Self::Partitioned(solver) => solver.partitions(),
        }
    }

    fn try_write_receipt(&self, peak_rss: Option<u64>) -> io::Result<()> {
        match self {
            Self::Default(solver) => sdpx_solver::receipt::try_write(solver, peak_rss),
            #[cfg(feature = "sdp")]
            Self::Partitioned(solver) => {
                sdpx_solver::receipt::try_write_partitioned(solver, peak_rss)
            }
        }
    }

    #[cfg(feature = "sdp")]
    fn cost_history(&self) -> CliResult<Option<CostHistory>> {
        match self {
            Self::Default(_) => Ok(None),
            Self::Partitioned(solver) => solver.cost_history().map_err(Into::into),
        }
    }
}

// Keep MPI stage control at the input/output boundary. All numerical work
// still uses the same solver as a serial caller.
fn stage<T>(mpi: MpiContext, result: CliResult<T>) -> CliResult<T> {
    if mpi.all_succeeded(result.is_ok()) {
        result
    } else {
        Err(result
            .err()
            .unwrap_or_else(|| "another MPI rank failed this stage".into()))
    }
}

struct Identity(Sha256);
impl Write for Identity {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn agree<T: Serialize>(mpi: MpiContext, value: &T) -> CliResult<()> {
    if mpi.size() == 1 {
        return Ok(());
    }
    let mut identity = Identity(Sha256::new());
    stage(
        mpi,
        serde_json::to_writer(&mut identity, value).map_err(Into::into),
    )?;
    if !mpi.agree_signature(identity.0.finalize().into()) {
        return Err("MPI ranks disagree on input or configuration".into());
    }
    Ok(())
}

fn parse_options<I>(args: I, rank: usize) -> CliResult<Option<Options>>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let mut out = Options::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                if rank == 0 {
                    println!("{USAGE}\nSupported precisions: {FRONTEND_PRECISION_HELP}");
                }
                return Ok(None);
            }
            "--version" | "-V" => {
                if rank == 0 {
                    println!("SDPX {}", env!("CARGO_PKG_VERSION"));
                }
                return Ok(None);
            }
            "--quiet" | "-q" => out.quiet = true,
            "--precision" | "--threads" | "--settings" | "--output" | "--partitions"
            | "--cost-history-in" | "--cost-history-out" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for {arg}"))?;
                match arg.as_str() {
                    "--precision" => out.precision = Some(value.parse()?),
                    "--threads" => out.threads = Some(value.parse()?),
                    "--settings" => out.settings = Some(value.into()),
                    "--output" => out.output = Some(value.into()),
                    "--partitions" => {
                        out.partitions = Some(if value == "auto" {
                            PartitionMode::Auto
                        } else {
                            let count: usize = value.parse()?;
                            if count == 0 {
                                return Err("--partitions must be greater than zero".into());
                            }
                            PartitionMode::Count(count)
                        });
                    }
                    "--cost-history-in" => out.cost_history_in = Some(value.into()),
                    "--cost-history-out" => out.cost_history_out = Some(value.into()),
                    _ => unreachable!(),
                }
            }
            "-" => {
                if out.input.replace(arg.into()).is_some() {
                    return Err("only one input is supported".into());
                }
            }
            _ if arg.starts_with('-') => return Err(format!("unknown option {arg}").into()),
            _ => {
                if out.input.replace(arg.into()).is_some() {
                    return Err("only one input is supported".into());
                }
            }
        }
    }
    if out.input.is_none() {
        return Err(format!(
            "missing input\n{USAGE}\nSupported precisions: {FRONTEND_PRECISION_HELP}"
        )
        .into());
    }
    Ok(Some(out))
}

fn options(mpi: MpiContext) -> CliResult<Option<Options>> {
    parse_options(env::args().skip(1), mpi.rank())
}

fn validate_partition_mode(partitions: Option<PartitionMode>, mpi_size: usize) -> CliResult<()> {
    let Some(partitions) = partitions else {
        return Ok(());
    };
    if partitions == PartitionMode::Count(0) {
        return Err("--partitions must be greater than zero".into());
    }
    #[cfg(not(feature = "sdp"))]
    {
        let _ = mpi_size;
        return Err("--partitions requires a build with an SDP backend".into());
    }
    #[cfg(feature = "sdp")]
    {
        if mpi_size > 1 {
            match partitions {
                PartitionMode::Auto => Ok(()),
                PartitionMode::Count(count) if count == mpi_size => Ok(()),
                PartitionMode::Count(_) => {
                    Err("with MPI, --partitions N must equal the MPI world size".into())
                }
            }
        } else {
            Ok(())
        }
    }
}

fn validate_cost_history_options(options: &Options) -> CliResult<()> {
    if (options.cost_history_in.is_some() || options.cost_history_out.is_some())
        && options.partitions.is_none()
    {
        return Err("--cost-history requires --partitions N|auto".into());
    }
    #[cfg(not(feature = "sdp"))]
    if options.cost_history_in.is_some() || options.cost_history_out.is_some() {
        return Err("--cost-history requires a build with an SDP backend".into());
    }
    Ok(())
}

fn run<T: FloatT + Serialize + DeserializeOwned + FromStr>(
    options: Options,
    mpi: MpiContext,
) -> CliResult<u8> {
    validate_partition_mode(options.partitions, mpi.size())?;
    validate_cost_history_options(&options)?;
    let start = Instant::now();
    let path = options.input.as_ref().unwrap();
    let partitions = options.partitions;
    #[cfg(feature = "sdp")]
    let cost_history = stage(
        mpi,
        (|| -> CliResult<Option<CostHistory>> {
            if let Some(path) = &options.cost_history_in {
                Ok(Some(CostHistory::read(BufReader::new(File::open(path)?))?))
            } else {
                Ok(None)
            }
        })(),
    )?;
    #[cfg(feature = "sdp")]
    agree(mpi, &(&cost_history, options.cost_history_out.is_some()))?;
    #[allow(unused_mut)]
    let mut objective_constant = T::zero();
    #[allow(unused_mut)]
    let mut sampled_metadata = None::<serde_json::Value>;
    #[allow(unused_mut)]
    let mut equalities = 0;
    let problem = stage(
        mpi,
        (|| -> CliResult<JsonProblem<T>> {
            let mut problem = if path.is_dir() {
                #[cfg(feature = "sdp")]
                {
                    let sampled = read_sdpb_sampled::<T>(path)?;
                    sdpx_solver::receipt::memory_mark("input read");
                    objective_constant = sampled.objective_constant;
                    equalities = sampled.num_equalities;
                    sampled_metadata = Some(serde_json::json!({
                        "objective_constant": objective_constant,
                        "num_equalities": equalities, "grams": sampled.grams,
                    }));
                    sampled.problem
                }
                #[cfg(not(feature = "sdp"))]
                return Err("SDPB sampled input requires a build with an SDP backend".into());
            } else if path.as_os_str() == "-" {
                if mpi.size() > 1 {
                    return Err("MPI stdin is not supported; use a shared input file".into());
                }
                JsonProblem::<T>::read(io::stdin().lock())?
            } else {
                JsonProblem::<T>::read(BufReader::new(File::open(path)?))?
            };
            if let Some(path) = &options.settings {
                problem.settings = serde_json::from_reader(BufReader::new(File::open(path)?))?;
                // Match the shared conic JSON encoding of an unlimited time limit.
                if problem.settings.time_limit == f64::MAX {
                    problem.settings.time_limit = f64::INFINITY;
                }
            }
            if let Some(threads) = options.threads {
                problem.settings.max_threads = threads;
            }
            if options.quiet {
                problem.settings.verbose = false;
            }
            Ok(problem)
        })(),
    )?;
    // Hash authoritative parsed values and effective settings without allocating
    // another full serialized problem. Recovery metadata is part of identity.
    agree(
        mpi,
        &(&problem, &sampled_metadata, T::precision_bits(), partitions),
    )?;
    let load_seconds = start.elapsed().as_secs_f64();
    let solve_start = Instant::now();
    let mut solver = stage(
        mpi,
        (|| -> CliResult<RuntimeSolver<T>> {
            if let Some(partitions) = partitions {
                #[cfg(feature = "sdp")]
                {
                    let solver = match partitions {
                        PartitionMode::Auto => problem
                            .into_auto_partitioned_solver_with_cost_history(CostHistoryOptions {
                                history: cost_history,
                                record: options.cost_history_out.is_some(),
                            })?,
                        PartitionMode::Count(count) => problem
                            .into_partitioned_solver_with_cost_history(
                                count,
                                CostHistoryOptions {
                                    history: cost_history,
                                    record: options.cost_history_out.is_some(),
                                },
                            )?,
                    };
                    return Ok(RuntimeSolver::Partitioned(solver));
                }
                #[cfg(not(feature = "sdp"))]
                {
                    let _ = partitions;
                    unreachable!("partitioned mode was rejected before setup");
                }
            }
            Ok(RuntimeSolver::Default(problem.into_solver()?))
        })(),
    )?;
    if mpi.rank() == 0 {
        solver.print_to_stream(Box::new(io::stderr()));
    } else {
        solver.print_to_sink();
    }
    solver.solve();
    let api_seconds = solve_start.elapsed().as_secs_f64();
    #[cfg(feature = "sdp")]
    if let Some(path) = &options.cost_history_out {
        stage(
            mpi,
            (|| -> CliResult<()> {
                let history = solver.cost_history()?;
                if mpi.rank() == 0 {
                    history
                        .ok_or("cost-history training was not enabled for this solver")?
                        .write(File::create(path)?)?;
                }
                Ok(())
            })(),
        )?;
    }
    let s = solver.solution();
    agree(mpi, &(s.status as u32, s.iterations))?;
    stage(
        mpi,
        if mpi.rank() == 0 || std::env::var_os("SDPX_RECEIPT_ALL_RANKS").is_some() {
            solver
                .try_write_receipt(sdpx_solver::receipt::peak_rss_bytes())
                .map_err(Into::into)
        } else {
            Ok(())
        },
    )?;
    let complete = matches!(
        s.status,
        SolverStatus::Solved | SolverStatus::PrimalInfeasible | SolverStatus::DualInfeasible
    );
    stage(
        mpi,
        (|| -> CliResult<()> {
            if mpi.rank() != 0 {
                return Ok(());
            }
            let mut receipt_settings = solver.settings().clone();
            if receipt_settings.time_limit.is_infinite() {
                receipt_settings.time_limit = f64::MAX;
            }
            let result = ResultRecord {
                version: env!("CARGO_PKG_VERSION"),
                precision_bits: T::precision_bits(),
                status: format!("{:?}", s.status),
                iterations: s.iterations,
                objective: s.obj_val.is_finite().then_some(s.obj_val),
                dual_objective: s.obj_val_dual.is_finite().then_some(s.obj_val_dual),
                primal_residual: s.r_prim.is_finite().then_some(s.r_prim),
                dual_residual: s.r_dual.is_finite().then_some(s.r_dual),
                dual_componentwise_residual: solver
                    .info()
                    .res_dual_componentwise
                    .filter(|v| v.is_finite()),
                x: &s.x,
                s: &s.s,
                z: &s.z,
                native_seconds: s.solve_time,
                api_seconds,
                load_seconds,
                threads_requested: solver.settings().max_threads,
                cone_threads: solver.cone_threads(),
                partitions: solver.partitions(),
                linear_solver: &solver.info().linsolver.name,
                linear_solver_threads: solver.info().linsolver.threads,
                kkt_form: if solver.info().linsolver.name == "owned_condensed"
                    || solver.info().linsolver.name.starts_with("condensed_")
                {
                    "condensed"
                } else {
                    "augmented"
                },
                // Only the small settings object is materialized as JSON; the point
                // remains streamed. This object can be reused with --settings.
                settings: serde_json::to_value(receipt_settings)?,
                original_objective: sampled_metadata
                    .as_ref()
                    .filter(|_| s.obj_val.is_finite())
                    .map(|_| objective_constant + s.obj_val),
                original_dual_objective: sampled_metadata
                    .as_ref()
                    .filter(|_| s.obj_val_dual.is_finite())
                    .map(|_| objective_constant + s.obj_val_dual),
                sampled_y: sampled_metadata
                    .as_ref()
                    .map(|_| s.z[..equalities].iter().map(|z| -*z).collect()),
                sampled: sampled_metadata,
                mpi_world_size: mpi.size(),
                output_rank: mpi.rank(),
            };
            let mut output: Box<dyn Write> = match options.output {
                Some(path) => Box::new(BufWriter::new(File::create(path)?)),
                None => Box::new(io::stdout().lock()),
            };
            serde_json::to_writer(&mut output, &result)?;
            writeln!(output)?;
            output.flush()?;
            Ok(())
        })(),
    )?;
    Ok(if complete { 0 } else { 2 })
}

/// Keep the exact kernels' multi-megabyte scratch buffers on the heap: with
/// glibc's default threshold each is an `mmap`/`munmap` pair, and every unmap
/// in a many-threaded process costs a TLB shootdown (about 5% of an MPFR SDP
/// solve at 32 threads). Explicit threshold or padding environment settings
/// leave allocator tuning to glibc.
fn tune_allocator() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        let configured = [
            "MALLOC_MMAP_THRESHOLD_",
            "MALLOC_TRIM_THRESHOLD_",
            "MALLOC_TOP_PAD_",
        ]
        .iter()
        .any(|name| std::env::var_os(name).is_some());
        if !configured {
            // SAFETY: mallopt only adjusts allocator parameters; it is called
            // once, before any worker threads exist.
            unsafe {
                libc::mallopt(libc::M_MMAP_THRESHOLD, 32 << 20);
                libc::mallopt(libc::M_TRIM_THRESHOLD, 256 << 20);
                libc::mallopt(libc::M_TOP_PAD, 64 << 20);
            }
        }
    }
}

macro_rules! precision_runner {
    ([] $(($bits:literal, $variant:ident, $scalar:ty),)*) => {
        fn run_precision(options: Options, mpi: MpiContext) -> CliResult<u8> {
            match options.precision.unwrap_or(53) {
                $($bits => run::<$scalar>(options, mpi),)*
                _ => Err(format!("unsupported precision; use {FRONTEND_PRECISION_HELP}").into()),
            }
        }
    };
}
with_frontend_precisions!(precision_runner);

fn main() -> ExitCode {
    tune_allocator();
    let mpi = MpiContext::initialize();
    if mpi.size() > 1 {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            previous(info);
            mpi.abort("panic in native MPI application");
        }));
    }
    let backend_threads: Vec<_> = [
        "RAYON_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]
    .map(|name| (name, env::var(name).ok()))
    .into_iter()
    .collect();
    let result = agree(
        mpi,
        &(
            env::args().skip(1).collect::<Vec<_>>(),
            backend_threads,
            env::var_os("SDPX_RNS_OPS").is_some(),
            env::var_os("SDPX_SERIAL_QDLDL").is_some(),
        ),
    )
    .and_then(|_| stage(mpi, options(mpi)))
    .and_then(|options| match options {
        None => Ok(0),
        Some(options) => run_precision(options, mpi),
    });
    mpi.finish();
    match result {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("sdpx (rank {}): {error}", mpi.rank());
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn partitions_requires_a_positive_count() {
        let options = parse_options(args(&["input.json", "--partitions", "3"]), 0)
            .unwrap()
            .unwrap();
        assert_eq!(options.partitions, Some(PartitionMode::Count(3)));
        assert!(parse_options(args(&["input.json", "--partitions", "0"]), 0).is_err());
        assert!(parse_options(args(&["input.json", "--partitions", "many"]), 0).is_err());
    }

    #[test]
    fn partitions_requires_an_argument() {
        assert!(parse_options(args(&["input.json", "--partitions"]), 0).is_err());
    }

    #[test]
    fn partitions_accepts_structural_auto() {
        let options = parse_options(args(&["input.json", "--partitions", "auto"]), 0)
            .unwrap()
            .unwrap();
        assert_eq!(options.partitions, Some(PartitionMode::Auto));
        #[cfg(feature = "sdp")]
        assert!(validate_partition_mode(Some(PartitionMode::Auto), 2).is_ok());
        #[cfg(not(feature = "sdp"))]
        assert!(validate_partition_mode(Some(PartitionMode::Auto), 2)
            .unwrap_err()
            .to_string()
            .contains("SDP"));
    }

    #[test]
    fn partitions_reject_mpi_before_setup() {
        let error = validate_partition_mode(Some(PartitionMode::Count(3)), 2)
            .unwrap_err()
            .to_string();
        #[cfg(feature = "sdp")]
        assert!(error.contains("world size"));
        #[cfg(not(feature = "sdp"))]
        assert!(error.contains("SDP"));
    }

    #[test]
    fn partitions_accept_matching_mpi_world_size() {
        #[cfg(feature = "sdp")]
        assert!(validate_partition_mode(Some(PartitionMode::Count(2)), 2).is_ok());
        #[cfg(not(feature = "sdp"))]
        assert!(validate_partition_mode(Some(PartitionMode::Count(2)), 2).is_err());
    }

    #[test]
    fn cost_history_paths_require_partitioned_mode() {
        let options = parse_options(
            args(&[
                "input.json",
                "--partitions",
                "auto",
                "--cost-history-in",
                "cost.json",
                "--cost-history-out",
                "trained.json",
            ]),
            0,
        )
        .unwrap()
        .unwrap();
        assert_eq!(options.partitions, Some(PartitionMode::Auto));
        assert!(options.cost_history_in.is_some());
        assert!(options.cost_history_out.is_some());
        let default = parse_options(
            args(&["input.json", "--cost-history-out", "trained.json"]),
            0,
        )
        .unwrap()
        .unwrap();
        assert!(validate_cost_history_options(&default).is_err());
    }

    #[test]
    fn mpi_history_training_and_input_are_allowed() {
        let mut training = parse_options(
            args(&[
                "input.json",
                "--partitions",
                "auto",
                "--cost-history-out",
                "trained.json",
            ]),
            0,
        )
        .unwrap()
        .unwrap();
        #[cfg(feature = "sdp")]
        assert!(validate_cost_history_options(&training).is_ok());
        #[cfg(not(feature = "sdp"))]
        assert!(validate_cost_history_options(&training).is_err());

        training.cost_history_out = None;
        training.cost_history_in = Some("history.json".into());
        #[cfg(feature = "sdp")]
        assert!(validate_cost_history_options(&training).is_ok());
        #[cfg(not(feature = "sdp"))]
        assert!(validate_cost_history_options(&training).is_err());
    }
}
