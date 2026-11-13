use sdpx_pmp::{post, Result};
use std::path::{Path, PathBuf};

const USAGE: &str = "sdpx-bootstrap — post-process an SDPX solution of an SDPB sampled SDP\n\
Usage:\n  sdpx-bootstrap spectrum --sdp DIR --solution FILE [--threshold T] [--subdivisions N]\n  \
sdpx-bootstrap sdpb-out --sdp DIR --solution FILE --out NEW_DIRECTORY\n  \
sdpx-bootstrap approx-objective --sdp NEW_DIR --base DIR --solution FILE\n  \
sdpx-bootstrap certify --sdp DIR --solution FILE\n\
FILE is the JSON result of `sdpx DIR`; its precision_bits selects the working precision.\n\
spectrum: zeros of the extremal functional on [0, last sample point] with lambda_min(F)/(|c|+sum|y_i B_i|) <= T\n\
  (default 1e-20), and weights solving the primal equations at those zeros (JSON to stdout).\n\
sdpb-out: SDPB's out.txt, y.txt, x_j.txt and X/Y_matrix_j_parity.txt (\"rows cols\" header).\n\
approx-objective: first-order objective of NEW_DIR at the base solution (same structure).\n\
certify: SDPB-style dual jump: minimal Gram correction for exact equalities, then Cholesky;\n  certified=true proves c - B y is a sum of squares (the point is excluded).";

macro_rules! dispatch {
    ([] $(($bits:literal, $name:ident, $ty:ty),)*) => {
        fn run_at(bits: usize, command: &str, a: &Args) -> Result<serde_json::Value> {
            match bits {
                $($bits => run::<$ty>(command, a),)*
                _ => Err(format!("unsupported precision {bits}; build with all-precisions").into()),
            }
        }
    };
}
sdpx_arithmetic::with_frontend_precisions!(dispatch);

#[derive(Default)]
struct Args {
    sdp: Option<PathBuf>,
    base: Option<PathBuf>,
    solution: Option<PathBuf>,
    out: Option<PathBuf>,
    threshold: Option<String>,
    subdivisions: Option<usize>,
}

fn run<T: sdpx_arithmetic::Scalar + std::str::FromStr>(
    command: &str,
    a: &Args,
) -> Result<serde_json::Value> {
    let need = |p: &Option<PathBuf>, name: &str| -> Result<PathBuf> {
        p.clone().ok_or_else(|| format!("missing {name}").into())
    };
    let (sdp, solution) = (need(&a.sdp, "--sdp")?, need(&a.solution, "--solution")?);
    match command {
        "spectrum" => post::spectrum::<T>(
            &sdp,
            &solution,
            a.threshold.as_deref().unwrap_or("1e-20"),
            a.subdivisions.unwrap_or(16),
        ),
        "sdpb-out" => post::sdpb_out::<T>(&sdp, &solution, &need(&a.out, "--out")?),
        "certify" => post::certify::<T>(&sdp, &solution),
        "approx-objective" => {
            post::approx_objective::<T>(&need(&a.base, "--base")?, &sdp, &solution)
        }
        _ => Err(format!("unknown command {command}\n{USAGE}").into()),
    }
}

fn main_inner() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let command = args.next().ok_or(USAGE)?;
    if matches!(command.as_str(), "--help" | "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    let mut a = Args::default();
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--sdp" => a.sdp = Some(value.into()),
            "--base" => a.base = Some(value.into()),
            "--solution" => a.solution = Some(value.into()),
            "--out" => a.out = Some(value.into()),
            "--threshold" => a.threshold = Some(value),
            "--subdivisions" => a.subdivisions = Some(value.parse()?),
            _ => return Err(format!("unknown option {flag}\n{USAGE}").into()),
        }
    }
    let solution = a.solution.clone().ok_or("missing --solution")?;
    let bits = precision(&solution)?;
    let result = run_at(bits, &command, &a)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn precision(path: &Path) -> Result<usize> {
    // The result header precedes the point; read only that field.
    let v: serde_json::Value =
        serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(path)?))?;
    Ok(v["precision_bits"]
        .as_u64()
        .ok_or("solution has no precision_bits")? as usize)
}

fn main() -> std::process::ExitCode {
    match main_inner() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("sdpx-bootstrap: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
