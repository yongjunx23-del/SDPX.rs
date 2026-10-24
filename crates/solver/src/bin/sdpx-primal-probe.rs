//! Explicit diagnostic: same-precision feasible-primal barrier continuation.
use sdpx_arithmetic::{with_frontend_precisions, FRONTEND_PRECISION_HELP};
use sdpx_solver::{algebra::FloatT, solver::JsonProblem};
use serde::{de::DeserializeOwned, Serialize};
use std::{fs::File, io::BufReader, path::Path};

fn run<T: FloatT + Serialize + DeserializeOwned>(
    args: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut p = JsonProblem::<T>::read(BufReader::new(File::open(&args[1])?))?;
    p.settings.max_threads = args[7].parse()?;
    let start: serde_json::Value = serde_json::from_reader(BufReader::new(File::open(&args[2])?))?;
    if start["precision_bits"].as_u64() != Some(T::precision_bits() as u64) {
        return Err("starting iterate must have the requested precision".into());
    }
    let x: Vec<T> = serde_json::from_value(start["x"].clone())?;
    let mu = T::from_str_radix(&args[6], 10).map_err(|_| "invalid mu")?;
    sdpx_solver::solver::primal_probe::run(p, x, mu, args[5].parse()?, Path::new(&args[3]))
}

macro_rules! precision_runner {
    ([] $(($bits:literal, $variant:ident, $scalar:ty),)*) => {
        fn run_precision(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
            match args[4].parse::<usize>()? {
                $($bits => run::<$scalar>(args),)*
                _ => Err(format!("unsupported precision; use {FRONTEND_PRECISION_HELP}").into()),
            }
        }
    };
}
with_frontend_precisions!(precision_runner);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 8 {
        return Err("usage: sdpx-primal-probe CASE START OUTPUT BITS MAX_ITER MU THREADS".into());
    }
    let threads = args[7].parse::<usize>()?;
    if threads == 0 {
        return Err("THREADS must be positive".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    run_precision(&args)
}
