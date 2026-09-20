//! Replay a lossless KKT snapshot (`SDPX_SNAPSHOT=<dir>`) through an LDL
//! backend and print a JSON audit report.
//!
//! Usage: kkt_replay <snap.bin> [direct_solve_method]   (default "auto")

use sdpx_arithmetic::MpFloat;
use sdpx_solver::algebra::FloatT;
use sdpx_solver::snapshot;
use sdpx_solver::solver::CoreSettings;
use std::path::Path;

fn run<T: FloatT>(path: &Path, method: &str) -> Result<(), String> {
    let snap = snapshot::load::<T>(path).map_err(|e| e.to_string())?;
    let mut settings = CoreSettings::<T>::default();
    settings.direct_solve_method = method.to_string();
    let rep = snapshot::replay(&snap, &settings)?;
    println!(
        concat!(
            "{{\"file\": {:?}, \"backend\": {:?}, \"refactored\": {}, ",
            "\"solved\": {}, \"normwise_residual\": {:.3e}, ",
            "\"max_abs_residual\": {:.3e}, \"bitwise_match\": {}, ",
            "\"factor_secs\": {:.4}, \"solve_secs\": {:.4}}}"
        ),
        path.file_name().unwrap().to_string_lossy(),
        rep.backend,
        rep.refactored,
        rep.solved,
        rep.normwise_residual,
        rep.max_abs_residual,
        rep.bitwise_match
            .map(|b| b.to_string())
            .unwrap_or_else(|| "null".into()),
        rep.factor_secs,
        rep.solve_secs,
    );
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: kkt_replay <snap.bin> [direct_solve_method]");
        std::process::exit(2);
    }
    let path = Path::new(&args[1]);
    let method = args.get(2).map(|s| s.as_str()).unwrap_or("auto");
    let prec = snapshot::header_precision(path).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2);
    });
    let r = match prec {
        53 => run::<f64>(path, method),
        128 => run::<MpFloat<2>>(path, method),
        256 => run::<MpFloat<4>>(path, method),
        512 => run::<MpFloat<8>>(path, method),
        768 => run::<MpFloat<12>>(path, method),
        1024 => run::<MpFloat<16>>(path, method),
        2048 => run::<MpFloat<32>>(path, method),
        p => Err(format!("unsupported precision {p}")),
    };
    if let Err(e) = r {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
