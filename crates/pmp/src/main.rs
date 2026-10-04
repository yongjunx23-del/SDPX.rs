use sdpx_pmp::Result;
use std::path::PathBuf;
fn usage() {
    println!("sdpx-pmp2sdp --input PMP.json|PMP.xml --output NEW_DIRECTORY [--precision BITS] [--threads N]\n\nBITS: MPFR widths from {} (default 768).\nN: concurrent blocks (default 1); more workers use more memory.\nWrites uncompressed SDPB sampled JSON. Existing output paths are refused.", sdpx_arithmetic::FRONTEND_PRECISION_HELP);
}
macro_rules! dispatch {
    ( [] (53,$variant:ident,$scalar:ty), $(($bits:literal,$name:ident,$ty:ty),)* ) => {
        fn convert(bits:usize,input:&std::path::Path,output:&std::path::Path,threads:usize)->Result<usize> {
            match bits { $($bits => sdpx_pmp::convert_file::<$ty>(input,output,threads),)*
                _ => Err(format!("unsupported precision; use an MPFR width from: {}", sdpx_arithmetic::FRONTEND_PRECISION_HELP).into()), }
        }
    }
}
sdpx_arithmetic::with_frontend_precisions!(dispatch);
fn run() -> Result<()> {
    let mut input = None;
    let mut output = None;
    let mut bits = 768;
    let mut threads = 1;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                usage();
                return Ok(());
            }
            "--version" => {
                println!("sdpx-pmp2sdp {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            "--input" | "-i" => {
                if input.is_some() {
                    return Err("duplicate --input".into());
                }
                input = Some(PathBuf::from(args.next().ok_or("missing input path")?));
            }
            "--output" | "-o" => {
                if output.is_some() {
                    return Err("duplicate --output".into());
                }
                output = Some(PathBuf::from(args.next().ok_or("missing output path")?));
            }
            "--precision" | "-p" => bits = args.next().ok_or("missing precision")?.parse()?,
            "--threads" | "-t" => threads = args.next().ok_or("missing thread count")?.parse()?,
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    let input = input.ok_or("--input is required (see --help)")?;
    let output = output.ok_or("--output is required (see --help)")?;
    let blocks = convert(bits, &input, &output, threads)?;
    println!(
        "Converted {} blocks at {bits} bits to {}",
        blocks,
        output.display()
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("sdpx-pmp2sdp: {error}");
        std::process::exit(1);
    }
}
