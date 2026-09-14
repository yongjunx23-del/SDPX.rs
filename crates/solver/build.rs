use std::error::Error;
#[cfg(feature = "buildinfo")]
use vergen::*;
fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "buildinfo")]
    Emitter::default()
        .add_instructions(&BuildBuilder::all_build()?)?
        .add_instructions(&CargoBuilder::all_cargo()?)?
        .add_instructions(&RustcBuilder::all_rustc()?)?
        .emit()?;
    Ok(())
}
