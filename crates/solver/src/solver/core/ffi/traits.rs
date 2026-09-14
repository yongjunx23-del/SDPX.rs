/// trait for defining FFI data counterparts as associated types
#[allow(missing_docs)]
pub trait SolverFFI<I> {
    type FFI: From<I>;
}
