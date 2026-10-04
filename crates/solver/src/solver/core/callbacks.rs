// ---------------------------------
// enum for managing callbacks
// ---------------------------------

pub trait TerminationCallback<I>: FnMut(&I) -> bool + Send + Sync {}
impl<I, T: FnMut(&I) -> bool + Send + Sync> TerminationCallback<I> for T {}

#[derive(Default)]
pub(crate) enum Callback<I> {
    #[default]
    None,
    Rust(Box<dyn TerminationCallback<I> + Send + Sync>),
}

impl<I> std::fmt::Debug for Callback<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Callback::None => write!(f, "Callback::None"),
            Callback::Rust(_) => write!(f, "Callback::Rust(<closure>)"),
        }
    }
}

impl<I> Callback<I>
where
    I: Clone + Sized,
{
    // Call the callback function
    fn call(&mut self, info: &I) -> bool {
        match self {
            Callback::None => false,
            Callback::Rust(f) => f(info),
        }
    }
}

#[derive(Debug)]
pub(crate) struct SolverCallbacks<I> {
    /// callback for termination
    pub termination_callback: Callback<I>,
    /// iterate checkpoint / restart requests
    pub checkpoint: crate::solver::core::checkpoint::CheckpointConfig,
}

impl<I> Default for SolverCallbacks<I> {
    // Create a new set of callbacks
    fn default() -> Self {
        Self {
            termination_callback: Callback::None,
            checkpoint: Default::default(),
        }
    }
}

impl<I> SolverCallbacks<I>
where
    I: Clone + Sized,
{
    pub(crate) fn check_termination(&mut self, info: &I) -> bool {
        // check termination conditions
        self.termination_callback.call(info)
    }
}
