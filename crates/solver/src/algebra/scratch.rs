/// Per-thread reusable scratch, keyed by type: taken out for one call and put
/// back, so repeated calls of one shape make no Rust allocations (the
/// caller-owned-workspace contract) and reentrant calls stay safe: a nested
/// call, e.g. from a task stolen during `rayon::join`, gets its own value.
pub fn with_scratch<V: Default + 'static, R>(f: impl FnOnce(&mut V) -> R) -> R {
    use std::any::{Any, TypeId};
    use std::cell::RefCell;
    thread_local! {
        static SCRATCH: RefCell<Vec<Box<dyn Any>>> = const { RefCell::new(Vec::new()) };
    }
    let taken = SCRATCH.with(|slot| {
        let mut slot = slot.borrow_mut();
        slot.iter()
            .position(|b| (**b).type_id() == TypeId::of::<V>())
            .map(|i| slot.swap_remove(i))
    });
    let mut boxed: Box<V> = match taken.map(|b| b.downcast::<V>()) {
        Some(Ok(v)) => v,
        _ => Box::default(),
    };
    let result = f(&mut boxed);
    SCRATCH.with(|slot| slot.borrow_mut().push(boxed));
    result
}

/// Allocations from this size up are fresh mappings (the CLI's
/// `M_MMAP_THRESHOLD`): the first write places each of their pages.
const FIRST_TOUCH_BYTES: usize = 32 << 20;

/// `(0..len).map(f)` written by `pool`'s workers when the vector is a fresh
/// mapping (`None` otherwise). Pages land on the NUMA node of the worker
/// that first writes them, so an array the whole pool works on spreads over
/// the pool's nodes instead of the node of the thread that built it.
pub(crate) fn spread_vec<T: Send>(
    len: usize,
    f: impl Fn(usize) -> T + Sync + Send,
    pool: &rayon::ThreadPool,
) -> Option<Vec<T>> {
    use rayon::prelude::*;
    (pool.current_num_threads() > 1 && len * std::mem::size_of::<T>() >= FIRST_TOUCH_BYTES)
        .then(|| pool.install(|| (0..len).into_par_iter().map(f).collect()))
}
