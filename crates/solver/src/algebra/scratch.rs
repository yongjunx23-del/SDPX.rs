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
