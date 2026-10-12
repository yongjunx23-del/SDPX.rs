//! Internal utility functions and helpers.

// assorted other functionality missing from std

// a drop-in replacement for the julia "findall" function,
// which serves as a vectorized version of the std::iter::position
// returning indices of *all* elements satisfying a predicate

use num_traits::Num;
use std::cmp::Ordering;
use std::iter::zip;

pub(crate) trait PositionAll<T>: Iterator<Item = T> {
    fn position_all<F>(&mut self, predicate: F) -> Vec<usize>
    where
        F: FnMut(&T) -> bool;
}

impl<T, I> PositionAll<T> for I
where
    I: Iterator<Item = T>,
{
    fn position_all<F>(&mut self, mut f: F) -> Vec<usize>
    where
        F: FnMut(&T) -> bool,
    {
        self.enumerate()
            .filter(|(_, item)| f(item))
            .map(|(index, _)| index)
            .collect::<Vec<_>>()
    }
}

// permutation and inverse permutation
// functions that require no allocation
// p must be a valid permutation vector
// in both cases for safety

pub(crate) fn permute<T: Copy>(x: &mut [T], b: &[T], p: &[usize]) {
    debug_assert!(p.is_empty() || *p.iter().max().unwrap() < x.len());
    unsafe {
        zip(p, x).for_each(|(p, x)| *x = *b.get_unchecked(*p));
    }
}

pub(crate) fn ipermute<T: Copy>(x: &mut [T], b: &[T], p: &[usize]) {
    debug_assert!(p.is_empty() || *p.iter().max().unwrap() < x.len());
    unsafe {
        zip(p, b).for_each(|(p, b)| *x.get_unchecked_mut(*p) = *b);
    }
}

// Construct an inverse permutation from a permutation
pub(crate) fn invperm(p: &[usize]) -> Vec<usize> {
    let mut b = vec![0; p.len()];
    for (i, j) in p.iter().enumerate() {
        assert!(*j < p.len() && b[*j] == 0);
        b[*j] = i;
    }
    b
}

pub(crate) fn sortperm_rev<T>(p: &mut [usize], v: &[T])
where
    T: Sized + Ord + Copy,
{
    assert_eq!(p.len(), v.len());
    sortperm_by(p, v, |&a, &b| b.cmp(&a));
}

pub(crate) fn sortperm_by<T, F>(p: &mut [usize], v: &[T], compare: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    assert_eq!(p.len(), v.len());
    p.iter_mut().enumerate().for_each(|(i, p)| *p = i);
    let mut f = compare;
    p.sort_by(|&i, &j| f(&v[i], &v[j]));
}

// maybe should be a vector trait, but T needs to admit
// non-float types (e.g. usize).  Would require partition of the
// vector math traits into those that require FloatT and those
// that only require Num + Ord.
pub(crate) fn findmax<T>(v: &[T]) -> Option<usize>
where
    T: Num + Copy + Ord,
{
    v.iter()
        .enumerate()
        .max_by_key(|(_, &value)| value)
        .map(|(idx, _)| idx)
}

// -------------
// testing

#[test]
fn test_position_all() {
    let test = [3, 1, 0, 5, 9];
    let idx = test.iter().position_all(|&v| *v > 2);
    assert_eq!(idx, vec![0, 3, 4]);

    let idx: Vec<usize> = test.iter().position_all(|&v| *v == 2);
    assert_eq!(idx, Vec::<usize>::new());
}

#[test]
fn test_permute() {
    let mut x = vec![0; 5];
    let b = [6, 7, 8, 9, 10];
    let p = [2, 4, 1, 3, 0];

    permute(&mut x, &b, &p);

    assert_eq!(x, [8, 10, 7, 9, 6]);
}

#[test]
fn test_ipermute() {
    let mut x = vec![0; 5];
    let b = [8, 10, 7, 9, 6];
    let p = [2, 4, 1, 3, 0];

    ipermute(&mut x, &b, &p);

    assert_eq!(x, [6, 7, 8, 9, 10]);
}

#[test]
fn test_sortperm_by() {
    let mut p = vec![0usize; 6];
    let v: Vec<isize> = vec![3, 1, 1, 5, 0, 6];
    let ptarg: Vec<usize> = vec![5, 3, 0, 1, 2, 4];

    // sort in descending order should preserve the
    // order of the 1's.   This is different from applying
    // sortperm and then reversing the result.
    sortperm_by(&mut p, &v, |&a, &b| b.cmp(&a));

    for i in 0..v.len() {
        assert_eq!(p[i], ptarg[i]);
    }
}

std::thread_local! {
    /// The [`run_queue`] whose item this thread runs (its address), or 0.
    static QUEUE_ITEM: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// A queued job: estimated seconds, size (tie break) and the work.
pub(crate) type Job<'a> = (f64, usize, Box<dyn FnOnce() + Send + 'a>);

/// Run `jobs` on the current pool as a queue: each of `workers` items takes
/// the largest remaining job (estimated seconds, then size), so the longest
/// jobs start first and the shortest fill the tail. An item started nested
/// in an item of the same queue (stolen by a thread waiting inside a split
/// job's inner join) returns at once instead of taking a whole job onto
/// that job's critical path; the outer item still drains the queue. Items
/// of another queue (owner kernels sharing one pool) run normally, or that
/// queue's jobs could be left unrun.
pub(crate) fn run_queue(mut jobs: Vec<Job<'_>>, workers: usize) {
    use rayon::prelude::*;
    // Ascending, so `pop` takes the largest.
    jobs.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1)));
    let jobs = std::sync::Mutex::new(jobs);
    let queue = &jobs as *const _ as usize;
    (0..workers.max(1)).into_par_iter().for_each(|_| {
        let outer = QUEUE_ITEM.with(|c| c.get());
        if outer == queue {
            return;
        }
        QUEUE_ITEM.with(|c| c.set(queue));
        struct Leave(usize);
        impl Drop for Leave {
            fn drop(&mut self) {
                QUEUE_ITEM.with(|c| c.set(self.0));
            }
        }
        let _leave = Leave(outer);
        loop {
            let Some((_, _, job)) = jobs.lock().unwrap().pop() else {
                break;
            };
            job();
        }
    });
}

/// Independent jobs that another phase's idle workers take, largest first
/// (as [`run_queue`] orders them); [`TailJobs::finish`] runs any left over.
pub(crate) struct TailJobs<'a> {
    jobs: std::sync::Mutex<Vec<Job<'a>>>,
}

impl<'a> TailJobs<'a> {
    pub(crate) fn new(mut jobs: Vec<Job<'a>>) -> Self {
        jobs.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1)));
        Self {
            jobs: std::sync::Mutex::new(jobs),
        }
    }

    /// Run the largest remaining job; false when none is left.
    pub(crate) fn run_one(&self) -> bool {
        let job = self.jobs.lock().unwrap().pop();
        job.map(|(_, _, job)| job()).is_some()
    }

    /// Run the remaining jobs as a queue on the current pool.
    pub(crate) fn finish(self, workers: usize) {
        let jobs = self.jobs.into_inner().unwrap();
        if !jobs.is_empty() {
            run_queue(jobs, workers);
        }
    }
}

#[test]
fn test_run_queue_nested_queue_runs_its_jobs() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    // One worker: the outer item's thread also runs every inner item, as an
    // owner kernel's queue does when started inside another owner's job.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let ran = AtomicUsize::new(0);
    pool.install(|| {
        let outer: Vec<Job<'_>> = (0..2)
            .map(|i| {
                let ran = &ran;
                let job = move || {
                    let inner: Vec<Job<'_>> = (0..3)
                        .map(|j| {
                            let job = move || {
                                ran.fetch_add(1, Ordering::Relaxed);
                            };
                            (j as f64, j, Box::new(job) as Box<dyn FnOnce() + Send + '_>)
                        })
                        .collect();
                    run_queue(inner, 4);
                };
                (i as f64, i, Box::new(job) as Box<dyn FnOnce() + Send + '_>)
            })
            .collect();
        run_queue(outer, 4);
    });
    assert_eq!(ran.load(Ordering::Relaxed), 6);
}
