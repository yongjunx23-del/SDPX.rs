//! Canonical collectives used by owner-local numerical phases.
//!
//! Numerical values are gathered through the [`Scalar`] wire codec provided by
//! [`crate::mpi::World`] and folded in rank order.  Native MPI reductions are
//! intentionally not used: MPFR values do not have an MPI datatype and a
//! reduction tree would make the rounding order depend on the communicator.
//!
//! The owner-local path performs all collectives from its solver/control thread.
//! Rayon workers may prepare local contributions, but they must join before
//! calling this interface.  `MockCollective` is a small in-process transport
//! for exercising that contract without starting MPI.  No full block-data
//! allgather is exposed: callers exchange only bounded scalar vectors needed
//! by coupled phases, while final solution assembly uses the explicit
//! root-only gather operation below.

use sdpx_arithmetic::Scalar;
#[cfg(test)]
use std::collections::BTreeMap;
#[cfg(test)]
use std::sync::{Arc, Condvar, Mutex};
#[cfg(test)]
use std::thread::{self, ThreadId};

fn ranges_exact(ranges: &[(usize, usize)], total: usize) -> bool {
    let mut next = 0usize;
    for &(offset, length) in ranges {
        if offset != next {
            return false;
        }
        let Some(end) = offset.checked_add(length) else {
            return false;
        };
        next = end;
    }
    next == total
}

/// A failure observed before a collective can publish a numerical result.
///
/// `WrongThread` is returned by the in-process transport.  The MPI adapter
/// aborts the MPI world for the same violation, because returning on one rank
/// while peers are entering the collective would deadlock the job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CollectiveError {
    /// A rank called a collective from a thread other than its bound control
    /// thread. Constructed by the in-process mock transport used in tests.
    #[cfg(test)]
    WrongThread,
    /// Ranks supplied different vector lengths to one numerical reduction.
    LengthMismatch,
    /// Ranks entered different operations at one collective sequence point.
    OperationMismatch,
    /// Ranks entered the same sequence point with different site identifiers.
    SiteMismatch,
    /// A rank entered the same sequence point more than once.
    #[cfg(test)]
    DuplicateContribution,
    /// The mock group was constructed with no ranks.
    #[cfg(test)]
    EmptyGroup,
}

/// The minimal transport needed by owner-local HSD phases.
///
/// Each rank must call every method in the same order.  `site` identifies the
/// numerical stream (for example, the border Schur or residual stream).  The
/// mock tracks a per-handle sequence, while the MPI adapter agrees on the
/// site and operation before entering payload exchange.  Implementations
/// return a consensus error instead of publishing a partial reduction.
pub(crate) trait Collective<T: Scalar>: Send + Sync {
    /// This rank's zero-based communicator rank.
    fn rank(&self) -> usize;
    /// Number of ranks participating in the communicator.
    fn size(&self) -> usize;
    /// Return true only when every rank supplied `value=true`.
    fn all_true(&self, site: usize, value: bool) -> Result<bool, CollectiveError>;
    /// Return true only when every rank supplied the same decision code.
    fn agree_u32(&self, site: usize, value: u32) -> Result<bool, CollectiveError>;
    /// Sum equal-length vectors in rank order and return the result on every
    /// rank.  Empty vectors are valid when every rank contributes empty data.
    fn reduce_sum(&self, site: usize, local: &[T]) -> Result<Vec<T>, CollectiveError>;
    /// Sum a persistent workspace without a temporary copy in serial mode.
    fn reduce_sum_in_place(&self, site: usize, values: &mut [T]) -> Result<(), CollectiveError> {
        let global = self.reduce_sum(site, values)?;
        if global.len() != values.len() {
            return Err(CollectiveError::LengthMismatch);
        }
        values.copy_from_slice(&global);
        Ok(())
    }
    /// Take the elementwise maximum in rank order and return it on every rank.
    fn reduce_max(&self, site: usize, local: &[T]) -> Result<Vec<T>, CollectiveError>;
    /// Concatenate equal-length vectors in rank order on every rank, so a
    /// caller can fold several reductions of one round in its own order.
    fn all_gather(&self, site: usize, local: &[T]) -> Result<Vec<T>, CollectiveError>;
    /// Gather rank-local vectors only on `root`. `ranges` gives each rank's
    /// destination interval in canonical rank order; non-root ranks return
    /// `Ok(None)` and do not receive the assembled payload.
    fn gather_root(
        &self,
        site: usize,
        local: &[T],
        ranges: &[(usize, usize)],
        root: usize,
    ) -> Result<Option<Vec<T>>, CollectiveError>;
}

/// The ordinary one-process transport.  It returns an owned vector just like
/// the distributed implementations so callers can use one generic path.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SerialCollective;

impl<T: Scalar> Collective<T> for SerialCollective {
    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        1
    }

    fn all_true(&self, _site: usize, value: bool) -> Result<bool, CollectiveError> {
        Ok(value)
    }

    fn agree_u32(&self, _site: usize, _value: u32) -> Result<bool, CollectiveError> {
        Ok(true)
    }

    fn reduce_sum(&self, _site: usize, local: &[T]) -> Result<Vec<T>, CollectiveError> {
        Ok(local.to_vec())
    }

    fn reduce_sum_in_place(&self, _site: usize, _values: &mut [T]) -> Result<(), CollectiveError> {
        Ok(())
    }

    fn reduce_max(&self, _site: usize, local: &[T]) -> Result<Vec<T>, CollectiveError> {
        Ok(local.to_vec())
    }

    fn all_gather(&self, _site: usize, local: &[T]) -> Result<Vec<T>, CollectiveError> {
        Ok(local.to_vec())
    }

    fn gather_root(
        &self,
        _site: usize,
        local: &[T],
        ranges: &[(usize, usize)],
        root: usize,
    ) -> Result<Option<Vec<T>>, CollectiveError> {
        if root != 0 || ranges.len() != 1 || ranges[0] != (0, local.len()) {
            return Err(CollectiveError::LengthMismatch);
        }
        Ok(Some(local.to_vec()))
    }
}

/// A process-local MPI adapter.  The actual MPI world remains optional and is
/// loaded lazily by [`crate::mpi::World`].
#[derive(Clone, Copy)]
pub(crate) struct WorldCollective(pub(crate) crate::mpi::World);

impl WorldCollective {
    fn check_round(&self, site: usize, operation: Operation) -> Result<(), CollectiveError> {
        self.handshake(site, operation, None, None).map(|_| ())
    }

    /// Agree on the site, operation and (optionally) a vector length, and
    /// carry an optional small payload, in one max-allreduce of paired
    /// `(x, -x)` entries (max and -min). Errors keep the order of the former
    /// one-check-per-collective sequence: site, operation, then length.
    /// Returns the payload's (max, min) over ranks.
    fn handshake(
        &self,
        site: usize,
        operation: Operation,
        len: Option<usize>,
        payload: Option<u32>,
    ) -> Result<(f64, f64), CollectiveError> {
        let started = std::time::Instant::now();
        self.0.ensure_collective_thread();
        let mapped = self.0.collective_site(site);
        let site_ok = u32::try_from(mapped).is_ok() && self.0.valid_collective_site(mapped);
        let len_ok =
            len.is_none_or(|l| u32::try_from(l).is_ok() && l.checked_mul(self.0.size()).is_some());
        let s = if site_ok { mapped as f64 } else { -1.0 };
        let op = f64::from(operation.code());
        let l = len.map_or(0.0, |l| l as f64);
        let p = f64::from(payload.unwrap_or(0));
        let mut v = [
            f64::from(u8::from(!site_ok)),
            f64::from(u8::from(!len_ok)),
            s,
            -s,
            op,
            -op,
            l,
            -l,
            p,
            -p,
        ];
        self.0.allreduce_max_f64_slice(&mut v);
        crate::receipt::site_record(site, started.elapsed());
        if v[0] != 0.0 || v[2] != -v[3] {
            return Err(CollectiveError::SiteMismatch);
        }
        if v[4] != -v[5] {
            return Err(CollectiveError::OperationMismatch);
        }
        if v[1] != 0.0 || v[6] != -v[7] {
            return Err(CollectiveError::LengthMismatch);
        }
        Ok((v[8], -v[9]))
    }

    fn ranges_for(
        world: &crate::mpi::World,
        local_len: usize,
    ) -> Result<(Vec<(usize, usize)>, usize), CollectiveError> {
        let total = local_len
            .checked_mul(world.size())
            .ok_or(CollectiveError::LengthMismatch)?;
        Ok((crate::mpi::ranges(total, world.size()), total))
    }

    /// Payload exchange after a handshake that agreed on `local.len()`.
    fn gather<T: Scalar>(&self, site: usize, local: &[T]) -> Result<Vec<T>, CollectiveError> {
        self.0.ensure_collective_thread();
        let (ranges, total) = Self::ranges_for(&self.0, local.len())?;
        let mut all = vec![T::zero(); total];
        self.0
            .gather_slice(self.0.collective_site(site), local, &ranges, &mut all);
        Ok(all)
    }

    fn fold_sum<T: Scalar>(all: &[T], size: usize, len: usize) -> Vec<T> {
        if len == 0 {
            return Vec::new();
        }
        let mut out = all[..len].to_vec();
        for rank in 1..size {
            let segment = &all[rank * len..(rank + 1) * len];
            for (dst, &value) in out.iter_mut().zip(segment) {
                *dst += value;
            }
        }
        out
    }

    fn fold_max<T: Scalar>(all: &[T], size: usize, len: usize) -> Vec<T> {
        if len == 0 {
            return Vec::new();
        }
        let mut out = all[..len].to_vec();
        for rank in 1..size {
            let segment = &all[rank * len..(rank + 1) * len];
            for (dst, &value) in out.iter_mut().zip(segment) {
                *dst = if dst.is_nan() || value.is_nan() {
                    T::nan()
                } else {
                    dst.max(value)
                };
            }
        }
        out
    }
}

impl<T: Scalar> Collective<T> for WorldCollective {
    fn rank(&self) -> usize {
        self.0.rank()
    }

    fn size(&self) -> usize {
        self.0.size()
    }

    fn all_true(&self, site: usize, value: bool) -> Result<bool, CollectiveError> {
        let (max, _) = self.handshake(site, Operation::AllTrue, None, Some(u32::from(!value)))?;
        Ok(max == 0.0)
    }

    fn agree_u32(&self, site: usize, value: u32) -> Result<bool, CollectiveError> {
        let (max, min) = self.handshake(site, Operation::AgreeU32, None, Some(value))?;
        Ok(max == min)
    }

    fn reduce_sum(&self, site: usize, local: &[T]) -> Result<Vec<T>, CollectiveError> {
        self.handshake(site, Operation::Sum, Some(local.len()), None)?;
        let all = self.gather(site, local)?;
        Ok(Self::fold_sum(&all, self.0.size(), local.len()))
    }

    fn reduce_max(&self, site: usize, local: &[T]) -> Result<Vec<T>, CollectiveError> {
        self.handshake(site, Operation::Max, Some(local.len()), None)?;
        let all = self.gather(site, local)?;
        Ok(Self::fold_max(&all, self.0.size(), local.len()))
    }

    fn all_gather(&self, site: usize, local: &[T]) -> Result<Vec<T>, CollectiveError> {
        self.handshake(site, Operation::AllGather, Some(local.len()), None)?;
        self.gather(site, local)
    }

    fn gather_root(
        &self,
        site: usize,
        local: &[T],
        ranges: &[(usize, usize)],
        root: usize,
    ) -> Result<Option<Vec<T>>, CollectiveError> {
        self.check_round(site, Operation::GatherRoot)?;
        let root = u32::try_from(root).ok();
        let root_ok = root.is_some() && (root.unwrap_or(0) as usize) < self.0.size();
        if !self.0.all_true(root_ok) {
            return Err(CollectiveError::SiteMismatch);
        }
        let root = root.expect("root conversion was agreed by all ranks") as usize;
        if !self.0.agree_u32(root as u32) {
            return Err(CollectiveError::SiteMismatch);
        }
        if !self.0.all_true(ranges.len() == self.0.size()) {
            return Err(CollectiveError::LengthMismatch);
        }
        let mut out_len = 0usize;
        let mut valid = true;
        for &(offset, length) in ranges {
            let Some(end) = offset.checked_add(length) else {
                valid = false;
                break;
            };
            out_len = out_len.max(end);
        }
        if !self.0.all_true(valid) {
            return Err(CollectiveError::LengthMismatch);
        }
        if !self.0.all_true(ranges_exact(ranges, out_len)) {
            return Err(CollectiveError::LengthMismatch);
        }
        Ok(self
            .0
            .gather_slice_root(self.0.collective_site(site), local, ranges, root, out_len))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Operation {
    AllTrue,
    AgreeU32,
    Sum,
    Max,
    GatherRoot,
    AllGather,
}

impl Operation {
    fn code(self) -> u32 {
        match self {
            Self::AllTrue => 1,
            Self::AgreeU32 => 2,
            Self::Sum => 3,
            Self::Max => 4,
            Self::GatherRoot => 5,
            Self::AllGather => 6,
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RoundKey {
    site: usize,
    operation: Operation,
    root: usize,
}

#[cfg(test)]
#[derive(Clone, Debug)]
enum RoundResult<T: Scalar> {
    Bool(bool),
    Agree(bool),
    Values(Vec<T>),
    RootValues(Vec<T>),
    Error(CollectiveError),
}

#[cfg(test)]
struct Round<T: Scalar> {
    key: RoundKey,
    layout: Option<Vec<(usize, usize)>>,
    vectors: Vec<Option<Vec<T>>>,
    bools: Vec<Option<bool>>,
    decisions: Vec<Option<u32>>,
    result: Option<RoundResult<T>>,
    released: usize,
}

#[cfg(test)]
struct MockState<T: Scalar> {
    size: usize,
    rounds: BTreeMap<usize, Round<T>>,
    fatal: Option<CollectiveError>,
}

/// A deterministic in-process collective group.
///
/// Construct a group with [`MockCollective::group`], move one handle to each
/// rank thread, and call methods in identical order.  The first call on a
/// handle binds it to that thread; later calls from another thread publish a
/// group-wide `WrongThread` failure and wake peers before returning.  This
/// makes accidental worker calls visible without leaving another rank behind
/// an incomplete barrier.
#[cfg(test)]
pub(crate) struct MockCollective<T: Scalar> {
    rank: usize,
    size: usize,
    state: Arc<(Mutex<MockState<T>>, Condvar)>,
    owner: Arc<Mutex<Option<ThreadId>>>,
    sequence: Mutex<usize>,
}

#[cfg(test)]
impl<T: Scalar> MockCollective<T> {
    /// Create one handle per rank in a deterministic in-process group.
    pub(crate) fn group(size: usize) -> Result<Vec<Self>, CollectiveError> {
        if size == 0 {
            return Err(CollectiveError::EmptyGroup);
        }
        let state = Arc::new((
            Mutex::new(MockState {
                size,
                rounds: BTreeMap::new(),
                fatal: None,
            }),
            Condvar::new(),
        ));
        Ok((0..size)
            .map(|rank| Self {
                rank,
                size,
                state: Arc::clone(&state),
                owner: Arc::new(Mutex::new(None)),
                sequence: Mutex::new(0),
            })
            .collect())
    }

    fn check_thread(&self) -> Result<(), CollectiveError> {
        let current = thread::current().id();
        let mut owner = self.owner.lock().expect("mock collective owner poisoned");
        let result = match *owner {
            Some(bound) if bound != current => Err(CollectiveError::WrongThread),
            Some(_) => Ok(()),
            None => {
                *owner = Some(current);
                Ok(())
            }
        };
        drop(owner);
        if let Err(error) = result {
            self.fail_group(error);
        }
        result
    }

    fn fail_group(&self, error: CollectiveError) {
        let (lock, wake) = &*self.state;
        let mut state = lock.lock().expect("mock collective state poisoned");
        if state.fatal.is_none() {
            state.fatal = Some(error);
        }
        wake.notify_all();
    }

    fn next_sequence(&self) -> usize {
        let mut sequence = self.sequence.lock().expect("mock sequence poisoned");
        let current = *sequence;
        *sequence += 1;
        current
    }

    fn vector_round(
        &self,
        site: usize,
        operation: Operation,
        local: &[T],
    ) -> Result<Vec<T>, CollectiveError> {
        self.check_thread()?;
        let sequence = self.next_sequence();
        let (lock, wake) = &*self.state;
        let mut state = lock.lock().expect("mock collective state poisoned");
        let size = state.size;
        if let Some(error) = state.fatal {
            return Err(error);
        }
        {
            let round = state.rounds.entry(sequence).or_insert_with(|| Round {
                key: RoundKey {
                    site,
                    operation,
                    root: usize::MAX,
                },
                layout: None,
                vectors: vec![None; size],
                bools: Vec::new(),
                decisions: Vec::new(),
                result: None,
                released: 0,
            });
            if round.result.is_none() {
                if round.key.site != site {
                    round.result = Some(RoundResult::Error(CollectiveError::SiteMismatch));
                } else if round.key.operation != operation {
                    round.result = Some(RoundResult::Error(CollectiveError::OperationMismatch));
                } else if round.vectors[self.rank].is_some() {
                    round.result = Some(RoundResult::Error(CollectiveError::DuplicateContribution));
                } else {
                    round.vectors[self.rank] = Some(local.to_vec());
                }
                if round.result.is_none() && round.vectors.iter().all(Option::is_some) {
                    let lengths = round.vectors.iter().map(|v| v.as_ref().map_or(0, Vec::len));
                    let first = lengths.clone().next().unwrap_or(0);
                    if lengths.into_iter().any(|len| len != first) {
                        round.result = Some(RoundResult::Error(CollectiveError::LengthMismatch));
                    } else {
                        let values: Vec<Vec<T>> = round
                            .vectors
                            .iter()
                            .map(|v| v.clone().expect("all vector contributions present"))
                            .collect();
                        let mut result = values[0].clone();
                        if operation == Operation::AllGather {
                            result = values.concat();
                        }
                        for segment in values
                            .iter()
                            .skip(1)
                            .filter(|_| operation != Operation::AllGather)
                        {
                            match operation {
                                Operation::Sum => {
                                    for (dst, &value) in result.iter_mut().zip(segment) {
                                        *dst += value;
                                    }
                                }
                                Operation::Max => {
                                    for (dst, &value) in result.iter_mut().zip(segment) {
                                        *dst = if dst.is_nan() || value.is_nan() {
                                            T::nan()
                                        } else {
                                            dst.max(value)
                                        };
                                    }
                                }
                                _ => unreachable!("vector round has a numerical operation"),
                            }
                        }
                        round.result = Some(RoundResult::Values(result));
                    }
                }
            }
        }
        if let Some(RoundResult::Error(error)) = state
            .rounds
            .get(&sequence)
            .and_then(|round| round.result.as_ref())
        {
            state.fatal = Some(*error);
        }
        wake.notify_all();
        loop {
            if let Some(error) = state.fatal {
                return Err(error);
            }
            let result = state
                .rounds
                .get(&sequence)
                .and_then(|round| round.result.clone());
            if let Some(result) = result {
                let remove = {
                    let round = state
                        .rounds
                        .get_mut(&sequence)
                        .expect("mock vector round disappeared before release");
                    round.released += 1;
                    round.released == size
                };
                if remove {
                    state.rounds.remove(&sequence);
                }
                return match result {
                    RoundResult::Values(values) => Ok(values),
                    RoundResult::Error(error) => Err(error),
                    _ => Err(CollectiveError::OperationMismatch),
                };
            }
            state = wake.wait(state).expect("mock collective wait poisoned");
        }
    }

    fn bool_round(
        &self,
        site: usize,
        operation: Operation,
        value: bool,
    ) -> Result<bool, CollectiveError> {
        self.check_thread()?;
        let sequence = self.next_sequence();
        let (lock, wake) = &*self.state;
        let mut state = lock.lock().expect("mock collective state poisoned");
        let size = state.size;
        if let Some(error) = state.fatal {
            return Err(error);
        }
        {
            let round = state.rounds.entry(sequence).or_insert_with(|| Round {
                key: RoundKey {
                    site,
                    operation,
                    root: usize::MAX,
                },
                layout: None,
                vectors: Vec::new(),
                bools: vec![None; size],
                decisions: Vec::new(),
                result: None,
                released: 0,
            });
            if round.result.is_none() {
                if round.key.site != site {
                    round.result = Some(RoundResult::Error(CollectiveError::SiteMismatch));
                } else if round.key.operation != operation {
                    round.result = Some(RoundResult::Error(CollectiveError::OperationMismatch));
                } else if round.bools[self.rank].is_some() {
                    round.result = Some(RoundResult::Error(CollectiveError::DuplicateContribution));
                } else {
                    round.bools[self.rank] = Some(value);
                }
                if round.result.is_none() && round.bools.iter().all(Option::is_some) {
                    let all_true = round
                        .bools
                        .iter()
                        .all(|v| v.expect("all boolean contributions present"));
                    round.result = Some(RoundResult::Bool(all_true));
                }
            }
        }
        if let Some(RoundResult::Error(error)) = state
            .rounds
            .get(&sequence)
            .and_then(|round| round.result.as_ref())
        {
            state.fatal = Some(*error);
        }
        wake.notify_all();
        loop {
            if let Some(error) = state.fatal {
                return Err(error);
            }
            let result = state
                .rounds
                .get(&sequence)
                .and_then(|round| round.result.clone());
            if let Some(result) = result {
                let remove = {
                    let round = state
                        .rounds
                        .get_mut(&sequence)
                        .expect("mock boolean round disappeared before release");
                    round.released += 1;
                    round.released == size
                };
                if remove {
                    state.rounds.remove(&sequence);
                }
                return match result {
                    RoundResult::Bool(value) => Ok(value),
                    RoundResult::Error(error) => Err(error),
                    _ => Err(CollectiveError::OperationMismatch),
                };
            }
            state = wake.wait(state).expect("mock collective wait poisoned");
        }
    }

    fn agree_round(&self, site: usize, value: u32) -> Result<bool, CollectiveError> {
        self.check_thread()?;
        let sequence = self.next_sequence();
        let (lock, wake) = &*self.state;
        let mut state = lock.lock().expect("mock collective state poisoned");
        let size = state.size;
        if let Some(error) = state.fatal {
            return Err(error);
        }
        {
            let round = state.rounds.entry(sequence).or_insert_with(|| Round {
                key: RoundKey {
                    site,
                    operation: Operation::AgreeU32,
                    root: usize::MAX,
                },
                layout: None,
                vectors: Vec::new(),
                bools: Vec::new(),
                decisions: vec![None; size],
                result: None,
                released: 0,
            });
            if round.result.is_none() {
                if round.key.site != site {
                    round.result = Some(RoundResult::Error(CollectiveError::SiteMismatch));
                } else if round.key.operation != Operation::AgreeU32 {
                    round.result = Some(RoundResult::Error(CollectiveError::OperationMismatch));
                } else if round.decisions[self.rank].is_some() {
                    round.result = Some(RoundResult::Error(CollectiveError::DuplicateContribution));
                } else {
                    round.decisions[self.rank] = Some(value);
                }
                if round.result.is_none() && round.decisions.iter().all(Option::is_some) {
                    let first = round.decisions[0].expect("all decisions present");
                    let agrees = round
                        .decisions
                        .iter()
                        .all(|v| v.expect("all decisions present") == first);
                    round.result = Some(RoundResult::Agree(agrees));
                }
            }
        }
        if let Some(RoundResult::Error(error)) = state
            .rounds
            .get(&sequence)
            .and_then(|round| round.result.as_ref())
        {
            state.fatal = Some(*error);
        }
        wake.notify_all();
        loop {
            if let Some(error) = state.fatal {
                return Err(error);
            }
            let result = state
                .rounds
                .get(&sequence)
                .and_then(|round| round.result.clone());
            if let Some(result) = result {
                let remove = {
                    let round = state
                        .rounds
                        .get_mut(&sequence)
                        .expect("mock decision round disappeared before release");
                    round.released += 1;
                    round.released == size
                };
                if remove {
                    state.rounds.remove(&sequence);
                }
                return match result {
                    RoundResult::Agree(value) => Ok(value),
                    RoundResult::Error(error) => Err(error),
                    _ => Err(CollectiveError::OperationMismatch),
                };
            }
            state = wake.wait(state).expect("mock collective wait poisoned");
        }
    }

    fn gather_root_round(
        &self,
        site: usize,
        local: &[T],
        ranges: &[(usize, usize)],
        root: usize,
    ) -> Result<Option<Vec<T>>, CollectiveError> {
        self.check_thread()?;
        let sequence = self.next_sequence();
        let (lock, wake) = &*self.state;
        let mut state = lock.lock().expect("mock collective state poisoned");
        let size = state.size;
        if let Some(error) = state.fatal {
            return Err(error);
        }
        {
            let round = state.rounds.entry(sequence).or_insert_with(|| Round {
                key: RoundKey {
                    site,
                    operation: Operation::GatherRoot,
                    root,
                },
                layout: Some(ranges.to_vec()),
                vectors: vec![None; size],
                bools: Vec::new(),
                decisions: Vec::new(),
                result: None,
                released: 0,
            });
            if round.result.is_none() {
                let expected = ranges.get(self.rank).copied();
                if round.key.site != site {
                    round.result = Some(RoundResult::Error(CollectiveError::SiteMismatch));
                } else if round.key.operation != Operation::GatherRoot {
                    round.result = Some(RoundResult::Error(CollectiveError::OperationMismatch));
                } else if round.key.root != root || round.layout.as_deref() != Some(ranges) {
                    round.result = Some(RoundResult::Error(CollectiveError::SiteMismatch));
                } else if root >= size || ranges.len() != size {
                    round.result = Some(RoundResult::Error(CollectiveError::LengthMismatch));
                } else if !ranges_exact(
                    ranges,
                    ranges
                        .last()
                        .and_then(|&(offset, length)| offset.checked_add(length))
                        .unwrap_or(0),
                ) {
                    round.result = Some(RoundResult::Error(CollectiveError::LengthMismatch));
                } else if expected.is_none_or(|(_, length)| length != local.len()) {
                    round.result = Some(RoundResult::Error(CollectiveError::LengthMismatch));
                } else if round.vectors[self.rank].is_some() {
                    round.result = Some(RoundResult::Error(CollectiveError::DuplicateContribution));
                } else {
                    round.vectors[self.rank] = Some(local.to_vec());
                }
                if round.result.is_none() && round.vectors.iter().all(Option::is_some) {
                    let mut out_len = 0usize;
                    let mut valid = true;
                    for &(offset, length) in ranges {
                        let Some(end) = offset.checked_add(length) else {
                            valid = false;
                            break;
                        };
                        out_len = out_len.max(end);
                    }
                    if !valid {
                        round.result = Some(RoundResult::Error(CollectiveError::LengthMismatch));
                    } else {
                        let mut output = vec![T::zero(); out_len];
                        for (rank, values) in round.vectors.iter().enumerate() {
                            let (offset, length) = ranges[rank];
                            let values = values
                                .as_ref()
                                .expect("all root-gather contributions present");
                            if values.len() != length {
                                round.result =
                                    Some(RoundResult::Error(CollectiveError::LengthMismatch));
                                break;
                            }
                            output[offset..offset + length].copy_from_slice(values);
                        }
                        if round.result.is_none() {
                            round.result = Some(RoundResult::RootValues(output));
                        }
                    }
                }
            }
        }
        if let Some(RoundResult::Error(error)) = state
            .rounds
            .get(&sequence)
            .and_then(|round| round.result.as_ref())
        {
            state.fatal = Some(*error);
        }
        wake.notify_all();
        loop {
            if let Some(error) = state.fatal {
                return Err(error);
            }
            let result = state.rounds.get(&sequence).and_then(|round| {
                round.result.as_ref().map(|result| match result {
                    RoundResult::RootValues(values) if self.rank == root => {
                        Ok(Some(values.clone()))
                    }
                    RoundResult::RootValues(_) => Ok(None),
                    RoundResult::Error(error) => Err(*error),
                    _ => Err(CollectiveError::OperationMismatch),
                })
            });
            if let Some(result) = result {
                let remove = {
                    let round = state
                        .rounds
                        .get_mut(&sequence)
                        .expect("mock root-gather round disappeared before release");
                    round.released += 1;
                    round.released == size
                };
                if remove {
                    state.rounds.remove(&sequence);
                }
                return result;
            }
            state = wake.wait(state).expect("mock collective wait poisoned");
        }
    }
}

#[cfg(test)]
impl<T: Scalar> Collective<T> for MockCollective<T> {
    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.size
    }

    fn all_true(&self, site: usize, value: bool) -> Result<bool, CollectiveError> {
        self.bool_round(site, Operation::AllTrue, value)
    }

    fn agree_u32(&self, site: usize, value: u32) -> Result<bool, CollectiveError> {
        self.agree_round(site, value)
    }

    fn reduce_sum(&self, site: usize, local: &[T]) -> Result<Vec<T>, CollectiveError> {
        self.vector_round(site, Operation::Sum, local)
    }

    fn reduce_max(&self, site: usize, local: &[T]) -> Result<Vec<T>, CollectiveError> {
        self.vector_round(site, Operation::Max, local)
    }

    fn all_gather(&self, site: usize, local: &[T]) -> Result<Vec<T>, CollectiveError> {
        self.vector_round(site, Operation::AllGather, local)
    }

    fn gather_root(
        &self,
        site: usize,
        local: &[T],
        ranges: &[(usize, usize)],
        root: usize,
    ) -> Result<Option<Vec<T>>, CollectiveError> {
        self.gather_root_round(site, local, ranges, root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sdpx_arithmetic::{Bits256, Scalar};
    use std::thread;

    #[test]
    fn serial_is_identity() {
        let comm = SerialCollective;
        let values = [1.25f64, -3.5, 8.75];
        assert_eq!(comm.reduce_sum(0, &values).unwrap(), values);
        let mut workspace = values;
        comm.reduce_sum_in_place(0, &mut workspace).unwrap();
        assert_eq!(workspace, values);
        assert_eq!(comm.reduce_max(1, &values).unwrap(), values);
        assert!(<SerialCollective as Collective<f64>>::all_true(&comm, 2, true).unwrap());
        assert!(!<SerialCollective as Collective<f64>>::all_true(&comm, 2, false).unwrap());
        assert!(<SerialCollective as Collective<f64>>::agree_u32(&comm, 3, 19).unwrap());
    }

    #[test]
    fn mock_reduces_in_fixed_rank_order_and_accepts_empty() {
        let handles = MockCollective::<f64>::group(3).unwrap();
        let threads: Vec<_> = handles
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| {
                thread::spawn(move || {
                    let local = match rank {
                        0 => vec![1.0e20, -0.0],
                        1 => vec![-1.0e20, 3.0],
                        _ => vec![0.25, -3.0],
                    };
                    let sum = comm.reduce_sum(10, &local).unwrap();
                    let mut workspace = local.clone();
                    comm.reduce_sum_in_place(13, &mut workspace).unwrap();
                    assert_eq!(workspace, sum);
                    let max = comm.reduce_max(11, &local).unwrap();
                    let empty = comm.reduce_sum(12, &[]).unwrap();
                    comm.reduce_sum_in_place(14, &mut []).unwrap();
                    (sum, max, empty)
                })
            })
            .collect();
        let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        for (sum, max, empty) in results {
            assert_eq!(sum, [0.25, 0.0]);
            assert_eq!(max, [1.0e20, 3.0]);
            assert!(empty.is_empty());
        }
    }

    #[test]
    fn max_reductions_propagate_nonfinite_values() {
        let handles = MockCollective::<f64>::group(3).unwrap();
        let threads: Vec<_> = handles
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| {
                thread::spawn(move || {
                    let value = if rank == 1 {
                        f64::NAN
                    } else {
                        rank as f64 + 1.0
                    };
                    comm.reduce_max(13, &[value]).unwrap()
                })
            })
            .collect();
        for result in threads.into_iter().map(|thread| thread.join().unwrap()) {
            assert!(result[0].is_nan());
        }

        let handles = MockCollective::<Bits256>::group(2).unwrap();
        let finite = "2".parse::<Bits256>().unwrap();
        let threads: Vec<_> = handles
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| {
                let finite = finite;
                thread::spawn(move || {
                    let value = if rank == 0 { Bits256::nan() } else { finite };
                    comm.reduce_max(14, &[value]).unwrap()
                })
            })
            .collect();
        for result in threads.into_iter().map(|thread| thread.join().unwrap()) {
            assert!(result[0].is_nan());
        }

        let folded = WorldCollective::fold_max(&[f64::NAN, 2.0], 2, 1);
        assert!(folded[0].is_nan());
    }

    #[test]
    fn mock_preserves_mpfr_values_without_float_conversion() {
        let handles = MockCollective::<Bits256>::group(2).unwrap();
        let expected = "1.23456789012345678901234567890123456789e-123"
            .parse::<Bits256>()
            .unwrap();
        let other = "-9.87654321098765432109876543210987654321e+77"
            .parse::<Bits256>()
            .unwrap();
        let threads: Vec<_> = handles
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| {
                let expected = expected;
                let other = other;
                thread::spawn(move || {
                    let local = if rank == 0 { [expected] } else { [other] };
                    comm.reduce_sum(20, &local).unwrap()
                })
            })
            .collect();
        let values: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        let sum = expected + other;
        assert_eq!(values, vec![vec![sum], vec![sum]]);
        assert!(sum.is_finite());
        assert!(sum != expected);
    }

    #[test]
    fn mock_failure_and_decision_consensus_are_global() {
        let handles = MockCollective::<f64>::group(3).unwrap();
        let threads: Vec<_> = handles
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| {
                thread::spawn(move || {
                    let all_ok = comm.all_true(30, rank != 1).unwrap();
                    let agree = comm.agree_u32(31, if rank == 0 { 7 } else { 9 }).unwrap();
                    (all_ok, agree)
                })
            })
            .collect();
        let values: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(values, vec![(false, false); 3]);
    }

    #[test]
    fn mock_rejects_length_mismatch_without_partial_result() {
        let handles = MockCollective::<f64>::group(2).unwrap();
        let threads: Vec<_> = handles
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| {
                thread::spawn(move || {
                    let local = if rank == 0 { vec![1.0] } else { vec![] };
                    comm.reduce_sum(40, &local)
                })
            })
            .collect();
        let values: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(values, vec![Err(CollectiveError::LengthMismatch); 2]);
    }

    #[test]
    fn mock_gathers_only_on_root_with_explicit_ranges() {
        let handles = MockCollective::<f64>::group(3).unwrap();
        let ranges = vec![(0, 2), (2, 0), (2, 1)];
        let threads: Vec<_> = handles
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| {
                let ranges = ranges.clone();
                thread::spawn(move || {
                    let local = match rank {
                        0 => vec![1.0, 2.0],
                        1 => Vec::new(),
                        _ => vec![9.0],
                    };
                    comm.gather_root(41, &local, &ranges, 0)
                })
            })
            .collect();
        let values: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(values[0], Ok(Some(vec![1.0, 2.0, 9.0])));
        assert_eq!(values[1], Ok(None));
        assert_eq!(values[2], Ok(None));
    }

    #[test]
    fn mock_root_gather_preserves_mpfr_wire_values() {
        let handles = MockCollective::<Bits256>::group(2).unwrap();
        let ranges = vec![(0, 1), (1, 1)];
        let values = [
            "1.23456789012345678901234567890123456789e-123"
                .parse::<Bits256>()
                .unwrap(),
            "-9.87654321098765432109876543210987654321e+77"
                .parse::<Bits256>()
                .unwrap(),
        ];
        let threads: Vec<_> = handles
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| {
                let ranges = ranges.clone();
                let value = values[rank];
                thread::spawn(move || comm.gather_root(42, &[value], &ranges, 1))
            })
            .collect();
        let gathered: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(gathered[0], Ok(None));
        assert_eq!(gathered[1], Ok(Some(values.to_vec())));
    }

    #[test]
    fn mock_rejects_operation_mismatch_consistently() {
        let mut handles = MockCollective::<f64>::group(2).unwrap();
        let rank0 = handles.remove(0);
        let rank1 = handles.remove(0);
        let threads = [
            thread::spawn(move || rank0.reduce_sum(45, &[1.0])),
            thread::spawn(move || rank1.reduce_max(45, &[1.0])),
        ];
        let values: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(values, vec![Err(CollectiveError::OperationMismatch); 2]);
    }

    #[test]
    fn wrong_thread_fails_before_touching_group_state() {
        let mut handles = MockCollective::<f64>::group(1).unwrap();
        let comm = handles.pop().unwrap();
        assert!(comm.all_true(50, true).unwrap());
        let other = thread::spawn(move || comm.all_true(51, true))
            .join()
            .unwrap();
        assert_eq!(other, Err(CollectiveError::WrongThread));
    }

    #[test]
    fn wrong_thread_wakes_other_ranks_with_consensus_failure() {
        let mut handles = MockCollective::<f64>::group(2).unwrap();
        let rank0 = handles.remove(0);
        let rank1 = handles.remove(0);
        rank1.check_thread().unwrap();
        let waiting = thread::spawn(move || rank0.reduce_sum(60, &[1.0]));
        let wrong = thread::spawn(move || rank1.reduce_sum(60, &[2.0]))
            .join()
            .unwrap();
        assert_eq!(wrong, Err(CollectiveError::WrongThread));
        assert_eq!(waiting.join().unwrap(), Err(CollectiveError::WrongThread));
    }
}
