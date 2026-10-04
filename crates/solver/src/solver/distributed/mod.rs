//! Owner-partitioned HSD implementation for MPI: block state stays on its
//! owning rank and ranks share one equality Schur complement. It plugs into the
//! core predictor/corrector loop through the same traits as `default`.

use crate::algebra::*;
use crate::solver::chordal::ChordalInfo;
use crate::solver::cones::CompositeCone;
#[cfg(test)]
use crate::solver::cones::Cone;
#[cfg(test)]
use crate::solver::core::traits::ProblemData;
use crate::solver::default::*;
use crate::solver::sampled::*;
use crate::solver::SupportedConeT;

/// Apply `$f` to every owner block (zipped sources, flat tuple items), on
/// `$pool` when present. Blocks are independent, so pooled and serial runs
/// perform identical per-block arithmetic.
macro_rules! for_blocks {
    ($pool:expr, ($($src:expr),+ $(,)?), $f:expr) => {
        match $pool {
            Some(pool) => pool.install(|| {
                rayon::iter::ParallelIterator::for_each(
                    rayon::iter::IntoParallelIterator::into_par_iter(($($src,)+)),
                    $f,
                )
            }),
            None => itertools::multizip(($($src,)+)).for_each($f),
        }
    };
}
/// `true` when `$f` succeeds on every owner block; see [`for_blocks`].
macro_rules! all_blocks {
    ($pool:expr, ($($src:expr),+ $(,)?), $f:expr) => {
        match $pool {
            Some(pool) => pool.install(|| {
                rayon::iter::ParallelIterator::reduce(
                    rayon::iter::ParallelIterator::map(
                        rayon::iter::IntoParallelIterator::into_par_iter(($($src,)+)),
                        $f,
                    ),
                    || true,
                    |a, b| a & b,
                )
            }),
            None => itertools::multizip(($($src,)+)).map($f).fold(true, |a, b| a & b),
        }
    };
}

pub(crate) mod collective;
mod costs;
mod hsd;
mod kkt;
mod layout;
mod partitioned;
mod state;

#[cfg(feature = "serde")]
pub(crate) use costs::input_fingerprint;
pub(crate) use costs::CostRuntimeConfig;
pub use costs::{CostComponent, CostHistory, CostHistoryOptions, CostOwnerSample};
#[cfg(test)]
pub(crate) use hsd::*;
pub(crate) use kkt::*;
pub(crate) use layout::*;
pub use partitioned::PartitionedSolver;
pub(crate) use state::*;
