use super::*;
#[path = "parallel.rs"]
mod cone_parallel;
use crate::algebra::triangular_number;
pub(crate) use cone_parallel::pin_worker;
use cone_parallel::ConeThreading;
use rayon::prelude::*;
use std::collections::HashMap;
use std::iter::zip;
use std::ops::Range;

// -------------------------------------
// default composite cone type
// -------------------------------------

pub struct CompositeCone<T: FloatT = f64> {
    cones: Vec<SupportedCone<T>>,
    local_only: bool,

    //Type count for each cone type
    pub(crate) type_counts: HashMap<SupportedConeTag, usize>,

    //overall size of the composite cone
    pub(crate) numel: usize,
    pub(crate) degree: usize,

    //ranges for the indices of the constituent cones
    pub(crate) rng_cones: Vec<Range<usize>>,

    //ranges for the indices of the constituent Hs blocks
    //associated with each cone
    pub(crate) rng_blocks: Vec<Range<usize>>,

    // the flag for symmetric cone check
    _is_symmetric: bool,
    threading: Option<ConeThreading>,
    sym_step_bounds: Vec<(T, T)>,
    // Measured per-cone scaling cost (ns), identical on every MPI rank; the
    // rank partition balances it once available.
    mpi_costs: Option<Vec<u64>>,
}

impl<T> CompositeCone<T>
where
    T: FloatT,
{
    /// Owner-local cones never discover or shard over the process MPI world.
    pub(crate) fn is_local_only(&self) -> bool {
        self.local_only
    }

    pub(crate) fn new_local(types: &[SupportedConeT<T>]) -> Self {
        let mut cones = Self::new(types);
        cones.local_only = true;
        cones
    }

    pub fn new(types: &[SupportedConeT<T>]) -> Self {
        // make an internal copy to protect from user modification
        let types = types.to_vec();
        let ncones = types.len();
        let mut cones: Vec<SupportedCone<T>> = Vec::with_capacity(ncones);

        // Count for the number of each cone type, indexed by SupportedConeTag
        // NB: ideally we could fix max capacity here,  but Enum::variant_count is not
        // yet a stable feature.  Capacity should be number of SupportedCone variants.
        // See: https://github.com/rust-lang/rust/issues/73662
        let mut type_counts = HashMap::new();

        // assumed symmetric to start
        let mut _is_symmetric = true;

        // create cones with the given dims
        for t in types.iter() {
            //make a new cone
            let cone = make_cone(t);

            //update global problem symmetry
            _is_symmetric = _is_symmetric && cone.is_symmetric();

            //increment type counts
            *type_counts.entry(cone.as_tag()).or_insert(0) += 1;

            cones.push(cone);
        }

        // count up elements and degree
        let numel = cones.iter().map(|c| c.numel()).sum();
        let degree = cones.iter().map(|c| c.degree()).sum();

        //ranges for the subvectors associated with each cone,
        //and the ranges for the corresponding entries
        //in the Hs sparse block

        let rng_cones = make_rng_cones(&cones);
        let rng_blocks = make_rng_blocks(&cones);

        Self {
            local_only: false,
            cones,
            //types,
            type_counts,
            numel,
            degree,
            rng_cones,
            rng_blocks,
            _is_symmetric,
            threading: None,
            sym_step_bounds: Vec::new(),
            mpi_costs: None,
        }
    }
}

fn make_rng_cones<T>(cones: &[SupportedCone<T>]) -> Vec<Range<usize>>
where
    T: FloatT,
{
    let mut rngs = Vec::with_capacity(cones.len());

    if !cones.is_empty() {
        let mut start = 0;
        for cone in cones {
            let stop = start + cone.numel();
            rngs.push(start..stop);
            start = stop;
        }
    }
    rngs
}

fn make_rng_blocks<T>(cones: &[SupportedCone<T>]) -> Vec<Range<usize>>
where
    T: FloatT,
{
    let mut rngs = Vec::with_capacity(cones.len());

    if !cones.is_empty() {
        let mut start = 0;
        for cone in cones {
            let nvars = cone.numel();
            let stop = start + {
                if cone.Hs_is_diagonal() {
                    nvars
                } else {
                    triangular_number(nvars)
                }
            };
            rngs.push(start..stop);
            start = stop;
        }
    }
    rngs
}

/// Unchanged step inputs (dz, ds, z, s, settings) for the parallel
/// nonsymmetric step pass.
type NonsymStep<'a, T> = (&'a [T], &'a [T], &'a [T], &'a [T], &'a CoreSettings<T>);

impl<T> CompositeCone<T>
where
    T: FloatT,
{
    fn fold_step_bounds(
        &mut self,
        αmax: T,
        cached_sym: bool,
        nonsym: Option<NonsymStep<'_, T>>,
        mut evaluate: impl FnMut(&mut SupportedCone<T>, std::ops::Range<usize>, T) -> (T, T),
    ) -> (T, T) {
        let all_symmetric = self.is_symmetric();

        // Force symmetric cones first.
        let (mut α, αz, αs) = self.fold_pass(αmax, true, cached_sym, &mut evaluate);
        if all_symmetric {
            // Separate bounds for separate primal and dual steps; their
            // minimum is the common step.
            return (αz, αs);
        }

        // if we have any nonsymmetric cones, then back off from full steps slightly
        // so that centrality checks and logarithms don't fail right at the boundaries
        if !all_symmetric {
            let ceil = T::one() - T::sqrt(T::epsilon());
            α = T::min(α, ceil);
        }

        // Force asymmetric cones last.
        if let Some(parallel) = nonsym.and_then(|args| self.nonsym_step_parallel(α, args)) {
            return (parallel, parallel);
        }
        α = self.fold_pass(α, false, cached_sym, &mut evaluate).0;

        (α, α)
    }

    fn fold_pass(
        &mut self,
        α: T,
        symcond: bool,
        cached_sym: bool,
        evaluate: &mut impl FnMut(&mut SupportedCone<T>, std::ops::Range<usize>, T) -> (T, T),
    ) -> (T, T, T) {
        // `α` folds both components at the running cap (the common step);
        // `αz`/`αs` fold each component alone at the initial cap.
        let (cap0, mut α, mut αz, mut αs) = (α, α, α, α);
        for (_index, (cone, rng)) in zip(&mut self.cones, &self.rng_cones).enumerate() {
            if cone.is_symmetric() != symcond {
                continue;
            }
            if cached_sym && cone.is_symmetric() {
                let (boundz, bounds) = self.sym_step_bounds[_index];
                // Preserve each component's clipping operand order,
                // including ties at signed zero, at the current cap.
                // Symmetric-cone step lengths are cap-insensitive, so
                // evaluating them at αmax folds to the same minimum.
                let (nextαz, nextαs) = (T::min(boundz, α), T::min(bounds, α));
                α = T::min(α, T::min(nextαz, nextαs));
                (αz, αs) = (T::min(αz, boundz), T::min(αs, bounds));
                continue;
            }
            // Symmetric cones are evaluated at the initial cap so each
            // component's bound is its own (cap-insensitive: same common α).
            let cap = if symcond { cap0 } else { α };
            let (nextαz, nextαs) = evaluate(cone, rng.clone(), cap);
            α = T::min(α, T::min(nextαz, nextαs));
            (αz, αs) = (T::min(αz, nextαz), T::min(αs, nextαs));
        }
        (α, αz, αs)
    }

    /// Nonsymmetric cones backtrack along the common chain α0·stepᵏ. The
    /// sequential fold (each cone starting from the previous cap) ends at the
    /// smallest per-cone result when membership is monotone along the ray, as
    /// for convex cones. So evaluate every cone from α0 in the pool, take the
    /// minimum and confirm every cone at it (one trial each); on any
    /// disagreement return `None` and keep the sequential fold.
    fn nonsym_step_parallel(&mut self, α0: T, args: NonsymStep<'_, T>) -> Option<T> {
        const MIN_CONES: usize = 64;
        let threading = self.threading.as_ref()?;
        if self.cones.iter().filter(|c| !c.is_symmetric()).count() < MIN_CONES {
            return None;
        }
        let (dz, ds, z, s, settings) = args;
        let rng = &self.rng_cones;
        let cones = &mut self.cones;
        let step_at = |i: usize, cone: &mut SupportedCone<T>, cap: T| {
            let r = rng[i].clone();
            let (a, b) = cone.step_length(
                &dz[r.clone()],
                &ds[r.clone()],
                &z[r.clone()],
                &s[r],
                settings,
                cap,
            );
            T::min(cap, T::min(a, b))
        };
        let αstar = threading.pool.install(|| {
            cones
                .par_iter_mut()
                .enumerate()
                .filter(|(_, c)| !c.is_symmetric())
                .map(|(i, c)| step_at(i, c, α0))
                .reduce(|| α0, |a, b| T::min(a, b))
        });
        if αstar == α0 {
            return Some(α0);
        }
        let confirmed = threading.pool.install(|| {
            cones
                .par_iter_mut()
                .enumerate()
                .filter(|(_, c)| !c.is_symmetric())
                .all(|(i, c)| step_at(i, c, αstar) == αstar)
        });
        confirmed.then_some(αstar)
    }

    fn shift_one(
        cone: &mut SupportedCone<T>,
        shift: &mut [T],
        dz: &mut [T],
        ds: &mut [T],
        σμ: T,
        prepared: bool,
    ) {
        if prepared {
            if let SupportedCone::PSDTriangleCone(cone) = cone {
                cone.combined_shift_prepared(shift, dz, ds, σμ);
                return;
            }
        }
        cone.combined_ds_shift(shift, dz, ds, σμ);
    }

    /// Gondzio corrections. Orthant rows: a trial product
    /// `(s + αΔs)(z + αΔz)` outside `[lo, hi]` adds `−t` to the scaled
    /// complementarity right-hand side (`zΔs + sΔz = −ds`). Second-order
    /// cones: the spectral values `v₀ ± ‖v̄‖` of the Jordan product of the
    /// NT-scaled trial point `W⁻ᵀ(s + αΔs) ∘ W(z + αΔz)` are pushed into the
    /// band (`λ ∘ (WΔz + W⁻ᵀΔs) = −ds`). Binary64 PSD cones do the same with
    /// the eigenvalues of the symmetrized product. Other cones keep their
    /// right-hand side.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn centrality_correction(
        &mut self,
        ds: &mut [T],
        s: &[T],
        z: &[T],
        step_s: &[T],
        step_z: &[T],
        α: T,
        lo: T,
        hi: T,
    ) -> bool {
        let serial = self.mpi_world().is_some() || self.cones.len() < 2;
        if !serial {
            if let Some(threading) = &self.threading {
                let changed = std::sync::atomic::AtomicBool::new(false);
                threading.pool.install(|| {
                    cone_parallel::apply(
                        &mut self.cones,
                        &threading.lanes,
                        threading.inner_parallel,
                        threading.paired,
                        threading.inner_ways,
                        ds,
                        &|cone, rows, ds| {
                            let mut work: [Vec<T>; 3] = Default::default();
                            if correct_cone(
                                cone,
                                ds,
                                &s[rows.clone()],
                                &z[rows.clone()],
                                &step_s[rows.clone()],
                                &step_z[rows],
                                α,
                                lo,
                                hi,
                                &mut work,
                            ) {
                                changed.store(true, std::sync::atomic::Ordering::Relaxed);
                            }
                            true
                        },
                    )
                });
                return changed.into_inner();
            }
        }
        let mut changed = false;
        let mut work: [Vec<T>; 3] = Default::default();
        for (cone, rows) in self.cones.iter_mut().zip(&self.rng_cones) {
            changed |= correct_cone(
                cone,
                &mut ds[rows.clone()],
                &s[rows.clone()],
                &z[rows.clone()],
                &step_s[rows.clone()],
                &step_z[rows.clone()],
                α,
                lo,
                hi,
                &mut work,
            );
        }
        changed
    }

    pub(crate) fn combined_shift_impl(
        &mut self,
        shift: &mut [T],
        step_z: &mut [T],
        step_s: &mut [T],
        σμ: T,
        prepared: bool,
    ) {
        if let Some(world) = self.mpi_world() {
            return self.combined_shift_sharded(world, shift, step_z, step_s, σμ, prepared);
        }
        if let Some(threading) = &self.threading {
            if let (Some(chunk), [SupportedCone::NonnegativeCone(cone)]) =
                (threading.orthant_chunk, self.cones.as_mut_slice())
            {
                threading.pool.install(|| {
                    cone.combined_ds_shift_parallel(shift, step_z, step_s, σμ, chunk);
                });
                return;
            }
            threading.pool.install(|| {
                cone_parallel::apply(
                    &mut self.cones,
                    &threading.lanes,
                    threading.inner_parallel,
                    threading.paired,
                    threading.inner_ways,
                    (shift, step_z, step_s),
                    &|cone, _rows, (shift, step_z, step_s)| {
                        if let (Some(chunk), SupportedCone::NonnegativeCone(c)) =
                            (threading.orthant_chunk, &mut *cone)
                        {
                            c.combined_ds_shift_parallel(shift, step_z, step_s, σμ, chunk);
                        } else {
                            Self::shift_one(cone, shift, step_z, step_s, σμ, prepared);
                        }
                        true
                    },
                )
            });
            return;
        }
        // Here we must first explicitly borrow the subvector
        // of cones, since trying to access it using self.iter_mut
        // causes a borrow conflict with ranges.

        // It is necessary for the function to mutate self since
        // nonsymmetric cones modify their internal state when
        // computing the ds_shift

        for (cone, rng) in zip(&mut self.cones, &self.rng_cones) {
            let shifti = &mut shift[rng.clone()];
            let step_zi = &mut step_z[rng.clone()];
            let step_si = &mut step_s[rng.clone()];
            Self::shift_one(cone, shifti, step_zi, step_si, σμ, prepared);
        }
    }

    pub(crate) fn prepare_affine_bounds(
        &mut self,
        dz: &mut [T],
        ds: &mut [T],
        z: &[T],
        s: &[T],
        settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T) {
        if self
            .cones
            .iter()
            .any(|c| matches!(c, SupportedCone::PSDTriangleCone(_)))
        {
            if let Some(world) = self.mpi_world() {
                let writes = self.mpi_step_bounds(world, dz, ds, z, s, settings, αmax, true);
                for (i, wz, ws) in writes {
                    let r = self.rng_cones[i].clone();
                    dz[r.clone()].copy_from_slice(&wz);
                    ds[r].copy_from_slice(&ws);
                }
                return self.fold_step_bounds(αmax, true, None, |cone, rows, cap| {
                    cone.step_length(
                        &dz[rows.clone()],
                        &ds[rows.clone()],
                        &z[rows.clone()],
                        &s[rows],
                        settings,
                        cap,
                    )
                });
            }
            let cached = if let Some(threading) = &self.threading {
                if αmax.is_finite() && αmax > T::zero() && threading.sym_step_lanes.len() > 1 {
                    self.sym_step_bounds.resize(self.cones.len(), (αmax, αmax));
                    threading.pool.install(|| {
                        cone_parallel::prepare_affine_bounds(
                            &mut self.cones,
                            &mut self.sym_step_bounds,
                            &threading.sym_step_lanes,
                            threading.inner_parallel,
                            threading.paired,
                            threading.inner_ways,
                            dz,
                            ds,
                            z,
                            s,
                            settings,
                            αmax,
                        )
                    });
                    true
                } else {
                    false
                }
            } else {
                false
            };
            return self.fold_step_bounds(αmax, cached, None, |cone, rows, cap| {
                if let SupportedCone::PSDTriangleCone(cone) = cone {
                    cone.prepare_affine_bounds(&mut dz[rows.clone()], &mut ds[rows], cap)
                } else {
                    cone.step_length(
                        &dz[rows.clone()],
                        &ds[rows.clone()],
                        &z[rows.clone()],
                        &s[rows],
                        settings,
                        cap,
                    )
                }
            });
        }
        self.step_length(dz, ds, z, s, settings, αmax)
    }

    /// Configure the reusable cone worker pool. Zero chooses the available
    /// CPU budget; one and structurally small workloads keep the serial path.
    /// Native BLAS must use one thread when cone workers execute PSD kernels.
    pub fn configure_threads(&mut self, threads: usize) -> Result<(), rayon::ThreadPoolBuildError> {
        let threading = ConeThreading::new(&self.cones, threads)?;
        self.threading = threading;
        Ok(())
    }

    /// Run this collection's cone lanes on an existing pool (shared with the
    /// caller, e.g. an MPI rank's owner pool) instead of building a new one.
    pub(crate) fn share_pool(&mut self, pool: std::sync::Arc<rayon::ThreadPool>) {
        self.threading = ConeThreading::with_pool(&self.cones, pool);
    }

    /// Number of workers actually selected after the small-work cutoff.
    pub fn cone_threads(&self) -> usize {
        self.threading
            .as_ref()
            .map_or(1, |t| t.pool.current_num_threads())
    }

    /// Share the same worker budget with sequential KKT block phases.
    pub(crate) fn thread_pool(&self) -> Option<std::sync::Arc<rayon::ThreadPool>> {
        self.threading.as_ref().map(|t| t.pool.clone())
    }

    pub fn len(&self) -> usize {
        self.cones.len()
    }
    pub fn is_empty(&self) -> bool {
        self.cones.is_empty()
    }
    pub fn iter(&self) -> std::slice::Iter<'_, SupportedCone<T>> {
        self.cones.iter()
    }
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, SupportedCone<T>> {
        self.cones.iter_mut()
    }
    pub(crate) fn get_type_count(&self, tag: SupportedConeTag) -> usize {
        if self.type_counts.contains_key(&tag) {
            self.type_counts[&tag]
        } else {
            0
        }
    }

    /// Per-cone scaling-state size exchanged between ranks: only PSD
    /// cones carry nontrivial state (R, Rinv, λ). Every other cone type
    /// recomputes its (cheap, deterministic) update on all ranks.
    fn scaling_state_len(cone: &SupportedCone<T>) -> usize {
        match cone {
            SupportedCone::PSDTriangleCone(c) => c.scaling_state_len(),
            _ => 0,
        }
    }

    /// Shardable per-cone work ∝ n³ (SVD, congruence products, eig); svec
    /// numel ~ n², so `numel^1.5` is the cost proxy. Stateless cones carry
    /// unit cost — they run on every rank anyway.
    fn mpi_work_cost(cone: &SupportedCone<T>) -> u64 {
        let numel = cone.numel() as f64;
        if Self::scaling_state_len(cone) > 0 {
            numel.powf(1.5).max(1.0) as u64
        } else {
            1
        }
    }

    /// Cost-balanced contiguous cone partition shared by every sharded
    /// method and its field gather — all ranks derive identical bounds.
    fn mpi_blocks(&self, world: crate::mpi::World) -> Vec<(usize, usize)> {
        let costs: Vec<u64> = match &self.mpi_costs {
            Some(costs) => costs.clone(),
            None => self.cones.iter().map(Self::mpi_work_cost).collect(),
        };
        crate::mpi::cost_ranges(&costs, world.size())
    }

    fn pack_scaling_state(cone: &SupportedCone<T>, out: &mut Vec<T>) {
        match cone {
            SupportedCone::PSDTriangleCone(c) => c.pack_scaling_state(out),
            _ => {}
        }
    }

    fn unpack_scaling_state(cone: &mut SupportedCone<T>, src: &[T]) {
        match cone {
            SupportedCone::PSDTriangleCone(c) => c.unpack_scaling_state(src),
            _ => {}
        }
    }

    /// The shared MPI world when the problem has at least one PSD cone
    /// (the only cone type worth sharding). Stateless cones still run on
    /// every rank, so the gate is identical on all ranks.
    fn mpi_world(&self) -> Option<crate::mpi::World> {
        if self.local_only {
            return None;
        }
        crate::mpi::World::get()
            .filter(|_| self.cones.iter().any(|c| Self::scaling_state_len(c) > 0))
    }

    /// Allgather packed per-cone fields over `world`. `widths[i]` is
    /// cone `i`'s field count in `T` elements (zero for cones that carry
    /// no exchanged state); `blocks` is the shared per-rank cone partition
    /// (see [`Self::mpi_blocks`]); `fields` holds `(index, data)` pairs for
    /// this rank's owned cones in cone order. Returns every cone's
    /// fields concatenated in cone order, bitwise identical on all ranks.
    fn gather_fields(
        world: crate::mpi::World,
        ncones: usize,
        widths: &[usize],
        blocks: &[(usize, usize)],
        fields: &[(usize, Vec<T>)],
    ) -> Vec<T> {
        let mut offsets = Vec::with_capacity(ncones + 1);
        offsets.push(0usize);
        for &w in widths {
            offsets.push(offsets.last().unwrap() + w);
        }
        let gather_ranges: Vec<(usize, usize)> = blocks
            .iter()
            .map(|&(b0, len)| (offsets[b0], offsets[b0 + len] - offsets[b0]))
            .collect();
        let owned = blocks[world.rank()];
        let (g0, g1) = (offsets[owned.0], offsets[owned.0 + owned.1]);
        let mut local = Vec::with_capacity(g1 - g0);
        for &(i, ref data) in fields {
            if !(owned.0..owned.0 + owned.1).contains(&i) || widths[i] == 0 {
                continue;
            }
            debug_assert_eq!(data.len(), widths[i]);
            local.extend_from_slice(data);
        }
        debug_assert_eq!(local.len(), g1 - g0);
        let mut all = vec![T::zero(); *offsets.last().unwrap()];
        world.gather_slice(crate::mpi::SITE_CONES, &local, &gather_ranges, &mut all);
        all
    }

    /// Prefix offsets matching `gather_fields`' layout, so callers can
    /// locate cone `i`'s fields inside the returned buffer.
    fn field_offsets(widths: &[usize]) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(widths.len() + 1);
        offsets.push(0usize);
        for &w in widths {
            offsets.push(offsets.last().unwrap() + w);
        }
        offsets
    }

    /// Runs `eval` on every cone this rank is responsible for under MPI —
    /// the owned range plus all stateless cones, which are cheap and
    /// needed locally on every rank — collecting `(index, fields)` pairs.
    /// `eval` receives the cone's slice range inside the shared vectors.
    fn mpi_eval_cones(
        &mut self,
        owned: &std::ops::Range<usize>,
        eval: impl Fn(usize, &mut SupportedCone<T>, std::ops::Range<usize>) -> Vec<T> + Sync,
    ) -> Vec<(usize, Vec<T>)> {
        let mask =
            |i: usize, c: &SupportedCone<T>| owned.contains(&i) || Self::scaling_state_len(c) == 0;
        let pool = self.threading.as_ref().map(|t| t.pool.clone());
        let (inner, paired) = self
            .threading
            .as_ref()
            .map_or((false, false), |t| (t.inner_parallel, t.paired));
        let rng = &self.rng_cones;
        let cones = &mut self.cones;
        match pool {
            Some(pool) => pool.install(|| {
                cones
                    .par_iter_mut()
                    .enumerate()
                    .filter(|(i, c)| mask(*i, c))
                    .map(|(i, c)| {
                        // Owned blocks can outnumber free workers; let heavy
                        // kernels offer inner work to the ambient pool when
                        // spare workers exist.
                        let _inner = (inner || paired).then(|| {
                            sdpx_arithmetic::inner_parallel::Guard::enter_levels(inner, paired)
                        });
                        (i, eval(i, c, rng[i].clone()))
                    })
                    .collect()
            }),
            None => cones
                .iter_mut()
                .enumerate()
                .filter(|(i, c)| mask(*i, c))
                .map(|(i, c)| (i, eval(i, c, rng[i].clone())))
                .collect(),
        }
    }

    /// Rank-sharded scaling update: each rank factorizes only its block
    /// range (the expensive SVD/W work), non-PSD cones still update on all
    /// ranks, then the packed `[R, Rinv, λ]` states are gathered in block
    /// order so every rank reconstructs the bitwise-serial cone state.
    fn update_scaling_sharded(
        &mut self,
        world: crate::mpi::World,
        s: &[T],
        z: &[T],
        μ: T,
        scaling_strategy: ScalingStrategy,
    ) -> bool {
        let blocks = self.mpi_blocks(world);
        let owned = {
            let (b0, len) = blocks[world.rank()];
            b0..b0 + len
        };
        let rng = &self.rng_cones;
        let cones = &mut self.cones;
        let elapsed: Vec<std::sync::atomic::AtomicU64> = (0..cones.len())
            .map(|_| std::sync::atomic::AtomicU64::new(0))
            .collect();
        let update_one =
            |i: usize, cone: &mut SupportedCone<T>, inner: bool, paired: bool| -> bool {
                if owned.contains(&i) || Self::scaling_state_len(cone) == 0 {
                    let _inner = (inner || paired).then(|| {
                        sdpx_arithmetic::inner_parallel::Guard::enter_levels(inner, paired)
                    });
                    let start = std::time::Instant::now();
                    let ok = cone.update_scaling(
                        &s[rng[i].clone()],
                        &z[rng[i].clone()],
                        μ,
                        scaling_strategy,
                    );
                    elapsed[i].store(
                        start.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    ok
                } else {
                    true
                }
            };
        let (inner, paired) = self
            .threading
            .as_ref()
            .map_or((false, false), |t| (t.inner_parallel, t.paired));
        let ok = match &self.threading {
            Some(threading) => threading.pool.install(|| {
                cones
                    .par_iter_mut()
                    .enumerate()
                    .map(|(i, cone)| update_one(i, cone, inner, paired))
                    .reduce(|| true, |a, b| a & b)
            }),
            None => cones
                .iter_mut()
                .enumerate()
                .map(|(i, cone)| update_one(i, cone, false, false))
                .fold(true, |a, b| a & b),
        };
        // Scaling success must agree on every rank before any collective:
        // a rank that returned early while another gathers would deadlock.
        let failed = world.allreduce_max_f64(if ok { 0.0 } else { 1.0 }) > 0.0;
        if failed {
            return false;
        }
        let lens: Vec<usize> = self.cones.iter().map(Self::scaling_state_len).collect();
        let fields: Vec<(usize, Vec<T>)> = owned
            .clone()
            .map(|i| {
                let mut data = Vec::with_capacity(lens[i]);
                Self::pack_scaling_state(&self.cones[i], &mut data);
                (i, data)
            })
            .collect();
        let all = Self::gather_fields(world, self.cones.len(), &lens, &blocks, &fields);
        let offsets = Self::field_offsets(&lens);
        for (i, cone) in self.cones.iter_mut().enumerate() {
            if lens[i] > 0 && !owned.contains(&i) {
                Self::unpack_scaling_state(cone, &all[offsets[i]..offsets[i] + lens[i]]);
            }
        }
        // Share the measured costs of the stateful cones (each owned by one
        // rank); the next partition balances them on every rank alike.
        let local: Vec<f64> = owned
            .clone()
            .map(|i| elapsed[i].load(std::sync::atomic::Ordering::Relaxed) as f64)
            .collect();
        let mut measured = vec![0.0f64; self.cones.len()];
        world.gather_slice(crate::mpi::SITE_CONES, &local, &blocks, &mut measured);
        if world.rank() == 0 && crate::receipt::profile_requested() {
            // Observation only: the slowest cones bound the phase.
            let mut top: Vec<(f64, usize)> =
                measured.iter().enumerate().map(|(i, &t)| (t, i)).collect();
            top.sort_by(|a, b| b.0.total_cmp(&a.0));
            let sums: Vec<f64> = blocks
                .iter()
                .map(|&(b0, len)| measured[b0..b0 + len].iter().sum::<f64>() * 1e-9)
                .collect();
            eprintln!(
                "CONE_COSTS rank_sums_s={:?} top_s={:?}",
                sums,
                top.iter()
                    .take(6)
                    .map(|&(t, i)| (i, self.cones[i].numel(), t * 1e-9))
                    .collect::<Vec<_>>()
            );
        }
        self.mpi_costs = Some(
            self.cones
                .iter()
                .zip(&measured)
                .map(|(c, &t)| {
                    if Self::scaling_state_len(c) > 0 {
                        (t as u64).max(1)
                    } else {
                        1
                    }
                })
                .collect(),
        );
        true
    }

    /// Rank-sharded `mul_Hs`: each rank evaluates only its owned PSD
    /// cones (two congruence products each) plus every stateless cone,
    /// then exchanges the `y` slices so all ranks hold the full output.
    fn mul_Hs_sharded(&mut self, world: crate::mpi::World, y: &mut [T], x: &[T]) {
        let blocks = self.mpi_blocks(world);
        let owned = {
            let (b0, len) = blocks[world.rank()];
            b0..b0 + len
        };
        let results = self.mpi_eval_cones(&owned, |_i, cone, r| {
            let mut buf = vec![T::zero(); r.len()];
            let mut w = vec![T::zero(); r.len()];
            cone.mul_Hs(&mut buf, &x[r], &mut w);
            buf
        });
        let widths: Vec<usize> = self
            .cones
            .iter()
            .map(|c| {
                if Self::scaling_state_len(c) > 0 {
                    c.numel()
                } else {
                    0
                }
            })
            .collect();
        let all = Self::gather_fields(world, self.cones.len(), &widths, &blocks, &results);
        let offsets = Self::field_offsets(&widths);
        for (i, data) in results {
            if widths[i] == 0 {
                y[self.rng_cones[i].clone()].copy_from_slice(&data);
            }
        }
        for (i, cone) in self.cones.iter().enumerate() {
            if widths[i] > 0 {
                let r = self.rng_cones[i].clone();
                y[r].copy_from_slice(&all[offsets[i]..offsets[i] + cone.numel()]);
            }
        }
    }

    /// Rank-sharded `Δs_from_Δz_offset` with the same gather pattern.
    fn Δs_sharded(&mut self, world: crate::mpi::World, out: &mut [T], ds: &[T], z: &[T]) {
        let blocks = self.mpi_blocks(world);
        let owned = {
            let (b0, len) = blocks[world.rank()];
            b0..b0 + len
        };
        let results = self.mpi_eval_cones(&owned, |_i, cone, r| {
            let mut buf = vec![T::zero(); r.len()];
            let mut w = vec![T::zero(); r.len()];
            cone.Δs_from_Δz_offset(&mut buf, &ds[r.clone()], &mut w, &z[r]);
            buf
        });
        let widths: Vec<usize> = self
            .cones
            .iter()
            .map(|c| {
                if Self::scaling_state_len(c) > 0 {
                    c.numel()
                } else {
                    0
                }
            })
            .collect();
        let all = Self::gather_fields(world, self.cones.len(), &widths, &blocks, &results);
        let offsets = Self::field_offsets(&widths);
        for (i, data) in results {
            if widths[i] == 0 {
                out[self.rng_cones[i].clone()].copy_from_slice(&data);
            }
        }
        for (i, cone) in self.cones.iter().enumerate() {
            if widths[i] > 0 {
                let r = self.rng_cones[i].clone();
                out[r].copy_from_slice(&all[offsets[i]..offsets[i] + cone.numel()]);
            }
        }
    }

    /// Rank-sharded `combined_shift_impl`: `[shift | step_z | step_s]`
    /// per cone packed as one field.
    fn combined_shift_sharded(
        &mut self,
        world: crate::mpi::World,
        shift: &mut [T],
        step_z: &mut [T],
        step_s: &mut [T],
        σμ: T,
        prepared: bool,
    ) {
        let blocks = self.mpi_blocks(world);
        let owned = {
            let (b0, len) = blocks[world.rank()];
            b0..b0 + len
        };
        let results = self.mpi_eval_cones(&owned, |_i, cone, r| {
            let n = r.len();
            let mut buf = vec![T::zero(); 3 * n];
            let (sh, rest) = buf.split_at_mut(n);
            let (sz, ss) = rest.split_at_mut(n);
            // step_z/step_s carry the affine directions as inputs in both
            // prepared and unprepared modes; shift is pure scratch.
            sz.copy_from_slice(&step_z[r.clone()]);
            ss.copy_from_slice(&step_s[r]);
            Self::shift_one(cone, sh, sz, ss, σμ, prepared);
            buf
        });
        let widths: Vec<usize> = self
            .cones
            .iter()
            .map(|c| {
                if Self::scaling_state_len(c) > 0 {
                    3 * c.numel()
                } else {
                    0
                }
            })
            .collect();
        let all = Self::gather_fields(world, self.cones.len(), &widths, &blocks, &results);
        let offsets = Self::field_offsets(&widths);
        let mut scatter = |i: usize, data: &[T], this: &mut Self| {
            let r = this.rng_cones[i].clone();
            let n = r.len();
            shift[r.clone()].copy_from_slice(&data[..n]);
            step_z[r.clone()].copy_from_slice(&data[n..2 * n]);
            step_s[r].copy_from_slice(&data[2 * n..3 * n]);
        };
        for (i, data) in &results {
            if widths[*i] == 0 {
                scatter(*i, data, self);
            }
        }
        for i in 0..self.cones.len() {
            if widths[i] > 0 {
                scatter(i, &all[offsets[i]..offsets[i] + widths[i]], self);
            }
        }
    }

    /// Rank-sharded `margins`: per-cone `(αi, βi)` exchanged and folded
    /// in cone order — bitwise identical to the serial fold.
    fn margins_sharded(
        &mut self,
        world: crate::mpi::World,
        z: &[T],
        pd: PrimalOrDualCone,
    ) -> (T, T) {
        let blocks = self.mpi_blocks(world);
        let owned = {
            let (b0, len) = blocks[world.rank()];
            b0..b0 + len
        };
        let results = self.mpi_eval_cones(&owned, |_i, cone, r| {
            let mut zi = z[r].to_vec();
            let (αi, βi) = cone.margins(&mut zi, pd);
            vec![αi, βi]
        });
        let widths: Vec<usize> = self
            .cones
            .iter()
            .map(|c| usize::from(Self::scaling_state_len(c) > 0) * 2)
            .collect();
        let all = Self::gather_fields(world, self.cones.len(), &widths, &blocks, &results);
        let offsets = Self::field_offsets(&widths);
        let mut per_cone: Vec<Option<(T, T)>> = vec![None; self.cones.len()];
        for (i, data) in &results {
            if widths[*i] == 0 {
                per_cone[*i] = Some((data[0], data[1]));
            }
        }
        for i in 0..self.cones.len() {
            if widths[i] > 0 {
                per_cone[i] = Some((all[offsets[i]], all[offsets[i] + 1]));
            }
        }
        let mut α = T::max_value();
        let mut β = T::zero();
        for v in per_cone.iter().flatten() {
            α = T::min(α, v.0);
            β += v.1;
        }
        (α, β)
    }

    /// Rank-sharded `compute_barrier`: per-cone barrier exchanged and
    /// summed in cone order on every rank.
    fn compute_barrier_sharded(
        &mut self,
        world: crate::mpi::World,
        z: &[T],
        s: &[T],
        dz: &[T],
        ds: &[T],
        α: T,
    ) -> T {
        let blocks = self.mpi_blocks(world);
        let owned = {
            let (b0, len) = blocks[world.rank()];
            b0..b0 + len
        };
        let results = self.mpi_eval_cones(&owned, |_i, cone, r| {
            vec![cone.compute_barrier(&z[r.clone()], &s[r.clone()], &dz[r.clone()], &ds[r], α)]
        });
        let widths: Vec<usize> = self
            .cones
            .iter()
            .map(|c| usize::from(Self::scaling_state_len(c) > 0))
            .collect();
        let all = Self::gather_fields(world, self.cones.len(), &widths, &blocks, &results);
        let offsets = Self::field_offsets(&widths);
        let mut per_cone: Vec<Option<T>> = vec![None; self.cones.len()];
        for (i, data) in &results {
            if widths[*i] == 0 {
                per_cone[*i] = Some(data[0]);
            }
        }
        for i in 0..self.cones.len() {
            if widths[i] > 0 {
                per_cone[i] = Some(all[offsets[i]]);
            }
        }
        let mut barrier = T::zero();
        for v in per_cone.iter().flatten() {
            barrier += *v;
        }
        barrier
    }

    /// Rank-sharded symmetric-cone step evaluation shared by
    /// `step_length` and `prepare_affine_bounds`. Each rank evaluates
    /// only its owned PSD cones (at the global cap — the min-fold is
    /// cap-insensitive, so the result is bitwise identical) plus every
    /// stateless symmetric cone, then exchanges per-cone bounds. Fills
    /// `sym_step_bounds` for every symmetric cone and, for `prepared`,
    /// returns the rewritten `(dz, ds)` slices of every PSD cone for the
    /// caller to scatter.
    #[allow(clippy::too_many_arguments)]
    fn mpi_step_bounds(
        &mut self,
        world: crate::mpi::World,
        dz: &[T],
        ds: &[T],
        z: &[T],
        s: &[T],
        settings: &CoreSettings<T>,
        αmax: T,
        prepared: bool,
    ) -> Vec<(usize, Vec<T>, Vec<T>)> {
        let blocks = self.mpi_blocks(world);
        let owned = {
            let (b0, len) = blocks[world.rank()];
            b0..b0 + len
        };
        let results = self.mpi_eval_cones(&owned, |_i, cone, r| {
            if !cone.is_symmetric() {
                return Vec::new();
            }
            if let SupportedCone::PSDTriangleCone(cone) = cone {
                if prepared {
                    let n = r.len();
                    let mut buf = vec![T::zero(); 2 * n + 2];
                    let (dz_i, rest) = buf.split_at_mut(n);
                    let (ds_i, tail) = rest.split_at_mut(n);
                    // dz/ds are in/out: seed them with the affine directions.
                    dz_i.copy_from_slice(&dz[r.clone()]);
                    ds_i.copy_from_slice(&ds[r]);
                    let (αz, αs) = cone.prepare_affine_bounds(dz_i, ds_i, αmax);
                    tail[0] = αz;
                    tail[1] = αs;
                    return buf;
                }
            }
            let (αz, αs) = cone.step_length(
                &dz[r.clone()],
                &ds[r.clone()],
                &z[r.clone()],
                &s[r],
                settings,
                αmax,
            );
            vec![αz, αs]
        });
        let widths: Vec<usize> = self
            .cones
            .iter()
            .map(|c| {
                if Self::scaling_state_len(c) > 0 {
                    if prepared {
                        2 * c.numel() + 2
                    } else {
                        2
                    }
                } else {
                    0
                }
            })
            .collect();
        let all = Self::gather_fields(world, self.cones.len(), &widths, &blocks, &results);
        let offsets = Self::field_offsets(&widths);
        self.sym_step_bounds.resize(self.cones.len(), (αmax, αmax));
        let mut writes = Vec::new();
        let mut absorb = |i: usize, data: &[T], this: &mut Self| {
            if !this.cones[i].is_symmetric() {
                return;
            }
            if widths[i] > 0 {
                if prepared {
                    let n = this.rng_cones[i].len();
                    writes.push((i, data[..n].to_vec(), data[n..2 * n].to_vec()));
                }
                this.sym_step_bounds[i] = (data[widths[i] - 2], data[widths[i] - 1]);
            } else {
                this.sym_step_bounds[i] = (data[0], data[1]);
            }
        };
        for (i, data) in &results {
            if widths[*i] == 0 {
                absorb(*i, data, self);
            }
        }
        for i in 0..self.cones.len() {
            if widths[i] > 0 {
                absorb(i, &all[offsets[i]..offsets[i] + widths[i]], self);
            }
        }
        writes
    }
}

impl<T> Cone<T> for CompositeCone<T>
where
    T: FloatT,
{
    fn thread_pool(&self) -> Option<std::sync::Arc<rayon::ThreadPool>> {
        CompositeCone::thread_pool(self)
    }

    fn degree(&self) -> usize {
        self.degree
    }

    fn numel(&self) -> usize {
        self.numel
    }

    fn is_symmetric(&self) -> bool {
        self._is_symmetric
    }

    fn is_sparse_expandable(&self) -> bool {
        //This should probably never be called
        //self.cones.iter().any(|cone| cone.is_sparse_expandable())
        unreachable!();
    }

    fn allows_primal_dual_scaling(&self) -> bool {
        self.cones
            .iter()
            .all(|cone| cone.allows_primal_dual_scaling())
    }

    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool {
        let mut any_changed = false;

        // we will update e <- δ .* e using return values
        // from this function.  default is to do nothing at all
        δ.fill(T::one());
        for (cone, rng) in zip(&self.cones, &self.rng_cones) {
            let δi = &mut δ[rng.clone()];
            let ei = &e[rng.clone()];
            any_changed |= cone.rectify_equilibration(δi, ei);
        }
        any_changed
    }

    fn margins(&mut self, z: &mut [T], pd: PrimalOrDualCone) -> (T, T) {
        if let Some(world) = self.mpi_world() {
            return self.margins_sharded(world, z, pd);
        }
        let mut α = T::max_value();
        let mut β = T::zero();
        // Each PSD margin is a full eigendecomposition: evaluate the cones in
        // the pool, then fold in cone order (the same min and sum sequence).
        let contiguous = self
            .rng_cones
            .iter()
            .scan(0usize, |end, rng| {
                let ok = rng.start == *end;
                *end = rng.end;
                Some(ok)
            })
            .all(|ok| ok);
        if let Some(pool) = self
            .thread_pool()
            .filter(|p| p.current_num_threads() > 1 && self.cones.len() > 1 && contiguous)
        {
            let mut parts: Vec<&mut [T]> = Vec::with_capacity(self.cones.len());
            let mut rest = &mut *z;
            for rng in &self.rng_cones {
                let (part, tail) = std::mem::take(&mut rest).split_at_mut(rng.len());
                parts.push(part);
                rest = tail;
            }
            let margins: Vec<(T, T)> = pool.install(|| {
                self.cones
                    .par_iter_mut()
                    .zip(parts.into_par_iter())
                    .map(|(cone, part)| cone.margins(part, pd))
                    .collect()
            });
            for (αi, βi) in margins {
                α = T::min(α, αi);
                β += βi;
            }
            return (α, β);
        }
        for (cone, rng) in zip(&mut self.cones, &self.rng_cones) {
            let (αi, βi) = cone.margins(&mut z[rng.clone()], pd);
            α = T::min(α, αi);
            β += βi;
        }
        (α, β)
    }

    fn scaled_unit_shift(&self, z: &mut [T], α: T, pd: PrimalOrDualCone) {
        for (cone, rng) in zip(&self.cones, &self.rng_cones) {
            cone.scaled_unit_shift(&mut z[rng.clone()], α, pd);
        }
    }

    fn unit_initialization(&self, z: &mut [T], s: &mut [T]) {
        for (cone, rng) in zip(&self.cones, &self.rng_cones) {
            cone.unit_initialization(&mut z[rng.clone()], &mut s[rng.clone()]);
        }
    }

    fn set_identity_scaling(&mut self) {
        for cone in self.iter_mut() {
            cone.set_identity_scaling();
        }
    }

    fn update_scaling(
        &mut self,
        s: &[T],
        z: &[T],
        μ: T,
        scaling_strategy: ScalingStrategy,
    ) -> bool {
        if let Some(world) = self.mpi_world() {
            return self.update_scaling_sharded(world, s, z, μ, scaling_strategy);
        }
        if let Some(threading) = &self.threading {
            if let Some(ok) = threading.update_scaling(&mut self.cones, s, z, μ, scaling_strategy)
            {
                return ok;
            }
            if let (Some(chunk), [SupportedCone::NonnegativeCone(cone)]) =
                (threading.orthant_chunk, self.cones.as_mut_slice())
            {
                threading.pool.install(|| {
                    cone.update_scaling_parallel(s, z, chunk);
                });
                return true;
            }
            return threading.pool.install(|| {
                cone_parallel::apply(
                    &mut self.cones,
                    &threading.lanes,
                    threading.inner_parallel,
                    threading.paired,
                    threading.inner_ways,
                    (),
                    &|cone, rows, ()| {
                        if let (Some(chunk), SupportedCone::NonnegativeCone(c)) =
                            (threading.orthant_chunk, &mut *cone)
                        {
                            c.update_scaling_parallel(&s[rows.clone()], &z[rows], chunk);
                            true
                        } else {
                            cone.update_scaling(&s[rows.clone()], &z[rows], μ, scaling_strategy)
                        }
                    },
                )
            });
        }
        let mut is_scaling_success;
        for (cone, rng) in zip(&mut self.cones, &self.rng_cones) {
            let si = &s[rng.clone()];
            let zi = &z[rng.clone()];
            is_scaling_success = cone.update_scaling(si, zi, μ, scaling_strategy);
            if !is_scaling_success {
                return false;
            }
        }
        true
    }

    fn Hs_is_diagonal(&self) -> bool {
        //This function should probably never be called since
        //we only us it to interrogate the blocks, but we can
        //implement something reasonable anyway
        self.cones.iter().all(|cone| cone.Hs_is_diagonal())
    }

    #[allow(non_snake_case)]
    fn get_Hs(&self, Hsblock: &mut [T]) {
        for (cone, rng) in zip(&self.cones, &self.rng_blocks) {
            cone.get_Hs(&mut Hsblock[rng.clone()]);
        }
    }

    fn mul_Hs(&mut self, y: &mut [T], x: &[T], work: &mut [T]) {
        if let Some(world) = self.mpi_world() {
            return self.mul_Hs_sharded(world, y, x);
        }
        if let Some(threading) = &self.threading {
            if let (Some(chunk), [SupportedCone::NonnegativeCone(cone)]) =
                (threading.orthant_chunk, self.cones.as_mut_slice())
            {
                threading.pool.install(|| {
                    cone.mul_Hs_parallel(y, x, chunk);
                });
                return;
            }
            threading.pool.install(|| {
                cone_parallel::apply(
                    &mut self.cones,
                    &threading.lanes,
                    threading.inner_parallel,
                    threading.paired,
                    threading.inner_ways,
                    (y, work),
                    &|cone, rows, (y, work)| {
                        if let (Some(chunk), SupportedCone::NonnegativeCone(c)) =
                            (threading.orthant_chunk, &mut *cone)
                        {
                            c.mul_Hs_parallel(y, &x[rows], chunk);
                        } else {
                            cone.mul_Hs(y, &x[rows], work);
                        }
                        true
                    },
                )
            });
            return;
        }
        for (cone, rng) in zip(&mut self.cones, &self.rng_cones) {
            cone.mul_Hs(&mut y[rng.clone()], &x[rng.clone()], &mut work[rng.clone()]);
        }
    }

    fn affine_ds(&self, ds: &mut [T], s: &[T]) {
        for (cone, rng) in zip(&self.cones, &self.rng_cones) {
            let dsi = &mut ds[rng.clone()];
            let si = &s[rng.clone()];
            cone.affine_ds(dsi, si);
        }
    }

    fn combined_ds_shift(&mut self, shift: &mut [T], step_z: &mut [T], step_s: &mut [T], σμ: T) {
        self.combined_shift_impl(shift, step_z, step_s, σμ, false);
    }

    fn Δs_from_Δz_offset(&mut self, out: &mut [T], ds: &[T], work: &mut [T], z: &[T]) {
        if let Some(world) = self.mpi_world() {
            return self.Δs_sharded(world, out, ds, z);
        }
        if let Some(threading) = &self.threading {
            if let (Some(chunk), [SupportedCone::NonnegativeCone(cone)]) =
                (threading.orthant_chunk, self.cones.as_mut_slice())
            {
                threading.pool.install(|| {
                    cone.offset_parallel(out, ds, z, chunk);
                });
                return;
            }
            threading.pool.install(|| {
                cone_parallel::apply(
                    &mut self.cones,
                    &threading.lanes,
                    threading.inner_parallel,
                    threading.paired,
                    threading.inner_ways,
                    (out, work),
                    &|cone, rows, (out, work)| {
                        if let (Some(chunk), SupportedCone::NonnegativeCone(c)) =
                            (threading.orthant_chunk, &mut *cone)
                        {
                            c.offset_parallel(out, &ds[rows.clone()], &z[rows], chunk);
                        } else {
                            cone.Δs_from_Δz_offset(out, &ds[rows.clone()], work, &z[rows]);
                        }
                        true
                    },
                )
            });
            return;
        }
        for (cone, rng) in zip(&mut self.cones, &self.rng_cones) {
            let outi = &mut out[rng.clone()];
            let dsi = &ds[rng.clone()];
            let worki = &mut work[rng.clone()];
            let zi = &z[rng.clone()];
            cone.Δs_from_Δz_offset(outi, dsi, worki, zi);
        }
    }

    fn step_length(
        &mut self,
        dz: &[T],
        ds: &[T],
        z: &[T],
        s: &[T],
        settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T) {
        let cached_sym = if let Some(world) = self.mpi_world() {
            self.mpi_step_bounds(world, dz, ds, z, s, settings, αmax, false);
            true
        } else if let Some(threading) = &self.threading {
            if let (Some(chunk), [SupportedCone::NonnegativeCone(cone)]) =
                (threading.orthant_chunk, self.cones.as_mut_slice())
            {
                return threading
                    .pool
                    .install(|| cone.step_length_parallel(dz, ds, z, s, chunk, αmax));
            }
            if threading.orthant_chunk.is_none()
                && αmax.is_finite()
                && αmax > T::zero()
                && threading.sym_step_lanes.len() > 1
            {
                self.sym_step_bounds.resize(self.cones.len(), (αmax, αmax));
                threading.pool.install(|| {
                    cone_parallel::sym_step_bounds(
                        &mut self.cones,
                        &mut self.sym_step_bounds,
                        &threading.sym_step_lanes,
                        threading.inner_parallel,
                        threading.paired,
                        threading.inner_ways,
                        dz,
                        ds,
                        z,
                        s,
                        settings,
                        αmax,
                    )
                });
                true
            } else {
                false
            }
        } else {
            false
        };

        let nonsym = self
            .mpi_world()
            .is_none()
            .then_some((dz, ds, z, s, settings));
        let orthant = self
            .threading
            .as_ref()
            .filter(|_| !cached_sym)
            .and_then(|threading| {
                threading
                    .orthant_chunk
                    .map(|chunk| (std::sync::Arc::clone(&threading.pool), chunk))
            });
        self.fold_step_bounds(αmax, cached_sym, nonsym, |cone, rows, cap| {
            if let (Some((pool, chunk)), SupportedCone::NonnegativeCone(c)) = (&orthant, &mut *cone)
            {
                return pool.install(|| {
                    c.step_length_parallel(
                        &dz[rows.clone()],
                        &ds[rows.clone()],
                        &z[rows.clone()],
                        &s[rows],
                        *chunk,
                        cap,
                    )
                });
            }
            cone.step_length(
                &dz[rows.clone()],
                &ds[rows.clone()],
                &z[rows.clone()],
                &s[rows],
                settings,
                cap,
            )
        })
    }

    fn compute_barrier(&mut self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        if let Some(world) = self.mpi_world() {
            return self.compute_barrier_sharded(world, z, s, dz, ds, α);
        }
        // Each nonsymmetric barrier costs several logarithms: evaluate the
        // cones in the pool, then add in cone order (the same sum).
        if let Some(threading) = self.threading.as_ref().filter(|_| self.cones.len() > 1) {
            let rng_cones = &self.rng_cones;
            let values: Vec<T> = threading.pool.install(|| {
                self.cones
                    .par_iter_mut()
                    .zip(rng_cones.par_iter())
                    .with_min_len(8)
                    .map(|(cone, rng)| {
                        cone.compute_barrier(
                            &z[rng.clone()],
                            &s[rng.clone()],
                            &dz[rng.clone()],
                            &ds[rng.clone()],
                            α,
                        )
                    })
                    .collect()
            });
            let mut barrier = T::zero();
            for value in values {
                barrier += value;
            }
            return barrier;
        }
        let mut barrier = T::zero();
        for (cone, rng) in zip(&mut self.cones, &self.rng_cones) {
            let zi = &z[rng.clone()];
            let si = &s[rng.clone()];
            let dzi = &dz[rng.clone()];
            let dsi = &ds[rng.clone()];
            barrier += cone.compute_barrier(zi, si, dzi, dsi, α);
        }
        barrier
    }
}

/// One cone's Gondzio correction on its own row slices (see
/// `CompositeCone::centrality_correction`).
#[allow(clippy::too_many_arguments)]
fn correct_cone<T: FloatT>(
    cone: &mut SupportedCone<T>,
    ds: &mut [T],
    s: &[T],
    z: &[T],
    step_s: &[T],
    step_z: &[T],
    α: T,
    lo: T,
    hi: T,
    work: &mut [Vec<T>; 3],
) -> bool {
    use crate::algebra::{MatrixShape, VectorMath};
    use crate::solver::default::band_correction;
    let [trial, ss, zz] = work;
    match cone {
        SupportedCone::NonnegativeCone(_) => {
            let mut changed = false;
            for i in 0..ds.len() {
                let v = (s[i] + α * step_s[i]) * (z[i] + α * step_z[i]);
                if let Some(t) = band_correction(v, lo, hi) {
                    ds[i] -= t;
                    changed = true;
                }
            }
            changed
        }
        SupportedCone::SecondOrderCone(soc) => {
            let n = ds.len();
            (ss.resize(n, T::zero()), zz.resize(n, T::zero()));
            trial.clear();
            trial.extend((0..n).map(|i| s[i] + α * step_s[i]));
            soc.mul_Winv(MatrixShape::T, ss, trial, T::one(), T::zero());
            trial.clear();
            trial.extend((0..n).map(|i| z[i] + α * step_z[i]));
            soc.mul_W(MatrixShape::N, zz, trial, T::one(), T::zero());
            // v = ss ∘ zz = (ss·zz, ss₀ z̄ + zz₀ s̄)
            let v0 = ss.dot(zz);
            trial.clear();
            trial.extend((1..n).map(|k| ss[0] * zz[k] + zz[0] * ss[k]));
            let norm = trial.norm();
            let (tp, tm) = (
                band_correction(v0 + norm, lo, hi),
                band_correction(v0 - norm, lo, hi),
            );
            if tp.is_none() && tm.is_none() {
                return false;
            }
            let (tp, tm) = (tp.unwrap_or(T::zero()), tm.unwrap_or(T::zero()));
            let half = T::from_f64(0.5).unwrap();
            ds[0] -= half * (tp + tm);
            if norm > T::zero() {
                let scale = half * (tp - tm) / norm;
                for (k, &w) in trial.iter().enumerate() {
                    ds[1 + k] -= scale * w;
                }
            }
            true
        }
        SupportedCone::PSDTriangleCone(psd) => {
            psd.centrality_correction(ds, s, z, step_s, step_z, α, lo, hi)
        }
        _ => false,
    }
}

#[cfg(test)]
#[path = "tests/thread.rs"]
mod thread_tests;
