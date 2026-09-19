use super::*;
#[path = "cone_parallel.rs"]
mod cone_parallel;
use crate::algebra::triangular_number;
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
}

impl<T> CompositeCone<T>
where
    T: FloatT,
{
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

impl<T> CompositeCone<T>
where
    T: FloatT,
{
    fn fold_step_bounds(
        &mut self,
        αmax: T,
        cached_sym: bool,
        mut evaluate: impl FnMut(&mut SupportedCone<T>, std::ops::Range<usize>, T) -> (T, T),
    ) -> (T, T) {
        let mut α = αmax;
        let all_symmetric = self.is_symmetric();
        let mut innerfcn = |α: T, symcond: bool| -> T {
            let mut α = α;
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
                    continue;
                }
                let (nextαz, nextαs) = evaluate(cone, rng.clone(), α);
                α = T::min(α, T::min(nextαz, nextαs));
            }
            α
        };

        // Force symmetric cones first.
        α = innerfcn(α, true);

        // if we have any nonsymmetric cones, then back off from full steps slightly
        // so that centrality checks and logarithms don't fail right at the boundaries
        if !all_symmetric {
            let ceil = T::one() - T::sqrt(T::epsilon());
            α = T::min(α, ceil);
        }

        // Force asymmetric cones last.
        α = innerfcn(α, false);

        (α, α)
    }

    fn shift_one(
        cone: &mut SupportedCone<T>,
        shift: &mut [T],
        dz: &mut [T],
        ds: &mut [T],
        σμ: T,
        prepared: bool,
    ) {
        #[cfg(feature = "sdp")]
        if prepared {
            if let SupportedCone::PSDTriangleCone(cone) = cone {
                cone.combined_shift_prepared(shift, dz, ds, σμ);
                return;
            }
        }
        cone.combined_ds_shift(shift, dz, ds, σμ);
    }

    pub(crate) fn combined_shift_impl(
        &mut self,
        shift: &mut [T],
        step_z: &mut [T],
        step_s: &mut [T],
        σμ: T,
        prepared: bool,
    ) {
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
                    (shift, step_z, step_s),
                    &|cone, _rows, (shift, step_z, step_s)| {
                        Self::shift_one(cone, shift, step_z, step_s, σμ, prepared);
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
        #[cfg(feature = "sdp")]
        if self
            .cones
            .iter()
            .any(|c| matches!(c, SupportedCone::PSDTriangleCone(_)))
        {
            let cached = if let Some(threading) = &self.threading {
                if αmax.is_finite() && αmax > T::zero() && threading.sym_step_lanes.len() > 1 {
                    self.sym_step_bounds.resize(self.cones.len(), (αmax, αmax));
                    threading.pool.install(|| {
                        cone_parallel::prepare_affine_bounds(
                            &mut self.cones,
                            &mut self.sym_step_bounds,
                            &threading.sym_step_lanes,
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
            return self.fold_step_bounds(αmax, cached, |cone, rows, cap| {
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
            #[cfg(feature = "sdp")]
            SupportedCone::PSDTriangleCone(c) => c.scaling_state_len(),
            _ => 0,
        }
    }

    fn pack_scaling_state(cone: &SupportedCone<T>, out: &mut Vec<T>) {
        match cone {
            #[cfg(feature = "sdp")]
            SupportedCone::PSDTriangleCone(c) => c.pack_scaling_state(out),
            _ => {}
        }
    }

    fn unpack_scaling_state(cone: &mut SupportedCone<T>, src: &[T]) {
        match cone {
            #[cfg(feature = "sdp")]
            SupportedCone::PSDTriangleCone(c) => c.unpack_scaling_state(src),
            _ => {}
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
        let owned = world.range(self.cones.len());
        let rng = &self.rng_cones;
        let cones = &mut self.cones;
        let update_one = |i: usize, cone: &mut SupportedCone<T>, inner: bool| -> bool {
            if owned.contains(&i) || Self::scaling_state_len(cone) == 0 {
                let _inner = inner.then(sdpx_arithmetic::inner_parallel::Guard::enter);
                cone.update_scaling(&s[rng[i].clone()], &z[rng[i].clone()], μ, scaling_strategy)
            } else {
                true
            }
        };
        let ok = match &self.threading {
            Some(threading) => threading.pool.install(|| {
                cones
                    .par_iter_mut()
                    .enumerate()
                    .map(|(i, cone)| update_one(i, cone, true))
                    .reduce(|| true, |a, b| a & b)
            }),
            None => cones
                .iter_mut()
                .enumerate()
                .map(|(i, cone)| update_one(i, cone, false))
                .fold(true, |a, b| a & b),
        };
        // Scaling success must agree on every rank before any collective:
        // a rank that returned early while another gathers would deadlock.
        let failed = world.allreduce_max_f64(if ok { 0.0 } else { 1.0 }) > 0.0;
        if failed {
            return false;
        }
        let lens: Vec<usize> = self.cones.iter().map(Self::scaling_state_len).collect();
        let mut offsets = Vec::with_capacity(self.cones.len() + 1);
        offsets.push(0usize);
        for &n in &lens {
            offsets.push(offsets.last().unwrap() + n);
        }
        let gather_ranges: Vec<(usize, usize)> = crate::mpi::ranges(self.cones.len(), world.size())
            .iter()
            .map(|&(b0, len)| (offsets[b0], offsets[b0 + len] - offsets[b0]))
            .collect();
        let (g0, g1) = (offsets[owned.start], offsets[owned.end]);
        let mut local = Vec::with_capacity(g1 - g0);
        for i in owned.clone() {
            Self::pack_scaling_state(&self.cones[i], &mut local);
        }
        debug_assert_eq!(local.len(), g1 - g0);
        let mut all = vec![T::zero(); *offsets.last().unwrap()];
        world.gather_slice(crate::mpi::SITE_CONES, &local, &gather_ranges, &mut all);
        for (i, cone) in self.cones.iter_mut().enumerate() {
            if lens[i] > 0 && !owned.contains(&i) {
                Self::unpack_scaling_state(cone, &all[offsets[i]..offsets[i] + lens[i]]);
            }
        }
        true
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
        let mut α = T::max_value();
        let mut β = T::zero();
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
        if let Some(world) = crate::mpi::World::get() {
            if self
                .cones
                .iter()
                .any(|c| Self::scaling_state_len(c) > 0)
            {
                return self.update_scaling_sharded(world, s, z, μ, scaling_strategy);
            }
        }
        if let Some(threading) = &self.threading {
            if let (Some(chunk), [SupportedCone::NonnegativeCone(cone)]) =
                (threading.orthant_chunk, self.cones.as_mut_slice())
            {
                threading.pool.install(|| {
                    cone.update_scaling_parallel(s, z, chunk);
                });
                return true;
            }
            return threading.pool.install(|| {
                cone_parallel::apply(&mut self.cones, &threading.lanes, (), &|cone, rows, ()| {
                    cone.update_scaling(&s[rows.clone()], &z[rows], μ, scaling_strategy)
                })
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
                    (y, work),
                    &|cone, rows, (y, work)| {
                        cone.mul_Hs(y, &x[rows], work);
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
                    (out, work),
                    &|cone, rows, (out, work)| {
                        cone.Δs_from_Δz_offset(out, &ds[rows.clone()], work, &z[rows]);
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
        let cached_sym = if let Some(threading) = &self.threading {
            if let (Some(chunk), [SupportedCone::NonnegativeCone(cone)]) =
                (threading.orthant_chunk, self.cones.as_mut_slice())
            {
                return threading
                    .pool
                    .install(|| cone.step_length_parallel(dz, ds, z, s, chunk, αmax));
            }
            if αmax.is_finite() && αmax > T::zero() && threading.sym_step_lanes.len() > 1 {
                self.sym_step_bounds.resize(self.cones.len(), (αmax, αmax));
                threading.pool.install(|| {
                    cone_parallel::sym_step_bounds(
                        &mut self.cones,
                        &mut self.sym_step_bounds,
                        &threading.sym_step_lanes,
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

        self.fold_step_bounds(αmax, cached_sym, |cone, rows, cap| {
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

#[cfg(test)]
#[path = "cone_thread_tests.rs"]
mod thread_tests;
