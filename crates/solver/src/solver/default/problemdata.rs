#![allow(non_snake_case)]
use itertools::izip;

use super::*;
use crate::algebra::*;
use crate::solver::SupportedConeT;
use crate::solver::{
    cones::{CompositeCone, Cone},
    core::traits::ProblemData,
};

use crate::solver::chordal::ChordalInfo;
use crate::solver::sampled::SampledOperator;

// ---------------
// Data type for default problem format
// ---------------

/// Standard-form solver type implementing the [`ProblemData`](crate::solver::core::traits::ProblemData) trait
pub struct DefaultProblemData<T> {
    /// The matrix P in the quadratic objective term
    pub P: CscMatrix<T>,
    /// The vector q in the quadratic objective term
    pub q: Vec<T>,
    /// The explicit matrix A in the constraints. For sampled solvers this
    /// holds only the linear component after KKT construction; use
    /// [`Self::materialize_A`] to obtain all coefficients.
    pub A: CscMatrix<T>,
    pub(crate) sampled_matrix_stats: Option<(T, usize)>,
    pub(crate) sampled: Option<std::sync::Arc<SampledOperator<T>>>,
    pub(crate) sampled_input: bool,
    /// The vector b in the constraints
    pub b: Vec<T>,
    /// Vector of cones in the problem
    pub cones: Vec<SupportedConeT<T>>,
    /// Number of variables
    pub n: usize,
    /// Number of constraints
    pub m: usize,
    /// Whether the optional operator-aware dual feasibility work scan is
    /// enabled. Keeping this beside the prepared data lets the residual pass
    /// avoid any extra traversal in the default global-only mode.
    pub(crate) componentwise_enabled: bool,
    /// Equilibration data for the problem
    pub equilibration: DefaultEquilibrationData<T>,

    // unscaled inf norms of linear terms.  Set to "None"
    // during data updating to allow for multiple updates, and
    // then recalculated during solve if needed
    pub(crate) normq: Option<T>,
    pub(crate) normb: Option<T>,

    pub(crate) presolver: Option<Presolver<T>>,
    pub(crate) dropped_zeros: usize, // number of eliminated structural zeros

    pub(crate) chordal_info: Option<ChordalInfo<T>>,

    /// Problem data before presolve and chordal decomposition, kept for the
    /// original-coordinate acceptance test when either changed the problem.
    pub(crate) original: Option<super::original_check::OriginalData<T>>,
}

impl<T> DefaultProblemData<T>
where
    T: FloatT,
{
    /// Checkpoint identity of the internal (presolved, equilibrated) problem.
    pub(crate) fn checkpoint_identity(&self) -> crate::solver::core::checkpoint::Identity {
        use crate::solver::core::checkpoint::{Fnv, Identity};
        let (mut shape, mut values) = (Fnv::new(), Fnv::new());
        shape.words(&[self.n, self.m]);
        for m in [&self.A, &self.P] {
            shape.words(&m.colptr);
            shape.words(&m.rowval);
            values.scalars(&m.nzval);
        }
        values.scalars(&self.q);
        values.scalars(&self.b);
        for cone in &self.cones {
            let tag = match cone {
                SupportedConeT::ZeroConeT(_) => 0,
                SupportedConeT::NonnegativeConeT(_) => 1,
                SupportedConeT::SecondOrderConeT(_) => 2,
                SupportedConeT::ExponentialConeT() => 3,
                SupportedConeT::PowerConeT(a) => {
                    shape.scalars(&[*a]);
                    4
                }
                SupportedConeT::GenPowerConeT(a, dim2) => {
                    shape.scalars(a);
                    shape.word(*dim2 as u64);
                    5
                }
                SupportedConeT::PSDTriangleConeT(_) => 6,
            };
            shape.words(&[tag, cone.nvars()]);
        }
        if let Some(op) = &self.sampled {
            for b in op.blocks() {
                shape.words(&[
                    b.row_start,
                    b.column_start,
                    b.dim,
                    b.basis_rows,
                    b.basis_cols,
                ]);
                values.scalars(&b.basis);
                values.scalars(&b.weights);
            }
        }
        Identity {
            structure: shape.finish(),
            values: values.finish(),
        }
    }

    /// Create a new `DefaultProblemData` object
    pub fn new(
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        settings: &DefaultSettings<T>,
    ) -> Self {
        Self::new_cow(
            std::borrow::Cow::Borrowed(P),
            std::borrow::Cow::Borrowed(q),
            std::borrow::Cow::Borrowed(A),
            std::borrow::Cow::Borrowed(b),
            cones,
            settings,
            None,
        )
    }

    /// Move already-owned inputs when preprocessing leaves them unchanged.
    pub(crate) fn new_cow(
        P_in: std::borrow::Cow<'_, CscMatrix<T>>,
        q_in: std::borrow::Cow<'_, [T]>,
        A_in: std::borrow::Cow<'_, CscMatrix<T>>,
        b_in: std::borrow::Cow<'_, [T]>,
        cones: &[SupportedConeT<T>],
        settings: &DefaultSettings<T>,
        sampled: Option<&SampledOperator<T>>,
    ) -> Self {
        use std::borrow::Cow;
        let (mut P, mut q, mut A, mut b) = (P_in, q_in, A_in, b_in);
        // Collapse repeated orthants, drop empty cones and turn singletons
        // into orthant rows, except for PSD factors in sampled input.
        let mut cones = SupportedConeT::new_collapsed(cones, sampled.is_none());
        if !P.is_triu() {
            P = Cow::Owned(P.to_triu());
        }
        let copy_original =
            |P: &CscMatrix<T>, q: &[T], A: &CscMatrix<T>, b: &[T], cones: &[SupportedConeT<T>]| {
                super::original_check::OriginalData {
                    P: P.clone(),
                    q: q.to_vec(),
                    A: A.clone(),
                    b: b.to_vec(),
                    cones: cones.to_vec(),
                }
            };
        let mut original = None;

        // Each stage replaces only what it changes, so borrowed API data is
        // copied at most once.
        let presolver = try_presolver(&P, &q, &A, &b, &cones, settings, sampled);
        if let Some(presolver) = &presolver {
            if settings.tol_original.is_some() {
                original = Some(copy_original(&P, &q, &A, &b, &cones));
            }
            let (A_new, b_new, cones_new, objective) = presolver.presolve(&P, &q, &A, &b, &cones);
            (A, b, cones) = (Cow::Owned(A_new), Cow::Owned(b_new), cones_new);
            if let Some((P_new, q_new)) = objective {
                (P, q) = (Cow::Owned(P_new), Cow::Owned(q_new));
            }
        }

        // ChordalInfo must be built on the *reduced* problem: its init_cones
        // and per-cone row ranges index the presolved A/b, and the cone_maps
        // it records map decomposed cones back to the presolved cone list.
        // Chordal lowering changes the sampled factors' operator.
        let mut chordal_info = if sampled.is_some() {
            None
        } else {
            try_chordal_info(&A, &b, &cones, settings)
        };
        if let Some(chordal_info) = &mut chordal_info {
            if original.is_none() && settings.tol_original.is_some() {
                original = Some(copy_original(&P, &q, &A, &b, &cones));
            }
            let (P_new, q_new, A_new, b_new, cones_new) =
                chordal_info.decomp_augment(&P, &q, &A, &b, settings);
            (P, q, A, b, cones) = (
                Cow::Owned(P_new),
                Cow::Owned(q_new),
                Cow::Owned(A_new),
                Cow::Owned(b_new),
                cones_new,
            );
        }

        // Scaling owns its inputs; borrowed API data is copied only here.
        let (mut P_new, q_new, mut A_new, mut b_new) = (
            P.into_owned(),
            q.into_owned(),
            A.into_owned(),
            b.into_owned(),
        );
        let cones_new = cones;

        //cap entries in b at INFINITY.  This is important
        //for inf values that were not in a reduced cone
        //this is not considered part of the "presolve", so
        //can always happen regardless of user settings
        let infbound = crate::get_infinity().as_T();
        b_new.scalarop(|x| T::min(x, infbound));

        // this ensures m is the *reduced* size m
        let (m, n) = A_new.size();

        // explicitly dropzeros on the copied data, since dropzeros
        // operates in place. Revisit this order of operations
        // once a proper presolver is implemented, since it might
        // be preferable to dropzeros then presolve
        let mut dropped_zeros = 0;
        if settings.input_sparse_dropzeros {
            dropped_zeros += P_new.dropzeros() + A_new.dropzeros();
        }

        let equilibration = DefaultEquilibrationData::<T>::new(n, m);

        let normq = Some(q_new.norm_inf());
        let normb = Some(b_new.norm_inf());

        Self {
            P: P_new,
            q: q_new,
            A: A_new,
            sampled_matrix_stats: None,
            sampled: None,
            sampled_input: false,
            b: b_new,
            cones: cones_new,
            n,
            m,
            componentwise_enabled: settings.tol_feas_componentwise.is_some(),
            equilibration,
            normq,
            normb,
            dropped_zeros,
            presolver,
            chordal_info,
            original,
        }
    }

    /// Original problem data for the acceptance test: the stored copy, or
    /// the internal data with the equilibration undone.
    pub(crate) fn original_data(
        &self,
    ) -> std::borrow::Cow<'_, super::original_check::OriginalData<T>> {
        if let Some(original) = &self.original {
            return std::borrow::Cow::Borrowed(original);
        }
        let eq = &self.equilibration;
        let cinv = T::recip(eq.c);
        let mut A = self.A.clone();
        for col in 0..A.n {
            for p in A.colptr[col]..A.colptr[col + 1] {
                let row = A.rowval[p];
                A.nzval[p] *= eq.einv[row] * eq.dinv[col];
            }
        }
        let mut P = self.P.clone();
        for col in 0..P.n {
            for p in P.colptr[col]..P.colptr[col + 1] {
                let row = P.rowval[p];
                P.nzval[p] *= eq.dinv[row] * eq.dinv[col] * cinv;
            }
        }
        let q = self
            .q
            .iter()
            .zip(&eq.dinv)
            .map(|(&v, &d)| v * d * cinv)
            .collect();
        let b = self.b.iter().zip(&eq.einv).map(|(&v, &e)| v * e).collect();
        std::borrow::Cow::Owned(super::original_check::OriginalData {
            P,
            q,
            A,
            b,
            cones: self.cones.clone(),
        })
    }

    /// Materialize the equilibrated constraint matrix, including sampled rows.
    /// This may allocate a large matrix; solver products use the factors directly.
    #[allow(non_snake_case)]
    pub fn materialize_A(&self) -> Result<CscMatrix<T>, String> {
        if let Some(operator) = &self.sampled {
            return operator.materialize_checked();
        }
        Ok(self.A.clone())
    }

    pub(crate) fn constraint_norm_inf(&self) -> T {
        self.sampled_matrix_stats
            .map_or_else(|| self.A.nzval.norm_inf(), |s| s.0)
    }

    pub(crate) fn constraint_nnz(&self) -> usize {
        self.sampled_matrix_stats
            .map_or_else(|| self.A.nnz(), |s| s.1)
    }

    pub(crate) fn compact_sampled_matrix(&mut self) {
        if let Some(operator) = &self.sampled {
            self.sampled_matrix_stats = Some((self.A.nzval.norm_inf(), self.A.nnz()));
            self.A = operator.linear().clone();
        }
    }

    pub(crate) fn get_normq(&mut self) -> T {
        if let Some(norm) = self.normq {
            norm
        } else {
            let dinv = &self.equilibration.dinv;
            let cinv = T::recip(self.equilibration.c);
            let norm = self.q.norm_inf_scaled(dinv) * cinv;
            self.normq = Some(norm);
            norm
        }
    }

    pub(crate) fn get_normb(&mut self) -> T {
        if let Some(norm) = self.normb {
            norm
        } else {
            let einv = &self.equilibration.einv;
            let norm = self.b.norm_inf_scaled(einv);
            self.normb = Some(norm);
            norm
        }
    }

    pub(crate) fn clear_normq(&mut self) {
        self.normq = None;
    }

    pub(crate) fn clear_normb(&mut self) {
        self.normb = None;
    }

    pub(crate) fn install_sampled(
        &mut self,
        mut operator: SampledOperator<T>,
        pool: Option<&rayon::ThreadPool>,
    ) {
        self.sampled_input = true;
        debug_assert!(!self.is_chordal_decomposed());
        if let Some(presolver) = &self.presolver {
            let keep = &presolver.reduce_map.as_ref().unwrap().keep_logical;
            debug_assert!(operator.blocks().iter().all(|b| {
                keep[b.row_start..b.row_start + b.row_count()]
                    .iter()
                    .all(|&v| v)
            }));
            let mut prefix = vec![0; keep.len() + 1];
            for (i, &retained) in keep.iter().enumerate() {
                prefix[i + 1] = prefix[i] + usize::from(retained);
            }
            let mut blocks = operator.blocks().to_vec();
            for block in &mut blocks {
                block.row_start = prefix[block.row_start];
            }
            let mut linear = operator.linear().select_rows(keep);
            if let Some(keep) = &presolver.keep_columns {
                let mut prefix = vec![0; keep.len() + 1];
                for (i, &retained) in keep.iter().enumerate() {
                    prefix[i + 1] = prefix[i] + usize::from(retained);
                }
                for block in &mut blocks {
                    debug_assert!(keep
                        [block.column_start..block.column_start + block.column_count()]
                        .iter()
                        .all(|&v| v));
                    block.column_start = prefix[block.column_start];
                }
                linear = super::presolver::select_columns(&linear, keep);
            }
            operator = SampledOperator::new(linear, blocks)
                .expect("presolve retains complete sampled blocks");
        }
        // Each sampled block takes its first row's scale; blocks own disjoint
        // rows, so they are rescaled in parallel (one division per row).
        {
            let eq = &mut self.equilibration;
            let mut spans: Vec<(usize, usize, T)> = operator
                .blocks()
                .iter()
                .map(|b| (b.row_start, b.row_count(), eq.e[b.row_start]))
                .collect();
            spans.sort_unstable_by_key(|s| s.0);
            let mut rest = (&mut self.b[..], &mut eq.e[..], &mut eq.einv[..]);
            let mut offset = 0;
            let mut parts = Vec::with_capacity(spans.len());
            for &(start, len, sigma) in &spans {
                let (_, b) = std::mem::take(&mut rest.0).split_at_mut(start - offset);
                let (_, e) = std::mem::take(&mut rest.1).split_at_mut(start - offset);
                let (_, einv) = std::mem::take(&mut rest.2).split_at_mut(start - offset);
                let (b, b_rest) = b.split_at_mut(len);
                let (e, e_rest) = e.split_at_mut(len);
                let (einv, einv_rest) = einv.split_at_mut(len);
                rest = (b_rest, e_rest, einv_rest);
                offset = start + len;
                parts.push((b, e, einv, sigma));
            }
            let rescale = |(b, e, einv, sigma): (&mut [T], &mut [T], &mut [T], T)| {
                for ((b, e), einv) in b.iter_mut().zip(e.iter_mut()).zip(einv.iter_mut()) {
                    *b *= sigma / *e;
                    *e = sigma;
                    *einv = sigma.recip();
                }
            };
            match pool {
                Some(p) if p.current_num_threads() > 1 => {
                    use rayon::prelude::*;
                    p.install(|| parts.into_par_iter().for_each(rescale))
                }
                _ => parts.into_iter().for_each(rescale),
            }
        }
        let (d, e) = (&self.equilibration.d, &self.equilibration.e);
        operator.scale_with(d, e, |linear| lrscale_pooled(linear, e, Some(d), pool));
        // The stored factors define this input. Assembly and operator products
        // are allowed their ordinary working-precision rounding differences.
        // Release the Ruiz-scaled copy before assembling its replacement.
        self.A = CscMatrix::zeros((0, 0));
        self.A = operator.materialize_pooled(pool);
        self.sampled = Some(std::sync::Arc::new(operator));
    }

    // data updating not supported following presolve
    //reduction or chordal decomposition
    pub(crate) fn is_presolved(&self) -> bool {
        self.presolver.is_some()
    }

    // data updating not supported if structural zeros
    // have been eliminated
    pub(crate) fn is_dropped_zeros(&self) -> bool {
        self.dropped_zeros != 0
    }

    pub(crate) fn is_chordal_decomposed(&self) -> bool {
        if self.chordal_info.is_some() {
            return true;
        }
        false
    }
}

impl<T> ProblemData<T> for DefaultProblemData<T>
where
    T: FloatT,
{
    type V = DefaultVariables<T>;
    type C = CompositeCone<T>;
    type SE = DefaultSettings<T>;

    fn scale_stats(&self) -> String {
        let f = |v: T| v.to_f64().unwrap_or(f64::NAN);
        let range = |v: &[T]| {
            let lo = v.iter().copied().fold(T::infinity(), T::min);
            let hi = v.iter().copied().fold(T::zero(), T::max);
            (f(lo), f(hi))
        };
        let eq = &self.equilibration;
        format!(
            "n={} m={} q_inf={:.3e} q_2={:.3e} b_inf={:.3e} b_2={:.3e} c={:.3e} d=[{:.3e},{:.3e}] e=[{:.3e},{:.3e}]",
            self.q.len(),
            self.b.len(),
            f(self.q.norm_inf()),
            f(self.q.norm()),
            f(self.b.norm_inf()),
            f(self.b.norm()),
            f(eq.c),
            range(&eq.d).0,
            range(&eq.d).1,
            range(&eq.e).0,
            range(&eq.e).1
        )
    }

    /// `min(1, 1/max(d))`: the largest column equilibration scale. A column
    /// scaled up by `d` carries data `d` times below the unit start, and
    /// the slacks balancing it reach that scale (Λ27: max d 7.5e27, final
    /// τ 5.5e-29; Λ19 6.1e25; ising11 2.2). Overshooting the solution scale
    /// costs about one iteration per decade, undershooting about ten.
    fn unit_start_tau(&self) -> Option<T> {
        let dmax = self.equilibration.d.iter().copied().fold(T::one(), T::max);
        // Replicated ranks agree on one start.
        let global = crate::mpi::max_all_f64(dmax.to_f64().unwrap_or(f64::INFINITY));
        let dmax = if global.is_finite() {
            T::from_f64(global).unwrap()
        } else {
            dmax
        };
        Some(T::min(T::one(), T::recip(dmax)))
    }

    fn equilibrate(&mut self, cones: &CompositeCone<T>, settings: &DefaultSettings<T>) {
        let data = self;
        let equil = &mut data.equilibration;

        // if equilibration is disabled, just return.  Note that
        // the default equilibration structure initializes with
        // identity scaling already.
        if !settings.equilibrate_enable {
            return;
        }

        // references to scaling matrices from workspace
        let (d, e) = (&mut equil.d, &mut equil.e);

        // use the inverse scalings as work vectors
        let dwork = &mut equil.dinv;
        let ework = &mut equil.einv;

        // references to problem data
        // note that P may be triu, but it shouldn't matter
        let (P, A, q, b) = (&mut data.P, &mut data.A, &mut data.q, &mut data.b);

        let scale_min = settings.equilibrate_min_scaling;
        let scale_max = settings.equilibrate_max_scaling;
        let pool = cones.thread_pool();

        // perform scaling operations for a fixed number of steps
        for _ in 0..settings.equilibrate_max_iter {
            kkt_col_norms_pooled(P, A, dwork, ework, pool.as_deref());

            //zero rows or columns should not get scaled
            dwork.scalarop(|x| if x == T::zero() { T::one() } else { x });
            ework.scalarop(|x| if x == T::zero() { T::one() } else { x });

            dwork.rsqrt();
            ework.rsqrt();

            // bound the cumulative scaling
            let bound = |w: &mut [T], s: &[T]| {
                for (w, &s) in izip!(w.iter_mut(), s.iter()) {
                    *w = T::clip(w, scale_min / s, scale_max / s);
                }
            };
            match pool.as_deref() {
                Some(p) if ework.len() >= POOLED_RUIZ_MIN => p.install(|| {
                    use rayon::prelude::*;
                    rayon::join(
                        || dwork.par_chunks_mut(RUIZ_CHUNK).zip(d.par_chunks(RUIZ_CHUNK)).for_each(|(w, s)| bound(w, s)),
                        || ework.par_chunks_mut(RUIZ_CHUNK).zip(e.par_chunks(RUIZ_CHUNK)).for_each(|(w, s)| bound(w, s)),
                    );
                }),
                _ => {
                    bound(dwork, d);
                    bound(ework, e);
                }
            }

            // Scale the problem data and update the
            // equilibration matrices
            scale_data(P, A, q, b, Some(dwork), ework, pool.as_deref());
            d.hadamard(dwork);
            e.hadamard(ework);

            // Reuse Dwork for the newly scaled P column norms when present.
            let mean_col_norm_P = if P.nnz() == 0 {
                T::zero()
            } else {
                P.col_norms(dwork);
                dwork.mean()
            };
            let inf_norm_q = q.norm_inf();

            if mean_col_norm_P != T::zero() && inf_norm_q != T::zero() {
                let scale_cost = T::max(inf_norm_q, mean_col_norm_P);
                let ctmp = T::recip(scale_cost);
                let ctmp = T::clip(&ctmp, scale_min / equil.c, scale_max / equil.c);

                // scale the penalty terms and overall scaling
                P.scale(ctmp);
                q.scale(ctmp);
                equil.c *= ctmp;
            }
        } //end Ruiz scaling loop

        // fix scalings in cones for which elementwise
        // scaling can't be applied. Rectification should
        //either do nothing or take a convex combination of
        //scalings over a cone, so shouldn't need to check
        //bounds on the scalings here
        if cones.rectify_equilibration(ework, e) {
            // only rescale again if some cones were rectified
            scale_data(P, A, q, b, None, ework, pool.as_deref());
            e.hadamard(ework);
        }

        // update the inverse scaling data
        equil.dinv.scalarop_from(T::recip, d);
        equil.einv.scalarop_from(T::recip, e);
    }
}

// ---------------
// utilities
// ---------------

fn kkt_col_norms<T: FloatT>(
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    norm_LHS: &mut [T],
    norm_RHS: &mut [T],
) {
    P.col_norms_sym(norm_LHS); // P can be triu
    A.col_norms_no_reset(norm_LHS); // incrementally from P norms
    A.row_norms(norm_RHS); // same as column norms of A'
}

/// Rows from which the Ruiz norm and bound passes run on the pool (mixed
/// Λ27/1024, 544k rows and 61M entries: about 3.9 s per serial pass).
const POOLED_RUIZ_MIN: usize = 1 << 14;
const RUIZ_CHUNK: usize = 1 << 12;

/// [`kkt_col_norms`] on the pool for large `A`: columns split over workers;
/// rows split into ranges, each scanning every column's sorted rows within
/// its range. Maxima of the same values, so the norms are identical.
fn kkt_col_norms_pooled<T: FloatT>(
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    norm_LHS: &mut [T],
    norm_RHS: &mut [T],
    pool: Option<&rayon::ThreadPool>,
) {
    use rayon::prelude::*;
    let Some(pool) = pool.filter(|p| p.current_num_threads() > 1 && A.m >= POOLED_RUIZ_MIN) else {
        return kkt_col_norms(P, A, norm_LHS, norm_RHS);
    };
    P.col_norms_sym(norm_LHS);
    let (colptr, rowval, nzval) = (&A.colptr, &A.rowval, &A.nzval);
    pool.install(|| {
        rayon::join(
            || {
                norm_LHS.par_iter_mut().enumerate().with_min_len(64).for_each(|(j, v)| {
                    *v = nzval[colptr[j]..colptr[j + 1]]
                        .iter()
                        .fold(*v, |m, &x| T::max(m, T::abs(x)));
                })
            },
            || {
                norm_RHS.par_chunks_mut(RUIZ_CHUNK).enumerate().for_each(|(k, out)| {
                    out.fill(T::zero());
                    let (r0, r1) = (k * RUIZ_CHUNK, k * RUIZ_CHUNK + out.len());
                    for j in 0..A.n {
                        let rows = &rowval[colptr[j]..colptr[j + 1]];
                        let lo = rows.partition_point(|&r| r < r0);
                        let hi = lo + rows[lo..].partition_point(|&r| r < r1);
                        for p in colptr[j] + lo..colptr[j] + hi {
                            let v = &mut out[rowval[p] - r0];
                            *v = T::max(*v, T::abs(nzval[p]));
                        }
                    }
                })
            },
        )
    });
}

fn scale_data<T: FloatT>(
    P: &mut CscMatrix<T>,
    A: &mut CscMatrix<T>,
    q: &mut [T],
    b: &mut [T],
    d: Option<&[T]>,
    e: &[T],
    pool: Option<&rayon::ThreadPool>,
) {
    match d {
        Some(d) => {
            P.lrscale(d, d); // P[:,:] = Ds*P*Ds
            lrscale_pooled(A, e, Some(d), pool);
            q.hadamard(d);
        }
        None => {
            lrscale_pooled(A, e, None, pool); // A[:,:] = Es*A
        }
    }
    b.hadamard(e);
}

// Ruiz rescaling of a large high-precision A is O(nnz) independent entry
// updates per pass. Column ranges of equal nnz run on the cone pool with the
// same per-entry expression as `lrscale`/`lscale`, so results are bitwise
// identical to the serial path.
fn lrscale_pooled<T: FloatT>(
    A: &mut CscMatrix<T>,
    l: &[T],
    r: Option<&[T]>,
    pool: Option<&rayon::ThreadPool>,
) {
    use rayon::prelude::*;
    let nnz = A.nzval.len();
    let workers = pool.map_or(1, |p| p.current_num_threads());
    // Below this size the serial loop is cheaper than dispatch.
    if workers <= 1 || nnz < 1 << 16 {
        match r {
            Some(r) => A.lrscale(l, r),
            None => A.lscale(l),
        }
        return;
    }
    let target = nnz.div_ceil(4 * workers).max(1);
    let mut parts = Vec::with_capacity(4 * workers + 1);
    let (colptr, rowval) = (&A.colptr, &A.rowval);
    let mut rest: &mut [T] = &mut A.nzval;
    let mut col = 0;
    while col < A.n {
        let (first, mut end) = (colptr[col], col + 1);
        while end < A.n && colptr[end] - first < target {
            end += 1;
        }
        let (chunk, tail) = std::mem::take(&mut rest).split_at_mut(colptr[end] - first);
        rest = tail;
        parts.push((col..end, chunk));
        col = end;
    }
    pool.unwrap().install(|| {
        parts.into_par_iter().for_each(|(cols, vals)| {
            let base = colptr[cols.start];
            for c in cols {
                let range = colptr[c] - base..colptr[c + 1] - base;
                let rows = &rowval[colptr[c]..colptr[c + 1]];
                match r {
                    Some(r) => {
                        for (val, row) in vals[range].iter_mut().zip(rows) {
                            *val *= l[*row] * r[c];
                        }
                    }
                    None => {
                        for (val, row) in vals[range].iter_mut().zip(rows) {
                            *val *= l[*row];
                        }
                    }
                }
            }
        });
    });
}

fn try_chordal_info<T>(
    A: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
    settings: &DefaultSettings<T>,
) -> Option<ChordalInfo<T>>
where
    T: FloatT,
{
    if !settings.chordal_decomposition_enable {
        return None;
    }

    // nothing to do if there are no PSD cones or they are all small
    if !cones
        .iter()
        .any(|c| matches!(c, SupportedConeT::PSDTriangleConeT(dim) if *dim > 3))
    {
        return None;
    }

    let chordal_info = ChordalInfo::new(A, b, cones, settings);

    // no decomposition possible, or not worth its overlap variables
    if !chordal_info.is_decomposed() || !chordal_info.decomposition_pays(A) {
        return None;
    }

    Some(chordal_info)
}

fn try_presolver<T>(
    P: &CscMatrix<T>,
    q: &[T],
    A: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
    settings: &DefaultSettings<T>,
    sampled: Option<&SampledOperator<T>>,
) -> Option<Presolver<T>>
where
    T: FloatT,
{
    if !settings.presolve_enable {
        return None;
    }

    let presolver = Presolver::new(P, q, A, b, cones, sampled);

    if !presolver.is_reduced() {
        return None;
    }

    Some(presolver)
}

#[cfg(test)]
mod start_tau_tests {
    use super::*;

    fn tau_for(scale: f64) -> f64 {
        // One well-scaled column and one scaled by `scale`.
        let A = CscMatrix::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 0.5, 0.25 * scale, scale],
        );
        let P = CscMatrix::zeros((2, 2));
        let cones = [SupportedConeT::NonnegativeConeT(2)];
        let settings = DefaultSettings::<f64>::default();
        let mut data = DefaultProblemData::new(&P, &[1.0, 1.0], &A, &[1.0, 1.0], &cones, &settings);
        let composite = CompositeCone::new(&cones);
        data.equilibrate(&composite, &settings);
        let tau = data.unit_start_tau().unwrap();
        let dmax = data.equilibration.d.iter().copied().fold(1.0, f64::max);
        assert_eq!(tau, f64::min(1.0, 1.0 / dmax));
        tau
    }

    #[test]
    fn unit_start_tau_follows_largest_column_scale() {
        // Balanced data keeps the unit start.
        assert_eq!(tau_for(1.0), 1.0);
        // A tiny column is scaled up (to the binary64 bound 1e4 here) and the
        // unit start moves out by the same factor.
        let tau = tau_for(1e-12);
        assert!(tau < 1e-3 && tau >= 1e-4, "tau {tau}");
    }
}
