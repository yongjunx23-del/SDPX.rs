#![allow(non_snake_case)]
use super::*;
use crate::algebra::*;
use crate::solver::SupportedConeT;

// ---------------
// Data type for default problem presolver
// ---------------

#[derive(Debug)]
pub(crate) struct PresolverRowReductionIndex {
    // vector of length = original RHS.   Entries are false
    // for those rows that should be eliminated before solve
    pub keep_logical: Vec<bool>,
}

/// Presolver data for the standard solver implementation

#[derive(Debug)]
pub(crate) struct Presolver<T> {
    // original cones of the problem
    pub(crate) _init_cones: Vec<SupportedConeT<T>>,

    // Original-row map for exact equality, infinite NN and SOC tail reductions
    pub(crate) reduce_map: Option<PresolverRowReductionIndex>,

    // Second-order cones whose constant tail coordinates were merged
    pub(crate) soc_tails: Vec<SocTail<T>>,

    // Variables fixed by their only (singleton) row; `keep_columns` is set
    // when any were removed, and `objective_offset` is their `Σ q_j x_j`.
    pub(crate) fixed: Vec<FixedColumn<T>>,
    pub(crate) keep_columns: Option<Vec<bool>>,
    pub(crate) objective_offset: T,

    // size of original and reduced RHS, respectively
    pub(crate) mfull: usize,
    pub(crate) mreduced: usize,

    // inf bound that was taken from the module level
    // and should be applied throughout.   Held here so
    // that any subsequent change to the module's state
    // won't mess up our solver mid-solve
    pub(crate) infbound: f64,
}

impl<T> Presolver<T>
where
    T: FloatT,
{
    /// create a new presolver object
    pub(crate) fn new(
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        _settings: &DefaultSettings<T>,
    ) -> Self {
        let infbound = crate::get_infinity();

        // make copy of cones to protect from user interference
        let init_cones = cones.to_vec();
        let mfull = b.len();

        let (mut reduce_map, mut mreduced) = make_reduction_map(A, cones, b, infbound.as_T());
        let soc_tails = soc_constant_tails(A, b, cones);
        if !soc_tails.is_empty() {
            let keep = &mut reduce_map
                .get_or_insert_with(|| PresolverRowReductionIndex {
                    keep_logical: vec![true; mfull],
                })
                .keep_logical;
            for tail in &soc_tails {
                for &row in &tail.rows {
                    if Some(row) != tail.carrier {
                        keep[row] = false;
                        mreduced -= 1;
                    }
                }
            }
        }
        let fixed = singleton_columns(
            P,
            q,
            A,
            &reduced_rhs(b, &soc_tails),
            cones,
            &soc_tails,
            reduce_map.as_ref().map(|map| &map.keep_logical[..]),
        );
        let mut keep_columns = None;
        let mut objective_offset = T::zero();
        if !fixed.is_empty() {
            let keep = &mut reduce_map
                .get_or_insert_with(|| PresolverRowReductionIndex {
                    keep_logical: vec![true; mfull],
                })
                .keep_logical;
            let columns = keep_columns.get_or_insert_with(|| vec![true; A.n]);
            for f in &fixed {
                keep[f.row] = false;
                columns[f.column] = false;
                mreduced -= 1;
            }
            objective_offset = T::dot_fma(fixed.iter().map(|f| (&q[f.column], &f.x)));
        }

        Self {
            _init_cones: init_cones,
            reduce_map,
            soc_tails,
            fixed,
            keep_columns,
            objective_offset,
            mfull,
            mreduced,
            infbound,
        }
    }

    /// true if the presolver has reduced the problem
    pub(crate) fn is_reduced(&self) -> bool {
        self.reduce_map.is_some()
    }
    /// returns number of constraints eliminated
    pub(crate) fn count_reduced(&self) -> usize {
        self.mfull - self.mreduced
    }

    /// Reduced `A`, `b` and cones, plus `P` and `q` when columns were removed.
    #[allow(clippy::type_complexity)]
    pub(crate) fn presolve(
        &self,
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
    ) -> (
        CscMatrix<T>,
        Vec<T>,
        Vec<SupportedConeT<T>>,
        Option<(CscMatrix<T>, Vec<T>)>,
    ) {
        let (A_new, b_new) = self.reduce_A_b(A, b);
        let cones_new = self.reduce_cones(cones);
        let objective = self
            .keep_columns
            .as_ref()
            .map(|keep| (select_principal(P, keep), q.select(keep)));

        (A_new, b_new, cones_new, objective)
    }

    fn reduce_A_b(&self, A: &CscMatrix<T>, b: &[T]) -> (CscMatrix<T>, Vec<T>) {
        assert!(self.reduce_map.is_some());
        let map = self.reduce_map.as_ref().unwrap();

        let mut A = A.select_rows(&map.keep_logical);
        if let Some(keep) = &self.keep_columns {
            A = select_columns(&A, keep);
        }
        let b = reduced_rhs(b, &self.soc_tails).select(&map.keep_logical);

        (A, b)
    }

    fn reduce_cones(&self, cones: &[SupportedConeT<T>]) -> Vec<SupportedConeT<T>> {
        assert!(self.reduce_map.is_some());
        let map = self.reduce_map.as_ref().unwrap();

        // assume that we will end up with the same
        // number of cones, despite small possibility
        // that some will be completely eliminated

        let mut cones_new = Vec::with_capacity(cones.len());
        let mut keep_iter = map.keep_logical.iter();

        for cone in cones {
            let numel_cone = cone.nvars();
            let markers = keep_iter.by_ref().take(numel_cone);

            if matches!(
                cone,
                SupportedConeT::NonnegativeConeT(_) | SupportedConeT::ZeroConeT(_)
            ) {
                let nkeep = markers.filter(|&b| *b).count();
                if nkeep > 0 {
                    cones_new.push(if matches!(cone, SupportedConeT::ZeroConeT(_)) {
                        SupportedConeT::ZeroConeT(nkeep)
                    } else {
                        SupportedConeT::NonnegativeConeT(nkeep)
                    });
                }
            } else if matches!(cone, SupportedConeT::SecondOrderConeT(_)) {
                cones_new.push(SupportedConeT::SecondOrderConeT(
                    markers.filter(|&b| *b).count(),
                ));
            } else {
                //NB: take() is lazy, so must consume this block
                //to force keep_iter to advance to the next cone

                // this clippy lint is a false positive
                #[allow(unknown_lints)] // suppress error in old versions
                #[allow(clippy::double_ended_iterator_last)]
                markers.last(); // skip this cone
                cones_new.push(cone.clone());
            }
        }

        // A fully constant SOC tail leaves a one-row cone, i.e. an orthant row.
        SupportedConeT::new_collapsed(&cones_new)
    }

    pub(crate) fn reverse_presolve(
        &self,
        solution: &mut DefaultSolution<T>,
        variables: &DefaultVariables<T>,
    ) {
        // Certificates have no constant part: tail slacks are -A x = 0, and
        // fixed columns and their rows carry zero certificate entries.
        let certificate = solution.status.is_infeasible();
        match &self.keep_columns {
            None => {
                solution.x.copy_from(&variables.x);
            }
            Some(keep) => {
                let mut reduced = variables.x.iter();
                for (x, &kept) in solution.x.iter_mut().zip(keep) {
                    if kept {
                        *x = *reduced.next().unwrap();
                    }
                }
                for f in &self.fixed {
                    solution.x[f.column] = if certificate { T::zero() } else { f.x };
                }
            }
        }

        let map = self.reduce_map.as_ref().unwrap();
        let mut ctr = 0;

        let zero_rows = self._init_cones.iter().flat_map(|c| {
            std::iter::repeat_n(matches!(c, SupportedConeT::ZeroConeT(_)), c.nvars())
        });
        for ((idx, &keep), zero) in map.keep_logical.iter().enumerate().zip(zero_rows) {
            if keep {
                solution.s[idx] = variables.s[ctr];
                solution.z[idx] = variables.z[ctr];
                ctr += 1;
            } else {
                solution.s[idx] = if zero {
                    T::zero()
                } else {
                    self.infbound.as_T()
                };
                solution.z[idx] = T::zero();
            }
        }
        for f in &self.fixed {
            solution.s[f.row] = T::zero();
            solution.z[f.row] = if certificate { T::zero() } else { f.z };
        }
        for tail in &self.soc_tails {
            let scale = |v: T| {
                if tail.norm == T::zero() {
                    T::zero()
                } else {
                    v / tail.norm
                }
            };
            match tail.carrier {
                // (s1, s_V, ν) and (z1, z_V, ζ) lift to s_C = s_ν·b_C/ν and
                // z_C = ζ·b_C/ν: same cone norms, inner products and bᵀz.
                Some(row) => {
                    let (s, z) = (solution.s[row], solution.z[row]);
                    for (&r, &v) in tail.rows.iter().zip(&tail.values) {
                        solution.s[r] = s * scale(v);
                        solution.z[r] = z * scale(v);
                    }
                }
                // Orthant row s1 - ν >= 0 with multiplier z1 lifts to
                // s = (s1, b_C) and z = z1·(1, -b_C/ν) on the cone boundary.
                None => {
                    let z = solution.z[tail.head];
                    if !certificate {
                        solution.s[tail.head] += tail.norm;
                    }
                    for (&r, &v) in tail.rows.iter().zip(&tail.values) {
                        solution.s[r] = if certificate { T::zero() } else { v };
                        solution.z[r] = -z * scale(v);
                    }
                }
            }
        }
    }
}

/// Coordinates `C` of a second-order cone tail whose rows of `A` are all zero
/// have constant slacks `b_C`, so `(s1, s_V, s_C) ∈ K` iff `(s1, s_V, ν) ∈ K`
/// with `ν = ‖b_C‖`. One carrier row keeps `ν` (none when `ν = 0`); with no
/// variable tail left the cone is the orthant row `s1 - ν >= 0`.
#[derive(Debug)]
pub(crate) struct SocTail<T> {
    head: usize,
    rows: Vec<usize>,
    values: Vec<T>,
    norm: T,
    carrier: Option<usize>,
    orthant: bool,
}

/// `b` after the SOC tail reductions (carrier `ν`, orthant head `b1 - ν`).
fn reduced_rhs<T: FloatT>(b: &[T], tails: &[SocTail<T>]) -> Vec<T> {
    let mut b = b.to_vec();
    for tail in tails {
        match tail.carrier {
            Some(row) => b[row] = tail.norm,
            None => b[tail.head] -= tail.norm,
        }
    }
    b
}

/// A variable `x_j` without quadratic terms whose only retained nonzero
/// `a = A_ij` lies in a single-entry orthant or equality row `i` is fixed at
/// `x_j = b_i/a`, with multiplier `z_i = -q_j/a` (dual row `q_j + a z_i = 0`)
/// and `s_i = 0`. An orthant row needs `z_i >= 0`; otherwise the column is
/// left to the solver. Row and column leave; `q_j x_j` is an objective constant.
#[derive(Debug)]
pub(crate) struct FixedColumn<T> {
    column: usize,
    row: usize,
    x: T,
    z: T,
}

fn singleton_columns<T: FloatT>(
    P: &CscMatrix<T>,
    q: &[T],
    A: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
    tails: &[SocTail<T>],
    keep: Option<&[bool]>,
) -> Vec<FixedColumn<T>> {
    const ORTHANT: u8 = 1;
    const EQUALITY: u8 = 2;
    let mut kind = vec![0u8; b.len()];
    let mut start = 0;
    for cone in cones {
        let dim = cone.nvars();
        let k = match cone {
            SupportedConeT::NonnegativeConeT(_) => ORTHANT,
            SupportedConeT::ZeroConeT(_) => EQUALITY,
            _ => 0,
        };
        kind[start..start + dim].fill(k);
        start += dim;
    }
    for tail in tails.iter().filter(|t| t.orthant) {
        kind[tail.head] = ORTHANT;
    }
    if !kind.iter().any(|&k| k != 0) {
        return Vec::new();
    }
    let kept = |r: usize| keep.is_none_or(|k| k[r]);
    let mut row_count = vec![0u32; b.len()];
    for (&r, &v) in A.rowval.iter().zip(&A.nzval) {
        if v != T::zero() && kept(r) {
            row_count[r] = row_count[r].saturating_add(1);
        }
    }
    let mut quadratic = vec![false; A.n];
    for j in 0..P.n {
        for k in P.colptr[j]..P.colptr[j + 1] {
            if P.nzval[k] != T::zero() {
                quadratic[j] = true;
                quadratic[P.rowval[k]] = true;
            }
        }
    }
    let mut fixed = Vec::new();
    for column in 0..A.n {
        if quadratic[column] {
            continue;
        }
        let mut entries = (A.colptr[column]..A.colptr[column + 1])
            .filter(|&k| A.nzval[k] != T::zero() && kept(A.rowval[k]));
        let (Some(k), None) = (entries.next(), entries.next()) else {
            continue;
        };
        let (row, a) = (A.rowval[k], A.nzval[k]);
        if row_count[row] != 1 || kind[row] == 0 {
            continue;
        }
        let (x, z) = (b[row] / a, -q[column] / a);
        if !x.is_finite() || !z.is_finite() || (kind[row] == ORTHANT && z < T::zero()) {
            continue;
        }
        fixed.push(FixedColumn { column, row, x, z });
    }
    fixed
}

fn select_columns<T: FloatT>(A: &CscMatrix<T>, keep: &[bool]) -> CscMatrix<T> {
    let mut colptr = vec![0];
    let (mut rowval, mut nzval) = (Vec::new(), Vec::new());
    for (j, _) in keep.iter().enumerate().filter(|(_, &k)| k) {
        rowval.extend_from_slice(&A.rowval[A.colptr[j]..A.colptr[j + 1]]);
        nzval.extend_from_slice(&A.nzval[A.colptr[j]..A.colptr[j + 1]]);
        colptr.push(rowval.len());
    }
    CscMatrix::new(A.m, colptr.len() - 1, colptr, rowval, nzval)
}

/// Principal submatrix of `P` on the retained columns; removed columns have
/// no nonzero entries.
fn select_principal<T: FloatT>(P: &CscMatrix<T>, keep: &[bool]) -> CscMatrix<T> {
    let keep = keep.to_vec();
    select_columns(P, &keep).select_rows(&keep)
}

fn soc_constant_tails<T: FloatT>(
    A: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
) -> Vec<SocTail<T>> {
    if !cones
        .iter()
        .any(|c| matches!(c, SupportedConeT::SecondOrderConeT(d) if *d > 1))
    {
        return Vec::new();
    }
    let mut used = vec![false; b.len()];
    for (&row, &value) in A.rowval.iter().zip(&A.nzval) {
        used[row] |= value != T::zero();
    }
    let mut tails = Vec::new();
    let mut start = 0;
    for cone in cones {
        let dim = cone.nvars();
        if let SupportedConeT::SecondOrderConeT(d) = cone {
            if *d > 1 {
                let rows: Vec<usize> = (start + 1..start + dim).filter(|&r| !used[r]).collect();
                let values: Vec<T> = rows.iter().map(|&r| b[r]).collect();
                let norm = values.norm();
                let orthant = rows.len() == dim - 1;
                // A single constant coordinate is already merged.
                if !rows.is_empty() && (orthant || rows.len() > 1) {
                    let carrier = (!orthant && norm != T::zero()).then_some(rows[0]);
                    tails.push(SocTail {
                        head: start,
                        rows,
                        values,
                        norm,
                        carrier,
                        orthant,
                    });
                }
            }
        }
        start += dim;
    }
    tails
}

fn make_reduction_map<T>(
    A: &CscMatrix<T>,
    cones: &[SupportedConeT<T>],
    b: &[T],
    infbound: T,
) -> (Option<PresolverRowReductionIndex>, usize)
where
    T: FloatT,
{
    //assume we keep everything initially
    let mut keep_logical = vec![true; b.len()];
    let mut mreduced = b.len();

    // only try to reduce nn cones.  Make a slight contraction
    // so that we are firmly "less than" here
    let infbound = (T::one() - T::epsilon() * (10.).as_T()) * infbound;

    // we loop through b and remove any entries that are both infinite
    // and in a nonnegative cone

    let mut idx = 0; // index into the b vector

    for cone in cones {
        let numel_cone = cone.nvars();

        if matches!(cone, SupportedConeT::NonnegativeConeT(_)) {
            for _ in 0..numel_cone {
                if b[idx] > infbound {
                    keep_logical[idx] = false;
                    mreduced -= 1;
                }
                idx += 1;
            }
        } else {
            // skip this cone
            idx += numel_cone;
        }
    }

    if let Some(redundant) = redundant_equalities(A, b, cones) {
        for row in redundant {
            keep_logical[row] = false;
            mreduced -= 1;
        }
    }

    let outoption = {
        if mreduced < b.len() {
            Some(PresolverRowReductionIndex { keep_logical })
        } else {
            None
        }
    };

    (outoption, mreduced)
}

// Prove dependence over exact input values, including RHS. Retained rows are
// unchanged, so zero multipliers restore a valid dual in original coordinates.
// Elimination is bounded by an operation budget (modular proof and rational
// fallback separately) and the per-coefficient size guard (`Exact::bounded`);
// past either, no row is removed.
const MODULAR_BUDGET: usize = 400_000_000;
const RATIONAL_BUDGET: usize = 40_000_000;
/// Elimination updates allowed per unit of one IPM iteration's estimated
/// work (stored coefficients and rows of `A`, plus `p^3` per PSD block for
/// its scaling). The proof is an optional reduction: this keeps it within a
/// few iterations' cost. The fixed budgets above let it run 1.4-3.3 s on
/// SOCP_sched_50_50 (2526 equality rows) against 0.11 s for the whole solve
/// without presolve, while the large Float64 SDP gains from the 1317 rows it
/// removes in 1.2 s of a 45-125 s solve.
const WORK_PER_UNIT: usize = 4;

fn redundant_equalities<T: FloatT>(
    A: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
) -> Option<Vec<usize>> {
    let timer = crate::receipt::start();
    let result = redundant_equalities_impl(A, b, cones);
    crate::receipt::finish("presolve_rank", timer);
    result
}

fn redundant_equalities_impl<T: FloatT>(
    A: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
) -> Option<Vec<usize>> {
    use sdpx_arithmetic::Exact;
    use std::collections::BTreeMap;
    let mut ids = Vec::new();
    let mut start = 0;
    for cone in cones {
        if matches!(cone, SupportedConeT::ZeroConeT(_)) {
            ids.extend(start..start + cone.nvars());
        }
        start += cone.nvars();
    }
    if ids.is_empty() {
        return None;
    }
    let mut lookup = vec![usize::MAX; b.len()];
    for (i, &r) in ids.iter().enumerate() {
        lookup[r] = i;
    }
    let psd_work = cones
        .iter()
        .map(|c| match c {
            SupportedConeT::PSDTriangleConeT(p) => p.saturating_pow(3),
            _ => 0,
        })
        .fold(0usize, usize::saturating_add);
    let work = (A.nnz() + b.len())
        .saturating_add(psd_work)
        .saturating_mul(WORK_PER_UNIT);
    if short_equalities_full_rank(A, b, &ids, &lookup, work) {
        return Some(Vec::new());
    }
    // A full row rank image proves that no exact row can be removed. A
    // singular image proves nothing: only then build the rational rows.
    let mut images: Vec<BTreeMap<usize, u64>> = (0..ids.len()).map(|_| BTreeMap::new()).collect();
    for (i, &r) in ids.iter().enumerate() {
        let value = b[r].mersenne31()?;
        if value != 0 {
            images[i].insert(A.n, u64::from(value));
        }
    }
    for c in 0..A.n {
        for k in A.colptr[c]..A.colptr[c + 1] {
            let i = lookup[A.rowval[k]];
            if i != usize::MAX {
                let value = A.nzval[k].mersenne31()?;
                if value != 0 {
                    images[i].insert(c, u64::from(value));
                }
            }
        }
    }
    match modular_full_row_rank(images, work.min(MODULAR_BUDGET)) {
        Some(true) => return Some(Vec::new()),
        // Out of budget: the costlier rational pass would not finish either.
        None => return None,
        Some(false) => {}
    }
    let mut rows: Vec<BTreeMap<usize, Exact>> = (0..ids.len()).map(|_| BTreeMap::new()).collect();
    for (i, &r) in ids.iter().enumerate() {
        if b[r] != T::zero() {
            rows[i].insert(A.n, b[r].exact()?);
        }
    }
    for c in 0..A.n {
        for k in A.colptr[c]..A.colptr[c + 1] {
            let i = lookup[A.rowval[k]];
            if i != usize::MAX && A.nzval[k] != T::zero() {
                rows[i].insert(c, A.nzval[k].exact()?);
            }
        }
    }
    let mut basis: BTreeMap<usize, BTreeMap<usize, Exact>> = BTreeMap::new();
    let mut redundant = Vec::new();
    let mut budget = work.min(RATIONAL_BUDGET);
    for (id, mut row) in ids.into_iter().zip(rows) {
        loop {
            let Some((&pivot, value)) = row.first_key_value() else {
                redundant.push(id);
                break;
            };
            let factor = value.clone();
            if let Some(previous) = basis.get(&pivot) {
                budget = budget.checked_sub(previous.len())?;
                for (&c, value) in previous {
                    let entry = row.entry(c).or_default();
                    entry.subtract_product(&factor, value);
                    if !entry.bounded() {
                        return None;
                    }
                    if entry.is_zero() {
                        row.remove(&c);
                    }
                }
            } else {
                for value in row.values_mut() {
                    value.divide(&factor);
                    if !value.bounded() {
                        return None;
                    }
                }
                basis.insert(pivot, row);
                break;
            }
        }
    }
    Some(redundant)
}

// A nonsingular minor over F_(2^31 - 1) proves independence over the exact
// input field. For a short, wide equality system, build that minor a column
// at a time and stop at full rank, before allocating rational rows for every
// coefficient. Bound dense workspace; all inconclusive cases keep the sparse
// modular/rational path below, including genuine dependencies.
fn short_equalities_full_rank<T: FloatT>(
    a: &CscMatrix<T>,
    b: &[T],
    ids: &[usize],
    lookup: &[usize],
    work: usize,
) -> bool {
    // The dense basis takes m² words (4 MiB at the limit); larger systems use
    // the sparse modular elimination below.
    let m = ids.len();
    if m == 0 || m > 1024 || a.n < m {
        return false;
    }
    // A full minor takes about m³/2 modular updates; past twice that (a
    // dependent system), defer to the sparse modular/rational path unchanged.
    // The dense modular minor is cheap per update: keep its 16M floor (SOCP
    // nb_L2_bessel proves its 123 rows in 5.6 ms) and bound larger systems
    // by the work estimate.
    let mut budget = (2 * m * m * m).max(16_000_000).min(work.max(16_000_000));
    // Any nonsingular minor is a proof, so visit the columns in a fixed
    // stride order: blocks of columns sharing a row pattern are then sampled
    // early instead of exhausting the budget one block at a time.
    let mut stride = ((a.n as f64 * 0.618_033_988_7) as usize).max(1);
    while gcd(stride, a.n) != 1 {
        stride += 1;
    }
    // Echelon row `pivot` lives at basis[pivot * m..][pivot..]; its earlier
    // entries are zero and never read.
    let mut basis = vec![0u32; m * m];
    let mut have = vec![false; m];
    let mut column = vec![0u32; m];
    let mut rank = 0;
    for step in 0..=a.n {
        let c = if step == a.n {
            a.n
        } else {
            (step as u128 * stride as u128 % a.n as u128) as usize
        };
        column.fill(0);
        if c == a.n {
            for (i, &r) in ids.iter().enumerate() {
                let Some(value) = b[r].mersenne31() else {
                    return false;
                };
                column[i] = value;
            }
        } else {
            for k in a.colptr[c]..a.colptr[c + 1] {
                let i = lookup[a.rowval[k]];
                if i != usize::MAX {
                    let Some(value) = a.nzval[k].mersenne31() else {
                        return false;
                    };
                    column[i] = value;
                }
            }
        }
        for pivot in 0..m {
            let factor = column[pivot];
            if factor == 0 {
                continue;
            }
            let Some(rest) = budget.checked_sub(m - pivot) else {
                return false;
            };
            budget = rest;
            let row = &mut basis[pivot * m + pivot..(pivot + 1) * m];
            if have[pivot] {
                for (value, &previous) in column[pivot..].iter_mut().zip(row.iter()) {
                    *value = sub_mersenne31(*value, mul_mersenne31(factor, previous));
                }
            } else {
                let inverse = inverse_mersenne31(u64::from(factor)) as u32;
                for (value, slot) in column[pivot..].iter_mut().zip(row.iter_mut()) {
                    *value = mul_mersenne31(*value, inverse);
                    *slot = *value;
                }
                have[pivot] = true;
                rank += 1;
                break;
            }
        }
        if rank == m {
            return true;
        }
    }
    false
}

/// `a·b mod (2^31 - 1)` for canonical `a, b`: two folds of `2^31 ≡ 1`
/// instead of a division (the dense minor's inner loop).
#[inline(always)]
fn mul_mersenne31(a: u32, b: u32) -> u32 {
    const P: u64 = (1 << 31) - 1;
    let t = u64::from(a) * u64::from(b);
    let t = (t & P) + (t >> 31);
    let t = ((t & P) + (t >> 31)) as u32;
    if t >= P as u32 {
        t - P as u32
    } else {
        t
    }
}

/// `a - b mod (2^31 - 1)` for canonical `a, b`.
#[inline(always)]
fn sub_mersenne31(a: u32, b: u32) -> u32 {
    const P: u32 = (1 << 31) - 1;
    let d = a + P - b;
    if d >= P {
        d - P
    } else {
        d
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn inverse_mersenne31(value: u64) -> u64 {
    const P: u64 = (1 << 31) - 1;
    let (mut power, mut exponent, mut inverse) = (value, P - 2, 1u64);
    while exponent != 0 {
        if exponent & 1 != 0 {
            inverse = inverse * power % P;
        }
        power = power * power % P;
        exponent >>= 1;
    }
    inverse
}

/// `Some(full rank)` over F_(2^31 - 1), or `None` when the budget runs out.
fn modular_full_row_rank(
    rows: Vec<std::collections::BTreeMap<usize, u64>>,
    budget: usize,
) -> Option<bool> {
    use std::collections::BTreeMap;
    const P: u64 = (1 << 31) - 1;
    let mut basis: BTreeMap<usize, BTreeMap<usize, u64>> = BTreeMap::new();
    let mut budget = budget;
    for mut row in rows {
        loop {
            let Some((&pivot, &factor)) = row.first_key_value() else {
                return Some(false);
            };
            if let Some(previous) = basis.get(&pivot) {
                let rest = budget.checked_sub(previous.len())?;
                budget = rest;
                for (&column, &value) in previous {
                    let entry = row.entry(column).or_default();
                    *entry = (*entry + P - factor * value % P) % P;
                    if *entry == 0 {
                        row.remove(&column);
                    }
                }
            } else {
                let inverse = inverse_mersenne31(factor);
                for value in row.values_mut() {
                    *value = *value * inverse % P;
                }
                basis.insert(pivot, row);
                break;
            }
        }
    }
    Some(true)
}

#[cfg(test)]
mod exact_tests {
    use super::*;
    fn check<T: FloatT>() {
        let one = T::one();
        let two = one + one;
        let zero = T::zero();
        // Last row differs below f64 resolution in MPFR; it must be retained.
        let delta = T::epsilon();
        let a = CscMatrix::new(
            5,
            2,
            vec![0, 4, 7],
            vec![0, 1, 2, 4, 0, 1, 4],
            vec![one, two, one, one, one, two, one + delta],
        );
        let b = vec![two, two + two, zero, zero, two];
        let cones = vec![SupportedConeT::ZeroConeT(5)];
        assert_eq!(redundant_equalities(&a, &b, &cones).unwrap(), vec![1, 3]);
        let mut inconsistent = b.clone();
        inconsistent[1] += delta * two * two * two;
        assert!(!redundant_equalities(&a, &inconsistent, &cones)
            .unwrap()
            .contains(&1));
    }
    fn peeled_chain<T: FloatT>() {
        // Rows 0..31 form a singleton cascade; the separate two-row component
        // has one exact dependence. A nonzero RHS alone must not peel a row.
        let n = 34;
        let mut i = Vec::new();
        let mut j = Vec::new();
        let mut v = Vec::new();
        for r in 0..32 {
            i.push(r);
            j.push(r);
            v.push(T::one());
            if r > 0 {
                i.push(r);
                j.push(r - 1);
                v.push(T::one());
            }
        }
        for r in 32..34 {
            for c in 32..34 {
                i.push(r);
                j.push(c);
                v.push(T::one());
            }
        }
        let a = CscMatrix::new_from_triplets(n, n, i, j, v);
        let mut b = vec![T::one(); n];
        let cones = [SupportedConeT::ZeroConeT(n)];
        assert_eq!(redundant_equalities(&a, &b, &cones).unwrap(), vec![33]);
        b[33] += T::one();
        assert!(redundant_equalities(&a, &b, &cones).unwrap().is_empty());
    }
    #[test]
    fn singleton_cascade_f64() {
        peeled_chain::<f64>();
    }
    #[test]
    fn singleton_cascade_256() {
        peeled_chain::<sdpx_arithmetic::Bits256>();
    }
    #[test]
    fn singleton_cascade_512() {
        peeled_chain::<sdpx_arithmetic::Bits512>();
    }
    #[test]
    fn exact_rows_f64() {
        check::<f64>();
    }
    #[test]
    fn exact_rows_256() {
        check::<sdpx_arithmetic::Bits256>();
    }
    #[test]
    fn exact_rows_512() {
        check::<sdpx_arithmetic::Bits512>();
    }
}

#[cfg(test)]
mod reduction_tests {
    use super::*;
    use crate::solver::{IPSolver, SolverStatus};
    use SupportedConeT::*;

    fn t<T: FloatT>(v: f64) -> T {
        T::from_f64(v).unwrap()
    }

    fn solve<T: FloatT>(
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        presolve: bool,
    ) -> DefaultSolver<T> {
        let settings = DefaultSettings {
            verbose: false,
            presolve_enable: presolve,
            ..DefaultSettings::default()
        };
        let mut solver = DefaultSolver::new(P, q, A, b, cones, settings).unwrap();
        solver.solve();
        solver
    }

    fn soc_gap<T: FloatT>(v: &[T]) -> T {
        v[0] - v[1..].norm()
    }

    // x0: SOC (2x0 - 1, 3, 4), an orthant row fixing x0 = 3; x1, x2: SOC
    // (x1 + 10, x2, 1, 2) whose constants merge into √5; x2 >= 1 in a row
    // singleton of a coupled column; x3: equality singleton 2x3 = 4.
    fn mixed<T: FloatT>() {
        let A = CscMatrix::new(
            9,
            4,
            vec![0, 1, 2, 4, 5],
            vec![0, 3, 4, 7, 8],
            vec![t(-2.), t(-1.), t(-1.), t(-1.), t(2.)],
        );
        let b: Vec<T> = [-1., 3., 4., 10., 0., 1., 2., -1., 4.].map(t).to_vec();
        let q: Vec<T> = [1., 1., 1., -1.].map(t).to_vec();
        let P = CscMatrix::new(4, 4, vec![0, 0, 1, 1, 1], vec![1], vec![t(1.)]);
        let cones = [
            SecondOrderConeT(3),
            SecondOrderConeT(4),
            NonnegativeConeT(1),
            ZeroConeT(1),
        ];
        let reduced = solve(&P, &q, &A, &b, &cones, true);
        let full = solve(&P, &q, &A, &b, &cones, false);
        let presolver = reduced.data.presolver.as_ref().unwrap();
        assert_eq!(presolver.fixed.len(), 2);
        assert_eq!(presolver.soc_tails.len(), 2);
        assert_eq!((reduced.data.n, reduced.data.m), (2, 4));
        for s in [&reduced, &full] {
            assert_eq!(s.solution.status, SolverStatus::Solved);
        }
        let (r, f) = (&reduced.solution, &full.solution);
        let tol = t::<T>(1e-6);
        assert!((r.obj_val - f.obj_val).abs() < tol);
        assert_eq!(r.x[0], t(3.));
        assert_eq!(r.x[3], t(2.));
        for (a, b) in r.x.iter().zip(&f.x) {
            assert!((*a - *b).abs() < tol);
        }
        // Original-coordinate KKT conditions of the lifted point.
        let mut res = b.clone();
        A.gemv(&mut res, &r.x, -T::one(), T::one());
        for (v, s) in res.iter().zip(&r.s) {
            assert!((*v - *s).abs() < tol);
        }
        let mut dual = q.clone();
        P.sym(MatrixTriangle::Triu)
            .symv(&mut dual, &r.x, T::one(), T::one());
        A.t().gemv(&mut dual, &r.z, T::one(), T::one());
        assert!(dual.norm_inf() < tol);
        for range in [0..3, 3..7] {
            assert!(soc_gap(&r.s[range.clone()]) > -tol);
            assert!(soc_gap(&r.z[range.clone()]) > -tol);
        }
        assert!(r.s.dot(&r.z).abs() < tol);
        assert!(r.z[7] >= T::zero() && r.s[7] >= T::zero());
        assert!((-b.dot(&r.z) - xpx_half(&P, &r.x) - r.obj_val).abs() < tol);
    }

    fn xpx_half<T: FloatT>(P: &CscMatrix<T>, x: &[T]) -> T {
        let mut px = vec![T::zero(); x.len()];
        P.sym(MatrixTriangle::Triu)
            .symv(&mut px, x, T::one(), T::zero());
        px.dot(x) / t(2.)
    }

    #[test]
    fn mixed_f64() {
        mixed::<f64>();
    }

    #[test]
    fn mixed_256() {
        mixed::<sdpx_arithmetic::Bits256>();
    }

    // (x0, 3, 4) in SOC with x0 <= 2: the orthant row x0 >= 5 conflicts,
    // and the lifted certificate must be one for the original cones.
    #[test]
    fn infeasible_certificate_lifts() {
        let A = CscMatrix::new(4, 1, vec![0, 2], vec![0, 3], vec![-1., 1.]);
        let b = vec![0., 3., 4., 2.];
        let cones = [SecondOrderConeT(3), NonnegativeConeT(1)];
        let s = solve(&CscMatrix::zeros((1, 1)), &[0.], &A, &b, &cones, true);
        assert_eq!(s.solution.status, SolverStatus::PrimalInfeasible);
        let z = &s.solution.z;
        let mut atz = vec![0f64];
        A.t().gemv(&mut atz, z, 1., 0.);
        assert!(atz[0].abs() < 1e-8 && b.dot(z) < 0.);
        assert!(soc_gap(&z[0..3]) > -1e-8 && z[3] >= 0.);
    }

    // min -x with x >= 0 needs z = -1 < 0: the column stays and the solver
    // reports the unbounded problem.
    #[test]
    fn inadmissible_multiplier_is_not_fixed() {
        let A = CscMatrix::new(1, 1, vec![0, 1], vec![0], vec![-1.]);
        let s = solve(
            &CscMatrix::zeros((1, 1)),
            &[-1.],
            &A,
            &[0.],
            &[NonnegativeConeT(1)],
            true,
        );
        assert!(s.data.presolver.is_none());
        assert_eq!(s.solution.status, SolverStatus::DualInfeasible);
    }
}
