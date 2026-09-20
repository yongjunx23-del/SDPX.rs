#![allow(non_snake_case)]
//! Multi-leaf arrow LDLᵀ solver for quasidefinite KKT systems whose
//! positive-sign variables decompose into several structurally disconnected
//! components.  Each component factors as an independent dense leaf while the
//! negative-sign variables form a small border Schur complement:
//!
//! K = [diag(H_i)  B_i;  B_iᵀ  C],   H_i ≻ 0,   C ≺ 0.
//!
//! Compared to the general sparse QDLDL this trades fill-reducing orderings
//! for dense leaf factorizations that parallelize trivially and reuse the
//! same coupling transforms across refactorizations.  Eligibility is decided
//! once from the structure (stored zeros count as edges); refactor failures
//! delegate to a lazily constructed QDLDL fallback, so the path can never do
//! worse than the baseline backend.
use crate::algebra::*;
use crate::solver::core::kktsolvers::direct::{
    BoxedDirectLDLSolver, DirectLDLSolver, DirectLDLSolverReqs,
};
use crate::solver::core::kktsolvers::{HasLinearSolverInfo, LinearSolverInfo};
use crate::solver::core::CoreSettings;
use rayon::prelude::*;
use std::sync::Arc;

/// Hard cap on the dense working set of the arrow representation.  Larger
/// systems keep using QDLDL rather than densifying without bound.
const ARROW_MAX_BYTES: u128 = 512 * 1024 * 1024;

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Unpivoted dense LDLᵀ with the same per-pivot dynamic regularization
/// semantics as QDLDL: a pivot that violates its expected sign by more than
/// `eps` is clamped to `delta * sign` and counted.
struct DenseLeaf<T> {
    n: usize,
    l: Vec<T>,
    dinv: Vec<T>,
}

impl<T: FloatT> DenseLeaf<T> {
    fn new(n: usize) -> Self {
        Self {
            n,
            l: vec![T::zero(); n * n],
            dinv: vec![T::zero(); n],
        }
    }

    fn factor(
        &mut self,
        a: &[T],
        sign: i8,
        reg: Option<(T, T)>,
        regularize_count: &mut usize,
    ) -> Result<(), &'static str> {
        self.l.copy_from_slice(a);
        let n = self.n;
        let s = T::from_i8(sign).unwrap();
        for k in 0..n {
            let mut d = self.l[k + k * n];
            if !d.is_finite() {
                return Err("nonfinite_pivot");
            }
            // Same dynamic regularization as QDLDL: a pivot violating its
            // expected sign past eps is clamped to delta*sign and counted.
            if let Some((eps, delta)) = reg {
                if d * s < eps {
                    d = delta * s;
                    *regularize_count += 1;
                }
            }
            if d.is_zero() {
                return Err("zero_pivot");
            }
            self.dinv[k] = T::one() / d;
            self.l[k + k * n] = T::one();
            for i in k + 1..n {
                self.l[i + k * n] *= self.dinv[k];
            }
            for j in k + 1..n {
                let v = self.l[j + k * n] * d;
                for i in j..n {
                    self.l[i + j * n] = (-self.l[i + k * n]).mul_add(v, self.l[i + j * n]);
                }
            }
        }
        Ok(())
    }

    fn forward(&self, x: &mut [T]) {
        for i in 0..self.n {
            for k in 0..i {
                x[i] = (-self.l[i + k * self.n]).mul_add(x[k], x[i]);
            }
        }
    }

    fn backward(&self, x: &mut [T]) {
        for i in (0..self.n).rev() {
            for k in i + 1..self.n {
                x[i] = (-self.l[k + i * self.n]).mul_add(x[k], x[i]);
            }
        }
    }

    fn solve(&self, x: &mut [T]) {
        self.forward(x);
        for (x, d) in x.iter_mut().zip(&self.dinv) {
            *x *= *d;
        }
        self.backward(x);
    }
}

struct Leaf<T> {
    ids: Vec<usize>,
    h: Vec<T>,
    b: Vec<T>,
    factor: DenseLeaf<T>,
    y: Vec<T>,
    z: Vec<T>,
    contribution: Vec<T>,
    w: Vec<T>,
    v: Vec<T>,
}

impl<T: FloatT> Leaf<T> {
    fn new(ids: Vec<usize>, t: usize) -> Self {
        let g = ids.len();
        Self {
            ids,
            h: vec![T::zero(); g * g],
            b: vec![T::zero(); g * t],
            factor: DenseLeaf::new(g),
            y: vec![T::zero(); g * t],
            z: vec![T::zero(); g * t],
            contribution: vec![T::zero(); t * t],
            w: vec![T::zero(); g],
            v: vec![T::zero(); g],
        }
    }

    /// H = LDLᵀ, Y = L⁻¹B, contribution = YᵀD⁻¹Y accumulated into S = C - ΣYᵀZ.
    fn refactor(
        &mut self,
        t: usize,
        reg: Option<(T, T)>,
        regularize_count: &mut usize,
    ) -> Result<(), &'static str> {
        let g = self.ids.len();
        self.factor.factor(&self.h, 1, reg, regularize_count)?;
        self.y.copy_from_slice(&self.b);
        for j in 0..t {
            self.factor.forward(&mut self.y[j * g..(j + 1) * g]);
        }
        for j in 0..t {
            for i in 0..g {
                self.z[i + j * g] = self.y[i + j * g] * self.factor.dinv[i];
            }
        }
        for j in 0..t {
            for i in 0..=j {
                let s = T::dot_fma((0..g).map(|k| (&self.y[k + i * g], &self.z[k + j * g])));
                self.contribution[i + j * t] = s;
                self.contribution[j + i * t] = s;
            }
        }
        Ok(())
    }

    fn first_solve(&mut self, rhs: &[T]) {
        for (w, &id) in self.w.iter_mut().zip(&self.ids) {
            *w = rhs[id];
        }
        self.factor.forward(&mut self.w);
        for i in 0..self.ids.len() {
            self.v[i] = self.w[i] * self.factor.dinv[i];
        }
    }

    fn second_solve(&mut self, xt: &[T]) {
        let g = self.ids.len();
        for i in 0..g {
            let s = T::dot_fma((0..xt.len()).map(|j| (&self.y[i + j * g], &xt[j])));
            self.w[i] = (self.w[i] - s) * self.factor.dinv[i];
        }
        self.factor.backward(&mut self.w);
    }
}

pub struct ArrowLDLSolver<T: FloatT> {
    matrix: CscMatrix<T>,
    signs: Vec<i8>,
    settings: CoreSettings<T>,
    n: usize,
    trunk: Vec<usize>,
    owner: Vec<usize>,
    local: Vec<usize>,
    leaves: Vec<Leaf<T>>,
    c: Vec<T>,
    s: Vec<T>,
    tf: DenseLeaf<T>,
    tx: Vec<T>,
    pool: Option<Arc<rayon::ThreadPool>>,
    regularize_count: usize,
    fallback: Option<BoxedDirectLDLSolver<T>>,
    use_arrow: bool,
}

impl<T: FloatT> ArrowLDLSolver<T> {
    /// Eligibility is structural only: connected components of the
    /// positive-sign subgraph become leaves, negative-sign variables become
    /// the border.  Returns `None` when the shape is not exploitable, so the
    /// caller transparently keeps QDLDL.
    pub fn try_new(k: &CscMatrix<T>, signs: &[i8], settings: &CoreSettings<T>) -> Option<Self> {
        let n = k.n;
        if k.m != n || signs.len() != n || signs.iter().any(|&s| s != 1 && s != -1) || n == 0 {
            return None;
        }
        let mut parent: Vec<usize> = (0..n).collect();
        for j in 0..n {
            for q in k.colptr[j]..k.colptr[j + 1] {
                let i = k.rowval[q];
                if i > j {
                    return None;
                }
                if signs[i] > 0 && signs[j] > 0 {
                    let a = find(&mut parent, i);
                    let b = find(&mut parent, j);
                    parent[a] = b;
                }
            }
        }
        let mut components: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        let mut trunk = Vec::new();
        for (i, &s) in signs.iter().enumerate() {
            if s < 0 {
                trunk.push(i);
            } else {
                let r = find(&mut parent, i);
                components.entry(r).or_default().push(i);
            }
        }
        let mut groups: Vec<Vec<usize>> = components.into_values().collect();
        groups.sort_by_key(|g| g[0]);
        let t = trunk.len();
        let n_pos = (n - t) as u128;
        let cells: u128 = groups
            .iter()
            .map(|g| {
                let g = g.len() as u128;
                4 * g * g + 4 * g * t as u128 + (t as u128).pow(2) + 8 * g
            })
            .sum::<u128>()
            + 6 * (t as u128).pow(2)
            + 8 * n as u128;
        // Dense work estimates: leaf factorizations ~ sum(g^3)/3, coupling
        // transforms ~ n_pos*t^2/2, trunk factor ~ t^3/3.  Arrow only pays
        // when real dense leaf blocks exist to amortize that dense
        // coupling/trunk work; singleton-leaf structures (e.g. a diagonal
        // positive block in an LP) degenerate to a dense Schur assembly and
        // lose to sparse QDLDL by an order of magnitude.
        let leaf_work: u128 = groups.iter().map(|g| (g.len() as u128).pow(3)).sum();
        let border_work = n_pos * (t as u128).pow(2) + (t as u128).pow(3);
        let work_ok = border_work <= 24 * leaf_work.max(1);
        if std::env::var_os("SDPX_PROFILE").is_some() {
            // Observation-only grouping stats (plan PR-06): positive-sign
            // components, leaf size spread, border size, leaf-border coupling
            // edges and the dense working-set estimate.
            let mut sizes: Vec<usize> = groups.iter().map(|g| g.len()).collect();
            sizes.sort_unstable();
            let coupling = (0..n)
                .flat_map(|j| (k.colptr[j]..k.colptr[j + 1]).map(move |q| (k.rowval[q], j)))
                .filter(|&(i, j)| signs[i] != signs[j])
                .count();
            eprintln!(
                "GROUP_STATS n={n} components={} leaf_min={} leaf_med={} leaf_max={} border={t} coupling_nnz={coupling} dense_mib={:.1} leaf_work={leaf_work} border_work={border_work} eligible={}",
                groups.len(),
                sizes.first().copied().unwrap_or(0),
                sizes.get(sizes.len() / 2).copied().unwrap_or(0),
                sizes.last().copied().unwrap_or(0),
                cells as f64 * std::mem::size_of::<T>() as f64 / 1048576.0,
                groups.len() >= 2
                    && work_ok
                    && cells * std::mem::size_of::<T>() as u128 <= ARROW_MAX_BYTES,
            );
        }
        if groups.len() < 2 {
            return None;
        }
        if !work_ok {
            return None;
        }
        if cells * std::mem::size_of::<T>() as u128 > ARROW_MAX_BYTES {
            return None;
        }
        let mut owner = vec![usize::MAX; n];
        let mut local = vec![0usize; n];
        for (gi, g) in groups.iter().enumerate() {
            for (li, &id) in g.iter().enumerate() {
                owner[id] = gi;
                local[id] = li;
            }
        }
        for (i, &id) in trunk.iter().enumerate() {
            local[id] = i;
        }
        let leaves = groups.into_iter().map(|g| Leaf::new(g, t)).collect();
        Some(Self {
            matrix: k.clone(),
            signs: signs.to_vec(),
            settings: settings.clone(),
            n,
            trunk,
            owner,
            local,
            leaves,
            c: vec![T::zero(); t * t],
            s: vec![T::zero(); t * t],
            tf: DenseLeaf::new(t),
            tx: vec![T::zero(); t],
            pool: None,
            regularize_count: 0,
            fallback: None,
            use_arrow: false,
        })
    }

    /// Rebuild dense leaf/coupling/border blocks from the authoritative CSC
    /// copy.  Structure is fixed; only values are refreshed.
    fn scatter(&mut self) -> bool {
        let t = self.trunk.len();
        for leaf in self.leaves.iter_mut() {
            leaf.h.fill(T::zero());
            leaf.b.fill(T::zero());
        }
        self.c.fill(T::zero());
        for j in 0..self.n {
            for q in self.matrix.colptr[j]..self.matrix.colptr[j + 1] {
                let i = self.matrix.rowval[q];
                let a = self.matrix.nzval[q];
                if !a.is_finite() {
                    return false;
                }
                match (self.owner[i], self.owner[j]) {
                    (x, y) if x == usize::MAX && y == usize::MAX => {
                        self.c[self.local[i] + self.local[j] * t] = a;
                        self.c[self.local[j] + self.local[i] * t] = a;
                    }
                    (x, y) if x != usize::MAX && y != usize::MAX => {
                        if x != y {
                            return false; // cross-leaf edge cannot appear
                        }
                        let g = self.leaves[x].ids.len();
                        self.leaves[x].h[self.local[i] + self.local[j] * g] = a;
                        self.leaves[x].h[self.local[j] + self.local[i] * g] = a;
                    }
                    (x, _) if x != usize::MAX => {
                        let g = self.leaves[x].ids.len();
                        self.leaves[x].b[self.local[i] + self.local[j] * g] = a;
                    }
                    (_, y) => {
                        let g = self.leaves[y].ids.len();
                        self.leaves[y].b[self.local[j] + self.local[i] * g] = a;
                    }
                }
            }
        }
        true
    }

    fn factor_arrow(&mut self) -> bool {
        if !self.scatter() {
            return false;
        }
        let reg = self.settings.dynamic_regularization_enable.then_some((
            self.settings.dynamic_regularization_eps,
            self.settings.dynamic_regularization_delta,
        ));
        let t = self.trunk.len();
        let leaves_ok = if let Some(pool) = &self.pool {
            // Regularization counts are diagnostic only; accumulate per-leaf
            // counts after the parallel section to stay deterministic.
            let mut counts = vec![0usize; self.leaves.len()];
            let ok = pool.install(|| {
                self.leaves
                    .par_iter_mut()
                    .zip(counts.par_iter_mut())
                    .map(|(leaf, c)| leaf.refactor(t, reg, c))
                    .collect::<Result<Vec<_>, _>>()
            });
            self.regularize_count += counts.iter().sum::<usize>();
            ok.is_ok()
        } else {
            let count = &mut self.regularize_count;
            self.leaves
                .iter_mut()
                .try_for_each(|leaf| leaf.refactor(t, reg, count))
                .is_ok()
        };
        if !leaves_ok {
            return false;
        }
        self.s.copy_from_slice(&self.c);
        // Deterministic merge order regardless of leaf scheduling.
        for leaf in &self.leaves {
            for (s, a) in self.s.iter_mut().zip(&leaf.contribution) {
                *s -= *a;
            }
        }
        let mut border_count = 0usize;
        let ok = self.tf.factor(&self.s, -1, reg, &mut border_count).is_ok();
        self.regularize_count += border_count;
        ok
    }

    fn factor_fallback(&mut self) -> bool {
        if self.fallback.is_none() {
            let solver: BoxedDirectLDLSolver<T> =
                Box::new(super::qdldl::QDLDLDirectLDLSolver::new(
                    &self.matrix,
                    &self.signs,
                    &self.settings,
                    None,
                ));
            self.fallback = Some(solver);
        }
        let solver = self.fallback.as_mut().unwrap();
        // The fallback's internal permuted copy must observe the same values
        // the caller has written into our authoritative CSC (including any
        // temporary regularization shifts already applied to it).
        let indices: Vec<usize> = (0..self.matrix.nzval.len()).collect();
        solver.update_values(&indices, &self.matrix.nzval);
        solver.set_pool(self.pool.clone());
        solver.refactor(&self.matrix)
    }

    fn solve_arrow(&mut self, x: &mut [T], b: &[T]) {
        let t = self.trunk.len();
        if let Some(pool) = &self.pool {
            pool.install(|| self.leaves.par_iter_mut().for_each(|l| l.first_solve(b)));
        } else {
            for leaf in &mut self.leaves {
                leaf.first_solve(b);
            }
        }
        for (v, &i) in self.tx.iter_mut().zip(&self.trunk) {
            *v = b[i];
        }
        for leaf in &self.leaves {
            let g = leaf.ids.len();
            for j in 0..t {
                self.tx[j] -= T::dot_fma((0..g).map(|i| (&leaf.y[i + j * g], &leaf.v[i])));
            }
        }
        self.tf.solve(&mut self.tx);
        let tx = &self.tx;
        if let Some(pool) = &self.pool {
            pool.install(|| self.leaves.par_iter_mut().for_each(|l| l.second_solve(tx)));
        } else {
            for leaf in &mut self.leaves {
                leaf.second_solve(tx);
            }
        }
        for leaf in &self.leaves {
            for (&id, &value) in leaf.ids.iter().zip(&leaf.w) {
                x[id] = value;
            }
        }
        for (&id, &value) in self.trunk.iter().zip(&self.tx) {
            x[id] = value;
        }
    }
}

impl<T: FloatT> DirectLDLSolverReqs for ArrowLDLSolver<T> {
    fn required_matrix_shape() -> MatrixTriangle {
        MatrixTriangle::Triu
    }
}

impl<T: FloatT> HasLinearSolverInfo for ArrowLDLSolver<T> {
    fn linear_solver_info(&self) -> LinearSolverInfo {
        if !self.use_arrow {
            if let Some(solver) = &self.fallback {
                return solver.linear_solver_info();
            }
        }
        LinearSolverInfo {
            name: "arrow".to_string(),
            threads: self.pool.as_ref().map_or(1, |p| p.current_num_threads()),
            direct: true,
            nnzA: self.matrix.nzval.len(),
            nnzL: self
                .leaves
                .iter()
                .map(|l| l.ids.len() * (l.ids.len() + 1) / 2)
                .sum::<usize>()
                + self.trunk.len() * (self.trunk.len() + 1) / 2,
        }
    }
}

impl<T: FloatT> DirectLDLSolver<T> for ArrowLDLSolver<T> {
    fn update_values(&mut self, index: &[usize], values: &[T]) {
        for (&i, &v) in index.iter().zip(values) {
            self.matrix.nzval[i] = v;
        }
    }

    fn scale_values(&mut self, index: &[usize], scale: T) {
        for &i in index {
            self.matrix.nzval[i] *= scale;
        }
    }

    fn offset_values(&mut self, index: &[usize], offset: T, signs: &[i8]) {
        for (&i, &s) in index.iter().zip(signs) {
            self.matrix.nzval[i] += offset * T::from_i8(s).unwrap();
        }
    }

    fn refactor(&mut self, _kkt: &CscMatrix<T>) -> bool {
        self.use_arrow = self.factor_arrow();
        self.use_arrow || self.factor_fallback()
    }

    fn solve(&mut self, kkt: &CscMatrix<T>, x: &mut [T], b: &mut [T]) {
        if !self.use_arrow {
            self.fallback.as_mut().unwrap().solve(kkt, x, b);
            return;
        }
        self.solve_arrow(x, b);
    }

    fn set_pool(&mut self, pool: Option<Arc<rayon::ThreadPool>>) {
        self.pool = pool;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{FromPrimitive, ToPrimitive, Zero};
    use sdpx_arithmetic::MpFloat;

    /// K = [H1 . . B1; . H2 . B2; . . H3 B3; B1' B2' B3' C] with three
    /// positive leaves and a negative border, stored as upper-triangular CSC.
    fn arrow_kkt() -> (CscMatrix<f64>, Vec<i8>) {
        arrow_kkt_merged(false)
    }

    fn arrow_kkt_merged(merge: bool) -> (CscMatrix<f64>, Vec<i8>) {
        // leaves: {0,1}, {2,3}, {4} ; border: {5,6}
        let n = 7;
        let mut dense = vec![vec![0f64; n]; n];
        let h = |i: usize, j: usize, v: f64, d: &mut Vec<Vec<f64>>| {
            d[i][j] = v;
            d[j][i] = v;
        };
        h(0, 0, 4.0, &mut dense);
        h(0, 1, 1.0, &mut dense);
        h(1, 1, 3.0, &mut dense);
        h(2, 2, 5.0, &mut dense);
        h(2, 3, -1.0, &mut dense);
        h(3, 3, 2.0, &mut dense);
        h(4, 4, 6.0, &mut dense);
        dense[5][5] = -2.0;
        dense[6][6] = -3.0;
        dense[0][5] = 0.5;
        dense[2][5] = -0.25;
        dense[4][6] = 0.75;
        dense[1][6] = 0.1;
        dense[5][6] = 0.2;
        if merge {
            dense[1][2] = 0.5; // positive-positive edges merge all leaves
            dense[3][4] = -0.4;
        }
        // Sparse CSC: only actual structural entries are stored.  Positive
        // components {0,1},{2,3},{4} connect only through border {5,6}.
        let mut colptr = vec![0usize];
        let mut rowval = Vec::new();
        let mut nzval = Vec::new();
        for j in 0..n {
            for i in 0..=j {
                if dense[i][j] != 0.0 || i == j {
                    rowval.push(i);
                    nzval.push(dense[i][j]);
                }
            }
            colptr.push(rowval.len());
        }
        (
            CscMatrix::new(n, n, colptr, rowval, nzval),
            vec![1, 1, 1, 1, 1, -1, -1],
        )
    }

    fn reference_solve(k: &CscMatrix<f64>, b: &[f64]) -> Vec<f64> {
        let mut x = vec![0f64; k.n];
        // dense Gaussian elimination for reference
        let mut a = vec![vec![0f64; k.n]; k.n];
        for j in 0..k.n {
            for q in k.colptr[j]..k.colptr[j + 1] {
                a[k.rowval[q]][j] = k.nzval[q];
                a[j][k.rowval[q]] = k.nzval[q];
            }
        }
        let mut rhs = b.to_vec();
        for c in 0..k.n {
            let p = (c..k.n)
                .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
                .unwrap();
            a.swap(c, p);
            rhs.swap(c, p);
            for r in c + 1..k.n {
                let f = a[r][c] / a[c][c];
                for k2 in c..k.n {
                    a[r][k2] -= f * a[c][k2];
                }
                rhs[r] -= f * rhs[c];
            }
        }
        for c in (0..k.n).rev() {
            let mut s = rhs[c];
            for k2 in c + 1..k.n {
                s -= a[c][k2] * x[k2];
            }
            x[c] = s / a[c][c];
        }
        x
    }

    #[test]
    fn arrow_solves_quasidefinite_and_falls_back() {
        let (k, signs) = arrow_kkt();
        let settings = CoreSettings::<f64>::default();
        let mut solver = ArrowLDLSolver::try_new(&k, &signs, &settings).unwrap();
        assert_eq!(solver.leaves.len(), 3);
        assert_eq!(solver.trunk, vec![5, 6]);
        assert!(solver.refactor(&k));
        assert!(solver.use_arrow);
        let b = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.25];
        let mut rhs = b.clone();
        let mut x = vec![0.0; k.n];
        solver.solve(&k, &mut x, &mut rhs);
        let expect = reference_solve(&k, &b);
        for i in 0..k.n {
            assert!((x[i] - expect[i]).abs() < 1e-10, "{x:?} != {expect:?}");
        }
        // single connected positive component: not eligible
        let (k1, s1) = arrow_kkt_merged(true);
        assert!(ArrowLDLSolver::try_new(&k1, &s1, &settings).is_none());
        // QDLDL parity: a wrong-sign leaf pivot is regularized, not rejected.
        // Solver-side updates go through update_values (the authoritative CSC
        // clone lives inside the solver).
        let (k2, s2) = arrow_kkt();
        let diag0 = k2.colptr[1] - 1; // (0,0) is the last entry of column 0
        let mut solver = ArrowLDLSolver::try_new(&k2, &s2, &settings).unwrap();
        solver.update_values(&[diag0], &[-100.0]);
        assert!(solver.refactor(&k2));
        assert!(solver.use_arrow);
        assert!(solver.regularize_count > 0);
        // zero leaf pivot without dynamic regularization: refactor falls
        // back to QDLDL, which applies its own internal regularization
        let (k3, s3) = arrow_kkt();
        let mut settings3 = CoreSettings::<f64>::default();
        settings3.dynamic_regularization_enable = false;
        let diag4 = k3.colptr[5] - 1; // (4,4)
        let mut solver = ArrowLDLSolver::try_new(&k3, &s3, &settings3).unwrap();
        solver.update_values(&[diag4], &[0.0]);
        assert!(solver.refactor(&k3));
        assert!(!solver.use_arrow);
        let mut rhs = b.clone();
        let mut x = vec![0.0; k3.n];
        solver.solve(&k3, &mut x, &mut rhs);
        // the fallback solved the perturbed system (K[4,4] = 0), so the
        // reference must use the same values; QDLDL applies its own pivot
        // floor to the singular leaf so a loose tolerance is expected
        let mut k3p = k3.clone();
        k3p.nzval[diag4] = 0.0;
        let expect = reference_solve(&k3p, &b);
        for i in 0..k3.n {
            assert!((x[i] - expect[i]).abs() < 1e-4, "{x:?} != {expect:?}");
        }
    }

    /// Same fixture under arbitrary precision: the production MPFR path must
    /// factor and solve identically.
    #[test]
    fn arrow_solves_quasidefinite_mpfr() {
        let (k64, signs) = arrow_kkt();
        let b64 = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.25];
        let n = k64.n;
        let k = CscMatrix::<MpFloat<2>>::new(
            n,
            n,
            k64.colptr.clone(),
            k64.rowval.clone(),
            k64.nzval
                .iter()
                .map(|&v| MpFloat::<2>::from_f64(v).unwrap())
                .collect(),
        );
        let b: Vec<MpFloat<2>> = b64
            .iter()
            .map(|&v| MpFloat::<2>::from_f64(v).unwrap())
            .collect();
        let expect = reference_solve(&k64, &b64);
        let settings = CoreSettings::<MpFloat<2>>::default();
        let mut solver = ArrowLDLSolver::try_new(&k, &signs, &settings).unwrap();
        assert!(solver.refactor(&k));
        assert!(solver.use_arrow);
        let mut rhs = b.clone();
        let mut x = vec![MpFloat::<2>::zero(); n];
        solver.solve(&k, &mut x, &mut rhs);
        for i in 0..n {
            assert!(
                (x[i].to_f64().unwrap() - expect[i]).abs() < 1e-10,
                "{x:?} != {expect:?}"
            );
        }
    }

    /// A diagonal positive block (the LP shape) yields only singleton
    /// leaves: the dense trunk/coupling work is not amortized by any real
    /// leaf factorization, so the arrow representation must decline and
    /// leave the system to sparse QDLDL.
    #[test]
    fn arrow_rejects_singleton_leaves() {
        let n = 40;
        let t = 8;
        // Column j stores rows i <= j; positive->border edges sit in the
        // border columns.
        let mut colptr = vec![0usize];
        let mut rowval = Vec::new();
        let mut nzval = Vec::new();
        for j in 0..n + t {
            if j < n {
                rowval.push(j);
                nzval.push(2.0);
            } else {
                for i in (0..n).filter(|i| i % t == j - n) {
                    rowval.push(i);
                    nzval.push(0.25);
                }
                rowval.push(j);
                nzval.push(-1.0);
            }
            colptr.push(rowval.len());
        }
        let k = CscMatrix::new(n + t, n + t, colptr, rowval, nzval);
        let signs = vec![1i8; n]
            .into_iter()
            .chain(vec![-1i8; t])
            .collect::<Vec<_>>();
        assert!(ArrowLDLSolver::try_new(&k, &signs, &CoreSettings::<f64>::default()).is_none());
    }
}
