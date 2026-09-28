//! Eliminate scalar variables with one or two local bound rows.
//!
//! Negative bound pivots are eliminated first, leaving one positive scalar
//! pivot per variable: P_ii + shift + sum(a_bound^2 / (Hs_bound + shift)).
//! This is the ordinary signed LDL operation on a 2x2 or 3x3 leaf. Equality
//! multipliers and unbounded variables form the border. The parent KKT solver
//! still regularizes and refines against the complete original operator.
use super::*;
use crate::solver::cones::{CompositeCone, SupportedCone};

impl<T: FloatT> ArrowLDLSolver<T> {
    pub(crate) fn try_local_bounds(
        k: &CscMatrix<T>,
        signs: &[i8],
        a: &CscMatrix<T>,
        cones: &CompositeCone<T>,
        settings: &CoreSettings<T>,
    ) -> Option<Self> {
        // Binary64 requires the packed faer Schur kernel.
        if T::precision_bits() <= 53 && !cfg!(feature = "faer-sparse") {
            return None;
        }
        let n = a.n;
        if k.n != n + a.m || k.m != k.n || signs.len() != k.n {
            return None;
        }
        let mut bound_rows = vec![false; a.m];
        let mut trunk = Vec::new();
        for (cone, rows) in cones.iter().zip(&cones.rng_cones) {
            match cone {
                SupportedCone::ZeroCone(_) => trunk.extend(rows.clone().map(|r| n + r)),
                SupportedCone::NonnegativeCone(_) => bound_rows[rows.clone()].fill(true),
                _ => return None,
            }
        }
        // Every nonnegative row must be a bound on exactly one variable.
        let mut seen = vec![false; a.m];
        let mut groups = Vec::new();
        for col in 0..n {
            let mut ids = Vec::new();
            for p in a.colptr[col]..a.colptr[col + 1] {
                let row = a.rowval[p];
                if bound_rows[row] {
                    if seen[row] {
                        return None;
                    }
                    seen[row] = true;
                    ids.push(n + row);
                }
            }
            if ids.is_empty() {
                // Eliminate the negative equality border before these positive
                // pivots, avoiding division by a free variable's tiny shift.
                trunk.push(col);
            } else {
                if ids.len() > 2 {
                    return None;
                }
                ids.push(col);
                groups.push(ids);
            }
        }
        if bound_rows
            .iter()
            .zip(&seen)
            .any(|(&bound, &used)| bound && !used)
            || groups.len() < 64
            || trunk.is_empty()
            || trunk.len() > 128
        {
            return None;
        }
        let t = trunk.len() as u128;
        let cells = groups
            .iter()
            .map(|g| {
                let g = g.len() as u128;
                2 * g * g + 3 * t + 4 * g
            })
            .sum::<u128>()
            + 6 * t * t
            + 8 * k.n as u128;
        let packed_bytes = if cfg!(feature = "faer-sparse")
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>()
        {
            (2 * groups.len() as u128 * t + t * t) * 8
        } else if T::precision_bits() > 64 {
            (2 * groups.len() as u128 * t + t * t) * std::mem::size_of::<T>() as u128
        } else {
            0
        };
        if cells * std::mem::size_of::<T>() as u128 + packed_bytes > ARROW_MAX_BYTES {
            return None;
        }
        let mut owner = vec![usize::MAX; k.n];
        for (group, ids) in groups.iter().enumerate() {
            for &id in ids {
                owner[id] = group;
            }
        }
        if signs
            .iter()
            .enumerate()
            .any(|(i, &s)| s != if i < n { 1 } else { -1 })
        {
            return None;
        }
        // Diagonal P only. Reject cross-leaf edges and malformed CSC before
        // installing the persistent block map; stored zeros remain edges.
        for j in 0..k.n {
            for p in k.colptr[j]..k.colptr[j + 1] {
                let i = k.rowval[p];
                if i > j
                    || (p > k.colptr[j] && k.rowval[p - 1] >= i)
                    || (j < n && i != j)
                    || (owner[i] != usize::MAX && owner[j] != usize::MAX && owner[i] != owner[j])
                {
                    return None;
                }
                if owner[i] != owner[j]
                    && ((owner[i] != usize::MAX && i >= n) || (owner[j] != usize::MAX && j >= n))
                {
                    return None;
                }
            }
        }
        Some(Self::from_groups(
            k,
            signs,
            settings,
            groups,
            trunk,
            Some(LocalStructure::Bounds),
        ))
    }
}

// Persistent column-major panels avoid gathering individual leaves for every
// Schur entry. Only actual binary64 problems enter this path; MPFR never casts.
#[cfg(feature = "faer-sparse")]
pub(super) struct BoundPanels {
    y: Vec<f64>,
    z: Vec<f64>,
    gram: Vec<f64>,
    rhs: Vec<f64>,
    border: Vec<f64>,
    pub(super) residual_entries: Vec<(usize, usize, usize)>,
}

#[cfg(feature = "faer-sparse")]
impl BoundPanels {
    pub(super) fn new(capacity: usize, border: usize) -> Self {
        Self {
            y: Vec::with_capacity(capacity * border),
            z: Vec::with_capacity(capacity * border),
            gram: vec![0.0; border * border],
            rhs: Vec::new(),
            border: Vec::new(),
            residual_entries: Vec::new(),
        }
    }
}

#[cfg(feature = "faer-sparse")]
impl<T: FloatT> ArrowLDLSolver<T> {
    pub(super) fn assemble_bound_schur_faer(&mut self) -> bool {
        let Some(panels) = self.bound_panels.as_mut() else {
            return false;
        };
        let n = self.leaves.len();
        let t = self.trunk.len();
        panels.y.resize(n * t, 0.0);
        panels.z.resize(n * t, 0.0);
        for j in 0..t {
            for (i, leaf) in self.leaves.iter().enumerate() {
                panels.y[i + j * n] = leaf.coupling_values()[j].to_f64().unwrap();
                panels.z[i + j * n] = (leaf.coupling_values()[j]
                    * leaf.factor.dinv[leaf.coupling_start])
                    .to_f64()
                    .unwrap();
            }
        }
        // Serial SIMD reduction keeps the same result at every requested
        // thread count. Leaf work can still use the caller's thread pool.
        faer::linalg::matmul::matmul(
            faer::MatMut::from_column_major_slice_mut(&mut panels.gram, t, t),
            faer::Accum::Replace,
            faer::MatRef::from_column_major_slice(&panels.y, n, t).transpose(),
            faer::MatRef::from_column_major_slice(&panels.z, n, t),
            1.0,
            faer::Par::Seq,
        );
        for j in 0..t {
            for i in j..t {
                let value = self.c[i + j * t] - T::from_f64(panels.gram[i + j * t]).unwrap();
                self.s[i + j * t] = value;
                self.s[j + i * t] = value;
            }
        }
        true
    }
}

#[cfg(feature = "faer-sparse")]
impl<T: FloatT> ArrowLDLSolver<T> {
    // Share single and batched RHS kernels, including refinement solves.
    pub(super) fn solve_bounds_faer(&mut self, x: &mut [T], b: &[T], cols: usize) -> bool {
        let Some(panels) = self.bound_panels.as_mut() else {
            return false;
        };
        let n = self.leaves.len();
        let t = self.trunk.len();
        panels.rhs.resize(n * cols, 0.0);
        panels.border.resize(t * cols, 0.0);
        for (i, leaf) in self.leaves.iter_mut().enumerate() {
            leaf.first_many(b, self.n, cols, 1);
            for c in 0..cols {
                panels.rhs[i + c * n] = leaf.batch_v[leaf.coupling_start * cols + c]
                    .to_f64()
                    .unwrap();
            }
        }
        faer::linalg::matmul::matmul(
            faer::MatMut::from_column_major_slice_mut(&mut panels.border, t, cols),
            faer::Accum::Replace,
            faer::MatRef::from_column_major_slice(&panels.y, n, t).transpose(),
            faer::MatRef::from_column_major_slice(&panels.rhs, n, cols),
            1.0,
            faer::Par::Seq,
        );
        for c in 0..cols {
            for (j, &id) in self.trunk.iter().enumerate() {
                self.tx[j] = b[id + c * self.n] - T::from_f64(panels.border[j + c * t]).unwrap();
            }
            self.tf.solve(&mut self.tx);
            for (j, &id) in self.trunk.iter().enumerate() {
                x[id + c * self.n] = self.tx[j];
                panels.border[j + c * t] = self.tx[j].to_f64().unwrap();
            }
        }
        faer::linalg::matmul::matmul(
            faer::MatMut::from_column_major_slice_mut(&mut panels.rhs, n, cols),
            faer::Accum::Replace,
            faer::MatRef::from_column_major_slice(&panels.y, n, t),
            faer::MatRef::from_column_major_slice(&panels.border, t, cols),
            1.0,
            faer::Par::Seq,
        );
        for (i, leaf) in self.leaves.iter_mut().enumerate() {
            for r in 0..leaf.ids.len() {
                for c in 0..cols {
                    let value = if r == leaf.coupling_start {
                        T::from_f64(panels.rhs[i + c * n]).unwrap()
                    } else {
                        T::zero()
                    };
                    leaf.batch_w[r * cols + c] =
                        (leaf.batch_w[r * cols + c] - value) * leaf.factor.dinv[r];
                }
            }
            leaf.factor.backward_many(&mut leaf.batch_w, cols);
            for (r, &id) in leaf.ids.iter().enumerate() {
                for c in 0..cols {
                    x[id + c * self.n] = leaf.batch_w[r * cols + c];
                }
            }
        }
        true
    }
}

#[cfg(feature = "faer-sparse")]
impl<T: FloatT> ArrowLDLSolver<T> {
    pub(super) fn residual_bounds_faer(
        &self,
        k: &CscMatrix<T>,
        out: &mut [T],
        rhs: &[T],
        point: &[T],
    ) -> Option<T> {
        let panels = self.bound_panels.as_ref()?;
        let (n, t) = (self.leaves.len(), self.trunk.len());
        let primal: Vec<f64> = self
            .leaves
            .iter()
            .map(|l| point[l.ids[l.coupling_start]].to_f64().unwrap())
            .collect();
        let border: Vec<f64> = self
            .trunk
            .iter()
            .map(|&i| point[i].to_f64().unwrap())
            .collect();
        let mut product = vec![0.0; n + t];
        let (leaf_product, border_product) = product.split_at_mut(n);
        faer::linalg::matmul::matmul(
            faer::MatMut::from_column_major_slice_mut(leaf_product, n, 1),
            faer::Accum::Replace,
            faer::MatRef::from_column_major_slice(&panels.y, n, t),
            faer::MatRef::from_column_major_slice(&border, t, 1),
            1.0,
            faer::Par::Seq,
        );
        faer::linalg::matmul::matmul(
            faer::MatMut::from_column_major_slice_mut(border_product, t, 1),
            faer::Accum::Replace,
            faer::MatRef::from_column_major_slice(&panels.y, n, t).transpose(),
            faer::MatRef::from_column_major_slice(&primal, n, 1),
            1.0,
            faer::Par::Seq,
        );
        out.copy_from_slice(rhs);
        for (i, l) in self.leaves.iter().enumerate() {
            out[l.ids[l.coupling_start]] -= T::from_f64(product[i]).unwrap();
        }
        for (i, &id) in self.trunk.iter().enumerate() {
            out[id] -= T::from_f64(product[n + i]).unwrap();
        }
        // Read current unshifted values from the parent's KKT. Factor storage
        // contains regularized diagonals and must never define this residual.
        for &(q, i, j) in &panels.residual_entries {
            out[i] = (-k.nzval[q]).mul_add(point[j], out[i]);
            if i != j {
                out[j] = (-k.nzval[q]).mul_add(point[i], out[j]);
            }
        }
        Some(if out.is_finite() {
            out.norm_inf()
        } else {
            T::infinity()
        })
    }
}

pub(super) struct ExactBoundPanels<T> {
    y: Vec<T>,
    z: Vec<T>,
    gram: Vec<T>,
}
impl<T: FloatT> ExactBoundPanels<T> {
    pub(super) fn new(n: usize, t: usize) -> Self {
        Self {
            y: vec![T::zero(); n * t],
            z: vec![T::zero(); n * t],
            gram: vec![T::zero(); t * t],
        }
    }
}
impl<T: FloatT> ArrowLDLSolver<T> {
    pub(super) fn assemble_bound_schur_exact(&mut self) -> bool {
        let Some(panels) = self.exact_bound_panels.as_mut() else {
            return false;
        };
        let (n, t) = (self.leaves.len(), self.trunk.len());
        for j in 0..t {
            for (i, leaf) in self.leaves.iter().enumerate() {
                panels.y[i + j * n] = leaf.coupling_values()[j];
                panels.z[i + j * n] =
                    leaf.coupling_values()[j] * leaf.factor.dinv[leaf.coupling_start];
            }
        }
        // Upper Z^T Y is the transpose of the old lower Y^T Z. This keeps
        // exactly the same chosen entries even though rounding Z can make
        // the unmirrored product differ in the last bit across the diagonal.
        {
            let (y, z) = (&panels.y, &panels.z);
            let column = |(j, col): (usize, &mut [T])| {
                for (i, v) in col.iter_mut().enumerate().take(j + 1) {
                    *v = T::dot_fma(z[i * n..(i + 1) * n].iter().zip(&y[j * n..(j + 1) * n]));
                }
            };
            if let Some(pool) = &self.pool {
                pool.install(|| panels.gram.par_chunks_mut(t).enumerate().for_each(column));
            } else {
                panels.gram.chunks_mut(t).enumerate().for_each(column);
            }
        }
        for j in 0..t {
            for i in 0..=j {
                let v = self.c[j + i * t] - panels.gram[i + j * t];
                self.s[j + i * t] = v;
                self.s[i + j * t] = v;
            }
        }
        true
    }
}
