//! SDPB-style dual feasibility "jump" for `find_dual_feasible`.
//!
//! Keep y (the zero-cone part of z) and ask whether c - B y is a sum of
//! squares in the sampled bases: give each Gram matrix the smallest
//! (Frobenius) correction that satisfies every sampled dual equality
//! exactly, then factor the corrected matrices. Columns pair with
//! weight * q_k' sym(Y_rs) q_k (see `SampledBlock`), so for one level pair
//! (r, s) the correction is sum_k w_k f q_k q_k' with f = 1 on the diagonal
//! and 1/2 off it, and w solves G w = residual with
//! G_kl = f * sum over parities of (q_k . q_l)^2.
use super::*;

/// Cholesky in place (row-major); false at the first nonpositive pivot.
fn cholesky<T: FloatT>(a: &mut [T], n: usize) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if !(d > T::zero()) {
            return false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut v = a[i * n + j];
            for k in 0..j {
                v -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = v / d;
        }
    }
    true
}

/// Full symmetric matrix from svec (upper triangle by columns, sqrt(2) off-diagonal).
fn unpack<T: FloatT>(svec: &[T], side: usize) -> Vec<T> {
    let mut m = vec![T::zero(); side * side];
    let mut k = 0;
    for c in 0..side {
        for r in 0..=c {
            let v = if r == c {
                svec[k]
            } else {
                svec[k] * T::FRAC_1_SQRT_2()
            };
            m[r * side + c] = v;
            m[c * side + r] = v;
            k += 1;
        }
    }
    m
}

impl<T: FloatT> SampledOperator<T> {
    /// Whether original-coordinate `z` passes the jump test for the input
    /// objective `q`. Blocks sharing a column range are the parities of one
    /// polynomial matrix block.
    pub(crate) fn dual_jump(&self, q: &[T], z: &[T]) -> bool {
        // t_p = -(q_p + (linear' z)_p) / weight_p: the value <Q_p, Y> must take.
        let lin = &self.linear;
        let column = |p: usize| {
            (lin.colptr[p]..lin.colptr[p + 1])
                .fold(q[p], |acc, i| acc + lin.nzval[i] * z[lin.rowval[i]])
        };
        let half: T = (0.5).as_T();
        let mut start = 0;
        while start < self.blocks.len() {
            let first = &self.blocks[start];
            let end = start
                + self.blocks[start..]
                    .iter()
                    .take_while(|b| b.column_start == first.column_start)
                    .count();
            let parts = &self.blocks[start..end];
            start = end;
            let (dim, np) = (first.dim, first.basis_cols);
            if np == 0 {
                continue;
            }
            debug_assert!(parts
                .iter()
                .all(|b| b.dim == dim && b.basis_cols == np && b.weights == first.weights));
            let mut grams: Vec<(usize, usize, Vec<T>)> = parts
                .iter()
                .map(|b| {
                    let side = dim * b.basis_rows;
                    let len = side * (side + 1) / 2;
                    (
                        b.basis_rows,
                        side,
                        unpack(&z[b.row_start..b.row_start + len], side),
                    )
                })
                .collect();
            // q_k' Y[r, s] q_l of one part.
            let form = |b: &SampledBlock<T>, y: &[T], side: usize, r: usize, s: usize, k: usize| {
                let h = b.basis_rows;
                let qk = &b.basis[k * h..(k + 1) * h];
                let mut total = T::zero();
                for a in 0..h {
                    let row = &y[(r * h + a) * side + s * h..(r * h + a) * side + s * h + h];
                    total += qk[a]
                        * row
                            .iter()
                            .zip(qk)
                            .fold(T::zero(), |acc, (&v, &w)| acc + v * w);
                }
                total
            };
            let mut pair = 0;
            for s in 0..dim {
                for r in 0..=s {
                    let factor = if r == s { T::one() } else { half };
                    let mut residual: Vec<T> = (0..np)
                        .map(|k| {
                            let p = pair * np + k;
                            let target = -column(first.column_start + p) / first.weights[p];
                            let value = parts.iter().zip(&grams).fold(
                                T::zero(),
                                |acc, (b, (_, side, y))| {
                                    acc + (form(b, y, *side, r, s, k) + form(b, y, *side, s, r, k))
                                        * half
                                },
                            );
                            target - value
                        })
                        .collect();
                    let mut g = vec![T::zero(); np * np];
                    for b in parts {
                        let h = b.basis_rows;
                        for k in 0..np {
                            for l in 0..=k {
                                let d = b.basis[k * h..(k + 1) * h]
                                    .iter()
                                    .zip(&b.basis[l * h..(l + 1) * h])
                                    .fold(T::zero(), |acc, (&x, &y)| acc + x * y);
                                g[k * np + l] += factor * d * d;
                            }
                        }
                    }
                    if !cholesky(&mut g, np) {
                        return false;
                    }
                    for i in 0..np {
                        for k in 0..i {
                            let v = g[i * np + k] * residual[k];
                            residual[i] -= v;
                        }
                        residual[i] /= g[i * np + i];
                    }
                    for i in (0..np).rev() {
                        for k in i + 1..np {
                            let v = g[k * np + i] * residual[k];
                            residual[i] -= v;
                        }
                        residual[i] /= g[i * np + i];
                    }
                    for (b, (h, side, y)) in parts.iter().zip(grams.iter_mut()) {
                        let (h, side) = (*h, *side);
                        for a in 0..h {
                            for c in 0..h {
                                let d = (0..np).fold(T::zero(), |acc, k| {
                                    acc + residual[k] * b.basis[k * h + a] * b.basis[k * h + c]
                                }) * factor;
                                y[(r * h + a) * side + s * h + c] += d;
                                if r != s {
                                    y[(s * h + a) * side + r * h + c] += d;
                                }
                            }
                        }
                    }
                    pair += 1;
                }
            }
            for (_, side, y) in grams.iter_mut() {
                if !cholesky(y, *side) {
                    return false;
                }
            }
        }
        true
    }
}
