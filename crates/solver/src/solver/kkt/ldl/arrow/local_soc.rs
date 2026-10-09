//! Local SOC3 elimination beneath the ordinary augmented KKT interface.
//!
//! Each leaf contains three negative cone coordinates followed by its two
//! positive primal coordinates. Eliminating the cone block first leaves a
//! two-by-two positive block. Only equality multipliers remain in the border.
//! The caller still owns static shifts, their escalation, and refinement
//! against the original augmented operator. No equality is dropped.
use super::*;
use crate::solver::cones::{CompositeCone, Cone, SupportedCone};

impl<T: FloatT> ArrowLDLSolver<T> {
    pub(crate) fn try_local_soc(
        k: &CscMatrix<T>,
        signs: &[i8],
        a: &CscMatrix<T>,
        cones: &CompositeCone<T>,
        settings: &CoreSettings<T>,
    ) -> Option<Self> {
        let n = a.n;
        let mut row_owner = vec![usize::MAX; a.m];
        let mut groups = Vec::new();
        let mut trunk = Vec::new();
        for (cone, rows) in cones.iter().zip(&cones.rng_cones) {
            match cone {
                SupportedCone::ZeroCone(_) => trunk.extend(rows.clone().map(|r| n + r)),
                SupportedCone::SecondOrderCone(c) if c.numel() == 3 => {
                    row_owner[rows.clone()].fill(groups.len());
                    groups.push(rows.clone().map(|r| n + r).collect::<Vec<_>>());
                }
                _ => return None,
            }
        }
        debug_assert!(k.n == n + a.m && k.m == k.n && signs.len() == k.n);
        // Tiny problems stay on QDLDL.
        if groups.len() < 8 || trunk.is_empty() {
            return None;
        }
        for col in 0..n {
            let mut owner = None;
            for p in a.colptr[col]..a.colptr[col + 1] {
                let group = row_owner[a.rowval[p]];
                if group == usize::MAX {
                    continue;
                }
                if owner.is_some_and(|previous| previous != group) {
                    return None;
                }
                owner = Some(group);
            }
            let group = owner?;
            groups[group].push(col);
        }
        if groups.iter().any(|g| g.len() != 5) {
            return None;
        }
        let t = trunk.len() as u128;
        // H/L, raw B row1 plus Y/Z, RHS work and the dense border. The dense
        // border must not be much larger than the couplings that fill it.
        let cells = groups.len() as u128 * (2 * 25 + 5 * t + 4 * 5) + 4 * t * t + 4 * k.n as u128;
        let coupling: usize = groups
            .iter()
            .flat_map(|g| &g[3..])
            .map(|&col| a.colptr[col + 1] - a.colptr[col])
            .sum();
        if cells * std::mem::size_of::<T>() as u128 > ARROW_MAX_BYTES
            || t * t > 8 * coupling as u128
        {
            return None;
        }
        let mut owner = vec![usize::MAX; k.n];
        for (group, ids) in groups.iter().enumerate() {
            for (position, &id) in ids.iter().enumerate() {
                debug_assert_eq!(signs[id], if position < 3 { -1 } else { 1 });
                owner[id] = group;
            }
        }
        debug_assert!(trunk.iter().all(|&id| signs[id] == -1));
        // Reject nonlocal P coupling before constructing the block map.
        // Stored zeros count as structural edges, so later data updates
        // cannot silently invalidate this decomposition.
        for j in 0..k.n {
            for p in k.colptr[j]..k.colptr[j + 1] {
                let i = k.rowval[p];
                if owner[i] != usize::MAX && owner[j] != usize::MAX && owner[i] != owner[j] {
                    return None;
                }
                // Only primal coordinates couple leaves to the equality border.
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
            Some(LocalStructure::Soc),
        ))
    }

    pub(super) fn assemble_local_schur(&mut self) {
        let timer = crate::receipt::start();
        #[cfg(feature = "faer-sparse")]
        if self.assemble_bound_schur_faer() {
            crate::receipt::finish("arrow.local_schur", timer);
            return;
        }
        if self.assemble_bound_schur_exact() {
            crate::receipt::finish("arrow.local_schur", timer);
            return;
        }
        let t = self.trunk.len();
        let leaves = &self.leaves;
        if matches!(
            self.local_structure,
            Some(LocalStructure::Soc | LocalStructure::Cones)
        ) && T::precision_bits() > 64
        {
            debug_assert!(self.ranks.is_none());
            let blocks: Vec<_> = leaves
                .iter()
                .filter(|leaf| !leaf.y.is_empty())
                .map(|leaf| {
                    (
                        &leaf.y[..],
                        &leaf.factor.dinv[leaf.coupling_start..],
                        &leaf.coupled[..],
                    )
                })
                .collect();
            // Upper ZᵀY selects the same products as lower YᵀZ below.
            if T::xgemm_blocks_upper_exact(t, &blocks, &mut self.s, self.pool.as_deref()) {
                for j in 0..t {
                    for i in 0..=j {
                        let value = self.c[i + j * t] - self.s[i + j * t];
                        self.s[i + j * t] = value;
                        self.s[j + i * t] = value;
                    }
                }
                crate::receipt::finish("arrow.local_schur", timer);
                return;
            }
            self.s.copy_from_slice(&self.c);
        }
        if T::precision_bits() <= 53 {
            self.assemble_local_schur_f64();
            crate::receipt::finish("arrow.local_schur", timer);
            return;
        }
        // Use the leaf's structurally nonzero coupling suffix.
        // MPFR's dot uses exact accumulation rounded once; each entry has the
        // same leaf/coordinate order at every thread count. No parallel sum.
        // Z = D⁻¹Y is not stored at MPFR; form it for this fallback only.
        let z: Vec<Vec<T>> = leaves
            .iter()
            .map(|leaf| {
                let (start, width) = (leaf.coupling_start, leaf.ids.len() - leaf.coupling_start);
                leaf.y
                    .iter()
                    .enumerate()
                    .map(|(e, &y)| y * leaf.factor.dinv[start + e % width])
                    .collect()
            })
            .collect();
        let classes = &self.classes;
        let column = |(j, values): (usize, &mut [T])| {
            for (i, value) in values.iter_mut().enumerate().skip(j) {
                *value -= T::dot_fma(
                    classes
                        .iter()
                        .filter(|c| c.position[i] != usize::MAX && c.position[j] != usize::MAX)
                        .flat_map(|class| {
                            let (pi, pj) = (class.position[i], class.position[j]);
                            let z = &z;
                            class.leaves.iter().flat_map(move |&l| {
                                let leaf = &leaves[l];
                                let width = leaf.ids.len() - leaf.coupling_start;
                                (0..width)
                                    .map(move |r| (&leaf.y[r + pi * width], &z[l][r + pj * width]))
                            })
                        }),
                );
            }
        };
        if let Some(pool) = &self.pool {
            pool.install(|| self.s.par_chunks_mut(t).enumerate().for_each(column));
        } else {
            self.s.chunks_mut(t).enumerate().for_each(column);
        }
        for j in 0..t {
            for i in j + 1..t {
                self.s[j + i * t] = self.s[i + j * t];
            }
        }
        crate::receipt::finish("arrow.local_schur", timer);
    }

    /// Binary64 `S -= Σ YᵀZ` over the leaves' coupling suffixes, streaming
    /// each leaf once. Fixed chunks of leaves accumulate lower-triangle
    /// partials in leaf order; partials are summed in chunk order, so the
    /// result does not depend on the thread count.
    fn assemble_local_schur_f64(&mut self) {
        const CHUNK: usize = 1024;
        let t = self.trunk.len();
        let tri = triangular_number(t);
        let partial = |leaves: &[Leaf<T>]| {
            let mut acc = vec![T::zero(); tri];
            for leaf in leaves {
                let width = leaf.ids.len() - leaf.coupling_start;
                for (cj, &j) in leaf.coupled.iter().enumerate() {
                    let z = &leaf.z[cj * width..(cj + 1) * width];
                    // Packed lower column j starts at j·t - j(j-1)/2.
                    let column = j * t - j * (j + 1) / 2;
                    for (ci, &i) in leaf.coupled.iter().enumerate().skip(cj) {
                        let y = &leaf.y[ci * width..(ci + 1) * width];
                        let at = column + i;
                        acc[at] = y.iter().zip(z).fold(acc[at], |v, (&a, &b)| a.mul_add(b, v));
                    }
                }
            }
            acc
        };
        let leaves = &self.leaves;
        let parts: Vec<Vec<T>> = match &self.pool {
            Some(pool) if leaves.len() > CHUNK => {
                pool.install(|| leaves.par_chunks(CHUNK).map(partial).collect())
            }
            _ => leaves.chunks(CHUNK).map(partial).collect(),
        };
        let mut at = 0;
        for j in 0..t {
            for i in j..t {
                let total = parts.iter().fold(T::zero(), |v, p| v + p[at]);
                let value = self.s[i + j * t] - total;
                self.s[i + j * t] = value;
                self.s[j + i * t] = value;
                at += 1;
            }
        }
    }
}
