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
        if k.n != n + a.m || k.m != k.n || signs.len() != k.n {
            return None;
        }
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
        // Small borders amortize dense assembly; tiny problems stay on QDLDL.
        if groups.len() < 8 || trunk.is_empty() || trunk.len() > 128 {
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
        // H/L, B/Y/Z, RHS work and the dense border; no per-leaf t² buffers.
        let cells =
            groups.len() as u128 * (2 * 25 + 3 * 2 * t + 4 * 5) + 4 * t * t + 4 * k.n as u128;
        if cells * std::mem::size_of::<T>() as u128 > ARROW_MAX_BYTES {
            return None;
        }
        let mut owner = vec![usize::MAX; k.n];
        for (group, ids) in groups.iter().enumerate() {
            for (position, &id) in ids.iter().enumerate() {
                if signs[id] != if position < 3 { -1 } else { 1 } {
                    return None;
                }
                owner[id] = group;
            }
        }
        if trunk.iter().any(|&id| signs[id] != -1) {
            return None;
        }
        // Reject nonlocal P coupling and any noncanonical CSC pattern before
        // constructing the block map. Stored zeros count as structural edges,
        // so later data updates cannot silently invalidate this decomposition.
        for j in 0..k.n {
            for p in k.colptr[j]..k.colptr[j + 1] {
                let i = k.rowval[p];
                if i > j || (p > k.colptr[j] && k.rowval[p - 1] >= i) {
                    return None;
                }
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
        // A border column couples only to the leaf's primal suffix.
        // MPFR's dot uses exact accumulation rounded once; each entry has the
        // same leaf/coordinate order at every thread count. No parallel sum.
        let column = |(j, values): (usize, &mut [T])| {
            for (i, value) in values.iter_mut().enumerate().skip(j) {
                *value -= T::dot_fma(leaves.iter().flat_map(|leaf| {
                    let width = leaf.ids.len() - leaf.coupling_start;
                    (0..width).map(move |r| (&leaf.y[r + i * width], &leaf.z[r + j * width]))
                }));
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
}
