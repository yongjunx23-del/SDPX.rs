//! Eliminate any mix of small local cones around an equality border.
//!
//! Each orthant row and each second-order cone is a unit. A variable touching
//! at most `SHARED_UNITS` units joins them, so units sharing such a variable
//! (or a P entry) merge into one leaf: its cone rows (negative) followed by
//! its variables (positive). Equality rows, free variables and variables
//! touching more units form the border. Leaves couple to the border through
//! their variables; a leaf whose cone rows also touch a border variable
//! couples through all of its rows. This covers standard-form problems
//! (`Ax = b`, `x ∈ K` with identity cone rows) as well as the specialized
//! SOC and bound layouts, which are tried first. MPFR only: its border Schur
//! complement is exact (residue products) and rounded once, while a binary64
//! border loses the accuracy of the augmented factorization on
//! ill-conditioned problems. The caller still owns static shifts, their
//! escalation, and refinement against the original augmented operator. No
//! equality is dropped.
use super::*;
use crate::solver::cones::{CompositeCone, Cone, SupportedCone};

/// Largest leaf (cone rows plus variables) factored densely.
const LEAF_MAX: usize = 32;
/// A variable touching more units than this is shared and stays in the border.
const SHARED_UNITS: usize = 4;
/// Largest border; its dense factor costs t³/3.
const BORDER_MAX: usize = 2048;

impl<T: FloatT> ArrowLDLSolver<T> {
    pub(crate) fn try_local_cones(
        k: &CscMatrix<T>,
        signs: &[i8],
        a: &CscMatrix<T>,
        cones: &CompositeCone<T>,
        settings: &CoreSettings<T>,
    ) -> Option<Self> {
        if T::precision_bits() <= 64 {
            return None;
        }
        let n = a.n;
        debug_assert!(k.n == n + a.m && k.m == k.n && signs.len() == k.n);
        let mut unit_of_row = vec![usize::MAX; a.m];
        let mut units: Vec<std::ops::Range<usize>> = Vec::new();
        let mut trunk = Vec::new();
        let mut soc = false;
        for (cone, rows) in cones.iter().zip(&cones.rng_cones) {
            match cone {
                SupportedCone::ZeroCone(_) => trunk.extend(rows.clone().map(|r| n + r)),
                SupportedCone::NonnegativeCone(_) => {
                    for r in rows.clone() {
                        unit_of_row[r] = units.len();
                        units.push(r..r + 1);
                    }
                }
                SupportedCone::SecondOrderCone(c) if c.numel() <= LEAF_MAX => {
                    soc = true;
                    unit_of_row[rows.clone()].fill(units.len());
                    units.push(rows.clone());
                }
                _ => return None,
            }
        }
        // Pure orthant problems keep the bound elimination or QDLDL.
        if !soc {
            return None;
        }
        let mut parent: Vec<usize> = (0..units.len()).collect();
        let mut home = vec![usize::MAX; n];
        let mut touched = Vec::with_capacity(SHARED_UNITS + 1);
        for col in 0..n {
            touched.clear();
            for &r in &a.rowval[a.colptr[col]..a.colptr[col + 1]] {
                let u = unit_of_row[r];
                if u != usize::MAX && !touched.contains(&u) {
                    touched.push(u);
                    if touched.len() > SHARED_UNITS {
                        break;
                    }
                }
            }
            if touched.is_empty() || touched.len() > SHARED_UNITS {
                trunk.push(col);
                continue;
            }
            home[col] = touched[0];
            for &u in &touched[1..] {
                let (x, y) = (find(&mut parent, touched[0]), find(&mut parent, u));
                parent[x] = y;
            }
        }
        // P entries between leaf variables merge their leaves.
        for j in 0..n {
            for &i in &k.rowval[k.colptr[j]..k.colptr[j + 1]] {
                if i < n && i != j && home[i] != usize::MAX && home[j] != usize::MAX {
                    let (x, y) = (find(&mut parent, home[i]), find(&mut parent, home[j]));
                    parent[x] = y;
                }
            }
        }
        let mut leaf_of = vec![usize::MAX; units.len()];
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for (u, rows) in units.iter().enumerate() {
            let root = find(&mut parent, u);
            if leaf_of[root] == usize::MAX {
                leaf_of[root] = groups.len();
                groups.push(Vec::new());
            }
            groups[leaf_of[root]].extend(rows.clone().map(|r| n + r));
        }
        for col in 0..n {
            if home[col] != usize::MAX {
                let root = find(&mut parent, home[col]);
                groups[leaf_of[root]].push(col);
            }
        }
        let t = trunk.len();
        if groups.len() < 8
            || t == 0
            || t > BORDER_MAX
            || groups.iter().any(|g| g.len() > LEAF_MAX)
            || trunk.iter().any(|&id| signs[id] != if id < n { 1 } else { -1 })
            || groups
                .iter()
                .flatten()
                .any(|&id| signs[id] != if id < n { 1 } else { -1 })
        {
            return None;
        }
        // Dense leaf blocks, B/Y/Z rows over each leaf's distinct border
        // columns, the dense border and solve vectors.
        let mut owner = vec![usize::MAX; k.n];
        for (group, ids) in groups.iter().enumerate() {
            for &id in ids {
                owner[id] = group;
            }
        }
        let mut border = vec![usize::MAX; k.n];
        for (position, &id) in trunk.iter().enumerate() {
            border[id] = position;
        }
        // Per leaf: border columns, coupling entries and cone-row coupling.
        let mut links: Vec<(Vec<usize>, u128, bool)> = vec![(Vec::new(), 0, false); groups.len()];
        for j in 0..k.n {
            for &i in &k.rowval[k.colptr[j]..k.colptr[j + 1]] {
                let (inner, outer) = match (owner[i], owner[j]) {
                    (x, usize::MAX) if x != usize::MAX => (i, j),
                    (usize::MAX, y) if y != usize::MAX => (j, i),
                    _ => continue,
                };
                let link = &mut links[owner[inner]];
                link.0.push(border[outer]);
                link.1 += 1;
                link.2 |= inner >= n;
            }
        }
        let mut cells = 4 * (t as u128).pow(2) + 4 * k.n as u128;
        for (g, link) in groups.iter().zip(&mut links) {
            link.0.sort_unstable();
            link.0.dedup();
            let rows = g.iter().filter(|&&id| id >= n).count();
            let width = if link.2 { g.len() } else { g.len() - rows } as u128;
            let (g, q) = (g.len() as u128, link.0.len() as u128);
            cells += 2 * g * g + (3 * width).saturating_sub(1) * q + 8 * g;
        }
        let entries: u128 = links.iter().map(|link| link.1).sum();
        if cells * std::mem::size_of::<T>() as u128
            > ARROW_MAX_BYTES + k.nzval.len() as u128 * std::mem::size_of::<T>() as u128
            || (t as u128).pow(2) > 8 * entries
        {
            return None;
        }
        Some(Self::from_groups(
            k,
            signs,
            settings,
            groups,
            trunk,
            Some(LocalStructure::Cones),
        ))
    }
}
