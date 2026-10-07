//! Eliminate any mix of small local cones around an equality border.
//!
//! Each orthant row and each second-order cone is a unit. Variables join the
//! units they touch (fewest units first) while the merged leaf stays within
//! `LEAF_MAX` coordinates, so units sharing such a variable (or a P entry)
//! merge into one leaf: its cone rows (negative) followed by its variables
//! (positive). Equality rows, free variables, variables shared by more than
//! `SHARED_UNITS` units and variables that would grow a leaf past the limit
//! form the border, as do cones too large for a leaf. Leaves couple to the border through
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

/// Largest leaf (cone rows, expansion coordinates and variables) factored
/// densely.
const LEAF_MAX: usize = 64;
/// Variables shared by more units stay in the border: merging them into one
/// leaf took about four times as many refinement solves (journal 2026-10-07).
const SHARED_UNITS: usize = 4;

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
        debug_assert!(k.n >= n + a.m && k.m == k.n && signs.len() == k.n);
        // Unit coordinates: cone rows, then a large SOC's two sparse-expansion
        // coordinates (numbered after all rows, in cone order).
        let mut unit_of_row = vec![usize::MAX; a.m];
        let mut units: Vec<Vec<usize>> = Vec::new();
        let mut trunk = Vec::new();
        let mut expansion = n + a.m;
        for (cone, rows) in cones.iter().zip(&cones.rng_cones) {
            match cone {
                SupportedCone::ZeroCone(_) => trunk.extend(rows.clone().map(|r| n + r)),
                SupportedCone::NonnegativeCone(_) => {
                    for r in rows.clone() {
                        unit_of_row[r] = units.len();
                        units.push(vec![n + r]);
                    }
                }
                SupportedCone::SecondOrderCone(_) => {
                    let mut ids: Vec<usize> = rows.clone().map(|r| n + r).collect();
                    if cone.is_sparse_expandable() {
                        ids.extend([expansion, expansion + 1]);
                        expansion += 2;
                    }
                    // A cone too large for a leaf joins the border.
                    if ids.len() > LEAF_MAX {
                        trunk.extend(ids);
                    } else {
                        unit_of_row[rows.clone()].fill(units.len());
                        units.push(ids);
                    }
                }
                _ => return None,
            }
        }
        debug_assert_eq!(expansion, k.n);
        let mut touches: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (col, touched) in touches.iter_mut().enumerate() {
            for &r in &a.rowval[a.colptr[col]..a.colptr[col + 1]] {
                let u = unit_of_row[r];
                if u != usize::MAX && !touched.contains(&u) {
                    touched.push(u);
                    if touched.len() > LEAF_MAX {
                        break;
                    }
                }
            }
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&col| (touches[col].len(), col));
        let mut parent: Vec<usize> = (0..units.len()).collect();
        let mut size: Vec<usize> = units.iter().map(Vec::len).collect();
        let mut home = vec![usize::MAX; n];
        let mut roots = Vec::new();
        for col in order {
            roots.clear();
            roots.extend(touches[col].iter().map(|&u| find(&mut parent, u)));
            roots.sort_unstable();
            roots.dedup();
            let total = roots.iter().map(|&r| size[r]).sum::<usize>() + 1;
            if roots.is_empty() || total > LEAF_MAX || touches[col].len() > SHARED_UNITS {
                trunk.push(col);
                continue;
            }
            for &r in &roots[1..] {
                parent[r] = roots[0];
            }
            size[roots[0]] = total;
            home[col] = roots[0];
        }
        // P entries between leaf variables merge their leaves.
        for j in 0..n {
            for &i in &k.rowval[k.colptr[j]..k.colptr[j + 1]] {
                if i < n && i != j && home[i] != usize::MAX && home[j] != usize::MAX {
                    let (x, y) = (find(&mut parent, home[i]), find(&mut parent, home[j]));
                    if x != y {
                        if size[x] + size[y] > LEAF_MAX {
                            return None;
                        }
                        parent[x] = y;
                        size[y] += size[x];
                    }
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
            groups[leaf_of[root]].extend(rows.iter().copied());
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
            || groups.iter().any(|g| g.len() > LEAF_MAX)
            || trunk
                .iter()
                .chain(groups.iter().flatten())
                .any(|&id| id < n + a.m && signs[id] != if id < n { 1 } else { -1 })
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
                debug_assert!(inner < n + a.m);
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
        // Border Schur couplings: each leaf's border columns form a clique,
        // plus direct border entries. Within each sign (equality rows before
        // free and shared variables), eliminate the least coupled border
        // coordinates first, as a minimum-degree ordering of the sparse
        // augmented system would; eliminating the densest rows first made
        // ill-conditioned final factorizations overflow (journal 2026-10-07).
        let words = t.div_ceil(64);
        let mut adjacent = vec![0u64; t * words];
        let mut seen = std::collections::HashSet::new();
        for (columns, _, _) in &links {
            if !seen.insert(&columns[..]) {
                continue;
            }
            for &a in columns {
                for &b in columns {
                    adjacent[a * words + b / 64] |= 1 << (b % 64);
                }
            }
        }
        for j in 0..k.n {
            for &i in &k.rowval[k.colptr[j]..k.colptr[j + 1]] {
                if border[i] != usize::MAX && border[j] != usize::MAX {
                    let (a, b) = (border[i], border[j]);
                    adjacent[a * words + b / 64] |= 1 << (b % 64);
                    adjacent[b * words + a / 64] |= 1 << (a % 64);
                }
            }
        }
        let degree: Vec<u32> = adjacent
            .chunks(words)
            .map(|row| row.iter().map(|w| w.count_ones()).sum())
            .collect();
        let mut order: Vec<usize> = (0..t).collect();
        order.sort_by_key(|&p| (signs[trunk[p]] > 0, degree[p], p));
        let trunk = order.into_iter().map(|p| trunk[p]).collect();
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
