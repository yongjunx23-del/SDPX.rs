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
//! SOC and bound layouts, which are tried first. MPFR forms the border Schur
//! complement exactly (residue products, rounded once). Binary64 admits the
//! layout only when its dense work beats the sparse LDL ([`CONE_FLOP_RATIO`])
//! and runs each coupling class as stacked BLAS panels ([`ConePanels`]). The
//! caller still owns static shifts, their escalation, and refinement against
//! the original augmented operator. No equality is dropped.
use super::*;
use crate::solver::cones::{CompositeCone, Cone, SupportedCone};

/// Largest leaf (cone rows, expansion coordinates and variables) factored
/// densely.
const LEAF_MAX: usize = 64;
/// Variables shared by more units stay in the border: merging them into one
/// leaf took about four times as many refinement solves (journal 2026-10-07).
const SHARED_UNITS: usize = 4;
/// Binary64 dense flops (class Schur products plus the border factor) per
/// sparse LDL flop admitted. Against QDLDL (journal 2026-10-19), nb (0.99x)
/// ran 59% and nb_L2_bessel (1.34x) 33% faster; sambal (2.3x), nb_L1 (5.4x),
/// axis_many (7.9x) and three LPs (12-19x) ran 1.5-4.5x slower.
const CONE_FLOP_RATIO: f64 = 2.0;

impl<T: FloatT> ArrowLDLSolver<T> {
    pub(crate) fn try_local_cones(
        k: &CscMatrix<T>,
        signs: &[i8],
        a: &CscMatrix<T>,
        cones: &CompositeCone<T>,
        settings: &CoreSettings<T>,
        amd: &std::cell::OnceCell<crate::solver::kkt::ldl::AmdOrdering>,
    ) -> Option<Self> {
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
        // Binary64 adds the class panels' Y and Z copies (its leaves keep no Z).
        let copies = if T::precision_bits() <= 64 { 4 } else { 3 };
        let mut cells = 4 * (t as u128).pow(2) + 4 * k.n as u128;
        let mut schur = 0;
        for (g, link) in groups.iter().zip(&mut links) {
            link.0.sort_unstable();
            link.0.dedup();
            let rows = g.iter().filter(|&&id| id >= n).count();
            let width = if link.2 { g.len() } else { g.len() - rows } as u128;
            let (g, q) = (g.len() as u128, link.0.len() as u128);
            cells += 2 * g * g + (copies * width).saturating_sub(1) * q + 8 * g;
            schur += width * q * q;
        }
        let entries: u128 = links.iter().map(|link| link.1).sum();
        if cells * std::mem::size_of::<T>() as u128
            > ARROW_MAX_BYTES + k.nzval.len() as u128 * std::mem::size_of::<T>() as u128
            || (t as u128).pow(2) > 8 * entries
        {
            return None;
        }
        if T::precision_bits() <= 64
            && !super::local_bounds::dense_border_pays(
                k,
                "LOCAL_CONES",
                schur as f64 + (t as f64).powi(3) / 3.0,
                CONE_FLOP_RATIO,
                amd,
            )
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

/// Binary64 coupling classes as stacked column-major panels: a class's leaf
/// suffixes (rows) over its border columns. The border Schur and both
/// coupling products become one GEMM per class.
pub(super) struct ConePanels<T> {
    panels: Vec<ConePanel<T>>,
    /// Each leaf's panel and first panel row (`usize::MAX`: uncoupled).
    slot: Vec<(usize, usize)>,
}

struct ConePanel<T> {
    rows: usize,
    y: Vec<T>,
    z: Vec<T>,
    // Solve scratch: rows × cols leaf values, q × cols border values.
    v: Vec<T>,
    u: Vec<T>,
}

impl<T: FloatT> ConePanels<T> {
    pub(super) fn new(classes: &[CouplingClass], leaves: &[Leaf<T>]) -> Self {
        let mut slot = vec![(usize::MAX, 0); leaves.len()];
        let panels = classes
            .iter()
            .enumerate()
            .map(|(p, class)| {
                let mut rows = 0;
                for &l in &class.leaves {
                    slot[l] = (p, rows);
                    rows += leaves[l].ids.len() - leaves[l].coupling_start;
                }
                let q = leaves[class.leaves[0]].coupled.len();
                ConePanel {
                    rows,
                    y: vec![T::zero(); rows * q],
                    z: vec![T::zero(); rows * q],
                    v: Vec::new(),
                    u: Vec::new(),
                }
            })
            .collect();
        Self { panels, slot }
    }
}

impl<T: FloatT> ArrowLDLSolver<T> {
    /// `S -= Σ YᵀZ` per class panel, Z = D⁻¹Y rounded entrywise as the leaf
    /// path forms it; only the lower triangle is read, then mirrored.
    pub(super) fn assemble_cone_schur(&mut self) -> bool {
        let Some(cp) = self.cone_panels.as_mut() else {
            return false;
        };
        let t = self.trunk.len();
        let mut g = Vec::new();
        for (panel, class) in cp.panels.iter_mut().zip(&self.classes) {
            let columns = &self.leaves[class.leaves[0]].coupled;
            let (rows, q) = (panel.rows, columns.len());
            let mut at = 0;
            for &l in &class.leaves {
                let leaf = &self.leaves[l];
                let width = leaf.ids.len() - leaf.coupling_start;
                let dinv = &leaf.factor.dinv[leaf.coupling_start..];
                for c in 0..q {
                    let (y, o) = (&leaf.y[c * width..(c + 1) * width], at + c * rows);
                    panel.y[o..o + width].copy_from_slice(y);
                    for ((z, &y), &d) in panel.z[o..o + width].iter_mut().zip(y).zip(dinv) {
                        *z = y * d;
                    }
                }
                at += width;
            }
            g.resize(q * q, T::zero());
            let (q32, rows32) = (q as i32, rows as i32);
            T::xgemm(
                b'T',
                b'N',
                q32,
                q32,
                rows32,
                T::one(),
                &panel.y,
                rows32,
                &panel.z,
                rows32,
                T::zero(),
                &mut g,
                q32,
            );
            for (cj, &j) in columns.iter().enumerate() {
                for (ci, &i) in columns.iter().enumerate().skip(cj) {
                    self.s[i + j * t] -= g[ci + cj * q];
                }
            }
        }
        for j in 0..t {
            for i in j + 1..t {
                self.s[j + i * t] = self.s[i + j * t];
            }
        }
        true
    }

    /// `cols` right-hand sides (column c of `b` and `x` at `c * n`) through
    /// the leaf sweeps, the class panels and the dense border.
    pub(super) fn solve_cones(&mut self, x: &mut [T], b: &[T], cols: usize) -> bool {
        let Some(cp) = self.cone_panels.as_mut() else {
            return false;
        };
        let (n, t) = (self.n, self.trunk.len());
        let leaves = &mut self.leaves;
        let timer = crate::receipt::start();
        let forward = |l: &mut Leaf<T>| l.first_many(b, n, cols, 1, false);
        match &self.pool {
            Some(pool) => pool.install(|| leaves.par_iter_mut().for_each(forward)),
            None => leaves.iter_mut().for_each(forward),
        }
        crate::receipt::finish("cones.forward", timer);
        let timer = crate::receipt::start();
        let tx = &mut self.batch_tx;
        tx.resize(t * cols, T::zero());
        for (j, &id) in self.trunk.iter().enumerate() {
            for c in 0..cols {
                tx[j * cols + c] = b[c * n + id];
            }
        }
        for (panel, class) in cp.panels.iter_mut().zip(&self.classes) {
            let columns = &leaves[class.leaves[0]].coupled;
            let (rows, q) = (panel.rows, columns.len());
            panel.v.resize(rows * cols, T::zero());
            panel.u.resize(q * cols, T::zero());
            let mut at = 0;
            for &l in &class.leaves {
                let leaf = &leaves[l];
                let width = leaf.ids.len() - leaf.coupling_start;
                for r in 0..width {
                    for c in 0..cols {
                        panel.v[at + r + c * rows] = leaf.batch_v[r * cols + c];
                    }
                }
                at += width;
            }
            let (q32, rows32, cols32) = (q as i32, rows as i32, cols as i32);
            T::xgemm(
                b'T',
                b'N',
                q32,
                cols32,
                rows32,
                T::one(),
                &panel.y,
                rows32,
                &panel.v,
                rows32,
                T::zero(),
                &mut panel.u,
                q32,
            );
            for (p, &j) in columns.iter().enumerate() {
                for c in 0..cols {
                    tx[j * cols + c] -= panel.u[p + c * q];
                }
            }
        }
        crate::receipt::finish("cones.couple", timer);
        let timer = crate::receipt::start();
        if cols == 1 {
            self.tf.solve_pooled(tx, self.pool.as_deref());
        } else {
            self.tf.solve_many(tx, cols);
        }
        crate::receipt::finish("cones.trunk", timer);
        let timer = crate::receipt::start();
        for (panel, class) in cp.panels.iter_mut().zip(&self.classes) {
            let columns = &leaves[class.leaves[0]].coupled;
            let (rows, q) = (panel.rows, columns.len());
            for (p, &j) in columns.iter().enumerate() {
                for c in 0..cols {
                    panel.u[p + c * q] = tx[j * cols + c];
                }
            }
            let (q32, rows32, cols32) = (q as i32, rows as i32, cols as i32);
            T::xgemm(
                b'N',
                b'N',
                rows32,
                cols32,
                q32,
                T::one(),
                &panel.y,
                rows32,
                &panel.u,
                q32,
                T::zero(),
                &mut panel.v,
                rows32,
            );
        }
        let (slot, panels) = (&cp.slot, &cp.panels);
        let backward = |(l, leaf): (usize, &mut Leaf<T>)| {
            let (start, (p, at)) = (leaf.coupling_start, slot[l]);
            for i in 0..leaf.ids.len() {
                for c in 0..cols {
                    let s = if i < start || p == usize::MAX {
                        T::zero()
                    } else {
                        panels[p].v[at + i - start + c * panels[p].rows]
                    };
                    let w = &mut leaf.batch_w[i * cols + c];
                    *w = (*w - s) * leaf.factor.dinv[i];
                }
            }
            leaf.factor.backward_many(&mut leaf.batch_w, cols);
        };
        match &self.pool {
            Some(pool) => pool.install(|| leaves.par_iter_mut().enumerate().for_each(backward)),
            None => leaves.iter_mut().enumerate().for_each(backward),
        }
        for leaf in leaves.iter() {
            for (i, &id) in leaf.ids.iter().enumerate() {
                for c in 0..cols {
                    x[c * n + id] = leaf.batch_w[i * cols + c];
                }
            }
        }
        for (j, &id) in self.trunk.iter().enumerate() {
            for c in 0..cols {
                x[c * n + id] = tx[j * cols + c];
            }
        }
        crate::receipt::finish("cones.backward", timer);
        true
    }
}
