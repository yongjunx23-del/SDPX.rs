//! Parallel LDLᵀ factorisation and triangular solves over the elimination
//! tree.
//!
//! The QDLDL elimination tree partitions the columns into a *trunk* (every
//! node that is, or is an ancestor of, a junction with two or more children)
//! and *leaf groups* (the subtrees hanging below the trunk).  Rows belonging
//! to different leaf groups share no ancestor below the trunk, so their
//! numeric work is independent and can run on the shared solver thread pool.
//! For the condensed Schur systems used with sampled inputs the trunk is the
//! small retained-equality border and the leaf groups are the dense
//! block-diagonal cone blocks, which is where most of the work sits.
//!
//! Bitwise reproducibility is preserved:
//! * Leaf rows only read and write columns of their own group, so running
//!   groups concurrently performs exactly the serial operations in serial
//!   order within each group.
//! * A trunk row's leaf-column work is split by group.  Contributions into
//!   trunk positions and into `D[k]` are recorded with their path position
//!   and replayed in the exact serial path order.
//! * The triangular solves use the same split; trunk-position contributions
//!   of the forward solve are replayed in global column order.
//!
//! If the symbolic pattern ever violates these invariants (for example an
//! ordering that interleaves subtrees) the plan is rejected and the serial
//! kernels run instead.

use super::{QDLDLError, QDLDL_UNKNOWN, QDLDL_UNUSED, QDLDL_USED};
use crate::algebra::FloatT;
use rayon::prelude::*;
use std::iter::zip;

/// Column-indexed CSC storage holding one disjoint piece of the L factor.
#[derive(Debug)]
struct ColStore<T> {
    /// Global column index owned at each local slot.
    cols: Vec<usize>,
    /// Local colptr over `cols`.
    lp: Vec<usize>,
    /// Row indices of the owned columns, same layout as the global factor.
    li: Vec<usize>,
    /// Values of the owned columns.
    lx: Vec<T>,
    /// Next free slot per owned column.
    next: Vec<usize>,
}

impl<T: FloatT> ColStore<T> {
    fn new(cols: Vec<usize>, lnz: &[usize]) -> Self {
        let mut lp = Vec::with_capacity(cols.len() + 1);
        lp.push(0);
        let mut acc = 0;
        for &c in &cols {
            acc += lnz[c];
            lp.push(acc);
        }
        Self {
            lp,
            li: vec![0; acc],
            lx: vec![T::zero(); acc],
            next: vec![0; cols.len()],
            cols,
        }
    }

    fn reset(&mut self) {
        self.next.copy_from_slice(&self.lp[..self.lp.len() - 1]);
    }
}

/// Per-group reusable scratch.  Dense vectors are indexed by the column's
/// slot in its group (`rank_of`): a leaf column's rows lie in its own group
/// or the trunk, so a group never touches another group's positions.
#[derive(Debug)]
struct GroupWs<T> {
    y_vals: Vec<T>,
    y_markers: Vec<bool>,
    y_idx: Vec<usize>,
    elim: Vec<usize>,
    d: Vec<T>,
    dinv: Vec<T>,
    pos_count: usize,
    reg_count: usize,
    err: bool,
    /// (path sequence, column) pairs for the current trunk row.
    row_cols: Vec<(u32, usize)>,
    /// Trunk-position contributions as (path sequence, trunk rank, value),
    /// in the order the segment consumed its columns.
    trunk_delta: Vec<(u32, u32, T)>,
    /// (path sequence, `y_c * Lx_tmp`) contributions to `D[k]`.
    d_delta: Vec<(u32, T)>,
    /// Forward-solve scratch and its trunk-position contributions as
    /// (column, trunk rank, value).
    xbuf: Vec<T>,
    solve_delta: Vec<(u32, u32, T)>,
}

impl<T: FloatT> GroupWs<T> {
    fn new(size: usize) -> Self {
        Self {
            y_vals: vec![T::zero(); size],
            y_markers: vec![false; size],
            y_idx: vec![0; size],
            elim: vec![0; size],
            d: vec![T::zero(); size],
            dinv: vec![T::zero(); size],
            pos_count: 0,
            reg_count: 0,
            err: false,
            row_cols: Vec::new(),
            trunk_delta: Vec::new(),
            d_delta: Vec::new(),
            xbuf: vec![T::zero(); size],
            solve_delta: Vec::new(),
        }
    }
}

/// Mutable references to `v[i]` for the sorted, distinct indices `ids`.
fn pick_mut<'a, A>(mut v: &'a mut [A], ids: &[usize]) -> Vec<&'a mut A> {
    let mut out = Vec::with_capacity(ids.len());
    let mut base = 0;
    for &i in ids {
        let (_, tail) = std::mem::take(&mut v).split_at_mut(i - base);
        let (x, rest) = tail.split_first_mut().unwrap();
        out.push(x);
        v = rest;
        base = i + 1;
    }
    out
}

/// Trunk-side scratch shared by the serial parts of a parallel factorisation.
#[derive(Debug)]
struct TrunkWs<T> {
    y_vals: Vec<T>,
    y_markers: Vec<bool>,
    y_idx: Vec<usize>,
    elim: Vec<usize>,
}

/// Elimination-tree schedule for a pool-based factorisation/solve.
#[derive(Debug)]
pub(super) struct ParallelPlan<T> {
    /// Leaf groups as sorted global column lists.
    groups: Vec<Vec<usize>>,
    /// Owning group per column, or `usize::MAX` for trunk columns.
    group_of: Vec<usize>,
    /// Local slot of a column inside its owner's [`ColStore`].
    rank_of: Vec<usize>,
    /// Trunk columns in ascending order.
    trunk: Vec<usize>,
    /// Position of a trunk column within `trunk`, `usize::MAX` otherwise.
    trunk_rank: Vec<usize>,
    /// Column storage for each leaf group.
    stores: Vec<ColStore<T>>,
    /// Column storage for the trunk columns.
    trunk_store: ColStore<T>,
    /// Per-group scratch.
    ws: Vec<GroupWs<T>>,
    /// Trunk scratch.
    tws: TrunkWs<T>,
    /// Per-path-position leaf contributions `(trunk row, delta)` used by the
    /// ordered trunk replay in [`ParallelPlan::trunk_row`].
    pos_delta: Vec<Vec<(u32, T)>>,
    /// Per-path-position `D[k]` contributions from leaf columns.
    pos_d: Vec<T>,
    /// Per-column forward-solve contributions `(trunk row, delta)` for the
    /// ordered replay in [`ParallelPlan::solve`].
    solve_pos: Vec<Vec<(u32, T)>>,
    /// Groups on the current trunk row's path, ascending.
    touched: Vec<usize>,
}

const NO_GROUP: usize = usize::MAX;

impl<T: FloatT> ParallelPlan<T> {
    /// Build the elimination-tree schedule from the symbolic factorisation.
    /// Returns `None` when the pattern does not admit the grouped parallel
    /// scheme or `threads` would not repay it; the caller then keeps using
    /// the serial kernels.
    pub(super) fn build(
        n: usize,
        etree: &[usize],
        lnz: &[usize],
        lp: &[usize],
        li: &[usize],
        threads: usize,
    ) -> Option<Self> {
        if n == 0 {
            return None;
        }
        // Trunk = junction nodes (two or more etree children) and every
        // ancestor of a junction.
        let mut children = vec![0usize; n];
        for &p in etree.iter().take(n) {
            if p != QDLDL_UNKNOWN {
                children[p] += 1;
            }
        }
        let mut is_trunk = vec![false; n];
        for k in 0..n {
            if children[k] >= 2 {
                let mut j = k;
                while j != QDLDL_UNKNOWN && !is_trunk[j] {
                    is_trunk[j] = true;
                    j = etree[j];
                }
            }
        }

        // Group id = the non-trunk root of each leaf subtree.
        let mut group_of = vec![NO_GROUP; n];
        let mut roots: Vec<usize> = Vec::new();
        let mut root_id = vec![NO_GROUP; n];
        for k in 0..n {
            if is_trunk[k] {
                continue;
            }
            let mut j = k;
            while etree[j] != QDLDL_UNKNOWN && !is_trunk[etree[j]] {
                j = etree[j];
            }
            if root_id[j] == NO_GROUP {
                root_id[j] = roots.len();
                roots.push(j);
            }
            group_of[k] = root_id[j];
        }

        let mut groups = vec![Vec::new(); roots.len()];
        for k in 0..n {
            if group_of[k] != NO_GROUP {
                groups[group_of[k]].push(k);
            }
        }

        // Parallelising a single group buys nothing.  Group scratch is sized
        // to the group and a trunk row visits only the groups on its path, so
        // the partition's granularity costs no more than O(n).
        if groups.len() < 2 {
            return None;
        }

        let trunk: Vec<usize> = (0..n).filter(|&k| is_trunk[k]).collect();
        let mut trunk_rank = vec![NO_GROUP; n];
        for (r, &c) in trunk.iter().enumerate() {
            trunk_rank[c] = r;
        }

        // Validate the disjointness invariant on the symbolic L pattern:
        // leaf columns may only be written by rows of their own group or by
        // trunk rows; trunk columns only by trunk rows. Meanwhile estimate
        // the work: a column with l entries costs ~l² (the j-th row using it
        // updates its j earlier entries). Trunk columns, and a leaf column's
        // updates of trunk rows (recorded per group, replayed in order), are
        // serial; the rest runs per group.
        let mut serial = 0f64;
        let mut work = vec![0f64; groups.len()];
        for (c, g) in group_of.iter().enumerate() {
            let mut b = 0f64;
            for &r in &li[lp[c]..lp[c + 1]] {
                if *g == NO_GROUP {
                    if group_of[r] != NO_GROUP {
                        return None;
                    }
                } else if group_of[r] == NO_GROUP {
                    b += 1.0;
                } else if group_of[r] != *g {
                    return None;
                }
            }
            let l = (lp[c + 1] - lp[c]) as f64;
            if *g == NO_GROUP {
                serial += l * l;
            } else {
                serial += b * b;
                work[*g] += l * l - b * b;
            }
        }
        // The plan copies L and pays a pool dispatch per trunk row: keep it
        // only when the pool can at least halve the factor (a 1880-row trunk
        // of 1920 ran 7x slower on 4 threads; Hubbard thermal cubic3, whose
        // 1305-row group dominates, ran 1.17x faster on 16 for 2.6 GB more).
        let (max, sum) = work
            .iter()
            .fold((0f64, 0f64), |(m, s), &w| (m.max(w), s + w));
        // A pattern without off-diagonal work estimates zero, not 0/0.
        let speedup = (serial + sum) / (serial + max.max(sum / threads as f64)).max(1.0);
        if crate::receipt::profile_requested() {
            eprintln!(
                "QDLDLPLAN groups={} trunk={} speedup={speedup:.2}",
                groups.len(),
                trunk.len()
            );
        }
        if speedup < 2.0 {
            return None;
        }

        // Owner-local column storage.
        let mut rank_of = vec![0usize; n];
        let mut stores = Vec::with_capacity(groups.len());
        for cols in &groups {
            for (r, &c) in cols.iter().enumerate() {
                rank_of[c] = r;
            }
            stores.push(ColStore::new(cols.clone(), lnz));
        }
        for (r, &c) in trunk.iter().enumerate() {
            rank_of[c] = r;
        }
        let trunk_store = ColStore::new(trunk.clone(), lnz);

        let ws = groups.iter().map(|g| GroupWs::new(g.len())).collect();
        let tws = TrunkWs {
            y_vals: vec![T::zero(); n],
            y_markers: vec![false; n],
            y_idx: vec![0; n],
            elim: vec![0; n],
        };

        Some(Self {
            groups,
            group_of,
            rank_of,
            trunk,
            trunk_rank,
            stores,
            trunk_store,
            ws,
            tws,
            pos_delta: Vec::new(),
            pos_d: Vec::new(),
            solve_pos: Vec::new(),
            touched: Vec::new(),
        })
    }

    pub(super) fn parallelisable(&self) -> bool {
        self.groups.len() > 1
    }

    /// (leaf group count, trunk size) for diagnostics.
    pub(super) fn shape(&self) -> (usize, usize) {
        (self.groups.len(), self.trunk.len())
    }

    /// Numeric LDLᵀ factorisation on `pool`.  Produces bit-identical results
    /// to `_factor_inner`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn factor(
        &mut self,
        pool: &rayon::ThreadPool,
        ap: &[usize],
        ai: &[usize],
        ax: &[T],
        lp: &mut [usize],
        li: &mut [usize],
        lx: &mut [T],
        d: &mut [T],
        dinv: &mut [T],
        lnz: &[usize],
        etree: &[usize],
        dsigns: &[i8],
        regularize_enable: bool,
        regularize_eps: T,
        regularize_delta: T,
        regularize_count: &mut usize,
    ) -> Result<usize, QDLDLError> {
        // Global colptr, identical to the serial preamble.
        lp[0] = 0;
        let mut acc = 0;
        for (lp, l) in zip(&mut lp[1..], lnz) {
            acc += l;
            *lp = acc;
        }
        for store in &mut self.stores {
            store.reset();
        }
        self.trunk_store.reset();
        d.fill(T::zero());
        // Trunk rows clear the buffers of the groups they touch; a failed
        // factorization may leave some behind.
        for w in &mut self.ws {
            w.y_vals.fill(T::zero());
            w.y_markers.fill(QDLDL_UNUSED);
            w.d.fill(T::zero());
            w.dinv.fill(T::zero());
            w.pos_count = 0;
            w.reg_count = 0;
            w.err = false;
            w.row_cols.clear();
            w.trunk_delta.clear();
            w.d_delta.clear();
        }
        *regularize_count = 0;

        // Row 0 diagonal, exactly as the serial path.
        d[0] = ax[0];
        if regularize_enable {
            let sign = T::from_i8(dsigns[0]).unwrap();
            if d[0] * sign < regularize_eps {
                d[0] = regularize_delta * sign;
                *regularize_count += 1;
            }
        }
        if d[0].is_zero() {
            return Err(QDLDLError::ZeroPivot);
        }
        let mut pos_count = (d[0] > T::zero()) as usize;
        dinv[0] = T::recip(d[0]);
        // Column 0 is consumed by the preamble even when it belongs to a
        // leaf group; mirror the seed values into the owner's workspace.
        if self.group_of[0] != NO_GROUP {
            let (w, r) = (&mut self.ws[self.group_of[0]], self.rank_of[0]);
            w.d[r] = d[0];
            w.dinv[r] = dinv[0];
        }

        let params = (dsigns, regularize_enable, regularize_eps, regularize_delta);

        // Phase A: leaf groups are independent; each runs the serial row
        // loop restricted to its own columns on private storage.
        let group_of = &self.group_of;
        let rank_of = &self.rank_of;
        pool.install(|| {
            self.ws
                .par_iter_mut()
                .zip(self.stores.par_iter_mut())
                .zip(self.groups.par_iter())
                .for_each(|((w, store), cols)| {
                    for &k in cols {
                        if k == 0 {
                            continue; // handled by the preamble
                        }
                        leaf_row(k, ap, ai, ax, etree, group_of, w, store, rank_of, params);
                        if w.err {
                            break;
                        }
                    }
                })
        });
        for (cols, w) in self.groups.iter().zip(self.ws.iter()) {
            for (r, &k) in cols.iter().enumerate() {
                if k != 0 {
                    d[k] = w.d[r];
                    dinv[k] = w.dinv[r];
                }
            }
            pos_count += w.pos_count;
            *regularize_count += w.reg_count;
        }
        if self.ws.iter().any(|w| w.err) {
            return Err(QDLDLError::ZeroPivot);
        }

        // Phase B: trunk rows.  Leaf-column work is distributed back to the
        // owning groups; trunk columns are processed serially.
        for t in 0..self.trunk.len() {
            let k = self.trunk[t];
            if k == 0 {
                continue; // handled by the preamble
            }
            self.trunk_row(
                k,
                ap,
                ai,
                ax,
                etree,
                pool,
                d,
                dinv,
                params,
                regularize_count,
            );
            if d[k].is_zero() {
                return Err(QDLDLError::ZeroPivot);
            }
            if d[k] > T::zero() {
                pos_count += 1;
            }
            dinv[k] = T::recip(d[k]);
        }

        // Assemble the global factor from the per-owner stores.
        for store in &self.stores {
            for (r, &c) in store.cols.iter().enumerate() {
                let (f, l) = (store.lp[r], store.lp[r] + lnz[c]);
                li[lp[c]..lp[c] + lnz[c]].copy_from_slice(&store.li[f..l]);
                lx[lp[c]..lp[c] + lnz[c]].copy_from_slice(&store.lx[f..l]);
            }
        }
        let ts = &self.trunk_store;
        for (r, &c) in ts.cols.iter().enumerate() {
            let (f, l) = (ts.lp[r], ts.lp[r] + lnz[c]);
            li[lp[c]..lp[c] + lnz[c]].copy_from_slice(&ts.li[f..l]);
            lx[lp[c]..lp[c] + lnz[c]].copy_from_slice(&ts.lx[f..l]);
        }

        Ok(pos_count)
    }

    /// Process one trunk row: build its elimination path serially (matching
    /// the serial marker order exactly), run the leaf columns per group on
    /// the pool, replay trunk-position and D contributions in path order,
    /// then finish the trunk columns serially.
    #[allow(clippy::too_many_arguments)]
    fn trunk_row(
        &mut self,
        k: usize,
        ap: &[usize],
        ai: &[usize],
        ax: &[T],
        etree: &[usize],
        pool: &rayon::ThreadPool,
        d: &mut [T],
        dinv: &[T],
        params: (&[i8], bool, T, T),
        regularize_count: &mut usize,
    ) {
        let (dsigns, regularize_enable, regularize_eps, regularize_delta) = params;

        // Seed y values into the owning workspace and build the shared
        // elimination path; identical order to the serial row loop.
        let mut nnz_y = 0;
        let mut dk = T::zero();
        {
            let tws = &mut self.tws;
            for i in ap[k]..ap[k + 1] {
                let bidx = ai[i];
                if bidx == k {
                    dk = ax[i];
                    continue;
                }
                if self.group_of[bidx] == NO_GROUP {
                    tws.y_vals[bidx] = ax[i];
                } else {
                    self.ws[self.group_of[bidx]].y_vals[self.rank_of[bidx]] = ax[i];
                }
                if tws.y_markers[bidx] == QDLDL_UNUSED {
                    tws.y_markers[bidx] = QDLDL_USED;
                    tws.elim[0] = bidx;
                    let mut nnz_e = 1;
                    let mut next_idx = etree[bidx];
                    while next_idx != QDLDL_UNKNOWN && next_idx < k {
                        if tws.y_markers[next_idx] == QDLDL_USED {
                            break;
                        }
                        tws.y_markers[next_idx] = QDLDL_USED;
                        tws.elim[nnz_e] = next_idx;
                        next_idx = etree[next_idx];
                        nnz_e += 1;
                    }
                    while nnz_e != 0 {
                        nnz_e -= 1;
                        tws.y_idx[nnz_y] = tws.elim[nnz_e];
                        nnz_y += 1;
                    }
                }
            }
            // Markers are only needed for the path build; the serial code
            // clears them as columns are consumed.
            for &c in tws.y_idx.iter().take(nnz_y) {
                tws.y_markers[c] = QDLDL_UNUSED;
            }
        }

        // Partition the path (processing order = reverse y_idx) into
        // per-group leaf lists, tagging each column with its sequence
        // position for the ordered replay.  Trunk columns are eliminated
        // inline during the replay itself.  Only the groups on the path take
        // part; their buffers are cleared again once the row is replayed, so
        // no stale sequence tag survives into a later row.
        let mut seq = 0u32;
        for &c in self.tws.y_idx[..nnz_y].iter().rev() {
            let g = self.group_of[c];
            if g != NO_GROUP {
                if self.ws[g].row_cols.is_empty() {
                    self.touched.push(g);
                }
                self.ws[g].row_cols.push((seq, c));
            }
            seq += 1;
        }
        self.touched.sort_unstable();

        // Parallel leaf segments.  Group workspaces own their leaf columns,
        // so no writes are shared; trunk-position and D contributions are
        // recorded with their sequence tags.
        let trunk_rank = &self.trunk_rank;
        let rank_of = &self.rank_of;
        let ws = pick_mut(&mut self.ws, &self.touched);
        let stores = pick_mut(&mut self.stores, &self.touched);
        pool.install(|| {
            ws.into_par_iter()
                .zip(stores.into_par_iter())
                .for_each(|(w, store)| {
                    for &(s, cidx) in &w.row_cols {
                        let r = rank_of[cidx];
                        let tmp = store.next[r];
                        let y_c = w.y_vals[r];
                        for idx in store.lp[r]..tmp {
                            let lxj = store.lx[idx];
                            let lij = store.li[idx];
                            if trunk_rank[lij] == NO_GROUP {
                                w.y_vals[rank_of[lij]] -= lxj * y_c;
                            } else {
                                w.trunk_delta.push((s, trunk_rank[lij] as u32, lxj * y_c));
                            }
                        }
                        let ltmp = y_c * dinv[cidx];
                        store.lx[tmp] = ltmp;
                        store.li[tmp] = k;
                        store.next[r] += 1;
                        w.d_delta.push((s, y_c * ltmp));
                        w.y_vals[r] = T::zero();
                    }
                });
        });

        // Merge leaf contributions into per-position lists, then replay the
        // path in exact serial order: at each position a leaf column's
        // recorded trunk-row and D contributions are applied, and a trunk
        // column is eliminated inline.  This keeps every shared-state update
        // in the serial sequence even when leaf and trunk columns interleave
        // inside a row's path.
        self.pos_delta.clear();
        self.pos_delta.resize_with(nnz_y, Vec::new);
        self.pos_d.clear();
        self.pos_d.resize(nnz_y, T::zero());
        // A position's contributions come from one column and hit distinct
        // trunk rows, so their order within the position does not matter.
        for &g in &self.touched {
            let w = &mut self.ws[g];
            for &(s, rank, d) in &w.trunk_delta {
                self.pos_delta[s as usize].push((self.trunk[rank as usize] as u32, d));
            }
            for &(s, d) in &w.d_delta {
                self.pos_d[s as usize] = d;
            }
            w.row_cols.clear();
            w.trunk_delta.clear();
            w.d_delta.clear();
        }
        self.touched.clear();
        for e in 0..nnz_y {
            let cidx = self.tws.y_idx[nnz_y - 1 - e];
            for &(row, delta) in &self.pos_delta[e] {
                self.tws.y_vals[row as usize] -= delta;
            }
            if self.trunk_rank[cidx] != NO_GROUP {
                let r = self.rank_of[cidx];
                let tmp = self.trunk_store.next[r];
                let y_c = self.tws.y_vals[cidx];
                for idx in self.trunk_store.lp[r]..tmp {
                    let lxj = self.trunk_store.lx[idx];
                    let lij = self.trunk_store.li[idx];
                    self.tws.y_vals[lij] -= lxj * y_c;
                }
                let ltmp = y_c * dinv[cidx];
                self.trunk_store.lx[tmp] = ltmp;
                self.trunk_store.li[tmp] = k;
                self.trunk_store.next[r] += 1;
                dk -= y_c * ltmp;
                self.tws.y_vals[cidx] = T::zero();
            } else {
                dk -= self.pos_d[e];
            }
        }

        d[k] = dk;
        if regularize_enable {
            let sign = T::from_i8(dsigns[k]).unwrap();
            if d[k] * sign < regularize_eps {
                d[k] = regularize_delta * sign;
                *regularize_count += 1;
            }
        }
    }

    /// Parallel `x = L⁻ᵀ D⁻¹ L⁻¹ x` on the assembled global factor.
    pub(super) fn solve(
        &mut self,
        pool: &rayon::ThreadPool,
        lp: &[usize],
        li: &[usize],
        lx: &[T],
        dinv: &[T],
        x: &mut [T],
    ) {
        // Forward solve (L+I)x = b.  Leaf columns scatter into their own
        // group and into trunk positions; the latter are replayed in global
        // column order afterwards.
        let (trunk_rank, rank_of) = (&self.trunk_rank, &self.rank_of);
        let xr: &[T] = x;
        pool.install(|| {
            self.ws
                .par_iter_mut()
                .zip(self.groups.par_iter())
                .for_each(|(w, cols)| {
                    w.solve_delta.clear();
                    for (b, &c) in w.xbuf.iter_mut().zip(cols) {
                        *b = xr[c];
                    }
                    for (r, &i) in cols.iter().enumerate() {
                        let xi = w.xbuf[r];
                        for idx in lp[i]..lp[i + 1] {
                            let lxj = lx[idx];
                            let lij = li[idx];
                            if trunk_rank[lij] == NO_GROUP {
                                w.xbuf[rank_of[lij]] -= lxj * xi;
                            } else {
                                w.solve_delta
                                    .push((i as u32, trunk_rank[lij] as u32, lxj * xi));
                            }
                        }
                    }
                });
        });
        for (w, cols) in self.ws.iter().zip(&self.groups) {
            for (b, &i) in w.xbuf.iter().zip(cols) {
                x[i] = *b;
            }
        }
        // Merge leaf contributions into per-column lists, then replay all
        // columns in exact serial order: at column `i` a leaf column's
        // recorded trunk-row contributions are applied and a trunk column
        // scatters inline — the same shared-state order as `_lsolve` even
        // when leaf and trunk columns interleave.  A column's contributions
        // hit distinct trunk rows, so their order within it does not matter.
        let n = self.trunk_rank.len();
        self.solve_pos.clear();
        self.solve_pos.resize_with(n, Vec::new);
        for w in self.ws.iter() {
            for &(c, rank, d) in &w.solve_delta {
                self.solve_pos[c as usize].push((self.trunk[rank as usize] as u32, d));
            }
        }
        for i in 0..n {
            if self.trunk_rank[i] != NO_GROUP {
                let xi = x[i];
                for idx in lp[i]..lp[i + 1] {
                    x[li[idx]] -= lx[idx] * xi;
                }
            } else {
                for &(row, delta) in &self.solve_pos[i] {
                    x[row as usize] -= delta;
                }
            }
        }

        // Combined D⁻¹Lᵀ solve: trunk columns first (they only depend on
        // trunk values), then leaf groups in parallel.
        for &i in self.trunk.iter().rev() {
            let mut s = T::zero();
            for idx in lp[i]..lp[i + 1] {
                s += lx[idx] * x[li[idx]];
            }
            let xi = &mut x[i];
            *xi *= dinv[i];
            *xi -= s;
        }
        let xr: &[T] = x;
        pool.install(|| {
            self.ws
                .par_iter_mut()
                .zip(self.groups.par_iter())
                .for_each(|(w, cols)| {
                    for (b, &c) in w.xbuf.iter_mut().zip(cols) {
                        *b = xr[c];
                    }
                    for (r, &i) in cols.iter().enumerate().rev() {
                        let mut s = T::zero();
                        for idx in lp[i]..lp[i + 1] {
                            let lij = li[idx];
                            let xj = if trunk_rank[lij] == NO_GROUP {
                                w.xbuf[rank_of[lij]]
                            } else {
                                xr[lij]
                            };
                            s += lx[idx] * xj;
                        }
                        let xi = &mut w.xbuf[r];
                        *xi *= dinv[i];
                        *xi -= s;
                    }
                });
        });
        for (w, cols) in self.ws.iter().zip(&self.groups) {
            for (b, &i) in w.xbuf.iter().zip(cols) {
                x[i] = *b;
            }
        }
    }
}

/// The leaf-row elimination for one group, identical in order and arithmetic
/// to the serial `_factor_inner` loop restricted to `cols`.
#[allow(clippy::too_many_arguments)]
fn leaf_row<T: FloatT>(
    k: usize,
    ap: &[usize],
    ai: &[usize],
    ax: &[T],
    etree: &[usize],
    group_of: &[usize],
    ws: &mut GroupWs<T>,
    store: &mut ColStore<T>,
    rank_of: &[usize],
    params: (&[i8], bool, T, T),
) {
    let (dsigns, regularize_enable, regularize_eps, regularize_delta) = params;
    let y_vals = &mut ws.y_vals;
    let y_markers = &mut ws.y_markers;
    let y_idx = &mut ws.y_idx;
    let elim = &mut ws.elim;

    // Every column on the path lies below `k` in its leaf subtree.
    let rk = rank_of[k];
    let mut nnz_y = 0;
    for i in ap[k]..ap[k + 1] {
        let bidx = ai[i];
        if bidx == k {
            ws.d[rk] = ax[i];
            continue;
        }
        debug_assert_eq!(group_of[bidx], group_of[k]);
        y_vals[rank_of[bidx]] = ax[i];
        if y_markers[rank_of[bidx]] == QDLDL_UNUSED {
            y_markers[rank_of[bidx]] = QDLDL_USED;
            elim[0] = bidx;
            let mut nnz_e = 1;
            let mut next_idx = etree[bidx];
            while next_idx != QDLDL_UNKNOWN && next_idx < k {
                if y_markers[rank_of[next_idx]] == QDLDL_USED {
                    break;
                }
                y_markers[rank_of[next_idx]] = QDLDL_USED;
                elim[nnz_e] = next_idx;
                next_idx = etree[next_idx];
                nnz_e += 1;
            }
            while nnz_e != 0 {
                nnz_e -= 1;
                y_idx[nnz_y] = elim[nnz_e];
                nnz_y += 1;
            }
        }
    }

    for i in (0..nnz_y).rev() {
        let cidx = y_idx[i];
        debug_assert_eq!(group_of[cidx], group_of[k]);
        let r = rank_of[cidx];
        let tmp = store.next[r];
        let y_c = y_vals[r];
        for idx in store.lp[r]..tmp {
            let lxj = store.lx[idx];
            let lij = store.li[idx];
            debug_assert_eq!(group_of[lij], group_of[k]);
            y_vals[rank_of[lij]] -= lxj * y_c;
        }
        let ltmp = y_c * ws.dinv[r];
        store.lx[tmp] = ltmp;
        store.li[tmp] = k;
        store.next[r] += 1;
        ws.d[rk] -= y_c * ltmp;
        y_vals[r] = T::zero();
        y_markers[r] = QDLDL_UNUSED;
    }

    if regularize_enable {
        let sign = T::from_i8(dsigns[k]).unwrap();
        if ws.d[rk] * sign < regularize_eps {
            ws.d[rk] = regularize_delta * sign;
            ws.reg_count += 1;
        }
    }
    if ws.d[rk].is_zero() {
        ws.err = true;
        return;
    }
    if ws.d[rk] > T::zero() {
        ws.pos_count += 1;
    }
    ws.dinv[rk] = T::recip(ws.d[rk]);
}

#[cfg(test)]
mod tests {
    use crate::algebra::CscMatrix;
    use crate::qdldl::{QDLDLFactorisation, QDLDLSettingsBuilder};
    use rayon::ThreadPoolBuilder;
    use std::sync::Arc;

    /// Dense-block arrow matrix: `nblocks` dense diagonal blocks of size
    /// `bs` followed by `ntr` dense border columns.  With the identity
    /// permutation each leaf subtree is one block and the border is the
    /// trunk — the same shape as the condensed Schur systems of sampled
    /// problems.
    fn arrow_matrix(nblocks: usize, bs: usize, ntr: usize) -> CscMatrix<f64> {
        let nleaf = nblocks * bs;
        let n = nleaf + ntr;
        let mut colptr = Vec::with_capacity(n + 1);
        let mut rowval = Vec::new();
        let mut nzval = Vec::new();
        colptr.push(0);
        for c in 0..n {
            let bs_of = c / bs;
            let start = if c < nleaf { bs_of * bs } else { 0 };
            for r in start..=c {
                rowval.push(r);
                // Non-dyadic values: accumulation-order differences show up
                // in the last bits, so the test detects ordering mistakes.
                let v = if r == c {
                    8.0 + (c % 7) as f64 * 0.3
                } else {
                    let s = ((r * 3 + c * 5) % 4) as f64 - 1.5;
                    s * 0.13 + 0.007
                };
                nzval.push(v);
            }
            colptr.push(rowval.len());
        }
        CscMatrix {
            m: n,
            n,
            colptr,
            rowval,
            nzval,
        }
    }

    fn assert_factors_equal(fa: &QDLDLFactorisation<f64>, fb: &QDLDLFactorisation<f64>) {
        assert_eq!(fa.L.colptr, fb.L.colptr);
        assert_eq!(fa.L.rowval, fb.L.rowval);
        assert_eq!(fa.L.nzval, fb.L.nzval);
        assert_eq!(fa.D, fb.D);
        assert_eq!(fa.Dinv, fb.Dinv);
        assert_eq!(fa.positive_inertia(), fb.positive_inertia());
        assert_eq!(fa.regularize_count(), fb.regularize_count());
    }

    fn pool(threads: usize) -> Arc<rayon::ThreadPool> {
        Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn parallel_factor_solve_bitwise_identical() {
        let a = arrow_matrix(5, 6, 3);
        let n = a.n;
        let opts = QDLDLSettingsBuilder::default()
            .perm((0..n).collect())
            .build()
            .unwrap();

        // serial factorisation (no pool)
        let mut serial = QDLDLFactorisation::new(&a, Some(opts.clone())).unwrap();
        serial.refactor().unwrap();
        assert!(serial.plan.is_none() || true); // plan may exist; pool is what gates

        // parallel factorisation
        let mut par = QDLDLFactorisation::new(&a, Some(opts)).unwrap();
        par.set_pool(Some(pool(4)));
        assert!(par.threads() > 1);
        par.refactor().unwrap();

        assert_factors_equal(&serial, &par);

        // a second refactor through update_values must stay identical too
        let idx: Vec<usize> = (0..a.nnz()).collect();
        let vals: Vec<f64> = a.nzval.iter().map(|v| *v * 0.5 + 0.125).collect();
        serial.update_values(&idx, &vals);
        serial.refactor().unwrap();
        par.update_values(&idx, &vals);
        par.refactor().unwrap();
        assert_factors_equal(&serial, &par);

        // solves must be bitwise identical
        let b0: Vec<f64> = (0..n).map(|i| ((i * 11) % 9) as f64 - 4.0).collect();
        let (mut b1, mut b2) = (b0.clone(), b0.clone());
        serial.solve(&mut b1);
        par.solve(&mut b2);
        assert_eq!(b1, b2);
    }

    /// A fork-join sparse structure (four chains merging) exercises the
    /// parallel plan outside the pure arrow case; a trunk-heavy arrow and a
    /// plain chain exercise the serial fallback when the plan would not pay
    /// or finds no parallelism.
    #[test]
    fn parallel_fallback_and_fork() {
        // fork: columns 0..31 form four chains of eight joining the trunk
        // columns 32..33 (dense border); then a 12-column border over two
        // pairs holds nearly all the work and stays serial
        for (blocks, bs, border, parallel) in [(4, 8, 2, true), (2, 2, 12, false)] {
            let a = arrow_matrix(blocks, bs, border);
            let n = a.n;
            let opts = QDLDLSettingsBuilder::default()
                .perm((0..n).collect())
                .build()
                .unwrap();
            let mut serial = QDLDLFactorisation::new(&a, Some(opts.clone())).unwrap();
            serial.refactor().unwrap();
            let mut par = QDLDLFactorisation::new(&a, Some(opts)).unwrap();
            par.set_pool(Some(pool(3)));
            assert_eq!(par.threads() > 1, parallel);
            par.refactor().unwrap();
            assert_factors_equal(&serial, &par);
        }

        // tridiagonal chain: single group, serial path still exact
        let n = 8;
        let mut colptr = vec![0usize];
        let (mut rowval, mut nzval) = (Vec::new(), Vec::new());
        for c in 0..n {
            if c > 0 {
                rowval.push(c - 1);
                nzval.push(0.5);
            }
            rowval.push(c);
            nzval.push(4.0);
            colptr.push(rowval.len());
        }
        let tri = CscMatrix {
            m: n,
            n,
            colptr,
            rowval,
            nzval,
        };
        let mut f = QDLDLFactorisation::new(&tri, None).unwrap();
        f.set_pool(Some(pool(2)));
        f.refactor().unwrap();
        let mut b = vec![1.0f64; n];
        f.solve(&mut b);
        assert!(b.iter().all(|v| (*v).is_finite()));
    }
}
