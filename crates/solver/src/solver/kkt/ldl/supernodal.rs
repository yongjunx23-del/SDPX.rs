//! Multifrontal supernodal LDLᵀ for wide types.
//!
//! Serial QDLDL pays one rounded scalar multiply-subtract per factor term.
//! Here every supernode's frontal matrix is factored by the arrow solver's
//! dense exact kernels ([`ldl_rec`]: exact dots and residue products), and
//! each front starts once its children are factored. Each kernel is exact or
//! runs in a fixed order, and children are assembled in a fixed order, so the
//! factor does not depend on the thread count or the schedule.
//!
//! The ordering is AMD without dense-row removal ([`ordering`]): QDLDL's
//! threshold (15·√n) treats every row of a dense-ish Schur complement as
//! dense and leaves those rows unordered (dE3's condensed Schur, order 6237:
//! 3.1e10 multiply-subtracts against 2.7e9).

use super::arrow::{exact_product, ldl_rec, solve_unit_lower_t};
use crate::algebra::*;
use crate::solver::core::CoreSettings;
use crate::solver::kkt::direct::DirectLDLSolver;
use crate::solver::kkt::{HasLinearSolverInfo, LinearSolverInfo};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const NONE: usize = usize::MAX;
/// Columns per solve panel, and rows per parallel task of its update.
const SOLVE_PANEL: usize = 64;
const SOLVE_ROWS: usize = 256;

/// AMD ordering that removes only completely dense rows.
pub(crate) fn ordering<T: FloatT>(k: &CscMatrix<T>) -> super::AmdOrdering {
    let mut control = amd::Control::default();
    control.dense = -1.0;
    amd::order(k.n, &k.colptr, &k.rowval, &control).unwrap()
}

/// Supernodal structure of one KKT pattern under a fixed ordering.
struct Symbolic {
    /// `perm[new] = old`.
    perm: Vec<usize>,
    /// Expected pivot signs in the new order.
    signs: Vec<i8>,
    /// Supernode `s` owns columns `start[s]..start[s + 1]` (postorder).
    start: Vec<usize>,
    /// Rows of each front: its own columns, then the rows below them.
    rows: Vec<Vec<usize>>,
    children: Vec<Vec<usize>>,
    parent: Vec<usize>,
    /// Positions in the parent's `rows` of each row below a supernode.
    relmap: Vec<Vec<usize>>,
    /// Per supernode: (KKT value index, offset in the column-major front).
    entries: Vec<Vec<(usize, usize)>>,
}

impl Symbolic {
    fn new<T: FloatT>(k: &CscMatrix<T>, signs: &[i8], perm0: Vec<usize>) -> Self {
        let n = k.n;
        // Elimination tree under `perm0`, then a postorder of it (same fill).
        let parent0 = etree(&upper_pattern(k, &invert(&perm0)), n);
        let perm: Vec<usize> = postorder(&parent0).iter().map(|&j| perm0[j]).collect();
        let iperm = invert(&perm);
        let mut lower: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
        for j in 0..n {
            for q in k.colptr[j]..k.colptr[j + 1] {
                let (a, b) = (iperm[k.rowval[q]], iperm[j]);
                lower[a.min(b)].push((a.max(b), q));
            }
        }
        let parent = etree(&upper_pattern(k, &iperm), n);
        let mut nchild = vec![0usize; n];
        let mut kids: Vec<Vec<usize>> = vec![Vec::new(); n];
        for j in 0..n {
            if parent[j] != NONE {
                nchild[parent[j]] += 1;
                kids[parent[j]].push(j);
            }
        }
        // Column structures (rows below the diagonal), bottom-up. A column
        // keeps its structure only while it may still end a supernode.
        let merges = |j: usize, cc: &[usize]| {
            j > 0 && parent[j - 1] == j && nchild[j] == 1 && cc[j - 1] == cc[j] + 1
        };
        let mut structs: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut cc = vec![0usize; n];
        let mut mark = vec![NONE; n];
        for j in 0..n {
            let mut s = Vec::new();
            for &(r, _) in &lower[j] {
                if r > j && mark[r] != j {
                    mark[r] = j;
                    s.push(r);
                }
            }
            for &c in &kids[j] {
                for &r in &structs[c] {
                    if r > j && mark[r] != j {
                        mark[r] = j;
                        s.push(r);
                    }
                }
            }
            s.sort_unstable();
            cc[j] = s.len();
            structs[j] = s;
            if merges(j, &cc) {
                structs[j - 1] = Vec::new();
            }
        }
        // Fundamental supernodes, then relaxed amalgamation of each last
        // child into its parent (CHOLMOD's default limits).
        struct Node {
            f: usize,
            l: usize,
            below: Vec<usize>,
            zeros: usize,
        }
        let stored = |w: usize, b: usize| w * (w - 1) / 2 + w * b;
        let mut nodes: Vec<Node> = Vec::new();
        let mut j = 0;
        while j < n {
            let f = j;
            j += 1;
            while j < n && merges(j, &cc) {
                j += 1;
            }
            let mut p = Node {
                f,
                l: j,
                below: std::mem::take(&mut structs[j - 1]),
                zeros: 0,
            };
            while let Some(c) = nodes.last() {
                let child = c.l == p.f && c.below.first().is_some_and(|&r| r < p.l);
                if !child {
                    break;
                }
                let (w, b) = (p.l - c.f, p.below.len());
                let added = stored(w, b) - stored(c.l - c.f, c.below.len()) - stored(p.l - p.f, b);
                let zeros = c.zeros + p.zeros + added;
                let frac = zeros as f64 / stored(w, b).max(1) as f64;
                if !(w <= 4 || (w <= 16 && frac < 0.8) || (w <= 48 && frac < 0.1) || frac < 0.05)
                {
                    break;
                }
                let c = nodes.pop().unwrap();
                p.f = c.f;
                p.zeros = zeros;
            }
            nodes.push(p);
        }
        let nsn = nodes.len();
        let mut start = Vec::with_capacity(nsn + 1);
        let mut sn_of = vec![0usize; n];
        for (s, node) in nodes.iter().enumerate() {
            start.push(node.f);
            sn_of[node.f..node.l].fill(s);
        }
        start.push(n);
        let rows: Vec<Vec<usize>> = nodes
            .into_iter()
            .map(|node| (node.f..node.l).chain(node.below).collect())
            .collect();
        let width = |s: usize| start[s + 1] - start[s];
        let mut parent = vec![NONE; nsn];
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); nsn];
        let mut relmap: Vec<Vec<usize>> = vec![Vec::new(); nsn];
        for s in 0..nsn {
            let Some(&r) = rows[s].get(width(s)) else {
                continue;
            };
            let p = sn_of[r];
            parent[s] = p;
            children[p].push(s);
            let (fp, wp) = (start[p], width(p));
            let below_p = &rows[p][wp..];
            relmap[s] = rows[s][width(s)..]
                .iter()
                .map(|&r| {
                    if r < fp + wp {
                        r - fp
                    } else {
                        wp + below_p.binary_search(&r).unwrap()
                    }
                })
                .collect();
        }
        let mut entries: Vec<Vec<(usize, usize)>> = vec![Vec::new(); nsn];
        for s in 0..nsn {
            let (f, w, r) = (start[s], width(s), rows[s].len());
            let below = &rows[s][w..];
            for c in f..f + w {
                for &(row, q) in &lower[c] {
                    let pos = if row < f + w {
                        row - f
                    } else {
                        w + below.binary_search(&row).unwrap()
                    };
                    entries[s].push((q, pos + (c - f) * r));
                }
            }
        }
        let signs = perm.iter().map(|&old| signs[old]).collect();
        Self {
            perm,
            signs,
            start,
            rows,
            children,
            parent,
            relmap,
            entries,
        }
    }

    fn width(&self, s: usize) -> usize {
        self.start[s + 1] - self.start[s]
    }

    /// Assemble and partially factor front `s`: its factor columns (`r × w`,
    /// unit lower on top), their D⁻¹, and the update block for the parent
    /// (`b × b`, lower triangle).
    #[allow(clippy::type_complexity)]
    fn front<T: FloatT>(
        &self,
        s: usize,
        kkt: &CscMatrix<T>,
        updates: &mut dyn FnMut(usize) -> Vec<T>,
        reg: Option<(T, T)>,
        count: &mut usize,
        ways: usize,
    ) -> Result<(Vec<T>, Vec<T>, Vec<T>), &'static str> {
        let (f, w, r) = (self.start[s], self.width(s), self.rows[s].len());
        let b = r - w;
        // The factor columns (r × w) and the parent's update (F22, b × b)
        // are assembled apart; the update block is never part of the front.
        let mut front = zeros::<T>(r * w, ways);
        let mut update = zeros::<T>(b * b, ways);
        for &(q, offset) in &self.entries[s] {
            front[offset] = kkt.nzval[q];
        }
        for &c in &self.children[s] {
            let child = updates(c);
            let map = &self.relmap[c];
            let bc = map.len();
            for jj in 0..bc {
                let (j, values) = (map[jj], &child[jj * bc..(jj + 1) * bc]);
                if j < w {
                    for ii in jj..bc {
                        front[map[ii] + j * r] += values[ii];
                    }
                } else {
                    for ii in jj..bc {
                        update[map[ii] - w + (j - w) * b] += values[ii];
                    }
                }
            }
        }
        let sign = |k: usize| T::from_i8(self.signs[f + k]).unwrap();
        let mut dinv = vec![T::zero(); w];
        ldl_rec(&mut front, r, w, &vec![0; w], &sign, reg, count, ways, &mut dinv)?;
        if b > 0 {
            let mut l11 = vec![T::zero(); w * w];
            for j in 0..w {
                l11[j * w..(j + 1) * w].copy_from_slice(&front[j * r..j * r + w]);
            }
            solve_unit_lower_t(b, w, &l11, w, &mut front[w..], r, &vec![0; b], ways);
            drop(l11);
            // W21 = L21·D1, then L21 in place; the update is F22 − L21·W21ᵀ.
            let mut w21 = vec![T::zero(); b * w];
            for j in 0..w {
                let column = &mut front[w + j * r..r + j * r];
                w21[j * b..(j + 1) * b].copy_from_slice(column);
                column.iter_mut().for_each(|v| *v *= dinv[j]);
            }
            let l21 = &front[w..];
            let mut packed = zeros::<T>(b * (b + 1) / 2, ways);
            let upper = crate::algebra::with_split_hint(ways, || {
                T::xgemm_upper_exact(b'N', b'T', b, b, w, &w21, b, l21, r, &mut packed, None, None)
            });
            if upper {
                for a in 0..b {
                    for c in a..b {
                        update[c + a * b] -= packed[c * (c + 1) / 2 + a];
                    }
                }
            } else {
                drop(packed);
                let mut full = zeros::<T>(b * b, ways);
                exact_product((b'N', b'T'), (b, b, w), (&w21, b), (l21, r), &mut full, ways);
                for a in 0..b {
                    for c in a..b {
                        update[c + a * b] -= full[a + c * b];
                    }
                }
            }
        }
        Ok((front, dinv, update))
    }
}

fn zeros<T: FloatT>(len: usize, ways: usize) -> Vec<T> {
    if ways > 1 {
        rayon::iter::repeat_n(T::zero(), len).collect()
    } else {
        vec![T::zero(); len]
    }
}

fn invert(perm: &[usize]) -> Vec<usize> {
    let mut iperm = vec![0; perm.len()];
    perm.iter().enumerate().for_each(|(i, &p)| iperm[p] = i);
    iperm
}

/// Rows `i < j` of each column `j` of the permuted symmetric pattern.
fn upper_pattern<T: FloatT>(k: &CscMatrix<T>, iperm: &[usize]) -> Vec<Vec<usize>> {
    let mut upper = vec![Vec::new(); k.n];
    for j in 0..k.n {
        for q in k.colptr[j]..k.colptr[j + 1] {
            let (a, b) = (iperm[k.rowval[q]], iperm[j]);
            if a != b {
                upper[a.max(b)].push(a.min(b));
            }
        }
    }
    upper
}

/// Elimination tree (Liu, with path compression).
fn etree(upper: &[Vec<usize>], n: usize) -> Vec<usize> {
    let mut parent = vec![NONE; n];
    let mut ancestor = vec![NONE; n];
    for j in 0..n {
        for &i in &upper[j] {
            let mut k = i;
            while ancestor[k] != NONE && ancestor[k] != j {
                let next = ancestor[k];
                ancestor[k] = j;
                k = next;
            }
            if ancestor[k] == NONE {
                ancestor[k] = j;
                parent[k] = j;
            }
        }
    }
    parent
}

/// Nodes of a forest in postorder (children ascending before parents).
fn postorder(parent: &[usize]) -> Vec<usize> {
    let n = parent.len();
    let mut kids: Vec<Vec<usize>> = vec![Vec::new(); n];
    for j in (0..n).rev() {
        if parent[j] != NONE {
            kids[parent[j]].push(j);
        }
    }
    let mut order = Vec::with_capacity(n);
    let mut stack = Vec::new();
    for root in (0..n).filter(|&j| parent[j] == NONE) {
        stack.push(root);
        while let Some(&j) = stack.last() {
            if let Some(c) = kids[j].pop() {
                stack.push(c);
            } else {
                order.push(j);
                stack.pop();
            }
        }
    }
    order
}

/// One numeric factorization in flight. A front starts when its last child
/// finishes, so small ancestor chains overlap the wide leaves (dE3 at 32
/// threads: fronts of 16-162 columns waited for a 1457-column leaf), and a
/// wide front still splits its own work over the pool.
struct Fronts<'a, T: FloatT> {
    sym: &'a Symbolic,
    kkt: &'a CscMatrix<T>,
    reg: Option<(T, T)>,
    ways: usize,
    /// Children of each front not yet factored.
    pending: Vec<AtomicUsize>,
    updates: Vec<Mutex<Vec<T>>>,
    /// Factor columns and D⁻¹ of each front.
    factors: Vec<Mutex<(Vec<T>, Vec<T>)>>,
    error: Mutex<Option<&'static str>>,
}

impl<'a, T: FloatT> Fronts<'a, T> {
    fn run<'s>(&'s self, scope: &rayon::Scope<'s>, s: usize) {
        let sym = self.sym;
        // Finished fronts free their scratch on the workers' arenas, which
        // keep those pages resident; return them before an assembled front.
        if !sym.children[s].is_empty() {
            crate::receipt::trim_heap(false);
        }
        let mut take = |c: usize| std::mem::take(&mut *self.updates[c].lock().unwrap());
        match sym.front(s, self.kkt, &mut take, self.reg, &mut 0, self.ways) {
            Ok((ls, ds, us)) => {
                *self.factors[s].lock().unwrap() = (ls, ds);
                *self.updates[s].lock().unwrap() = us;
                let p = sym.parent[s];
                if p != NONE && self.pending[p].fetch_sub(1, Ordering::AcqRel) == 1 {
                    scope.spawn(move |scope| self.run(scope, p));
                }
            }
            Err(error) => {
                self.error.lock().unwrap().get_or_insert(error);
            }
        }
    }
}

pub struct SupernodalLDLSolver<T: FloatT> {
    sym: Symbolic,
    /// Factor columns of each supernode (`rows × width`, column-major).
    l: Vec<Vec<T>>,
    dinv: Vec<T>,
    reg: Option<(T, T)>,
    pool: Option<Arc<rayon::ThreadPool>>,
    nnz: usize,
    work: Vec<T>,
}

impl<T: FloatT> SupernodalLDLSolver<T> {
    pub fn new(k: &CscMatrix<T>, signs: &[i8], settings: &CoreSettings<T>, perm: Vec<usize>) -> Self {
        let sym = Symbolic::new(k, signs, perm);
        if crate::receipt::profile_requested() {
            let nsn = sym.start.len() - 1;
            let widest = (0..nsn).map(|s| sym.rows[s].len()).max().unwrap_or(0);
            eprintln!(
                "SUPERNODAL n={} supernodes={nsn} leaves={} front_max={widest} lnz={}",
                k.n,
                sym.children.iter().filter(|c| c.is_empty()).count(),
                (0..nsn).map(|s| sym.width(s) * sym.rows[s].len()).sum::<usize>(),
            );
        }
        let nsn = sym.start.len() - 1;
        Self {
            l: (0..nsn).map(|_| Vec::new()).collect(),
            dinv: vec![T::zero(); k.n],
            reg: settings.dynamic_regularization_enable.then_some((
                settings.dynamic_regularization_eps,
                settings.dynamic_regularization_delta,
            )),
            pool: None,
            nnz: k.nnz(),
            work: vec![T::zero(); k.n],
            sym,
        }
    }

    fn factor(&mut self, kkt: &CscMatrix<T>) -> Result<(), &'static str> {
        let sym = &self.sym;
        let nsn = sym.start.len() - 1;
        let fronts = Fronts {
            sym,
            kkt,
            reg: self.reg,
            ways: rayon::current_num_threads(),
            pending: sym.children.iter().map(|c| AtomicUsize::new(c.len())).collect(),
            updates: (0..nsn).map(|_| Mutex::new(Vec::new())).collect(),
            factors: (0..nsn).map(|_| Mutex::new((Vec::new(), Vec::new()))).collect(),
            error: Mutex::new(None),
        };
        let cx = &fronts;
        rayon::scope(|scope| {
            for s in (0..nsn).filter(|&s| sym.children[s].is_empty()) {
                scope.spawn(move |scope| cx.run(scope, s));
            }
        });
        if let Some(error) = fronts.error.into_inner().unwrap() {
            return Err(error);
        }
        for (s, factor) in fronts.factors.into_iter().enumerate() {
            let (ls, ds) = factor.into_inner().unwrap();
            self.l[s] = ls;
            self.dinv[sym.start[s]..sym.start[s + 1]].copy_from_slice(&ds);
        }
        crate::receipt::trim_heap(false);
        Ok(())
    }

    /// `y = L⁻ᵀ D⁻¹ L⁻¹ y` in the factor's order, by panels of each front.
    /// Forward: each row subtracts one exact dot of its panel columns, rows
    /// split over the pool. Backward: each panel column takes the rows after
    /// its panel as one exact dot, then the panel's own solved rows as
    /// another. Panels are fixed, so the result is thread-invariant.
    fn solve_permuted(&self, y: &mut [T]) {
        let sym = &self.sym;
        let nsn = sym.start.len() - 1;
        let par = rayon::current_num_threads() > 1;
        for s in 0..nsn {
            let (w, rows) = (sym.width(s), &sym.rows[s]);
            let (l, r) = (&self.l[s], rows.len());
            let mut g: Vec<T> = rows.iter().map(|&row| y[row]).collect();
            for k0 in (0..w).step_by(SOLVE_PANEL) {
                let k1 = (k0 + SOLVE_PANEL).min(w);
                for i in k0 + 1..k1 {
                    let solved = T::dot_fma((k0..i).map(|k| (&l[i + k * r], &g[k])));
                    g[i] -= solved;
                }
                let (head, tail) = g.split_at_mut(k1);
                let x = &head[k0..k1];
                let update = |(c, chunk): (usize, &mut [T])| {
                    let i0 = k1 + c * SOLVE_ROWS;
                    for (t, v) in chunk.iter_mut().enumerate() {
                        let i = i0 + t;
                        *v -= T::dot_fma(x.iter().enumerate().map(|(kk, xk)| (&l[i + (k0 + kk) * r], xk)));
                    }
                };
                if par && tail.len() > SOLVE_ROWS {
                    tail.par_chunks_mut(SOLVE_ROWS).enumerate().for_each(update);
                } else {
                    tail.chunks_mut(SOLVE_ROWS).enumerate().for_each(update);
                }
            }
            for (&row, v) in rows.iter().zip(g) {
                y[row] = v;
            }
        }
        y.iter_mut().zip(&self.dinv).for_each(|(v, &d)| *v *= d);
        for s in (0..nsn).rev() {
            let (w, rows) = (sym.width(s), &sym.rows[s]);
            let (l, r) = (&self.l[s], rows.len());
            let mut g: Vec<T> = rows.iter().map(|&row| y[row]).collect();
            for k0 in (0..w).step_by(SOLVE_PANEL).rev() {
                let k1 = (k0 + SOLVE_PANEL).min(w);
                let (head, tail) = g.split_at_mut(k1);
                let dot = |k: usize| T::dot_slices(&l[k1 + k * r..r + k * r], tail);
                let t: Vec<T> = if par && tail.len() * (k1 - k0) > SOLVE_ROWS * SOLVE_PANEL {
                    (k0..k1).into_par_iter().map(dot).collect()
                } else {
                    (k0..k1).map(dot).collect()
                };
                for k in (k0..k1).rev() {
                    let solved = T::dot_slices(&l[k + 1 + k * r..k1 + k * r], &head[k + 1..k1]);
                    head[k] = head[k] - t[k - k0] - solved;
                }
            }
            for (&row, v) in rows[..w].iter().zip(&g[..w]) {
                y[row] = *v;
            }
        }
    }
}

impl<T: FloatT> HasLinearSolverInfo for SupernodalLDLSolver<T> {
    fn linear_solver_info(&self) -> LinearSolverInfo {
        LinearSolverInfo {
            name: "supernodal".to_string(),
            threads: self.pool.as_ref().map_or(1, |p| p.current_num_threads()),
            direct: true,
            nnzA: self.nnz,
            nnzL: (0..self.sym.start.len() - 1)
                .map(|s| {
                    let (w, r) = (self.sym.width(s), self.sym.rows[s].len());
                    w * r - w * (w + 1) / 2
                })
                .sum(),
        }
    }
}

impl<T: FloatT> DirectLDLSolver<T> for SupernodalLDLSolver<T> {
    // Values are read from the KKT at each refactor.
    fn update_values(&mut self, _index: &[usize], _values: &[T]) {}

    fn scale_values(&mut self, _index: &[usize], _scale: T) {}

    fn solve(&mut self, _kkt: &CscMatrix<T>, x: &mut [T], b: &mut [T]) {
        let mut y = std::mem::take(&mut self.work);
        for (v, &old) in y.iter_mut().zip(&self.sym.perm) {
            *v = b[old];
        }
        match &self.pool {
            Some(pool) => pool.install(|| self.solve_permuted(&mut y)),
            None => self.solve_permuted(&mut y),
        }
        for (&v, &old) in y.iter().zip(&self.sym.perm) {
            x[old] = v;
        }
        self.work = y;
    }

    fn refactor(&mut self, kkt: &CscMatrix<T>) -> bool {
        let timer = crate::receipt::start();
        let ok = match self.pool.clone() {
            Some(pool) => pool.install(|| self.factor(kkt)),
            None => self.factor(kkt),
        }
        .is_ok();
        crate::receipt::finish("supernodal.factor", timer);
        ok && self.dinv.iter().all(|d| d.is_finite())
    }

    fn set_pool(&mut self, pool: Option<Arc<rayon::ThreadPool>>) {
        self.pool = pool;
    }
}
