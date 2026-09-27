//! Exact fixed-precision matrix products on double-precision BLAS.
//!
//! This is the model of SDPB's `bigint_syrk` (after FLINT's `mul_blas`):
//! each operand is aligned to one binary exponent, so every entry is an exact
//! integer; the integers are reduced modulo word-sized primes; each prime's
//! product is one `dgemm` that is exact in binary64; the Chinese remainder
//! theorem rebuilds every exact dot product; and each output is rounded once,
//! nearest-even. The result therefore equals the correctly rounded exact dot
//! product, the same value `MpFloat::dot_fma` (exactdot) returns.
//!
//! Both residue encoding and CRT reconstruction are also posed as `dgemm`:
//! entries are split into short bit chunks and multiplied against tables of
//! `2^(c·j) mod p` (encode) or chunks of `M/p` (decode). Every f64 product
//! and sum is an integer below 2^53, so no step rounds.
//!
//! A top-level call splits across primes (encode, product) and outputs
//! (decode); calls already running as pool tasks stay serial (see `split_plan`)
//! and stream primes in small groups with an incremental CRT, so a call's
//! working set fits a core's share of cache. Constant operands can keep their
//! residues in a fingerprint-checked [`ResidueCache`].
use rayon::prelude::*;
use sdpx_arithmetic::{DyadicKind, MpFloat};
use num_traits::Zero;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

type F<const N: usize> = MpFloat<N>;

/// Exactness budget of a binary64 integer accumulation.
const EXACT_BITS: f64 = 53.0;
const MIN_PRIME_BITS: u32 = 18;
const MAX_PRIME_BITS: u32 = 26;
/// Primes available per width; bounds the reconstructable dynamic range
/// (about 24k bits at 24-bit primes).
const MAX_PRIMES: usize = 1024;
/// Largest admitted per-operand exponent spread (bits).
const MAX_SPREAD: i64 = 4096;

/// Process-wide cache of plan-dependent tables (never invalidated: they
/// depend only on the prime width, count and chunk width).
fn cached<K: std::hash::Hash + Eq, V>(
    cache: &'static OnceLock<Mutex<HashMap<K, Arc<V>>>>,
    key: K,
    build: impl FnOnce() -> V,
) -> Arc<V> {
    let map = cache.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(v) = map.lock().unwrap().get(&key) {
        return v.clone();
    }
    let v = Arc::new(build());
    map.lock().unwrap().entry(key).or_insert(v).clone()
}

fn is_prime(n: u64) -> bool {
    if n < 2 || n % 2 == 0 {
        return n == 2;
    }
    let mut d = 3;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

/// Descending primes below `2^bits`, computed once per width.
fn primes(bits: u32) -> Arc<Vec<u64>> {
    static CACHE: OnceLock<Mutex<Vec<Option<Arc<Vec<u64>>>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(vec![None; MAX_PRIME_BITS as usize + 1]));
    let mut guard = cache.lock().unwrap();
    guard[bits as usize]
        .get_or_insert_with(|| {
            let mut out = Vec::with_capacity(MAX_PRIMES);
            let mut n = (1u64 << bits) - 1;
            while out.len() < MAX_PRIMES {
                if is_prime(n) {
                    out.push(n);
                }
                n -= 2;
            }
            Arc::new(out)
        })
        .clone()
}

fn mulmod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

fn inverse(a: u64, p: u64) -> u64 {
    let (mut t, mut nt) = (0i128, 1i128);
    let (mut r, mut nr) = (p as i128, (a % p) as i128);
    while nr != 0 {
        let q = r / nr;
        (t, nt) = (nt, t - q * nt);
        (r, nr) = (nr, r - q * nr);
    }
    (if t < 0 { t + p as i128 } else { t }) as u64
}

fn symmetric(r: u64, p: u64) -> f64 {
    if r > p / 2 {
        r as f64 - p as f64
    } else {
        r as f64
    }
}

/// Symmetric residue of an exact integer-valued `v` with `|v| < 2^53`.
#[inline]
fn reduce(v: f64, p: f64, pinv: f64) -> f64 {
    // Nearest integer of |v·pinv| < 2^51 by the 1.5·2^52 shift; any nearby
    // integer works because the result is re-centered below.
    const SHIFT: f64 = 6755399441055744.0;
    let r = v - ((v * pinv + SHIFT) - SHIFT) * p;
    if r > 0.5 * p {
        r - p
    } else if r < -0.5 * p {
        r + p
    } else {
        r
    }
}

/// Bits `[start, start + width)` of the little-endian integer `limbs`;
/// positions below zero read as zero bits.
#[inline]
fn bits_at(limbs: &[u64], start: i64, width: u32) -> u64 {
    if start + width as i64 <= 0 {
        return 0;
    }
    let pad = if start < 0 { (-start) as u32 } else { 0 };
    let s = start.max(0) as usize;
    let (li, off) = (s / 64, s % 64);
    let lo = limbs.get(li).copied().unwrap_or(0) as u128;
    let hi = limbs.get(li + 1).copied().unwrap_or(0) as u128;
    let x = (((hi << 64) | lo) >> off) as u64;
    let w = width - pad;
    (x & ((1u64 << w) - 1)) << pad
}

fn ceil_log2(x: usize) -> u32 {
    usize::BITS - x.saturating_sub(1).leading_zeros()
}

thread_local! {
    static SPLIT_HINT: std::cell::Cell<usize> = const { std::cell::Cell::new(1) };
    static BUFFERS: std::cell::RefCell<Vec<Vec<f64>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Buffers kept per thread (enough for one call's operands, scratch and
/// accumulator) and their total size cap, in entries (32 MiB).
const POOLED_BUFFERS: usize = 16;
const POOLED_ENTRIES: usize = 4 << 20;

/// A zeroed buffer of `len` entries, reusing one this thread released.
/// Kernel calls repeat with the same shapes every iteration; reuse avoids
/// fresh large allocations (page faults, `mmap`/`munmap`) that stop a
/// block's kernels scaling across threads. Contents never carry over.
fn take_buffer(len: usize) -> Vec<f64> {
    let mut v = BUFFERS
        .with(|b| {
            let mut b = b.borrow_mut();
            // Smallest buffer that fits, else the largest (it grows once).
            let i = (0..b.len()).min_by_key(|&i| match b[i].capacity() {
                c if c >= len => (0, c),
                c => (1, usize::MAX - c),
            })?;
            Some(b.swap_remove(i))
        })
        .unwrap_or_default();
    v.clear();
    v.resize(len, 0.0);
    v
}

fn release_buffer(v: Vec<f64>) {
    if v.capacity() == 0 {
        return;
    }
    BUFFERS.with(|b| {
        let mut b = b.borrow_mut();
        let held: usize = b.iter().map(Vec::capacity).sum();
        if b.len() < POOLED_BUFFERS && held + v.capacity() <= POOLED_ENTRIES {
            b.push(v);
        }
    });
}

/// Run `f` with this thread's residue products split into `ways` prime
/// groups on the ambient pool. Callers give extra ways only to blocks whose
/// measured cost exceeds a worker's fair share (SDPB-style worst-fit), so
/// light blocks stay serial and cache use stays bounded.
/// Ways granted to a block of measured `cost` when the pool's fair share is
/// `share` (SDPB-style measured load balancing).
pub(crate) fn measured_ways(cost: f64, share: f64) -> usize {
    ((cost / share).floor() as usize).clamp(1, 8)
}

pub(crate) fn with_split_hint<R>(ways: usize, f: impl FnOnce() -> R) -> R {
    let previous = SPLIT_HINT.with(|c| c.replace(ways.max(1)));
    let result = f();
    SPLIT_HINT.with(|c| c.set(previous));
    result
}

/// How one product is split. A top-level call (not on a rayon worker) with a
/// pool splits across all primes. A call already running as a pool task is
/// one of many concurrent block tasks: it stays serial unless its caller
/// granted it extra ways (`with_split_hint`), splitting across primes for
/// every call interleaves unrelated products and was measured to slow wide
/// pools.
enum Split<'a> {
    Pool(&'a rayon::ThreadPool),
    Ways(usize),
    Serial,
}

fn split_plan(pool: Option<&rayon::ThreadPool>) -> Split<'_> {
    if let Some(p) = pool.filter(|p| p.current_num_threads() > 1 && rayon::current_thread_index().is_none()) {
        return Split::Pool(p);
    }
    let ways = SPLIT_HINT.with(|c| c.get());
    if ways > 1 && rayon::current_thread_index().is_some() {
        Split::Ways(ways)
    } else {
        Split::Serial
    }
}

/// Exact product of integer-valued f64 matrices whose partial sums stay below
/// 2^53 (every caller checks that bound): `c = op(a)·op(b)` with `ldc`. Any
/// summation order gives the same integers, so any GEMM kernel is exact here.
#[allow(clippy::too_many_arguments)]
fn int_gemm(
    ta: u8,
    tb: u8,
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    local_gemm(ta, tb, m, n, k, a, lda, b, ldb, c, ldc, false);
}

/// `c += op(a)·op(b)` under the same exactness bound as [`int_gemm`].
#[allow(clippy::too_many_arguments)]
fn int_gemm_acc(
    ta: u8,
    tb: u8,
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    local_gemm(ta, tb, m, n, k, a, lda, b, ldb, c, ldc, true);
}

/// Column-major `c = op(a)·op(b)` (or `c +=` when `accumulate`) on the
/// calling thread with thread-local packing. Every product and partial sum
/// is an exact integer below 2^53, so the result does not depend on the
/// kernel's summation order.
#[allow(clippy::too_many_arguments)]
fn local_gemm(
    ta: u8,
    tb: u8,
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
    accumulate: bool,
) {
    // (row stride, column stride) of op(x) for a column-major x.
    let strides = |t: u8, ld: usize| if t == b'N' { (1, ld as isize) } else { (ld as isize, 1) };
    let (a_rs, a_cs) = strides(ta, lda);
    let (b_rs, b_cs) = strides(tb, ldb);
    let a_end = if ta == b'N' { (m - 1) + (k.max(1) - 1) * lda } else { (k.max(1) - 1) + (m - 1) * lda };
    let b_end = if tb == b'N' { (k.max(1) - 1) + (n - 1) * ldb } else { (n - 1) + (k.max(1) - 1) * ldb };
    assert!(k == 0 || (a_end < a.len() && b_end < b.len()));
    assert!((m - 1) + (n - 1) * ldc < c.len());
    // SAFETY: the asserts bound every strided access of the three operands.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            ldc as isize,
            1,
            accumulate,
            a.as_ptr(),
            a_cs,
            a_rs,
            b.as_ptr(),
            b_cs,
            b_rs,
            1.0,
            1.0,
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
    }
}

/// Dense operand view: `rows × cols` stored column-major with leading dimension `ld`.
#[derive(Clone, Copy)]
struct View<'a, const N: usize> {
    data: &'a [F<N>],
    rows: usize,
    cols: usize,
    ld: usize,
}

impl<const N: usize> View<'_, N> {
    fn len(&self) -> usize {
        if self.rows == 0 || self.cols == 0 {
            0
        } else {
            (self.cols - 1) * self.ld + self.rows
        }
    }
    fn stored(&self, e: usize) -> bool {
        e % self.ld < self.rows
    }
    /// `(min, max)` of `exponent - PRECISION_BITS` over non-zero entries;
    /// `None` for a non-finite entry. An all-zero operand gives `(MAX, MIN)`.
    fn exponent_range(&self) -> Option<(i64, i64)> {
        let (mut lo, mut hi) = (i64::MAX, i64::MIN);
        for (e, v) in self.data[..self.len()].iter().enumerate() {
            if !self.stored(e) {
                continue;
            }
            let view = v.dyadic_view();
            match view.kind {
                DyadicKind::Zero => {}
                DyadicKind::Finite { .. } => {
                    let x = view.exponent as i64 - F::<N>::PRECISION_BITS as i64;
                    lo = lo.min(x);
                    hi = hi.max(x);
                }
                _ => return None,
            }
        }
        Some((lo, hi))
    }
}

/// Primes and derived constants for one product.
struct Plan {
    bits: u32,
    p: Vec<f64>,
    pinv: Vec<f64>,
    primes: Vec<u64>,
    /// Inner-dimension chunk that keeps one per-prime `dgemm` exact.
    k_chunk: usize,
}

impl Plan {
    fn new(k: usize, needed_bits: f64) -> Option<Self> {
        // |residue| < 2^(bits-1); k·2^(2bits-2) must stay below 2^53.
        let bits = ((EXACT_BITS + 2.0 - ceil_log2(k.max(1)) as f64) / 2.0).floor() as u32;
        let bits = bits.clamp(MIN_PRIME_BITS, MAX_PRIME_BITS);
        let k_chunk = 1usize << (53 + 2 - 2 * bits).min(40);
        let table = primes(bits);
        let (mut have, mut count) = (0.0f64, 0usize);
        while have < needed_bits {
            if count == table.len() {
                return None;
            }
            have += (table[count] as f64).log2();
            count += 1;
        }
        let primes = table[..count].to_vec();
        let p: Vec<f64> = primes.iter().map(|&q| q as f64).collect();
        let pinv = p.iter().map(|q| 1.0 / q).collect();
        Some(Self {
            bits,
            p,
            pinv,
            primes,
            k_chunk,
        })
    }
    fn count(&self) -> usize {
        self.primes.len()
    }

    /// The first `count` primes of width `bits` (a prefix of the same table
    /// every plan of this width draws from).
    fn with_count(bits: u32, count: usize) -> Self {
        let primes = primes(bits)[..count].to_vec();
        let p: Vec<f64> = primes.iter().map(|&q| q as f64).collect();
        let pinv = p.iter().map(|q| 1.0 / q).collect();
        Self {
            bits,
            p,
            pinv,
            primes,
            k_chunk: 1usize << (53 + 2 - 2 * bits).min(40),
        }
    }
}

/// Primes per streamed group: small enough that one group's residues,
/// products and CRT update stay in a core's share of a 16 MB CCX L3.
const STREAM_GROUP: usize = 16;

/// An operand's exact integer image split into `width`-bit chunks, entry-
/// contiguous (`e[j + r·chunks]`). Residues of any prime group follow from one
/// small `dgemm` against the weights `2^(width·j) mod p`, so the chunk matrix
/// is built once and reused by every group.
struct ChunkMatrix {
    e: Vec<f64>,
    chunks: usize,
    width: u32,
    len: usize,
}

impl Drop for ChunkMatrix {
    fn drop(&mut self) {
        release_buffer(std::mem::take(&mut self.e));
    }
}

/// Rows encoded per way, at least; below this a split costs more than it saves.
const MIN_ROWS_PER_WAY: usize = 256;

fn chunk_matrix<const N: usize>(x: View<'_, N>, lo: i64, spread: i64, plan: &Plan, split: &Split<'_>) -> ChunkMatrix {
    let len = x.len();
    let int_bits = F::<N>::PRECISION_BITS as i64 + spread;
    let mut c = 30u32;
    let chunks = loop {
        let j = (int_bits as usize).div_ceil(c as usize);
        if (j as f64).log2() + c as f64 + (plan.bits - 1) as f64 <= EXACT_BITS - 0.5 || c == 8 {
            break j;
        }
        c -= 1;
    };
    let mut e = take_buffer(len * chunks);
    // Rows are independent; a split call encodes them over its ways.
    let fill = |r0: usize, rows: &mut [f64]| {
        for (i, row) in rows.chunks_mut(chunks).enumerate() {
            let r = r0 + i;
            if !x.stored(r) {
                continue;
            }
            let value = &x.data[r];
            let view = value.dyadic_view();
            let negative = match view.kind {
                DyadicKind::Finite { negative } => negative,
                _ => continue,
            };
            let delta = view.exponent as i64 - F::<N>::PRECISION_BITS as i64 - lo;
            let limbs = value.exact_encode().2;
            let j0 = (delta / c as i64) as usize;
            let j1 = ((delta + F::<N>::PRECISION_BITS as i64) as usize).div_ceil(c as usize).min(chunks);
            for (j, slot) in row.iter_mut().enumerate().take(j1).skip(j0) {
                let bits = bits_at(limbs, c as i64 * j as i64 - delta, c) as f64;
                *slot = if negative { -bits } else { bits };
            }
        }
    };
    let ways = split_ways(split).min(len / MIN_ROWS_PER_WAY).max(1);
    if ways <= 1 || chunks == 0 {
        fill(0, &mut e);
    } else {
        let per = len.div_ceil(ways);
        let mut run = || e.par_chunks_mut(per * chunks).enumerate().for_each(|(w, rows)| fill(w * per, rows));
        match split {
            Split::Pool(p) => p.install(run),
            _ => run(),
        }
    }
    ChunkMatrix { e, chunks, width: c, len }
}

fn weights(plan: &Plan, width: u32, chunks: usize) -> Arc<Vec<f64>> {
    static WEIGHTS: OnceLock<Mutex<HashMap<(u32, usize, u32, usize), Arc<Vec<f64>>>>> = OnceLock::new();
    let kp = plan.count();
    cached(&WEIGHTS, (plan.bits, kp, width, chunks), || {
        let mut w = vec![0.0f64; chunks * kp];
        for (q, &p) in plan.primes.iter().enumerate() {
            let step = (1u64 << width) % p;
            let mut v = 1 % p;
            for j in 0..chunks {
                w[j + q * chunks] = symmetric(v, p);
                v = mulmod(v, step, p);
            }
        }
        w
    })
}

/// Symmetric residues of primes `q0..q1`, prime-major: `out[(q-q0)·len + e]`.
fn group_residues(m: &ChunkMatrix, plan: &Plan, q0: usize, q1: usize, out: &mut Vec<f64>) {
    let g = q1 - q0;
    out.clear();
    out.resize(m.len * g, 0.0);
    let w = weights(plan, m.width, m.chunks);
    int_gemm(b'T', b'N', m.len, g, m.chunks, &m.e, m.chunks, &w[q0 * m.chunks..q1 * m.chunks], m.chunks, out, m.len);
    for (qi, col) in out.chunks_mut(m.len.max(1)).enumerate().take(g) {
        let q = q0 + qi;
        for x in col {
            *x = reduce(*x, plan.p[q], plan.pinv[q]);
        }
    }
}

/// Residues of an operand that stays constant across many products (the
/// sampled basis for a whole solve, a scaling factor for an iteration),
/// stored as f32 (exact: |r| < 2^24 for prime widths <= 25 bits). An entry is
/// used only when the operand's exact-bit fingerprint, alignment and prime
/// width match, so a changed operand can never reuse stale residues; it is
/// simply re-encoded.
#[derive(Default)]
pub struct ResidueCache {
    entry: Option<CacheEntry>,
}

struct CacheEntry {
    fingerprint: u64,
    bits: u32,
    lo: i64,
    spread: i64,
    len: usize,
    count: usize,
    res: Vec<f32>,
}

/// Spare primes encoded beyond the current plan, so later calls whose other
/// operand has a slightly wider exponent spread still hit the cache.
const CACHE_SPARE_PRIMES: usize = 8;
const CACHE_MAX_BITS: u32 = 25;

/// 64-bit mix of every stored entry's exact representation.
fn fingerprint<const N: usize>(x: View<'_, N>) -> u64 {
    let mut h = 0x9e37_79b9_7f4a_7c15u64 ^ (x.rows as u64) ^ ((x.cols as u64) << 32);
    let mut mix = |v: u64| {
        h = (h ^ v).wrapping_mul(0x0100_0000_01b3).rotate_left(29);
    };
    for (e, v) in x.data[..x.len()].iter().enumerate() {
        if !x.stored(e) {
            continue;
        }
        let (kind, exponent, limbs) = v.exact_encode();
        mix(kind as u64);
        mix(exponent as u64);
        for &l in limbs.iter() {
            mix(l);
        }
    }
    h
}

/// Where one operand's per-group residues come from.
enum Operand<'a> {
    Chunks(ChunkMatrix),
    Cached(&'a CacheEntry),
}

impl Operand<'_> {
    fn residues(&self, plan: &Plan, q0: usize, q1: usize, out: &mut Vec<f64>) {
        match self {
            Operand::Chunks(m) => group_residues(m, plan, q0, q1, out),
            Operand::Cached(c) => {
                out.clear();
                out.extend(c.res[q0 * c.len..q1 * c.len].iter().map(|&v| v as f64));
            }
        }
    }
}

impl ResidueCache {
    /// The operand's residues for `plan` (`lo`/`spread` are its alignment),
    /// from the cache when it matches, otherwise freshly encoded and stored.
    fn operand<'a, const N: usize>(
        &'a mut self,
        x: View<'_, N>,
        lo: i64,
        spread: i64,
        plan: &Plan,
        split: &Split<'_>,
    ) -> Operand<'a> {
        if plan.bits > CACHE_MAX_BITS {
            return Operand::Chunks(chunk_matrix(x, lo, spread, plan, split));
        }
        let fp = fingerprint(x);
        let hit = self.entry.as_ref().is_some_and(|c| {
            c.fingerprint == fp && c.bits == plan.bits && c.lo == lo && c.spread == spread
                && c.len == x.len() && c.count >= plan.count()
        });
        if !hit {
            let count = (plan.count() + CACHE_SPARE_PRIMES).min(primes(plan.bits).len());
            let wide = Plan::with_count(plan.bits, count);
            let chunks = chunk_matrix(x, lo, spread, &wide, split);
            let mut res = Vec::new();
            group_residues(&chunks, &wide, 0, count, &mut res);
            self.entry = Some(CacheEntry {
                fingerprint: fp,
                bits: plan.bits,
                lo,
                spread,
                len: x.len(),
                count,
                res: res.iter().map(|&v| v as f32).collect(),
            });
        }
        Operand::Cached(self.entry.as_ref().unwrap())
    }
}

/// Streamed CRT: `X = Σ_q r'_q·(M/p_q) − round(Σ_q r'_q/p_q)·M` is linear in
/// the primes, so each group's residues update `Y` (digit sums) and `frac`
/// and are then discarded. Every `Y` entry is an integer below 2^53 at every
/// step (the plan bounds the full sum), so the result equals [`decode`]'s.
struct CrtAccumulator {
    crt: Arc<Crt>,
    selected: Vec<usize>,
    y: Vec<f64>,
    frac: Vec<f64>,
    r: Vec<f64>,
}

impl CrtAccumulator {
    fn new(plan: &Plan, selected: Vec<usize>) -> Self {
        static CRT: OnceLock<Mutex<HashMap<(u32, usize), Arc<Crt>>>> = OnceLock::new();
        let crt = cached(&CRT, (plan.bits, plan.count()), || crt(plan));
        let n = selected.len();
        let y = take_buffer(n * crt.chunks);
        Self { crt, selected, y, frac: take_buffer(n), r: take_buffer(0) }
    }

    /// Fold primes `q0..q1`, whose reduced residues are `prod[(q-q0)·outputs + o]`.
    fn add(&mut self, plan: &Plan, q0: usize, q1: usize, prod: &[f64], outputs: usize) {
        let (n, g, rows) = (self.selected.len(), q1 - q0, plan.count() + 1);
        self.r.clear();
        self.r.resize(n * g, 0.0);
        for q in q0..q1 {
            let (p, pinv, u) = (plan.p[q], plan.pinv[q], self.crt.u[q]);
            let src = &prod[(q - q0) * outputs..(q - q0 + 1) * outputs];
            let dst = &mut self.r[(q - q0) * n..(q - q0 + 1) * n];
            for ((d, f), &o) in dst.iter_mut().zip(self.frac.iter_mut()).zip(&self.selected) {
                // |r|·|u| < 2^(2bits-2) <= 2^52: exact in binary64.
                let v = reduce(src[o] * u, p, pinv);
                *d = v;
                *f += v * pinv;
            }
        }
        int_gemm_acc(b'N', b'N', n, self.crt.chunks, g, &self.r, n, &self.crt.table[q0..], rows, &mut self.y, n);
    }

    /// Fold another range's accumulator in. Digit sums are exact integers
    /// below 2^53, and `frac` only feeds `round()` with a 1/4 margin, so the
    /// merged state rounds exactly like a single serial pass.
    fn merge(&mut self, other: &Self) {
        for (y, &o) in self.y.iter_mut().zip(&other.y) {
            *y += o;
        }
        for (f, &o) in self.frac.iter_mut().zip(&other.frac) {
            *f += o;
        }
    }

    /// Subtract `round(frac)·M`, then pack and round each output once;
    /// outputs are independent, so a split call shares them over its ways.
    fn finish<const N: usize>(mut self, plan: &Plan, scale: i64, out: &mut [F<N>], split: &Split<'_>) -> bool {
        let (n, rows, kp) = (self.selected.len(), plan.count() + 1, plan.count());
        let ways = split_ways(split).min(n.div_ceil(64)).max(1);
        let failed = std::sync::atomic::AtomicBool::new(false);
        let yptr = SendPtr(self.y.as_mut_ptr());
        let optr = SendValues(out.as_mut_ptr());
        let (crt, frac, selected) = (&self.crt, &self.frac, &self.selected);
        let range = |w: usize| {
            let mut mag = Vec::new();
            for i in w * n / ways..(w + 1) * n / ways {
                // SAFETY: output i owns y[i + l·n] for every l and out[selected[i]].
                let y = |l: usize| unsafe { yptr.get().add(i + l * n) };
                let kk = frac[i].round();
                if kk != 0.0 {
                    for l in 0..crt.chunks {
                        unsafe { *y(l) -= kk * crt.table[kp + l * rows] };
                    }
                }
                let digits = (0..crt.chunks).map(|l| unsafe { *y(l) });
                let value = match pack(digits, crt.width, crt.chunks, &mut mag) {
                    Some(_) if mag.is_empty() => F::zero(),
                    Some(negative) => F::from_scaled_integer(negative, &mag, scale),
                    None => {
                        failed.store(true, std::sync::atomic::Ordering::Relaxed);
                        return;
                    }
                };
                unsafe { *optr.get().add(selected[i]) = value };
            }
        };
        run_ways(split, ways, range);
        !failed.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Drop for CrtAccumulator {
    fn drop(&mut self) {
        for v in [&mut self.y, &mut self.frac, &mut self.r] {
            release_buffer(std::mem::take(v));
        }
    }
}

/// Per-way scratch of the streamed kernels (pooled buffers).
struct Scratch {
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    prod: Vec<f64>,
    t: Vec<f64>,
    u: Vec<f64>,
}

impl Default for Scratch {
    fn default() -> Self {
        Self {
            a: take_buffer(0),
            b: take_buffer(0),
            c: take_buffer(0),
            prod: take_buffer(0),
            t: take_buffer(0),
            u: take_buffer(0),
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        for v in [&mut self.a, &mut self.b, &mut self.c, &mut self.prod, &mut self.t, &mut self.u] {
            release_buffer(std::mem::take(v));
        }
    }
}

/// Fewest primes worth a way of their own.
const MIN_PRIMES_PER_WAY: usize = 8;

fn split_ways(split: &Split<'_>) -> usize {
    match split {
        Split::Serial => 1,
        Split::Ways(w) => *w,
        Split::Pool(p) => p.current_num_threads(),
    }
}

/// Run `f(0..ways)` on the calling thread, the current pool (`Ways`) or the
/// given pool (`Pool`).
fn run_ways(split: &Split<'_>, ways: usize, f: impl Fn(usize) + Sync + Send) {
    if ways <= 1 {
        (0..ways.max(1)).for_each(f);
        return;
    }
    match split {
        Split::Pool(p) => p.install(|| (0..ways).into_par_iter().for_each(&f)),
        _ => (0..ways).into_par_iter().for_each(&f),
    }
}

/// SDPB-style sharing of one heavy call by several workers: primes `0..kp`
/// split into contiguous ranges, one per way; each way streams its range in
/// groups of at most [`STREAM_GROUP`] through `group` with its own scratch
/// and accumulator, and the accumulators merge in range order. With one way
/// this is exactly the serial streamed pass.
fn stream_primes(
    plan: &Plan,
    split: &Split<'_>,
    selected: &[usize],
    group: impl Fn(&mut CrtAccumulator, usize, usize, &mut Scratch) + Sync + Send,
) -> CrtAccumulator {
    let kp = plan.count();
    let ways = split_ways(split).min(kp.div_ceil(MIN_PRIMES_PER_WAY)).max(1);
    let run = |w: usize| {
        let (lo, hi) = (w * kp / ways, (w + 1) * kp / ways);
        let mut acc = CrtAccumulator::new(plan, selected.to_vec());
        let mut scratch = Scratch::default();
        let mut q0 = lo;
        while q0 < hi {
            let q1 = (q0 + STREAM_GROUP).min(hi);
            group(&mut acc, q0, q1, &mut scratch);
            q0 = q1;
        }
        acc
    };
    if ways == 1 {
        return run(0);
    }
    let parts: Vec<CrtAccumulator> = match split {
        Split::Pool(p) => p.install(|| (0..ways).into_par_iter().map(run).collect()),
        _ => (0..ways).into_par_iter().map(run).collect(),
    };
    let mut parts = parts.into_iter();
    let mut total = parts.next().unwrap();
    let rest: Vec<CrtAccumulator> = parts.collect();
    for part in &rest {
        total.merge(part);
    }
    // Release the merged parts' buffers on pool workers, not all on the
    // calling thread, so every worker's buffer pool stays stocked.
    match split {
        Split::Pool(p) => p.install(|| rest.into_par_iter().for_each(drop)),
        _ => rest.into_par_iter().for_each(drop),
    }
    total
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    fn get(self) -> *mut f64 {
        self.0
    }
}

/// CRT reconstruction constants: `u_q = (M/p_q)^{-1} mod p_q` and the
/// `(K+1) × L` chunk table of `M/p_q` (rows `q < K`) and `-M` scaled row.
struct Crt {
    /// Symmetric `u_q`.
    u: Vec<f64>,
    table: Vec<f64>,
    width: u32,
    chunks: usize,
}

fn crt(plan: &Plan) -> Crt {
    let kp = plan.count();
    let mut modulus = vec![1u64];
    for &p in &plan.primes {
        mul_word(&mut modulus, p);
    }
    let cofactors: Vec<Vec<u64>> = plan
        .primes
        .iter()
        .map(|&p| div_word(&modulus, p))
        .collect();
    let u = plan
        .primes
        .iter()
        .zip(&cofactors)
        .map(|(&p, cof)| symmetric(inverse(rem_word(cof, p), p), p))
        .collect();
    // (K+1)·2^(bits-1)·2^width < 2^53.
    let width = (EXACT_BITS - 0.5 - (plan.bits - 1) as f64 - ((kp + 1) as f64).log2()).floor() as u32;
    let total_bits = modulus.len() * 64;
    let chunks = total_bits.div_ceil(width as usize) + 1;
    let rows = kp + 1;
    let mut table = vec![0.0f64; rows * chunks];
    for l in 0..chunks {
        let start = (l * width as usize) as i64;
        for (q, cof) in cofactors.iter().enumerate() {
            table[q + l * rows] = bits_at(cof, start, width) as f64;
        }
        table[kp + l * rows] = bits_at(&modulus, start, width) as f64;
    }
    Crt {
        u,
        table,
        width,
        chunks,
    }
}

fn mul_word(x: &mut Vec<u64>, m: u64) {
    let mut carry = 0u128;
    for limb in x.iter_mut() {
        let cur = *limb as u128 * m as u128 + carry;
        *limb = cur as u64;
        carry = cur >> 64;
    }
    if carry != 0 {
        x.push(carry as u64);
    }
}

fn div_word(x: &[u64], d: u64) -> Vec<u64> {
    let mut out = vec![0u64; x.len()];
    let mut rem = 0u128;
    for i in (0..x.len()).rev() {
        let cur = (rem << 64) | x[i] as u128;
        out[i] = (cur / d as u128) as u64;
        rem = cur % d as u128;
    }
    out
}

fn rem_word(x: &[u64], d: u64) -> u64 {
    let mut rem = 0u128;
    for &limb in x.iter().rev() {
        rem = ((rem << 64) | limb as u128) % d as u128;
    }
    rem as u64
}

/// Pack signed base-`2^width` digit sums into a sign and magnitude limbs.
/// Returns `None` if the value does not fit the digit range (never for a
/// correctly sized plan).
fn pack(digits: impl Iterator<Item = f64>, width: u32, chunks: usize, mag: &mut Vec<u64>) -> Option<bool> {
    let nbits = width as usize * chunks;
    mag.clear();
    mag.resize(nbits.div_ceil(64) + 1, 0);
    // |y| < 2^53 and |carry| < 2^(53-width+1): i64 never overflows.
    let mut carry = 0i64;
    let mask = (1i64 << width) - 1;
    for (l, y) in digits.enumerate() {
        let t = y as i64 + carry;
        let d = t & mask;
        carry = t >> width;
        let pos = l * width as usize;
        let (li, off) = (pos / 64, pos % 64);
        let v = (d as u128) << off;
        mag[li] |= v as u64;
        if off + width as usize > 64 {
            mag[li + 1] |= (v >> 64) as u64;
        }
    }
    match carry {
        0 => {}
        -1 => {
            // Two's complement over `nbits` bits gives |X| = 2^nbits - D.
            let mut borrow_one = true;
            for limb in mag.iter_mut() {
                *limb = !*limb;
                if borrow_one {
                    let (v, o) = limb.overflowing_add(1);
                    *limb = v;
                    borrow_one = o;
                }
            }
            let used = nbits % 64;
            let top = nbits / 64;
            if used != 0 {
                mag[top] &= (1u64 << used) - 1;
            } else {
                mag[top] = 0;
            }
            for limb in mag.iter_mut().skip(top + 1) {
                *limb = 0;
            }
        }
        _ => return None,
    }
    while mag.last() == Some(&0) {
        mag.pop();
    }
    Some(carry == -1)
}

/// Correctly rounded `op(A)·op(B)` (`m × n`, inner `k`), written as
/// `out[i + j·m]`. With `upper_only`, only `i <= j` is produced (the rest is
/// left untouched). Returns `false` when the product does not admit an exact
/// plan (non-finite entry, extreme spread, too many primes); `out` is then
/// unwritten, except for a reconstruction-range failure the plan excludes.
#[allow(clippy::too_many_arguments)]
pub(super) fn gemm<const N: usize>(
    ta: u8,
    tb: u8,
    m: usize,
    n: usize,
    k: usize,
    a: &[F<N>],
    lda: usize,
    b: &[F<N>],
    ldb: usize,
    upper_only: bool,
    pool: Option<&rayon::ThreadPool>,
    out: &mut [F<N>],
    cache_b: Option<&mut ResidueCache>,
) -> bool {
    let ta_n = ta.to_ascii_uppercase() == b'N';
    let tb_n = tb.to_ascii_uppercase() == b'N';
    let av = View {
        data: a,
        rows: if ta_n { m } else { k },
        cols: if ta_n { k } else { m },
        ld: lda,
    };
    let bv = View {
        data: b,
        rows: if tb_n { k } else { n },
        cols: if tb_n { n } else { k },
        ld: ldb,
    };
    let (Some((lo_a, hi_a)), Some((lo_b, hi_b))) = (av.exponent_range(), bv.exponent_range()) else {
        return false;
    };
    if lo_a == i64::MAX || lo_b == i64::MAX {
        // One operand is zero: the exact product is zero.
        for j in 0..n {
            for i in 0..if upper_only { (j + 1).min(m) } else { m } {
                out[i + j * m] = F::zero();
            }
        }
        return true;
    }
    let (da, db) = (hi_a - lo_a, hi_b - lo_b);
    if da > MAX_SPREAD || db > MAX_SPREAD {
        return false;
    }
    // |S| <= k·2^(P+da)·2^(P+db); a CRT modulus above 4|S| keeps the signed
    // lift's fractional estimate inside (-1/4, 1/4).
    let p_bits = F::<N>::PRECISION_BITS as f64;
    let needed = 2.0 * p_bits + (da + db) as f64 + (k.max(1) as f64).log2() + 3.0;
    let Some(plan) = Plan::new(k, needed) else {
        return false;
    };
    let (len_a, len_b) = (av.len(), bv.len());
    let outputs = m * n;
    let shape = GemmShape { ta, tb, m, n, k, lda, ldb };
    let split = split_plan(pool);
    let ca = chunk_matrix(av, lo_a, da, &plan, &split);
    let cb = match cache_b {
        Some(cache) => cache.operand(bv, lo_b, db, &plan, &split),
        None => Operand::Chunks(chunk_matrix(bv, lo_b, db, &plan, &split)),
    };
    let acc = stream_primes(&plan, &split, &selection(m, n, upper_only), |acc, q0, q1, s| {
        group_residues(&ca, &plan, q0, q1, &mut s.a);
        cb.residues(&plan, q0, q1, &mut s.b);
        s.prod.clear();
        s.prod.resize((q1 - q0) * outputs, 0.0);
        for q in q0..q1 {
            let qi = q - q0;
            shape.prime_product(
                &plan,
                q,
                &s.a[qi * len_a..(qi + 1) * len_a],
                &s.b[qi * len_b..(qi + 1) * len_b],
                &mut s.prod[qi * outputs..(qi + 1) * outputs],
                &mut s.t,
            );
        }
        acc.add(&plan, q0, q1, &s.prod, outputs);
    });
    acc.finish(&plan, lo_a + lo_b, out, &split)
}

/// Shape of one per-prime product `op(A)·op(B)`.
#[derive(Clone, Copy)]
struct GemmShape {
    ta: u8,
    tb: u8,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
}

impl GemmShape {
    /// Exact residue product for prime `q` into `cq` (`m × n`), reduced; the
    /// inner dimension is chunked so every binary64 sum stays below 2^53.
    fn prime_product(&self, plan: &Plan, q: usize, aq: &[f64], bq: &[f64], cq: &mut [f64], part: &mut Vec<f64>) {
        let (ta_n, tb_n) = (self.ta.to_ascii_uppercase() == b'N', self.tb.to_ascii_uppercase() == b'N');
        let (m, n, k, outputs) = (self.m, self.n, self.k, self.m * self.n);
        let mut k0 = 0;
        while k0 < k {
            let kc = plan.k_chunk.min(k - k0);
            let a_off = if ta_n { k0 * self.lda } else { k0 };
            let b_off = if tb_n { k0 } else { k0 * self.ldb };
            if k0 == 0 {
                int_gemm(self.ta, self.tb, m, n, kc, &aq[a_off..], self.lda, &bq[b_off..], self.ldb, cq, m);
                for x in cq.iter_mut() {
                    *x = reduce(*x, plan.p[q], plan.pinv[q]);
                }
            } else {
                part.clear();
                part.resize(outputs, 0.0);
                int_gemm(self.ta, self.tb, m, n, kc, &aq[a_off..], self.lda, &bq[b_off..], self.ldb, part, m);
                for (x, y) in cq.iter_mut().zip(part.iter()) {
                    *x = reduce(*x + reduce(*y, plan.p[q], plan.pinv[q]), plan.p[q], plan.pinv[q]);
                }
            }
            k0 += kc;
        }
    }
}

/// Correctly rounded `op(A)·X·op(A)ᵀ` (`m × m`) for a `k × k` matrix `X`,
/// written as `out[i + j·m]`. The intermediate `X·op(A)ᵀ` stays exact in
/// residues, so every output is the exact congruence rounded once — never
/// less accurate than two rounded products. `upper_only` as in [`gemm`].
#[allow(clippy::too_many_arguments)]
pub(super) fn congruence<const N: usize>(
    ta: u8,
    m: usize,
    k: usize,
    a: &[F<N>],
    lda: usize,
    x: &[F<N>],
    ldx: usize,
    upper_only: bool,
    pool: Option<&rayon::ThreadPool>,
    out: &mut [F<N>],
    cache_a: Option<&mut ResidueCache>,
) -> bool {
    let ta_n = ta.to_ascii_uppercase() == b'N';
    let av = View {
        data: a,
        rows: if ta_n { m } else { k },
        cols: if ta_n { k } else { m },
        ld: lda,
    };
    let xv = View {
        data: x,
        rows: k,
        cols: k,
        ld: ldx,
    };
    let (Some((lo_a, hi_a)), Some((lo_x, hi_x))) = (av.exponent_range(), xv.exponent_range()) else {
        return false;
    };
    if lo_a == i64::MAX || lo_x == i64::MAX {
        for j in 0..m {
            for i in 0..if upper_only { j + 1 } else { m } {
                out[i + j * m] = F::zero();
            }
        }
        return true;
    }
    let (da, dx) = (hi_a - lo_a, hi_x - lo_x);
    if da > MAX_SPREAD || dx > MAX_SPREAD {
        return false;
    }
    // |C| <= k²·2^(2(P+da))·2^(P+dx), with the same 4|C| CRT margin.
    let p_bits = F::<N>::PRECISION_BITS as f64;
    let log_k = (k.max(1) as f64).log2();
    let needed = 3.0 * p_bits + (2 * da + dx) as f64 + 2.0 * log_k + 3.0;
    let Some(plan) = Plan::new(k, needed) else {
        return false;
    };
    if plan.k_chunk < k {
        return false;
    }
    let (len_a, len_x) = (av.len(), xv.len());
    let outputs = m * m;
    let tb = if ta_n { b'T' } else { b'N' };
    // T = X·op(A)ᵀ (k × m), reduced; then C = op(A)·T.
    let prime = |q: usize, aq: &[f64], xq: &[f64], cq: &mut [f64], t: &mut Vec<f64>| {
        let (p, pinv) = (plan.p[q], plan.pinv[q]);
        t.clear();
        t.resize(k * m, 0.0);
        int_gemm(b'N', tb, k, m, k, xq, ldx, aq, lda, t, k);
        for v in t.iter_mut() {
            *v = reduce(*v, p, pinv);
        }
        int_gemm(ta, b'N', m, m, k, aq, lda, t, k, cq, m);
        for v in cq.iter_mut() {
            *v = reduce(*v, p, pinv);
        }
    };
    let split = split_plan(pool);
    let ca = match cache_a {
        Some(cache) => cache.operand(av, lo_a, da, &plan, &split),
        None => Operand::Chunks(chunk_matrix(av, lo_a, da, &plan, &split)),
    };
    let cx = chunk_matrix(xv, lo_x, dx, &plan, &split);
    let acc = stream_primes(&plan, &split, &selection(m, m, upper_only), |acc, q0, q1, s| {
        ca.residues(&plan, q0, q1, &mut s.a);
        group_residues(&cx, &plan, q0, q1, &mut s.b);
        s.prod.clear();
        s.prod.resize((q1 - q0) * outputs, 0.0);
        for q in q0..q1 {
            let qi = q - q0;
            prime(
                q,
                &s.a[qi * len_a..(qi + 1) * len_a],
                &s.b[qi * len_x..(qi + 1) * len_x],
                &mut s.prod[qi * outputs..(qi + 1) * outputs],
                &mut s.t,
            );
        }
        acc.add(&plan, q0, q1, &s.prod, outputs);
    });
    acc.finish(&plan, 2 * lo_a + lo_x, out, &split)
}

/// Correctly rounded `v[k] = Σ_{i≤j} c_ij·x_t·q_ik·q_jk` for `k < kmax`, where
/// `x` is an svec (`t = tri(j) + i`), `q` is `h × kmax` column-major, `c_ii = 1`
/// and `c_ij = sqrt2` (the working-precision constant). The whole weighted
/// quadratic form is one exact residue expression, rounded once: per prime
/// `M = 2^(-σ)·diag(x) + sqrt2_int·triu(x)`, `T = M·Q`, `v_k = Σ_i Q_ik·T_ik`,
/// with `sqrt2 = sqrt2_int·2^σ`.
#[allow(clippy::too_many_arguments)]
pub(super) fn svec_quadratic<const N: usize>(
    h: usize,
    kmax: usize,
    q: &[F<N>],
    x: &[F<N>],
    sqrt2: F<N>,
    pool: Option<&rayon::ThreadPool>,
    out: &mut [F<N>],
    cache_q: Option<&mut ResidueCache>,
) -> bool {
    let trih = h * (h + 1) / 2;
    let qv = View { data: q, rows: h, cols: kmax, ld: h };
    let xv = View { data: x, rows: trih, cols: 1, ld: trih };
    let sv = View { data: std::slice::from_ref(&sqrt2), rows: 1, cols: 1, ld: 1 };
    let (Some((lo_q, hi_q)), Some((lo_x, hi_x)), Some((sigma, _))) =
        (qv.exponent_range(), xv.exponent_range(), sv.exponent_range())
    else {
        return false;
    };
    if lo_q == i64::MAX || lo_x == i64::MAX {
        out[..kmax].iter_mut().for_each(|v| *v = F::zero());
        return true;
    }
    let (dq, dx) = (hi_q - lo_q, hi_x - lo_x);
    let p_bits = F::<N>::PRECISION_BITS as i64;
    // sqrt2 < 2^P·2^σ with σ = e - P <= 0, so 2^(-σ) is an integer weight.
    if dq > MAX_SPREAD || dx > MAX_SPREAD || sigma > 0 || -sigma > p_bits + 64 {
        return false;
    }
    // |M| < 2^(P+dx)·2^max(P, -σ); |v| <= h²·|M|·2^(2(P+dq)).
    let m_bits = (p_bits + dx + p_bits.max(-sigma)) as f64;
    let needed = m_bits + 2.0 * (p_bits + dq) as f64 + 2.0 * (h.max(1) as f64).log2() + 3.0;
    let Some(plan) = Plan::new(h, needed) else {
        return false;
    };
    if plan.k_chunk < h {
        return false;
    }
    let (len_q, len_x) = (qv.len(), xv.len());
    let prime = |qi: usize, qq: &[f64], xq: &[f64], s: f64, vq: &mut [f64], m: &mut Vec<f64>, t: &mut Vec<f64>| {
        let (p, pinv, pu) = (plan.p[qi], plan.pinv[qi], plan.primes[qi]);
        let diag = symmetric(pow2_mod(-sigma as u64, pu), pu);
        m.clear();
        m.resize(h * h, 0.0);
        for j in 0..h {
            for i in 0..j {
                m[i + j * h] = reduce(s * xq[j * (j + 1) / 2 + i], p, pinv);
            }
            m[j + j * h] = reduce(diag * xq[j * (j + 1) / 2 + j], p, pinv);
        }
        t.clear();
        t.resize(h * kmax, 0.0);
        int_gemm(b'N', b'N', h, kmax, h, m, h, qq, h, t, h);
        for (k, v) in vq.iter_mut().enumerate() {
            let mut acc = 0.0;
            for i in 0..h {
                acc += qq[i + k * h] * reduce(t[i + k * h], p, pinv);
            }
            *v = reduce(acc, p, pinv);
        }
    };
    let scale = sigma + lo_x + 2 * lo_q;
    let split = split_plan(pool);
    let cq = match cache_q {
        Some(cache) => cache.operand(qv, lo_q, dq, &plan, &split),
        None => Operand::Chunks(chunk_matrix(qv, lo_q, dq, &plan, &split)),
    };
    let (cx, cs) = (chunk_matrix(xv, lo_x, dx, &plan, &split), chunk_matrix(sv, sigma, 0, &plan, &split));
    let selected: Vec<usize> = (0..kmax).collect();
    let acc = stream_primes(&plan, &split, &selected, |acc, q0, q1, s| {
        cq.residues(&plan, q0, q1, &mut s.a);
        group_residues(&cx, &plan, q0, q1, &mut s.b);
        group_residues(&cs, &plan, q0, q1, &mut s.c);
        s.prod.clear();
        s.prod.resize((q1 - q0) * kmax, 0.0);
        for qi in q0..q1 {
            let g = qi - q0;
            prime(
                qi,
                &s.a[g * len_q..(g + 1) * len_q],
                &s.b[g * len_x..(g + 1) * len_x],
                s.c[g],
                &mut s.prod[g * kmax..(g + 1) * kmax],
                &mut s.t,
                &mut s.u,
            );
        }
        acc.add(&plan, q0, q1, &s.prod, kmax);
    });
    acc.finish(&plan, scale, out, &split)
}

fn pow2_mod(e: u64, p: u64) -> u64 {
    let (mut base, mut e, mut acc) = (2 % p, e, 1 % p);
    while e > 0 {
        if e & 1 == 1 {
            acc = mulmod(acc, base, p);
        }
        base = mulmod(base, base, p);
        e >>= 1;
    }
    acc
}

/// Column-major output indices: all of `m × n`, or its upper triangle.
fn selection(m: usize, n: usize, upper_only: bool) -> Vec<usize> {
    if upper_only {
        (0..n)
            .flat_map(|j| (0..(j + 1).min(m)).map(move |i| i + j * m))
            .collect()
    } else {
        (0..m * n).collect()
    }
}

#[derive(Clone, Copy)]
struct SendValues<T>(*mut T);
unsafe impl<T> Send for SendValues<T> {}
unsafe impl<T> Sync for SendValues<T> {}
impl<T> SendValues<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

#[cfg(test)]
#[path = "rns_blas_tests.rs"]
mod tests;
