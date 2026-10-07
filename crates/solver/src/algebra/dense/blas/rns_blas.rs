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
//! residues in a [`ResidueCache`]. General products check fingerprints;
//! constant diagonal operands rely on their owner to invalidate the cache.
use num_traits::{FromPrimitive, Zero};
use rayon::prelude::*;
use sdpx_arithmetic::{DyadicKind, MpFloat};
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
/// i128 slots of all ways of one exact block product (256 MiB).
const SLOT_BUDGET: usize = 16 << 20;

/// Retained plan tables are capped at 64 MiB per table family. Eviction
/// releases only the cache's references; active kernels keep their Arc.
trait CacheBytes {
    fn cache_bytes(&self) -> usize;
}
impl CacheBytes for Vec<f64> {
    fn cache_bytes(&self) -> usize {
        self.capacity() * 8
    }
}
fn cached<K: std::hash::Hash + Eq, V: CacheBytes>(
    cache: &'static OnceLock<Mutex<HashMap<K, Arc<V>>>>,
    key: K,
    build: impl FnOnce() -> V,
) -> Arc<V> {
    const LIMIT: usize = 64 << 20;
    let map = cache.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(v) = map.lock().unwrap().get(&key) {
        return v.clone();
    }
    let v = Arc::new(build());
    let mut map = map.lock().unwrap();
    if let Some(existing) = map.get(&key) {
        return existing.clone();
    }
    let bytes = v.cache_bytes();
    if bytes > LIMIT {
        return v;
    }
    if map.values().map(|v| v.cache_bytes()).sum::<usize>() + bytes > LIMIT {
        map.clear();
    }
    map.insert(key, v.clone());
    v
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

/// Balanced operands have |x*y/p| < p/4 < 2^24. Reciprocal/product
/// error is < 2^-28, below the 1/(2p) distance to a half-integer for odd p.
/// The rounded quotient is exact, so no re-centering is needed.
#[inline]
fn residue_product(x: f64, y: f64, p: f64, pinv: f64) -> f64 {
    const SHIFT: f64 = 6755399441055744.0;
    let v = x * y;
    v - ((v * pinv + SHIFT) - SHIFT) * p
}

fn ceil_log2(x: usize) -> u32 {
    usize::BITS - x.saturating_sub(1).leading_zeros()
}

fn prime_bits(k: usize) -> u32 {
    // |residue| < 2^(bits-1); k·2^(2bits-2) must stay below 2^53.
    let bits = ((EXACT_BITS + 2.0 - ceil_log2(k.max(1)) as f64) / 2.0).floor() as u32;
    bits.clamp(MIN_PRIME_BITS, MAX_PRIME_BITS)
}

thread_local! {
    static SPLIT_HINT: std::cell::Cell<usize> = const { std::cell::Cell::new(1) };
    static BUFFERS: std::cell::RefCell<BufferPool> = const { std::cell::RefCell::new(BufferPool(Vec::new())) };
}

/// Buffers kept per thread (enough for one call's operands, scratch and
/// accumulator) and their total size cap, in entries (32 MiB).
const POOLED_BUFFERS: usize = 16;
const POOLED_ENTRIES: usize = 4 << 20;
/// Buffers above this size are not pooled: the very tall operands of
/// Schur-style products would otherwise pin hundreds of MiB per thread.
const POOLED_MAX_ENTRIES: usize = 2 << 20;
// The per-thread limit prevents monopolization; this process-wide cap keeps
// idle storage bounded even with hundreds of workers or multiple solver pools.
const GLOBAL_POOLED_ENTRIES: usize = 16 << 20;
static POOLED_TOTAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
struct BufferPool(Vec<Vec<f64>>);
impl Drop for BufferPool {
    fn drop(&mut self) {
        POOLED_TOTAL.fetch_sub(
            self.0.iter().map(Vec::capacity).sum(),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

/// A zeroed buffer of `len` entries, reusing one this thread released.
/// Kernel calls repeat with the same shapes every iteration; reuse avoids
/// fresh large allocations (page faults, `mmap`/`munmap`) that stop a
/// block's kernels scaling across threads. Contents never carry over.
fn take_buffer(len: usize) -> Vec<f64> {
    let mut v = BUFFERS
        .with(|b| {
            let mut pool = b.borrow_mut();
            let b = &mut pool.0;
            // Smallest buffer that fits, else the largest (it grows once).
            let i = (0..b.len()).min_by_key(|&i| match b[i].capacity() {
                c if c >= len => (0, c),
                c => (1, usize::MAX - c),
            })?;
            let v = b.swap_remove(i);
            POOLED_TOTAL.fetch_sub(v.capacity(), std::sync::atomic::Ordering::Relaxed);
            Some(v)
        })
        .unwrap_or_default();
    v.clear();
    v.resize(len, 0.0);
    v
}

fn release_buffer(v: Vec<f64>) {
    if v.capacity() == 0 || v.capacity() > POOLED_MAX_ENTRIES {
        return;
    }
    BUFFERS.with(|b| {
        let mut pool = b.borrow_mut();
        let b = &mut pool.0;
        let held: usize = b.iter().map(Vec::capacity).sum();
        if b.len() < POOLED_BUFFERS && held + v.capacity() <= POOLED_ENTRIES {
            let capacity = v.capacity();
            if POOLED_TOTAL
                .fetch_update(
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                    |held| (held + capacity <= GLOBAL_POOLED_ENTRIES).then_some(held + capacity),
                )
                .is_ok()
            {
                b.push(v);
            }
        }
    });
}

/// SDPB-style worker assignment for blocks of measured `costs` on `workers`:
/// with at least as many workers as active blocks, the smallest makespan `M`
/// with `Σ ceil(cost/M) ≤ workers` gives each block `ceil(cost/M)` ways (at
/// most `MAX_WAYS`), so spare workers go to the heaviest blocks; otherwise
/// every block keeps one way. Zero (unmeasured) costs keep one way.
pub(crate) fn makespan_ways(costs: &[f64], workers: usize) -> Vec<usize> {
    const MAX_WAYS: usize = 8;
    let active = costs.iter().filter(|&&c| c > 0.0).count();
    let total: f64 = costs.iter().filter(|&&c| c > 0.0).sum();
    let largest = costs.iter().copied().fold(0.0f64, f64::max);
    if workers <= 1 || total <= 0.0 || active > workers {
        return vec![1; costs.len()];
    }
    let ways = |m: f64, c: f64| ((c / m).ceil() as usize).clamp(1, MAX_WAYS);
    let need = |m: f64| -> usize { costs.iter().filter(|&&c| c > 0.0).map(|&c| ways(m, c)).sum() };
    // need(M) is non-increasing in M and need(largest) = active <= workers.
    let (mut lo, mut hi) = ((total / workers as f64).max(largest / MAX_WAYS as f64), largest);
    if need(lo) <= workers {
        hi = lo;
    } else {
        for _ in 0..50 {
            let mid = 0.5 * (lo + hi);
            if need(mid) <= workers {
                hi = mid;
            } else {
                lo = mid;
            }
        }
    }
    costs.iter().map(|&c| if c > 0.0 { ways(hi, c) } else { 1 }).collect()
}

/// Run `f` with this thread's residue products split into `ways` on the
/// ambient pool. Callers grant extra ways only to their heaviest blocks
/// ([`makespan_ways`]), so light blocks stay serial.
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

/// Granted ways are capped so that each prime group carries at least
/// `WAY_WORK` residue multiply-adds (`work` counts them over all primes):
/// splitting a small product only adds CRT accumulators to merge.
/// [`split_plan`] without granted ways: products whose split shares one CRT
/// state over prime groups (GEMM, svec quadratic) ran slower split than
/// serial inside a busy pool (EPYC: a 53-row quadratic 3.5 ms serial, 4.6-5.6
/// ms at 2-4 ways), so only a top-level pool splits them.
fn unhinted_plan(pool: Option<&rayon::ThreadPool>) -> Split<'_> {
    match pool.filter(|p| p.current_num_threads() > 1 && rayon::current_thread_index().is_none()) {
        Some(p) => Split::Pool(p),
        None => Split::Serial,
    }
}

fn split_plan(pool: Option<&rayon::ThreadPool>, work: u128) -> Split<'_> {
    if let Some(p) =
        pool.filter(|p| p.current_num_threads() > 1 && rayon::current_thread_index().is_none())
    {
        return Split::Pool(p);
    }
    let ways = SPLIT_HINT.with(|c| c.get()).min((work / way_work()).min(usize::MAX as u128) as usize);
    if ways > 1 && rayon::current_thread_index().is_some() {
        Split::Ways(ways)
    } else {
        Split::Serial
    }
}

/// Residue multiply-adds one extra way must carry.
const WAY_WORK: u128 = 1 << 18;

fn way_work() -> u128 {
    WAY_WORK
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

/// Residue products go through faer's kernel except with Accelerate (macOS).
/// Every block task issues its own small GEMMs, and concurrent single-thread
/// OpenBLAS 0.3.29 calls collapse: a 53³ `dgemm` ran at 27 GF alone but 3.5 GF
/// per thread with 64 concurrent callers (EPYC 7742), so wide pools and
/// in-process owners stalled where separate MPI processes did not. Products
/// are exact integers, so the kernel never changes the bits.
/// `SDPX_INT_GEMM=blas|faer` overrides the choice.
#[cfg(feature = "faer-sparse")]
fn faer_int_gemm() -> bool {
    static CHOICE: OnceLock<bool> = OnceLock::new();
    *CHOICE.get_or_init(|| match std::env::var("SDPX_INT_GEMM").as_deref() {
        Ok("faer") => true,
        Ok("blas") => false,
        _ => !cfg!(target_os = "macos"),
    })
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
    let a_end = if ta == b'N' {
        (m - 1) + (k.max(1) - 1) * lda
    } else {
        (k.max(1) - 1) + (m - 1) * lda
    };
    let b_end = if tb == b'N' {
        (k.max(1) - 1) + (n - 1) * ldb
    } else {
        (n - 1) + (k.max(1) - 1) * ldb
    };
    assert!(k == 0 || (a_end < a.len() && b_end < b.len()));
    assert!((m - 1) + (n - 1) * ldc < c.len());
    if k == 0 {
        if !accumulate {
            for col in c.chunks_mut(ldc.max(1)).take(n) {
                col[..m].fill(0.0);
            }
        }
        return;
    }
    #[cfg(feature = "faer-sparse")]
    if faer_int_gemm() {
        // SAFETY: the asserts above bound every strided access.
        unsafe {
            let (ars, acs) = if ta == b'N' { (1, lda) } else { (lda, 1) };
            let (brs, bcs) = if tb == b'N' { (1, ldb) } else { (ldb, 1) };
            let a = faer::MatRef::from_raw_parts(a.as_ptr(), m, k, ars as isize, acs as isize);
            let b = faer::MatRef::from_raw_parts(b.as_ptr(), k, n, brs as isize, bcs as isize);
            let c = faer::MatMut::from_raw_parts_mut(c.as_mut_ptr(), m, n, 1, ldc as isize);
            let accum = if accumulate {
                faer::Accum::Add
            } else {
                faer::Accum::Replace
            };
            faer::linalg::matmul::matmul(c, accum, a, b, 1.0, faer::Par::Seq);
        }
        return;
    }
    // Linked BLAS (Accelerate, OpenBLAS, MKL): every per-prime product is an
    // exact integer GEMM, so provider summation order cannot change the bits.
    // SAFETY: the asserts bound every strided access of the three operands.
    unsafe {
        blas::dgemm(
            ta,
            tb,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a,
            lda as i32,
            b,
            ldb as i32,
            if accumulate { 1.0 } else { 0.0 },
            c,
            ldc as i32,
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

#[inline]
fn scan_exponent<const N: usize>(value: &F<N>, lo: &mut i64, hi: &mut i64) -> Option<bool> {
    let view = value.dyadic_view();
    match view.kind {
        DyadicKind::Zero => Some(false),
        DyadicKind::Finite { .. } => {
            let x = view.exponent as i64 - F::<N>::PRECISION_BITS as i64;
            *lo = (*lo).min(x);
            *hi = (*hi).max(x);
            Some(true)
        }
        _ => None,
    }
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
            scan_exponent(v, &mut lo, &mut hi)?;
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
        let bits = prime_bits(k);
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

/// Total streamed operand bytes allowed per way per group. Caps the group
/// width so that very tall operands keep peak RSS near the streamed size
/// instead of multiplying it by `STREAM_GROUP` and the way count.
const GROUP_SCRATCH_BYTES: usize = 8 << 20;

fn stream_group_cap(entries: usize) -> usize {
    (GROUP_SCRATCH_BYTES / (entries.max(1) * 8)).clamp(1, STREAM_GROUP)
}

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

impl ChunkMatrix {
    fn new<const N: usize>(len: usize, spread: i64, plan: &Plan) -> Self {
        let int_bits = F::<N>::PRECISION_BITS as i64 + spread;
        let mut width = 30u32;
        let chunks = loop {
            let j = (int_bits as usize).div_ceil(width as usize);
            if (j as f64).log2() + width as f64 + (plan.bits - 1) as f64 <= EXACT_BITS - 0.5
                || width == 8
            {
                break j;
            }
            width -= 1;
        };
        Self {
            e: take_buffer(len * chunks),
            chunks,
            width,
            len,
        }
    }

    fn view(&self) -> ChunkView<'_> {
        ChunkView {
            e: &self.e,
            len: self.len,
            chunks: self.chunks,
            width: self.width,
        }
    }
    /// Entries `r0..r1`; entry-major storage makes row ranges contiguous.
    fn rows(&self, r0: usize, r1: usize) -> ChunkView<'_> {
        ChunkView {
            e: &self.e[r0 * self.chunks..r1 * self.chunks],
            len: r1 - r0,
            chunks: self.chunks,
            width: self.width,
        }
    }
}

/// Borrowed entry-major chunk view, optionally covering a row range.
#[derive(Clone, Copy)]
struct ChunkView<'a> {
    e: &'a [f64],
    len: usize,
    chunks: usize,
    width: u32,
}

impl Drop for ChunkMatrix {
    fn drop(&mut self) {
        release_buffer(std::mem::take(&mut self.e));
    }
}

/// Rows encoded per way, at least; below this a split costs more than it saves.
const MIN_ROWS_PER_WAY: usize = 256;

#[inline]
fn fill_chunk<const N: usize>(value: &F<N>, row: &mut [f64], lo: i64, width: u32) {
    let view = value.dyadic_view();
    let negative = match view.kind {
        DyadicKind::Finite { negative } => negative,
        _ => return,
    };
    let delta = view.exponent as i64 - F::<N>::PRECISION_BITS as i64 - lo;
    let limbs = value.exact_encode().2;
    let j0 = (delta / width as i64) as usize;
    let j1 = ((delta + F::<N>::PRECISION_BITS as i64) as usize)
        .div_ceil(width as usize)
        .min(row.len());
    for (j, slot) in row.iter_mut().enumerate().take(j1).skip(j0) {
        let bits = bits_at(limbs, width as i64 * j as i64 - delta, width) as f64;
        *slot = if negative { -bits } else { bits };
    }
}

fn chunk_matrix<const N: usize>(
    x: View<'_, N>,
    lo: i64,
    spread: i64,
    plan: &Plan,
    split: &Split<'_>,
) -> ChunkMatrix {
    let len = x.len();
    let mut matrix = ChunkMatrix::new::<N>(len, spread, plan);
    let (chunks, width) = (matrix.chunks, matrix.width);
    // Rows are independent; a split call encodes them over its ways.
    let fill = |r0: usize, rows: &mut [f64]| {
        for (i, row) in rows.chunks_mut(chunks).enumerate() {
            let r = r0 + i;
            if !x.stored(r) {
                continue;
            }
            fill_chunk(&x.data[r], row, lo, width);
        }
    };
    let ways = split_ways(split).min(len / MIN_ROWS_PER_WAY).max(1);
    if ways <= 1 || chunks == 0 {
        fill(0, &mut matrix.e);
    } else {
        let per = len.div_ceil(ways);
        let mut run = || {
            matrix.e.par_chunks_mut(per * chunks)
                .enumerate()
                .for_each(|(w, rows)| fill(w * per, rows))
        };
        match split {
            Split::Pool(p) => p.install(run),
            _ => run(),
        }
    }
    matrix
}

fn weights(plan: &Plan, width: u32, chunks: usize) -> Arc<Vec<f64>> {
    static WEIGHTS: OnceLock<Mutex<HashMap<(u32, usize, u32, usize), Arc<Vec<f64>>>>> =
        OnceLock::new();
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
fn group_residues(m: ChunkView<'_>, plan: &Plan, q0: usize, q1: usize, out: &mut Vec<f64>) {
    let g = q1 - q0;
    // No clearing: the product overwrites every entry (k = 0 is zeroed by
    // `local_gemm`).
    out.resize(m.len * g, 0.0);
    let w = weights(plan, m.width, m.chunks);
    int_gemm(
        b'T',
        b'N',
        m.len,
        g,
        m.chunks,
        m.e,
        m.chunks,
        &w[q0 * m.chunks..q1 * m.chunks],
        m.chunks,
        out,
        m.len,
    );
    for (qi, col) in out.chunks_mut(m.len.max(1)).enumerate().take(g) {
        let q = q0 + qi;
        for x in col {
            *x = reduce(*x, plan.p[q], plan.pinv[q]);
        }
    }
}

/// Residues of an operand that stays constant across many products (the
/// sampled basis for a whole solve, a scaling factor for an iteration),
/// stored as signed 24-bit integers or f32, both exact at their prime width.
/// General products match the exact-bit fingerprint, alignment and prime
/// width, re-encoding changed operands. Constant-A diagonal congruences skip
/// the operand scan; their owner must reset the cache when A changes.
#[derive(Clone, Debug, Default)]
pub struct ResidueCache {
    entry: Arc<Mutex<Option<Arc<CacheEntry>>>>,
}

#[derive(Debug)]
struct CacheEntry {
    fingerprint: u64,
    bits: u32,
    lo: i64,
    spread: i64,
    len: usize,
    count: usize,
    res: CachedResidues,
}

#[derive(Debug)]
enum CachedResidues {
    // Balanced residues for primes below 2^24 lie strictly inside ±2^23.
    Narrow(Vec<[u8; 3]>),
    Wide(Vec<f32>),
}

impl CachedResidues {
    fn from_chunks(chunks: &ChunkMatrix, plan: &Plan) -> Self {
        let len = chunks.len * plan.count();
        let mut stored = if plan.bits <= 24 {
            Self::Narrow(Vec::with_capacity(len))
        } else {
            Self::Wide(Vec::with_capacity(len))
        };
        // Compress each prime group immediately. A full f64 encoding would
        // coexist with the cache and multiply the construction memory peak.
        let mut scratch = Vec::new();
        for first in (0..plan.count()).step_by(STREAM_GROUP) {
            group_residues(
                chunks.view(),
                plan,
                first,
                (first + STREAM_GROUP).min(plan.count()),
                &mut scratch,
            );
            match &mut stored {
                Self::Narrow(values) => values.extend(scratch.iter().map(|&value| {
                    let [a, b, c, _] = (value as i32).to_le_bytes();
                    [a, b, c]
                })),
                Self::Wide(values) => values.extend(scratch.iter().map(|&value| value as f32)),
            }
        }
        stored
    }

    /// Contiguous operands, block by block: each block's chunk matrix and one
    /// prime group are the only transients, and its residues are written in
    /// place, instead of a whole-operand chunk matrix plus per-way parts that
    /// are merged afterwards. Every residue comes from the same exact integer
    /// sum and reduction as [`Self::from_chunks`], so the store is identical.
    fn from_view_blocked<const N: usize>(
        x: View<'_, N>,
        lo: i64,
        spread: i64,
        plan: &Plan,
        split: &Split<'_>,
        reusable: Option<Self>,
    ) -> Self {
        const BLOCK: usize = 1 << 15;
        let (len, count) = (x.len(), plan.count());
        let mut stored = reusable.unwrap_or_else(|| {
            if plan.bits <= 24 {
                Self::Narrow(vec![[0u8; 3]; len * count])
            } else {
                Self::Wide(vec![0f32; len * count])
            }
        });
        let mut scratch = Vec::new();
        for r0 in (0..len).step_by(BLOCK) {
            let r1 = (r0 + BLOCK).min(len);
            let rows = r1 - r0;
            let block = View {
                data: &x.data[r0..r1],
                rows,
                cols: 1,
                ld: rows,
            };
            let chunks = chunk_matrix(block, lo, spread, plan, split);
            for q0 in (0..count).step_by(STREAM_GROUP) {
                let q1 = (q0 + STREAM_GROUP).min(count);
                group_residues(chunks.view(), plan, q0, q1, &mut scratch);
                for (qi, column) in scratch.chunks(rows).enumerate() {
                    let at = (q0 + qi) * len + r0;
                    match &mut stored {
                        Self::Narrow(values) => {
                            for (slot, &value) in values[at..at + rows].iter_mut().zip(column) {
                                let [a, b, c, _] = (value as i32).to_le_bytes();
                                *slot = [a, b, c];
                            }
                        }
                        Self::Wide(values) => {
                            for (slot, &value) in values[at..at + rows].iter_mut().zip(column) {
                                *slot = value as f32;
                            }
                        }
                    }
                }
            }
        }
        stored
    }

    /// Entry-split variant of [`Self::from_chunks`] for streamed callers: each
    /// way encodes every prime for its own contiguous entry range, so the
    /// encode is parallel and per-way scratch shrinks with the way count.
    /// Every stored value is computed by the same `group_residues` GEMM as
    /// the serial path, so residues are identical bit for bit.
    fn from_chunks_split(chunks: &ChunkMatrix, plan: &Plan, split: &Split<'_>) -> Self {
        let len = chunks.len;
        let ways = split_ways(split).min(len / MIN_ROWS_PER_WAY).max(1);
        if ways <= 1 {
            return Self::from_chunks(chunks, plan);
        }
        let count = plan.count();
        let per = len.div_ceil(ways);
        let work = |w: usize| -> CachedResidues {
            let (r0, r1) = (w * per, ((w + 1) * per).min(len));
            let view = chunks.rows(r0, r1);
            let len_w = r1 - r0;
            let mut part = if plan.bits <= 24 {
                Self::Narrow(Vec::with_capacity(len_w * count))
            } else {
                Self::Wide(Vec::with_capacity(len_w * count))
            };
            let mut scratch = Vec::new();
            for q0 in (0..count).step_by(STREAM_GROUP) {
                let q1 = (q0 + STREAM_GROUP).min(count);
                group_residues(view, plan, q0, q1, &mut scratch);
                match &mut part {
                    Self::Narrow(values) => values.extend(scratch.iter().map(|&value| {
                        let [a, b, c, _] = (value as i32).to_le_bytes();
                        [a, b, c]
                    })),
                    Self::Wide(values) => values.extend(scratch.iter().map(|&value| value as f32)),
                }
            }
            part
        };
        let encode = || {
            let parts: Vec<CachedResidues> = (0..ways).into_par_iter().map(work).collect();
            // Merge prime-major part rows into the shared `[q·len + e]` layout.
            if plan.bits <= 24 {
                let mut out = vec![[0u8; 3]; len * count];
                out.par_chunks_mut(len).enumerate().for_each(|(q, col)| {
                    let mut off = 0;
                    for part in &parts {
                        let Self::Narrow(values) = part else {
                            continue;
                        };
                        let lw = values.len() / count;
                        col[off..off + lw].copy_from_slice(&values[q * lw..(q + 1) * lw]);
                        off += lw;
                    }
                });
                Self::Narrow(out)
            } else {
                let mut out = vec![0.0f32; len * count];
                out.par_chunks_mut(len).enumerate().for_each(|(q, col)| {
                    let mut off = 0;
                    for part in &parts {
                        let Self::Wide(values) = part else {
                            continue;
                        };
                        let lw = values.len() / count;
                        col[off..off + lw].copy_from_slice(&values[q * lw..(q + 1) * lw]);
                        off += lw;
                    }
                });
                Self::Wide(out)
            }
        };
        match split {
            Split::Pool(p) => p.install(encode),
            _ => encode(),
        }
    }

    fn extend(&self, range: std::ops::Range<usize>, out: &mut Vec<f64>) {
        match self {
            Self::Narrow(values) => out.extend(values[range].iter().map(|&[a, b, c]| {
                // Sign-extend the stored 24-bit two's-complement integer.
                ((u32::from_le_bytes([a, b, c, 0]) << 8) as i32 >> 8) as f64
            })),
            Self::Wide(values) => out.extend(values[range].iter().map(|&value| value as f64)),
        }
    }
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
enum Operand {
    Chunks(ChunkMatrix),
    /// Transient compressed residues (this call only): the per-way group
    /// refill copies scalars instead of re-running the chunk GEMM, which
    /// keeps tall skinny operands cheap at small `stream_group_cap` widths.
    Packed(CachedResidues, usize),
    Cached(Arc<CacheEntry>),
}

/// Operands streaming at least this many residues go through the packed
/// store; below it the per-group chunk GEMM is already cheap.
const PACK_OPERAND_MIN: usize = 1 << 20;

fn packed_operand<const N: usize>(
    x: View<'_, N>,
    lo: i64,
    spread: i64,
    plan: &Plan,
    split: &Split<'_>,
) -> Operand {
    if plan.bits > CACHE_MAX_BITS || x.len().saturating_mul(plan.count()) < PACK_OPERAND_MIN {
        return Operand::Chunks(chunk_matrix(x, lo, spread, plan, split));
    }
    let chunks = chunk_matrix(x, lo, spread, plan, split);
    Operand::Packed(
        CachedResidues::from_chunks_split(&chunks, plan, split),
        chunks.len,
    )
}

impl Operand {
    fn residues(&self, plan: &Plan, q0: usize, q1: usize, out: &mut Vec<f64>) {
        match self {
            Operand::Chunks(m) => group_residues(m.view(), plan, q0, q1, out),
            Operand::Packed(res, len) => {
                out.clear();
                res.extend(q0 * len..q1 * len, out);
            }
            Operand::Cached(c) => {
                out.clear();
                c.res.extend(q0 * c.len..q1 * c.len, out);
            }
        }
    }
}

impl ResidueCache {
    /// Products with these inner dimensions use the same prime sequence, so
    /// one operand cache can serve both without alternating prime widths.
    pub(crate) fn same_prime_width(first: usize, second: usize) -> bool {
        prime_bits(first) == prime_bits(second)
    }

    /// The operand's residues for `plan` (`lo`/`spread` are its alignment),
    /// from the cache when it matches, otherwise freshly encoded and stored.
    fn operand<const N: usize>(
        &mut self,
        x: View<'_, N>,
        lo: i64,
        spread: i64,
        plan: &Plan,
        split: &Split<'_>,
    ) -> Operand {
        if plan.bits > CACHE_MAX_BITS {
            return Operand::Chunks(chunk_matrix(x, lo, spread, plan, split));
        }
        let fp = fingerprint(x);
        let stale = {
            let mut entry = self.entry.lock().unwrap();
            if let Some(c) = entry.as_ref() {
                if c.fingerprint == fp
                    && c.bits == plan.bits
                    && c.lo == lo
                    && c.spread == spread
                    && c.len == x.len()
                    && c.count >= plan.count()
                {
                    return Operand::Cached(c.clone());
                }
            }
            entry.take()
        };
        let count = (plan.count() + CACHE_SPARE_PRIMES).min(primes(plan.bits).len());
        // Reuse only a sole-owned, equal-sized payload: the contiguous encoder
        // overwrites every residue. Release other invalid storage before encoding,
        // outside the lock; active products retain their own Arc.
        let reusable = stale
            .and_then(|entry| Arc::try_unwrap(entry).ok())
            .map(|entry| entry.res)
            .filter(|res| {
                if x.ld != x.rows && x.cols != 1 {
                    return false;
                }
                match res {
                    CachedResidues::Narrow(values) => {
                        plan.bits <= 24 && values.len() == x.len() * count
                    }
                    CachedResidues::Wide(values) => {
                        plan.bits > 24 && values.len() == x.len() * count
                    }
                }
            });
        {
            let wide = Plan::with_count(plan.bits, count);
            let res = if x.ld == x.rows || x.cols == 1 {
                CachedResidues::from_view_blocked(x, lo, spread, &wide, split, reusable)
            } else {
                CachedResidues::from_chunks_split(
                    &chunk_matrix(x, lo, spread, &wide, split),
                    &wide,
                    split,
                )
            };
            let entry = Arc::new(CacheEntry {
                fingerprint: fp,
                bits: plan.bits,
                lo,
                spread,
                len: x.len(),
                count,
                res,
            });
            *self.entry.lock().unwrap() = Some(entry.clone());
            Operand::Cached(entry)
        }
    }
}

/// Streamed CRT: `X = Σ_q r'_q·(M/p_q) − round(Σ_q r'_q/p_q)·M` is linear in
/// the primes, so each group's residues update `Y` (digit sums) and `frac`
/// and are then discarded. Every `Y` entry is an integer below 2^53 at every
/// step (the plan bounds the full sum), so the result equals [`decode`]'s.
struct CrtAccumulator<'a> {
    crt: Arc<Crt>,
    /// Empty for contiguous outputs; otherwise explicit source/destination indices.
    selected: &'a [usize],
    y: Vec<f64>,
    frac: Vec<f64>,
}

impl<'a> CrtAccumulator<'a> {
    fn new(plan: &Plan, n: usize, selected: &'a [usize]) -> Self {
        static CRT: OnceLock<Mutex<HashMap<(u32, usize), Arc<Crt>>>> = OnceLock::new();
        let crt = cached(&CRT, (plan.bits, plan.count()), || crt(plan));
        let y = take_buffer(n * crt.chunks);
        Self {
            crt,
            selected,
            y,
            frac: take_buffer(n),
        }
    }

    /// Fold primes `q0..q1`, whose reduced residues are `prod[(q-q0)·outputs + o]`.
    fn add(
        &mut self,
        plan: &Plan,
        q0: usize,
        q1: usize,
        prod: &mut [f64],
        outputs: usize,
        split: &Split<'_>,
    ) {
        let (n, g, rows) = (self.frac.len(), q1 - q0, plan.count() + 1);
        // Sorted selections put each packed destination at or before its source;
        // the caller discards these residues after the CRT update.
        for q in q0..q1 {
            let (p, pinv, u) = (plan.p[q], plan.pinv[q], self.crt.u[q]);
            for (i, f) in self.frac.iter_mut().enumerate() {
                let o = if self.selected.is_empty() {
                    i
                } else {
                    self.selected[i]
                };
                // |r|·|u| < 2^(2bits-2) <= 2^52: exact in binary64.
                let v = residue_product(prod[(q - q0) * outputs + o], u, p, pinv);
                prod[(q - q0) * n + i] = v;
                *f += v * pinv;
            }
        }
        let digits = self.crt.chunks.div_ceil(split_ways(split));
        let table = &self.crt.table;
        // Digit columns are contiguous, disjoint pieces of the one accumulator.
        // Each update is an integer below 2^53, independent of GEMM summation order.
        let update = |first: usize, cols: usize, y: &mut [f64]| {
            int_gemm_acc(
                b'N',
                b'N',
                n,
                cols,
                g,
                &prod[..n * g],
                n,
                &table[q0 + first * rows..],
                rows,
                y,
                n,
            );
        };
        match split {
            Split::Serial => update(0, self.crt.chunks, &mut self.y),
            Split::Pool(p) => p.install(|| {
                self.y
                    .par_chunks_mut(n * digits)
                    .enumerate()
                    .for_each(|(i, y)| update(i * digits, y.len() / n, y))
            }),
            Split::Ways(_) => self
                .y
                .par_chunks_mut(n * digits)
                .enumerate()
                .for_each(|(i, y)| update(i * digits, y.len() / n, y)),
        }
    }

    /// Subtract `round(frac)·M`, then pack and round each output once;
    /// outputs are independent, so a split call shares them over its ways.
    fn finish<const N: usize>(
        mut self,
        plan: &Plan,
        scale: i64,
        out: &mut [F<N>],
        split: &Split<'_>,
    ) -> bool {
        let (n, rows, kp) = (self.frac.len(), plan.count() + 1, plan.count());
        let ways = split_ways(split).min(n.div_ceil(64)).max(1);
        let failed = std::sync::atomic::AtomicBool::new(false);
        let yptr = SendPtr(self.y.as_mut_ptr());
        let optr = SendValues(out.as_mut_ptr());
        let (crt, frac, selected) = (&self.crt, &self.frac, &self.selected);
        let range = |w: usize| {
            let mut mag = Vec::new();
            for i in w * n / ways..(w + 1) * n / ways {
                let o = if selected.is_empty() { i } else { selected[i] };
                // SAFETY: output i owns y[i + l·n] for every l and out[o].
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
                unsafe { *optr.get().add(o) = value };
            }
        };
        run_ways(split, ways, range);
        !failed.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Add each packed upper output's exact integer, times `2^shift`, to the
    /// `width` slots of its position among the global packed outputs; the
    /// local columns `columns` index the global ones.
    fn accumulate(
        self,
        plan: &Plan,
        shift: usize,
        columns: &[usize],
        slots: &mut [i128],
        width: usize,
    ) {
        let (n, rows, kp) = (self.frac.len(), plan.count() + 1, plan.count());
        let (crt, y) = (&self.crt, &self.y);
        let mut i = 0;
        for (j, &column) in columns.iter().enumerate() {
            for &row in &columns[..=j] {
                let o = column * (column + 1) / 2 + row;
                let dest = &mut slots[o * width..(o + 1) * width];
                // Digit sums minus round(frac)·M are exact integers below 2^53.
                let kk = self.frac[i].round();
                for l in 0..crt.chunks {
                    let d = y[i + l * n] - kk * crt.table[kp + l * rows];
                    if d != 0.0 {
                        add_shifted(dest, d as i64, shift + l * crt.width as usize);
                    }
                }
                i += 1;
            }
        }
    }
}

impl Drop for CrtAccumulator<'_> {
    fn drop(&mut self) {
        for v in [&mut self.y, &mut self.frac] {
            release_buffer(std::mem::take(v));
        }
    }
}

/// Scratch of the shared streamed kernels (pooled buffers).
struct Scratch {
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    prod: Vec<f64>,
    t: Vec<f64>,
}

impl Default for Scratch {
    fn default() -> Self {
        Self {
            a: take_buffer(0),
            b: take_buffer(0),
            c: take_buffer(0),
            prod: take_buffer(0),
            t: take_buffer(0),
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        for v in [
            &mut self.a,
            &mut self.b,
            &mut self.c,
            &mut self.prod,
            &mut self.t,
        ] {
            release_buffer(std::mem::take(v));
        }
    }
}

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

/// Stream one CRT state; independent prime products and digit columns split
/// over the granted workers.
fn stream_products<'a>(
    plan: &Plan,
    split: &Split<'_>,
    n: usize,
    selected: &'a [usize],
    group_cap: usize,
    group: impl Fn(&mut CrtAccumulator<'a>, usize, usize, &mut Scratch, &Split<'_>) + Sync + Send,
) -> CrtAccumulator<'a> {
    let mut acc = CrtAccumulator::new(plan, n, selected);
    let mut scratch = Scratch::default();
    for q0 in (0..plan.count()).step_by(group_cap) {
        group(
            &mut acc,
            q0,
            (q0 + group_cap).min(plan.count()),
            &mut scratch,
            split,
        );
    }
    acc
}

/// Prime products own disjoint output slices and short pooled scratch.
fn prime_products(
    split: &Split<'_>,
    outputs: usize,
    prod: &mut [f64],
    t: &mut Vec<f64>,
    product: impl Fn(usize, &mut [f64], &mut Vec<f64>) + Sync + Send,
) {
    if split_ways(split) == 1 {
        for (qi, prod) in prod.chunks_mut(outputs).enumerate() {
            product(qi, prod, t);
        }
        return;
    }
    let primes = (prod.len() / outputs).div_ceil(split_ways(split));
    let mut run = || {
        prod.par_chunks_mut(primes * outputs)
            .enumerate()
            .for_each(|(i, chunk)| {
                let mut t = take_buffer(0);
                for (j, prod) in chunk.chunks_mut(outputs).enumerate() {
                    product(i * primes + j, prod, &mut t);
                }
                release_buffer(t);
            });
    };
    match split {
        Split::Pool(p) => p.install(run),
        _ => run(),
    }
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

impl CacheBytes for Crt {
    fn cache_bytes(&self) -> usize {
        (self.u.capacity() + self.table.capacity()) * 8
    }
}

fn crt(plan: &Plan) -> Crt {
    let kp = plan.count();
    let mut modulus = vec![1u64];
    for &p in &plan.primes {
        mul_word(&mut modulus, p);
    }
    // (K+1)·2^(bits-1)·2^width < 2^53.
    let width =
        (EXACT_BITS - 0.5 - (plan.bits - 1) as f64 - ((kp + 1) as f64).log2()).floor() as u32;
    let total_bits = modulus.len() * 64;
    let chunks = total_bits.div_ceil(width as usize) + 1;
    let rows = kp + 1;
    let mut table = vec![0.0f64; rows * chunks];
    let mut u = Vec::with_capacity(kp);
    for (q, &p) in plan.primes.iter().enumerate() {
        let cofactor = div_word(&modulus, p);
        u.push(symmetric(inverse(rem_word(&cofactor, p), p), p));
        for (l, column) in table.chunks_mut(rows).enumerate() {
            let start = (l * width as usize) as i64;
            column[q] = bits_at(&cofactor, start, width) as f64;
        }
    }
    for (l, column) in table.chunks_mut(rows).enumerate() {
        let start = (l * width as usize) as i64;
        column[kp] = bits_at(&modulus, start, width) as f64;
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
fn pack(
    digits: impl Iterator<Item = f64>,
    width: u32,
    chunks: usize,
    mag: &mut Vec<u64>,
) -> Option<bool> {
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
/// left untouched); a square output of length n*(n+1)/2 is packed by column.
/// Returns `false` when the product does not admit an exact
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
    let packed_output = upper_only && m == n && out.len() == n * (n + 1) / 2;
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
    let same_operand =
        a.as_ptr() == b.as_ptr() && av.rows == bv.rows && av.cols == bv.cols && av.ld == bv.ld;
    let range_a = av.exponent_range();
    let range_b = if same_operand {
        range_a
    } else {
        bv.exponent_range()
    };
    let (Some((lo_a, hi_a)), Some((lo_b, hi_b))) = (range_a, range_b) else {
        return false;
    };
    if lo_a == i64::MAX || lo_b == i64::MAX {
        // One operand is zero: the exact product is zero.
        for j in 0..n {
            let start = if packed_output {
                j * (j + 1) / 2
            } else {
                j * m
            };
            for i in 0..if upper_only { (j + 1).min(m) } else { m } {
                out[start + i] = F::zero();
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
    let selected = if upper_only && !packed_output {
        selection(m, n, upper_only)
    } else {
        Vec::new()
    };
    let outputs = if packed_output {
        out.len()
    } else if upper_only {
        selected.len()
    } else {
        m * n
    };
    let shape = GemmShape {
        ta,
        tb,
        m,
        n,
        k,
        lda,
        ldb,
    };
    let split = unhinted_plan(pool);
    let ca = packed_operand(av, lo_a, da, &plan, &split);
    let cb = (!same_operand).then(|| match cache_b {
        Some(cache) => cache.operand(bv, lo_b, db, &plan, &split),
        None => packed_operand(bv, lo_b, db, &plan, &split),
    });
    // A small group cap bounds per-way residue scratch only when both operand
    // fills are scalar copies; a chunk-matrix source needs fat GEMM groups.
    let scalar_fill =
        |o: &Option<Operand>| matches!(o, None | Some(Operand::Packed(..) | Operand::Cached(..)));
    let group_cap = if scalar_fill(&cb) && matches!(ca, Operand::Packed(..) | Operand::Cached(..)) {
        stream_group_cap(len_a + len_b)
    } else {
        STREAM_GROUP
    };
    let group = |acc: &mut CrtAccumulator, q0, q1, s: &mut Scratch, split: &Split<'_>| {
        ca.residues(&plan, q0, q1, &mut s.a);
        if let Some(cb) = &cb {
            cb.residues(&plan, q0, q1, &mut s.b);
        }
        s.prod.clear();
        s.prod.resize((q1 - q0) * outputs, 0.0);
        let product = |qi, prod: &mut [f64], t: &mut Vec<f64>| {
            shape.prime_product(
                &plan,
                q0 + qi,
                &s.a[qi * len_a..(qi + 1) * len_a],
                if same_operand {
                    &s.a[qi * len_a..(qi + 1) * len_a]
                } else {
                    &s.b[qi * len_b..(qi + 1) * len_b]
                },
                prod,
                t,
                upper_only,
            );
        };
        prime_products(split, outputs, &mut s.prod, &mut s.t, product);
        acc.add(&plan, q0, q1, &mut s.prod, outputs, split);
    };
    let timer = (split_ways(&split) > 1)
        .then(crate::receipt::start)
        .flatten();
    let mut acc = stream_products(&plan, &split, outputs, &[], group_cap, group);
    crate::receipt::finish("rns.gemm.shared_crt", timer);
    if upper_only && !packed_output {
        acc.selected = &selected;
    }
    acc.finish(&plan, lo_a + lo_b, out, &split)
}

/// Upper triangle of `Σ_b Z_bᵀ·Y_b` with `Z_b = diag(d_b)·Y_b` rounded
/// entrywise, summed exactly across every block and rounded once into
/// `out[i + j*m]`. Block `b` is `(y, d, columns)`: `y` is column-major `w × q`
/// over its sorted global columns (`q = columns.len()`, any width `w`) and
/// `d` holds the `w` row scales; `Z` is formed while encoding, never stored.
///
/// Blocks sharing a column list form a class; within a class, local rows
/// with equal supports share one product. Rows of one support may be summed
/// in any order, so they are sorted by exponent window and cut into groups
/// of at most 256 rows. Each group encodes against its own window and
/// reconstructs its exact integer with its own primes: a few rows carrying
/// cancellation residues no longer widen every group's encoding and prime
/// count. Group integers are added exactly at their offsets from the lowest
/// group base, then rounded once.
pub(super) fn gemm_blocks_upper<const N: usize>(
    m: usize,
    blocks: &[(&[F<N>], &[F<N>], &[usize])],
    out: &mut [F<N>],
    pool: Option<&rayon::ThreadPool>,
) -> bool {
    let width = |block: &(&[F<N>], &[F<N>], &[usize])| block.1.len();
    let mut first_row = Vec::with_capacity(blocks.len() + 1);
    first_row.push(0);
    for block in blocks {
        debug_assert_eq!(block.0.len(), width(block) * block.2.len());
        first_row.push(first_row.last().unwrap() + width(block));
    }
    // Blocks sharing a column list form one class.
    let mut classes: Vec<(&[usize], Vec<usize>)> = Vec::new();
    {
        let mut index: HashMap<&[usize], usize> = HashMap::new();
        for (b, block) in blocks.iter().enumerate() {
            let class = *index.entry(block.2).or_insert_with(|| {
                classes.push((block.2, Vec::new()));
                classes.len() - 1
            });
            classes[class].1.push(b);
        }
    }
    // Per-row windows `[lo_a, hi_a, lo_b, hi_b]` and per-class supports of
    // each local row (positions within the class columns).
    let (mut lo_a, mut lo_b) = (i64::MAX, i64::MAX);
    let mut windows = vec![[0i64; 4]; *first_row.last().unwrap()];
    let mut supports: Vec<Vec<Vec<bool>>> = Vec::with_capacity(classes.len());
    for (columns, members) in &classes {
        let q = columns.len();
        let mut class_supports: Vec<Vec<bool>> = Vec::new();
        for &bi in members {
            let (y, d, _) = blocks[bi];
            let w = width(&blocks[bi]);
            if class_supports.len() < w {
                class_supports.resize(w, vec![false; q]);
            }
            for local in 0..w {
                let (mut lb, mut hb) = (i64::MAX, i64::MIN);
                for c in 0..q {
                    let Some(active) = scan_exponent(&y[local + c * w], &mut lb, &mut hb) else {
                        return false;
                    };
                    class_supports[local][c] |= active;
                }
                // A rounded product's exponent is e_y + e_d or e_y + e_d - 1.
                let view = d[local].dyadic_view();
                let (la, ha) = match view.kind {
                    DyadicKind::Finite { .. } if lb <= hb => {
                        let e = view.exponent as i64;
                        (lb + e - 1, hb + e)
                    }
                    DyadicKind::Finite { .. } | DyadicKind::Zero => (i64::MAX, i64::MIN),
                    _ => return false,
                };
                (lo_a, lo_b) = (lo_a.min(la), lo_b.min(lb));
                windows[first_row[bi] + local] = [la, ha, lb, hb];
            }
        }
        supports.push(class_supports);
    }
    let outputs = m * (m + 1) / 2;
    if lo_a == i64::MAX || lo_b == i64::MAX {
        for j in 0..m {
            out[j * m..j * m + j + 1].fill(F::zero());
        }
        return true;
    }
    let timer = crate::receipt::start();
    let split = split_plan(pool, u128::MAX);
    struct Group {
        /// Global output columns and their positions in the blocks' columns.
        columns: Vec<usize>,
        positions: Vec<usize>,
        /// `(block, local row)` pairs.
        rows: Vec<(usize, usize)>,
        lo: (i64, i64),
        spread: (i64, i64),
        plan: Plan,
        cost: f64,
    }
    let precision = F::<N>::PRECISION_BITS as f64;
    let mut groups = Vec::new();
    for ((columns, members), class_supports) in classes.iter().zip(supports) {
        // A row's common support includes both operands across the class.
        // Equal supports share one product.
        let mut supports: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
        for (local, support) in class_supports.into_iter().enumerate() {
            let positions: Vec<_> = support.into_iter().enumerate()
                .filter_map(|(c, active)| active.then_some(c)).collect();
            if positions.is_empty() {
                continue;
            }
            if let Some(at) = supports.iter().position(|(p, _)| p == &positions) {
                supports[at].1.push(local);
            } else {
                supports.push((positions, vec![local]));
            }
        }
        for (positions, locals) in supports {
            let global: Vec<usize> = positions.iter().map(|&c| columns[c]).collect();
            // Rows with a zero side contribute nothing.
            let mut ids: Vec<(usize, usize)> = members
                .iter()
                .flat_map(|&b| {
                    let w = width(&blocks[b]);
                    locals.iter().filter(move |&&local| local < w).map(move |&local| (b, local))
                })
                .filter(|&(b, local)| {
                    let w = windows[first_row[b] + local];
                    w[0] <= w[1] && w[2] <= w[3]
                })
                .collect();
            ids.sort_unstable_by_key(|&(b, local)| {
                let w = windows[first_row[b] + local];
                (w[0] + w[2], b, local)
            });
            for chunk in ids.chunks(256) {
                let (mut la, mut ha, mut lb, mut hb, mut top) =
                    (i64::MAX, i64::MIN, i64::MAX, i64::MIN, i64::MIN);
                for &(b, local) in chunk {
                    let w = windows[first_row[b] + local];
                    (la, ha, lb, hb) = (la.min(w[0]), ha.max(w[1]), lb.min(w[2]), hb.max(w[3]));
                    top = top.max(w[1] + w[3]);
                }
                // |Σ a·b| < k·2^(2p + top - la - lb) over the group's rows.
                let k = chunk.len();
                let needed = 2.0 * precision + (top - la - lb) as f64 + (k as f64).log2() + 3.0;
                let Some(plan) = Plan::new(k, needed) else {
                    return false;
                };
                let cols = global.len() as f64;
                let cost = plan.count() as f64
                    * k as f64
                    * cols
                    * ((2.0 * precision + (ha - la + hb - lb) as f64) / 28.0 + cols / 2.0);
                groups.push(Group {
                    columns: global.clone(),
                    positions: positions.clone(),
                    rows: chunk.to_vec(),
                    lo: (la, lb),
                    spread: (ha - la, hb - lb),
                    plan,
                    cost,
                });
            }
        }
    }
    // Heaviest groups first; ways pull the next group, and the exact sums
    // do not depend on which way adds which group.
    groups.sort_by(|x, y| y.cost.total_cmp(&x.cost));
    let base = groups.iter().map(|g| g.lo.0 + g.lo.1).min().unwrap_or(0);
    // Slots hold signed 64-bit-aligned partial sums of each output's integer.
    let top = groups
        .iter()
        .map(|g| (g.lo.0 + g.lo.1 - base) as usize + g.plan.count() * g.plan.bits as usize + 192)
        .max()
        .unwrap_or(0);
    let slots_per = top.div_ceil(64) + 2;
    let ways = split_ways(&split).min(groups.len()).max(1);
    // Group windows make any exponent spread encodable; only the exact
    // accumulator grows with it. Past this, exact dots are cheaper.
    if outputs * slots_per * ways > SLOT_BUDGET {
        crate::receipt::finish("rns.block_gemm", timer);
        return false;
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let build = |_| {
        let mut slots = vec![0i128; outputs * slots_per];
        let (mut ar, mut br, mut product, mut temp) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        loop {
            let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let Some(group) = groups.get(index) else {
                break;
            };
            let (cols, kr, plan) = (group.columns.len(), group.rows.len(), &group.plan);
            let len = kr * cols;
            let mut ca = ChunkMatrix::new::<N>(len, group.spread.0, plan);
            let mut cb = ChunkMatrix::new::<N>(len, group.spread.1, plan);
            for (j, &position) in group.positions.iter().enumerate() {
                for (i, &(b, local)) in group.rows.iter().enumerate() {
                    let (y, d, _) = blocks[b];
                    let at = local + position * width(&blocks[b]);
                    let (a0, b0) = ((j * kr + i) * ca.chunks, (j * kr + i) * cb.chunks);
                    let z = y[at] * d[local];
                    fill_chunk(&z, &mut ca.e[a0..a0 + ca.chunks], group.lo.0, ca.width);
                    fill_chunk(&y[at], &mut cb.e[b0..b0 + cb.chunks], group.lo.1, cb.width);
                }
            }
            let shape = GemmShape {
                ta: b'T', tb: b'N', m: cols, n: cols, k: kr, lda: kr, ldb: kr,
            };
            let n = cols * (cols + 1) / 2;
            let mut acc = CrtAccumulator::new(plan, n, &[]);
            for q0 in (0..plan.count()).step_by(STREAM_GROUP) {
                let q1 = (q0 + STREAM_GROUP).min(plan.count());
                group_residues(ca.view(), plan, q0, q1, &mut ar);
                group_residues(cb.view(), plan, q0, q1, &mut br);
                product.resize((q1 - q0) * n, 0.0);
                for (qi, q) in (q0..q1).enumerate() {
                    shape.prime_product(
                        plan, q,
                        &ar[qi * len..(qi + 1) * len],
                        &br[qi * len..(qi + 1) * len],
                        &mut product[qi * n..(qi + 1) * n], &mut temp, true,
                    );
                }
                acc.add(plan, q0, q1, &mut product, n, &Split::Serial);
            }
            let shift = (group.lo.0 + group.lo.1 - base) as usize;
            acc.accumulate(plan, shift, &group.columns, &mut slots, slots_per);
        }
        slots
    };
    let parts: Vec<Vec<i128>> = match &split {
        Split::Serial => vec![build(0)],
        Split::Pool(p) => p.install(|| (0..ways).into_par_iter().map(build).collect()),
        Split::Ways(_) => (0..ways).into_par_iter().map(build).collect(),
    };
    let mut parts = parts.into_iter();
    let mut total = parts.next().unwrap();
    for part in parts {
        for (x, y) in total.iter_mut().zip(&part) {
            *x += y;
        }
    }
    let mut mag = Vec::with_capacity(slots_per);
    let mut o = 0;
    for j in 0..m {
        for i in 0..=j {
            out[i + j * m] = round_slots(&total[o * slots_per..(o + 1) * slots_per], base, &mut mag);
            o += 1;
        }
    }
    crate::receipt::finish("rns.block_gemm", timer);
    true
}

/// Adds `value · 2^bit` to 64-bit-aligned signed slots.
#[inline]
fn add_shifted(slots: &mut [i128], value: i64, bit: usize) {
    let v = (value as i128) << (bit % 64);
    slots[bit / 64] += (v as u64) as i128;
    slots[bit / 64 + 1] += v >> 64;
}

/// The integer `Σ slots[s]·2^(64s)` times `2^scale`, rounded once.
fn round_slots<const N: usize>(slots: &[i128], scale: i64, mag: &mut Vec<u64>) -> F<N> {
    mag.clear();
    let mut carry = 0i128;
    for &s in slots {
        let t = s + carry;
        mag.push(t as u64);
        carry = t >> 64;
    }
    // The top slots only carry sign extension: `carry` is 0 or -1.
    let negative = carry < 0;
    if negative {
        let mut one = true;
        for limb in mag.iter_mut() {
            *limb = !*limb;
            if one {
                (*limb, one) = limb.overflowing_add(1);
            }
        }
    }
    F::from_scaled_integer(negative, mag, scale)
}

/// Correctly rounded `Aᵀ·diag(d)·A` (`m × m`) for a column-major `k × m`
/// operand `A` (leading dimension `k`), written as `out[i + j·m]`. Each
/// entry is the exact sum `Σ_r a_ri·d_r·a_rj` rounded once. `A` is meant to
/// be constant across calls: its residues come from `cache_a`, so only the
/// `k` scaling values are encoded per call. Reset `cache_a` when `A` changes.
/// `upper_only` as in [`gemm`].
#[allow(clippy::too_many_arguments)]
pub(super) fn diag_congruence<const N: usize>(
    m: usize,
    k: usize,
    a: &[F<N>],
    d: &[F<N>],
    upper_only: bool,
    pool: Option<&rayon::ThreadPool>,
    out: &mut [F<N>],
    cache_a: &mut ResidueCache,
) -> bool {
    let av = View {
        data: a,
        rows: k,
        cols: m,
        ld: k,
    };
    let dv = View {
        data: d,
        rows: k,
        cols: 1,
        ld: k,
    };
    let cached = cache_a.entry.lock().unwrap().clone();
    let range_a = cached.as_ref().map_or_else(
        || av.exponent_range(),
        |entry| Some((entry.lo, entry.lo + entry.spread)),
    );
    let (Some((lo_a, hi_a)), Some((lo_d, hi_d))) = (range_a, dv.exponent_range()) else {
        return false;
    };
    if lo_a == i64::MAX || lo_d == i64::MAX {
        for j in 0..m {
            for i in 0..if upper_only { j + 1 } else { m } {
                out[i + j * m] = F::zero();
            }
        }
        return true;
    }
    let (da, dd) = (hi_a - lo_a, hi_d - lo_d);
    if da > MAX_SPREAD || dd > MAX_SPREAD {
        return false;
    }
    // |C| <= k·2^(2(P+da))·2^(P+dd), with the 4|C| CRT margin.
    let p_bits = F::<N>::PRECISION_BITS as f64;
    let needed = 3.0 * p_bits + (2 * da + dd) as f64 + (k.max(1) as f64).log2() + 3.0;
    let Some(plan) = Plan::new(k, needed) else {
        return false;
    };
    let split = split_plan(pool, (m * m * k) as u128 * plan.count() as u128);
    let entry = match cached.filter(|e| e.bits == plan.bits && e.count >= plan.count()) {
        Some(entry) => entry,
        None => {
            let Operand::Cached(entry) = cache_a.operand(av, lo_a, da, &plan, &split) else {
                return false;
            };
            entry
        }
    };
    let count = plan.count();
    let mut dres = take_buffer(count * k);
    group_residues(
        chunk_matrix(dv, lo_d, dd, &plan, &split).view(),
        &plan,
        0,
        count,
        &mut dres,
    );
    let len = k * m;
    let outputs = if upper_only { m * (m + 1) / 2 } else { m * m };
    // Ways own disjoint row blocks and run every prime over them; partial
    // residues add exactly mod p, so the result is independent of the split.
    // Scratch per way is one row block × m plus the selected residues per prime.
    // Give each way the same number of cache-sized blocks. A plain 4096-row
    // cap can leave one way with twice the work (five blocks on four ways).
    let ways = split_ways(&split).min((k / 256).max(1)).max(1);
    let blocks_per_way = k.div_ceil(ways * 4096);
    let rows_per = k.div_ceil(ways * blocks_per_way);
    let blocks = k.div_ceil(rows_per);
    let parts: Vec<Mutex<Vec<f64>>> = (0..ways).map(|_| Mutex::new(Vec::new())).collect();
    run_ways(&split, ways, |w| {
        let mut part = take_buffer(count * outputs);
        let (mut ab, mut cb) = (take_buffer(rows_per * m), take_buffer(rows_per * m));
        let tile = if upper_only { 32 } else { m };
        let mut u = take_buffer(m * tile.min(m));
        for b in (w..blocks).step_by(ways) {
            let (r0, rows) = (b * rows_per, rows_per.min(k - b * rows_per));
            for q in 0..count {
                let (p, pinv) = (plan.p[q], plan.pinv[q]);
                ab.clear();
                for j in 0..m {
                    let at = q * len + j * k + r0;
                    entry.res.extend(at..at + rows, &mut ab);
                }
                // Balanced residues below p/2 < 2^25: each product is exact.
                let dq = &dres[q * k + r0..][..rows];
                cb.clear();
                for column in ab.chunks(rows) {
                    cb.extend(
                        column
                            .iter()
                            .zip(dq)
                            .map(|(&x, &w)| residue_product(x, w, p, pinv)),
                    );
                }
                // Rectangular tiles cover the upper triangle. Products and
                // modular sums remain exact, independent of tile/way order.
                // The prime width bounds the full k, or (at its minimum)
                // k_chunk is 2^19: every <=4096-row block fits one product.
                debug_assert!(rows <= plan.k_chunk);
                for j0 in (0..m).step_by(tile) {
                    let cols = tile.min(m - j0);
                    let height = if upper_only { j0 + cols } else { m };
                    int_gemm(
                        b'T',
                        b'N',
                        height,
                        cols,
                        rows,
                        &ab,
                        rows,
                        &cb[j0 * rows..],
                        rows,
                        &mut u[..height * cols],
                        height,
                    );
                    for j in 0..cols {
                        let end = if upper_only { j0 + j + 1 } else { height };
                        let column = j0 + j;
                        let offset = if upper_only {
                            column * (column + 1) / 2
                        } else {
                            column * m
                        };
                        for i in 0..end {
                            let x = &mut part[q * outputs + offset + i];
                            *x = reduce(*x + reduce(u[j * height + i], p, pinv), p, pinv);
                        }
                    }
                }
            }
        }
        for v in [ab, cb, u] {
            release_buffer(v);
        }
        *parts[w].lock().unwrap() = part;
    });
    release_buffer(dres);
    let mut parts = parts.into_iter().map(|p| p.into_inner().unwrap());
    let mut total = parts.next().unwrap();
    for part in parts {
        for q in 0..count {
            let (p, pinv) = (plan.p[q], plan.pinv[q]);
            for (x, &y) in total[q * outputs..][..outputs]
                .iter_mut()
                .zip(&part[q * outputs..])
            {
                *x = reduce(*x + y, p, pinv);
            }
        }
        release_buffer(part);
    }
    let selected;
    let mut acc = CrtAccumulator::new(&plan, outputs, &[]);
    for q0 in (0..count).step_by(STREAM_GROUP) {
        let q1 = (q0 + STREAM_GROUP).min(count);
        acc.add(
            &plan,
            q0,
            q1,
            &mut total[q0 * outputs..q1 * outputs],
            outputs,
            &Split::Serial,
        );
    }
    selected = if upper_only {
        selection(m, m, true)
    } else {
        Vec::new()
    };
    if upper_only {
        // CRT state stays in packed order; only final output addresses change.
        acc.selected = &selected;
    }
    let result = acc.finish(&plan, 2 * lo_a + lo_d, out, &split);
    release_buffer(total);
    result
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
    fn prime_product(
        &self,
        plan: &Plan,
        q: usize,
        aq: &[f64],
        bq: &[f64],
        cq: &mut [f64],
        part: &mut Vec<f64>,
        upper_only: bool,
    ) {
        let (ta_n, tb_n) = (
            self.ta.to_ascii_uppercase() == b'N',
            self.tb.to_ascii_uppercase() == b'N',
        );
        let (m, n, k, outputs) = (self.m, self.n, self.k, self.m * self.n);
        if upper_only {
            // Rectangle tiles cover the requested triangle; only selected
            // residues survive into CRT, in column-major packed order.
            const TILE: usize = 32;
            part.clear();
            part.resize(m * TILE.min(n), 0.0);
            for j0 in (0..n).step_by(TILE) {
                let cols = TILE.min(n - j0);
                let height = m.min(j0 + cols);
                for k0 in (0..k).step_by(plan.k_chunk) {
                    let kc = plan.k_chunk.min(k - k0);
                    let a_off = if ta_n { k0 * self.lda } else { k0 };
                    let b_off = if tb_n {
                        k0 + j0 * self.ldb
                    } else {
                        j0 + k0 * self.ldb
                    };
                    int_gemm(
                        self.ta,
                        self.tb,
                        height,
                        cols,
                        kc,
                        &aq[a_off..],
                        self.lda,
                        &bq[b_off..],
                        self.ldb,
                        &mut part[..height * cols],
                        height,
                    );
                    for j in 0..cols {
                        let column = j0 + j;
                        let offset = if column < m {
                            column * (column + 1) / 2
                        } else {
                            m * (m + 1) / 2 + (column - m) * m
                        };
                        for i in 0..m.min(column + 1) {
                            let value = reduce(part[i + j * height], plan.p[q], plan.pinv[q]);
                            let dest = &mut cq[offset + i];
                            *dest = if k0 == 0 {
                                value
                            } else {
                                reduce(*dest + value, plan.p[q], plan.pinv[q])
                            };
                        }
                    }
                }
            }
            return;
        }
        let mut k0 = 0;
        while k0 < k {
            let kc = plan.k_chunk.min(k - k0);
            let a_off = if ta_n { k0 * self.lda } else { k0 };
            let b_off = if tb_n { k0 } else { k0 * self.ldb };
            if k0 == 0 {
                int_gemm(
                    self.ta,
                    self.tb,
                    m,
                    n,
                    kc,
                    &aq[a_off..],
                    self.lda,
                    &bq[b_off..],
                    self.ldb,
                    cq,
                    m,
                );
                for x in cq.iter_mut() {
                    *x = reduce(*x, plan.p[q], plan.pinv[q]);
                }
            } else {
                part.clear();
                part.resize(outputs, 0.0);
                int_gemm(
                    self.ta,
                    self.tb,
                    m,
                    n,
                    kc,
                    &aq[a_off..],
                    self.lda,
                    &bq[b_off..],
                    self.ldb,
                    part,
                    m,
                );
                for (x, y) in cq.iter_mut().zip(part.iter()) {
                    *x = reduce(
                        *x + reduce(*y, plan.p[q], plan.pinv[q]),
                        plan.p[q],
                        plan.pinv[q],
                    );
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
    let (Some((lo_a, hi_a)), Some((lo_x, hi_x))) = (av.exponent_range(), xv.exponent_range())
    else {
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
    let split = split_plan(pool, (m * k * (m + k)) as u128 * plan.count() as u128);
    let ca = match cache_a {
        Some(cache) => cache.operand(av, lo_a, da, &plan, &split),
        None => Operand::Chunks(chunk_matrix(av, lo_a, da, &plan, &split)),
    };
    let cx = chunk_matrix(xv, lo_x, dx, &plan, &split);
    // A granted split shares output columns rather than primes: no CRT
    // accumulator is merged, and one fork covers the whole product.
    let ways = split_ways(&split).min(m / CONGRUENCE_MIN_COLS).max(1);
    if ways > 1 {
        let timer = crate::receipt::start();
        let shape = CongruenceShape {
            ta,
            ta_n,
            m,
            k,
            lda,
            ldx,
            len_a,
            len_x,
            upper_only,
        };
        if plan.count() * outputs * 8 <= ways * GROUP_SCRATCH_BYTES {
            let ok = congruence_prime_groups(shape, &plan, &ca, &cx, &split, ways, 2 * lo_a + lo_x, out);
            crate::receipt::finish("rns.congruence.prime_groups", timer);
            return ok;
        }
        let ok = congruence_columns(
            CongruenceShape {
                ta,
                ta_n,
                m,
                k,
                lda,
                ldx,
                len_a,
                len_x,
                upper_only,
            },
            &plan,
            &ca,
            &cx,
            &split,
            ways,
            2 * lo_a + lo_x,
            out,
        );
        crate::receipt::finish("rns.congruence.columns", timer);
        return ok;
    }
    let selected = selection(m, m, upper_only);
    let timer = (split_ways(&split) > 1)
        .then(crate::receipt::start)
        .flatten();
    let acc = stream_products(
        &plan,
        &split,
        if upper_only { selected.len() } else { outputs },
        &selected,
        STREAM_GROUP,
        |acc, q0, q1, s, split| {
            ca.residues(&plan, q0, q1, &mut s.a);
            group_residues(cx.view(), &plan, q0, q1, &mut s.b);
            s.prod.clear();
            s.prod.resize((q1 - q0) * outputs, 0.0);
            prime_products(split, outputs, &mut s.prod, &mut s.t, |qi, prod, t| {
                prime(
                    q0 + qi,
                    &s.a[qi * len_a..(qi + 1) * len_a],
                    &s.b[qi * len_x..(qi + 1) * len_x],
                    prod,
                    t,
                );
            });
            acc.add(&plan, q0, q1, &mut s.prod, outputs, split);
        },
    );
    crate::receipt::finish("rns.congruence.shared_crt", timer);
    acc.finish(&plan, 2 * lo_a + lo_x, out, &split)
}

/// Fewest output columns one way of a split congruence computes.
const CONGRUENCE_MIN_COLS: usize = 4;

#[derive(Clone, Copy)]
struct CongruenceShape {
    ta: u8,
    ta_n: bool,
    m: usize,
    k: usize,
    lda: usize,
    ldx: usize,
    len_a: usize,
    len_x: usize,
    upper_only: bool,
}

/// `C = op(A)·X·op(A)ᵀ` split in two phases for products whose per-prime
/// outputs fit the ways' scratch: prime groups first (full-size per-prime
/// GEMMs, as in the unsplit product), then output columns for the CRT. Narrow
/// column panels left the 53 x 53 GEMMs of a sampled Ising cone 3x slower
/// summed over 4 ways on EPYC. Each output's residues, CRT digits and rounding
/// are those of the unsplit product, so the result is bitwise identical.
#[allow(clippy::too_many_arguments)]
fn congruence_prime_groups<const N: usize>(
    shape: CongruenceShape,
    plan: &Plan,
    ca: &Operand,
    cx: &ChunkMatrix,
    split: &Split<'_>,
    ways: usize,
    scale: i64,
    out: &mut [F<N>],
) -> bool {
    let CongruenceShape {
        ta,
        ta_n,
        m,
        k,
        lda,
        ldx,
        len_a,
        len_x,
        upper_only,
    } = shape;
    let count = plan.count();
    let outputs = m * m;
    let tb = if ta_n { b'T' } else { b'N' };
    let mut prod = take_buffer(count * outputs);
    let per = count.div_ceil(ways);
    let products = |(w, chunk): (usize, &mut [f64])| {
        let (q0, q1) = (w * per, (w * per + per).min(count));
        let (mut a, mut x, mut t) = (take_buffer(0), take_buffer(0), take_buffer(k * m));
        ca.residues(plan, q0, q1, &mut a);
        group_residues(cx.view(), plan, q0, q1, &mut x);
        for q in q0..q1 {
            let (p, pinv) = (plan.p[q], plan.pinv[q]);
            let aq = &a[(q - q0) * len_a..(q - q0 + 1) * len_a];
            let xq = &x[(q - q0) * len_x..(q - q0 + 1) * len_x];
            int_gemm(b'N', tb, k, m, k, xq, ldx, aq, lda, &mut t, k);
            for v in t.iter_mut() {
                *v = reduce(*v, p, pinv);
            }
            let cq = &mut chunk[(q - q0) * outputs..(q - q0 + 1) * outputs];
            int_gemm(ta, b'N', m, m, k, aq, lda, &t, k, cq, m);
            for v in cq.iter_mut() {
                *v = reduce(*v, p, pinv);
            }
        }
        for v in [a, x, t] {
            release_buffer(v);
        }
    };
    let mut run = || {
        prod.par_chunks_mut(per * outputs)
            .enumerate()
            .for_each(products)
    };
    match split {
        Split::Pool(p) => p.install(run),
        _ => run(),
    }
    // Contiguous column ranges; an upper triangle weights later columns more.
    let weight = |j: usize| if upper_only { j as u128 + 1 } else { m as u128 };
    let total: u128 = (0..m).map(weight).sum();
    let mut bounds = vec![0];
    let mut acc_w = 0u128;
    for j in 0..m {
        acc_w += weight(j);
        if acc_w * ways as u128 >= total * bounds.len() as u128 && bounds.len() < ways {
            bounds.push(j + 1);
        }
    }
    bounds.push(m);
    bounds.dedup();
    let failed = std::sync::atomic::AtomicBool::new(false);
    let optr = SendValues(out.as_mut_ptr());
    let products_ref = &prod;
    let part = |w: usize| {
        let (j0, j1) = (bounds[w], bounds[w + 1]);
        let selected: Vec<usize> = (j0..j1)
            .flat_map(|j| (0..if upper_only { j + 1 } else { m }).map(move |i| i + j * m))
            .collect();
        let n = selected.len();
        let mut acc = CrtAccumulator::new(plan, n, &[]);
        let mut local = take_buffer(STREAM_GROUP * n);
        for q0 in (0..count).step_by(STREAM_GROUP) {
            let q1 = (q0 + STREAM_GROUP).min(count);
            for q in q0..q1 {
                let row = &products_ref[q * outputs..(q + 1) * outputs];
                for (dst, &o) in local[(q - q0) * n..(q - q0 + 1) * n].iter_mut().zip(&selected) {
                    *dst = row[o];
                }
            }
            acc.add(plan, q0, q1, &mut local, n, &Split::Serial);
        }
        release_buffer(local);
        let mut values = vec![F::<N>::zero(); n];
        if !acc.finish(plan, scale, &mut values, &Split::Serial) {
            failed.store(true, std::sync::atomic::Ordering::Relaxed);
            return;
        }
        for (&o, value) in selected.iter().zip(values) {
            // SAFETY: column ranges, hence output positions, are disjoint across ways.
            unsafe { *optr.get().add(o) = value };
        }
    };
    run_ways(split, bounds.len() - 1, part);
    release_buffer(prod);
    !failed.load(std::sync::atomic::Ordering::Relaxed)
}

/// `C = op(A)·X·op(A)ᵀ` with its output columns shared over `ways`. Both
/// operands' residues are formed once for every prime (prime groups split
/// over the ways); each way then forms `T = X·op(A)[cols,:]ᵀ` and
/// `C[:, cols] = op(A)·T` per prime and reconstructs its own outputs. Every
/// output's exact integer, CRT digits and rounding are those of the
/// unsplit product, so the result is bitwise identical.
#[allow(clippy::too_many_arguments)]
fn congruence_columns<const N: usize>(
    shape: CongruenceShape,
    plan: &Plan,
    ca: &Operand,
    cx: &ChunkMatrix,
    split: &Split<'_>,
    ways: usize,
    scale: i64,
    out: &mut [F<N>],
) -> bool {
    let CongruenceShape {
        ta,
        ta_n,
        m,
        k,
        lda,
        ldx,
        len_a,
        len_x,
        upper_only,
    } = shape;
    let count = plan.count();
    let mut ares = take_buffer(count * len_a);
    let mut xres = take_buffer(count * len_x);
    let per = count.div_ceil(ways);
    let mut fill = || {
        ares.par_chunks_mut(per * len_a)
            .zip(xres.par_chunks_mut(per * len_x))
            .enumerate()
            .for_each(|(w, (a, x))| {
                let (q0, q1) = (w * per, (w * per + per).min(count));
                let mut buf = Vec::new();
                ca.residues(plan, q0, q1, &mut buf);
                a.copy_from_slice(&buf[..(q1 - q0) * len_a]);
                group_residues(cx.view(), plan, q0, q1, &mut buf);
                x.copy_from_slice(&buf[..(q1 - q0) * len_x]);
            })
    };
    match split {
        Split::Pool(p) => p.install(fill),
        _ => fill(),
    }
    // Contiguous column ranges; an upper triangle weights later columns more.
    let weight = |j: usize| if upper_only { j as u128 + 1 } else { m as u128 };
    let total: u128 = (0..m).map(weight).sum();
    let mut bounds = vec![0];
    let mut acc_w = 0u128;
    for j in 0..m {
        acc_w += weight(j);
        if acc_w * ways as u128 >= total * bounds.len() as u128 && bounds.len() < ways {
            bounds.push(j + 1);
        }
    }
    bounds.push(m);
    bounds.dedup();
    let parts = bounds.len() - 1;
    let failed = std::sync::atomic::AtomicBool::new(false);
    let optr = SendValues(out.as_mut_ptr());
    let (ar, xr) = (&ares, &xres);
    let tb = if ta_n { b'T' } else { b'N' };
    let part = |w: usize| {
        let (j0, j1) = (bounds[w], bounds[w + 1]);
        let width = j1 - j0;
        let outputs = m * width;
        let selected: Vec<usize> = if upper_only {
            (j0..j1)
                .flat_map(|j| (0..=j).map(move |i| i + (j - j0) * m))
                .collect()
        } else {
            Vec::new()
        };
        let mut acc =
            CrtAccumulator::new(plan, if upper_only { selected.len() } else { outputs }, &selected);
        // Rows j0.. of op(A) (A itself when ta = 'N', else its columns).
        let a_cols = if ta_n { j0 } else { j0 * lda };
        let group = stream_group_cap(outputs);
        let mut prod = take_buffer(group * outputs);
        let mut t = take_buffer(k * width);
        for q0 in (0..count).step_by(group) {
            let q1 = (q0 + group).min(count);
            for q in q0..q1 {
                let (p, pinv) = (plan.p[q], plan.pinv[q]);
                let aq = &ar[q * len_a..(q + 1) * len_a];
                let xq = &xr[q * len_x..(q + 1) * len_x];
                int_gemm(b'N', tb, k, width, k, xq, ldx, &aq[a_cols..], lda, &mut t, k);
                for v in t.iter_mut() {
                    *v = reduce(*v, p, pinv);
                }
                let cq = &mut prod[(q - q0) * outputs..(q - q0 + 1) * outputs];
                int_gemm(ta, b'N', m, width, k, aq, lda, &t, k, cq, m);
                for v in cq.iter_mut() {
                    *v = reduce(*v, p, pinv);
                }
            }
            acc.add(plan, q0, q1, &mut prod, outputs, &Split::Serial);
        }
        release_buffer(prod);
        release_buffer(t);
        let mut local = vec![F::<N>::zero(); outputs];
        if !acc.finish(plan, scale, &mut local, &Split::Serial) {
            failed.store(true, std::sync::atomic::Ordering::Relaxed);
            return;
        }
        for j in j0..j1 {
            for i in 0..if upper_only { j + 1 } else { m } {
                // SAFETY: column ranges are disjoint across ways.
                unsafe { *optr.get().add(i + j * m) = local[i + (j - j0) * m] };
            }
        }
    };
    run_ways(split, parts, part);
    release_buffer(ares);
    release_buffer(xres);
    !failed.load(std::sync::atomic::Ordering::Relaxed)
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
    abs_q: bool,
    x: &[F<N>],
    sqrt2: F<N>,
    pool: Option<&rayon::ThreadPool>,
    out: &mut [F<N>],
    cache_q: Option<&mut ResidueCache>,
) -> bool {
    let trih = h * (h + 1) / 2;
    let qv = View {
        data: q,
        rows: h,
        cols: kmax,
        ld: h,
    };
    let xv = View {
        data: x,
        rows: trih,
        cols: 1,
        ld: trih,
    };
    let sv = View {
        data: std::slice::from_ref(&sqrt2),
        rows: 1,
        cols: 1,
        ld: 1,
    };
    let (Some((lo_q, hi_q)), Some((lo_x, hi_x)), Some((mut sigma, _))) = (
        qv.exponent_range(),
        xv.exponent_range(),
        sv.exponent_range(),
    ) else {
        return false;
    };
    // Sampled RHS quadratics use an exact off-diagonal multiplier of 2.
    // Encode that integer directly instead of carrying P-2 trailing zero
    // bits through every prime product and the CRT reconstruction.
    let integer_two = sqrt2 == F::<N>::from_u32(2).unwrap();
    if integer_two {
        sigma = 0;
    }
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
    // |M| < 2^(P+dx)·2^max(multiplier_bits, -σ).
    // |v| <= h²·|M|·2^(2(P+dq)).
    let multiplier_bits = if integer_two { 2 } else { p_bits };
    let m_bits = (p_bits + dx + multiplier_bits.max(-sigma)) as f64;
    let needed = m_bits + 2.0 * (p_bits + dq) as f64 + 2.0 * (h.max(1) as f64).log2() + 3.0;
    let Some(plan) = Plan::new(h, needed) else {
        return false;
    };
    if plan.k_chunk < h {
        return false;
    }
    let (len_q, len_x) = (qv.len(), xv.len());
    let prime = |qi: usize,
                 qq: &[f64],
                 xq: &[f64],
                 s: f64,
                 vq: &mut [f64],
                 temp: &mut Vec<f64>| {
        let (p, pinv, pu) = (plan.p[qi], plan.pinv[qi], plan.primes[qi]);
        let diag = symmetric(pow2_mod(-sigma as u64, pu), pu);
        temp.resize(h * h + h * kmax, 0.0);
        let (m, t) = temp.split_at_mut(h * h);
        for j in 0..h {
            for i in 0..j {
                m[i + j * h] = residue_product(s, xq[j * (j + 1) / 2 + i], p, pinv);
            }
            m[j + j * h] = residue_product(diag, xq[j * (j + 1) / 2 + j], p, pinv);
        }
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
    let split = unhinted_plan(pool);
    let cq = match cache_q {
        Some(cache) => cache.operand(qv, lo_q, dq, &plan, &split),
        None => Operand::Chunks(chunk_matrix(qv, lo_q, dq, &plan, &split)),
    };
    let cx = chunk_matrix(xv, lo_x, dx, &plan, &split);
    let cs = (!integer_two).then(|| chunk_matrix(sv, sigma, 0, &plan, &split));
    // Balanced residues are odd (odd primes), so negating those of each
    // negative entry gives exactly the residues of |q|; q's cache serves both.
    let negative: Vec<usize> = if abs_q {
        (0..len_q).filter(|&e| q[e] < F::zero()).collect()
    } else {
        Vec::new()
    };
    let timer = (split_ways(&split) > 1)
        .then(crate::receipt::start)
        .flatten();
    let acc = stream_products(&plan, &split, kmax, &[], STREAM_GROUP, |acc, q0, q1, s, split| {
        cq.residues(&plan, q0, q1, &mut s.a);
        for g in 0..q1 - q0 {
            for &e in &negative {
                s.a[g * len_q + e] = -s.a[g * len_q + e];
            }
        }
        group_residues(cx.view(), &plan, q0, q1, &mut s.b);
        if let Some(cs) = &cs {
            group_residues(cs.view(), &plan, q0, q1, &mut s.c);
        } else {
            s.c.clear();
            s.c.resize(q1 - q0, 2.0);
        }
        s.prod.resize((q1 - q0) * kmax, 0.0);
        prime_products(split, kmax, &mut s.prod, &mut s.t, |g, prod, temp| {
            prime(
                q0 + g,
                &s.a[g * len_q..(g + 1) * len_q],
                &s.b[g * len_x..(g + 1) * len_x],
                s.c[g],
                prod,
                temp,
            );
        });
        acc.add(&plan, q0, q1, &mut s.prod, kmax, split);
    });
    crate::receipt::finish("rns.quadratic.shared_crt", timer);
    acc.finish(&plan, scale, out, &split)
}

/// Selected exact q_aᵀ X q_b forms. X is symmetric and packed without svec
/// factors; Q is column-major. Only requested forms are reconstructed in MPFR.
#[allow(clippy::too_many_arguments)]
pub(super) fn symmetric_bilinear<const N: usize>(
    h: usize,
    columns: usize,
    q: &[F<N>],
    x: &[F<N>],
    pairs: &[(usize, usize)],
    out: &mut [F<N>],
    cache_q: &mut ResidueCache,
) -> bool {
    let Some(qlen) = h.checked_mul(columns) else {
        return false;
    };
    let Some(trih) = h
        .checked_add(1)
        .and_then(|v| h.checked_mul(v))
        .map(|v| v / 2)
    else {
        return false;
    };
    if h == 0
        || q.len() < qlen
        || x.len() < trih
        || out.len() < pairs.len()
        || pairs.iter().any(|&(a, b)| a >= columns || b >= columns)
    {
        return false;
    }
    let qv = View {
        data: q,
        rows: h,
        cols: columns,
        ld: h,
    };
    let xv = View {
        data: x,
        rows: trih,
        cols: 1,
        ld: trih,
    };
    let (Some((lo_q, hi_q)), Some((lo_x, hi_x))) = (qv.exponent_range(), xv.exponent_range())
    else {
        return false;
    };
    if lo_q == i64::MAX || lo_x == i64::MAX {
        out[..pairs.len()].iter_mut().for_each(|v| *v = F::zero());
        return true;
    }
    let (dq, dx) = (hi_q - lo_q, hi_x - lo_x);
    if dq > MAX_SPREAD || dx > MAX_SPREAD {
        return false;
    }
    // Each output has at most h² signed products of three integer mantissas.
    let bits = F::<N>::PRECISION_BITS as i64;
    let needed = (3 * bits + 2 * dq + dx) as f64 + 2.0 * (h.max(1) as f64).log2() + 3.0;
    let Some(plan) = Plan::new(h, needed) else {
        return false;
    };
    if plan.k_chunk < h {
        return false;
    }
    let split = split_plan(None, (h * h * columns) as u128 * plan.count() as u128);
    let cq = cache_q.operand(qv, lo_q, dq, &plan, &split);
    let cx = chunk_matrix(xv, lo_x, dx, &plan, &split);
    let (len_q, len_x, count) = (qv.len(), xv.len(), pairs.len());
    if count == 0 {
        return true;
    }
    let timer = (split_ways(&split) > 1)
        .then(crate::receipt::start)
        .flatten();
    let acc = stream_products(&plan, &split, count, &[], STREAM_GROUP, |acc, q0, q1, s, split| {
        cq.residues(&plan, q0, q1, &mut s.a);
        group_residues(cx.view(), &plan, q0, q1, &mut s.b);
        s.prod.clear();
        s.prod.resize((q1 - q0) * count, 0.0);
        prime_products(split, count, &mut s.prod, &mut s.t, |g, prod, temp| {
            let (p, pinv) = (plan.p[q0 + g], plan.pinv[q0 + g]);
            let qq = &s.a[g * len_q..(g + 1) * len_q];
            let xx = &s.b[g * len_x..(g + 1) * len_x];
            temp.resize(h * h + h * columns, 0.0);
            let (m, t) = temp.split_at_mut(h * h);
            for j in 0..h {
                for i in 0..=j {
                    let value = xx[j * (j + 1) / 2 + i];
                    m[i + j * h] = value;
                    m[j + i * h] = value;
                }
            }
            int_gemm(b'N', b'N', h, columns, h, m, h, qq, h, t, h);
            for (k, &(a, b)) in pairs.iter().enumerate() {
                let mut value = 0.0;
                for i in 0..h {
                    value += qq[i + a * h] * reduce(t[i + b * h], p, pinv);
                }
                prod[k] = reduce(value, p, pinv);
            }
        });
        acc.add(&plan, q0, q1, &mut s.prod, count, split);
    });
    crate::receipt::finish("rns.bilinear.shared_crt", timer);
    acc.finish(&plan, lo_x + 2 * lo_q, out, &split)
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

/// Explicit column-major upper indices; full output uses the identity mapping.
fn selection(m: usize, n: usize, upper_only: bool) -> Vec<usize> {
    if upper_only {
        (0..n)
            .flat_map(|j| (0..(j + 1).min(m)).map(move |i| i + j * m))
            .collect()
    } else {
        Vec::new()
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
