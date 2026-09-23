//! Residue number system (RNS) exact dot products.
//!
//! A bulk dot product `sum_l a_l * b_l` over fixed-precision values is computed
//! exactly in residue space: each operand is aligned to a common binary
//! exponent, reduced modulo a set of ~60-bit primes, accumulated in `u128`, and
//! reconstructed via Garner CRT into an exact integer that rounds once at the
//! destination precision. This is the same model SDPB uses for `bigint_syrk`:
//! MPFR still owns decompositions; only bilinear pairing products move to
//! residues.
//!
//! Exactness contract: the reconstructed integer equals the real sum times a
//! power of two with no truncation, so the result is the correctly rounded
//! value of the exact dot. When operands are non-finite or the exponent spread
//! exceeds the plan window the caller must keep the MPFR path — the plan
//! returns `None` rather than approximating.
//!
//! Prime shape `p = 2^60 - c` with `c < 2^32` (pseudo-Mersenne) keeps the
//! reduction of a `u128` accumulator to a few multiply/fold steps.

use crate::{DyadicKind, MpFloat};
use gmp_mpfr_sys::gmp;
use std::mem::MaybeUninit;

/// Prime bit width. Products stay below `2^120`, so `2^8 = 256` of them fit a
/// `u128` accumulator before a reduction is required.
const PRIME_BITS: u32 = 60;
/// Batch of terms accumulated between reductions of the `u128` accumulators.
const REDUCE_EVERY: usize = 64;
/// Largest admitted per-matrix exponent spread `max(e) - min(e)`. A spread this
/// wide means values differing by `2^4096` coexist in one product, which the
/// solver's iterates never exhibit; beyond it the caller keeps the MPFR path.
const MAX_SPREAD: i64 = 4096;
/// Number of `2^(2^j) mod p` tables entries; covers `delta < 2^16`.
const POW2_STEPS: usize = 16;

/// Primality test for `n < 2^64` (deterministic Miller–Rabin bases).
fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for &p in &[2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n == p {
            return true;
        }
        if n % p == 0 {
            return false;
        }
    }
    let d = n - 1;
    let s = d.trailing_zeros();
    let d = d >> s;
    'witness: for &a in &[2u64, 325, 9375, 28178, 450775, 9780504, 1795265022] {
        let a = a % n;
        if a == 0 {
            continue;
        }
        let mut x = powmod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mulmod(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

fn powmod(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut acc = 1u64;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mulmod(acc, base, m);
        }
        base = mulmod(base, base, m);
        exp >>= 1;
    }
    acc
}

fn mulmod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Exponent range `(lo, hi)` of the dyadic window `exponent - PRECISION_BITS`
/// over a slice; `None` on non-finite values. Identically-zero slices return
/// `(i64::MAX, i64::MIN)` so callers can detect them.
pub fn exponent_range<const N: usize>(m: &[MpFloat<N>]) -> Option<(i64, i64)> {
    let mut lo = i64::MAX;
    let mut hi = i64::MIN;
    for v in m {
        let view = v.dyadic_view();
        match view.kind {
            DyadicKind::Zero => continue,
            DyadicKind::Finite { .. } => {
                // Extreme exponents (subnormal-range values) can sit at
                // the MPFR exponent floor; a saturating window check is
                // enough since such values always exceed MAX_SPREAD.
                let e = (view.exponent as i64)
                    .saturating_sub(MpFloat::<N>::PRECISION_BITS as i64);
                lo = lo.min(e);
                hi = hi.max(e);
            }
            _ => return None,
        }
    }
    Some((lo, hi))
}

/// One modulus `p = 2^60 - c`, `c < 2^32`, plus its Montgomery-free helpers.
#[derive(Clone, Copy)]
struct Prime {
    p: u64,
    /// `c = 2^60 - p`.
    c: u64,
    /// `2^64 mod p` for the Horner fold of mantissa limbs.
    r64: u64,
}

impl Prime {
    /// Reduce `t < 2^128` modulo `p = 2^60 - c` by repeated folding.
    #[inline]
    fn reduce(&self, t: u128) -> u64 {
        const MASK: u128 = (1u128 << PRIME_BITS) - 1;
        // t = t1*2^60 + t0, and 2^60 ≡ c (mod p). Keep t1 in u128: inputs up to
        // 2^128 leave t1 as wide as 2^68.
        let mut r = (t >> PRIME_BITS) * self.c as u128 + (t & MASK);
        r = (r >> PRIME_BITS) * self.c as u128 + (r & MASK);
        r = (r >> PRIME_BITS) * self.c as u128 + (r & MASK);
        let mut r = r as u64;
        while r >= self.p {
            r -= self.p;
        }
        r
    }
}

/// Find `count` primes `2^60 - c`, scanning downward, `c` odd, `c < 2^32`.
fn find_primes(count: usize) -> Vec<Prime> {
    let mut out = Vec::with_capacity(count);
    let mut c = 1u64;
    while out.len() < count {
        let p = (1u64 << PRIME_BITS) - c;
        if is_prime(p) {
            out.push(Prime {
                p,
                c,
                r64: ((16u128 * c as u128) % p as u128) as u64, // 2^64 = 16*2^60
            });
        }
        c += 2;
        debug_assert!(c < (1u64 << 32), "prime search exhausted 2^32 window");
    }
    out
}

/// The first 64 primes of shape `2^60 - c` (~3840 exact bits of dynamic
/// range), computed once — Miller–Rabin per prime dominated plan builds.
static PRIMES: std::sync::OnceLock<Vec<Prime>> = std::sync::OnceLock::new();

fn primes() -> &'static [Prime] {
    PRIMES.get_or_init(|| find_primes(64))
}

/// Tables that depend only on the prime count `k`: encode weights, Garner
/// inverses, prefix products and the modulus. Shared between plans through
/// `Arc` so building a per-call plan is just the bound computation and the
/// two exponent shifts — the O(k²) tables are paid once per `k` per process.
struct RnsTables {
    primes: Vec<Prime>,
    /// `pow2[k][j] = 2^(2^j) mod p_k`.
    pow2: Vec<[u64; POW2_STEPS]>,
    /// `inv[j][i] = p_j^{-1} mod p_i` for `j < i`, packed row-major by `i`.
    inv: Vec<Vec<u64>>,
    /// `prefix[i]` = `prod_{j<i} p_j` as little-endian limbs (`prefix[0] = [1]`).
    prefix: Vec<Vec<u64>>,
    /// `M = prod(p_j)` little-endian limbs.
    modulus: Vec<u64>,
    /// `floor(M / 2)` for the symmetric sign test.
    half: Vec<u64>,
}

static TABLES: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<usize, std::sync::Arc<RnsTables>>>,
> = std::sync::OnceLock::new();

fn build_tables(k: usize) -> RnsTables {
    let primes = primes()[..k].to_vec();
    let pow2: Vec<[u64; POW2_STEPS]> = primes
        .iter()
        .map(|pr| {
            let mut t = [0u64; POW2_STEPS];
            t[0] = 2 % pr.p;
            for j in 1..POW2_STEPS {
                t[j] = pr.reduce(t[j - 1] as u128 * t[j - 1] as u128);
            }
            t
        })
        .collect();
    // Garner inverse table: inv[j][i] = p_j^{-1} mod p_i for j < i.
    let inv: Vec<Vec<u64>> = (0..k)
        .map(|i| (0..i).map(|j| mod_inverse(primes[j].p % primes[i].p, primes[i].p)).collect())
        .collect();
    // Prefix products and modulus as little-endian limb vectors.
    let mut prefix = Vec::with_capacity(k);
    let mut modulus = vec![1u64];
    for pr in &primes {
        prefix.push(modulus.clone());
        mul_add_word(&mut modulus, pr.p, 0);
    }
    let mut half = modulus.clone();
    // floor(M/2): M is odd (all primes odd), so (M-1)/2 = shift right once.
    let mut rem = 0u64;
    for limb in half.iter_mut().rev() {
        let cur = (rem << 63) | (*limb >> 1);
        rem = *limb & 1;
        *limb = cur;
    }
    RnsTables {
        primes,
        pow2,
        inv,
        prefix,
        modulus,
        half,
    }
}

fn tables_for(k: usize) -> Option<std::sync::Arc<RnsTables>> {
    if k == 0 || k > primes().len() {
        return None;
    }
    let map = TABLES.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    Some(
        map.lock()
            .unwrap()
            .entry(k)
            .or_insert_with(|| std::sync::Arc::new(build_tables(k)))
            .clone(),
    )
}

/// Alignment and modulus plan for one residue product `A · B`.
///
/// Each operand matrix is encoded as exact integers `I = value / 2^shift`
/// where `shift` is the matrix minimum of `exponent - PRECISION_BITS`. The
/// product integer then carries `2^(shift_a + shift_b)` exactly.
pub struct RnsPlan {
    tables: std::sync::Arc<RnsTables>,
    shift_a: i64,
    shift_b: i64,
}

/// One operand matrix encoded as residues: `res[elem][k]`, `K` contiguous per
/// element. Element order matches the caller's slice order.
pub struct Residues {
    /// `elems * K` residues.
    pub data: Vec<u64>,
    /// Number of primes `K`.
    pub k: usize,
    /// Number of encoded elements.
    pub elems: usize,
}

/// Per-call workspace for [`RnsPlan::dot_residues_into`] /
/// [`RnsPlan::reconstruct_into`]. Allocating per output dominates small dots;
/// the scratch is reused across every output of one matrix product.
pub struct RnsScratch {
    acc: Vec<u128>,
    res: Vec<u64>,
    c: Vec<u64>,
    x: Vec<u64>,
    mag: Vec<u64>,
    /// Pooled GMP/MPFR reconstruction objects: `mpz_import` and
    /// `mpfr::set_z` reuse their limb allocations, so the only per-output
    /// cost is the copy itself. `f_prec` tracks the mpfr precision; a
    /// precision change reallocates `f`.
    z: Option<gmp::mpz_t>,
    f: Option<gmp_mpfr_sys::mpfr::mpfr_t>,
    f_prec: usize,
}

impl Drop for RnsScratch {
    fn drop(&mut self) {
        unsafe {
            if let Some(z) = self.z.as_mut() {
                gmp::mpz_clear(z);
            }
            if let Some(f) = self.f.as_mut() {
                gmp_mpfr_sys::mpfr::clear(f);
            }
        }
    }
}

thread_local! {
    static RNS_SCRATCH: std::cell::RefCell<RnsScratch> = std::cell::RefCell::new(RnsScratch {
        acc: Vec::new(),
        res: Vec::new(),
        c: Vec::new(),
        x: Vec::new(),
        mag: Vec::new(),
        z: None,
        f: None,
        f_prec: 0,
    });
}

impl RnsPlan {
    /// Cheap necessary condition for `profitable`. Call before scanning
    /// operands or locking the shared tables; true still requires the full
    /// exponent-window and amortization checks.
    pub fn worth_planning(terms: usize, outputs: usize) -> bool {
        terms >= 24 && outputs != 0
    }

    /// Build a plan for `sum over `terms` products of `a`-by-`b` elements.
    ///
    /// Returns `None` when any value is non-finite or the combined exponent
    /// spread exceeds [`MAX_SPREAD`]; the caller then keeps the MPFR path.
    /// The prime count follows the strict bound `|S| < terms * 2^(P+Da+P+Db)`.
    pub fn for_pair<const N: usize>(
        a: &[MpFloat<N>],
        b: &[MpFloat<N>],
        terms: usize,
    ) -> Option<Self> {
        let (lo_a, hi_a) = exponent_range(a)?;
        let (lo_b, hi_b) = exponent_range(b)?;
        Self::for_ranges::<N>((lo_a, hi_a), (lo_b, hi_b), terms)
    }

    /// Variant of [`RnsPlan::for_pair`] with precomputed exponent ranges, so a
    /// constant operand is scanned once per solve rather than once per call.
    pub fn for_ranges<const N: usize>(
        a_range: (i64, i64),
        b_range: (i64, i64),
        terms: usize,
    ) -> Option<Self> {
        if terms == 0 {
            return None;
        }
        let (lo_a, hi_a) = a_range;
        let (lo_b, hi_b) = b_range;
        if lo_a == i64::MAX || lo_b == i64::MAX {
            return None; // an operand is identically zero; nothing to gain
        }
        let (da, db) = (hi_a.saturating_sub(lo_a), hi_b.saturating_sub(lo_b));
        if da > MAX_SPREAD || db > MAX_SPREAD {
            return None;
        }
        // |S_int| < terms * 2^(P + Da) * 2^(P + Db); signed CRT needs M > 2|S|.
        let p = MpFloat::<N>::PRECISION_BITS as u64;
        let bound = 2 * p + (da + db) as u64 + (64 - (terms as u64).leading_zeros() as u64) + 2;
        let k = bound.div_ceil(PRIME_BITS as u64) as usize;
        let tables = tables_for(k)?; // wider operand window than the cached prime range

        Some(Self {
            tables,
            shift_a: lo_a,
            shift_b: lo_b,
        })
    }

    /// Number of primes; also the residue stride per element.
    pub fn primes(&self) -> usize {
        self.tables.primes.len()
    }

    /// The `a`-side alignment shift this plan was built with. Constant
    /// operands encoded against `(shift_a, primes)` stay reusable while the
    /// varying side is re-encoded per call.
    pub fn shift_a(&self) -> i64 {
        self.shift_a
    }

    /// Heuristic cost gate: RNS wins only when the per-term dot saving,
    /// summed over every output, outweighs the one-shot plan build plus
    /// the `encode_elems × primes` operand encoding.
    ///
    /// `outputs` is the number of independent dot products; `encode_elems`
    /// counts the operand elements encoded once (a + b for GEMM, a for
    /// SYRK); `limbs` is the operand mantissa width. Measured constants at
    /// 512–768-bit: `mpfr_fma` ≈ 300ns/term including per-call overhead
    /// (release builds range ~110ns bare-MPFR to ~365ns end-to-end;
    /// underestimating it wrongly rejects profitable operator dots),
    /// residue accumulation ≈ 2ns per term per prime, Garner
    /// reconstruction ≈ 1.6µs/output, encode ≈ ~8ns per limb per prime
    /// with the weighted table, cached plan ≈ ~30µs.
    pub fn profitable(
        &self,
        terms: usize,
        outputs: usize,
        encode_elems: usize,
        limbs: usize,
    ) -> bool {
        if !Self::worth_planning(terms, outputs) {
            return false;
        }
        let k = self.tables.primes.len() as f64;
        let mpfr = outputs as f64 * terms as f64 * 300.0;
        let dots = outputs as f64 * (terms as f64 * 2.0 * k + 1600.0);
        let encode = encode_elems as f64 * k * limbs as f64 * 8.0;
        let plan = 30_000.0f64;
        (dots + encode + plan) * 1.25 < mpfr
    }

    /// Encode `m` (slice order) into residues against the `a`- or `b`-shift.
    ///
    /// `side` selects `shift_a`/`shift_b`. Returns `None` if a value is
    /// non-finite (the scan in `for_pair` should already have excluded this).
    ///
    /// Per-element work is a weighted limb sum: `v mod p = sum_l limb_l *
    /// 2^(64l) mod p`, accumulated in `u128` with one reduction per 8-limb
    /// batch — roughly half the cost of the per-limb Horner fold.
    pub fn encode<const N: usize>(&self, m: &[MpFloat<N>], side: EncodeSide) -> Option<Residues> {
        self.encode_with(m, side, &self.tables.primes, &self.tables.pow2)
    }

    /// Encode against the full cached prime table rather than this plan's
    /// first `K`. Plans draw primes as a fixed prefix of that table, so a
    /// constant operand encoded once at the wider stride stays valid for
    /// every later plan: [`RnsPlan::dot_residues_into`] reads only the
    /// first `k` residues of each row. Costs more than a plan-width
    /// encode and is meant for the cached constant side of a product.
    pub fn encode_wide<const N: usize>(&self, m: &[MpFloat<N>], side: EncodeSide) -> Option<Residues> {
        let t = tables_for(primes().len())?;
        self.encode_with(m, side, &t.primes, &t.pow2)
    }

    fn encode_with<const N: usize>(
        &self,
        m: &[MpFloat<N>],
        side: EncodeSide,
        primes: &[Prime],
        pow2: &[[u64; POW2_STEPS]],
    ) -> Option<Residues> {
        let k = primes.len();
        let shift = match side {
            EncodeSide::A => self.shift_a,
            EncodeSide::B => self.shift_b,
        };
        // w[l][k] = 2^(64*l) mod p_k. Small fixed table per call.
        let nl = m.iter().map(|v| v.limbs.len()).max().unwrap_or(0);
        let mut w = vec![vec![0u64; k]; nl.max(1)];
        for (kk, pr) in primes.iter().enumerate() {
            w[0][kk] = 1 % pr.p;
            for l in 1..nl {
                w[l][kk] = pr.reduce(w[l - 1][kk] as u128 * pr.r64 as u128);
            }
        }
        let encode_one = |e: usize, row: &mut [u64]| -> bool {
            let v = &m[e];
            let view = v.dyadic_view();
            let delta = match view.kind {
                DyadicKind::Zero => return true, // residues stay zero
                DyadicKind::Finite { .. } => (view.exponent as i64)
                    .saturating_sub(MpFloat::<N>::PRECISION_BITS as i64)
                    .saturating_sub(shift),
                _ => return false,
            };
            if delta < 0 || delta >= (1i64 << POW2_STEPS) {
                return false;
            }
            for (kk, pr) in primes.iter().enumerate() {
                // Sum limb_l * 2^(64l) mod p in u128, folding every 8 limbs:
                // each term < 2^124, so a batch of 8 stays below 2^127.
                let mut acc = 0u128;
                let mut batch = 0usize;
                for (l, &limb) in v.limbs.iter().enumerate() {
                    acc += limb as u128 * w[l][kk] as u128;
                    batch += 1;
                    if batch == 8 {
                        acc = pr.reduce(acc) as u128;
                        batch = 0;
                    }
                }
                let mut r = pr.reduce(acc);
                if delta != 0 && r != 0 {
                    let mut d = delta as u64;
                    let mut j = 0usize;
                    while d != 0 {
                        if d & 1 == 1 {
                            r = pr.reduce(r as u128 * pow2[kk][j] as u128);
                        }
                        d >>= 1;
                        j += 1;
                    }
                }
                if view.kind.is_negative() && r != 0 {
                    r = pr.p - r;
                }
                row[kk] = r;
            }
            true
        };
        let mut data = vec![0u64; m.len() * k];
        for (e, row) in data.chunks_mut(k).enumerate() {
            if !encode_one(e, row) {
                return None;
            }
        }
        Some(Residues {
            data,
            k,
            elems: m.len(),
        })
    }

    /// Borrow the thread-local scratch (zeroed allocations amortized across
    /// every output of a matrix product).
    pub fn with_scratch<R>(f: impl FnOnce(&mut RnsScratch) -> R) -> R {
        RNS_SCRATCH.with(|s| f(&mut s.borrow_mut()))
    }

    /// One exact dot: accumulate `terms` residue pairs, then reconstruct with
    /// a single rounding. Reuses thread-local scratch (no allocation per
    /// output after the first call on each worker).
    pub fn dot<const N: usize>(
        &self,
        ra: &Residues,
        a0: usize,
        da: usize,
        rb: &Residues,
        b0: usize,
        db: usize,
        terms: usize,
    ) -> MpFloat<N> {
        Self::with_scratch(|s| {
            self.dot_residues_into(ra, a0, da, rb, b0, db, terms, s);
            self.reconstruct_scratch(s)
        })
    }

    /// Exact dot of two residue columns into `scratch.res[..K]`.
    ///
    /// `a0`/`b0` are element indices; `da`/`db` step between consecutive terms
    /// in element units.
    pub fn dot_residues_into(
        &self,
        ra: &Residues,
        mut a0: usize,
        da: usize,
        rb: &Residues,
        mut b0: usize,
        db: usize,
        terms: usize,
        scratch: &mut RnsScratch,
    ) {
        let k = self.tables.primes.len();
        // Rows may be stored at a wider stride than `k` (see
        // [`RnsPlan::encode_wide`]); only the first `k` residues of each
        // row participate since plans draw a fixed prefix of primes.
        debug_assert!(ra.k >= k && rb.k >= k);
        scratch.acc.clear();
        scratch.acc.resize(k, 0u128);
        scratch.res.clear();
        scratch.res.resize(k, 0u64);
        let acc = &mut scratch.acc;
        let mut l = 0usize;
        while l < terms {
            let end = (l + REDUCE_EVERY).min(terms);
            for _ in l..end {
                let rowa = &ra.data[a0 * ra.k..a0 * ra.k + k];
                let rowb = &rb.data[b0 * rb.k..b0 * rb.k + k];
                for j in 0..k {
                    acc[j] += rowa[j] as u128 * rowb[j] as u128;
                }
                a0 += da;
                b0 += db;
            }
            for (j, pr) in self.tables.primes.iter().enumerate() {
                acc[j] = pr.reduce(acc[j]) as u128;
            }
            l = end;
        }
        for j in 0..k {
            scratch.res[j] = acc[j] as u64;
        }
    }

    /// Exact dot of two residue columns, allocating the residue vector.
    pub fn dot_residues(&self, ra: &Residues, a0: usize, da: usize, rb: &Residues, b0: usize, db: usize, terms: usize) -> Vec<u64> {
        let mut out = vec![0u64; self.tables.primes.len()];
        Self::with_scratch(|s| {
            self.dot_residues_into(ra, a0, da, rb, b0, db, terms, s);
            out.copy_from_slice(&s.res);
        });
        out
    }

    /// Reconstruct the rounded value of `sum` from its `K` residues.
    ///
    /// Garner mixed-radix gives `x = S mod M` in `[0, M)`; values above `M/2`
    /// map to negative sums. One `mpz -> mpfr` conversion plus an exact
    /// `2^(shift_a + shift_b)` scale performs the single rounding.
    pub fn reconstruct<const N: usize>(&self, residues: &[u64]) -> MpFloat<N> {
        Self::with_scratch(|s| {
            s.res.clear();
            s.res.extend_from_slice(residues);
            self.reconstruct_scratch(s)
        })
    }

    /// [`RnsPlan::reconstruct`] over the residues already in `scratch.res`.
    pub fn reconstruct_scratch<const N: usize>(&self, scratch: &mut RnsScratch) -> MpFloat<N> {
        let k = self.tables.primes.len();
        debug_assert!(scratch.res.len() >= k);
        // Garner coefficients.
        scratch.c.clear();
        scratch.c.resize(k, 0u64);
        for i in 0..k {
            let mut t = scratch.res[i];
            for j in 0..i {
                let p = self.tables.primes[i];
                // t = (t - c_j) * inv(j, i) mod p_i
                // c_j is canonical modulo p_j, not necessarily modulo p_i.
                // In particular the prime table is descending. Reduce first
                // so the unsigned modular subtraction cannot underflow.
                // The division is confined to the uncommon out-of-range case.
                let mut cj = scratch.c[j];
                if cj >= p.p {
                    cj %= p.p;
                }
                let diff = if t >= cj { t - cj } else { t + p.p - cj };
                t = p.reduce(diff as u128 * self.tables.inv[i][j] as u128);
            }
            scratch.c[i] = t;
        }
        // x = sum_i c_i * prefix_i, kept below M by a conditional subtract per
        // addend: each addend is < M, so x stays in [0, M) throughout.
        scratch.x.clear();
        scratch.x.resize(self.tables.modulus.len() + 1, 0u64);
        let x = &mut scratch.x;
        for (i, &ci) in scratch.c.iter().enumerate() {
            if ci == 0 {
                continue;
            }
            let mut carry = 0u128;
            for (d, &m) in x.iter_mut().zip(self.tables.prefix[i].iter()) {
                let cur = *d as u128 + ci as u128 * m as u128 + carry;
                *d = cur as u64;
                carry = cur >> 64;
            }
            let mut idx = self.tables.prefix[i].len();
            while carry != 0 && idx < x.len() {
                let cur = x[idx] as u128 + carry;
                x[idx] = cur as u64;
                carry = cur >> 64;
                idx += 1;
            }
            if cmp_limbs(&x, &self.tables.modulus) != std::cmp::Ordering::Less {
                sub_limbs(&mut *x, &self.tables.modulus);
            }
        }
        // Sign: S mod M in [0, M); if x > M/2 the sum is negative, |S| = M - x.
        let negative = cmp_limbs(x, &self.tables.half) == std::cmp::Ordering::Greater;
        scratch.mag.clear();
        if negative {
            // |S| = M - x, computed into `mag`.
            scratch.mag.extend_from_slice(&self.tables.modulus);
            scratch.mag.resize(x.len(), 0);
            sub_limbs(&mut scratch.mag, x);
        } else {
            scratch.mag.extend_from_slice(x);
        }
        let mag = &scratch.mag;
        // Import into the pooled mpz, convert to the pooled MPFR, apply the
        // 2^(shift_a+shift_b) scale. Both objects keep their allocations
        // across every output of the product.
        if scratch.z.is_none() {
            let mut z = MaybeUninit::<gmp::mpz_t>::uninit();
            unsafe { gmp::mpz_init(z.as_mut_ptr()) };
            scratch.z = Some(unsafe { z.assume_init() });
        }
        if scratch.f_prec != MpFloat::<N>::PRECISION_BITS {
            if let Some(mut f) = scratch.f.take() {
                unsafe { gmp_mpfr_sys::mpfr::clear(&mut f) };
            }
            let mut f = MaybeUninit::<gmp_mpfr_sys::mpfr::mpfr_t>::uninit();
            unsafe {
                gmp_mpfr_sys::mpfr::init2(f.as_mut_ptr(), MpFloat::<N>::PRECISION_BITS as _)
            };
            scratch.f = Some(unsafe { f.assume_init() });
            scratch.f_prec = MpFloat::<N>::PRECISION_BITS;
        }
        let z = scratch.z.as_mut().unwrap();
        let f = scratch.f.as_mut().unwrap();
        unsafe {
            gmp::mpz_import(
                z,
                mag.len(),
                -1, // least significant word first
                8,
                0,
                0,
                mag.as_ptr().cast(),
            );
            if negative {
                gmp::mpz_neg(z, z);
            }
            gmp_mpfr_sys::mpfr::set_z(f, z, gmp_mpfr_sys::mpfr::rnd_t::RNDN);
            let scale = self.shift_a + self.shift_b;
            if scale >= 0 {
                gmp_mpfr_sys::mpfr::mul_2exp(f, f, scale as u64, gmp_mpfr_sys::mpfr::rnd_t::RNDN);
            } else {
                gmp_mpfr_sys::mpfr::div_2exp(
                    f,
                    f,
                    scale.unsigned_abs(),
                    gmp_mpfr_sys::mpfr::rnd_t::RNDN,
                );
            }
            MpFloat::<N>::from_mpfr_descriptor(f)
        }
    }
}

/// Which operand shift an encode pass uses.
#[derive(Clone, Copy)]
pub enum EncodeSide {
    A,
    B,
}

fn mod_inverse(a: u64, m: u64) -> u64 {
    // Extended Euclid; inputs are coprime primes.
    let (mut t, mut new_t) = (0i128, 1i128);
    let (mut r, mut new_r) = (m as i128, a as i128);
    while new_r != 0 {
        let q = r / new_r;
        (t, new_t) = (new_t, t - q * new_t);
        (r, new_r) = (new_r, r - q * new_r);
    }
    let t = if t < 0 { t + m as i128 } else { t };
    t as u64
}

/// `x += word * m` little-endian limb multiply-add.
fn mul_add_word(x: &mut Vec<u64>, m: u64, add: u64) {
    let mut carry = add as u128;
    for limb in x.iter_mut() {
        let cur = *limb as u128 * m as u128 + carry;
        *limb = cur as u64;
        carry = cur >> 64;
    }
    while carry != 0 {
        x.push(carry as u64);
        carry >>= 64;
    }
}

fn cmp_limbs(a: &[u64], b: &[u64]) -> std::cmp::Ordering {
    let norm = |v: &[u64]| {
        let mut n = v.len();
        while n > 0 && v[n - 1] == 0 {
            n -= 1;
        }
        n
    };
    let (na, nb) = (norm(a), norm(b));
    if na != nb {
        return na.cmp(&nb);
    }
    for i in (0..na).rev() {
        match a[i].cmp(&b[i]) {
            std::cmp::Ordering::Equal => continue,
            o => return o,
        }
    }
    std::cmp::Ordering::Equal
}

/// `x -= m` (limbs), requires `x >= m`.
fn sub_limbs(x: &mut [u64], m: &[u64]) {
    let mut borrow = 0i128;
    for i in 0..x.len() {
        let mv = if i < m.len() { m[i] } else { 0 };
        let cur = x[i] as i128 - mv as i128 - borrow;
        if cur < 0 {
            x[i] = (cur + (1i128 << 64)) as u64;
            borrow = 1;
        } else {
            x[i] = cur as u64;
            borrow = 0;
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn rns_timing_smoke() {
        use std::time::Instant;
        // ~Λ=15 shapes: dot length 44 and 800 at 512-bit.
        for &k in &[44usize, 128, 800] {
            let a: Vec<Bits512> = (0..k)
                .map(|i| Bits512::from_f64(((i * 37) % 97) as f64 * 0.31 - 14.0).unwrap())
                .collect();
            let b: Vec<Bits512> = (0..k)
                .map(|i| Bits512::from_f64(((i * 53) % 89) as f64 * 0.27 + 0.5).unwrap())
                .collect();
            let reps = 50;
            let t0 = Instant::now();
            for _ in 0..reps { std::hint::black_box(dot_mpfr(&a, &b)); }
            let mpfr = t0.elapsed().as_nanos() as f64 / reps as f64;
            let t0 = Instant::now();
            let mut plan_ns = 0u128;
            let mut enc_ns = 0u128;
            for _ in 0..reps {
                let t1 = Instant::now();
                let plan = RnsPlan::for_pair(&a, &b, k).unwrap();
                plan_ns += t1.elapsed().as_nanos();
                let t1 = Instant::now();
                let ra = plan.encode(&a, EncodeSide::A).unwrap();
                let rb = plan.encode(&b, EncodeSide::B).unwrap();
                enc_ns += t1.elapsed().as_nanos();
                let res = plan.dot_residues(&ra, 0, 1, &rb, 0, 1, k);
                std::hint::black_box(plan.reconstruct::<8>(&res));
            }
            let rns = t0.elapsed().as_nanos() as f64 / reps as f64;
            eprintln!("k={k} primes={} mpfr={mpfr:.0}ns rns={rns:.0}ns ratio={:.2} plan={:.0}ns enc={:.0}ns",
                RnsPlan::for_pair(&a, &b, k).unwrap().primes(), mpfr / rns,
                plan_ns as f64 / reps as f64, enc_ns as f64 / reps as f64);
        }
    }

    use super::*;
    use crate::integer::exact_product;
    use crate::{Bits256, Bits512, Scalar};
    use num_traits::{FromPrimitive, One, ToPrimitive, Zero};

    fn dot_mpfr<const N: usize>(a: &[MpFloat<N>], b: &[MpFloat<N>]) -> MpFloat<N> {
        MpFloat::dot_fma(a.iter().zip(b.iter()))
    }

    fn dot_rns<const N: usize>(a: &[MpFloat<N>], b: &[MpFloat<N>]) -> Option<MpFloat<N>> {
        let plan = RnsPlan::for_pair(a, b, a.len())?;
        let ra = plan.encode(a, EncodeSide::A)?;
        let rb = plan.encode(b, EncodeSide::B)?;
        let res = plan.dot_residues(&ra, 0, 1, &rb, 0, 1, a.len());
        Some(plan.reconstruct(&res))
    }


    #[test]
    fn rns_debug_single_product() {
        use num_traits::FromPrimitive;
        let a = vec![Bits512::from_f64(1.5).unwrap()];
        let b = vec![Bits512::from_f64(-3.5).unwrap()];
        let plan = RnsPlan::for_pair(&a, &b, 1).unwrap();
        eprintln!("primes={} shift_a={} shift_b={}", plan.primes(), plan.shift_a, plan.shift_b);
        let ra = plan.encode(&a, EncodeSide::A).unwrap();
        let rb = plan.encode(&b, EncodeSide::B).unwrap();
        eprintln!("ra={:?}", &ra.data[..8.min(ra.k)]);
        eprintln!("rb={:?}", &rb.data[..8.min(rb.k)]);
        let res = plan.dot_residues(&ra, 0, 1, &rb, 0, 1, 1);
        eprintln!("res={:?}", &res[..8.min(res.len())]);
        let got: Bits512 = plan.reconstruct(&res);
        let want = exact_product(&a, &b).unwrap().to_mpfloat::<8>();
        eprintln!("got={got} want={want}");
        assert_eq!(got, want);
    }

    #[test]
    fn rns_dot_matches_exact_integer_reference() {
        // Mixed magnitudes and signs; compare against the GMP exact accumulator.
        let a: Vec<Bits512> = [
            1.5f64, -2.25, 1e30, -4e-18, 0.0, 7.75, -1e-40, 3.0,
            1.0000000000000002, -0.5, 6.25, -9.5,
        ]
        .iter()
        .map(|&x| Bits512::from_f64(x).unwrap())
        .collect();
        let b: Vec<Bits512> = [
            -3.5f64, 0.75, -2e28, 5e-17, 9.0, -0.125, 1e38, 4.0,
            2.5, 1.75, -8.0, 0.0625,
        ]
        .iter()
        .map(|&x| Bits512::from_f64(x).unwrap())
        .collect();
        let exact = exact_product(&a, &b).unwrap().to_mpfloat::<8>();
        let rns = dot_rns(&a, &b).expect("plan");
        assert_eq!(rns, exact, "RNS must equal the exact integer sum");
        // The exact sum also equals dot_fma here only up to last-ulp; require
        // relative agreement within a few ulps.
        let mpfr = dot_mpfr(&a, &b);
        let num = (rns - mpfr).to_f64().unwrap_or(f64::NAN);
        let den = mpfr.to_f64().unwrap_or(f64::NAN).abs().max(1e-300);
        assert!(num.abs() / den < 1e-14, "rns {num} vs mpfr drift");
    }

    #[test]
    fn rns_dot_handles_cancellation_and_zero() {
        let one = Bits512::one();
        let tiny = Bits512::from_f64(2f64.powi(-400)).unwrap();
        let a = vec![one, tiny, Bits512::zero()];
        let b = vec![one, one, one];
        // exact = 1 + 2^-400; check the tiny term survives.
        let rns = dot_rns(&a, &b).expect("plan");
        assert_eq!(rns, one + tiny);
        // Pure cancellation to zero.
        let a2 = vec![one, -one];
        let b2 = vec![one, one];
        let rns2 = dot_rns(&a2, &b2).expect("plan");
        assert!(rns2.is_zero());
    }

    #[test]
    fn rns_dot_strided_and_256bit() {
        let a: Vec<Bits256> = (0..40)
            .map(|i| Bits256::from_f64((i as f64 - 20.0) * 0.37 + 0.11).unwrap())
            .collect();
        let b: Vec<Bits256> = (0..40)
            .map(|i| Bits256::from_f64((i as f64 + 3.0) * 1.13 - 0.77).unwrap())
            .collect();
        let plan = RnsPlan::for_pair(&a, &b, 20).expect("plan");
        let ra = plan.encode(&a, EncodeSide::A).unwrap();
        let rb = plan.encode(&b, EncodeSide::B).unwrap();
        // stride 2: terms a[0],a[2],...,a[38] with b[0..20]
        let res = plan.dot_residues(&ra, 0, 2, &rb, 0, 1, 20);
        let got: Bits256 = plan.reconstruct(&res);
        let want = exact_product(
            &a.iter().step_by(2).copied().collect::<Vec<_>>(),
            &b[..20],
        )
        .unwrap()
        .to_mpfloat::<4>();
        assert_eq!(got, want);
    }

    #[test]
    fn rns_plan_rejects_nonfinite_and_huge_spread() {
        let a = vec![Bits512::one(), Bits512::infinity()];
        let b = vec![Bits512::one(); 2];
        assert!(RnsPlan::for_pair(&a, &b, 2).is_none());
        // Exponent spread beyond the window: 1 vs 2^5000.
        let a2 = vec![
            Bits512::one(),
            Bits512::from_f64(f64::MAX).unwrap(),
        ];
        let big = Bits512::from_f64(f64::MAX).unwrap();
        let huge = big * big * big * big * big * big * big * big * big * big * big * big;
        let a3 = vec![Bits512::one(), huge];
        let b3 = vec![Bits512::one(); 2];
        // 2^~7560 spread exceeds MAX_SPREAD=4096.
        assert!(RnsPlan::for_pair(&a3, &b3, 2).is_none());
        let _ = a2;
    }

    #[test]
    fn rns_profitability_gate_prefers_long_dots() {
        let a = vec![Bits512::one(); 64];
        let b = vec![Bits512::one(); 64];
        let plan = RnsPlan::for_pair(&a, &b, 64).unwrap();
        // Long dot, many outputs, few encoded elements: profitable.
        assert!(plan.profitable(64, 100_000, 128, 8));
        // Tiny output count cannot amortize encode + plan.
        assert!(!plan.profitable(64, 8, 128, 8));
        assert!(!plan.profitable(4, 100_000, 128, 8));
    }
}
