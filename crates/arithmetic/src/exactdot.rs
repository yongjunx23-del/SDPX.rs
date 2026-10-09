//! Correctly rounded dot products by exact fixed-point accumulation.
//!
//! Every finite `MpFloat<N>` is `M · 2^(e - P)` with an `N`-limb mantissa `M`,
//! so a product of two values is the exact `2N`-limb integer `Ma · Mb` scaled by
//! `2^(ea + eb - 2P)`. Aligning all products to the smallest such scale and
//! summing them with GMP's `mpn` layer gives the dot product exactly; one
//! `mpfr_set_z_2exp` then rounds it once, nearest-even, at the destination.
//!
//! This is the contract of the residue (RNS) path: exact accumulation, one
//! rounding. The result is never less accurate than the FMA chain it replaces
//! and does not depend on the term order. A non-finite operand, or an exponent
//! spread wider than [`MAX_ACC_LIMBS`], declines so the caller keeps the chain.
use crate::MpFloat;
use gmp_mpfr_sys::{gmp, mpfr};
use std::cell::RefCell;
use std::ptr::NonNull;

/// Accumulator bound per sign, in limbs (about 32k bits of exponent spread).
const MAX_ACC_LIMBS: usize = 512;
/// Widest supported mantissa, in limbs (2048-bit values).
pub(crate) const MAX_N: usize = 32;
/// Widest mantissa taking the inlined product; GMP's assembly wins above
/// (measured on Apple M4: 4 limbs 16.3 -> 11.9 ns/term, 8 limbs 30 -> 38).
pub(crate) const INLINE_N: usize = 4;
/// Widest mantissa multiplied by the inlined schoolbook; GMP's assembly
/// `mpn_mul_n` wins above (Apple M4: 8 limbs 30 -> 38 ns/term inline).
pub(crate) const SCHOOLBOOK_N: usize = 4;

/// Per-thread scratch, retained across calls so a dot product allocates only
/// on first use. Operands are held by address for the duration of one call.
#[derive(Default)]
struct Scratch {
    terms: Vec<(usize, usize)>,
    pos: Vec<u64>,
    neg: Vec<u64>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

type Pair<'a, const N: usize> = (&'a MpFloat<N>, &'a MpFloat<N>);

/// Exact dot product rounded once. `chain` evaluates the FMA chain and is used
/// when the exact path declines or the scratch is unavailable (reentrancy).
pub(crate) fn dot<'a, const N: usize>(
    pairs: impl IntoIterator<Item = Pair<'a, N>>,
    chain: impl Fn(&mut dyn Iterator<Item = Pair<'a, N>>) -> MpFloat<N>,
) -> MpFloat<N> {
    dot_with(pairs, chain, N <= INLINE_N)
}

/// `dot` with the per-term kernel chosen explicitly: the inlined product or
/// GMP's `mpn` calls. Both accumulate the same exact integer.
fn dot_with<'a, const N: usize>(
    pairs: impl IntoIterator<Item = Pair<'a, N>>,
    chain: impl Fn(&mut dyn Iterator<Item = Pair<'a, N>>) -> MpFloat<N>,
    inline: bool,
) -> MpFloat<N> {
    let mut pairs = pairs.into_iter();
    if N > MAX_N {
        return chain(&mut pairs);
    }
    let scratch =
        SCRATCH.try_with(|s| s.try_borrow_mut().ok().map(|mut s| std::mem::take(&mut *s)));
    let Ok(Some(mut scratch)) = scratch else {
        return chain(&mut pairs);
    };
    scratch.terms.clear();
    let (mut emin, mut emax, mut finite) = (i64::MAX, i64::MIN, true);
    for (a, b) in pairs {
        let regular = |k: i32| k.abs() == mpfr::REGULAR_KIND;
        let zero = |k: i32| k.abs() == mpfr::ZERO_KIND;
        if regular(a.kind) && regular(b.kind) {
            let e = a.exponent as i64 + b.exponent as i64;
            emin = emin.min(e);
            emax = emax.max(e);
        } else if (zero(a.kind) && (zero(b.kind) || regular(b.kind)))
            || (zero(b.kind) && regular(a.kind))
        {
            // An exact zero product adds nothing (the sign of a zero result is
            // not significant to the solver).
            continue;
        } else {
            finite = false;
        }
        scratch
            .terms
            .push((a as *const _ as usize, b as *const _ as usize));
    }
    // SAFETY: the addresses were taken from references that outlive this call.
    let view = |&(a, b): &(usize, usize)| -> Pair<'a, N> {
        unsafe { (&*(a as *const MpFloat<N>), &*(b as *const MpFloat<N>)) }
    };
    let exact = if !finite {
        None
    } else if scratch.terms.is_empty() {
        Some(MpFloat::<N>::default())
    } else {
        let Scratch { terms, pos, neg } = &mut scratch;
        accumulate(pos, neg, emin, emax, inline, terms.iter().map(view))
    };
    let out = exact.unwrap_or_else(|| chain(&mut scratch.terms.iter().map(view)));
    let _ = SCRATCH.try_with(|s| {
        if let Ok(mut slot) = s.try_borrow_mut() {
            *slot = scratch;
        }
    });
    out
}

/// Exact dot over two equal-length slices: one exponent scan, one
/// accumulation pass, no collected term list. Same value as `dot`.
pub(crate) fn dot_slices<'a, const N: usize>(
    a: &'a [MpFloat<N>],
    b: &'a [MpFloat<N>],
    chain: impl Fn(&mut dyn Iterator<Item = Pair<'a, N>>) -> MpFloat<N>,
) -> MpFloat<N> {
    assert_eq!(a.len(), b.len());
    if N > MAX_N {
        return chain(&mut a.iter().zip(b));
    }
    let regular = |k: i32| k.abs() == mpfr::REGULAR_KIND;
    let zero = |k: i32| k.abs() == mpfr::ZERO_KIND;
    let (mut emin, mut emax, mut finite) = (i64::MAX, i64::MIN, true);
    for (x, y) in a.iter().zip(b) {
        if regular(x.kind) && regular(y.kind) {
            let e = x.exponent as i64 + y.exponent as i64;
            emin = emin.min(e);
            emax = emax.max(e);
        } else if !(zero(x.kind) && (zero(y.kind) || regular(y.kind))
            || zero(y.kind) && regular(x.kind))
        {
            finite = false;
        }
    }
    if !finite {
        return dot(a.iter().zip(b), chain);
    }
    if emin == i64::MAX {
        return MpFloat::<N>::default();
    }
    let scratch =
        SCRATCH.try_with(|s| s.try_borrow_mut().ok().map(|mut s| std::mem::take(&mut *s)));
    let Ok(Some(mut scratch)) = scratch else {
        return chain(&mut a.iter().zip(b));
    };
    let terms = a
        .iter()
        .zip(b)
        .filter(|(x, y)| regular(x.kind) && regular(y.kind));
    let exact = accumulate(
        &mut scratch.pos,
        &mut scratch.neg,
        emin,
        emax,
        N <= INLINE_N,
        terms,
    );
    let _ = SCRATCH.try_with(|s| {
        if let Ok(mut slot) = s.try_borrow_mut() {
            *slot = scratch;
        }
    });
    exact.unwrap_or_else(|| dot(a.iter().zip(b), chain))
}

fn accumulate<'a, const N: usize>(
    pos: &mut Vec<u64>,
    neg: &mut Vec<u64>,
    emin: i64,
    emax: i64,
    inline: bool,
    terms: impl Iterator<Item = Pair<'a, N>> + Clone,
) -> Option<MpFloat<N>> {
    // Bit 0 of the accumulator weighs 2^(emin - 2P). A term with exponent sum
    // e lands at bit offset e - emin; its 2N-limb product plus one shift limb
    // ends at or below limb (emax - emin)/64 + 2N + 1. One further limb absorbs
    // the carry growth of up to 2^64 terms.
    let len = usize::try_from(emax - emin).ok()? / 64 + 2 * N + 3;
    if len > MAX_ACC_LIMBS {
        return None;
    }
    for acc in [&mut *pos, &mut *neg] {
        acc.clear();
        acc.resize(len, 0);
    }
    if inline && N <= INLINE_N {
        // The accumulator is picked by the product's sign bit, not a branch:
        // signs of consecutive terms are data-dependent and mispredict.
        let accs = [pos.as_mut_ptr(), neg.as_mut_ptr()];
        // One product buffer per dot; every term overwrites its 2N limbs.
        let mut scratch = [0u64; 2 * MAX_N];
        for (a, b) in terms.clone() {
            let offset = (a.exponent as i64 + b.exponent as i64 - emin) as usize;
            let negative = ((a.kind < 0) != (b.kind < 0)) as usize;
            let limb = offset / 64;
            // SAFETY: both accumulators hold `len` limbs and `limb < len`; the
            // bound on `len` covers the shifted product and its carry. The two
            // pointers address distinct vectors.
            let target =
                unsafe { std::slice::from_raw_parts_mut(accs[negative].add(limb), len - limb) };
            add_shifted_product::<N>(
                target,
                &a.limbs,
                &b.limbs,
                (offset % 64) as u32,
                &mut scratch,
            );
        }
    }
    let mut prod = [0u64; 2 * MAX_N + 1];
    for (a, b) in terms.filter(|_| !(inline && N <= INLINE_N)) {
        let offset = (a.exponent as i64 + b.exponent as i64 - emin) as usize;
        let target = if (a.kind < 0) != (b.kind < 0) {
            &mut *neg
        } else {
            &mut *pos
        };
        let (limb, shift) = (offset / 64, (offset % 64) as u32);
        // SAFETY: `prod` holds 2N+1 <= 2*MAX_N+1 limbs; `target[limb..len]`
        // covers the shifted product and its carry by the bound on `len`.
        // In-place mpn operands are identical, which GMP permits.
        unsafe {
            mpn_mul_n::<N>(prod.as_mut_ptr(), a.limbs.as_ptr(), b.limbs.as_ptr());
            let width = if shift == 0 {
                2 * N
            } else {
                prod[2 * N] =
                    gmp::mpn_lshift(prod.as_mut_ptr(), prod.as_ptr(), (2 * N) as _, shift);
                2 * N + 1
            };
            let dst = target.as_mut_ptr().add(limb);
            if gmp::mpn_add_n(dst, dst, prod.as_ptr(), width as _) != 0 {
                let rest = dst.add(width);
                gmp::mpn_add_1(rest, rest, (len - limb - width) as _, 1);
            }
        }
    }
    // SAFETY: both accumulators hold `len` limbs; the difference is written in
    // place into the larger one.
    let (sign, mag) = unsafe {
        match gmp::mpn_cmp(pos.as_ptr(), neg.as_ptr(), len as _) {
            0 => return Some(MpFloat::<N>::default()),
            c if c > 0 => {
                gmp::mpn_sub_n(pos.as_mut_ptr(), pos.as_ptr(), neg.as_ptr(), len as _);
                (1, &*pos)
            }
            _ => {
                gmp::mpn_sub_n(neg.as_mut_ptr(), neg.as_ptr(), pos.as_ptr(), len as _);
                (-1, &*neg)
            }
        }
    };
    let used = mag.iter().rposition(|&l| l != 0)? + 1;
    // Read-only integer view of the magnitude; MPFR does not retain it.
    let z = gmp::mpz_t {
        alloc: used as _,
        size: sign * used as i32,
        d: NonNull::new(mag.as_ptr() as *mut gmp::limb_t)?,
    };
    let scale = emin - 2 * MpFloat::<N>::PRECISION_BITS as i64;
    Some(MpFloat::<N>::output(|r| unsafe {
        mpfr::set_z_2exp(r, &z, scale as mpfr::exp_t, mpfr::rnd_t::RNDN);
    }))
}

/// `rp[..2N] = up * vp` for `N` limbs. GMP's fat x86_64 build gives every
/// Zen part Zen 1's `MUL_TOOM22_THRESHOLD` of 16, so 1024-bit (16-limb)
/// products went through Toom-22 (EPYC 7742 at 1024 bits: `toom22`, `add_n`
/// and `sub_n` 12% of cycles); GMP's own Zen 2 tuning keeps the basecase
/// through 18 limbs. The product is exact either way.
///
/// # Safety
/// `rp` holds `2N` limbs and does not overlap the `N`-limb inputs.
#[inline(always)]
pub(crate) unsafe fn mpn_mul_n<const N: usize>(rp: *mut u64, up: *const u64, vp: *const u64) {
    #[cfg(target_arch = "x86_64")]
    if N < 19 {
        extern "C" {
            fn __gmpn_mul_basecase(rp: *mut u64, up: *const u64, un: gmp::size_t, vp: *const u64, vn: gmp::size_t);
        }
        return __gmpn_mul_basecase(rp, up, N as _, vp, N as _);
    }
    gmp::mpn_mul_n(rp, up, vp, N as _);
}

/// `out[..2N] = a * b` for `N`-limb little-endian mantissas: an inlined
/// schoolbook up to `SCHOOLBOOK_N` limbs (row 0 writes, later rows
/// accumulate, so `out` needs no zeroing), GMP's assembly `mpn_mul_n` above.
#[inline(always)]
pub(crate) fn mul_limbs<const N: usize>(a: &[u64; N], b: &[u64; N], out: &mut [u64]) {
    debug_assert!(out.len() >= 2 * N);
    if N > SCHOOLBOOK_N {
        // SAFETY: `out` holds 2N limbs and does not overlap the inputs.
        unsafe { mpn_mul_n::<N>(out.as_mut_ptr(), a.as_ptr(), b.as_ptr()) };
        return;
    }
    let mut carry = 0u64;
    for j in 0..N {
        let t = a[0] as u128 * b[j] as u128 + carry as u128;
        out[j] = t as u64;
        carry = (t >> 64) as u64;
    }
    out[N] = carry;
    for i in 1..N {
        let mut carry = 0u64;
        for j in 0..N {
            let t = a[i] as u128 * b[j] as u128 + out[i + j] as u128 + carry as u128;
            out[i + j] = t as u64;
            carry = (t >> 64) as u64;
        }
        out[i + N] = carry;
    }
}

/// Run `f` on a zeroed scratch of at least `2N + 2` limbs sized to `N`, so
/// narrow types do not clear a buffer sized for the widest.
#[inline(always)]
pub(crate) fn with_limbs<const N: usize, R>(f: impl FnOnce(&mut [u64]) -> R) -> R {
    if N <= 4 {
        f(&mut [0u64; 10])
    } else if N <= 8 {
        f(&mut [0u64; 18])
    } else if N <= 16 {
        f(&mut [0u64; 34])
    } else {
        f(&mut [0u64; 2 * MAX_N + 2])
    }
}

/// `dst += (a * b) << shift` for `N`-limb mantissas, `shift < 64`, with the
/// carry propagated through `dst`. Schoolbook product and shifted add are
/// inlined for the const width: the exact integer result of GMP's
/// `mpn_mul_n` + `mpn_lshift` + `mpn_add_n` without three calls per term.
#[inline(always)]
fn add_shifted_product<const N: usize>(
    dst: &mut [u64],
    a: &[u64; N],
    b: &[u64; N],
    shift: u32,
    prod: &mut [u64],
) {
    {
        let prod = &mut prod[..2 * N];
        mul_limbs::<N>(a, b, prod);
        let mut carry = false;
        let mut add = |slot: &mut u64, v: u64| {
            let (s, c1) = slot.overflowing_add(v);
            let (s, c2) = s.overflowing_add(carry as u64);
            *slot = s;
            carry = c1 | c2;
        };
        let width = if shift == 0 {
            for t in 0..2 * N {
                add(&mut dst[t], prod[t]);
            }
            2 * N
        } else {
            add(&mut dst[0], prod[0] << shift);
            for t in 1..2 * N {
                add(
                    &mut dst[t],
                    (prod[t] << shift) | (prod[t - 1] >> (64 - shift)),
                );
            }
            add(&mut dst[2 * N], prod[2 * N - 1] >> (64 - shift));
            2 * N + 1
        };
        let mut t = width;
        while carry {
            let (s, c) = dst[t].overflowing_add(1);
            dst[t] = s;
            carry = c;
            t += 1;
        }
    }
}

impl<const N: usize> MpFloat<N> {
    /// The value `(-1)^negative · mag · 2^scale` rounded once, nearest-even.
    /// `mag` holds little-endian 64-bit limbs of an exact integer. Used by
    /// residue products that reconstruct the exact sum before rounding.
    pub fn from_scaled_integer(negative: bool, mag: &[u64], scale: i64) -> Self {
        let mut n = mag.len();
        while n > 0 && mag[n - 1] == 0 {
            n -= 1;
        }
        if n == 0 {
            return Self::default();
        }
        let mag = &mag[..n];
        let bits = n * 64 - mag[n - 1].leading_zeros() as usize;
        let p = Self::PRECISION_BITS;
        // Nearest-even rounding of the top `p` bits, done on the limbs. The
        // mantissa is `M·2^(e-p)` with `2^(p-1) <= M < 2^p` (see `dyadic`).
        let (mut limbs, mut shift) = ([0u64; N], bits as i64 - p as i64);
        let bit = |i: i64| -> bool { i >= 0 && (mag[(i / 64) as usize] >> (i % 64)) & 1 == 1 };
        for (l, limb) in limbs.iter_mut().enumerate() {
            // Limb `l` of M holds bits [shift + 64l, shift + 64l + 64) of mag.
            let start = shift + 64 * l as i64;
            let mut v = 0u64;
            for part in 0..2 {
                let src = start.div_euclid(64) + part;
                let off = start.rem_euclid(64);
                if src < 0 || src as usize >= n {
                    continue;
                }
                let w = mag[src as usize];
                v |= if part == 0 {
                    w >> off
                } else if off == 0 {
                    0
                } else {
                    w << (64 - off)
                };
            }
            *limb = v;
        }
        if shift > 0 {
            let round = bit(shift - 1);
            // Any set bit strictly below the round bit.
            let below = (shift - 1) as usize;
            let (full, rem) = (below / 64, below % 64);
            let sticky = mag[..full].iter().any(|&w| w != 0)
                || (rem > 0 && mag[full] & ((1u64 << rem) - 1) != 0);
            if round && (sticky || limbs[0] & 1 == 1) {
                let mut carry = true;
                for limb in limbs.iter_mut() {
                    let (v, o) = limb.overflowing_add(carry as u64);
                    *limb = v;
                    carry = o;
                    if !carry {
                        break;
                    }
                }
                if carry {
                    limbs[N - 1] = 1u64 << 63;
                    shift += 1;
                }
            }
        }
        let exponent = scale + shift + p as i64;
        let (emin, emax) = unsafe { (mpfr::get_emin() as i64, mpfr::get_emax() as i64) };
        if exponent < emin || exponent > emax {
            // Overflow or underflow: MPFR's own rounding and range handling.
            let z = gmp::mpz_t {
                alloc: n as _,
                size: if negative { -(n as i32) } else { n as i32 },
                d: NonNull::new(mag.as_ptr() as *mut gmp::limb_t).expect("non-empty"),
            };
            return MpFloat::<N>::output(|r| unsafe {
                mpfr::set_z_2exp(r, &z, scale as mpfr::exp_t, mpfr::rnd_t::RNDN);
            });
        }
        let kind = if negative {
            -mpfr::REGULAR_KIND
        } else {
            mpfr::REGULAR_KIND
        };
        Self::exact_decode(kind, exponent, limbs)
    }
}

#[cfg(test)]
mod tests {
    use crate::integer::exact_product;
    use crate::MpFloat;
    use num_traits::{FromPrimitive, One, Zero};

    // Deterministic xorshift values with a controllable exponent spread and
    // sign mix, including exact cancellation between terms.
    fn values<const N: usize>(seed: u64, k: usize, spread: i32) -> Vec<MpFloat<N>> {
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        (0..k)
            .map(|_| {
                let m = MpFloat::<N>::from_f64(next() as f64 / u64::MAX as f64 - 0.5).unwrap();
                let third =
                    MpFloat::<N>::one() / MpFloat::<N>::from_u32(3 + (next() % 5) as u32).unwrap();
                let e = if spread == 0 {
                    0
                } else {
                    (next() % (2 * spread as u64 + 1)) as i32 - spread
                };
                (m + third).scale_pow2(e as i64)
            })
            .collect()
    }

    fn check<const N: usize>() {
        for (seed, k, spread) in [
            (1, 1, 0),
            (2, 2, 0),
            (3, 7, 3),
            (4, 64, 40),
            (5, 33, 300),
            (6, 128, 900),
        ] {
            let a = values::<N>(seed, k, spread);
            let mut b = values::<N>(seed * 7919, k, spread);
            // Force a near-total cancellation in the middle of the sum.
            if k > 3 {
                b[k / 2] = -(a[0] * b[0]) / a[k / 2];
            }
            let expected = exact_product(&a, &b).unwrap().to_mpfloat::<N>();
            assert_eq!(
                MpFloat::<N>::dot_fma(a.iter().zip(&b)),
                expected,
                "N={N} k={k} spread={spread}"
            );
            assert_eq!(
                super::dot_slices(&a, &b, |t| MpFloat::<N>::dot_fma_chain(t)),
                expected,
                "slices N={N} k={k} spread={spread}"
            );
            // Order independence: reversing the terms gives the same bits.
            assert_eq!(
                MpFloat::<N>::dot_fma(a.iter().rev().zip(b.iter().rev())),
                expected
            );
        }
        // Exact cancellation yields zero; empty and all-zero sums are zero.
        let x = values::<N>(9, 4, 10);
        let neg: Vec<_> = x.iter().map(|v| -*v).collect();
        let both: Vec<_> = x.iter().chain(&neg).copied().collect();
        let ones = vec![MpFloat::<N>::one(); both.len()];
        assert!(MpFloat::<N>::dot_fma(both.iter().zip(&ones)).is_zero());
        let z = [MpFloat::<N>::zero(); 3];
        assert!(MpFloat::<N>::dot_fma(z.iter().zip(&x[..3])).is_zero());
        assert!(super::dot_slices(&z, &x[..3], |t| MpFloat::<N>::dot_fma_chain(t)).is_zero());
        assert!(super::dot_slices(&both, &ones, |t| MpFloat::<N>::dot_fma_chain(t)).is_zero());
        // An exponent spread wider than the window falls back to the chain.
        let wide = [
            MpFloat::<N>::one(),
            MpFloat::<N>::one().scale_pow2(-100_000),
        ];
        let ones = [MpFloat::<N>::one(); 2];
        assert_eq!(
            MpFloat::<N>::dot_fma(wide.iter().zip(&ones)),
            MpFloat::<N>::dot_fma_chain(wide.iter().zip(&ones))
        );
    }

    // The inlined kernel accumulates the same exact integer as GMP's mpn
    // calls: identical bits on spread, cancelling and carry-heavy inputs.
    fn inline_matches_gmp<const N: usize>() {
        let run = |a: &[MpFloat<N>], b: &[MpFloat<N>], inline| {
            super::dot_with(a.iter().zip(b), |t| MpFloat::<N>::dot_fma_chain(t), inline)
        };
        for (seed, k, spread) in [(11, 5, 0), (12, 97, 20), (13, 400, 63), (14, 257, 700)] {
            let a = values::<N>(seed, k, spread);
            let b = values::<N>(seed * 31, k, spread);
            assert_eq!(
                run(&a, &b, true),
                run(&a, &b, false),
                "N={N} k={k} spread={spread}"
            );
        }
        // All-ones mantissas: every limb add carries, and staggered shifts
        // push carries through the accumulator.
        let p = MpFloat::<N>::PRECISION_BITS as i64;
        let top = MpFloat::<N>::one() - MpFloat::<N>::one().scale_pow2(-p);
        for shifts in [1usize, 3, 64, 130] {
            let a: Vec<_> = (0..300)
                .map(|i| top.scale_pow2(-((i % shifts) as i64)))
                .collect();
            let b = vec![top; a.len()];
            assert_eq!(
                run(&a, &b, true),
                run(&a, &b, false),
                "N={N} carry {shifts}"
            );
            let neg: Vec<_> = b.iter().map(|v| -*v).collect();
            let mixed: Vec<_> = b
                .iter()
                .zip(&neg)
                .enumerate()
                .map(|(i, (x, y))| if i % 2 == 0 { *x } else { *y })
                .collect();
            assert_eq!(
                run(&a, &mixed, true),
                run(&a, &mixed, false),
                "N={N} mixed {shifts}"
            );
        }
    }

    #[test]
    fn inline_kernel_matches_gmp_kernel() {
        inline_matches_gmp::<1>();
        inline_matches_gmp::<2>();
        inline_matches_gmp::<3>();
        inline_matches_gmp::<4>();
    }

    #[test]
    fn exact_dot_matches_integer_oracle_all_precisions() {
        check::<2>();
        check::<4>();
        check::<8>();
        check::<12>();
        check::<16>();
        check::<32>();
    }
}
