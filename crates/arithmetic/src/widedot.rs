//! Wide fixed-point accumulation for `MpFloat` dot products.
//!
//! A dot product of `MpFloat` values otherwise pays one full MPFR
//! normalization + rounding per term. Here every product's 2N-limb
//! significand is fused limb-by-limb into a fixed-point window anchored above
//! the largest product exponent, accumulated as a two's-complement integer,
//! and normalized/rounded once. Terms landing entirely below the window
//! cannot move the `64N`-bit-rounded result; cancellation deep enough to
//! expose the noise floor, non-finite inputs and exponent-range extremes all
//! fall back to the caller's sequential path.

use crate::MpFloat;
use gmp_mpfr_sys::mpfr;

// Limbs of carry headroom above the largest product.
const GUARD_LIMBS: usize = 2;

/// Add (or, when `neg`, subtract) `v << pos` into the two's-complement
/// window. `v` is a u128 limb-pair product; `pos` is the absolute bit offset
/// of its low bit inside `acc`. Positions below zero clip the low bits —
/// returns whether any bit was dropped (each clip loses < 1 window unit).
#[inline(always)]
fn accum_u128(acc: &mut [u64], mut v: u128, mut pos: i64, neg: bool) -> bool {
    let mut clipped = false;
    if pos < 0 {
        let drop = (-pos).min(128) as u32;
        clipped = drop >= 128 || (v << (128 - drop)) != 0;
        v = v.checked_shr(drop).unwrap_or(0);
        pos = 0;
        if v == 0 {
            return clipped;
        }
    }
    let w = acc.len();
    let l = (pos / 64) as usize;
    let bit = (pos % 64) as u32;
    if l >= w {
        return clipped;
    }
    // v << bit spans at most three limbs.
    let w0 = (v as u64).wrapping_shl(bit);
    let w1 = if bit == 0 {
        (v >> 64) as u64
    } else {
        ((v >> 64) as u64).wrapping_shl(bit) | ((v as u64) >> (64 - bit))
    };
    let w2 = if bit == 0 {
        0
    } else {
        ((v >> 64) as u64) >> (64 - bit)
    };
    // Chain the three partial limbs into the window: acc[k] += p + carry_in,
    // carry propagates limb-by-limb then ripples upward.
    let mut carry = false;
    let mut k = l;
    for &p in &[w0, w1, w2] {
        if k >= w {
            return clipped;
        }
        let (r, c1) = if neg {
            acc[k].overflowing_sub(p)
        } else {
            acc[k].overflowing_add(p)
        };
        let (r, c2) = if carry {
            if neg { r.overflowing_sub(1) } else { r.overflowing_add(1) }
        } else {
            (r, false)
        };
        acc[k] = r;
        carry = c1 || c2;
        k += 1;
    }
    while carry && k < w {
        let (r, c) = if neg {
            acc[k].overflowing_sub(1)
        } else {
            acc[k].overflowing_add(1)
        };
        acc[k] = r;
        carry = c;
        k += 1;
    }
    clipped
}

/// Two's-complement negate over the whole window.
fn negate(acc: &mut [u64]) {
    let mut carry = true;
    for d in acc.iter_mut() {
        let (r, c) = (!*d).overflowing_add(carry as u64);
        *d = r;
        carry = c;
    }
}

/// Bit length of the magnitude window (highest set bit index + 1).
fn bit_len(acc: &[u64]) -> usize {
    for (i, &d) in acc.iter().enumerate().rev() {
        if d != 0 {
            return i * 64 + (64 - d.leading_zeros() as usize);
        }
    }
    0
}

/// Test bit `i` of the magnitude window.
#[inline(always)]
fn test_bit(acc: &[u64], i: usize) -> bool {
    (acc[i / 64] >> (i % 64)) & 1 != 0
}

/// Any set bit strictly below `i`.
fn any_below(acc: &[u64], i: usize) -> bool {
    let (limb, bit) = (i / 64, i % 64);
    if acc[..limb].iter().any(|&d| d != 0) {
        return true;
    }
    bit > 0 && acc[limb] & ((1u64 << bit) - 1) != 0
}

/// True when the `sh`-bit dropped fraction G sits within `n` window units of
/// the rounding boundary `2^(sh-1)`. Clipped mass (< n units total, unknown
/// sign) could then push the true sum across the boundary.
fn near_boundary(acc: &[u64], sh: usize, n: u64) -> bool {
    let h = sh - 1; // boundary bit position
    let (tl, tb) = (h / 64, h % 64);
    let top_mask = if tb == 0 { 0 } else { (1u64 << tb) - 1 };
    // f spans bits [0, h): limbs [1, tl) plus limb tl's low tb bits (for
    // tl = 0 that partial limb is acc[0] itself).
    let f_lo = if tl == 0 { acc[0] & top_mask } else { acc[0] };
    let high_clear = (1..tl).all(|i| acc[i] == 0) && (tl == 0 || acc[tl] & top_mask == 0);
    if acc[tl] >> tb & 1 != 0 {
        // G = 2^h + f; |G - B| = f — near iff f < n.
        high_clear && f_lo < n
    } else {
        // G = f < 2^h; B - G = c + 1 with c = 2^h - 1 - f — near iff c < n - 1.
        let c_lo = if tl == 0 { !acc[0] & top_mask } else { !acc[0] };
        let c_high_clear =
            (1..tl).all(|i| acc[i] == u64::MAX) && (tl == 0 || acc[tl] & top_mask == top_mask);
        n >= 2 && c_high_clear && c_lo <= n - 2
    }
}

/// Extract the top `prec` bits (a whole number of limbs) with round-to-
/// nearest-even on the discarded low bits; on carry-out the significand
/// wraps to exactly `2^prec` and the caller bumps the exponent.
fn normalize_rne<const N: usize>(acc: &[u64], bits: usize) -> ([u64; N], bool) {
    let prec = 64 * N;
    let sh = bits - prec;
    let mut out = [0u64; N];
    for (i, d) in out.iter_mut().enumerate() {
        let limb = sh / 64 + i;
        let bit = sh % 64;
        let lo = acc.get(limb).copied().unwrap_or(0) >> bit;
        let hi = if bit == 0 {
            0
        } else {
            acc.get(limb + 1).copied().unwrap_or(0) << (64 - bit)
        };
        *d = lo | hi;
    }
    let guard = test_bit(acc, sh - 1);
    let sticky = any_below(acc, sh - 1);
    if guard && (sticky || out[0] & 1 != 0) {
        let mut carry = true;
        for d in out.iter_mut() {
            let (r, c) = d.overflowing_add(carry as u64);
            *d = r;
            carry = c;
        }
        if carry {
            // M = 2^prec: renormalize to 2^(prec-1) with exponent + 1.
            out = [0u64; N];
            out[N - 1] = 1 << 63;
            return (out, true);
        }
    }
    (out, false)
}

/// Attempt a wide fixed-point dot product over operand pairs. `None` requests
/// the caller's sequential fallback (specials, overflow, deep cancellation).
/// Allocation-free: `scan` reads operand metadata, `accum` replays the same
/// iterator for accumulation — both are clones of the caller's iterator.
pub(crate) fn wide_dot<'a, const N: usize, I>(scan: I, accum: I) -> Option<MpFloat<N>>
where
    I: Iterator<Item = (&'a MpFloat<N>, &'a MpFloat<N>)>,
{
    const MAXW: usize = 4 * 32 + GUARD_LIMBS; // largest supported MpFloat window
    let prec = 64 * N;
    let w = 4 * N + GUARD_LIMBS;
    if w > MAXW {
        return None;
    }
    let mut e_top = i64::MIN;
    let mut n = 0u64;
    for (x, y) in scan {
        // Kinds order as NAN < INF < ZERO < REGULAR in absolute value.
        let (ax, ay) = (x.kind.abs(), y.kind.abs());
        if (ax == mpfr::ZERO_KIND && ay >= mpfr::ZERO_KIND)
            || (ay == mpfr::ZERO_KIND && ax >= mpfr::ZERO_KIND)
        {
            continue; // exact-zero times finite — drops out of the sum
        }
        if ax != mpfr::REGULAR_KIND || ay != mpfr::REGULAR_KIND {
            return None; // inf/nan (incl. 0·inf) — sequential path propagates
        }
        e_top = e_top.max(x.exponent.saturating_add(y.exponent));
        n += 1;
    }
    if n == 0 {
        return Some(MpFloat::default());
    }
    // Window [base, top): top = e_top + GUARD carry limbs. Exponents near the
    // i64 edges can't anchor a window — take the sequential path.
    let margin = (128 * N + 64 * w + 128) as i64;
    if e_top > i64::MAX - margin || e_top < i64::MIN + margin {
        return None;
    }
    let top = e_top + 64 * GUARD_LIMBS as i64;
    let base = top - 64 * w as i64;
    let mut window = [0u64; MAXW];
    let acc = &mut window[..w];
    let mut clipped = false; // any product bits dropped below the window
    for (x, y) in accum {
        let (ax, ay) = (x.kind.abs(), y.kind.abs());
        if ax != mpfr::REGULAR_KIND || ay != mpfr::REGULAR_KIND {
            continue; // zeros dropped by the scan; specials already rejected
        }
        // term = ±Ma·Mb·2^(e - 128N): offset of the product's low bit. A
        // saturated-low exponent lands the whole product below the window.
        let neg = x.kind.signum() != y.kind.signum();
        let Some(sh) = x
            .exponent
            .saturating_add(y.exponent)
            .checked_sub((128 * N) as i64)
            .and_then(|s| s.checked_sub(base))
        else {
            clipped = true;
            continue;
        };
        if sh <= -((128 * N) as i64) {
            clipped = true; // product entirely below the window
            continue;
        }
        for i in 0..N {
            let ai = x.limbs[i];
            if ai == 0 {
                continue;
            }
            for j in 0..N {
                let bj = y.limbs[j];
                if bj == 0 {
                    continue;
                }
                let pos = sh + 64 * (i + j) as i64;
                if pos >= 64 * w as i64 {
                    return None; // unreachable by window sizing — defensive
                }
                if pos <= -128 {
                    clipped = true;
                    continue;
                }
                clipped |= accum_u128(acc, ai as u128 * bj as u128, pos, neg);
            }
        }
    }
    let negative = (acc[w - 1] >> 63) != 0;
    if negative {
        negate(acc);
    }
    let bits = bit_len(acc);
    if bits == 0 {
        return Some(MpFloat::default());
    }
    // Dropped fractions total < n window units; keep the significand well
    // above that noise floor (deep cancellation falls back).
    let log2n = u64::BITS as usize - (n - 1).leading_zeros() as usize;
    if bits < prec + log2n + 4 {
        return None;
    }
    // With clipping, the true sum T = window ± <n units: if the dropped
    // fraction is within n of the rounding boundary, T could round the other
    // way — only a provable same-side gap keeps the single rounding exact.
    let sh_out = bits - prec;
    if clipped && near_boundary(acc, sh_out, n) {
        return None;
    }
    let e_out = base + bits as i64;
    let (m, carry) = normalize_rne::<N>(acc, bits);
    let e_final = e_out + carry as i64;
    let (emin, emax) = unsafe { (mpfr::get_emin(), mpfr::get_emax()) };
    if e_final < emin || e_final > emax {
        return None; // range edge — sequential path handles over/underflow
    }
    let mut out = MpFloat::<N>::default();
    out.limbs = m;
    out.exponent = e_final;
    out.kind = if negative {
        -mpfr::REGULAR_KIND
    } else {
        mpfr::REGULAR_KIND
    };
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bits256, Scalar};
    use num_traits::{FromPrimitive, One, Zero};

    /// The sequential-FMA semantics `dot_fma` falls back to.
    fn sequential<'a, const N: usize>(pairs: &[(&'a MpFloat<N>, &'a MpFloat<N>)]) -> MpFloat<N> {
        pairs
            .iter()
            .fold(MpFloat::zero(), |acc, &(x, y)| x.mul_add(*y, acc))
    }

    fn dot<const N: usize>(xs: &[MpFloat<N>], ys: &[MpFloat<N>]) -> MpFloat<N> {
        let pairs: Vec<_> = xs.iter().zip(ys.iter()).collect();
        MpFloat::dot_exact(pairs.iter().copied())
    }

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    #[test]
    fn random_dots_match_sequential() {
        let mut seed = 0x9e3779b97f4a7c15u64;
        for round in 0..64 {
            let n = 1 + (lcg(&mut seed).abs() * 48.0) as usize;
            let xs: Vec<Bits256> = (0..n)
                .map(|_| Bits256::from_f64(lcg(&mut seed) * 10f64.powi((lcg(&mut seed) * 20.0) as i32)).unwrap())
                .collect();
            let ys: Vec<Bits256> = (0..n)
                .map(|_| Bits256::from_f64(lcg(&mut seed) * 10f64.powi((lcg(&mut seed) * 20.0) as i32)).unwrap())
                .collect();
            let pairs: Vec<_> = xs.iter().zip(ys.iter()).collect();
            let wide = MpFloat::dot_exact(pairs.iter().copied());
            let seq = sequential(&pairs);
            // Sanity: the wide path actually engages for these generic dots.
            assert!(wide_dot(pairs.iter().copied(), pairs.iter().copied()).is_some(), "round {round} fell back");
            // The wide path rounds the exact sum once; the sequential path
            // rounds per step. They agree to a few ulp of the result.
            let diff = (wide - seq).abs();
            let tol = seq.abs() * Bits256::from_f64(2f64.powi(-240)).unwrap()
                + Bits256::from_f64(1e-280).unwrap();
            assert!(diff <= tol, "round {round}: wide={wide} seq={seq}");
        }
    }

    #[test]
    fn exact_integer_dots() {
        let xs: Vec<Bits256> = [3u64, 7, 11, 13, 17]
            .iter()
            .map(|&v| Bits256::from_u64(v).unwrap())
            .collect();
        let ys: Vec<Bits256> = [5u64, 19, 23, 29, 31]
            .iter()
            .map(|&v| Bits256::from_u64(v).unwrap())
            .collect();
        let expected: u64 = [3 * 5, 7 * 19, 11 * 23, 13 * 29, 17 * 31]
            .iter()
            .sum();
        assert_eq!(dot(&xs, &ys), Bits256::from_u64(expected).unwrap());
        // Mixed signs exercise the borrow (two's-complement) path.
        let xs: Vec<Bits256> = [3i64, -7, 11, -13, 17]
            .iter()
            .map(|&v| Bits256::from_i64(v).unwrap())
            .collect();
        let expected: i64 = 3 * 5 - 7 * 19 + 11 * 23 - 13 * 29 + 17 * 31;
        assert_eq!(dot(&xs, &ys), Bits256::from_i64(expected).unwrap());
        // All-negative sum.
        let xs: Vec<Bits256> = [-3i64, -7].iter().map(|&v| Bits256::from_i64(v).unwrap()).collect();
        let ys: Vec<Bits256> = [5i64, 19].iter().map(|&v| Bits256::from_i64(v).unwrap()).collect();
        assert_eq!(dot(&xs, &ys), Bits256::from_i64(-148).unwrap());
    }

    #[test]
    fn cancellation_and_specials() {
        let a = Bits256::from_f64(1.5).unwrap();
        let b = Bits256::from_f64(-2.25).unwrap();
        // Exact cancellation.
        let pairs = [(&a, &b), (&a, &(-b))];
        assert_eq!(MpFloat::dot_exact(pairs.iter().copied()), Bits256::zero());
        // A huge term plus a tiny one far below the result's precision.
        let big = Bits256::one();
        let tiny = Bits256::from_f64(2f64.powi(-200)).unwrap();
        assert_eq!(dot(&[big], &[tiny]), tiny);
        let pairs = [(&big, &big), (&big, &tiny)];
        assert_eq!(
            MpFloat::dot_exact(pairs.iter().copied()),
            big + tiny // 1 + 2^-200 is exactly representable at 256 bits
        );
        // Inf and NaN propagate through the sequential fallback.
        let inf = Bits256::infinity();
        assert!(MpFloat::dot_fma([(&inf, &big)].iter().copied()).is_infinite());
        let nan = Bits256::nan();
        assert!(MpFloat::dot_fma([(&nan, &big)].iter().copied()).is_nan());
        // 0·inf must produce NaN, not be silently dropped.
        let zero = Bits256::zero();
        assert!(MpFloat::dot_fma([(&zero, &inf)].iter().copied()).is_nan());
        assert!(MpFloat::dot_fma([(&inf, &zero)].iter().copied()).is_nan());
        // Deep cancellation below the window takes the fallback and stays exact.
        let c = Bits256::from_f64(1.0).unwrap() + Bits256::epsilon();
        let pairs = [(&c, &Bits256::one()), (&(-Bits256::one()), &Bits256::one())];
        assert_eq!(
            MpFloat::dot_exact(pairs.iter().copied()),
            Bits256::epsilon()
        );
    }

}
