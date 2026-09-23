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
const MAX_N: usize = 32;

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
    let mut pairs = pairs.into_iter();
    if N > MAX_N {
        return chain(&mut pairs);
    }
    let scratch = SCRATCH.try_with(|s| s.try_borrow_mut().ok().map(|mut s| std::mem::take(&mut *s)));
    let Ok(Some(mut scratch)) = scratch else {
        return chain(&mut pairs);
    };
    scratch.terms.clear();
    let (mut emin, mut emax, mut finite) = (i64::MAX, i64::MIN, true);
    for (a, b) in pairs {
        let regular = |k: i32| k.abs() == mpfr::REGULAR_KIND;
        let zero = |k: i32| k.abs() == mpfr::ZERO_KIND;
        if (zero(a.kind) && (zero(b.kind) || regular(b.kind)))
            || (zero(b.kind) && regular(a.kind))
        {
            // An exact zero product adds nothing (the sign of a zero result is
            // not significant to the solver).
            continue;
        }
        if regular(a.kind) && regular(b.kind) {
            let e = a.exponent as i64 + b.exponent as i64;
            emin = emin.min(e);
            emax = emax.max(e);
        } else {
            finite = false;
        }
        scratch.terms.push((a as *const _ as usize, b as *const _ as usize));
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
        accumulate::<N>(&mut scratch, emin, emax)
    };
    let out = exact.unwrap_or_else(|| chain(&mut scratch.terms.iter().map(view)));
    let _ = SCRATCH.try_with(|s| {
        if let Ok(mut slot) = s.try_borrow_mut() {
            *slot = scratch;
        }
    });
    out
}

fn accumulate<const N: usize>(scratch: &mut Scratch, emin: i64, emax: i64) -> Option<MpFloat<N>> {
    // Bit 0 of the accumulator weighs 2^(emin - 2P). A term with exponent sum
    // e lands at bit offset e - emin; its 2N-limb product plus one shift limb
    // ends at or below limb (emax - emin)/64 + 2N + 1. One further limb absorbs
    // the carry growth of up to 2^64 terms.
    let len = usize::try_from(emax - emin).ok()? / 64 + 2 * N + 3;
    if len > MAX_ACC_LIMBS {
        return None;
    }
    let Scratch { terms, pos, neg } = scratch;
    for acc in [&mut *pos, &mut *neg] {
        acc.clear();
        acc.resize(len, 0);
    }
    let mut prod = [0u64; 2 * MAX_N + 1];
    for &(a, b) in terms.iter() {
        // SAFETY: see `dot`; both operands are live, finite MpFloat<N> values.
        let (a, b) = unsafe { (&*(a as *const MpFloat<N>), &*(b as *const MpFloat<N>)) };
        let offset = (a.exponent as i64 + b.exponent as i64 - emin) as usize;
        let (limb, shift) = (offset / 64, (offset % 64) as u32);
        let target = if (a.kind < 0) != (b.kind < 0) { &mut *neg } else { &mut *pos };
        // SAFETY: `prod` holds 2N+1 <= 2*MAX_N+1 limbs; `target[limb..len]`
        // covers the shifted product and its carry by the bound on `len`.
        // In-place mpn operands are identical, which GMP permits.
        unsafe {
            gmp::mpn_mul_n(prod.as_mut_ptr(), a.limbs.as_ptr(), b.limbs.as_ptr(), N as _);
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
                let third = MpFloat::<N>::one() / MpFloat::<N>::from_u32(3 + (next() % 5) as u32).unwrap();
                let e = if spread == 0 { 0 } else { (next() % (2 * spread as u64 + 1)) as i32 - spread };
                (m + third).scale_pow2(e as i64)
            })
            .collect()
    }

    fn check<const N: usize>() {
        for (seed, k, spread) in [(1, 1, 0), (2, 2, 0), (3, 7, 3), (4, 64, 40), (5, 33, 300), (6, 128, 900)] {
            let a = values::<N>(seed, k, spread);
            let mut b = values::<N>(seed * 7919, k, spread);
            // Force a near-total cancellation in the middle of the sum.
            if k > 3 {
                b[k / 2] = -(a[0] * b[0]) / a[k / 2];
            }
            let expected = exact_product(&a, &b).unwrap().to_mpfloat::<N>();
            assert_eq!(MpFloat::<N>::dot_fma(a.iter().zip(&b)), expected, "N={N} k={k} spread={spread}");
            // Order independence: reversing the terms gives the same bits.
            assert_eq!(MpFloat::<N>::dot_fma(a.iter().rev().zip(b.iter().rev())), expected);
        }
        // Exact cancellation yields zero; empty and all-zero sums are zero.
        let x = values::<N>(9, 4, 10);
        let neg: Vec<_> = x.iter().map(|v| -*v).collect();
        let both: Vec<_> = x.iter().chain(&neg).copied().collect();
        let ones = vec![MpFloat::<N>::one(); both.len()];
        assert!(MpFloat::<N>::dot_fma(both.iter().zip(&ones)).is_zero());
        let z = [MpFloat::<N>::zero(); 3];
        assert!(MpFloat::<N>::dot_fma(z.iter().zip(&x[..3])).is_zero());
        // An exponent spread wider than the window falls back to the chain.
        let wide = [MpFloat::<N>::one(), MpFloat::<N>::one().scale_pow2(-100_000)];
        let ones = [MpFloat::<N>::one(); 2];
        assert_eq!(
            MpFloat::<N>::dot_fma(wide.iter().zip(&ones)),
            MpFloat::<N>::dot_fma_chain(wide.iter().zip(&ones))
        );
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
