//! Correctly rounded dot products by exact fixed-point accumulation.
//!
//! Every finite `MpFloat<N>` is `M · 2^(e - P)` with an `N`-limb mantissa `M`,
//! so a product of two values is the exact `2N`-limb integer `Ma · Mb` scaled by
//! `2^(ea + eb - 2P)`. Aligning all products to the smallest such scale and
//! summing their 64-bit limbs into 128-bit slots, with carries resolved once at
//! the end, gives the dot product exactly; one `mpfr_set_z_2exp` then rounds it
//! once, nearest-even, at the destination.
//!
//! This is the contract of the residue (RNS) path: exact accumulation, one
//! rounding. The result is never less accurate than the FMA chain it replaces
//! and does not depend on the term order. Wider exponent spreads use exact
//! products and MPFR summation; non-finite operands keep the caller's chain.
use crate::MpFloat;
use gmp_mpfr_sys::{gmp, mpfr};
use std::cell::RefCell;
use std::mem::MaybeUninit;
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
/// Widest mantissa multiplied by [`mul_rows`] on aarch64. Apple M4, ns per
/// product against GMP's `mpn_mul_n`: 8 limbs 22.2 -> 12.7, 10 limbs
/// 29.6 -> 19.4, 12 limbs 42.9 -> 35.3; 16 limbs 88 -> 150 spills.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
const ROWS_N: usize = 12;

/// Per-thread scratch, retained across calls so a dot product allocates only
/// on first use. Operands are held by address for the duration of one call.
#[derive(Default)]
struct Scratch {
    terms: Vec<(usize, usize)>,
    pos: Vec<u128>,
    neg: Vec<u128>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

type Pair<'a, const N: usize> = (&'a MpFloat<N>, &'a MpFloat<N>);

/// Exact dot product rounded once. `chain` preserves the caller's handling
/// of non-finite operands.
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
    let pairs = pairs.into_iter();
    if N > MAX_N {
        return dot_mpfr(pairs, chain);
    }
    let scratch =
        SCRATCH.try_with(|s| s.try_borrow_mut().ok().map(|mut s| std::mem::take(&mut *s)));
    let Ok(Some(mut scratch)) = scratch else {
        return dot_mpfr(pairs, chain);
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
    let out = exact.unwrap_or_else(|| dot_mpfr(scratch.terms.iter().map(view), chain));
    let _ = SCRATCH.try_with(|s| {
        if let Ok(mut slot) = s.try_borrow_mut() {
            *slot = scratch;
        }
    });
    out
}

/// Slow exact path without storage proportional to the exponent spread.
/// MPFR's `dot` first multiplies at 2P bits, but those products can overflow
/// or underflow before cancellation. Build their normalized mantissas and
/// exponents directly instead; `sum` rounds only the final value.
fn dot_mpfr<'a, const N: usize>(
    pairs: impl IntoIterator<Item = Pair<'a, N>>,
    chain: impl Fn(&mut dyn Iterator<Item = Pair<'a, N>>) -> MpFloat<N>,
) -> MpFloat<N> {
    let pairs: Vec<_> = pairs.into_iter().collect();
    let finite = |x: &MpFloat<N>| matches!(x.kind.abs(), mpfr::ZERO_KIND | mpfr::REGULAR_KIND);
    if pairs.iter().any(|(a, b)| !finite(a) || !finite(b)) {
        return chain(&mut pairs.into_iter());
    }
    let products: Vec<_> = pairs
        .into_iter()
        .filter(|(a, b)| !zero_product(a, b))
        .map(|(a, b)| {
            let mut limbs = vec![0u64; 2 * N];
            let mut exponent = a.exponent + b.exponent;
            let kind = if (a.kind < 0) != (b.kind < 0) {
                -mpfr::REGULAR_KIND
            } else {
                mpfr::REGULAR_KIND
            };
            // SAFETY: the product has 2N limbs; each input has N limbs.
            // Normalized factors give either 2P or 2P-1 significant bits.
            unsafe {
                mpn_mul_n::<N>(limbs.as_mut_ptr(), a.limbs.as_ptr(), b.limbs.as_ptr());
                if limbs[2 * N - 1] >> 63 == 0 {
                    gmp::mpn_lshift(limbs.as_mut_ptr(), limbs.as_ptr(), (2 * N) as _, 1);
                    exponent -= 1;
                }
                let mut desc = MaybeUninit::uninit();
                mpfr::custom_init_set(
                    desc.as_mut_ptr(),
                    kind,
                    exponent,
                    (2 * MpFloat::<N>::PRECISION_BITS) as _,
                    limbs.as_mut_ptr().cast(),
                );
                (limbs, desc.assume_init())
            }
        })
        .collect();
    let ptrs: Vec<_> = products
        .iter()
        .map(|(_, d)| (d as *const mpfr::mpfr_t).cast_mut())
        .collect();
    MpFloat::<N>::output(|r| unsafe {
        // Product descriptors borrow their owned limb buffers. Their exponents
        // may exceed the active range; only the sum is range-checked. MPFR's
        // summation visits bounded blocks rather than filling exponent gaps.
        let inexact = mpfr::sum(r, ptrs.as_ptr(), ptrs.len() as _, mpfr::rnd_t::RNDN);
        mpfr::check_range(r, inexact, mpfr::rnd_t::RNDN);
    })
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
        return dot_mpfr(a.iter().zip(b), chain);
    }
    if let Some(v) = window_dot(a, b) {
        return v;
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
        return dot_mpfr(a.iter().zip(b), chain);
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
    exact.unwrap_or_else(|| dot_mpfr(a.iter().zip(b), chain))
}

/// Exponent-sum reach, in limbs, of [`window_dot`] on each side of a dot's
/// first nonzero term. BFSS arrow-leaf dots span under 16 limbs; a wider
/// dot takes the scanned two-pass path.
const WINDOW: usize = 16;

/// [`dot_slices`] in one pass, without the exponent scan: bit 0 of the
/// limb accumulators weighs 2^(e0 - 64·WINDOW - 2P) for the first nonzero
/// term's exponent sum e0, and each product adds by one carry chain. The
/// sum is the same exact integer at another alignment, so the value is
/// that of the scanned path. `None` when a term is non-finite or lands
/// more than `WINDOW` limbs from e0.
fn window_dot<const N: usize>(a: &[MpFloat<N>], b: &[MpFloat<N>]) -> Option<MpFloat<N>> {
    // Kinds are 0 (NaN), ±1 (inf), ±2 (zero) and ±3 (regular), signed by
    // the value. The product of two kinds has the product's sign, and its
    // square is 81 for regular factors, 16 or 36 for an exact zero of finite
    // factors and below 16 when a factor is NaN or infinite (no sign to
    // branch on).
    const BOTH: i32 = (mpfr::REGULAR_KIND * mpfr::REGULAR_KIND).pow(2);
    const FINITE: i32 = (mpfr::ZERO_KIND * mpfr::ZERO_KIND).pow(2);
    let class = |x: &MpFloat<N>, y: &MpFloat<N>| (x.kind * y.kind).pow(2);
    let Some(first) = a.iter().zip(b).position(|(x, y)| {
        let k = class(x, y);
        k == BOTH || k < FINITE
    }) else {
        // Every product is an exact zero.
        return Some(MpFloat::<N>::default());
    };
    let (x, y) = (&a[first], &b[first]);
    if class(x, y) != BOTH {
        return None;
    }
    let base = x.exponent as i64 + y.exponent as i64 - 64 * WINDOW as i64;
    // The farthest term ends at limb 2·WINDOW + 2N + 1; one limb more holds
    // the carries of up to 2^64 terms.
    let len = 2 * WINDOW + 2 * N + 3;
    #[cfg(target_arch = "x86_64")]
    let fast = x86::fast();
    with_window::<N, _>(|acc| {
        let (pos, neg) = acc.split_at_mut(len);
        let neg = &mut neg[..len];
        let accs = [pos.as_mut_ptr(), neg.as_mut_ptr()];
        for (x, y) in a[first..].iter().zip(&b[first..]) {
            let kind = x.kind * y.kind;
            if kind * kind != BOTH {
                if kind * kind < FINITE {
                    return None;
                }
                continue;
            }
            let offset = (x.exponent as i64 + y.exponent as i64 - base) as u64;
            if offset > 128 * WINDOW as u64 {
                return None;
            }
            // SAFETY: the term ends at limb offset/64 + 2N + 1 < len - 1.
            unsafe {
                let dst = accs[(kind < 0) as usize].add((offset / 64) as usize);
                let shift = (offset % 64) as u32;
                #[cfg(target_arch = "x86_64")]
                if fast {
                    x86::add_term::<N>(dst, &x.limbs, &y.limbs, shift);
                    continue;
                }
                let mut prod = [0u64; 2 * MAX_N];
                mul_limbs::<N>(&x.limbs, &y.limbs, &mut prod[..2 * N]);
                add_chain(dst, &prod[..2 * N], shift);
            }
        }
        round_acc(pos, neg, len, base)
    })
}

/// Run `f` on a zeroed scratch of at least two [`window_dot`] accumulators
/// sized to `N`.
#[inline(always)]
fn with_window<const N: usize, R>(f: impl FnOnce(&mut [u64]) -> R) -> R {
    const W: usize = 2 * WINDOW + 3;
    if N <= 4 {
        f(&mut [0u64; 2 * (W + 8)])
    } else if N <= 8 {
        f(&mut [0u64; 2 * (W + 16)])
    } else if N <= 16 {
        f(&mut [0u64; 2 * (W + 32)])
    } else {
        f(&mut [0u64; 2 * (W + 2 * MAX_N)])
    }
}

/// `dst[0..] += limbs << shift` for `shift < 64`: one carry chain over
/// `limbs.len() + 1` limbs, then the carry onward.
///
/// # Safety
/// `dst` addresses limbs that hold the carried sum.
#[inline(always)]
unsafe fn add_chain(dst: *mut u64, limbs: &[u64], shift: u32) {
    let (mut below, mut c) = (0u64, false);
    for (t, &l) in limbs.iter().enumerate() {
        let piece = (l << shift) | ((below >> 1) >> (63 - shift));
        (*dst.add(t), c) = adc(*dst.add(t), piece, c);
        below = l;
    }
    let mut t = limbs.len();
    (*dst.add(t), c) = adc(*dst.add(t), (below >> 1) >> (63 - shift), c);
    while c {
        t += 1;
        (*dst.add(t), c) = (*dst.add(t)).overflowing_add(1);
    }
}

/// [`window_dot`] terms with BMI2 and ADX (Zen 2 and later; Intel since
/// Broadwell): `mulx` runs at one per cycle on Zen 2 where `mul` takes 1.7,
/// two carry chains (`adcx`/`adox`) form the product, `shlx`/`shrx` shift it
/// (the compiler's `shld` is microcoded on Zen), and the shifted product
/// adds to memory as one `adc` chain; wider products add as 4x4 blocks.
/// EPYC 7742, dump2223 factor leaf, ns per term against the portable
/// two-pass kernel: 2 limbs 11.2 -> 6.9, 4 limbs 24.0 -> 13.9, 8 limbs
/// 59.4 -> 46.6, 12 limbs 106 -> 95, 16 limbs 169 -> 168.
#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::asm;

    pub(super) fn fast() -> bool {
        std::is_x86_feature_detected!("bmi2") && std::is_x86_feature_detected!("adx")
    }

    /// `dst[0..] += (a * b) << shift` for `shift < 64`.
    ///
    /// # Safety
    /// The CPU has BMI2 and ADX; `dst` addresses limbs that hold the
    /// carried sum.
    #[inline(always)]
    pub(super) unsafe fn add_term<const N: usize>(
        dst: *mut u64,
        a: &[u64; N],
        b: &[u64; N],
        shift: u32,
    ) {
        let (a, b) = (a.as_ptr(), b.as_ptr());
        if N == 2 {
            chain(dst, mul2(a, b), shift);
        } else if N % 4 == 0 {
            // 4-limb blocks add at their own offsets, so the full product is
            // never stored.
            for i in (0..N).step_by(4) {
                for j in (0..N).step_by(4) {
                    chain(dst.add(i + j), mul4(a.add(i), b.add(j)), shift);
                }
            }
        } else {
            let mut prod = [0u64; 2 * super::MAX_N];
            super::mpn_mul_n::<N>(prod.as_mut_ptr(), a, b);
            chain_limbs(dst, &prod[..2 * N], shift);
        }
    }

    /// `a * b` for 2-limb `a` and `b`.
    #[inline(always)]
    unsafe fn mul2(a: *const u64, b: *const u64) -> [u64; 4] {
        let mut p = [0u64; 4];
        asm!(
            "mov rdx, [{a}]",
            "mulx {p1}, {p0}, [{b}]",
            "mulx {p2}, {t}, [{b} + 8]",
            "add {p1}, {t}",
            "adc {p2}, 0",
            "mov rdx, [{a} + 8]",
            "mulx {u}, {t}, [{b}]",
            "mulx {p3}, {a}, [{b} + 8]",
            "add {u}, {a}",
            "adc {p3}, 0",
            "add {p1}, {t}",
            "adc {p2}, {u}",
            "adc {p3}, 0",
            a = inout(reg) a => _,
            b = in(reg) b,
            p0 = out(reg) p[0], p1 = out(reg) p[1], p2 = out(reg) p[2], p3 = out(reg) p[3],
            t = out(reg) _,
            u = out(reg) _,
            out("rdx") _,
            options(pure, readonly, nostack),
        );
        p
    }

    /// `a * b` for 4-limb `a` and `b`: row 0 by one chain, rows 1-3 adding
    /// low halves by `adcx` and high halves by `adox`.
    #[inline(always)]
    unsafe fn mul4(a: *const u64, b: *const u64) -> [u64; 8] {
        let mut r = [0u64; 8];
        asm!(
            "mov rdx, [{a}]",
            "mulx {r1}, {r0}, [{b}]",
            "mulx {r2}, {l}, [{b} + 8]",
            "add {r1}, {l}",
            "mulx {r3}, {l}, [{b} + 16]",
            "adc {r2}, {l}",
            "mulx {r4}, {l}, [{b} + 24]",
            "adc {r3}, {l}",
            "adc {r4}, 0",
            "mov rdx, [{a} + 8]",
            "xor {r5:e}, {r5:e}",
            "mulx {h}, {l}, [{b}]",
            "adcx {r1}, {l}",
            "adox {r2}, {h}",
            "mulx {h}, {l}, [{b} + 8]",
            "adcx {r2}, {l}",
            "adox {r3}, {h}",
            "mulx {h}, {l}, [{b} + 16]",
            "adcx {r3}, {l}",
            "adox {r4}, {h}",
            "mulx {h}, {l}, [{b} + 24]",
            "adcx {r4}, {l}",
            "adox {r5}, {h}",
            "adc {r5}, 0",
            "mov rdx, [{a} + 16]",
            "xor {r6:e}, {r6:e}",
            "mulx {h}, {l}, [{b}]",
            "adcx {r2}, {l}",
            "adox {r3}, {h}",
            "mulx {h}, {l}, [{b} + 8]",
            "adcx {r3}, {l}",
            "adox {r4}, {h}",
            "mulx {h}, {l}, [{b} + 16]",
            "adcx {r4}, {l}",
            "adox {r5}, {h}",
            "mulx {h}, {l}, [{b} + 24]",
            "adcx {r5}, {l}",
            "adox {r6}, {h}",
            "adc {r6}, 0",
            "mov rdx, [{a} + 24]",
            "xor {r7:e}, {r7:e}",
            "mulx {h}, {l}, [{b}]",
            "adcx {r3}, {l}",
            "adox {r4}, {h}",
            "mulx {h}, {l}, [{b} + 8]",
            "adcx {r4}, {l}",
            "adox {r5}, {h}",
            "mulx {h}, {l}, [{b} + 16]",
            "adcx {r5}, {l}",
            "adox {r6}, {h}",
            "mulx {h}, {l}, [{b} + 24]",
            "adcx {r6}, {l}",
            "adox {r7}, {h}",
            "adc {r7}, 0",
            a = in(reg) a,
            b = in(reg) b,
            r0 = out(reg) r[0], r1 = out(reg) r[1], r2 = out(reg) r[2], r3 = out(reg) r[3],
            r4 = out(reg) r[4], r5 = out(reg) r[5], r6 = out(reg) r[6], r7 = out(reg) r[7],
            l = out(reg) _,
            h = out(reg) _,
            out("rdx") _,
            options(pure, readonly, nostack),
        );
        r
    }

    /// `dst[0..] += p << shift` for a 4- or 8-limb product held in
    /// registers: flagless shifts (`lea` joins the pieces) feed one `adc`
    /// chain over `M + 2` limbs, then the rare carry onward. A zero shift
    /// (one term in 64) takes the portable chain.
    #[inline(always)]
    unsafe fn chain<const M: usize>(dst: *mut u64, p: [u64; M], shift: u32) {
        if shift == 0 {
            return super::add_chain(dst, &p, 0);
        }
        let (s, r) = (shift as u64, 64 - shift as u64);
        if M == 4 {
            asm!(
                "shrx {t}, {p0}, {r}",
                "shlx {p0}, {p0}, {s}",
                "add [{d}], {p0}",
                "shrx {u}, {p1}, {r}",
                "shlx {p1}, {p1}, {s}",
                "lea {p1}, [{p1} + {t}]",
                "adc [{d} + 8], {p1}",
                "shrx {t}, {p2}, {r}",
                "shlx {p2}, {p2}, {s}",
                "lea {p2}, [{p2} + {u}]",
                "adc [{d} + 16], {p2}",
                "shrx {u}, {p3}, {r}",
                "shlx {p3}, {p3}, {s}",
                "lea {p3}, [{p3} + {t}]",
                "adc [{d} + 24], {p3}",
                "adc [{d} + 32], {u}",
                "adc qword ptr [{d} + 40], 0",
                "jnc 3f",
                "2:",
                "lea {d}, [{d} + 8]",
                "adc qword ptr [{d} + 40], 0",
                "jc 2b",
                "3:",
                d = inout(reg) dst => _,
                p0 = inout(reg) p[0] => _, p1 = inout(reg) p[1] => _,
                p2 = inout(reg) p[2] => _, p3 = inout(reg) p[3] => _,
                s = in(reg) s,
                r = in(reg) r,
                t = out(reg) _,
                u = out(reg) _,
                options(nostack),
            );
        } else {
            asm!(
                "shrx {t}, {p0}, {r}",
                "shlx {p0}, {p0}, {s}",
                "add [{d}], {p0}",
                "shrx {u}, {p1}, {r}",
                "shlx {p1}, {p1}, {s}",
                "lea {p1}, [{p1} + {t}]",
                "adc [{d} + 8], {p1}",
                "shrx {t}, {p2}, {r}",
                "shlx {p2}, {p2}, {s}",
                "lea {p2}, [{p2} + {u}]",
                "adc [{d} + 16], {p2}",
                "shrx {u}, {p3}, {r}",
                "shlx {p3}, {p3}, {s}",
                "lea {p3}, [{p3} + {t}]",
                "adc [{d} + 24], {p3}",
                "shrx {t}, {p4}, {r}",
                "shlx {p4}, {p4}, {s}",
                "lea {p4}, [{p4} + {u}]",
                "adc [{d} + 32], {p4}",
                "shrx {u}, {p5}, {r}",
                "shlx {p5}, {p5}, {s}",
                "lea {p5}, [{p5} + {t}]",
                "adc [{d} + 40], {p5}",
                "shrx {t}, {p6}, {r}",
                "shlx {p6}, {p6}, {s}",
                "lea {p6}, [{p6} + {u}]",
                "adc [{d} + 48], {p6}",
                "shrx {u}, {p7}, {r}",
                "shlx {p7}, {p7}, {s}",
                "lea {p7}, [{p7} + {t}]",
                "adc [{d} + 56], {p7}",
                "adc [{d} + 64], {u}",
                "adc qword ptr [{d} + 72], 0",
                "jnc 3f",
                "2:",
                "lea {d}, [{d} + 8]",
                "adc qword ptr [{d} + 72], 0",
                "jc 2b",
                "3:",
                d = inout(reg) dst => _,
                p0 = inout(reg) p[0] => _, p1 = inout(reg) p[1] => _,
                p2 = inout(reg) p[2] => _, p3 = inout(reg) p[3] => _,
                p4 = inout(reg) p[4] => _, p5 = inout(reg) p[5] => _,
                p6 = inout(reg) p[6] => _, p7 = inout(reg) p[7] => _,
                s = in(reg) s,
                r = in(reg) r,
                t = out(reg) _,
                u = out(reg) _,
                options(nostack),
            );
        }
    }

    /// `dst[0..] += p << shift` for any length: per limb one flagless shift
    /// pair and one `adc` to memory (`lea` and `dec` keep the carry).
    #[inline(always)]
    unsafe fn chain_limbs(dst: *mut u64, p: &[u64], shift: u32) {
        if shift == 0 {
            return super::add_chain(dst, p, 0);
        }
        let (s, r) = (shift as u64, 64 - shift as u64);
        asm!(
            "xor {below:e}, {below:e}",
            "2:",
            "mov {x}, [{p}]",
            "shrx {hi}, {x}, {r}",
            "shlx {x}, {x}, {s}",
            "lea {x}, [{x} + {below}]",
            "mov {below}, {hi}",
            "adc [{d}], {x}",
            "lea {p}, [{p} + 8]",
            "lea {d}, [{d} + 8]",
            "dec {n}",
            "jnz 2b",
            "adc [{d}], {below}",
            "3:",
            "lea {d}, [{d} + 8]",
            "adc qword ptr [{d}], 0",
            "jc 3b",
            d = inout(reg) dst => _,
            p = inout(reg) p.as_ptr() => _,
            n = inout(reg) p.len() => _,
            s = in(reg) s,
            r = in(reg) r,
            x = out(reg) _,
            hi = out(reg) _,
            below = out(reg) _,
            options(nostack),
        );
    }
}

/// Accumulator limbs for exponent sums in `[emin, emax]`, if supported.
/// Bit 0 of the accumulator weighs 2^(emin - 2P). A term with exponent sum
/// e lands at bit offset e - emin; its 2N-limb product plus one shift limb
/// ends at or below limb (emax - emin)/64 + 2N + 1. One further limb absorbs
/// the carry growth of up to 2^64 terms.
/// [`dot_slices`] over `chunks` contiguous pieces whose exact partial sums
/// `par_for(chunks, f)` may compute concurrently (calling `f(k)` once for
/// every `k < chunks`). The partial sums share one alignment and add exactly,
/// so the value is the same as `dot_slices`.
pub(crate) fn dot_slices_chunked<'a, const N: usize>(
    a: &'a [MpFloat<N>],
    b: &'a [MpFloat<N>],
    chunks: usize,
    par_for: &dyn Fn(usize, &(dyn Fn(usize) + Sync)),
    chain: impl Fn(&mut dyn Iterator<Item = Pair<'a, N>>) -> MpFloat<N>,
) -> MpFloat<N> {
    assert_eq!(a.len(), b.len());
    let chunks = chunks.min(a.len()).max(1);
    if chunks < 2 || N > MAX_N {
        return dot_slices(a, b, chain);
    }
    let size = a.len().div_ceil(chunks);
    let chunks = a.len().div_ceil(size);
    let regular = |k: i32| k.abs() == mpfr::REGULAR_KIND;
    let zero = |k: i32| k.abs() == mpfr::ZERO_KIND;
    let scans = std::sync::Mutex::new(vec![None; chunks]);
    par_for(chunks, &|k| {
        let (mut emin, mut emax, mut finite) = (i64::MAX, i64::MIN, true);
        let r = k * size..((k + 1) * size).min(a.len());
        for (x, y) in a[r.clone()].iter().zip(&b[r]) {
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
        scans.lock().unwrap()[k] = Some((emin, emax, finite));
    });
    let (emin, emax, finite) = scans.into_inner().unwrap().into_iter().fold(
        (i64::MAX, i64::MIN, true),
        |(a, b, c), scan| {
            let (x, y, z) = scan.expect("chunk scheduler skipped exponent scan");
            (a.min(x), b.max(y), c && z)
        },
    );
    let len = match (finite, emin == i64::MAX) {
        (_, true) if finite => return MpFloat::<N>::default(),
        (true, false) => acc_len::<N>(emin, emax),
        _ => None,
    };
    let Some(len) = len else {
        return dot_slices(a, b, chain);
    };
    let parts = std::sync::Mutex::new(vec![None; chunks]);
    par_for(chunks, &|k| {
        let (mut pos, mut neg) = (vec![0u128; len], vec![0u128; len]);
        let r = k * size..((k + 1) * size).min(a.len());
        let terms = a[r.clone()]
            .iter()
            .zip(&b[r])
            .filter(|(x, y)| regular(x.kind) && regular(y.kind));
        add_terms(&mut pos, &mut neg, emin, len, N <= INLINE_N, terms);
        parts.lock().unwrap()[k] = Some((pos, neg));
    });
    let mut parts = parts
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|part| part.expect("chunk scheduler skipped accumulation"));
    let (mut pos, mut neg) = parts.next().unwrap();
    for (p, n) in parts {
        add_slots(&mut pos, &p);
        add_slots(&mut neg, &n);
    }
    round_slots(&mut pos, &mut neg, emin).unwrap_or_else(|| dot_slices(a, b, chain))
}

fn acc_len<const N: usize>(emin: i64, emax: i64) -> Option<usize> {
    let len = usize::try_from(emax - emin).ok()? / 64 + 2 * N + 3;
    (len <= MAX_ACC_LIMBS).then_some(len)
}

fn accumulate<'a, const N: usize>(
    pos: &mut Vec<u128>,
    neg: &mut Vec<u128>,
    emin: i64,
    emax: i64,
    inline: bool,
    terms: impl Iterator<Item = Pair<'a, N>> + Clone,
) -> Option<MpFloat<N>> {
    let len = acc_len::<N>(emin, emax)?;
    for acc in [&mut *pos, &mut *neg] {
        acc.clear();
        acc.resize(len, 0);
    }
    add_terms(pos, neg, emin, len, inline, terms);
    round_slots(pos, neg, emin)
}

/// Add every term's exact product into the `len`-slot accumulators.
fn add_terms<'a, const N: usize>(
    pos: &mut [u128],
    neg: &mut [u128],
    emin: i64,
    len: usize,
    inline: bool,
    terms: impl Iterator<Item = Pair<'a, N>> + Clone,
) {
    debug_assert!(pos.len() >= len && neg.len() >= len);
    let accs = [pos.as_mut_ptr(), neg.as_mut_ptr()];
    // One product buffer per dot; every term overwrites its limbs.
    let mut scratch = [0u64; 2 * MAX_N];
    for term in terms {
        // SAFETY: both accumulators hold `len` slots and are distinct.
        unsafe { add_term::<N>(accs, emin, len, inline, term, &mut scratch) };
    }
}

/// Add one regular term's exact product into `accs[0]` (positive) or
/// `accs[1]` (negative), by the product's sign.
///
/// # Safety
/// Both pointers address distinct accumulators of `len` slots whose bit 0
/// weighs 2^(emin - 2P), with `len` from [`acc_len`] over a range holding
/// this term's exponent sum.
#[inline(always)]
unsafe fn add_term<const N: usize>(
    accs: [*mut u128; 2],
    emin: i64,
    len: usize,
    inline: bool,
    (a, b): Pair<'_, N>,
    scratch: &mut [u64; 2 * MAX_N],
) {
    let offset = (a.exponent as i64 + b.exponent as i64 - emin) as usize;
    // The accumulator is picked by the product's sign bit, not a branch:
    // signs of consecutive terms are data-dependent and mispredict.
    let negative = ((a.kind < 0) != (b.kind < 0)) as usize;
    let (limb, shift) = (offset / 64, (offset % 64) as u32);
    debug_assert!(limb + 2 * N < len);
    // SAFETY: `limb < len`; the bound on `len` covers the shifted product.
    let dst = accs[negative].add(limb);
    if inline && N <= INLINE_N {
        // A local product buffer stays in registers.
        let mut prod = [0u64; 2 * INLINE_N];
        mul_limbs::<N>(&a.limbs, &b.limbs, &mut prod[..2 * N]);
        add_shifted(dst, &prod[..2 * N], shift);
        return;
    }
    // SAFETY: `scratch` holds 2N <= 2*MAX_N limbs and does not overlap the
    // inputs.
    mpn_mul_n::<N>(scratch.as_mut_ptr(), a.limbs.as_ptr(), b.limbs.as_ptr());
    add_shifted(dst, &scratch[..2 * N], shift);
}

/// `dst[0..=n] += limbs << shift` for `n` limbs and `shift < 64`, one 64-bit
/// piece per 128-bit slot: no carry chain, so consecutive terms overlap.
/// A slot absorbs 2^64 pieces before it could overflow.
///
/// # Safety
/// `dst` addresses `limbs.len() + 1` slots.
#[inline(always)]
unsafe fn add_shifted(dst: *mut u128, limbs: &[u64], shift: u32) {
    // `(x >> 1) >> (63 - shift)` is `x >> (64 - shift)`, and 0 at shift 0.
    let mut below = 0u64;
    for (t, &l) in limbs.iter().enumerate() {
        *dst.add(t) += ((l << shift) | ((below >> 1) >> (63 - shift))) as u128;
        below = l;
    }
    *dst.add(limbs.len()) += ((below >> 1) >> (63 - shift)) as u128;
}

/// `acc += other`, slot by slot. Each slot stays below 2^128: it sums at most
/// one 64-bit piece per term.
fn add_slots(acc: &mut [u128], other: &[u128]) {
    for (x, &y) in acc.iter_mut().zip(other) {
        *x += y;
    }
}

/// Resolve the carries of `slots` into its exact little-endian 64-bit limbs,
/// written in place over the first `slots.len()` limbs of the same memory.
fn fold_slots(slots: &mut [u128]) -> &mut [u64] {
    let len = slots.len();
    let (src, limbs) = (slots.as_mut_ptr(), slots.as_mut_ptr() as *mut u64);
    let mut carry = 0u128;
    for t in 0..len {
        // SAFETY: slot t is read before limb t is written; limb t occupies
        // bytes [8t, 8t + 8) of slot t/2 <= t, which was read already.
        unsafe {
            let x = src.add(t).read() + carry;
            limbs.add(t).write(x as u64);
            carry = x >> 64;
        }
    }
    // The bound on the accumulator length keeps the total inside it.
    debug_assert_eq!(carry, 0);
    // SAFETY: the `len` limbs lie inside the slots' storage and were written.
    unsafe { std::slice::from_raw_parts_mut(limbs, len) }
}

/// [`round_acc`] of `pos - neg` held in slots of equal length.
fn round_slots<const N: usize>(
    pos: &mut [u128],
    neg: &mut [u128],
    emin: i64,
) -> Option<MpFloat<N>> {
    let len = pos.len();
    debug_assert_eq!(neg.len(), len);
    round_acc(fold_slots(pos), fold_slots(neg), len, emin)
}

/// Exact sums of `outputs` dot products whose terms `terms(c)` yields as
/// `(output, a, b)` for chunks `c < chunks`; `par_for(chunks, f)` may run
/// the chunks concurrently (calling `f(c)` once for every `c`). An output's
/// partial sums share one alignment and add exactly, so each value is `dot`
/// over that output's terms in any order. Non-finite outputs remain `None`;
/// finite outputs outside the limb window use the MPFR sum path.
///
/// `bounds[k]`, when given, is a range holding the exponent sum of every
/// finite nonzero term of output k (empty when it has none). The terms are
/// then read once, without an exponent scan; a term outside its range, or a
/// non-finite one, makes the call scan after all. The alignment never
/// changes a value, so the result is the same either way.
pub(crate) fn dot_outputs_chunked<'a, const N: usize, I>(
    outputs: usize,
    chunks: usize,
    bounds: Option<&[(i64, i64)]>,
    terms: &(dyn Fn(usize) -> I + Sync),
    par_for: &dyn Fn(usize, &(dyn Fn(usize) + Sync)),
) -> Vec<Option<MpFloat<N>>>
where
    I: Iterator<Item = (usize, &'a MpFloat<N>, &'a MpFloat<N>)>,
{
    let chunks = chunks.max(1);
    if N > MAX_N {
        return mpfr_outputs(vec![None; outputs], chunks, terms);
    }
    if let Some(bounds) = bounds {
        assert_eq!(bounds.len(), outputs);
        let range: Vec<_> = bounds.iter().map(|&(lo, hi)| (lo, hi, true)).collect();
        if let Some(out) = sum_outputs::<N, I, true>(&range, chunks, terms, par_for) {
            return out;
        }
    }
    let regular = |k: i32| k.abs() == mpfr::REGULAR_KIND;
    let scans = std::sync::Mutex::new(vec![None; chunks]);
    par_for(chunks, &|c| {
        let mut scan = vec![(i64::MAX, i64::MIN, true); outputs];
        for (k, x, y) in terms(c) {
            let (emin, emax, finite) = &mut scan[k];
            if regular(x.kind) && regular(y.kind) {
                let e = x.exponent as i64 + y.exponent as i64;
                *emin = (*emin).min(e);
                *emax = (*emax).max(e);
            } else if !zero_product(x, y) {
                *finite = false;
            }
        }
        scans.lock().unwrap()[c] = Some(scan);
    });
    let mut range = vec![(i64::MAX, i64::MIN, true); outputs];
    for scan in scans.into_inner().unwrap() {
        let scan = scan.expect("chunk scheduler skipped exponent scan");
        for (r, s) in range.iter_mut().zip(scan) {
            *r = (r.0.min(s.0), r.1.max(s.1), r.2 && s.2);
        }
    }
    let out = sum_outputs::<N, I, false>(&range, chunks, terms, par_for)
        .expect("scanned exponent ranges hold every term");
    if out.iter().zip(&range).any(|(v, r)| v.is_none() && r.2) {
        mpfr_outputs(out, chunks, terms)
    } else {
        out
    }
}

/// Fill declined finite outputs together, reading the chunk streams once.
fn mpfr_outputs<'a, const N: usize, I>(
    mut out: Vec<Option<MpFloat<N>>>,
    chunks: usize,
    terms: &(dyn Fn(usize) -> I + Sync),
) -> Vec<Option<MpFloat<N>>>
where
    I: Iterator<Item = (usize, &'a MpFloat<N>, &'a MpFloat<N>)>,
{
    let mut pending: Vec<_> = out.iter().map(|v| v.is_none().then(Vec::new)).collect();
    for c in 0..chunks {
        for (k, a, b) in terms(c) {
            if let Some(pairs) = &mut pending[k] {
                pairs.push((a, b));
            }
        }
    }
    let finite = |x: &MpFloat<N>| matches!(x.kind.abs(), mpfr::ZERO_KIND | mpfr::REGULAR_KIND);
    for (value, pairs) in out.iter_mut().zip(pending) {
        if let Some(pairs) = pairs {
            if pairs.iter().all(|(a, b)| finite(a) && finite(b)) {
                *value = Some(dot_mpfr(pairs, |_| unreachable!("finite output")));
            }
        }
    }
    out
}

/// Whether `a·b` is an exact zero of finite factors, which adds nothing
/// (the sign of a zero result is not significant to the solver).
#[inline(always)]
fn zero_product<const N: usize>(a: &MpFloat<N>, b: &MpFloat<N>) -> bool {
    let regular = |k: i32| k.abs() == mpfr::REGULAR_KIND;
    let zero = |k: i32| k.abs() == mpfr::ZERO_KIND;
    zero(a.kind) && (zero(b.kind) || regular(b.kind)) || zero(b.kind) && regular(a.kind)
}

/// The accumulation pass of [`dot_outputs_chunked`] over per-output
/// exponent-sum ranges `(emin, emax, finite)`. `GUARD` checks every term
/// against its range and returns `None` on a miss; without it the ranges
/// come from a scan of these very terms.
fn sum_outputs<'a, const N: usize, I, const GUARD: bool>(
    range: &[(i64, i64, bool)],
    chunks: usize,
    terms: &(dyn Fn(usize) -> I + Sync),
    par_for: &dyn Fn(usize, &(dyn Fn(usize) + Sync)),
) -> Option<Vec<Option<MpFloat<N>>>>
where
    I: Iterator<Item = (usize, &'a MpFloat<N>, &'a MpFloat<N>)>,
{
    let regular = |k: i32| k.abs() == mpfr::REGULAR_KIND;
    let mut out = vec![None; range.len()];
    // Output k accumulates in `len` slots from `start` at bit 0 weighing
    // 2^(emin - 2P); none when it declines or holds no nonzero product.
    let mut total = 0;
    let mut slots = Vec::with_capacity(range.len());
    for (k, &(emin, emax, finite)) in range.iter().enumerate() {
        let len = match (finite, emin > emax) {
            (true, true) => {
                out[k] = Some(MpFloat::<N>::default());
                0
            }
            (true, false) => match acc_len::<N>(emin, emax) {
                Some(len) => len,
                // Too wide a bound: the scan may find a narrower range.
                None if GUARD => return None,
                None => 0,
            },
            _ => 0,
        };
        total += len;
        slots.push((total - len, len, emin, emax));
    }
    if total == 0 && !GUARD {
        return Some(out);
    }
    let missed = std::sync::atomic::AtomicBool::new(false);
    let parts = std::sync::Mutex::new(vec![None; chunks]);
    par_for(chunks, &|c| {
        let (mut pos, mut neg) = (vec![0u128; total], vec![0u128; total]);
        let mut scratch = [0u64; 2 * MAX_N];
        let mut miss = false;
        for (k, x, y) in terms(c) {
            let (s, len, emin, emax) = slots[k];
            if !(regular(x.kind) && regular(y.kind)) {
                miss |= GUARD && !zero_product(x, y);
                continue;
            }
            if GUARD {
                let e = x.exponent as i64 + y.exponent as i64;
                if e < emin || e > emax {
                    miss = true;
                    continue;
                }
            } else if len == 0 {
                continue;
            }
            // SAFETY: output k owns `len` slots at `s` in both accumulators,
            // sized by `acc_len` over a range holding this term's exponent
            // sum (checked above, or scanned from these terms).
            unsafe {
                let accs = [pos.as_mut_ptr().add(s), neg.as_mut_ptr().add(s)];
                add_term::<N>(accs, emin, len, N <= INLINE_N, (x, y), &mut scratch);
            }
        }
        if miss {
            missed.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        parts.lock().unwrap()[c] = Some((pos, neg));
    });
    let mut parts = parts
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|part| part.expect("chunk scheduler skipped accumulation"));
    if missed.into_inner() {
        return None;
    }
    let (mut pos, mut neg) = parts.next().unwrap();
    for (p, n) in parts {
        // Slots carry nothing into the next output's slots.
        add_slots(&mut pos, &p);
        add_slots(&mut neg, &n);
    }
    for (k, &(s, len, emin, _)) in slots.iter().enumerate() {
        if len > 0 {
            out[k] = round_slots(&mut pos[s..s + len], &mut neg[s..s + len], emin);
        }
    }
    Some(out)
}

/// Round `pos - neg` (bit 0 weighing 2^(emin - 2P)) once, nearest-even.
fn round_acc<const N: usize>(
    pos: &mut [u64],
    neg: &mut [u64],
    len: usize,
    emin: i64,
) -> Option<MpFloat<N>> {
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
/// through 18 limbs. On aarch64, GMP's `addmul_1` rows (one call per limb
/// of `up`) run at about half the multiplier throughput; [`mul_rows`] takes
/// those widths. The product is exact either way.
///
/// # Safety
/// `rp` holds `2N` limbs and does not overlap the `N`-limb inputs.
#[inline(always)]
pub(crate) unsafe fn mpn_mul_n<const N: usize>(rp: *mut u64, up: *const u64, vp: *const u64) {
    #[cfg(target_arch = "aarch64")]
    if N <= ROWS_N {
        return mul_rows::<N>(rp, up, vp);
    }
    #[cfg(target_arch = "x86_64")]
    if N < 19 {
        extern "C" {
            fn __gmpn_mul_basecase(
                rp: *mut u64,
                up: *const u64,
                un: gmp::size_t,
                vp: *const u64,
                vn: gmp::size_t,
            );
        }
        return __gmpn_mul_basecase(rp, up, N as _, vp, N as _);
    }
    gmp::mpn_mul_n(rp, up, vp, N as _);
}

/// `a + b + c` and its carry.
#[inline(always)]
fn adc(a: u64, b: u64, c: bool) -> (u64, bool) {
    let (s, o1) = a.overflowing_add(b);
    let (s, o2) = s.overflowing_add(c as u64);
    (s, o1 | o2)
}

/// `rp[..2N] = up * vp` for `N <= ROWS_N` limbs, unrolled for constant `N`:
/// each row's `N` products are formed first, then their low and high halves
/// go in by two separate carry chains, so the multiplies of a row overlap
/// instead of waiting on one chain as in GMP's `addmul_1`.
///
/// # Safety
/// As [`mpn_mul_n`].
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
#[inline(always)]
unsafe fn mul_rows<const N: usize>(rp: *mut u64, up: *const u64, vp: *const u64) {
    debug_assert!(N <= ROWS_N);
    let (a, b) = (&*(up as *const [u64; N]), &*(vp as *const [u64; N]));
    let mut r = [0u64; 2 * ROWS_N];
    for i in 0..N {
        let (mut lo, mut hi) = ([0u64; N], [0u64; N]);
        for j in 0..N {
            let p = a[i] as u128 * b[j] as u128;
            (lo[j], hi[j]) = (p as u64, (p >> 64) as u64);
        }
        // Rows 0..i fill limbs 0..i+N, so limb i+N takes the low carry and
        // the high chain ends without one.
        let mut c = false;
        for j in 0..N {
            (r[i + j], c) = adc(r[i + j], lo[j], c);
        }
        r[i + N] = c as u64;
        let mut c = false;
        for j in 0..N {
            (r[i + 1 + j], c) = adc(r[i + 1 + j], hi[j], c);
        }
        debug_assert!(!c);
    }
    std::ptr::copy_nonoverlapping(r.as_ptr(), rp, 2 * N);
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
    use gmp_mpfr_sys::mpfr;
    use num_traits::{FromPrimitive, One, Zero};

    // `mul_rows` against GMP over all-ones, one-hot and xorshift operands.
    fn mul_rows_case<const N: usize>(seed: u64) {
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for case in 0..200 {
            let (mut a, mut b) = ([0u64; N], [0u64; N]);
            for j in 0..N {
                (a[j], b[j]) = match case {
                    0 => (u64::MAX, u64::MAX),
                    1 => (u64::MAX, (j == N - 1) as u64),
                    2 => ((j == 0) as u64 * u64::MAX, u64::MAX),
                    _ => (next(), next()),
                };
            }
            let (mut got, mut want) = ([0u64; 2 * super::MAX_N], [0u64; 2 * super::MAX_N]);
            unsafe {
                super::mul_rows::<N>(got.as_mut_ptr(), a.as_ptr(), b.as_ptr());
                gmp_mpfr_sys::gmp::mpn_mul_n(want.as_mut_ptr(), a.as_ptr(), b.as_ptr(), N as _);
            }
            assert_eq!(got[..2 * N], want[..2 * N], "N={N} case={case}");
        }
    }

    #[test]
    fn mul_rows_matches_gmp() {
        mul_rows_case::<1>(1);
        mul_rows_case::<2>(2);
        mul_rows_case::<3>(3);
        mul_rows_case::<4>(4);
        mul_rows_case::<5>(5);
        mul_rows_case::<6>(6);
        mul_rows_case::<7>(7);
        mul_rows_case::<8>(8);
        mul_rows_case::<9>(9);
        mul_rows_case::<10>(10);
        mul_rows_case::<11>(11);
        mul_rows_case::<12>(12);
    }

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
        // The exact slow path also preserves a negligible low term.
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

    fn wide_dot<const N: usize>() {
        use crate::Scalar;
        let one = MpFloat::<N>::one();
        let huge = one.scale_pow2(40_000);
        let p = MpFloat::<N>::PRECISION_BITS as i64;
        let run = |a: &[MpFloat<N>], b: &[MpFloat<N>], expected| {
            let chain = |_: &mut dyn Iterator<Item = super::Pair<'_, N>>| {
                panic!("finite wide dot used the rounded chain")
            };
            assert_eq!(super::dot(a.iter().zip(b), chain), expected, "N={N}");
            assert_eq!(
                super::dot(a.iter().rev().zip(b.iter().rev()), chain),
                expected
            );
            assert_eq!(super::dot_slices(a, b, chain), expected);
            let par_for = |n: usize, f: &(dyn Fn(usize) + Sync)| (0..n).for_each(f);
            assert_eq!(
                <MpFloat<N> as Scalar>::dot_slices_chunked(a, b, 2, &par_for),
                expected
            );
            let terms = |c| {
                a.iter()
                    .zip(b)
                    .enumerate()
                    .filter(move |(i, _)| i % 2 == c)
                    .map(|(_, (a, b))| (0, a, b))
            };
            let sum = <MpFloat<N> as Scalar>::dot_outputs_chunked(1, 2, None, &terms, &par_for)[0]
                .expect("finite output outside the limb window");
            assert_eq!(sum, expected);
        };
        run(&[huge, one, -huge], &[one; 3], one);
        let three_halves = one + one.scale_pow2(-1);
        run(
            &[huge, three_halves, -huge],
            &[one, three_halves, one],
            one.scale_pow2(1) + one.scale_pow2(-2),
        );
        // Ties round to even; a lower sticky bit takes the next value.
        let half = one.scale_pow2(-p);
        run(&[huge, one, half, -huge], &[one; 4], one);
        run(
            &[huge, one, half, half.scale_pow2(-1), -huge],
            &[one; 5],
            one + one.scale_pow2(1 - p),
        );
        // The product itself must remain exact before cancellation.
        let eps = one.scale_pow2(1 - p);
        run(
            &[huge, one + eps, -one, -huge],
            &[one, one - eps, one, one],
            -one.scale_pow2(2 - 2 * p),
        );
        // Product exponents outside the active MPFR range must survive until
        // cancellation or combination, without changing that thread's range.
        let (emin, emax) = unsafe { (mpfr::get_emin(), mpfr::get_emax()) };
        let large = one.scale_pow2(emax as i64 - 2);
        let four = one.scale_pow2(2);
        run(&[large, one, -large], &[four, one, four], one);
        let tiny = one.scale_pow2(emin as i64 - 1);
        let half = one.scale_pow2(-1);
        run(&[huge, tiny, tiny, -huge], &[one, half, half, one], tiny);
        run(
            &[huge, tiny, -huge],
            &[one, half, one],
            MpFloat::<N>::zero(),
        );
        run(
            &[huge, tiny, -huge],
            &[one, half + half.scale_pow2(-1), one],
            tiny,
        );
        assert_eq!(
            unsafe { (mpfr::get_emin(), mpfr::get_emax()) },
            (emin, emax)
        );
    }

    #[test]
    fn wide_exponent_dot_keeps_exact_cancellation_and_rounding() {
        macro_rules! widths {
            ($($n:literal),*) => { $(wide_dot::<$n>();)* };
        }
        widths!(
            2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 32
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
            let slices = |a: &[MpFloat<N>], b: &[MpFloat<N>]| {
                super::dot_slices(a, b, |t| MpFloat::<N>::dot_fma_chain(t))
            };
            assert_eq!(
                run(&a, &b, true),
                run(&a, &b, false),
                "N={N} carry {shifts}"
            );
            assert_eq!(slices(&a, &b), run(&a, &b, false), "N={N} slices {shifts}");
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
            assert_eq!(
                slices(&a, &mixed),
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
        inline_matches_gmp::<8>();
        inline_matches_gmp::<16>();
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
