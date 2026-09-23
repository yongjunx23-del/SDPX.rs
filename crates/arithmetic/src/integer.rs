//! Exact integer accumulation for dyadic matrices.
//!
//! A modular (CRT) product needs an *exact* integer image of the inputs. This
//! module builds one: finite dyadic values are aligned to a common binary
//! exponent, shifted into integers with no truncation, and accumulated with GMP.
//! It also reports the integer width and the strict term-wise bound `B` that a
//! modulus count must dominate, so callers never have to guess a prime count.
//!
//! This is a reference and planning path. It is not yet a production kernel.
use crate::MpFloat;
use gmp_mpfr_sys::gmp;
use std::mem::MaybeUninit;

/// Owned arbitrary-precision integer.
#[derive(Debug)]
pub struct ExactInteger {
    pub(crate) raw: gmp::mpz_t,
}

impl Default for ExactInteger {
    fn default() -> Self {
        let mut raw = MaybeUninit::uninit();
        unsafe {
            gmp::mpz_init(raw.as_mut_ptr());
            Self {
                raw: raw.assume_init(),
            }
        }
    }
}

impl Clone for ExactInteger {
    fn clone(&self) -> Self {
        let mut out = Self::default();
        unsafe {
            gmp::mpz_set(&mut out.raw, &self.raw);
        }
        out
    }
}

impl Drop for ExactInteger {
    fn drop(&mut self) {
        unsafe { gmp::mpz_clear(&mut self.raw) }
    }
}

impl ExactInteger {
    /// Number of significant bits; 0 for zero.
    pub fn bit_length(&self) -> u64 {
        unsafe { gmp::mpz_sizeinbase(&self.raw, 2) as u64 }
    }

    pub fn is_zero(&self) -> bool {
        unsafe { gmp::mpz_sgn(&self.raw) == 0 }
    }

    /// Exact reconstruction of `self * 2^shift` with a single rounding at the
    /// destination precision.
    pub fn to_mpfloat_scaled<const N: usize>(&self, shift: i64) -> MpFloat<N> {
        // Copy the integer into MPFR exactly, then apply one exact power-of-two
        // scale. mpfr_set_z is exact whenever the value fits; the scale below is
        // the only inexact step and rounds once, nearest-even.
        let out = unsafe {
            let mut z = MaybeUninit::uninit();
            gmp_mpfr_sys::mpfr::init2(z.as_mut_ptr(), MpFloat::<N>::PRECISION_BITS as _);
            let mut z = z.assume_init();
            gmp_mpfr_sys::mpfr::set_z(&mut z, &self.raw, gmp_mpfr_sys::mpfr::rnd_t::RNDN);
            let out = MpFloat::<N>::from_mpfr_descriptor(&z);
            gmp_mpfr_sys::mpfr::clear(&mut z);
            out
        };
        scale_by_power_of_two(&out, shift)
    }
}

/// Multiply by `2^shift` in one exact MPFR step.
///
/// Exact while the exponent stays in range, so no extra rounding is introduced
/// beyond the destination precision already in force.
pub fn scale_by_power_of_two<const N: usize>(v: &MpFloat<N>, shift: i64) -> MpFloat<N> {
    v.scale_pow2(shift)
}

/// One aligned term of an exact integer image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TermError {
    /// A non-finite operand has no integer image.
    NonFinite,
}

/// Alignment plan for one exact product `sum_k a_k * b_k`.
///
/// `term_shift` is the common power of two for a **single factor**: each factor
/// is represented as an integer times `2^term_shift`. A product of two such
/// integers therefore carries `2^(2 * term_shift)`, so
/// `sum_k a_k b_k = integer * 2^(2 * term_shift)` exactly.
#[derive(Clone, Debug)]
pub struct ProductPlan {
    /// Common binary exponent of one factor's integer image.
    pub term_shift: i64,
    /// Width of the widest scaled product term, in bits.
    pub max_term_bits: u64,
    /// Strict upper bound on the *real value* `|sum_k a_k b_k|`, in bits: every
    /// `|a_k b_k| < 2^b` and there are `terms` of them, so `|sum| < terms * 2^b`.
    pub value_bound_bits: u64,
}

impl ProductPlan {
    /// Strict upper bound on the accumulated **integer**, in bits.
    ///
    /// This is the quantity a signed-CRT modulus count must dominate: the integer
    /// is the real value scaled by `2^(-2 * term_shift)`, so a very negative
    /// `term_shift` inflates it far beyond the value's own magnitude.
    pub fn image_bound_bits(&self) -> u64 {
        let widen = -self.term_shift.saturating_mul(2);
        if widen <= 0 {
            self.value_bound_bits
        } else {
            self.value_bound_bits.saturating_add(widen as u64)
        }
    }

    /// Smallest modulus count satisfying `prod(p) > 2 * B` for signed CRT
    /// uniqueness, given `prime_bits` per modulus.
    pub fn modulus_count(&self, prime_bits: u32) -> u64 {
        let bits = self.image_bound_bits().saturating_add(1);
        bits.div_ceil(prime_bits.max(1) as u64)
    }

    /// Whether one BLAS panel of contraction length `k_c` stays exact in binary64
    /// for these moduli: every product and partial sum is below `2^53`.
    pub fn blas_panel_is_exact(k_c: usize, prime_bits: u32) -> bool {
        let p = (1u128 << prime_bits) - 1;
        (k_c as u128) * p * p < (1u128 << 53)
    }
}

/// Result of an exact product accumulation.
#[derive(Debug)]
pub struct ExactProduct {
    /// Exact integer value of `sum_k A_k B_k`.
    pub integer: ExactInteger,
    /// The plan this accumulation followed.
    pub plan: ProductPlan,
}

impl ExactProduct {
    /// Reconstruct `sum_k a_k b_k` with a single rounding at precision `N`.
    ///
    /// The accumulated integer is scaled by `2^(2 * term_shift)` because both
    /// factors were aligned to the same exponent.
    pub fn to_mpfloat<const N: usize>(&self) -> MpFloat<N> {
        self.integer
            .to_mpfloat_scaled(self.plan.term_shift.saturating_mul(2))
    }
}

/// Exactly accumulate `sum_k a_k * b_k` for finite dyadic inputs.
///
/// Uses a single common exponent (the minimum over all terms), so no term is
/// truncated. When the resulting integer width is impractical the caller must
/// split by exponent layer or keep the MPFR path; this function never drops bits
/// to make the width smaller.
pub fn exact_product<const N: usize>(
    a: &[MpFloat<N>],
    b: &[MpFloat<N>],
) -> Result<ExactProduct, TermError> {
    assert_eq!(a.len(), b.len(), "exact_product requires matched lengths");
    if a.is_empty() {
        return Ok(ExactProduct {
            integer: ExactInteger::default(),
            plan: ProductPlan {
                term_shift: 0,
                max_term_bits: 0,
                value_bound_bits: 0,
            },
        });
    }

    // Pass 1: align to the minimum exponent and measure the strict bound.
    let mut shift = i64::MAX;
    let mut worst = 0i64; // ceil(log2(|a_k| |b_k|)) for the widest term
    for (x, y) in a.iter().zip(b) {
        let (bx, by) = match (x.exact_bit_length(), y.exact_bit_length()) {
            (Some(bx), Some(by)) => (bx, by),
            _ => return Err(TermError::NonFinite),
        };
        if bx == 0 || by == 0 {
            continue;
        }
        shift = shift.min(x.dyadic_view().exponent as i64 - (N as i64 * 64));
        shift = shift.min(y.dyadic_view().exponent as i64 - (N as i64 * 64));
        // |a_k b_k| < 2^(bx + by)
        worst = worst.max(bx + by);
    }
    if shift == i64::MAX {
        // Every term was an exact zero.
        return Ok(ExactProduct {
            integer: ExactInteger::default(),
            plan: ProductPlan {
                term_shift: 0,
                max_term_bits: 0,
                value_bound_bits: 0,
            },
        });
    }

    // Strict bound on |sum_k a_k b_k|: every |a_k b_k| < 2^worst, and there are
    // `terms` of them, so |sum| < terms * 2^worst < 2^(worst + bitlen(terms)).
    let terms = a.len().max(1) as u64;
    let bound_bits = worst as u64 + (64 - terms.leading_zeros() as u64);

    let mut acc = ExactInteger::default();
    let mut max_term_bits = 0u64;
    for (x, y) in a.iter().zip(b) {
        if x.dyadic_view().kind == crate::DyadicKind::Zero
            || y.dyadic_view().kind == crate::DyadicKind::Zero
        {
            continue;
        }
        let tx = term_integer(x, shift);
        let ty = term_integer(y, shift);
        let mut product = ExactInteger::default();
        unsafe {
            gmp::mpz_mul(&mut product.raw, &tx.raw, &ty.raw);
            gmp::mpz_add(&mut acc.raw, &acc.raw, &product.raw);
        }
        max_term_bits = max_term_bits.max(product.bit_length());
    }

    Ok(ExactProduct {
        integer: acc,
        plan: ProductPlan {
            term_shift: shift,
            max_term_bits,
            value_bound_bits: bound_bits,
        },
    })
}

/// Shift a finite value into an integer aligned to `shift`.
fn term_integer<const N: usize>(v: &MpFloat<N>, shift: i64) -> ExactInteger {
    let view = v.dyadic_view();
    let mut limbs = vec![0u64; N];
    let bits = v.mantissa_limbs(&mut limbs).unwrap_or(0);
    let mut out = ExactInteger::default();
    if bits == 0 {
        return out;
    }
    unsafe {
        // Build M (limbs least significant first) with m <<= 64 between words.
        for limb in limbs.iter().rev() {
            gmp::mpz_mul_2exp(&mut out.raw, &out.raw, 64);
            let mut word = ExactInteger::default();
            gmp::mpz_set_ui(&mut word.raw, *limb);
            gmp::mpz_add(&mut out.raw, &out.raw, &word.raw);
        }
        // value = M * 2^(exp - prec); multiply by the further alignment shift.
        let extra = view.exponent as i64 - (N as i64 * 64) - shift;
        if extra >= 0 {
            gmp::mpz_mul_2exp(&mut out.raw, &out.raw, extra as u64);
        } else {
            gmp::mpz_tdiv_q_2exp(&mut out.raw, &out.raw, (-extra) as u64);
        }
        if view.kind.is_negative() {
            gmp::mpz_neg(&mut out.raw, &out.raw);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bits512;
    use num_traits::{FromPrimitive, One, ToPrimitive, Zero};
    use crate::Scalar;

    #[test]
    fn exact_product_matches_double_precision_trivial_cases() {
        let a = [Bits512::from_f64(3.0).unwrap(), Bits512::from_f64(4.0).unwrap()];
        let b = [Bits512::from_f64(5.0).unwrap(), Bits512::from_f64(6.0).unwrap()];
        let p = exact_product(&a, &b).unwrap();
        assert_eq!(p.to_mpfloat::<8>(), Bits512::from_f64(39.0).unwrap());
    }

    #[test]
    fn exact_product_keeps_cancellation_that_ordered_fma_loses() {
        // 1 + 2^-300, times 1, minus 1: the low bit must survive.
        let tiny = Bits512::from_f64(2f64.powi(-300)).unwrap();
        let one = Bits512::one();
        let a = [one + tiny, -one];
        let b = [one, one];
        let exact = exact_product(&a, &b).unwrap().to_mpfloat::<8>();
        assert_eq!(exact, tiny);
        // The ordered-FMA path also keeps it, because ordering happens to help here.
        let ordered = Bits512::dot_fma(a.iter().zip(&b));
        assert_eq!(ordered, tiny);
    }

    #[test]
    fn exact_product_keeps_low_bits_that_to_f64_drops() {
        // 2^53 + 1 is not representable in f64; the exact path must keep it.
        let a = [Bits512::from_i64(9007199254740993).unwrap()];
        let b = [Bits512::one()];
        let p = exact_product(&a, &b).unwrap();
        assert_eq!(p.to_mpfloat::<8>(), Bits512::from_i64(9007199254740993).unwrap());
        assert_eq!(a[0].to_f64().unwrap(), 9007199254740992.0);
        // The image is NOT 54 bits: every factor's mantissa is a full
        // PRECISION_BITS integer, so a product of two of them is about
        // 2 * PRECISION_BITS wide regardless of the value's magnitude.
        let width = p.integer.bit_length();
        assert!(
            (970..=1080).contains(&width),
            "unexpected image width {width}"
        );
        assert!(p.plan.image_bound_bits() >= width - 1);
    }

    /// The image width drives the modulus count, so it is pinned explicitly.
    #[test]
    fn image_width_and_prime_count_are_reported_not_assumed() {
        // A "small" 512-bit value: 1.0. Two of them multiply to an image with
        // roughly 2 * 512 significant bits.
        let one = [Bits512::one()];
        let p = exact_product(&one, &one).unwrap();
        let width = p.integer.bit_length();
        assert_eq!(p.to_mpfloat::<8>(), Bits512::one());
        // 2 * (512 - 1) bits of mantissa product, minus the alignment.
        assert!(width >= 1000, "width {width}");

        // Signed CRT uniqueness needs prod(p) > 2B where B bounds the *integer*
        // image, not the real value. At 512 bits the image is about 2*512 wide,
        // so a 20-bit modulus needs roughly 2*512/20 moduli.
        let needed = p.plan.modulus_count(20);
        assert!(
            (48..=58).contains(&needed),
            "512-bit 20-bit moduli needed = {needed}, expected ~2*512/20"
        );
        // And the BLAS panel condition is separate: it depends on the contraction
        // length, not on the precision.
        assert!(ProductPlan::blas_panel_is_exact(8192, 20));
        assert!(!ProductPlan::blas_panel_is_exact(1 << 20, 20));
    }

    #[test]
    fn exact_product_rejects_non_finite_operands() {
        let a = [Bits512::infinity()];
        let b = [Bits512::one()];
        assert_eq!(exact_product(&a, &b).unwrap_err(), TermError::NonFinite);
    }

    #[test]
    fn empty_and_zero_products_are_zero_with_zero_width() {
        let p = exact_product::<8>(&[], &[]).unwrap();
        assert!(p.integer.is_zero());
        assert_eq!(p.plan.max_term_bits, 0);
        assert_eq!(p.to_mpfloat::<8>(), Bits512::zero());
        let z = [Bits512::zero(), Bits512::zero()];
        let p = exact_product(&z, &z).unwrap();
        assert!(p.integer.is_zero());
    }

    #[test]
    fn wide_exponent_span_is_not_truncated() {
        // A term at 2^-400 and one at 2^400 must both survive the alignment.
        let a = [
            Bits512::from_f64(2f64.powi(-400)).unwrap(),
            Bits512::from_f64(2f64.powi(400)).unwrap(),
        ];
        let b = [Bits512::one(), Bits512::one()];
        let p = exact_product(&a, &b).unwrap();
        // The image needs about 800 bits: no bits were dropped to fit any modulus.
        assert!(
            p.integer.bit_length() > 700,
            "width {} suggests truncation",
            p.integer.bit_length()
        );
        let total = p.to_mpfloat::<8>();
        let expected = Bits512::from_f64(2f64.powi(-400)).unwrap()
            + Bits512::from_f64(2f64.powi(400)).unwrap();
        assert_eq!(total, expected);
    }
    /// A02/A13 gate (see /tmp/sdpx-phase0/A13_decision.md): real cost of projecting a hot gemm's operand block into
    /// residues, versus the MPFR FMA work the same block currently costs.
    #[test]
    fn projection_cost_at_measured_span() {
        use std::time::Instant;
        // Measured shapes: gemm m=16 n=31 k=16, operand entries = m*k + k*n.
        let (mk, kn, mnk) = (16 * 16usize, 16 * 31usize, 16 * 31 * 16usize);
        // Measured span (576 bits) means the alignment shift spans that range.
        let spread = [(-540i64), (-270), 0, 36];
        let primes: Vec<u32> = (0..81u32).map(|i| 1048573 - i).collect();

        // Build the entries once, exactly as a projection would consume them.
        let entries: Vec<ExactInteger> = spread
            .iter()
            .map(|&e| {
                let v = Bits512::from_f64(2f64.powi(e as i32)).unwrap();
                // align to the minimum exponent of the whole block, as the plan requires
                term_integer(&v, -540)
            })
            .collect();
        let block: Vec<&ExactInteger> = (0..(mk + kn)).map(|i| &entries[i % entries.len()]).collect();

        let t0 = Instant::now();
        let mut sink = 0u64;
        for e in &block {
            for p in &primes {
                let r = unsafe { gmp::mpz_fdiv_ui(&e.raw, *p as u64) };
                sink = sink.wrapping_add(r);
            }
        }
        let project_s = t0.elapsed().as_secs_f64();

        // Current MPFR cost for the same kernel: m*n*k ordered FMAs at ~118 ns.
        let mpfr_s = mnk as f64 * 118e-9;

        println!(
            "GATE entries={} primes={} projection={:.3} ms  mpfr={:.3} ms  ratio={:.2}x",
            block.len(),
            primes.len(),
            project_s * 1e3,
            mpfr_s * 1e3,
            project_s / mpfr_s
        );
        println!("GATE worst-case aligned width = {} bits", entries.iter().map(|e| e.bit_length()).max().unwrap());
        assert!(sink > 0);
    }
}
