//! Exact read-only dyadic view of a fixed-precision value.
//!
//! Every finite `MpFloat<N>` is an exact dyadic rational
//! `value = sign · M · 2^(exponent - PRECISION_BITS)`, where `M` is the integer
//! formed by the value's limbs **least significant limb first** and normalized
//! so that `2^(PRECISION_BITS-1) <= M < 2^PRECISION_BITS`. This module exposes
//! that pair without text round-trips, heap allocation, or any change to the
//! value's ownership or rounding behaviour.
//!
//! Both facts are pinned by `mantissa_layout_is_lsb_first_and_normalized`, which
//! checks 1.0, 2.0, 0.5, 1.5 and 3.0 limb-by-limb against their exact mantissas.
//!
//! Consumers use it to build exact integer images of matrices (for example a
//! modular product), where a lossy `to_f64()` would silently drop every bit
//! above 53.
use crate::{MpFloat, Scalar};
use gmp_mpfr_sys::mpfr;

/// Sign of an exact view, or a non-finite classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DyadicKind {
    /// Exactly zero (the sign of zero is not preserved by the solver paths).
    Zero,
    /// A finite non-zero value in `sign · M · 2^shift`.
    Finite {
        negative: bool,
    },
    Infinity {
        negative: bool,
    },
    Nan,
}

/// Borrowed, allocation-free exact view of one value.
#[derive(Clone, Copy, Debug)]
pub struct DyadicView {
    /// Classification of the source value.
    pub kind: DyadicKind,
    /// Binary exponent `e` in `value = sign · M · 2^(e - PRECISION_BITS)`.
    /// Meaningless unless `kind` is `Finite`.
    pub exponent: mpfr::exp_t,
    /// Number of significant bits in the mantissa (`PRECISION_BITS` for a
    /// normalized finite value, `0` for zero).
    pub mantissa_bits: usize,
}

impl DyadicKind {
    /// True when the value is finite and representable as `M · 2^shift`.
    pub fn is_finite(self) -> bool {
        matches!(self, DyadicKind::Zero | DyadicKind::Finite { .. })
    }
    /// Sign for finite non-zero values, `false` otherwise.
    pub fn is_negative(self) -> bool {
        matches!(
            self,
            DyadicKind::Finite { negative: true } | DyadicKind::Infinity { negative: true }
        )
    }
}

impl<const N: usize> MpFloat<N> {
    /// Classify and locate this value in the dyadic representation.
    pub fn dyadic_view(&self) -> DyadicView {
        Self::check_precision();
        // MPFR encodes the sign in `kind`, so -0 is `-ZERO_KIND`; solver paths
        // do not preserve the sign of zero and its exponent field is undefined.
        if self.kind.abs() == mpfr::ZERO_KIND {
            return DyadicView {
                kind: DyadicKind::Zero,
                exponent: 0,
                mantissa_bits: 0,
            };
        }
        if self.is_nan() {
            return DyadicView {
                kind: DyadicKind::Nan,
                exponent: 0,
                mantissa_bits: 0,
            };
        }
        if self.is_infinite() {
            return DyadicView {
                kind: DyadicKind::Infinity {
                    negative: self.is_sign_negative(),
                },
                exponent: 0,
                mantissa_bits: 0,
            };
        }
        DyadicView {
            kind: DyadicKind::Finite {
                negative: self.is_sign_negative(),
            },
            exponent: self.exponent,
            mantissa_bits: Self::PRECISION_BITS,
        }
    }

    /// Copy the mantissa limbs, least significant first (bit `i` of the integer
    /// `M` is bit `i % 64` of ``out[i / 64]``).
    ///
    /// Returns the number of significant bits written, or `None` when the value
    /// is not finite. `out` must hold at least `N` limbs; callers that passed a
    /// shorter slice get `None` rather than a partial copy.
    pub fn mantissa_limbs(&self, out: &mut [u64]) -> Option<usize> {
        let view = self.dyadic_view();
        if !view.kind.is_finite() {
            return None;
        }
        if out.len() < N {
            return None;
        }
        if matches!(view.kind, DyadicKind::Zero) {
            out[..N].fill(0);
            return Some(0);
        }
        out[..N].copy_from_slice(&self.limbs);
        Some(view.mantissa_bits)
    }

    /// Exact number of bits in `|value|`, i.e. `bit_length(M)` offset by the
    /// exponent. Signed: `0` for zero, `None` for non-finite.
    ///
    /// This is the quantity a modular plan needs for its bound on `|C_ij|`.
    pub fn exact_bit_length(&self) -> Option<i64> {
        let view = self.dyadic_view();
        match view.kind {
            DyadicKind::Zero => Some(0),
            DyadicKind::Nan | DyadicKind::Infinity { .. } => None,
            DyadicKind::Finite { .. } => {
                let mut limbs = [0u64; N];
                self.mantissa_limbs(&mut limbs)?;
                // Limbs are least significant first, so the leading limb is the
                // last non-zero one. Subnormal values simply have fewer bits.
                let top = limbs.iter().rposition(|&l| l != 0)?;
                let hi = limbs[top];
                let hi_bits = 64 - hi.leading_zeros() as i64;
                let below = top as i64 * 64;
                Some(hi_bits + below + view.exponent as i64 - Self::PRECISION_BITS as i64)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bits256, Bits512};
    use num_traits::{FromPrimitive, One, ToPrimitive, Zero};

    /// Rebuild `sign · M · 2^(exp - prec)` using exact power-of-two scaling only.
    fn rebuild<const N: usize>(v: &MpFloat<N>) -> Option<MpFloat<N>> {
        let view = v.dyadic_view();
        if matches!(view.kind, DyadicKind::Zero) {
            return Some(MpFloat::zero());
        }
        if !view.kind.is_finite() {
            return None;
        }
        let mut limbs = vec![0u64; N];
        v.mantissa_limbs(&mut limbs);
        let two = MpFloat::<N>::one() + MpFloat::<N>::one();
        // 2^64 by squaring, then the exact mantissa M (limbs are least significant first).
        let two64 = pow2::<N>(&two, 64);
        let mut m = MpFloat::<N>::zero();
        for limb in limbs.iter().rev() {
            m = m * two64 + MpFloat::<N>::from_u64(*limb).unwrap();
        }
        let shift = view.exponent as i64 - (N * 64) as i64;
        // Binary exponentiation: MPFR exponents reach 2^30, so a linear loop is unusable.
        let scale = pow2_signed::<N>(&two, shift);
        Some(if view.kind.is_negative() {
            -(m * scale)
        } else {
            m * scale
        })
    }

    fn pow2<const N: usize>(two: &MpFloat<N>, n: u32) -> MpFloat<N> {
        let mut result = MpFloat::<N>::one();
        let mut base = *two;
        let mut k = n;
        while k > 0 {
            if k & 1 == 1 {
                result = result * base;
            }
            base = base * base;
            k >>= 1;
        }
        result
    }

    fn pow2_signed<const N: usize>(two: &MpFloat<N>, n: i64) -> MpFloat<N> {
        let magnitude = n.unsigned_abs() as u32;
        let p = pow2(two, magnitude);
        if n >= 0 {
            p
        } else {
            MpFloat::<N>::one() / p
        }
    }

    #[test]
    fn mantissa_layout_is_lsb_first_and_normalized() {
        // Measured layout: `limb[0]` is the least significant word.
        let cases: [(f64, i64, u64); 5] = [
            (1.0, 1, 0x8000000000000000),
            (2.0, 2, 0x8000000000000000),
            (0.5, 0, 0x8000000000000000),
            (1.5, 1, 0xc000000000000000),
            (3.0, 2, 0xc000000000000000),
        ];
        for (value, exponent, top_limb) in cases {
            let v = Bits256::from_f64(value).unwrap();
            let view = v.dyadic_view();
            assert_eq!(view.exponent, exponent as mpfr::exp_t, "value {value}");
            let mut limbs = [0u64; 4];
            assert_eq!(v.mantissa_limbs(&mut limbs), Some(256));
            assert_eq!(limbs[3], top_limb, "value {value}");
            assert_eq!(limbs[..3], [0u64; 3], "value {value}");
        }
    }

    fn oracle_matches<const N: usize>(v: MpFloat<N>) {
        if v.is_zero() {
            assert_eq!(v.dyadic_view().kind, DyadicKind::Zero);
            return;
        }
        assert!(v.is_finite());
        let rebuilt = rebuild(&v).unwrap();
        assert_eq!(rebuilt, v, "dyadic view does not reproduce {v}");
    }

    #[test]
    fn dyadic_view_reproduces_exact_values_256() {
        let f = |x: f64| Bits256::from_f64(x).unwrap();
        oracle_matches::<4>(Bits256::one());
        oracle_matches::<4>(Bits256::one() + Bits256::one());
        oracle_matches::<4>(f(3.5));
        oracle_matches::<4>(-f(1.0 / 3.0));
        oracle_matches::<4>(f(-1234.5625));
        oracle_matches::<4>(Bits256::zero());
        oracle_matches::<4>(f(2f64.powi(-900)));
        oracle_matches::<4>(f(2f64.powi(900)));
        oracle_matches::<4>(Bits256::epsilon());
    }

    #[test]
    fn dyadic_view_reproduces_exact_values_512() {
        let f = |x: f64| Bits512::from_f64(x).unwrap();
        // -0 keeps kind `-ZERO_KIND`; the view must classify it as Zero.
        oracle_matches::<8>(f(-0.0));
        oracle_matches::<8>(Bits512::one());
        oracle_matches::<8>(f(1e300));
        oracle_matches::<8>(-f(1e-300));
        oracle_matches::<8>(Bits512::from_i64(9007199254740993).unwrap());
        oracle_matches::<8>(Bits512::epsilon());
        oracle_matches::<8>(Bits512::max_value());
        oracle_matches::<8>(-Bits512::max_value());
    }

    #[test]
    fn non_finite_values_are_classified_not_rebuilt() {
        for v in [Bits512::infinity(), Bits512::neg_infinity(), Bits512::nan()] {
            let view = v.dyadic_view();
            assert!(!view.kind.is_finite());
            let mut limbs = [0u64; 8];
            assert!(v.mantissa_limbs(&mut limbs).is_none());
            assert!(v.exact_bit_length().is_none());
            assert!(rebuild(&v).is_none());
        }
    }

    #[test]
    fn bit_length_tracks_magnitude() {
        for k in [-300i32, -1, 0, 1, 52, 53, 300] {
            let v = Bits512::from_f64(2f64.powi(k)).unwrap();
            assert_eq!(v.exact_bit_length(), Some((k + 1) as i64), "k={k}");
            assert_eq!((-v).exact_bit_length(), Some((k + 1) as i64), "-k={k}");
        }
        // An odd 54-bit integer keeps its low bit: this is exactly what f64 loses.
        let v = Bits512::from_i64(9007199254740993).unwrap();
        assert_eq!(v.exact_bit_length(), Some(54));
        assert_eq!(v.to_f64().unwrap(), 9007199254740992.0);
        assert_eq!(Bits512::zero().exact_bit_length(), Some(0));
    }

    #[test]
    fn short_destination_is_rejected_rather_than_truncated() {
        let v = Bits512::one();
        let mut short = [0u64; 4];
        assert!(v.mantissa_limbs(&mut short).is_none());
        assert_eq!(short, [0u64; 4]);
    }
}
