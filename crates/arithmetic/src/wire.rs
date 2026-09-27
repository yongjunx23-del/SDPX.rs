//! Canonical, pointer-free scalar payloads for optional MPI state exchange.
//!
//! The tag is carried out of band by the caller.  Payloads deliberately have
//! no host-layout or serde dependency: IEEE values are their raw bits in
//! little-endian order, while MPFR values carry the custom descriptor fields
//! in a fixed, explicit layout.

use gmp_mpfr_sys::mpfr;
use std::sync::OnceLock;

const MAGIC: u64 = 0x5344_5058; // "SDPX"
const VERSION: u64 = 1;
const TYPE_F32: u64 = 1;
const TYPE_F64: u64 = 2;
const TYPE_MP: u64 = 3;
const PRECISION_MASK: usize = (1 << 20) - 1;

/// Build the stable format/type/precision tag shared by all scalar payloads.
///
/// Bits 63..32 are the ASCII `SDPX` marker, bits 31..24 are the wire version,
/// bits 23..20 are the scalar family, and bits 19..0 are the precision in
/// binary digits.  The common precisions used by SDPX are all representable.
const fn tag(kind: u64, precision: usize) -> u64 {
    (MAGIC << 32) | (VERSION << 24) | (kind << 20) | (precision as u64 & PRECISION_MASK as u64)
}

pub(crate) fn primitive_tag(width: usize) -> u64 {
    match width {
        4 => tag(TYPE_F32, f32::MANTISSA_DIGITS as usize),
        8 => tag(TYPE_F64, f64::MANTISSA_DIGITS as usize),
        _ => 0,
    }
}

pub(crate) fn mp_tag(precision: usize) -> u64 {
    if precision == 0 || precision > PRECISION_MASK || precision > mpfr::PREC_MAX as usize {
        0
    } else {
        tag(TYPE_MP, precision)
    }
}

pub(crate) fn mp_size<const N: usize>() -> Option<usize> {
    if N == 0 || N > (mpfr::PREC_MAX as usize) / 64 || N > PRECISION_MASK / 64 {
        return None;
    }
    N.checked_mul(8)?.checked_add(12)
}

fn regular_exponent_floor() -> Option<i64> {
    // MPFR reserves the three lowest descriptor exponents for zero, NaN and
    // infinity.  A regular custom descriptor must be strictly above EXP_INF.
    let max = i64::try_from(mpfr::exp_t::MAX).ok()?;
    2i64.checked_sub(max)
}

fn exponent_limits() -> Option<(i64, i64)> {
    static LIMITS: OnceLock<Option<(i64, i64)>> = OnceLock::new();
    *LIMITS.get_or_init(|| unsafe {
        Some((
            i64::try_from(mpfr::get_emin_min()).ok()?,
            i64::try_from(mpfr::get_emax_max()).ok()?,
        ))
    })
}

fn classify_kind(kind: i32) -> Option<i32> {
    if kind == mpfr::NAN_KIND {
        return Some(mpfr::NAN_KIND);
    }
    let magnitude = kind.checked_abs()?;
    match magnitude {
        mpfr::INF_KIND | mpfr::ZERO_KIND | mpfr::REGULAR_KIND => Some(magnitude),
        _ => None,
    }
}

fn valid_regular<const N: usize>(kind: i32, exponent: i64, limbs: &[u64; N]) -> bool {
    if classify_kind(kind) != Some(mpfr::REGULAR_KIND) || N == 0 {
        return false;
    }
    let Some(floor) = regular_exponent_floor() else {
        return false;
    };
    let Some((emin_min, emax_max)) = exponent_limits() else {
        return false;
    };
    if exponent <= floor
        || exponent < emin_min
        || exponent > emax_max
        || mpfr::exp_t::try_from(exponent).is_err()
    {
        return false;
    }
    // MPFR's custom significand is normalized: the most significant limb has
    // its top bit set.  Rejecting a non-normalized payload avoids silently
    // constructing a different numerical value on the receiver.
    limbs[N - 1] & (1u64 << 63) != 0
}

fn valid_special<const N: usize>(kind: i32, exponent: i64, limbs: &[u64; N]) -> bool {
    let Some(magnitude) = classify_kind(kind) else {
        return false;
    };
    if magnitude == mpfr::REGULAR_KIND {
        return false;
    }
    // Special values have no payload in this representation.  The wire form
    // is canonical even when a descriptor arrived with stale storage.
    exponent == 0 && limbs.iter().all(|&limb| limb == 0)
}

pub(crate) fn write_mp<const N: usize>(
    kind: i32,
    exponent: mpfr::exp_t,
    limbs: &[u64; N],
    out: &mut [u8],
) -> bool {
    let Some(size) = mp_size::<N>() else {
        return false;
    };
    if out.len() != size {
        return false;
    }
    let Some(exponent) = i64::try_from(exponent).ok() else {
        return false;
    };
    let Some(magnitude) = classify_kind(kind) else {
        return false;
    };
    if magnitude == mpfr::REGULAR_KIND {
        if !valid_regular(kind, exponent, limbs) {
            return false;
        }
    } else {
        // Accept stale special descriptor fields and canonicalize them to
        // zero.  Their numerical meaning is determined entirely by `kind`.
        if !matches!(magnitude, mpfr::NAN_KIND | mpfr::INF_KIND | mpfr::ZERO_KIND) {
            return false;
        }
    }
    out[..4].copy_from_slice(&kind.to_le_bytes());
    let canonical_exponent = if magnitude == mpfr::REGULAR_KIND {
        exponent
    } else {
        0
    };
    out[4..12].copy_from_slice(&canonical_exponent.to_le_bytes());
    let mut offset = 12;
    for &limb in limbs {
        let end = offset + 8;
        let word = if magnitude == mpfr::REGULAR_KIND {
            limb
        } else {
            0
        };
        out[offset..end].copy_from_slice(&word.to_le_bytes());
        offset = end;
    }
    true
}

pub(crate) fn read_mp<const N: usize>(bytes: &[u8]) -> Option<(i32, i64, [u64; N])> {
    let size = mp_size::<N>()?;
    if bytes.len() != size {
        return None;
    }
    let kind = i32::from_le_bytes(bytes[..4].try_into().ok()?);
    let exponent = i64::from_le_bytes(bytes[4..12].try_into().ok()?);
    let mut limbs = [0u64; N];
    let mut offset = 12;
    for limb in &mut limbs {
        let end = offset + 8;
        *limb = u64::from_le_bytes(bytes[offset..end].try_into().ok()?);
        offset = end;
    }
    let magnitude = classify_kind(kind)?;
    if magnitude == mpfr::REGULAR_KIND {
        valid_regular(kind, exponent, &limbs).then_some((kind, exponent, limbs))
    } else {
        valid_special(kind, exponent, &limbs).then_some((kind, exponent, limbs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bits1024, Bits128, Bits2048, Bits256, Bits512, Bits768, MpFloat, Scalar};
    use num_traits::{One, Zero};

    #[test]
    fn primitive_payloads_are_explicit_little_endian() {
        let mut f32_bytes = [0u8; 4];
        assert!(1.5f32.write_wire(&mut f32_bytes));
        assert_eq!(f32_bytes, [0, 0, 0xc0, 0x3f]);
        assert_eq!(f32::read_wire(&f32_bytes), Some(1.5));
        assert!(f32::from_bits(0x7fc0_1234).write_wire(&mut f32_bytes));
        assert_eq!(f32::read_wire(&f32_bytes).unwrap().to_bits(), 0x7fc0_1234);
        assert_eq!(f32::wire_size(), Some(4));
        assert_eq!(f32::wire_tag(), 0x5344_5058_0110_0018);

        let mut f64_bytes = [0u8; 8];
        assert!(f64::from_bits(0x8000_0000_0000_0000).write_wire(&mut f64_bytes));
        assert_eq!(f64_bytes, [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80]);
        assert_eq!(
            f64::read_wire(&f64_bytes).unwrap().to_bits(),
            0x8000_0000_0000_0000
        );
        assert_eq!(f64::wire_size(), Some(8));
        assert_eq!(f64::wire_tag(), 0x5344_5058_0120_0035);
        assert!(!1.0f32.write_wire(&mut [0u8; 3]));
        assert!(f32::read_wire(&[0u8; 3]).is_none());
    }

    fn regular_roundtrip<const N: usize>() {
        let value: MpFloat<N> = "-1.234567890123456789e-4096".parse().unwrap();
        let size = MpFloat::<N>::wire_size().unwrap();
        let mut bytes = vec![0u8; size];
        assert!(value.write_wire(&mut bytes));
        let decoded = MpFloat::<N>::read_wire(&bytes).unwrap();
        assert_eq!(decoded.exact_encode(), value.exact_encode());

        let tiny = (MpFloat::<N>::one() + MpFloat::<N>::epsilon()).mul_add(
            MpFloat::<N>::one() - MpFloat::<N>::epsilon(),
            -MpFloat::<N>::one(),
        );
        assert!(tiny.is_finite() && !tiny.is_zero());
        assert!(tiny.write_wire(&mut bytes));
        assert_eq!(
            MpFloat::<N>::read_wire(&bytes).unwrap().exact_encode(),
            tiny.exact_encode()
        );
    }

    #[test]
    fn all_supported_mpfr_precisions_roundtrip() {
        regular_roundtrip::<2>();
        regular_roundtrip::<4>();
        regular_roundtrip::<8>();
        regular_roundtrip::<12>();
        regular_roundtrip::<16>();
        regular_roundtrip::<32>();
    }

    #[test]
    fn special_values_are_canonical_and_sign_preserving() {
        fn check<const N: usize>() {
            for value in [
                MpFloat::<N>::zero(),
                -MpFloat::<N>::zero(),
                MpFloat::<N>::infinity(),
                -MpFloat::<N>::infinity(),
                MpFloat::<N>::nan(),
            ] {
                let mut bytes = vec![0u8; MpFloat::<N>::wire_size().unwrap()];
                assert!(value.write_wire(&mut bytes));
                let decoded = MpFloat::<N>::read_wire(&bytes).unwrap();
                if value.is_nan() {
                    assert!(decoded.is_nan());
                } else {
                    assert_eq!(decoded, value);
                    assert_eq!(decoded.is_sign_negative(), value.is_sign_negative());
                }
                assert!(bytes[4..12].iter().all(|&byte| byte == 0));
                assert!(bytes[12..].iter().all(|&byte| byte == 0));
            }
        }
        check::<2>();
        check::<4>();
        check::<8>();
        check::<12>();
        check::<16>();
        check::<32>();
    }

    #[test]
    fn malformed_mpfr_payloads_are_rejected() {
        type F = Bits128;
        let mut bytes = vec![0u8; F::wire_size().unwrap()];
        let short_len = bytes.len() - 1;
        assert!(!F::one().write_wire(&mut bytes[..short_len]));
        assert!(F::read_wire(&bytes[..short_len]).is_none());

        // Start each malformed-field check from an otherwise valid payload so
        // that the assertion cannot be satisfied by a different stale field.
        assert!(F::one().write_wire(&mut bytes));
        // Unknown kind.
        bytes[..4].copy_from_slice(&99i32.to_le_bytes());
        assert!(F::read_wire(&bytes).is_none());
        // Regular significands must be normalized.
        assert!(F::one().write_wire(&mut bytes));
        bytes[..4].copy_from_slice(&mpfr::REGULAR_KIND.to_le_bytes());
        bytes[20..28].copy_from_slice(&0u64.to_le_bytes());
        assert!(F::read_wire(&bytes).is_none());
        // Reserved descriptor exponents denote special values, not regulars.
        assert!(F::one().write_wire(&mut bytes));
        bytes[4..12].copy_from_slice(&(i64::MIN + 3).to_le_bytes());
        assert!(F::read_wire(&bytes).is_none());
        assert!(F::one().write_wire(&mut bytes));
        bytes[4..12].copy_from_slice(&i64::MAX.to_le_bytes());
        assert!(F::read_wire(&bytes).is_none());
        // Special values carry no exponent or limb payload.
        assert!(F::infinity().write_wire(&mut bytes));
        bytes[..4].copy_from_slice(&mpfr::INF_KIND.to_le_bytes());
        bytes[4..12].copy_from_slice(&1i64.to_le_bytes());
        assert!(F::read_wire(&bytes).is_none());
        assert!(F::infinity().write_wire(&mut bytes));
        bytes[4..12].copy_from_slice(&0i64.to_le_bytes());
        bytes[12..20].copy_from_slice(&1u64.to_le_bytes());
        assert!(F::read_wire(&bytes).is_none());
    }

    #[test]
    fn tags_and_sizes_are_precision_specific() {
        assert_eq!(Bits128::wire_size(), Some(28));
        assert_eq!(Bits256::wire_size(), Some(44));
        assert_eq!(Bits512::wire_size(), Some(76));
        assert_eq!(Bits768::wire_size(), Some(108));
        assert_eq!(Bits1024::wire_size(), Some(140));
        assert_eq!(Bits2048::wire_size(), Some(268));
        assert_ne!(Bits128::wire_tag(), Bits256::wire_tag());
        assert_eq!(Bits128::wire_tag(), 0x5344_5058_0130_0080);
        assert_eq!(Bits2048::wire_tag(), 0x5344_5058_0130_0800);
    }
}
