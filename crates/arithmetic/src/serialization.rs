//! Decimal strings avoid a binary64 intermediate in high-precision JSON.
use crate::{MpFloat, Scalar};
use num_traits::Zero;
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

impl<const N: usize> Serialize for MpFloat<N> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_decimal(None))
    }
}

impl<'de, const N: usize> Deserialize<'de> for MpFloat<N> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Decimal<const N: usize>;
        impl<'de, const N: usize> de::Visitor<'de> for Decimal<N> {
            type Value = MpFloat<N>;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str(
                    "a finite decimal string (MPFR JSON must not use floating-point numbers)",
                )
            }

            fn visit_str<E: de::Error>(self, text: &str) -> Result<Self::Value, E> {
                let value: MpFloat<N> = text.parse().map_err(E::custom)?;
                if !value.is_finite()
                    || !text.bytes().all(|c| {
                        c.is_ascii_digit() || matches!(c, b'+' | b'-' | b'.' | b'e' | b'E')
                    })
                {
                    return Err(E::custom("MPFR input must be a finite decimal string"));
                }
                let mantissa = text.split(['e', 'E']).next().unwrap_or("");
                if value.is_zero() && mantissa.bytes().any(|c| matches!(c, b'1'..=b'9')) {
                    return Err(E::custom("MPFR input underflow"));
                }
                Ok(value)
            }

            // Integer JSON tokens have exact integer transport. Fractional JSON
            // numbers deliberately have no visitor: serde would round to f64.
            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                self.visit_str(&value.to_string())
            }
            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                self.visit_str(&value.to_string())
            }
        }
        deserializer.deserialize_any(Decimal::<N>)
    }
}
