//! Owned GMP rationals for exact, bounded preprocessing only.
use gmp_mpfr_sys::gmp;
use std::mem::MaybeUninit;
pub struct Exact {
    pub(crate) raw: gmp::mpq_t,
}
impl Default for Exact {
    fn default() -> Self {
        let mut raw = MaybeUninit::uninit();
        unsafe {
            gmp::mpq_init(raw.as_mut_ptr());
            Self {
                raw: raw.assume_init(),
            }
        }
    }
}
impl Clone for Exact {
    fn clone(&self) -> Self {
        let mut out = Self::default();
        unsafe {
            gmp::mpq_set(&mut out.raw, &self.raw);
        }
        out
    }
}
impl Drop for Exact {
    fn drop(&mut self) {
        unsafe {
            gmp::mpq_clear(&mut self.raw);
        }
    }
}
impl Exact {
    pub(crate) fn from_f64(value: f64) -> Self {
        let mut out = Self::default();
        unsafe {
            gmp::mpq_set_d(&mut out.raw, value);
        }
        out
    }
    pub fn is_zero(&self) -> bool {
        unsafe { gmp::mpq_sgn(&self.raw) == 0 }
    }
    pub fn bounded(&self) -> bool {
        unsafe {
            gmp::mpz_sizeinbase(&self.raw.num, 2) <= 2048
                && gmp::mpz_sizeinbase(&self.raw.den, 2) <= 2048
        }
    }
    /// Exact image in the prime field F_(2^31 - 1), when the denominator
    /// is invertible. A nonzero minor here proves rational independence.
    pub fn modulo_mersenne31(&self) -> Option<u32> {
        const P: u64 = (1 << 31) - 1;
        let (num, den) = unsafe {
            (
                gmp::mpz_fdiv_ui(&self.raw.num, P as _) as u64,
                gmp::mpz_fdiv_ui(&self.raw.den, P as _) as u64,
            )
        };
        if den == 0 {
            return None;
        }
        let (mut power, mut exponent, mut inverse) = (den, P - 2, 1u64);
        while exponent != 0 {
            if exponent & 1 != 0 {
                inverse = inverse * power % P;
            }
            power = power * power % P;
            exponent >>= 1;
        }
        Some((num * inverse % P) as u32)
    }
    pub fn divide(&mut self, other: &Self) {
        assert!(!other.is_zero());
        unsafe {
            let p = &mut self.raw as *mut _;
            gmp::mpq_div(p, p, &other.raw);
        }
    }
    pub fn subtract_product(&mut self, a: &Self, b: &Self) {
        let mut product = Self::default();
        unsafe {
            gmp::mpq_mul(&mut product.raw, &a.raw, &b.raw);
            let p = &mut self.raw as *mut _;
            gmp::mpq_sub(p, p, &product.raw);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Bits512, Scalar};
    #[test]
    fn exact_conversion_owns_storage_and_preserves_sub_f64_bits() {
        let one = <Bits512 as num_traits::One>::one();
        let tiny = <Bits512 as Scalar>::epsilon();
        let a = one.exact().unwrap();
        let mut b = (one + tiny).exact().unwrap();
        let saved = b.clone();
        b.subtract_product(&a, &a);
        assert!(!b.is_zero());
        let mut c = saved.clone();
        c.subtract_product(&saved, &a);
        assert!(c.is_zero());
        assert!(!saved.is_zero());
        assert!(<Bits512 as num_traits::Zero>::zero()
            .exact()
            .unwrap()
            .is_zero());
        assert!(<Bits512 as Scalar>::infinity().exact().is_none());
        assert!(<Bits512 as Scalar>::max_value().exact().is_none());
    }
}
