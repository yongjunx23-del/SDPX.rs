//! Fixed precision, independently owned floating point values.

//!
//! MPFR descriptors exist only during calls. A value contains no pointers, so
//! Rust copies, moves and vector reallocations preserve independent ownership.
//! Arithmetic always specifies nearest, ties-to-even rounding. MPFR's exponent
//! range is used without changing its process/thread configuration.
#[cfg(feature = "serde")]
mod serialization;
use gmp_mpfr_sys::{gmp, mpfr};
use num_traits::{FloatConst, FromPrimitive, Num, NumAssign, One, ToPrimitive, Zero};
use std::{
    cmp::Ordering,
    ffi::{CStr, CString},
    fmt,
    mem::MaybeUninit,
    ops::*,
    str::FromStr,
};

mod precisions;
pub use precisions::FRONTEND_PRECISION_HELP;

mod dyadic;
pub use dyadic::{DyadicKind, DyadicView};
mod exact;
mod exactdot;
/// Exact-integer reference conversions. Test-only oracle for the residue
/// kernels (see the module docs): not part of the production arithmetic.
#[cfg(test)]
mod integer;
pub use exact::Exact;
mod wire;

/// Operations required by the solver, without Float's 64-bit integer_decode.
pub trait Scalar:
    'static
    + Send
    + Sync
    + Copy
    + Default
    + PartialOrd
    + NumAssign
    + Neg<Output = Self>
    + FromPrimitive
    + ToPrimitive
    + FloatConst
    + fmt::Display
    + fmt::LowerExp
    + fmt::Debug
{
    /// Exact value for bounded structural presolve; unsupported types retain rows.
    fn exact(&self) -> Option<Exact> {
        None
    }
    /// Image of the exact value in F_(2^31 - 1), for presolve rank proofs.
    fn mersenne31(&self) -> Option<u32> {
        self.exact().and_then(|v| v.modulo_mersenne31())
    }
    /// Decimal text matching Display without formatting flags. Backends may
    /// avoid the extra formatted String allocation when producing JSON strings.
    fn decimal_string(&self) -> String {
        self.to_string()
    }
    /// Number of significant binary digits in arithmetic.
    fn precision_bits() -> usize;
    fn epsilon() -> Self;
    fn max_value() -> Self;
    fn min_value() -> Self;
    fn min_positive_value() -> Self;
    fn infinity() -> Self;
    fn neg_infinity() -> Self;
    fn nan() -> Self;
    /// Fixed-size canonical payload used by optional MPI state exchange.
    /// Unsupported scalar implementations return `None`/`0`/`false`.
    #[doc(hidden)]
    fn wire_size() -> Option<usize> {
        None
    }
    #[doc(hidden)]
    fn wire_tag() -> u64 {
        0
    }
    #[doc(hidden)]
    fn write_wire(self, _out: &mut [u8]) -> bool {
        false
    }
    #[doc(hidden)]
    fn read_wire(_bytes: &[u8]) -> Option<Self> {
        None
    }
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn cbrt(self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
    /// Accumulate products in iterator order, rounding each FMA at this precision.
    /// The default folds `mul_add`; owned high-precision types may reuse one
    /// accumulator instead of producing a fresh result per product.
    fn dot_fma<'a>(pairs: impl IntoIterator<Item = (&'a Self, &'a Self)>) -> Self {
        pairs
            .into_iter()
            .fold(Self::zero(), |acc, (x, y)| x.mul_add(*y, acc))
    }
    /// `dot_fma` over two equal-length slices. Same value; high-precision
    /// types can scan contiguous operands without collecting the terms.
    fn dot_slices(a: &[Self], b: &[Self]) -> Self {
        debug_assert_eq!(a.len(), b.len());
        Self::dot_fma(a.iter().zip(b))
    }
    fn ln(self) -> Self;
    fn exp(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powf(self, n: Self) -> Self;
    fn recip(self) -> Self {
        Self::one() / self
    }
    fn signum(self) -> Self;
    fn is_nan(self) -> bool;
    fn is_infinite(self) -> bool;
    fn is_finite(self) -> bool;
    fn is_sign_negative(self) -> bool;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
}
macro_rules! primitive_scalar {
    ($t:ty) => {
        impl Scalar for $t {
            fn exact(&self) -> Option<Exact> {
                self.is_finite().then(|| Exact::from_f64(*self as f64))
            }
            fn mersenne31(&self) -> Option<u32> {
                let v = *self as f64;
                if !v.is_finite() {
                    return None;
                }
                let bits = v.to_bits();
                let (e, m) = (((bits >> 52) & 0x7ff) as i64, bits & ((1 << 52) - 1));
                let (mantissa, exponent) = if e == 0 {
                    (m, -1074)
                } else {
                    (m | 1 << 52, e - 1075)
                };
                Some(exact::mersenne31_dyadic(v < 0.0, &[mantissa], exponent))
            }
            fn wire_size() -> Option<usize> {
                Some(std::mem::size_of::<$t>())
            }
            fn wire_tag() -> u64 {
                crate::wire::primitive_tag(std::mem::size_of::<$t>())
            }
            fn write_wire(self, out: &mut [u8]) -> bool {
                if out.len() != std::mem::size_of::<$t>() {
                    return false;
                }
                out.copy_from_slice(&self.to_bits().to_le_bytes());
                true
            }
            fn read_wire(bytes: &[u8]) -> Option<Self> {
                let bytes: [u8; std::mem::size_of::<$t>()] = bytes.try_into().ok()?;
                let mut word = [0u8; 8];
                word[..bytes.len()].copy_from_slice(&bytes);
                Some(Self::from_bits(u64::from_le_bytes(word) as _))
            }
            #[inline]
            fn precision_bits() -> usize {
                Self::MANTISSA_DIGITS as usize
            }
            #[inline]
            fn epsilon() -> Self {
                Self::EPSILON
            }
            #[inline]
            fn max_value() -> Self {
                Self::MAX
            }
            #[inline]
            fn min_value() -> Self {
                Self::MIN
            }
            #[inline]
            fn min_positive_value() -> Self {
                Self::MIN_POSITIVE
            }
            #[inline]
            fn infinity() -> Self {
                Self::INFINITY
            }
            #[inline]
            fn neg_infinity() -> Self {
                Self::NEG_INFINITY
            }
            #[inline]
            fn nan() -> Self {
                Self::NAN
            }
            #[inline]
            fn abs(self) -> Self {
                self.abs()
            }
            #[inline]
            fn sqrt(self) -> Self {
                self.sqrt()
            }
            #[inline]
            fn cbrt(self) -> Self {
                self.cbrt()
            }
            #[inline]
            fn mul_add(self, a: Self, b: Self) -> Self {
                self.mul_add(a, b)
            }
            #[inline]
            fn ln(self) -> Self {
                self.ln()
            }
            #[inline]
            fn exp(self) -> Self {
                self.exp()
            }
            #[inline]
            fn sin(self) -> Self {
                self.sin()
            }
            #[inline]
            fn cos(self) -> Self {
                self.cos()
            }
            #[inline]
            fn atan(self) -> Self {
                self.atan()
            }
            #[inline]
            fn atan2(self, other: Self) -> Self {
                self.atan2(other)
            }
            #[inline]
            fn powi(self, n: i32) -> Self {
                self.powi(n)
            }
            #[inline]
            fn powf(self, n: Self) -> Self {
                self.powf(n)
            }
            #[inline]
            fn signum(self) -> Self {
                self.signum()
            }
            #[inline]
            fn is_nan(self) -> bool {
                self.is_nan()
            }
            #[inline]
            fn is_infinite(self) -> bool {
                self.is_infinite()
            }
            #[inline]
            fn is_finite(self) -> bool {
                self.is_finite()
            }
            #[inline]
            fn is_sign_negative(self) -> bool {
                self.is_sign_negative()
            }
            #[inline]
            fn min(self, b: Self) -> Self {
                self.min(b)
            }
            #[inline]
            fn max(self, b: Self) -> Self {
                self.max(b)
            }
        }
    };
}
primitive_scalar!(f32);
primitive_scalar!(f64);

const ROUND: mpfr::rnd_t = mpfr::rnd_t::RNDN;
// The inline representation is deliberately limited to 64-bit GMP builds.
const _: () = assert!(gmp::NUMB_BITS == 64 && std::mem::size_of::<gmp::limb_t>() == 8);

#[derive(Clone, Copy)]
pub struct MpFloat<const N: usize> {
    pub(crate) limbs: [u64; N],
    pub(crate) kind: i32,
    pub(crate) exponent: mpfr::exp_t,
}
pub type Bits128 = MpFloat<2>;
pub type Bits256 = MpFloat<4>;
pub type Bits512 = MpFloat<8>;
pub type Bits768 = MpFloat<12>;
pub type Bits1024 = MpFloat<16>;
pub type Bits1216 = MpFloat<19>;
pub type Bits2048 = MpFloat<32>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    pub precision_bits: usize,
    pub message: &'static str,
}
impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}-bit arithmetic)",
            self.message, self.precision_bits
        )
    }
}
impl std::error::Error for ParseError {}

impl<const N: usize> Default for MpFloat<N> {
    fn default() -> Self {
        Self {
            limbs: [0; N],
            kind: mpfr::ZERO_KIND,
            exponent: 0,
        }
    }
}
impl<const N: usize> MpFloat<N> {
    pub const PRECISION_BITS: usize = N * 64;
    fn check_precision() {
        assert!(
            N > 0 && N <= (mpfr::PREC_MAX as usize) / 64,
            "invalid MPFR precision"
        );
    }
    // These descriptors must never escape the call that borrows their owner.
    fn descriptor(&self) -> mpfr::mpfr_t {
        Self::check_precision();
        unsafe {
            let mut d = MaybeUninit::uninit();
            mpfr::custom_init_set(
                d.as_mut_ptr(),
                self.kind,
                self.exponent,
                Self::PRECISION_BITS as _,
                self.limbs.as_ptr().cast_mut().cast(),
            );
            d.assume_init()
        }
    }
    fn output(f: impl FnOnce(*mut mpfr::mpfr_t)) -> Self {
        let mut result = Self::default();
        // Construct from the mutable borrow, never write through a shared reference.
        Self::check_precision();
        unsafe {
            let mut d = MaybeUninit::uninit();
            mpfr::custom_init(result.limbs.as_mut_ptr().cast(), Self::PRECISION_BITS as _);
            mpfr::custom_init_set(
                d.as_mut_ptr(),
                result.kind,
                0,
                Self::PRECISION_BITS as _,
                result.limbs.as_mut_ptr().cast(),
            );
            let mut d = d.assume_init();
            f(&mut d);
            result.kind = mpfr::custom_get_kind(&d);
            result.exponent = mpfr::custom_get_exp(&d);
        }
        result
    }
    /// Adopt the value held by a native descriptor at this precision.
    ///
    /// Rounded to this precision; exact when the source is representable.
    ///
    /// # Safety
    /// `desc` must be an initialized MPFR value with valid limb storage that
    /// remains readable and is not mutated for the duration of this call.
    pub unsafe fn from_mpfr_descriptor(desc: &mpfr::mpfr_t) -> Self {
        Self::output(|r| unsafe {
            mpfr::set(r, desc, ROUND);
        })
    }

    /// Lossless encoding: `(kind, exponent, limbs)`. Round-trips
    /// every representable state — finite values, ±0, ±inf, NaN — without
    /// text or f64 intermediates. Only the three public fields are exposed;
    /// no MPFR descriptor pointers are serialized.
    pub fn exact_encode(&self) -> (i32, i64, &[u64; N]) {
        (self.kind, self.exponent as i64, &self.limbs)
    }
    /// Inverse of [`Self::exact_encode`]. The caller must supply a triple produced
    /// at the same precision; no validation beyond a precision check is done.
    pub fn exact_decode(kind: i32, exponent: i64, limbs: [u64; N]) -> Self {
        Self::check_precision();
        Self {
            limbs,
            kind,
            exponent: exponent as mpfr::exp_t,
        }
    }

    /// Dot product accumulated exactly and rounded once at this precision
    /// (see `exactdot`). Non-finite operands, or an exponent spread too wide
    /// for the fixed-point window, keep [`Self::dot_fma_chain`].
    #[inline]
    pub fn dot_fma<'a>(pairs: impl IntoIterator<Item = (&'a Self, &'a Self)>) -> Self {
        exactdot::dot(pairs, |terms| Self::dot_fma_chain(terms))
    }

    /// Correctly rounded sum of exactly two products. MPFR evaluates the
    /// unrounded products together, avoiding the general dot accumulator.
    #[inline]
    pub fn dot_fma2(a: &Self, b: &Self, c: &Self, d: &Self) -> Self {
        if [a, b, c, d]
            .iter()
            .any(|v| !matches!(v.kind.abs(), mpfr::ZERO_KIND | mpfr::REGULAR_KIND))
        {
            return Self::dot_fma_chain([(a, b), (c, d)]);
        }
        let fmma = || {
            let (a, b, c, d) = (
                a.descriptor(),
                b.descriptor(),
                c.descriptor(),
                d.descriptor(),
            );
            Self::output(|r| unsafe {
                mpfr::fmma(r, &a, &b, &c, &d, ROUND);
            })
        };
        // The narrow exact kernel declines to fmma, preserving one rounding.
        // Zero operands use MPFR so the sign of an all-zero sum is retained.
        if N <= exactdot::INLINE_N
            && [a, b, c, d]
                .iter()
                .all(|v| v.kind.abs() == mpfr::REGULAR_KIND)
        {
            Self::fmma_regular(a, b, c, d)
                .unwrap_or_else(|| exactdot::dot([(a, b), (c, d)], |_| fmma()))
        } else {
            fmma()
        }
    }

    /// Apply `[c s; -s c]` to adjacent entries of independent strided rows.
    /// Each destination is the correctly rounded sum of two products, exactly
    /// as in two calls to `dot_fma2`. Coefficient descriptors are reused across
    /// rows; descriptors never outlive the owned values they borrow.
    pub fn rotate_adjacent_rows(rows: &mut [Self], stride: usize, p: usize, c: &Self, s: &Self) {
        assert!(stride > 0 && p + 1 < stride && rows.len() % stride == 0);
        let negative_s = -*s;
        let (cd, sd, nsd) = (c.descriptor(), s.descriptor(), negative_s.descriptor());
        for row in rows.chunks_exact_mut(stride) {
            if N <= exactdot::INLINE_N {
                let (x, y) = (row[p], row[p + 1]);
                row[p] = Self::dot_fma2(c, &x, s, &y);
                row[p + 1] = Self::dot_fma2(&negative_s, &x, c, &y);
                continue;
            }
            if [c, s, &row[p], &row[p + 1]]
                .iter()
                .any(|v| !matches!(v.kind.abs(), mpfr::ZERO_KIND | mpfr::REGULAR_KIND))
            {
                let (x, y) = (row[p], row[p + 1]);
                row[p] = Self::dot_fma2(c, &x, s, &y);
                row[p + 1] = Self::dot_fma2(&negative_s, &x, c, &y);
                continue;
            }
            let (x, y) = (row[p].descriptor(), row[p + 1].descriptor());
            let first = Self::output(|r| unsafe {
                mpfr::fmma(r, &cd, &x, &sd, &y, ROUND);
            });
            let second = Self::output(|r| unsafe {
                mpfr::fmma(r, &nsd, &x, &cd, &y, ROUND);
            });
            row[p] = first;
            row[p + 1] = second;
        }
    }

    /// Accumulate products in iterator order, rounding each FMA at this precision.
    /// The accumulator owns its storage and never aliases an input operand.
    pub fn dot_fma_chain<'a>(pairs: impl IntoIterator<Item = (&'a Self, &'a Self)>) -> Self {
        Self::output(|r| {
            for (a, b) in pairs {
                let x = a.descriptor();
                let y = b.descriptor();
                // MPFR permits the destination to alias an input. Here only the
                // previous accumulator is reused; input limb arrays stay read-only.
                unsafe {
                    mpfr::fma(r, &x, &y, r, ROUND);
                }
            }
        })
    }

    /// Correctly rounded Euclidean norm, without squaring at the input scale.
    pub fn hypot(self, other: Self) -> Self {
        self.binary(other, mpfr::hypot)
    }

    /// Exact scaling by `2^shift` in one MPFR step.
    ///
    /// `mpfr_mul_2si` is exact while the result stays in the exponent range,
    /// so no rounding is introduced beyond the destination precision already
    /// in force; out-of-range results round once to infinity or zero. Where
    /// `c_long` is narrower than `i64`, two exact half-steps cover the full
    /// `shift` range with the same semantics.
    pub fn scale_pow2(self, shift: i64) -> Self {
        if shift == 0 {
            return self;
        }
        match std::os::raw::c_long::try_from(shift) {
            Ok(n) => {
                let d = self.descriptor();
                Self::output(|r| unsafe {
                    mpfr::mul_2si(r, &d, n, ROUND);
                })
            }
            Err(_) => self.scale_pow2(shift / 2).scale_pow2(shift - shift / 2),
        }
    }

    fn unary(
        self,
        f: unsafe extern "C" fn(*mut mpfr::mpfr_t, *const mpfr::mpfr_t, mpfr::rnd_t) -> i32,
    ) -> Self {
        let a = self.descriptor();
        Self::output(|r| unsafe {
            f(r, &a, ROUND);
        })
    }
    /// Correctly rounded (nearest-even) product of two regular values,
    /// rounded inline after the limb product. Round-to-nearest
    /// has one answer, so this equals `mpfr_mul` bit for bit; `None` (zero,
    /// infinity, NaN, wider mantissas, exponents near the MPFR range limits)
    /// leaves the product to MPFR.
    #[inline]
    fn mul_regular(&self, b: &Self) -> Option<Self> {
        // Inline rounding wins through 19 limbs on Apple M4 (512 bits: 35 vs
        // 46 ns, 1216 bits: 142 vs 152 ns); MPFR is faster at 32 limbs.
        if N > 20 || self.kind.abs() != mpfr::REGULAR_KIND || b.kind.abs() != mpfr::REGULAR_KIND {
            return None;
        }
        // Bound inline exponent arithmetic; current MPFR limits are checked
        // after rounding.
        const SAFE: i64 = 1 << 29;
        let e = self.exponent as i64 + b.exponent as i64;
        if !(-SAFE..=SAFE).contains(&e) {
            return None;
        }
        let power =
            |a: &Self| a.limbs[N - 1] == 1 << 63 && a.limbs[..N - 1].iter().all(|&l| l == 0);
        let other = if power(self) {
            Some(b)
        } else if power(b) {
            Some(self)
        } else {
            None
        };
        if let Some(other) = other {
            let exponent = e - 1;
            if !Self::exponent_in_range(exponent) {
                return None;
            }
            let negative = (self.kind < 0) != (b.kind < 0);
            return Some(Self {
                kind: if negative {
                    -mpfr::REGULAR_KIND
                } else {
                    mpfr::REGULAR_KIND
                },
                exponent: exponent as mpfr::exp_t,
                ..*other
            });
        }
        if N > exactdot::SCHOOLBOOK_N {
            // GMP writes all 2N product limbs: no buffer clearing.
            let mut buf = MaybeUninit::<[u64; 2 * exactdot::MAX_N]>::uninit();
            let ptr = buf.as_mut_ptr().cast::<u64>();
            // SAFETY: the buffer holds 2N <= 2*MAX_N limbs and does not overlap
            // the operands; mpn_mul_n initializes exactly those 2N limbs, and
            // only they are referenced afterwards.
            let prod = unsafe {
                gmp::mpn_mul_n(ptr, self.limbs.as_ptr(), b.limbs.as_ptr(), N as _);
                std::slice::from_raw_parts_mut(ptr, 2 * N)
            };
            return self.round_product(b, prod, e);
        }
        exactdot::with_limbs::<N, _>(|prod| {
            let prod = &mut prod[..2 * N];
            exactdot::mul_limbs::<N>(&self.limbs, &b.limbs, prod);
            self.round_product(b, prod, e)
        })
    }
    #[inline]
    fn exponent_in_range(e: i64) -> bool {
        unsafe { mpfr::get_emin() as i64 <= e && e <= mpfr::get_emax() as i64 }
    }
    /// Normalize and round an exact `2N`-limb product to nearest-even.
    #[inline(always)]
    fn round_product(&self, b: &Self, prod: &mut [u64], mut e: i64) -> Option<Self> {
        // 0.Ma * 0.Mb lies in [1/4, 1): normalize by at most one bit.
        if prod[2 * N - 1] >> 63 == 0 {
            let mut carry = 0;
            for limb in prod.iter_mut() {
                let next = *limb >> 63;
                *limb = (*limb << 1) | carry;
                carry = next;
            }
            e -= 1;
        }
        let round = prod[N - 1] >> 63 == 1;
        let sticky = prod[N - 1] << 1 != 0 || prod[..N - 1].iter().any(|&l| l != 0);
        let mut limbs = [0u64; N];
        limbs.copy_from_slice(&prod[N..]);
        if round && (sticky || limbs[0] & 1 == 1) {
            let mut carry = true;
            for limb in limbs.iter_mut() {
                let (v, c) = limb.overflowing_add(carry as u64);
                *limb = v;
                carry = c;
                if !carry {
                    break;
                }
            }
            if carry {
                // Rounded up to 1.0: the mantissa becomes 0.1000... one binade up.
                limbs[N - 1] = 1 << 63;
                e += 1;
            }
        }
        if !Self::exponent_in_range(e) {
            return None;
        }
        let negative = (self.kind < 0) != (b.kind < 0);
        Some(Self {
            limbs,
            kind: if negative {
                -mpfr::REGULAR_KIND
            } else {
                mpfr::REGULAR_KIND
            },
            exponent: e as mpfr::exp_t,
        })
    }
    /// One exact product plus an exact addend in a bounded limb window.
    #[inline]
    fn fma_regular(&self, a: &Self, b: &Self) -> Option<Self> {
        if N > exactdot::SCHOOLBOOK_N
            || [self, a, b]
                .iter()
                .any(|v| v.kind.abs() != mpfr::REGULAR_KIND)
        {
            return None;
        }
        const SAFE: i64 = 1 << 29;
        if [self, a, b]
            .iter()
            .any(|v| !(-SAFE..=SAFE).contains(&(v.exponent as i64)))
        {
            return None;
        }
        let p = Self::PRECISION_BITS as i64;
        let ep = self.exponent as i64 + a.exponent as i64;
        let d = ep - b.exponent as i64;
        // 2N+2 limbs hold the full product, shifted addend, and one carry bit.
        if !(-127..=p + 127).contains(&d) {
            return None;
        }
        let sp = ep - 2 * p;
        let sb = b.exponent as i64 - p;
        let scale = sp.min(sb);
        let sx = (self.kind < 0) != (a.kind < 0);
        let sy = b.kind < 0;
        Some(exactdot::with_limbs::<N, _>(|xb| {
            exactdot::with_limbs::<N, _>(|yb| {
                let w = 2 * N + 2;
                let xb = &mut xb[..w];
                let yb = &mut yb[..w];
                exactdot::mul_limbs::<N>(&self.limbs, &a.limbs, &mut xb[..2 * N]);
                let shift = (sp - scale) as usize;
                Self::shift_exact_limbs(xb, shift);
                let shift = (sb - scale) as usize;
                let (q, r) = (shift / 64, shift % 64);
                for (i, &limb) in b.limbs.iter().enumerate() {
                    yb[q + i] |= limb << r;
                    if r != 0 {
                        yb[q + i + 1] |= limb >> (64 - r);
                    }
                }
                Self::add_exact_limbs(xb, sx, yb, sy, scale)
            })
        }))
    }
    #[inline(always)]
    fn shift_exact_limbs(limbs: &mut [u64], shift: usize) {
        if shift != 0 {
            let (q, r) = (shift / 64, shift % 64);
            for i in (0..limbs.len()).rev() {
                let lo = if i >= q { limbs[i - q] } else { 0 };
                let hi = if i > q { limbs[i - q - 1] } else { 0 };
                limbs[i] = if r == 0 {
                    lo
                } else {
                    (lo << r) | (hi >> (64 - r))
                };
            }
        }
    }
    #[inline(always)]
    fn add_exact_limbs(xb: &mut [u64], sx: bool, yb: &[u64], sy: bool, scale: i64) -> Self {
        let w = xb.len();
        let negative;
        if sx == sy {
            let mut carry = false;
            for (x, &y) in xb.iter_mut().zip(yb.iter()) {
                let (v, c1) = x.overflowing_add(y);
                let (v, c2) = v.overflowing_add(carry as u64);
                *x = v;
                carry = c1 | c2;
            }
            debug_assert!(!carry);
            negative = sx;
        } else {
            let x_ge = xb.iter().rev().cmp(yb.iter().rev()) != Ordering::Less;
            let mut borrow = false;
            for i in 0..w {
                let (big, small) = if x_ge { (xb[i], yb[i]) } else { (yb[i], xb[i]) };
                let (v, b1) = big.overflowing_sub(small);
                let (v, b2) = v.overflowing_sub(borrow as u64);
                xb[i] = v;
                borrow = b1 | b2;
            }
            debug_assert!(!borrow);
            negative = if x_ge { sx } else { sy };
        }
        Self::from_scaled_integer(negative, xb, scale)
    }
    /// Two unrounded products in the same bounded exact window as scalar FMA.
    #[inline]
    fn fmma_regular(a: &Self, b: &Self, c: &Self, d: &Self) -> Option<Self> {
        debug_assert!(N <= exactdot::INLINE_N);
        const SAFE: i64 = 1 << 29;
        if [a, b, c, d]
            .iter()
            .any(|v| !(-SAFE..=SAFE).contains(&(v.exponent as i64)))
        {
            return None;
        }
        let ep = a.exponent as i64 + b.exponent as i64;
        let eq = c.exponent as i64 + d.exponent as i64;
        let delta = ep - eq;
        // Each product needs 2P bits; 127 shift bits leave one bit for carry.
        if !(-127..=127).contains(&delta) {
            return None;
        }
        let scale = ep.min(eq) - 2 * Self::PRECISION_BITS as i64;
        let sx = (a.kind < 0) != (b.kind < 0);
        let sy = (c.kind < 0) != (d.kind < 0);
        Some(exactdot::with_limbs::<N, _>(|xb| {
            exactdot::with_limbs::<N, _>(|yb| {
                let w = 2 * N + 2;
                let (xb, yb) = (&mut xb[..w], &mut yb[..w]);
                exactdot::mul_limbs::<N>(&a.limbs, &b.limbs, &mut xb[..2 * N]);
                exactdot::mul_limbs::<N>(&c.limbs, &d.limbs, &mut yb[..2 * N]);
                if delta >= 0 {
                    Self::shift_exact_limbs(xb, delta as usize);
                } else {
                    Self::shift_exact_limbs(yb, (-delta) as usize);
                }
                Self::add_exact_limbs(xb, sx, yb, sy, scale)
            })
        }))
    }
    /// Correctly rounded (nearest-even) `self + b` (or `self - b` when
    /// `negate_b`) for regular operands. Exact in a 2N+2-limb buffer, then one rounding: equal to
    /// `mpfr_add`/`mpfr_sub` bit for bit. `None` defers to MPFR.
    #[inline]
    fn add_regular(&self, b: &Self, negate_b: bool) -> Option<Self> {
        // Inline wins through 8 limbs on Apple M4 (512 bits: 18 vs 19 ns);
        // MPFR's O(N) assembly is faster from 12 limbs.
        if N > 8 || self.kind.abs() != mpfr::REGULAR_KIND || b.kind.abs() != mpfr::REGULAR_KIND {
            return None;
        }
        const SAFE: i64 = 1 << 29;
        let (ea, eb) = (self.exponent as i64, b.exponent as i64);
        if !(-SAFE..=SAFE).contains(&ea) || !(-SAFE..=SAFE).contains(&eb) {
            return None;
        }
        let sa = self.kind < 0;
        let sb = (b.kind < 0) != negate_b;
        // x is the operand with the larger exponent (ties: either).
        let (x, sx, ex, y, sy, ey) = if ea >= eb {
            (self, sa, ea, b, sb, eb)
        } else {
            (b, sb, eb, self, sa, ea)
        };
        let p = Self::PRECISION_BITS as i64;
        let d = ex - ey;
        let regular = |negative: bool| {
            if negative {
                -mpfr::REGULAR_KIND
            } else {
                mpfr::REGULAR_KIND
            }
        };
        if d >= p + 2 {
            // |y| < 2^(ex-P-2): below half an ulp of x on either side, even
            // when x is a power of two. Nearest-even returns x.
            if !Self::exponent_in_range(ex) {
                return None;
            }
            return Some(Self {
                kind: regular(sx),
                ..*x
            });
        }
        // X = Mx * 2^(64(N+1)), Y = My * 2^(64(N+1) - d), both exact in W limbs
        // (top limb spare for the carry); value = (X +- Y) * 2^(ex - P - 64(N+1)).
        exactdot::with_limbs::<N, _>(|xb| {
            exactdot::with_limbs::<N, _>(|yb| Self::add_aligned(x, sx, ex, y, sy, d, xb, yb))
        })
    }
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn add_aligned(
        x: &Self,
        sx: bool,
        ex: i64,
        y: &Self,
        sy: bool,
        d: i64,
        xb: &mut [u64],
        yb: &mut [u64],
    ) -> Option<Self> {
        let p = Self::PRECISION_BITS as i64;
        let regular = |negative: bool| {
            if negative {
                -mpfr::REGULAR_KIND
            } else {
                mpfr::REGULAR_KIND
            }
        };
        let w = 2 * N + 2;
        xb[N + 1..2 * N + 1].copy_from_slice(&x.limbs);
        {
            let (q, r) = ((d / 64) as usize, (d % 64) as u32);
            // y placed at limbs N+1.., shifted right by d bits.
            for (i, &l) in y.limbs.iter().enumerate() {
                let at = N + 1 + i - q;
                if r == 0 {
                    yb[at] |= l;
                } else {
                    yb[at] |= l >> r;
                    yb[at - 1] |= l << (64 - r);
                }
            }
        }
        let xb = &mut xb[..w];
        let yb = &yb[..w];
        let negative;
        if sx == sy {
            let mut carry = false;
            for (a, &c) in xb.iter_mut().zip(yb) {
                let (v, c1) = a.overflowing_add(c);
                let (v, c2) = v.overflowing_add(carry as u64);
                *a = v;
                carry = c1 | c2;
            }
            debug_assert!(!carry);
            negative = sx;
        } else {
            // |X| vs |Y|: subtract the smaller magnitude from the larger.
            let x_ge = xb
                .iter()
                .rev()
                .zip(yb.iter().rev())
                .find(|(a, c)| a != c)
                .map_or(true, |(a, c)| a > c);
            // In place: each limb of X is read once before it is written.
            let mut borrow = false;
            for i in 0..w {
                let (big, small) = if x_ge { (xb[i], yb[i]) } else { (yb[i], xb[i]) };
                let (v, b1) = big.overflowing_sub(small);
                let (v, b2) = v.overflowing_sub(borrow as u64);
                xb[i] = v;
                borrow = b1 | b2;
            }
            negative = if x_ge { sx } else { sy };
        }
        let Some(top) = xb.iter().rposition(|&l| l != 0) else {
            // Exact cancellation: +0 under nearest-even.
            return Some(Self::default());
        };
        // Highest set bit h; mantissa = bits [h-P+1, h].
        let h = top as i64 * 64 + 63 - xb[top].leading_zeros() as i64;
        let mut e = h + ex - p - 64 * (N as i64 + 1) + 1;
        let shift = h + 1 - p; // right shift of X giving the mantissa (may be <= 0)
        let mut limbs = [0u64; N];
        let bit = |i: i64| -> u64 {
            if i < 0 || i >= 64 * w as i64 {
                0
            } else {
                (xb[(i / 64) as usize] >> (i % 64)) & 1
            }
        };
        let (round, sticky) = if shift > 0 {
            let (q, r) = ((shift / 64) as usize, (shift % 64) as u32);
            for (k, limb) in limbs.iter_mut().enumerate() {
                let lo = xb.get(q + k).copied().unwrap_or(0);
                let hi = xb.get(q + k + 1).copied().unwrap_or(0);
                *limb = if r == 0 {
                    lo
                } else {
                    (lo >> r) | (hi << (64 - r))
                };
            }
            let round = bit(shift - 1) == 1;
            let rb = shift - 1;
            let sticky = (0..(rb / 64) as usize).any(|i| xb[i] != 0)
                || (rb % 64 != 0 && xb[(rb / 64) as usize] & ((1u64 << (rb % 64)) - 1) != 0);
            (round, sticky)
        } else {
            // Fewer than P significant bits: exact, shift left.
            let left = -shift;
            let (q, r) = ((left / 64) as usize, (left % 64) as u32);
            for k in 0..N {
                let src = k as i64 - q as i64;
                let lo = if src >= 0 { xb[src as usize] } else { 0 };
                let below = if src >= 1 { xb[src as usize - 1] } else { 0 };
                limbs[k] = if r == 0 {
                    lo
                } else {
                    (lo << r) | (below >> (64 - r))
                };
            }
            (false, false)
        };
        if round && (sticky || limbs[0] & 1 == 1) {
            let mut carry = true;
            for limb in limbs.iter_mut() {
                let (v, c) = limb.overflowing_add(carry as u64);
                *limb = v;
                carry = c;
                if !carry {
                    break;
                }
            }
            if carry {
                limbs[N - 1] = 1 << 63;
                e += 1;
            }
        }
        if !Self::exponent_in_range(e) {
            return None;
        }
        Some(Self {
            limbs,
            kind: regular(negative),
            exponent: e as mpfr::exp_t,
        })
    }
    fn binary(
        self,
        b: Self,
        f: unsafe extern "C" fn(
            *mut mpfr::mpfr_t,
            *const mpfr::mpfr_t,
            *const mpfr::mpfr_t,
            mpfr::rnd_t,
        ) -> i32,
    ) -> Self {
        let a = self.descriptor();
        let b = b.descriptor();
        Self::output(|r| unsafe {
            f(r, &a, &b, ROUND);
        })
    }
    fn parse_radix(s: &str, radix: u32) -> Result<Self, ParseError> {
        let error = |message| ParseError {
            precision_bits: Self::PRECISION_BITS,
            message,
        };
        if !(2..=36).contains(&radix) {
            return Err(error("unsupported radix"));
        }
        if s.is_empty() || s.trim() != s {
            return Err(error("invalid number"));
        }
        let s = CString::new(s).map_err(|_| error("embedded NUL in number"))?;
        let mut status = 0;
        let value = Self::output(|r| unsafe {
            status = mpfr::set_str(r, s.as_ptr(), radix as _, ROUND);
        });
        if status != 0 {
            Err(error("invalid number"))
        } else {
            Ok(value)
        }
    }
    /// Decimal scientific notation; absent digit count guarantees binary round-trip.
    pub fn to_decimal(self, digits: Option<usize>) -> String {
        if self.is_nan() {
            return "NaN".into();
        }
        if self.is_infinite() {
            return if self.is_sign_negative() {
                "-inf"
            } else {
                "inf"
            }
            .into();
        }
        if self.is_zero() {
            return if self.is_sign_negative() { "-0" } else { "0" }.into();
        }
        let d = self.descriptor();
        let mut exponent = 0;
        unsafe {
            let ptr = mpfr::get_str(
                std::ptr::null_mut(),
                &mut exponent,
                10,
                digits.map(|n| n.max(1)).unwrap_or(0),
                &d,
                ROUND,
            );
            assert!(!ptr.is_null(), "MPFR decimal formatting failed");
            let mantissa = CStr::from_ptr(ptr).to_string_lossy().into_owned();
            mpfr::free_str(ptr);
            let (sign, m) = if let Some(m) = mantissa.strip_prefix('-') {
                ("-", m)
            } else {
                ("", mantissa.as_str())
            };
            if m.len() == 1 {
                format!("{}{}e{}", sign, m, exponent - 1)
            } else {
                format!("{}{}.{}e{}", sign, &m[..1], &m[1..], exponent - 1)
            }
        }
    }
}
impl<const N: usize> FromStr for MpFloat<N> {
    type Err = ParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_radix(s, 10)
    }
}
impl<const N: usize> fmt::Display for MpFloat<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = self.to_decimal(f.precision().map(|p| p.saturating_add(1)));
        let positive = !text.starts_with('-');
        f.pad_integral(positive, "", text.strip_prefix('-').unwrap_or(&text))
    }
}
impl<const N: usize> fmt::LowerExp for MpFloat<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl<const N: usize> fmt::Debug for MpFloat<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MpFloat<{}>({})", Self::PRECISION_BITS, self)
    }
}
impl<const N: usize> PartialEq for MpFloat<N> {
    #[inline]
    fn eq(&self, b: &Self) -> bool {
        Self::check_precision();
        if self.kind == mpfr::NAN_KIND || b.kind == mpfr::NAN_KIND {
            return false;
        }
        if self.is_zero() && b.is_zero() {
            return true;
        }
        // Regular significands are normalized; special values may carry stale
        // exponent/limb storage, which is not part of their numerical value.
        self.kind == b.kind
            && (self.kind.abs() != mpfr::REGULAR_KIND
                || (self.exponent == b.exponent && self.limbs == b.limbs))
    }
}
impl<const N: usize> PartialOrd for MpFloat<N> {
    #[inline]
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        Self::check_precision();
        if self.kind.abs() == mpfr::REGULAR_KIND && b.kind.abs() == mpfr::REGULAR_KIND {
            if self.kind != b.kind {
                return Some(self.kind.cmp(&b.kind));
            }
            let magnitude = self
                .exponent
                .cmp(&b.exponent)
                .then_with(|| self.limbs.iter().rev().cmp(b.limbs.iter().rev()));
            return Some(if self.kind < 0 {
                magnitude.reverse()
            } else {
                magnitude
            });
        }
        if self.is_nan() || b.is_nan() {
            None
        } else {
            Some(unsafe { mpfr::cmp(&self.descriptor(), &b.descriptor()) }.cmp(&0))
        }
    }
}
macro_rules! binary_op {
    ($trait:ident,$method:ident,$assign:ident,$assign_method:ident,$mp:ident) => {
        impl<const N: usize> $trait for MpFloat<N> {
            type Output = Self;
            fn $method(self, b: Self) -> Self {
                self.binary(b, mpfr::$mp)
            }
        }
        impl<const N: usize> $assign for MpFloat<N> {
            fn $assign_method(&mut self, b: Self) {
                *self = (*self).$method(b);
            }
        }
    };
}
impl<const N: usize> Add for MpFloat<N> {
    type Output = Self;
    #[inline]
    fn add(self, b: Self) -> Self {
        self.add_regular(&b, false)
            .unwrap_or_else(|| self.binary(b, mpfr::add))
    }
}
impl<const N: usize> AddAssign for MpFloat<N> {
    fn add_assign(&mut self, b: Self) {
        *self = *self + b;
    }
}
impl<const N: usize> Sub for MpFloat<N> {
    type Output = Self;
    #[inline]
    fn sub(self, b: Self) -> Self {
        self.add_regular(&b, true)
            .unwrap_or_else(|| self.binary(b, mpfr::sub))
    }
}
impl<const N: usize> SubAssign for MpFloat<N> {
    fn sub_assign(&mut self, b: Self) {
        *self = *self - b;
    }
}
impl<const N: usize> Mul for MpFloat<N> {
    type Output = Self;
    #[inline]
    fn mul(self, b: Self) -> Self {
        self.mul_regular(&b)
            .unwrap_or_else(|| self.binary(b, mpfr::mul))
    }
}
impl<const N: usize> MulAssign for MpFloat<N> {
    fn mul_assign(&mut self, b: Self) {
        *self = *self * b;
    }
}
binary_op!(Div, div, DivAssign, div_assign, div);
binary_op!(Rem, rem, RemAssign, rem_assign, fmod);
impl<const N: usize> Neg for MpFloat<N> {
    type Output = Self;
    fn neg(mut self) -> Self {
        self.kind = -self.kind;
        self
    }
}
impl<const N: usize> Zero for MpFloat<N> {
    fn zero() -> Self {
        Self::default()
    }
    fn is_zero(&self) -> bool {
        self.kind.abs() == mpfr::ZERO_KIND
    }
}
impl<const N: usize> One for MpFloat<N> {
    fn one() -> Self {
        Self::check_precision();
        // GMP limbs are least-significant first; MPFR stores 1 as 0.5 * 2^1.
        let mut limbs = [0; N];
        limbs[N - 1] = 1 << 63;
        Self {
            limbs,
            kind: mpfr::REGULAR_KIND,
            exponent: 1,
        }
    }
}
impl<const N: usize> Num for MpFloat<N> {
    type FromStrRadixErr = ParseError;
    fn from_str_radix(s: &str, r: u32) -> Result<Self, ParseError> {
        Self::parse_radix(s, r)
    }
}
impl<const N: usize> FromPrimitive for MpFloat<N> {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self::output(|r| unsafe {
            mpfr::set_sj(r, n, ROUND);
        }))
    }
    fn from_u64(n: u64) -> Option<Self> {
        Some(Self::output(|r| unsafe {
            mpfr::set_uj(r, n, ROUND);
        }))
    }
    fn from_i128(n: i128) -> Option<Self> {
        n.to_string().parse().ok()
    }
    fn from_u128(n: u128) -> Option<Self> {
        n.to_string().parse().ok()
    }
    fn from_f64(n: f64) -> Option<Self> {
        Some(Self::output(|r| unsafe {
            mpfr::set_d(r, n, ROUND);
        }))
    }
}
impl<const N: usize> ToPrimitive for MpFloat<N> {
    fn to_i64(&self) -> Option<i64> {
        let d = self.descriptor();
        unsafe {
            if mpfr::fits_intmax_p(&d, mpfr::rnd_t::RNDZ) != 0 {
                Some(mpfr::get_sj(&d, mpfr::rnd_t::RNDZ) as i64)
            } else {
                None
            }
        }
    }
    fn to_u64(&self) -> Option<u64> {
        let d = self.descriptor();
        unsafe {
            if mpfr::fits_uintmax_p(&d, mpfr::rnd_t::RNDZ) != 0 {
                Some(mpfr::get_uj(&d, mpfr::rnd_t::RNDZ) as u64)
            } else {
                None
            }
        }
    }
    fn to_f64(&self) -> Option<f64> {
        Some(unsafe { mpfr::get_d(&self.descriptor(), ROUND) })
    }
}
impl<const N: usize> Scalar for MpFloat<N> {
    fn decimal_string(&self) -> String {
        self.to_decimal(None)
    }

    fn mersenne31(&self) -> Option<u32> {
        let view = self.dyadic_view();
        match view.kind {
            DyadicKind::Zero => Some(0),
            DyadicKind::Finite { negative } => Some(exact::mersenne31_dyadic(
                negative,
                &self.limbs,
                view.exponent as i64 - Self::PRECISION_BITS as i64,
            )),
            _ => None,
        }
    }

    fn exact(&self) -> Option<Exact> {
        if *self == Self::zero() {
            return Some(Exact::default());
        }
        if !self.is_finite() || self.exponent.unsigned_abs() > 8192 || N * 64 > 8192 {
            return None;
        }
        let mut value = Exact::default();
        unsafe {
            mpfr::get_q(&mut value.raw, &self.descriptor());
        }
        Some(value)
    }
    fn wire_size() -> Option<usize> {
        crate::wire::mp_size::<N>()
    }
    fn wire_tag() -> u64 {
        crate::wire::mp_tag(Self::PRECISION_BITS)
    }
    fn write_wire(self, out: &mut [u8]) -> bool {
        crate::wire::write_mp(self.kind, self.exponent, &self.limbs, out)
    }
    fn read_wire(bytes: &[u8]) -> Option<Self> {
        let (kind, exponent, limbs) = crate::wire::read_mp::<N>(bytes)?;
        Some(Self::exact_decode(kind, exponent, limbs))
    }
    fn precision_bits() -> usize {
        Self::PRECISION_BITS
    }
    fn epsilon() -> Self {
        Self::output(|r| unsafe {
            mpfr::set_ui_2exp(r, 1, 1 - Self::PRECISION_BITS as mpfr::exp_t, ROUND);
        })
    }
    fn max_value() -> Self {
        Self::output(|r| unsafe {
            mpfr::set_inf(r, 1);
            mpfr::nextbelow(r);
        })
    }
    fn min_value() -> Self {
        -Self::max_value()
    }
    fn min_positive_value() -> Self {
        Self::output(|r| unsafe {
            mpfr::set_ui_2exp(r, 1, mpfr::get_emin() - 1, ROUND);
        })
    }
    fn infinity() -> Self {
        Self {
            kind: mpfr::INF_KIND,
            ..Self::default()
        }
    }
    fn neg_infinity() -> Self {
        -Self::infinity()
    }
    fn nan() -> Self {
        Self {
            kind: mpfr::NAN_KIND,
            ..Self::default()
        }
    }
    fn abs(mut self) -> Self {
        self.kind = self.kind.abs();
        self
    }
    fn sqrt(self) -> Self {
        self.unary(mpfr::sqrt)
    }
    fn cbrt(self) -> Self {
        self.unary(mpfr::cbrt)
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        if let Some(value) = self.fma_regular(&a, &b) {
            return value;
        }
        let x = self.descriptor();
        let y = a.descriptor();
        let z = b.descriptor();
        Self::output(|r| unsafe {
            mpfr::fma(r, &x, &y, &z, ROUND);
        })
    }
    fn dot_fma<'a>(pairs: impl IntoIterator<Item = (&'a Self, &'a Self)>) -> Self {
        MpFloat::dot_fma(pairs)
    }
    fn dot_slices(a: &[Self], b: &[Self]) -> Self {
        exactdot::dot_slices(a, b, |terms| Self::dot_fma_chain(terms))
    }
    fn ln(self) -> Self {
        self.unary(mpfr::log)
    }
    fn exp(self) -> Self {
        self.unary(mpfr::exp)
    }
    fn sin(self) -> Self {
        self.unary(mpfr::sin)
    }
    fn cos(self) -> Self {
        self.unary(mpfr::cos)
    }
    fn atan(self) -> Self {
        self.unary(mpfr::atan)
    }
    fn atan2(self, other: Self) -> Self {
        self.binary(other, mpfr::atan2)
    }
    fn powi(self, n: i32) -> Self {
        let d = self.descriptor();
        Self::output(|r| unsafe {
            mpfr::pow_si(r, &d, n as _, ROUND);
        })
    }
    fn powf(self, n: Self) -> Self {
        self.binary(n, mpfr::pow)
    }
    fn signum(self) -> Self {
        if self.is_nan() {
            self
        } else if self.is_sign_negative() {
            -Self::one()
        } else {
            Self::one()
        }
    }
    fn is_nan(self) -> bool {
        self.kind == mpfr::NAN_KIND
    }
    fn is_infinite(self) -> bool {
        self.kind.abs() == mpfr::INF_KIND
    }
    fn is_finite(self) -> bool {
        self.kind.abs() >= mpfr::ZERO_KIND
    }
    fn is_sign_negative(self) -> bool {
        self.kind < 0
    }
    fn min(self, b: Self) -> Self {
        if b.is_nan() || self < b || (self.is_zero() && b.is_zero() && self.is_sign_negative()) {
            self
        } else {
            b
        }
    }
    fn max(self, b: Self) -> Self {
        if b.is_nan() || self > b || (self.is_zero() && b.is_zero() && !self.is_sign_negative()) {
            self
        } else {
            b
        }
    }
}

// MPFR constants evaluated as enclosing intervals. Increase working precision
// until both endpoints round to the same destination value (Ziv strategy).
// No global/default precision is read or modified.
struct Temp(mpfr::mpfr_t);
impl Temp {
    fn new(bits: usize) -> Self {
        unsafe {
            let mut x = MaybeUninit::uninit();
            mpfr::init2(x.as_mut_ptr(), bits as _);
            Self(x.assume_init())
        }
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        unsafe {
            mpfr::clear(&mut self.0);
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Constant {
    Pi,
    E,
    Sqrt2,
    Ln2,
    Ln10,
    Log2E,
    Log10E,
    Frac1Pi,
    Frac2Pi,
    Frac2SqrtPi,
    Frac1Sqrt2,
    FracPi2,
    FracPi3,
    FracPi4,
    FracPi6,
    FracPi8,
}
// Precision is part of the key. Cached values contain owned limbs, never MPFR
// pointers; copying a result cannot mutate the cache. Thread-local storage
// avoids a lock in the cone/operator pools and does not change MPFR precision.
thread_local! {
    static CONSTANTS: std::cell::RefCell<std::collections::HashMap<(usize, Constant), (i32, i64, Vec<u64>)>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}
impl<const N: usize> MpFloat<N> {
    fn constant(c: Constant) -> Self {
        if let Some(value) = CONSTANTS.with(|cache| {
            cache.borrow().get(&(N, c)).map(|(kind, exponent, data)| {
                let mut limbs = [0; N];
                limbs.copy_from_slice(data);
                Self::exact_decode(*kind, *exponent, limbs)
            })
        }) {
            return value;
        }
        let value = Self::constant_uncached(c);
        CONSTANTS.with(|cache| {
            cache.borrow_mut().insert(
                (N, c),
                (value.kind, value.exponent as i64, value.limbs.to_vec()),
            );
        });
        value
    }

    fn constant_uncached(c: Constant) -> Self {
        let mut bits = Self::PRECISION_BITS + 32;
        loop {
            let mut lo = Temp::new(bits);
            let mut hi = Temp::new(bits);
            for (x, rnd) in [
                (&mut lo.0, mpfr::rnd_t::RNDD),
                (&mut hi.0, mpfr::rnd_t::RNDU),
            ] {
                unsafe {
                    use Constant::*;
                    let inverse = matches!(
                        c,
                        Log2E | Log10E | Frac1Pi | Frac2Pi | Frac2SqrtPi | Frac1Sqrt2
                    );
                    let base_rnd = if inverse {
                        if matches!(rnd, mpfr::rnd_t::RNDD) {
                            mpfr::rnd_t::RNDU
                        } else {
                            mpfr::rnd_t::RNDD
                        }
                    } else {
                        rnd
                    };
                    match c {
                        E => {
                            mpfr::set_ui(x, 1, base_rnd);
                            mpfr::exp(x, x, base_rnd);
                        }
                        Sqrt2 | Frac1Sqrt2 => {
                            mpfr::set_ui(x, 2, base_rnd);
                            mpfr::sqrt(x, x, base_rnd);
                        }
                        Ln2 | Log2E => {
                            mpfr::const_log2(x, base_rnd);
                        }
                        Ln10 | Log10E => {
                            mpfr::set_ui(x, 10, base_rnd);
                            mpfr::log(x, x, base_rnd);
                        }
                        _ => {
                            mpfr::const_pi(x, base_rnd);
                        }
                    }
                    if matches!(c, Frac2SqrtPi) {
                        mpfr::sqrt(x, x, base_rnd);
                    }
                    if inverse {
                        mpfr::ui_div(
                            x,
                            if matches!(c, Frac2Pi | Frac2SqrtPi) {
                                2
                            } else {
                                1
                            },
                            x,
                            rnd,
                        );
                    }
                    let divisor = match c {
                        FracPi2 => 2,
                        FracPi3 => 3,
                        FracPi4 => 4,
                        FracPi6 => 6,
                        FracPi8 => 8,
                        _ => 1,
                    };
                    if divisor != 1 {
                        mpfr::div_ui(x, x, divisor, rnd);
                    }
                }
            }
            let a = Self::output(|r| unsafe {
                mpfr::set(r, &lo.0, ROUND);
            });
            let b = Self::output(|r| unsafe {
                mpfr::set(r, &hi.0, ROUND);
            });
            if a == b {
                return a;
            }
            bits = bits.checked_mul(2).expect("constant precision overflow");
        }
    }
}
#[allow(non_snake_case)]
impl<const N: usize> FloatConst for MpFloat<N> {
    fn E() -> Self {
        Self::constant(Constant::E)
    }
    fn FRAC_1_PI() -> Self {
        Self::constant(Constant::Frac1Pi)
    }
    fn FRAC_1_SQRT_2() -> Self {
        Self::constant(Constant::Frac1Sqrt2)
    }
    fn FRAC_2_PI() -> Self {
        Self::constant(Constant::Frac2Pi)
    }
    fn FRAC_2_SQRT_PI() -> Self {
        Self::constant(Constant::Frac2SqrtPi)
    }
    fn FRAC_PI_2() -> Self {
        Self::constant(Constant::FracPi2)
    }
    fn FRAC_PI_3() -> Self {
        Self::constant(Constant::FracPi3)
    }
    fn FRAC_PI_4() -> Self {
        Self::constant(Constant::FracPi4)
    }
    fn FRAC_PI_6() -> Self {
        Self::constant(Constant::FracPi6)
    }
    fn FRAC_PI_8() -> Self {
        Self::constant(Constant::FracPi8)
    }
    fn LN_10() -> Self {
        Self::constant(Constant::Ln10)
    }
    fn LN_2() -> Self {
        Self::constant(Constant::Ln2)
    }
    fn LOG10_E() -> Self {
        Self::constant(Constant::Log10E)
    }
    fn LOG2_E() -> Self {
        Self::constant(Constant::Log2E)
    }
    fn PI() -> Self {
        Self::constant(Constant::Pi)
    }
    fn SQRT_2() -> Self {
        Self::constant(Constant::Sqrt2)
    }
}

// Inner-operator parallelism gate. Solver-pool lanes set this flag so heavy
// kernels (e.g. the MPFR SVD) may re-offer independent inner work to the
// ambient Rayon pool. It lives here because the BLAS sources are `include!`d
// into standalone test crates where solver-internal paths do not resolve.
#[doc(hidden)]
pub mod inner_parallel {
    use std::cell::Cell;

    std::thread_local! {
        static ENABLED: Cell<bool> = const { Cell::new(false) };
        static PAIRED: Cell<bool> = const { Cell::new(false) };
    }

    /// True while this thread is running a lane on the solver pool and the
    /// pool has enough spare workers for multi-way inner splits.
    #[inline]
    pub fn active() -> bool {
        ENABLED.with(Cell::get)
    }

    /// True while any spare worker exists beyond the cone lanes: a paired
    /// `rayon::join` of two independent operations pays off with a single
    /// stealer, so it needs far less headroom than column-level splits.
    #[inline]
    pub fn paired() -> bool {
        PAIRED.with(Cell::get)
    }

    /// Work, in 64-bit limb products, that one extra pool task must carry
    /// before an inner region is split.
    pub const GRAIN: u128 = 1 << 15;

    /// The inner-split grain ([`GRAIN`]).
    #[inline]
    pub fn grain() -> u128 {
        GRAIN
    }

    /// Limb-product weight of one multiply-add at `bits` of precision.
    #[inline]
    pub fn weight(bits: usize) -> u128 {
        let limbs = bits.div_ceil(64).max(1) as u128;
        limbs * limbs
    }

    /// Tasks worth creating for `work` limb products spread over `items`
    /// independent pieces when `enabled`: at least one grain per task, at
    /// most one task per piece. One means stay serial.
    #[inline]
    pub fn tasks_if(enabled: bool, work: u128, items: usize) -> usize {
        if !enabled || items < 2 {
            return 1;
        }
        (work / grain().max(1)).min(items as u128).max(1) as usize
    }

    /// [`tasks_if`] under the inner-parallel gate ([`active`]).
    #[inline]
    pub fn tasks(work: u128, items: usize) -> usize {
        tasks_if(active(), work, items)
    }

    /// RAII guard enabling [`active`]/[`paired`] until dropped.
    pub struct Guard(bool, bool);

    impl Guard {
        /// Enable both levels (heavy inner splits and pairing).
        pub fn enter() -> Self {
            Self::enter_levels(true, true)
        }

        /// Enable each level independently.
        pub fn enter_levels(inner: bool, paired: bool) -> Self {
            Self(
                ENABLED.with(|c| c.replace(inner)),
                PAIRED.with(|c| c.replace(paired)),
            )
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            ENABLED.with(|c| c.set(self.0));
            PAIRED.with(|c| c.set(self.1));
        }
    }
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
