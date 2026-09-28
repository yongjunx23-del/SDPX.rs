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
pub use precisions::{DEFAULT_FRONTEND_PRECISION_HELP, FRONTEND_PRECISION_HELP};

mod dyadic;
pub use dyadic::{DyadicKind, DyadicView};
mod exact;
mod exactdot;
/// Exact-integer reference conversions. Test-only oracle for the residue
/// kernels (see the module docs): not part of the production arithmetic.
#[cfg(test)]
mod integer;
pub use exact::Exact;
mod rns;
pub use rns::{EncodeSide, Residues, RnsPlan};
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
    /// Exact residue-domain dot support (arbitrary-precision backends only).
    /// The default reports `None` so callers keep `dot_fma`.
    #[doc(hidden)]
    fn rns_plan(_a: &[Self], _b: &[Self], _terms: usize) -> Option<crate::rns::RnsPlan> {
        None
    }
    /// [`Scalar::rns_plan`] variant with precomputed `(lo, hi)` exponent ranges.
    #[doc(hidden)]
    fn rns_plan_ranges(
        _a_range: (i64, i64),
        _b_range: (i64, i64),
        _terms: usize,
    ) -> Option<crate::rns::RnsPlan> {
        None
    }
    #[doc(hidden)]
    fn rns_encode(
        _plan: &crate::rns::RnsPlan,
        _side: crate::rns::EncodeSide,
        _m: &[Self],
    ) -> Option<crate::rns::Residues> {
        None
    }
    /// Full-width encode for cached constant operands; see
    /// [`crate::rns::RnsPlan::encode_wide`].
    #[doc(hidden)]
    fn rns_encode_wide(
        _plan: &crate::rns::RnsPlan,
        _side: crate::rns::EncodeSide,
        _m: &[Self],
    ) -> Option<crate::rns::Residues> {
        None
    }
    #[doc(hidden)]
    fn rns_dot(
        _plan: &crate::rns::RnsPlan,
        _ra: &crate::rns::Residues,
        _a0: usize,
        _da: usize,
        _rb: &crate::rns::Residues,
        _b0: usize,
        _db: usize,
        _terms: usize,
    ) -> Self {
        unimplemented!("rns_dot requires rns_plan support")
    }
    /// Reusable exponent range scan shared by `rns_plan` implementations.
    #[doc(hidden)]
    fn rns_exponent_range(_m: &[Self]) -> Option<(i64, i64)> {
        None
    }
    /// Lossless value encoding for solver-state snapshots: `(kind, exponent,
    /// limbs)`. `None` (the default) means the type has no exact snapshot
    /// encoding. Only concrete payload words are serialized — never
    /// descriptor pointers.
    #[doc(hidden)]
    fn scalar_exact_encode(&self) -> Option<(i32, i64, Vec<u64>)> {
        None
    }
    /// Inverse of [`Scalar::scalar_exact_encode`].
    #[doc(hidden)]
    fn scalar_exact_decode(_kind: i32, _exponent: i64, _limbs: &[u64]) -> Option<Self> {
        None
    }
}
macro_rules! primitive_scalar {
    ($t:ty) => {
        impl Scalar for $t {
            fn exact(&self) -> Option<Exact> {
                self.is_finite().then(|| Exact::from_f64(*self as f64))
            }
            fn scalar_exact_encode(&self) -> Option<(i32, i64, Vec<u64>)> {
                Some((0, 0, vec![self.to_bits() as u64]))
            }
            fn scalar_exact_decode(kind: i32, exponent: i64, limbs: &[u64]) -> Option<Self> {
                if kind == 0 && exponent == 0 && limbs.len() == 1 {
                    Some(Self::from_bits(limbs[0] as _))
                } else {
                    None
                }
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
            fn precision_bits() -> usize {
                Self::MANTISSA_DIGITS as usize
            }
            fn epsilon() -> Self {
                Self::EPSILON
            }
            fn max_value() -> Self {
                Self::MAX
            }
            fn min_value() -> Self {
                Self::MIN
            }
            fn min_positive_value() -> Self {
                Self::MIN_POSITIVE
            }
            fn infinity() -> Self {
                Self::INFINITY
            }
            fn neg_infinity() -> Self {
                Self::NEG_INFINITY
            }
            fn nan() -> Self {
                Self::NAN
            }
            fn abs(self) -> Self {
                self.abs()
            }
            fn sqrt(self) -> Self {
                self.sqrt()
            }
            fn cbrt(self) -> Self {
                self.cbrt()
            }
            fn mul_add(self, a: Self, b: Self) -> Self {
                self.mul_add(a, b)
            }
            fn ln(self) -> Self {
                self.ln()
            }
            fn exp(self) -> Self {
                self.exp()
            }
            fn sin(self) -> Self {
                self.sin()
            }
            fn cos(self) -> Self {
                self.cos()
            }
            fn atan(self) -> Self {
                self.atan()
            }
            fn atan2(self, other: Self) -> Self {
                self.atan2(other)
            }
            fn powi(self, n: i32) -> Self {
                self.powi(n)
            }
            fn powf(self, n: Self) -> Self {
                self.powf(n)
            }
            fn signum(self) -> Self {
                self.signum()
            }
            fn is_nan(self) -> bool {
                self.is_nan()
            }
            fn is_infinite(self) -> bool {
                self.is_infinite()
            }
            fn is_finite(self) -> bool {
                self.is_finite()
            }
            fn is_sign_negative(self) -> bool {
                self.is_sign_negative()
            }
            fn min(self, b: Self) -> Self {
                self.min(b)
            }
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

    /// Lossless snapshot encoding: `(kind, exponent, limbs)`. Round-trips
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
        let (a, b, c, d) = (
            a.descriptor(),
            b.descriptor(),
            c.descriptor(),
            d.descriptor(),
        );
        Self::output(|r| unsafe {
            mpfr::fmma(r, &a, &b, &c, &d, ROUND);
        })
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
        {
            let text = self.to_decimal(f.precision().map(|p| p.saturating_add(1)));
            let positive = !text.starts_with('-');
            f.pad_integral(positive, "", text.strip_prefix('-').unwrap_or(&text))
        }
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
binary_op!(Add, add, AddAssign, add_assign, add);
binary_op!(Sub, sub, SubAssign, sub_assign, sub);
binary_op!(Mul, mul, MulAssign, mul_assign, mul);
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
    fn scalar_exact_encode(&self) -> Option<(i32, i64, Vec<u64>)> {
        let (k, e, l) = self.exact_encode();
        Some((k, e, l.to_vec()))
    }
    fn scalar_exact_decode(kind: i32, exponent: i64, limbs: &[u64]) -> Option<Self> {
        let a: [u64; N] = limbs.try_into().ok()?;
        Some(Self::exact_decode(kind, exponent, a))
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
    fn rns_plan(a: &[Self], b: &[Self], terms: usize) -> Option<crate::rns::RnsPlan> {
        crate::rns::RnsPlan::for_pair(a, b, terms)
    }
    fn rns_plan_ranges(
        a_range: (i64, i64),
        b_range: (i64, i64),
        terms: usize,
    ) -> Option<crate::rns::RnsPlan> {
        crate::rns::RnsPlan::for_ranges::<N>(a_range, b_range, terms)
    }
    fn rns_encode(
        plan: &crate::rns::RnsPlan,
        side: crate::rns::EncodeSide,
        m: &[Self],
    ) -> Option<crate::rns::Residues> {
        plan.encode(m, side)
    }
    fn rns_encode_wide(
        plan: &crate::rns::RnsPlan,
        side: crate::rns::EncodeSide,
        m: &[Self],
    ) -> Option<crate::rns::Residues> {
        plan.encode_wide(m, side)
    }
    fn rns_dot(
        plan: &crate::rns::RnsPlan,
        ra: &crate::rns::Residues,
        a0: usize,
        da: usize,
        rb: &crate::rns::Residues,
        b0: usize,
        db: usize,
        terms: usize,
    ) -> Self {
        plan.dot::<N>(ra, a0, da, rb, b0, db, terms)
    }
    fn rns_exponent_range(m: &[Self]) -> Option<(i64, i64)> {
        crate::rns::exponent_range(m)
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
        if b.is_nan() || self < b {
            self
        } else if self.is_zero() && b.is_zero() && self.is_sign_negative() {
            self
        } else {
            b
        }
    }
    fn max(self, b: Self) -> Self {
        if b.is_nan() || self > b {
            self
        } else if self.is_zero() && b.is_zero() && !self.is_sign_negative() {
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
