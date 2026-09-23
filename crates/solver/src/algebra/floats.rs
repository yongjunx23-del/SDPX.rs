#![allow(non_snake_case)]
use crate::solver::kkt::ldl::config::LDLConfiguration;
use num_traits::FromPrimitive;
use sdpx_arithmetic::Scalar;

#[cfg(feature = "sdp")]
use crate::algebra::dense::BlasFloatT;

/// Arithmetic required by the single solver engine, including owned MPFR scalars.
pub trait CoreFloatT: Scalar {}
impl<T: Scalar> CoreFloatT for T {}

cfg_if::cfg_if! {
    if #[cfg(feature="sdp")] {
        /// Scalar types with a complete dense SDP numerical provider.
        #[doc(hidden)]
        pub trait MaybeBlasFloatT: BlasFloatT {}
        impl<T: BlasFloatT> MaybeBlasFloatT for T {}
    } else {
        #[doc(hidden)]
        pub trait MaybeBlasFloatT {}
        impl<T> MaybeBlasFloatT for T {}
    }
}

/// Arithmetic, dense operations and sparse factorization supported by SDPX.
/// Faer's RealField requirement is local to its provider, not the solver engine.
pub trait FloatT: CoreFloatT + MaybeBlasFloatT + LDLConfiguration {}
impl<T: CoreFloatT + MaybeBlasFloatT + LDLConfiguration> FloatT for T {}

/// Trait for converting Rust primitives to [`FloatT`](crate::algebra::FloatT)
///
/// This convenience trait is implemented on f32/64 and u32/64.  This trait
/// is required internally by the solver for converting constant primitives
/// to [`FloatT`](crate::algebra::FloatT).  It is also used by the
/// [user settings](crate::solver::default::DefaultSettings)
/// for converting defaults of primitive type to [`FloatT`](crate::algebra::FloatT).
//
// NB: `AsFloatT` is a convenience trait for f32/64 and u32/64
// so that we can do things like (2.0).as_T() everywhere on
// constants, rather than the awful T::from_f32(2.0).unwrap()
pub(crate) trait AsFloatT<T>: 'static {
    fn as_T(&self) -> T;
}

macro_rules! impl_as_FloatT {
    ($ty:ty, $ident:ident) => {
        impl<T> AsFloatT<T> for $ty
        where
            T: std::ops::Mul<T, Output = T> + FromPrimitive + 'static,
        {
            #[inline]
            fn as_T(&self) -> T {
                T::$ident(*self).unwrap()
            }
        }
    };
}
impl_as_FloatT!(u32, from_u32);
impl_as_FloatT!(u64, from_u64);
impl_as_FloatT!(usize, from_usize);
impl_as_FloatT!(f32, from_f32);
impl_as_FloatT!(f64, from_f64);
