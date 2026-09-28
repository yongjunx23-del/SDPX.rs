//! Shared exact precision dispatch for the CLI and C ABI.

/// Precision choices built into the executable frontends. The native Rust API
/// also accepts `MpFloat<N>` outside this range.
pub const FRONTEND_PRECISION_HELP: &str =
    "53 (Float64), or multiples of 64 from 128 through 2048 (MPFR)";

/// Invoke an internal callback with `(bits, variant, scalar)` entries.
#[doc(hidden)]
#[macro_export]
macro_rules! with_frontend_precisions {
    ($callback:ident $(, $arg:tt)*) => {
        $callback! {
            [$($arg),*]
            (53, F64, f64),
            (128, B128, $crate::MpFloat<2>),
            (192, B192, $crate::MpFloat<3>),
            (256, B256, $crate::MpFloat<4>),
            (320, B320, $crate::MpFloat<5>),
            (384, B384, $crate::MpFloat<6>),
            (448, B448, $crate::MpFloat<7>),
            (512, B512, $crate::MpFloat<8>),
            (576, B576, $crate::MpFloat<9>),
            (640, B640, $crate::MpFloat<10>),
            (704, B704, $crate::MpFloat<11>),
            (768, B768, $crate::MpFloat<12>),
            (832, B832, $crate::MpFloat<13>),
            (896, B896, $crate::MpFloat<14>),
            (960, B960, $crate::MpFloat<15>),
            (1024, B1024, $crate::MpFloat<16>),
            (1088, B1088, $crate::MpFloat<17>),
            (1152, B1152, $crate::MpFloat<18>),
            (1216, B1216, $crate::MpFloat<19>),
            (1280, B1280, $crate::MpFloat<20>),
            (1344, B1344, $crate::MpFloat<21>),
            (1408, B1408, $crate::MpFloat<22>),
            (1472, B1472, $crate::MpFloat<23>),
            (1536, B1536, $crate::MpFloat<24>),
            (1600, B1600, $crate::MpFloat<25>),
            (1664, B1664, $crate::MpFloat<26>),
            (1728, B1728, $crate::MpFloat<27>),
            (1792, B1792, $crate::MpFloat<28>),
            (1856, B1856, $crate::MpFloat<29>),
            (1920, B1920, $crate::MpFloat<30>),
            (1984, B1984, $crate::MpFloat<31>),
            (2048, B2048, $crate::MpFloat<32>),
        }
    };
}
