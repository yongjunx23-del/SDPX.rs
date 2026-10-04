# SDPX solver

One generic Rust HSD interior-point engine for Float64 and MPFR, adapted
from [Clarabel.rs](https://github.com/oxfordcontrol/Clarabel.rs). This crate
provides the native Rust API and `sdpx` CLI; `sdpx-ffi` provides the C ABI.

See the [workspace README](../../README.md) for setup and usage,
[Rust examples](examples/rust/) for API examples, and
[architecture](../../docs/ARCHITECTURE.md) for arithmetic, KKT backends and
build features. BLAS/LAPACK are linked in every solver build.

Apache-2.0. Upstream attribution, source mapping and original hashes are
retained in [provenance](../../provenance/).
