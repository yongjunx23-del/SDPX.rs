// Solver sources included via #[path] (see the mpfr_* and rns_scaling
// modules) resolve `crate::algebra` / `crate::receipt` against this root.
#[allow(unused_imports)]
use sdpx_solver::{algebra, receipt};

mod api_dimension_checks;
mod basic_eq_constrained;
mod basic_expcone;
mod basic_genpowcone;
mod basic_lp;
mod basic_powcone;
mod basic_qp;
mod basic_sdp;
mod basic_socp;
mod basic_unconstrained;
mod callbacks;
mod checkpoint;
mod data_updating;
mod direction_checks;
mod equilibration_bounds;
mod json_io;
mod mixed_conic;
mod mpfr_dense;
mod mpfr_gemm_syrk;
mod mpfr_parallel;
mod mpfr_workspace;
mod native_input;
mod pmp_solve;
mod presolve;
mod print_streams;
mod rns_scaling;
mod sampled_integration;
mod sampled_solver;
mod sdp_chordal;
