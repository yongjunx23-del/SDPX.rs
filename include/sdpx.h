#ifndef SDPX_H
#define SDPX_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* ABI 4. ABI 1/2/3 settings are rejected; settings/info sizes are 88/112 bytes
 * on 64-bit platforms. All pointers must reference valid aligned storage for their stated
 * lengths for the duration of the call. Output storage must not overlap input
 * storage or other outputs. Inputs are copied. Indices are zero
 * based, sorted and unique within each CSC column. P is upper triangular.
 * A handle is owned by its creator: destroy once, never use after destroy;
 * destroy must not race another call. Other handle operations reject overlap.
 * Return codes: 0 OK, 1 invalid input, 2 unsupported, 3 busy, 4 poisoned,
 * 5 panic, 6 insufficient output capacity, 7 no solved result.
 * Solver status is independent of ABI return codes (see sdpx_info.status).
 */
#define SDPX_ABI_VERSION 4u
#define SDPX_PREPROCESS_RUIZ 1u
#define SDPX_PREPROCESS_PRESOLVE 2u
#define SDPX_PREPROCESS_CHORDAL 4u
#define SDPX_PREPROCESS_ALL 7u

typedef struct sdpx_handle sdpx_handle;
typedef struct { uint32_t kind; uint32_t reserved; uint64_t count; const double *f64; const char *const *decimal; } sdpx_scalars; /* kind 0=f64, 1=decimal */
typedef struct { uint64_t rows, cols, nnz; const uint64_t *colptr, *rowval; sdpx_scalars values; } sdpx_csc;
/* kind 0 zero,1 nonnegative,2 SOC,3 PSD triangle,4 exponential,5 power,
 * 6 generalized power. dim is row count except PSD matrix order and GENPOW
 * right-hand Euclidean dimension. POW has one alpha, GENPOW has alpha vector.
 * PSD entries use upper triangular column order, sqrt(2) off-diagonal scaling. */
typedef struct { uint32_t kind, reserved; uint64_t dim; sdpx_scalars alpha; } sdpx_cone;
/* Factor-authoritative PSD block: starts are zero based; basis is column major.
 * There are dim*(dim+1)/2*basis_cols consecutive variables, with primitive
 * order s=0..dim-1, r=0..s, k=0..basis_cols-1. Each column is weights[p]
 * times svec(sym(e_r e_s') tensor (basis[:,k] basis[:,k]')).
 * The block occupies one PSD cone of order dim*basis_rows. A_linear must
 * be zero on those rows. dim and basis_rows must be positive; basis_cols may
 * be zero with empty basis and weights. Dimensions and cone alignment are validated. */
typedef struct {
 uint64_t row_start, column_start, dim, basis_rows, basis_cols;
 sdpx_scalars basis, weights;
} sdpx_sampled_block;
typedef struct {
 uint32_t abi_version, struct_size, precision_bits, max_iter;
 /* max_threads budgets the cone worker pool and eligible KKT factorization.
  * It is not a total process thread limit. Native BLAS threads are separate.
  * kkt_form: 0 auto, 1 augmented, 2 condensed. */
 /* preprocessing_flags: bit 0 Ruiz equilibration, bit 1 presolve,
  * bit 2 chordal decomposition. Defaults to SDPX_PREPROCESS_ALL (7).
  * Clear individual bits to disable; unknown bits are rejected. */
 uint32_t verbose, preprocessing_flags, max_threads, kkt_form;
 uint32_t reserved_0; /* Reserved ABI slot; must be zero. */
 double time_limit;
 /* NULL tolerances select precision-aware defaults. All decimal strings. */
 const char *tol_gap_abs, *tol_gap_rel, *tol_feas, *tol_infeas_abs, *tol_infeas_rel;
} sdpx_settings;
typedef struct {
 uint32_t abi_version, struct_size, status, iterations;
 /* Actual factorization threads, cone pool workers, and KKT form (1 or 2).
  * High precision uses serial QDLDL with independently parallel cone work. */
 uint32_t working_bits, backend_threads, cone_threads, kkt_form;
 uint32_t reserved_0; /* Reserved ABI slot; always zero. */
 uint64_t n, m;
 double solve_time, objective, dual_objective, primal_residual, dual_residual, gap_abs, gap_rel;
} sdpx_info;
/* Status: 0 unsolved,1 solved,2 primal infeasible,3 dual infeasible,
 * 4 almost solved,5 almost primal infeasible,6 almost dual infeasible,
 * 7 max iterations,8 max time,9 numerical error,10 insufficient progress,
 * 11 callback terminated. Info floating scalars are approximate summaries.
 * Exact numeric results are obtained through decimal bulk output below. */
/* Versioned first handshake prevents old clients from allocating ABI-3 storage. */
int32_t sdpx_default_settings_v4(sdpx_settings *out);
#define sdpx_default_settings sdpx_default_settings_v4
int32_t sdpx_prepare(const sdpx_csc *P, const sdpx_scalars *q, const sdpx_csc *A, const sdpx_scalars *b, const sdpx_cone *cones, uint64_t cone_count, const sdpx_settings *settings, sdpx_handle **out);
/* Sampled input entrypoint, introduced in ABI 3; uses current settings layout.
 * Factor arrays and blocks are copied before return, as with CSC inputs. */
int32_t sdpx_prepare_sampled(const sdpx_csc *P, const sdpx_scalars *q, const sdpx_csc *A_linear, const sdpx_scalars *b, const sdpx_cone *cones, uint64_t cone_count, const sdpx_settings *settings, const sdpx_sampled_block *blocks, uint64_t block_count, sdpx_handle **out);
int32_t sdpx_solve(sdpx_handle *handle);
/* Both q and b required; dimensions fixed. Successful update invalidates result.
 * prepare may change the internal structure when preprocessing is enabled.
 * Updates retain upstream rejection if presolve/chordal actually changed it.
 * To guarantee reusable q/b updates, clear SDPX_PREPROCESS_PRESOLVE and
 * SDPX_PREPROCESS_CHORDAL before prepare; Ruiz may remain enabled. */
int32_t sdpx_update(sdpx_handle *handle, const sdpx_scalars *q, const sdpx_scalars *b);
int32_t sdpx_get_info(sdpx_handle *handle, sdpx_info *out);
/* Actual factorization name (e.g. qdldl, faer, condensed_qdldl).
 * Available after prepare; locks the handle like other handle operations.
 * Query with NULL,0. required receives bytes including trailing NUL.
 * Insufficient capacity returns 6 without copying a partial name. */
int32_t sdpx_get_solver_name(sdpx_handle *handle, char *out, uint64_t capacity, uint64_t *required);
/* Result order: x[n], z[m], s[m], objective, dual_objective,
 * primal_residual, dual_residual, gap_abs, gap_rel. count = n+2m+6.
 * f64 bulk output converts explicitly; decimal is a NUL-separated UTF-8
 * stream in the same order (including a trailing NUL). Query with NULL,0.
 * required always receives number of doubles/bytes required. */
int32_t sdpx_result_f64(sdpx_handle *handle, double *out, uint64_t capacity, uint64_t *required);
int32_t sdpx_result_decimal(sdpx_handle *handle, char *out, uint64_t capacity, uint64_t *required);
/* Thread-local most recent error: copy NUL-terminated text; returns bytes
 * including NUL needed, truncates safely when capacity is smaller. */
uint64_t sdpx_last_error(char *out, uint64_t capacity);
int32_t sdpx_destroy(sdpx_handle *handle);
#ifdef __cplusplus
}
#endif
#endif
