//! Rust-owned SDPX ABI. See include/sdpx.h for pointer and ownership contracts.
use sdpx_arithmetic::MpFloat;
use sdpx_solver::solver::traits::Settings as SolverSettings;
use sdpx_solver::{
    algebra::{CscMatrix, FloatT},
    solver::{
        DefaultSettings, DefaultSolver, IPSolver, SampledBlock as CoreSampledBlock, SupportedConeT,
    },
};
use std::{
    cell::RefCell,
    ffi::{c_char, CStr},
    panic::{catch_unwind, AssertUnwindSafe},
    ptr, slice,
    sync::{Mutex, TryLockError},
};

const ABI_VERSION: u32 = 3;
const PREPROCESS_RUIZ: u32 = 1;
const PREPROCESS_PRESOLVE: u32 = 2;
const PREPROCESS_CHORDAL: u32 = 4;
const PREPROCESS_ALL: u32 = PREPROCESS_RUIZ | PREPROCESS_PRESOLVE | PREPROCESS_CHORDAL;

type Result<T> = std::result::Result<T, (i32, String)>;
fn invalid(s: impl Into<String>) -> (i32, String) {
    (1, s.into())
}
fn peak_rss_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmHWM"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}
thread_local! { static ERROR: RefCell<String> = const { RefCell::new(String::new()) }; }
fn boundary(f: impl FnOnce() -> Result<()>) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        Ok(Err((c, s))) => {
            ERROR.with(|e| *e.borrow_mut() = s);
            c
        }
        Err(_) => {
            ERROR.with(|e| *e.borrow_mut() = "Rust panic at ABI boundary".into());
            5
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Scalars {
    pub kind: u32,
    pub reserved: u32,
    pub count: u64,
    pub f64: *const f64,
    pub decimal: *const *const c_char,
}
#[repr(C)]
pub struct Csc {
    pub rows: u64,
    pub cols: u64,
    pub nnz: u64,
    pub colptr: *const u64,
    pub rowval: *const u64,
    pub values: Scalars,
}
#[repr(C)]
pub struct Cone {
    pub kind: u32,
    pub reserved: u32,
    pub dim: u64,
    pub alpha: Scalars,
}
/// Factor-authoritative sampled PSD block. Starts are zero based.
#[repr(C)]
pub struct SampledBlock {
    pub row_start: u64,
    pub column_start: u64,
    pub dim: u64,
    pub basis_rows: u64,
    pub basis_cols: u64,
    pub basis: Scalars,
    pub weights: Scalars,
}
unsafe fn sampled_blocks<T: Scalar>(input: &[SampledBlock]) -> Result<Vec<CoreSampledBlock<T>>> {
    input
        .iter()
        .map(|block| {
            let index = |v| usize::try_from(v).map_err(|_| invalid("sampled dimension overflow"));
            let row_start = index(block.row_start)?;
            let column_start = index(block.column_start)?;
            let dim = index(block.dim)?;
            let basis_rows = index(block.basis_rows)?;
            let basis_cols = index(block.basis_cols)?;
            if dim == 0 || basis_rows == 0 {
                return Err(invalid("sampled dim and basis_rows must be positive"));
            }
            let basis_count = basis_rows
                .checked_mul(basis_cols)
                .ok_or_else(|| invalid("sampled basis length overflow"))?;
            let primitive_count = dim
                .checked_add(1)
                .and_then(|d| dim.checked_mul(d))
                .map(|d| d / 2)
                .and_then(|d| d.checked_mul(basis_cols))
                .ok_or_else(|| invalid("sampled weights length overflow"))?;
            if block.basis.count != basis_count as u64
                || block.weights.count != primitive_count as u64
            {
                return Err(invalid("inconsistent sampled factor lengths"));
            }
            Ok(CoreSampledBlock {
                row_start,
                column_start,
                dim,
                basis_rows,
                basis_cols,
                basis: scalars(&block.basis)?,
                weights: scalars(&block.weights)?,
            })
        })
        .collect()
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Settings {
    pub abi_version: u32,
    pub struct_size: u32,
    pub precision_bits: u32,
    pub max_iter: u32,
    pub verbose: u32,
    pub preprocessing_flags: u32,
    pub max_threads: u32,
    pub kkt_form: u32,
    pub time_limit: f64,
    pub tol_gap_abs: *const c_char,
    pub tol_gap_rel: *const c_char,
    pub tol_feas: *const c_char,
    pub tol_infeas_abs: *const c_char,
    pub tol_infeas_rel: *const c_char,
}
#[repr(C)]
pub struct Info {
    pub abi_version: u32,
    pub struct_size: u32,
    pub status: u32,
    pub iterations: u32,
    pub working_bits: u32,
    pub backend_threads: u32,
    pub cone_threads: u32,
    pub kkt_form: u32,
    pub n: u64,
    pub m: u64,
    pub solve_time: f64,
    pub objective: f64,
    pub dual_objective: f64,
    pub primal_residual: f64,
    pub dual_residual: f64,
    pub gap_abs: f64,
    pub gap_rel: f64,
}
fn pointer<T>(p: *const T) -> Result<()> {
    if p.is_null() || (p as usize) % std::mem::align_of::<T>() != 0 {
        Err(invalid("null or misaligned pointer"))
    } else {
        Ok(())
    }
}
unsafe fn reference<'a, T>(p: *const T) -> Result<&'a T> {
    pointer(p)?;
    Ok(&*p)
}
unsafe fn array<'a, T>(p: *const T, n: u64) -> Result<&'a [T]> {
    let n = usize::try_from(n).map_err(|_| invalid("length overflow"))?;
    if n > isize::MAX as usize / std::mem::size_of::<T>() {
        return Err(invalid("array too large"));
    }
    if n == 0 {
        return Ok(&[]);
    };
    reference(p)?;
    Ok(slice::from_raw_parts(p, n))
}
unsafe fn string<'a>(p: *const c_char) -> Result<&'a str> {
    if p.is_null() {
        return Err(invalid("null decimal string"));
    }
    CStr::from_ptr(p)
        .to_str()
        .map_err(|_| invalid("decimal is not UTF-8"))
}
trait Scalar: FloatT {
    fn decimal(s: &str) -> Result<Self>;
}
impl Scalar for f64 {
    fn decimal(s: &str) -> Result<Self> {
        s.parse().map_err(|_| invalid("invalid decimal"))
    }
}
impl<const N: usize> Scalar for MpFloat<N>
where
    MpFloat<N>: FloatT,
{
    fn decimal(s: &str) -> Result<Self> {
        s.parse().map_err(|_| invalid("invalid decimal"))
    }
}
unsafe fn scalars<T: Scalar>(a: &Scalars) -> Result<Vec<T>> {
    if a.reserved != 0 {
        return Err(invalid("nonzero scalar reserved field"));
    }
    let v: Vec<T> = match a.kind {
        0 => array(a.f64, a.count)?
            .iter()
            .map(|x| T::from_f64(*x).ok_or_else(|| invalid("numeric conversion failed")))
            .collect::<Result<_>>()?,
        1 => array(a.decimal, a.count)?
            .iter()
            .map(|p| T::decimal(string(*p)?))
            .collect::<Result<_>>()?,
        _ => return Err(invalid("unknown scalar encoding")),
    };
    if v.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite input"));
    }
    Ok(v)
}
unsafe fn matrix<T: Scalar>(a: &Csc, upper: bool) -> Result<CscMatrix<T>> {
    let m = usize::try_from(a.rows).map_err(|_| invalid("rows overflow"))?;
    let n = usize::try_from(a.cols).map_err(|_| invalid("cols overflow"))?;
    let cp = array(
        a.colptr,
        a.cols
            .checked_add(1)
            .ok_or_else(|| invalid("cols overflow"))?,
    )?;
    let ri = array(a.rowval, a.nnz)?;
    if a.values.count != a.nnz || cp.first() != Some(&0) || cp.last() != Some(&a.nnz) {
        return Err(invalid("inconsistent CSC lengths"));
    }
    for j in 0..n {
        if cp[j] > cp[j + 1] || cp[j + 1] > a.nnz {
            return Err(invalid("invalid CSC column pointers"));
        }
        let rows = &ri[cp[j] as usize..cp[j + 1] as usize];
        if rows
            .iter()
            .any(|r| *r >= a.rows || (upper && *r > j as u64))
            || rows.windows(2).any(|r| r[0] >= r[1])
        {
            return Err(invalid(
                "CSC rows must be sorted unique in range; P upper triangular",
            ));
        }
    }
    Ok(CscMatrix::new(
        m,
        n,
        cp.iter().map(|x| *x as usize).collect(),
        ri.iter().map(|x| *x as usize).collect(),
        scalars(&a.values)?,
    ))
}
unsafe fn cones<T: Scalar>(input: &[Cone], m: usize) -> Result<Vec<SupportedConeT<T>>> {
    use SupportedConeT::*;
    let mut out = Vec::new();
    let mut rows = 0usize;
    for c in input {
        if c.reserved != 0 {
            return Err(invalid("nonzero cone reserved field"));
        }
        let d = usize::try_from(c.dim).map_err(|_| invalid("cone dimension overflow"))?;
        let a = scalars::<T>(&c.alpha)?;
        if d == 0 {
            return Err(invalid("cone dimension must be positive"));
        }
        let (cone, size) = match c.kind {
            0 => (ZeroConeT(d), d),
            1 => (NonnegativeConeT(d), d),
            2 => (SecondOrderConeT(d), d),
            3 => (
                PSDTriangleConeT(d),
                d.checked_mul(
                    d.checked_add(1)
                        .ok_or_else(|| invalid("PSD dimension overflow"))?,
                )
                .and_then(|v| v.checked_div(2))
                .ok_or_else(|| invalid("PSD dimension overflow"))?,
            ),
            4 if d == 3 => (ExponentialConeT(), 3),
            5 if d == 3 && a.len() == 1 && a[0] > T::zero() && a[0] < T::one() => {
                (PowerConeT(a[0]), 3)
            }
            6 if !a.is_empty() && a.iter().all(|x| *x > T::zero() && *x < T::one()) => {
                let sum = a.iter().fold(T::zero(), |s, x| s + *x);
                let eps = T::epsilon() * T::from_usize(a.len() * 8).unwrap();
                if (sum - T::one()).abs() > eps {
                    return Err(invalid("generalized power alpha must sum to one"));
                }
                let len = a
                    .len()
                    .checked_add(d)
                    .ok_or_else(|| invalid("cone size overflow"))?;
                (GenPowerConeT(a, d), len)
            }
            _ => return Err(invalid("invalid cone kind/dimension/parameters")),
        };
        if c.kind != 5 && c.kind != 6 && c.alpha.count != 0 {
            return Err(invalid("unexpected cone alpha"));
        }
        rows = rows
            .checked_add(size)
            .ok_or_else(|| invalid("cone size overflow"))?;
        out.push(cone);
    }
    if rows != m {
        return Err(invalid("cone dimensions do not sum to rows"));
    }
    Ok(out)
}
fn defaults() -> Settings {
    Settings {
        abi_version: ABI_VERSION,
        struct_size: std::mem::size_of::<Settings>() as u32,
        precision_bits: 53,
        max_iter: 200,
        verbose: 0,
        preprocessing_flags: PREPROCESS_ALL,
        max_threads: 1,
        kkt_form: 0,
        time_limit: f64::INFINITY,
        tol_gap_abs: ptr::null(),
        tol_gap_rel: ptr::null(),
        tol_feas: ptr::null(),
        tol_infeas_abs: ptr::null(),
        tol_infeas_rel: ptr::null(),
    }
}
fn validate_settings_layout(s: &Settings) -> Result<()> {
    if s.abi_version != ABI_VERSION {
        return Err(invalid(format!(
            "unsupported SDPX ABI version {}; this library requires ABI 3",
            s.abi_version
        )));
    }
    if s.struct_size as usize != std::mem::size_of::<Settings>()
        || s.kkt_form > 2
        || s.verbose > 1
        || s.preprocessing_flags & !PREPROCESS_ALL != 0
        || s.time_limit.is_nan()
        || s.time_limit < 0.0
    {
        return Err(invalid("invalid settings layout or values"));
    }
    Ok(())
}
unsafe fn settings<T: Scalar>(s: &Settings) -> Result<DefaultSettings<T>> {
    validate_settings_layout(s)?;
    let mut v = DefaultSettings::<T>::default();
    v.max_iter = s.max_iter;
    v.verbose = s.verbose != 0;
    v.equilibrate_enable = s.preprocessing_flags & PREPROCESS_RUIZ != 0;
    v.time_limit = s.time_limit;
    v.max_threads = s.max_threads.max(1);
    // Reuse the core's symbolic fill/work estimate. A thread budget alone
    // does not determine whether supernodal factorization is worthwhile.
    // "auto" keeps QDLDL semantics for MpFloat while allowing the
    // structurally-detected arrow factorization when it applies.
    // SDPX_DIRECT_SOLVE pins the backend for A/B measurements.
    v.direct_solve_method = match std::env::var("SDPX_DIRECT_SOLVE") {
        Ok(m) if m == "qdldl" || m == "auto" => m,
        _ => "auto".to_string(),
    };
    v.kkt_form = match s.kkt_form {
        0 => "auto",
        1 => "augmented",
        2 => "condensed",
        _ => unreachable!("validated KKT form"),
    }
    .into();
    v.presolve_enable = s.preprocessing_flags & PREPROCESS_PRESOLVE != 0;
    v.input_sparse_dropzeros = false;
    v.chordal_decomposition_enable = s.preprocessing_flags & PREPROCESS_CHORDAL != 0;
    let tolerance = v.tol_feas;
    let parse = |p| -> Result<T> {
        let x = if p == ptr::null() {
            tolerance
        } else {
            T::decimal(string(p)?)?
        };
        if !x.is_finite() || x <= T::zero() {
            Err(invalid("tolerance must be finite positive"))
        } else {
            Ok(x)
        }
    };
    v.tol_gap_abs = parse(s.tol_gap_abs)?;
    v.tol_gap_rel = parse(s.tol_gap_rel)?;
    v.tol_feas = parse(s.tol_feas)?;
    v.tol_infeas_abs = parse(s.tol_infeas_abs)?;
    v.tol_infeas_rel = parse(s.tol_infeas_rel)?;
    // Keep the core's independent kappa/tau default; it is not tol_feas.
    // The reduced ("almost solved") tolerances keep the core defaults: Float64
    // gets upstream Clarabel's 5e-5/1e-4/5e-12 values and MPFR gets the
    // precision-scaled ones. Overwriting them with the strict tolerances made
    // `AlmostSolved` unreachable, so a stalled iterate that already met the
    // strict *external* gate was reported as numerical_failure/stalled.
    v.validate().map_err(|e| invalid(e.to_string()))?;
    Ok(v)
}
struct Typed<T: Scalar> {
    solver: DefaultSolver<T>,
    bits: u32,
    solved: bool,
}
impl<T: Scalar> Typed<T> {
    unsafe fn prepare(
        p: &Csc,
        q: &Scalars,
        a: &Csc,
        b: &Scalars,
        c: &[Cone],
        s: &Settings,
        blocks: Option<&[SampledBlock]>,
    ) -> Result<Self> {
        if p.rows != p.cols || p.cols != a.cols || q.count != a.cols || b.count != a.rows {
            return Err(invalid("inconsistent problem dimensions"));
        }
        let p = matrix(p, true)?;
        let a = matrix(a, false)?;
        let q = scalars(q)?;
        let b = scalars(b)?;
        let cones = cones(c, a.m)?;
        let settings = settings(s)?;
        let solver = match blocks {
            Some(blocks) => DefaultSolver::new_sampled(
                &p,
                &q,
                &a,
                &b,
                &cones,
                sampled_blocks(blocks)?,
                settings,
            ),
            None => DefaultSolver::new(&p, &q, &a, &b, &cones, settings),
        }
        .map_err(|e| invalid(e.to_string()))?;
        Ok(Self {
            solver,
            bits: s.precision_bits,
            solved: false,
        })
    }
    fn solve(&mut self) -> Result<()> {
        self.solved = false;
        // Reset per-solve phase aggregation so the receipt covers this solve.
        sdpx_solver::receipt::drain();
        self.solver.solve();
        self.solved = true;
        self.emit_receipt();
        Ok(())
    }
    /// JSON execution receipt (plan PR-01), written when `SDPX_RECEIPT` names
    /// a path. Records identities, counts and aggregated phase timings; never
    /// fails the solve on I/O errors.
    fn emit_receipt(&self) {
        let Some(path) = std::env::var_os("SDPX_RECEIPT") else {
            return;
        };
        let phases = sdpx_solver::receipt::drain();
        let mut phase_map = serde_json::Map::new();
        for (name, s) in &phases {
            phase_map.insert(
                (*name).to_string(),
                serde_json::json!({
                    "count": s.count,
                    "total_s": s.total.as_secs_f64(),
                    "median_s": s.median().as_secs_f64(),
                    "max_s": s.max.as_secs_f64(),
                }),
            );
        }
        let ctr = self.solver.kktsystem.counters();
        let env = |k: &str| match std::env::var(k) {
            Ok(v) => serde_json::json!(v),
            Err(_) => serde_json::Value::Null,
        };
        let v = serde_json::json!({
            "schema_version": 1,
            "crate_version": env!("CARGO_PKG_VERSION"),
            "git_hash": option_env!("SDPX_GIT_HASH"),
            "precision_bits": self.bits,
            "backend": self.solver.info.linsolver.name,
            "threads": {
                "backend": self.solver.info.linsolver.threads,
                "cones": self.solver.cones.cone_threads(),
            },
            "dimensions": {"n": self.solver.solution.x.len(), "m": self.solver.solution.z.len()},
            "iterations": self.solver.solution.iterations,
            "status": format!("{:?}", self.solver.solution.status),
            "solve_time_s": self.solver.solution.solve_time,
            "counters": {
                "factorizations": ctr.factorizations,
                "rhs_applied": ctr.rhs_applied,
                "batches": ctr.batches,
            },
            "phases": phase_map,
            "memory": {"peak_rss_bytes": peak_rss_bytes()},
            "env": {
                "SDPX_SVD_ROT": env("SDPX_SVD_ROT"),
                "SDPX_DIRECT_SOLVE": env("SDPX_DIRECT_SOLVE"),
                "SDPX_RNS_OPS": env("SDPX_RNS_OPS"),
                "SDPX_INPUT_ID": env("SDPX_INPUT_ID"),
            },
        });
        if let Err(e) = std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap()) {
            eprintln!("SDPX_RECEIPT write failed for {path:?}: {e}");
        }
    }
    unsafe fn update(&mut self, q: &Scalars, b: &Scalars) -> Result<()> {
        if q.count != self.solver.solution.x.len() as u64
            || b.count != self.solver.solution.z.len() as u64
        {
            return Err(invalid("update dimensions differ from prepared problem"));
        }
        let q: Vec<T> = scalars(q)?;
        let b: Vec<T> = scalars(b)?;
        self.solved = false;
        self.solver
            .update_q(&q)
            .map_err(|e| invalid(e.to_string()))?;
        self.solver
            .update_b(&b)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(())
    }
    fn info(&self) -> Info {
        let s = &self.solver.solution;
        let i = &self.solver.info;
        Info {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<Info>() as u32,
            status: if self.solved { s.status as u32 } else { 0 },
            iterations: if self.solved { s.iterations } else { 0 },
            working_bits: self.bits,
            backend_threads: i.linsolver.threads as u32,
            cone_threads: self.solver.cones.cone_threads() as u32,
            kkt_form: if i.linsolver.name.starts_with("condensed_") {
                2
            } else {
                1
            },
            n: s.x.len() as u64,
            m: s.z.len() as u64,
            solve_time: if self.solved { s.solve_time } else { 0.0 },
            objective: s.obj_val.to_f64().unwrap_or(f64::NAN),
            dual_objective: s.obj_val_dual.to_f64().unwrap_or(f64::NAN),
            primal_residual: s.r_prim.to_f64().unwrap_or(f64::NAN),
            dual_residual: s.r_dual.to_f64().unwrap_or(f64::NAN),
            gap_abs: i.gap_abs.to_f64().unwrap_or(f64::NAN),
            gap_rel: i.gap_rel.to_f64().unwrap_or(f64::NAN),
        }
    }
    fn result(&self) -> Result<Vec<T>> {
        if !self.solved {
            return Err((7, "solve required before reading result".into()));
        }
        let s = &self.solver.solution;
        let mut r = Vec::with_capacity(s.x.len() + s.z.len() + s.s.len() + 6);
        r.extend_from_slice(&s.x);
        r.extend_from_slice(&s.z);
        r.extend_from_slice(&s.s);
        r.extend_from_slice(&[
            s.obj_val,
            s.obj_val_dual,
            s.r_prim,
            s.r_dual,
            self.solver.info.gap_abs,
            self.solver.info.gap_rel,
        ]);
        Ok(r)
    }
    fn f64_result(&self) -> Result<Vec<f64>> {
        Ok(self
            .result()?
            .iter()
            .map(|x| x.to_f64().unwrap_or(f64::NAN))
            .collect())
    }
    fn decimal_result(&self) -> Result<Vec<u8>> {
        let digits = (self.bits as usize * 30103 / 100000) + 4;
        let mut bytes = Vec::new();
        for x in self.result()? {
            bytes.extend_from_slice(format!("{x:.digits$e}").as_bytes());
            bytes.push(0)
        }
        Ok(bytes)
    }
}
enum Engine {
    F64(Typed<f64>),
    B128(Typed<MpFloat<2>>),
    B256(Typed<MpFloat<4>>),
    B512(Typed<MpFloat<8>>),
    B768(Typed<MpFloat<12>>),
    B1024(Typed<MpFloat<16>>),
    B2048(Typed<MpFloat<32>>),
}
macro_rules! dispatch {
    ($engine:expr,$s:ident,$body:expr) => {
        match $engine {
            Engine::F64($s) => $body,
            Engine::B128($s) => $body,
            Engine::B256($s) => $body,
            Engine::B512($s) => $body,
            Engine::B768($s) => $body,
            Engine::B1024($s) => $body,
            Engine::B2048($s) => $body,
        }
    };
}
struct State {
    engine: Engine,
    poisoned: bool,
}
#[repr(C)]
pub struct Handle {
    state: Mutex<State>,
}
unsafe fn operate(h: *mut Handle, f: impl FnOnce(&mut Engine) -> Result<()>) -> Result<()> {
    let h = reference(h)?;
    let mut state = match h.state.try_lock() {
        Ok(s) => s,
        Err(TryLockError::WouldBlock) => return Err((3, "handle is busy".into())),
        Err(TryLockError::Poisoned(_)) => return Err((4, "handle is poisoned".into())),
    };
    if state.poisoned {
        return Err((4, "handle is poisoned".into()));
    }
    match catch_unwind(AssertUnwindSafe(|| f(&mut state.engine))) {
        Ok(r) => r,
        Err(_) => {
            state.poisoned = true;
            Err((5, "solver panic; handle poisoned, destroy required".into()))
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn sdpx_default_settings(out: *mut Settings) -> i32 {
    boundary(|| {
        pointer(out)?;
        ptr::write(out, defaults());
        Ok(())
    })
}
#[no_mangle]
pub unsafe extern "C" fn sdpx_prepare(
    p: *const Csc,
    q: *const Scalars,
    a: *const Csc,
    b: *const Scalars,
    c: *const Cone,
    nc: u64,
    s: *const Settings,
    out: *mut *mut Handle,
) -> i32 {
    prepare_entry(p, q, a, b, c, nc, s, None, out)
}
#[no_mangle]
pub unsafe extern "C" fn sdpx_prepare_sampled(
    p: *const Csc,
    q: *const Scalars,
    a: *const Csc,
    b: *const Scalars,
    c: *const Cone,
    nc: u64,
    s: *const Settings,
    blocks: *const SampledBlock,
    block_count: u64,
    out: *mut *mut Handle,
) -> i32 {
    prepare_entry(p, q, a, b, c, nc, s, Some((blocks, block_count)), out)
}
unsafe fn prepare_entry(
    p: *const Csc,
    q: *const Scalars,
    a: *const Csc,
    b: *const Scalars,
    c: *const Cone,
    nc: u64,
    s: *const Settings,
    blocks: Option<(*const SampledBlock, u64)>,
    out: *mut *mut Handle,
) -> i32 {
    boundary(|| {
        pointer(out)?;
        ptr::write(out, ptr::null_mut());
        let (p, q, a, b, c, s) = (
            reference(p)?,
            reference(q)?,
            reference(a)?,
            reference(b)?,
            array(c, nc)?,
            reference(s)?,
        );
        validate_settings_layout(s)?;
        let blocks = blocks.map(|(p, n)| array(p, n)).transpose()?;
        let engine = match s.precision_bits {
            53 => Engine::F64(Typed::prepare(p, q, a, b, c, s, blocks)?),
            128 => Engine::B128(Typed::prepare(p, q, a, b, c, s, blocks)?),
            256 => Engine::B256(Typed::prepare(p, q, a, b, c, s, blocks)?),
            512 => Engine::B512(Typed::prepare(p, q, a, b, c, s, blocks)?),
            768 => Engine::B768(Typed::prepare(p, q, a, b, c, s, blocks)?),
            1024 => Engine::B1024(Typed::prepare(p, q, a, b, c, s, blocks)?),
            2048 => Engine::B2048(Typed::prepare(p, q, a, b, c, s, blocks)?),
            _ => {
                return Err((
                    2,
                    "unsupported precision; use 53,128,256,512,768,1024,2048".into(),
                ))
            }
        };
        ptr::write(
            out,
            Box::into_raw(Box::new(Handle {
                state: Mutex::new(State {
                    engine,
                    poisoned: false,
                }),
            })),
        );
        Ok(())
    })
}
#[no_mangle]
pub unsafe extern "C" fn sdpx_solve(h: *mut Handle) -> i32 {
    boundary(|| operate(h, |e| dispatch!(e, s, s.solve())))
}
#[no_mangle]
pub unsafe extern "C" fn sdpx_update(h: *mut Handle, q: *const Scalars, b: *const Scalars) -> i32 {
    boundary(|| {
        let (q, b) = (reference(q)?, reference(b)?);
        operate(h, |e| dispatch!(e, s, s.update(q, b)))
    })
}
#[no_mangle]
pub unsafe extern "C" fn sdpx_get_info(h: *mut Handle, out: *mut Info) -> i32 {
    boundary(|| {
        pointer(out)?;
        operate(h, |e| {
            ptr::write(out, dispatch!(e, s, s.info()));
            Ok(())
        })
    })
}
#[no_mangle]
pub unsafe extern "C" fn sdpx_get_solver_name(
    h: *mut Handle,
    out: *mut c_char,
    capacity: u64,
    required: *mut u64,
) -> i32 {
    boundary(|| {
        operate(h, |e| {
            let mut name = dispatch!(e, s, s.solver.info.linsolver.name.as_bytes().to_vec());
            name.push(0);
            output(&name, out.cast(), capacity, required)
        })
    })
}
unsafe fn output<T: Copy>(
    values: &[T],
    out: *mut T,
    capacity: u64,
    required: *mut u64,
) -> Result<()> {
    pointer(required)?;
    ptr::write(required, values.len() as u64);
    if out.is_null() && capacity == 0 {
        return Ok(());
    }
    if capacity < values.len() as u64 {
        return Err((6, "insufficient output capacity".into()));
    }
    if !values.is_empty() {
        pointer(out)?;
        ptr::copy_nonoverlapping(values.as_ptr(), out, values.len());
    }
    Ok(())
}
#[no_mangle]
pub unsafe extern "C" fn sdpx_result_f64(
    h: *mut Handle,
    out: *mut f64,
    capacity: u64,
    required: *mut u64,
) -> i32 {
    boundary(|| {
        operate(h, |e| {
            output(&dispatch!(e, s, s.f64_result())?, out, capacity, required)
        })
    })
}
#[no_mangle]
pub unsafe extern "C" fn sdpx_result_decimal(
    h: *mut Handle,
    out: *mut c_char,
    capacity: u64,
    required: *mut u64,
) -> i32 {
    boundary(|| {
        operate(h, |e| {
            output(
                &dispatch!(e, s, s.decimal_result())?,
                out.cast(),
                capacity,
                required,
            )
        })
    })
}
#[no_mangle]
pub unsafe extern "C" fn sdpx_last_error(out: *mut c_char, capacity: u64) -> u64 {
    catch_unwind(AssertUnwindSafe(|| {
        ERROR.with(|e| {
            let e = e.borrow();
            let n = e.len() + 1;
            if !out.is_null() && capacity > 0 {
                let k = e.len().min((capacity - 1).min(isize::MAX as u64) as usize);
                ptr::copy_nonoverlapping(e.as_ptr(), out.cast(), k);
                *out.add(k) = 0;
            }
            n as u64
        })
    }))
    .unwrap_or(0)
}
#[no_mangle]
pub unsafe extern "C" fn sdpx_destroy(h: *mut Handle) -> i32 {
    boundary(|| {
        if h.is_null() {
            return Ok(());
        }
        {
            let h = reference(h)?;
            match h.state.try_lock() {
                Ok(_) => (),
                Err(TryLockError::WouldBlock) => return Err((3, "handle is busy".into())),
                Err(TryLockError::Poisoned(_)) => (),
            }
        }
        drop(Box::from_raw(h));
        Ok(())
    })
}


#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
