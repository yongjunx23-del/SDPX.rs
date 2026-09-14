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
    v.direct_solve_method = if s.precision_bits == 53 {
        "auto"
    } else {
        "qdldl"
    }
    .into();
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
    v.tol_ktratio = tolerance;
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
        self.solver.solve();
        self.solved = true;
        Ok(())
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
mod tests {
    use super::*;
    fn arr(v: &[f64]) -> Scalars {
        Scalars {
            kind: 0,
            reserved: 0,
            count: v.len() as u64,
            f64: v.as_ptr(),
            decimal: ptr::null(),
        }
    }
    fn lp() -> *mut Handle {
        lp_settings(defaults())
    }
    fn lp_settings(settings: Settings) -> *mut Handle {
        unsafe {
            let p = Csc {
                rows: 1,
                cols: 1,
                nnz: 0,
                colptr: [0, 0].as_ptr(),
                rowval: ptr::null(),
                values: arr(&[]),
            };
            let a = Csc {
                rows: 1,
                cols: 1,
                nnz: 1,
                colptr: [0, 1].as_ptr(),
                rowval: [0].as_ptr(),
                values: arr(&[-1.0]),
            };
            let c = Cone {
                kind: 1,
                reserved: 0,
                dim: 1,
                alpha: arr(&[]),
            };
            let mut h = ptr::null_mut();
            assert_eq!(
                sdpx_prepare(
                    &p,
                    &arr(&[1.0]),
                    &a,
                    &arr(&[-1.0]),
                    &c,
                    1,
                    &settings,
                    &mut h
                ),
                0
            );
            h
        }
    }
    #[test]
    fn sampled_descriptor_validation() {
        unsafe {
            assert_eq!(std::mem::size_of::<SampledBlock>(), 104);
            let mut block = SampledBlock {
                row_start: 0,
                column_start: 0,
                dim: 1,
                basis_rows: 1,
                basis_cols: 1,
                basis: arr(&[1.0]),
                weights: arr(&[-1.0]),
            };
            let parsed = sampled_blocks::<f64>(std::slice::from_ref(&block)).unwrap();
            assert_eq!(parsed[0].basis, vec![1.0]);
            assert_eq!(parsed[0].weights, vec![-1.0]);
            block.weights.count = 2;
            assert!(sampled_blocks::<f64>(std::slice::from_ref(&block)).is_err());
            block.weights.count = 1;
            block.dim = u64::MAX;
            assert!(sampled_blocks::<f64>(std::slice::from_ref(&block)).is_err());
            block.dim = 1;
            block.basis_cols = 0;
            block.basis = arr(&[]);
            block.weights = arr(&[]);
            let empty = sampled_blocks::<f64>(std::slice::from_ref(&block)).unwrap();
            assert!(empty[0].basis.is_empty());
            assert!(empty[0].weights.is_empty());
            let mut out = 1usize as *mut Handle;
            assert_eq!(
                sdpx_prepare_sampled(
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    1,
                    &mut out
                ),
                1
            );
            assert!(out.is_null());
        }
    }
    #[test]
    fn sampled_public_abi_scalar_psd() {
        unsafe {
            for bits in [53, 512] {
                let mut settings = defaults();
                settings.precision_bits = bits;
                let p = Csc {
                    rows: 1,
                    cols: 1,
                    nnz: 0,
                    colptr: [0, 0].as_ptr(),
                    rowval: ptr::null(),
                    values: arr(&[]),
                };
                let a = Csc {
                    rows: 1,
                    cols: 1,
                    nnz: 0,
                    colptr: [0, 0].as_ptr(),
                    rowval: ptr::null(),
                    values: arr(&[]),
                };
                let cone = Cone {
                    kind: 3,
                    reserved: 0,
                    dim: 1,
                    alpha: arr(&[]),
                };
                let block = SampledBlock {
                    row_start: 0,
                    column_start: 0,
                    dim: 1,
                    basis_rows: 1,
                    basis_cols: 1,
                    basis: arr(&[1.0]),
                    weights: arr(&[-1.0]),
                };
                let mut h = ptr::null_mut();
                assert_eq!(
                    sdpx_prepare_sampled(
                        &p,
                        &arr(&[1.0]),
                        &a,
                        &arr(&[-1.0]),
                        &cone,
                        1,
                        &settings,
                        &block,
                        1,
                        &mut h
                    ),
                    0
                );
                assert!(!h.is_null());
                assert_eq!(sdpx_solve(h), 0);
                let mut info = std::mem::MaybeUninit::<Info>::uninit();
                assert_eq!(sdpx_get_info(h, info.as_mut_ptr()), 0);
                let info = info.assume_init();
                assert_eq!(info.status, 1);
                assert_eq!(info.working_bits, bits);
                // High precision is checked without f64 conversion in Julia tests.
                assert!((info.objective - 1.0).abs() <= 2e-8);
                assert_eq!(sdpx_destroy(h), 0);
            }
        }
    }
    #[test]
    fn null_and_settings_errors() {
        unsafe {
            assert_eq!(sdpx_solve(ptr::null_mut()), 1);
            assert_eq!(sdpx_default_settings(ptr::null_mut()), 1);
            assert_eq!(sdpx_destroy(ptr::null_mut()), 0);
            let mut s = std::mem::MaybeUninit::<Settings>::uninit();
            assert_eq!(sdpx_default_settings(s.as_mut_ptr()), 0);
            let s = s.assume_init();
            assert_eq!(s.abi_version, 3);
            assert_eq!(s.preprocessing_flags, 7);
            assert_eq!(s.kkt_form, 0);
            assert_eq!(std::mem::size_of::<Settings>(), 80);
            assert_eq!(std::mem::size_of::<Info>(), 104);
        }
    }
    #[test]
    fn abi_version_and_kkt_form_validation() {
        unsafe {
            let mut s = defaults();
            for version in [0, 1, 2, 4] {
                s.abi_version = version;
                let error = settings::<f64>(&s).err().unwrap();
                assert_eq!(error.0, 1);
                assert!(error.1.contains("requires ABI 3"));
            }
            s.abi_version = ABI_VERSION;
            s.kkt_form = 3;
            assert!(settings::<f64>(&s).is_err());
            for (code, name) in [(0, "auto"), (1, "augmented"), (2, "condensed")] {
                s.kkt_form = code;
                assert_eq!(settings::<f64>(&s).unwrap().kkt_form, name);
            }
            s.max_threads = 4;
            let f64_settings = settings::<f64>(&s).unwrap();
            assert_eq!(f64_settings.max_threads, 4);
            assert_eq!(f64_settings.direct_solve_method, "auto");
            s.precision_bits = 128;
            let hp_settings = settings::<MpFloat<2>>(&s).unwrap();
            assert_eq!(hp_settings.max_threads, 4);
            assert_eq!(hp_settings.direct_solve_method, "qdldl");
        }
    }
    fn preprocessing_mapping<T: Scalar>(bits: u32) {
        let core = DefaultSettings::<T>::default();
        assert!(core.equilibrate_enable);
        assert!(core.presolve_enable);
        assert!(core.chordal_decomposition_enable);
        unsafe {
            for flags in 0..=PREPROCESS_ALL {
                let mut s = defaults();
                s.precision_bits = bits;
                s.preprocessing_flags = flags;
                let mapped = settings::<T>(&s).unwrap();
                assert_eq!(mapped.equilibrate_enable, flags & PREPROCESS_RUIZ != 0);
                assert_eq!(mapped.presolve_enable, flags & PREPROCESS_PRESOLVE != 0);
                assert_eq!(
                    mapped.chordal_decomposition_enable,
                    flags & PREPROCESS_CHORDAL != 0
                );
                assert!(!mapped.input_sparse_dropzeros);
                assert_eq!(mapped.tol_feas, core.tol_feas);
                assert_eq!(
                    mapped.static_regularization_constant,
                    core.static_regularization_constant
                );
                assert_eq!(mapped.max_threads, 1);
            }
        }
    }

    #[test]
    fn preprocessing_flags_map_all_modes() {
        preprocessing_mapping::<f64>(53);
        preprocessing_mapping::<MpFloat<2>>(128);
        preprocessing_mapping::<MpFloat<4>>(256);
        preprocessing_mapping::<MpFloat<8>>(512);
        preprocessing_mapping::<MpFloat<12>>(768);
        preprocessing_mapping::<MpFloat<16>>(1024);
        preprocessing_mapping::<MpFloat<32>>(2048);
    }

    #[test]
    fn preprocessing_unknown_bits_are_rejected() {
        unsafe {
            for flags in [8, PREPROCESS_ALL | 8, 1 << 31, u32::MAX] {
                let mut s = defaults();
                s.preprocessing_flags = flags;
                assert_eq!(settings::<f64>(&s).err().unwrap().0, 1);
                s.precision_bits = 512;
                assert_eq!(settings::<MpFloat<8>>(&s).err().unwrap().0, 1);
            }
        }
    }

    #[test]
    fn prepared_solver_receives_preprocessing_flags() {
        unsafe {
            for bits in [53, 512] {
                for flags in 0..=PREPROCESS_ALL {
                    let mut s = defaults();
                    s.precision_bits = bits;
                    s.preprocessing_flags = flags;
                    let h = lp_settings(s);
                    operate(h, |engine| {
                        dispatch!(engine, typed, {
                            let actual = typed.solver.settings();
                            assert_eq!(actual.equilibrate_enable, flags & PREPROCESS_RUIZ != 0);
                            assert_eq!(actual.presolve_enable, flags & PREPROCESS_PRESOLVE != 0);
                            assert_eq!(
                                actual.chordal_decomposition_enable,
                                flags & PREPROCESS_CHORDAL != 0
                            );
                            assert!(!actual.input_sparse_dropzeros);
                            Ok(())
                        })
                    })
                    .unwrap();
                    assert_eq!(sdpx_destroy(h), 0);
                }
            }
        }
    }

    #[test]
    fn solver_name_query_and_actual_info() {
        unsafe {
            let h = lp();
            let mut n = 0;
            assert_eq!(sdpx_get_solver_name(h, ptr::null_mut(), 0, &mut n), 0);
            assert_eq!(n, 6); // qdldl plus NUL
            let mut short = [0x55_u8; 2];
            assert_eq!(
                sdpx_get_solver_name(h, short.as_mut_ptr().cast(), 2, &mut n),
                6
            );
            assert_eq!(short, [0x55; 2]);
            let mut name = vec![0_u8; n as usize];
            assert_eq!(
                sdpx_get_solver_name(h, name.as_mut_ptr().cast(), n, &mut n),
                0
            );
            assert_eq!(name, b"qdldl\0");
            assert_eq!(
                sdpx_get_solver_name(h, ptr::null_mut(), 0, ptr::null_mut()),
                1
            );
            {
                let _guard = (*h).state.lock().unwrap();
                assert_eq!(sdpx_get_solver_name(h, ptr::null_mut(), 0, &mut n), 3);
            }
            let mut info = std::mem::MaybeUninit::<Info>::uninit();
            assert_eq!(sdpx_get_info(h, info.as_mut_ptr()), 0);
            let info = info.assume_init();
            assert_eq!(info.abi_version, 3);
            assert_eq!(info.kkt_form, 1);
            assert_eq!(info.backend_threads, 1);
            assert_eq!(info.cone_threads, 1);
            assert_eq!(sdpx_destroy(h), 0);
        }
    }
    #[test]
    fn reject_unsorted_csc() {
        unsafe {
            let c = Csc {
                rows: 2,
                cols: 1,
                nnz: 2,
                colptr: [0, 2].as_ptr(),
                rowval: [1, 0].as_ptr(),
                values: arr(&[1.0, 1.0]),
            };
            assert!(matrix::<f64>(&c, false).is_err());
        }
    }
    #[test]
    fn zero_time_limit_returns_max_time() {
        unsafe {
            let mut settings = defaults();
            settings.time_limit = 0.0;
            let h = lp_settings(settings);
            assert_eq!(sdpx_solve(h), 0);
            let mut info = std::mem::MaybeUninit::<Info>::uninit();
            assert_eq!(sdpx_get_info(h, info.as_mut_ptr()), 0);
            assert_eq!(info.assume_init().status, 8);
            assert_eq!(sdpx_destroy(h), 0);
        }
    }
    #[test]
    fn lifecycle_and_update() {
        unsafe {
            // Structural preprocessing can invalidate reusable updates. Keep
            // Ruiz enabled while explicitly disabling presolve and chordal.
            let mut settings = defaults();
            settings.preprocessing_flags = PREPROCESS_RUIZ;
            let h = lp_settings(settings);
            let mut n = 0;
            assert_eq!(sdpx_result_f64(h, ptr::null_mut(), 0, &mut n), 7);
            assert_eq!(sdpx_solve(h), 0);
            assert_eq!(sdpx_result_f64(h, ptr::null_mut(), 0, &mut n), 0);
            let mut v = vec![0.0; n as usize];
            assert_eq!(sdpx_result_f64(h, v.as_mut_ptr(), n, &mut n), 0);
            assert!((v[0] - 1.0).abs() < 1e-7);
            assert_eq!(sdpx_update(h, &arr(&[f64::NAN]), &arr(&[-2.0])), 1);
            assert_eq!(sdpx_result_f64(h, ptr::null_mut(), 0, &mut n), 0);
            assert_eq!(sdpx_update(h, &arr(&[1.0]), &arr(&[-2.0])), 0);
            assert_eq!(sdpx_result_f64(h, ptr::null_mut(), 0, &mut n), 7);
            assert_eq!(sdpx_solve(h), 0);
            assert_eq!(sdpx_result_f64(h, v.as_mut_ptr(), n, &mut n), 0);
            assert!((v[0] - 2.0).abs() < 1e-7);
            assert_eq!(sdpx_destroy(h), 0);
        }
    }
    #[test]
    fn panic_poisons_and_busy_rejects() {
        unsafe {
            let h = lp();
            {
                let _guard = (*h).state.lock().unwrap();
                assert_eq!(sdpx_solve(h), 3);
            }
            assert_eq!(boundary(|| operate(h, |_| panic!("test"))), 5);
            assert_eq!(sdpx_solve(h), 4);
            assert_eq!(sdpx_destroy(h), 0);
        }
    }
}
