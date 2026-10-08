//! Optional MPI world for cross-node block sharding.
//!
//! Loaded lazily through `dlopen` so the dependency stays optional. A process
//! with no MPI launch metadata takes the ordinary serial path; an activated
//! multi-rank launch fails closed if MPI cannot initialize or its ABI/thread
//! prerequisites are unavailable, rather than silently solving independently
//! on every rank. A single-process world does not touch MPI unless explicitly
//! requested through `SDPX_MPI`.
//!
//! The shard discipline is rank-ordered block partitioning: every block is
//! evaluated on exactly one rank with identical arithmetic, partial outputs
//! are exchanged with `MPI_Allgatherv`, and scalar reductions use `MPI_MAX`
//! (order-free), so results are bitwise identical for any rank count.

#![allow(non_snake_case)]

use sdpx_arithmetic::Scalar;
use std::ffi::{c_char, c_int, c_void, CString};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    OnceLock,
};
use std::thread::ThreadId;

type MpiComm = *mut c_void;
type MpiDatatype = *mut c_void;
type MpiOp = *mut c_void;

extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
    fn atexit(callback: extern "C" fn()) -> c_int;
}
const RTLD_NOW: c_int = 2;
// RTLD_LOCAL is 0 on Linux/macOS; the value 4 is RTLD_NOLOAD, which returns
// null for not-yet-loaded libraries without setting dlerror.
const RTLD_LOCAL: c_int = 0;
const MPI_THREAD_MULTIPLE: c_int = 3;
const MPI_SUCCESS: c_int = 0;
const MPI_INT_MAX: usize = c_int::MAX as usize;
const WIRE_META_BYTES: usize = 56;
const WIRE_META_OK: u64 = 1;
const WIRE_META_BAD: u64 = 0;

// Main-thread control decisions must agree before a branch enters another
// numerical collective. Serial calls preserve their existing behavior.
pub(crate) fn all_succeeded(value: bool) -> bool {
    World::get().map_or(value, |world| {
        let started = std::time::Instant::now();
        let agreed = world.all_true(value);
        crate::receipt::site_record(SITE_SUCCESS, started.elapsed());
        agreed
    })
}

pub(crate) fn decision_agrees(value: u32) -> bool {
    World::get().is_none_or(|world| {
        let started = std::time::Instant::now();
        let agreed = world.agree_u32(value);
        crate::receipt::site_record(SITE_DECISION, started.elapsed());
        agreed
    })
}

/// Whether any rank holds `value` (false when not running under MPI ranks).
pub(crate) fn any_true(value: bool) -> bool {
    World::get().map_or(value, |world| !world.all_true(!value))
}

/// All replicated ranks must hold the same `value`; a mismatch aborts.
pub(crate) fn assert_agree(value: u32, message: &str) {
    if let Some(world) = World::get() {
        if !world.agree_u32(value) {
            world.abort(message);
        }
    }
}

/// Maximum of `v` over all ranks (`v` itself in serial runs).
pub(crate) fn max_all_f64(v: f64) -> f64 {
    World::get().map_or(v, |world| world.allreduce_max_f64(v))
}

/// True on rank zero and in serial runs.
pub(crate) fn is_root() -> bool {
    World::get().is_none_or(|world| world.rank() == 0)
}

/// Receipt labels for the unnumbered world agreements above.
const SITE_SUCCESS: usize = 9000;
const SITE_DECISION: usize = 9001;

/// Branching on a local step decision must never change collective order.
pub(crate) fn agreed_branch(value: bool) -> bool {
    if let Some(world) = World::get() {
        if !world.agree_u32(u32::from(value)) {
            world.abort("inconsistent replicated solver step decision");
        }
    }
    value
}

/// Shard sites run collectives on dedicated communicators so concurrent
/// gathers from different solver phases never interleave on one stream.
pub(crate) const SITE_FORWARD: usize = 0;
pub(crate) const SITE_ADJOINT: usize = 1;
pub(crate) const SITE_SCALING: usize = 2;
pub(crate) const SITE_GRAM: usize = 3;
pub(crate) const SITE_RX: usize = 4;
pub(crate) const SITE_RZ: usize = 5;
pub(crate) const SITE_CONES: usize = 6;
pub(crate) const SITE_ARROW: usize = 7;
const NSITES: usize = 8;

#[derive(Clone, Copy)]
struct Fns {
    init_thread: unsafe extern "C" fn(*mut c_int, *mut *mut *mut u8, c_int, *mut c_int) -> c_int,
    initialized: unsafe extern "C" fn(*mut c_int) -> c_int,
    query_thread: unsafe extern "C" fn(*mut c_int) -> c_int,
    finalized: unsafe extern "C" fn(*mut c_int) -> c_int,
    finalize: unsafe extern "C" fn() -> c_int,
    abort: unsafe extern "C" fn(MpiComm, c_int) -> c_int,
    comm_size: unsafe extern "C" fn(MpiComm, *mut c_int) -> c_int,
    comm_rank: unsafe extern "C" fn(MpiComm, *mut c_int) -> c_int,
    comm_dup: unsafe extern "C" fn(MpiComm, *mut MpiComm) -> c_int,
    comm_free: unsafe extern "C" fn(*mut MpiComm) -> c_int,
    allgatherv: unsafe extern "C" fn(
        *const c_void,
        c_int,
        MpiDatatype,
        *mut c_void,
        *const c_int,
        *const c_int,
        MpiDatatype,
        MpiComm,
    ) -> c_int,
    gatherv: unsafe extern "C" fn(
        *const c_void,
        c_int,
        MpiDatatype,
        *mut c_void,
        *const c_int,
        *const c_int,
        MpiDatatype,
        c_int,
        MpiComm,
    ) -> c_int,
    allreduce: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        c_int,
        MpiDatatype,
        MpiOp,
        MpiComm,
    ) -> c_int,
    comm_world: MpiComm,
    byte: MpiDatatype,
    double: MpiDatatype,
    max: MpiOp,
}

// OpenMPI handles are pointers to exported objects. Do not guess integer
// constants for another MPI implementation: an incorrectly typed predefined
// handle can crash inside the first collective (as happened with MPI_MAX).
unsafe fn resolve_handle(lib: *mut c_void, variable: &str) -> Option<*mut c_void> {
    let name = CString::new(variable).unwrap();
    let symbol = unsafe { dlsym(lib, name.as_ptr()) };
    (!symbol.is_null()).then_some(symbol)
}

/// `MPI_COMM_WORLD`, `MPI_BYTE`, `MPI_DOUBLE`, `MPI_MAX` for the library's
/// handle ABI. OpenMPI exports handles as global objects. The MPICH ABI
/// (MPICH, Intel MPI, MVAPICH) uses fixed integer handles, carried here in the
/// pointer-sized slot: x86_64 and AArch64 pass `int` arguments in the low half
/// of a register, and handle outputs land in zero-initialized slots.
unsafe fn handle_abi(lib: *mut c_void) -> Option<[*mut c_void; 4]> {
    let ompi = [
        "ompi_mpi_comm_world",
        "ompi_mpi_byte",
        "ompi_mpi_double",
        "ompi_mpi_op_max",
    ]
    .map(|name| unsafe { resolve_handle(lib, name) });
    if let [Some(comm), Some(byte), Some(double), Some(max)] = ompi {
        return Some([comm, byte, double, max]);
    }
    // MPI_Get_library_version may be called before MPI_Init.
    let version = unsafe { dlsym(lib, c"MPI_Get_library_version".as_ptr()) };
    if version.is_null() {
        return None;
    }
    let version: unsafe extern "C" fn(*mut c_char, *mut c_int) -> c_int =
        unsafe { std::mem::transmute(version) };
    let mut text = vec![0 as c_char; 8192];
    let mut len = 0;
    if unsafe { version(text.as_mut_ptr(), &mut len) } != MPI_SUCCESS {
        return None;
    }
    let text = unsafe { std::ffi::CStr::from_ptr(text.as_ptr()) }.to_string_lossy();
    ["MPICH", "Intel(R) MPI", "MVAPICH"]
        .iter()
        .any(|family| text.contains(family))
        .then(|| {
            [0x4400_0000usize, 0x4c00_010d, 0x4c00_080b, 0x5800_0001].map(|h| h as *mut c_void)
        })
}

macro_rules! sym {
    ($lib:expr, $name:literal) => {{
        let p = dlsym($lib, concat!($name, "\0").as_ptr().cast());
        if p.is_null() {
            return None;
        }
        std::mem::transmute::<*mut c_void, _>(p)
    }};
}

fn load() -> Option<Fns> {
    #[cfg(target_os = "linux")]
    const NAMES: &[&str] = &["libmpi.so.40", "libmpi.so.12", "libmpi.so"];
    #[cfg(target_os = "macos")]
    const NAMES: &[&str] = &["libmpi.dylib", "libmpi.40.dylib"];
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    const NAMES: &[&str] = &[];
    let debug = std::env::var_os("SDPX_MPI_DEBUG").is_some();
    // Ranks launched via orted may not inherit the submitter's
    // LD_LIBRARY_PATH, so plain sonames can fail to resolve even when the
    // library exists. MPI_LIB (or SDPX_MPI_LIBDIR) gives an absolute dir to
    // fall back to.
    // Prefer the library family of the launcher when several are installed:
    // an OpenMPI library under Hydra (Intel MPI, MPICH) or vice versa aborts.
    let mut names = NAMES.to_vec();
    if std::env::var_os("OMPI_COMM_WORLD_SIZE").is_none() {
        names.sort_by_key(|n| !n.contains(".so.12"));
    }
    let mut candidates: Vec<CString> = names.iter().map(|n| CString::new(*n).unwrap()).collect();
    for var in ["SDPX_MPI_LIBDIR", "MPI_LIB"] {
        if let Ok(dir) = std::env::var(var) {
            // OpenMPI's installed libmpi carries a dead build-host RPATH and
            // needs libopen-pal/libopen-rte from its own dir; preload them by
            // absolute path so the later soname lookup hits loaded objects.
            for dep in ["libopen-pal.so.40", "libopen-rte.so.40"] {
                if let Ok(p) = CString::new(format!("{dir}/{dep}")) {
                    let h = unsafe { dlopen(p.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
                    if debug && h.is_null() {
                        eprintln!("mpi: dep preload {dep} failed");
                    }
                }
            }
            for n in &names {
                if let Ok(p) = CString::new(format!("{dir}/{n}")) {
                    candidates.push(p);
                }
            }
        }
    }
    for name in &candidates {
        let lib = unsafe { dlopen(name.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
        if lib.is_null() {
            if debug {
                let err = unsafe { dlerror() };
                let msg = if err.is_null() {
                    "unknown".to_string()
                } else {
                    unsafe { std::ffi::CStr::from_ptr(err) }
                        .to_string_lossy()
                        .into_owned()
                };
                eprintln!("mpi: dlopen({}) failed: {msg}", name.to_string_lossy());
            }
            continue;
        }
        unsafe {
            if dlsym(lib, c"MPI_Comm_rank".as_ptr()).is_null() {
                if debug {
                    eprintln!("mpi: {} lacks MPI_Comm_rank", name.to_string_lossy());
                }
                continue;
            }
            let Some(handles) = handle_abi(lib) else {
                if debug {
                    eprintln!("mpi: {} has an unknown handle ABI", name.to_string_lossy());
                }
                continue;
            };
            return Some(Fns {
                init_thread: sym!(lib, "MPI_Init_thread"),
                initialized: sym!(lib, "MPI_Initialized"),
                query_thread: sym!(lib, "MPI_Query_thread"),
                finalized: sym!(lib, "MPI_Finalized"),
                finalize: sym!(lib, "MPI_Finalize"),
                abort: sym!(lib, "MPI_Abort"),
                comm_size: sym!(lib, "MPI_Comm_size"),
                comm_rank: sym!(lib, "MPI_Comm_rank"),
                comm_dup: sym!(lib, "MPI_Comm_dup"),
                comm_free: sym!(lib, "MPI_Comm_free"),
                allgatherv: sym!(lib, "MPI_Allgatherv"),
                gatherv: sym!(lib, "MPI_Gatherv"),
                allreduce: sym!(lib, "MPI_Allreduce"),
                comm_world: handles[0],
                byte: handles[1],
                double: handles[2],
                max: handles[3],
            });
        }
    }
    None
}

/// The number of ranks the environment advertises, without loading MPI.
fn advertised_size() -> usize {
    for var in [
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_SIZE",
        "MV2_COMM_WORLD_SIZE",
        "SLURM_NTASKS",
        "SDPX_MPI_SIZE",
    ] {
        if let Ok(v) = std::env::var(var) {
            if let Ok(n) = v.trim().parse::<usize>() {
                if n > 0 {
                    return n;
                }
            }
        }
    }
    1
}

/// A live MPI world. The function table and communicator handle are
/// process-global and constant after initialization; collectives are only
/// invoked from the calling thread.
#[derive(Clone, Copy)]
pub(crate) struct World {
    fns: &'static Fns,
    comms: &'static [MpiComm; NSITES],
    rank: i32,
    size: i32,
    owns_init: bool,
    init_thread: Option<ThreadId>,
    /// Thread which owns the solver/control-side collective sequence.
    ///
    /// MPI is initialized with `MPI_THREAD_MULTIPLE` for compatibility with
    /// hosts that have their own worker pool, but SDPX's canonical reductions
    /// are deliberately serialized on this thread.  Keeping this identity in
    /// the world makes a misplaced worker call fail by aborting the MPI job
    /// instead of allowing ranks to enter mismatched collectives.
    collective_thread: ThreadId,
}

unsafe impl Send for World {}
unsafe impl Sync for World {}
unsafe impl Sync for Fns {}

static WORLD: OnceLock<Option<World>> = OnceLock::new();
static OWNED_SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// Control operations for a replicated MPI application using the solver.
/// Numerical operations remain in the shared solver core. Every rank must
/// enter these operations in the same order, with no outstanding solver work.
#[derive(Clone, Copy)]
pub struct MpiContext(Option<World>);

impl MpiContext {
    pub(crate) fn from_world(world: Option<World>) -> Self {
        Self(world)
    }

    pub(crate) fn world(self) -> Option<World> {
        self.0
    }

    /// Activate optional MPI on the calling thread before reading input.
    /// Without an MPI launch or explicit opt-in, this is a serial context.
    pub fn initialize() -> Self {
        Self(World::get())
    }

    /// Actual rank, or zero in a serial process.
    pub fn rank(self) -> usize {
        self.0.map_or(0, |w| w.rank())
    }

    /// Actual number of ranks, or one in a serial process.
    pub fn size(self) -> usize {
        self.0.map_or(1, |w| w.size())
    }

    /// True only when every rank completed the current application stage.
    pub fn all_succeeded(self, success: bool) -> bool {
        self.0.map_or(success, |w| w.all_true(success))
    }

    /// Compare a caller-computed SHA-256 identity across all ranks. The
    /// digest covers configuration or input, not the local filesystem path.
    pub fn agree_signature(self, signature: [u8; 32]) -> bool {
        let Some(world) = self.0 else { return true };
        let local = signature.map(f64::from);
        let ranges: Vec<_> = (0..world.size()).map(|r| (r * 32, 32)).collect();
        let mut all = vec![0.0; world.size() * 32];
        world.gather_slice(NSITES - 1, &local, &ranges, &mut all);
        all.chunks_exact(32).all(|part| part == local)
    }

    /// Abort a distributed application after a failure that prevents stage
    /// synchronization, such as a panic during a numerical collective.
    pub fn abort(self, reason: &str) -> ! {
        if let Some(world) = self.0 {
            world.abort(reason);
        }
        std::process::abort()
    }

    /// Shut down only SDPX-owned MPI, on its initialization thread, after
    /// all ranks finish their final application stage. Host-owned MPI stays live.
    pub fn finish(self) {
        finalize_owned();
    }
}

/// Finalize an MPI runtime initialized by this module, without lazily
/// activating MPI. Host-owned MPI is never finalized. Call this after all
/// solver work has joined, from the thread that initialized MPI.
pub(crate) fn finalize_owned() {
    if let Some(Some(world)) = WORLD.get() {
        world.shutdown_owned_inner("explicit shutdown");
    }
}

fn activation_requested() -> (usize, bool) {
    let advertised = advertised_size();
    (
        advertised,
        advertised > 1 || std::env::var_os("SDPX_MPI").is_some(),
    )
}

/// A multi-rank launch must never silently turn into one independent serial
/// solve per rank. If MPI cannot be activated, abort the process when the
/// launcher advertised multiple ranks; this is the only safe action before a
/// communicator exists. Explicit opt-in is a hard request and therefore also
/// fails closed if activation cannot complete.
fn activation_failed(advertised: usize, requested: bool, reason: &str) -> Option<World> {
    if requested {
        eprintln!(
            "mpi: activation failed (advertised ranks={advertised}): {reason}; only OpenMPI's exported-handle ABI is supported"
        );
    }
    // An explicit SDPX_MPI opt-in is also a hard request. Returning `None`
    // after self-initializing MPI would leave an owned runtime alive and could
    // let one rank continue independently of its peers.
    if advertised > 1 || requested {
        std::process::abort();
    }
    None
}

fn activation_failed_with_abort(
    fns: &Fns,
    comm: MpiComm,
    advertised: usize,
    requested: bool,
    reason: &str,
) -> Option<World> {
    eprintln!(
        "mpi: activation failed (advertised ranks={advertised}): {reason}; aborting MPI world"
    );
    // MPI_Abort is collective at the job/runtime level and prevents peers
    // from entering a later collective after this rank rejects activation.
    let rc = unsafe { (fns.abort)(comm, 1) };
    if rc != MPI_SUCCESS {
        eprintln!("mpi: MPI_Abort returned error code {rc}");
    }
    if requested || advertised > 1 {
        std::process::abort();
    }
    None
}

extern "C" fn finalize_owned_world() {
    let Some(Some(world)) = WORLD.get() else {
        return;
    };
    world.shutdown_owned_inner("atexit");
}

impl World {
    /// Require a canonical collective to run on the control thread.
    ///
    /// A worker that violates this invariant cannot safely return a local
    /// error: peer ranks may already be waiting in the same MPI operation.
    /// Abort therefore gives every rank the same failure mode and avoids a
    /// partial reduction or a deadlock.
    pub(crate) fn ensure_collective_thread(&self) {
        if std::thread::current().id() != self.collective_thread {
            eprintln!(
                "mpi: collective from a non-control thread:\n{}",
                std::backtrace::Backtrace::force_capture()
            );
            self.abort("MPI collective called from a non-control thread");
        }
    }

    pub(crate) fn valid_collective_site(&self, site: usize) -> bool {
        site < NSITES
    }

    /// Map an owner-local numerical stream id onto one of the fixed MPI
    /// communicators. Sequence and operation handshakes still distinguish
    /// successive calls, so callers may use a larger deterministic site space
    /// without allocating one communicator per phase.
    pub(crate) fn collective_site(&self, site: usize) -> usize {
        site % NSITES
    }

    pub(crate) fn all_true(&self, value: bool) -> bool {
        self.allreduce_max_f64(if value { 0.0 } else { 1.0 }) == 0.0
    }

    pub(crate) fn agree_u32(&self, value: u32) -> bool {
        let value = f64::from(value);
        // One reduction of (v, -v) yields both max and -min.
        let mut pair = [value, -value];
        self.allreduce_max_f64_slice(&mut pair);
        pair[0] == -pair[1]
    }

    /// The shared world, initialized once. `None` unless the environment
    /// advertises more than one rank and MPI initializes successfully.
    pub(crate) fn get() -> Option<Self> {
        *WORLD.get_or_init(Self::init)
    }

    fn init() -> Option<Self> {
        let debug = std::env::var_os("SDPX_MPI_DEBUG").is_some();
        let (advertised, requested) = activation_requested();
        if !requested {
            if debug {
                eprintln!("mpi: no advertised size");
            }
            return None;
        }
        let fns = match load() {
            Some(f) => f,
            None => {
                if debug {
                    eprintln!("mpi: dlopen/symbol resolution failed");
                }
                return activation_failed(
                    advertised,
                    requested,
                    "MPI library or required OpenMPI symbols unavailable",
                );
            }
        };
        if debug {
            eprintln!("mpi: library loaded");
        }
        let mut flag = 0;
        if unsafe { (fns.initialized)(&mut flag) } != MPI_SUCCESS {
            return activation_failed(advertised, requested, "MPI_Initialized returned an error");
        }
        let mut owns_init = false;
        if flag == 0 {
            // MPI_THREAD_MULTIPLE: gathers may be issued from rayon worker
            // threads on their dedicated per-site communicators.
            let mut provided = 0;
            let rc = unsafe {
                (fns.init_thread)(
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    MPI_THREAD_MULTIPLE,
                    &mut provided,
                )
            };
            if debug {
                eprintln!("mpi: init_thread rc={rc} provided={provided}");
            }
            if rc != MPI_SUCCESS {
                return activation_failed(advertised, requested, "MPI_Init_thread failed");
            }
            owns_init = true;
        }
        let mut finalized = 0;
        if unsafe { (fns.finalized)(&mut finalized) } != MPI_SUCCESS {
            return activation_failed(advertised, requested, "MPI_Finalized returned an error");
        }
        if finalized != 0 {
            return activation_failed(advertised, requested, "MPI is already finalized");
        }
        let mut provided = 0;
        if unsafe { (fns.query_thread)(&mut provided) } != MPI_SUCCESS {
            return activation_failed(advertised, requested, "MPI_Query_thread returned an error");
        }
        if provided < MPI_THREAD_MULTIPLE {
            return activation_failed(
                advertised,
                requested,
                "MPI thread level is below MPI_THREAD_MULTIPLE",
            );
        }
        let (mut size, mut rank) = (0, 0);
        unsafe {
            if (fns.comm_size)(fns.comm_world, &mut size) != MPI_SUCCESS
                || (fns.comm_rank)(fns.comm_world, &mut rank) != MPI_SUCCESS
            {
                return activation_failed_with_abort(
                    &fns,
                    fns.comm_world,
                    advertised,
                    requested,
                    "MPI_Comm_size/ MPI_Comm_rank returned an error",
                );
            }
        }
        if debug {
            eprintln!("mpi: size={size} rank={rank}");
        }
        if size < 1 || rank < 0 || rank >= size {
            return activation_failed_with_abort(
                &fns,
                fns.comm_world,
                advertised,
                requested,
                "MPI returned an invalid rank/size",
            );
        }
        if size == 1 {
            if owns_init {
                let rc = unsafe { (fns.finalize)() };
                if rc != MPI_SUCCESS {
                    eprintln!("mpi: MPI_Finalize failed for single-rank owned world (rc={rc})");
                    std::process::abort();
                }
            }
            if advertised > 1 {
                return activation_failed(
                    advertised,
                    requested,
                    "launcher advertised multiple ranks but MPI_COMM_WORLD has size one",
                );
            }
            return None;
        }
        let comms: &'static mut [MpiComm; NSITES] =
            Box::leak(Box::new([std::ptr::null_mut(); NSITES]));
        for c in comms.iter_mut() {
            if unsafe { (fns.comm_dup)(fns.comm_world, c) } != MPI_SUCCESS {
                return activation_failed_with_abort(
                    &fns,
                    fns.comm_world,
                    advertised,
                    requested,
                    "MPI_Comm_dup returned an error",
                );
            }
        }
        let world = Self {
            fns: Box::leak(Box::new(fns)),
            comms,
            rank,
            size,
            owns_init,
            init_thread: owns_init.then(|| std::thread::current().id()),
            collective_thread: std::thread::current().id(),
        };
        if owns_init {
            // Register only for MPI initialized by SDPX. A host that owns
            // MPI remains responsible for finalization.
            let rc = unsafe { atexit(finalize_owned_world) };
            if rc != MPI_SUCCESS {
                let abort_rc = unsafe { (world.fns.abort)(world.fns.comm_world, 1) };
                if abort_rc != MPI_SUCCESS {
                    eprintln!("mpi: MPI_Abort returned error code {abort_rc}");
                }
                std::process::abort();
            }
        }
        Some(world)
    }

    pub(crate) fn size(&self) -> usize {
        self.size as usize
    }

    pub(crate) fn rank(&self) -> usize {
        self.rank as usize
    }

    fn shutdown_owned_inner(&self, source: &str) {
        if !self.owns_init {
            return;
        }
        if OWNED_SHUTDOWN.load(Ordering::Acquire) {
            return;
        }
        let mut finalized = 0;
        let finalized_rc = unsafe { (self.fns.finalized)(&mut finalized) };
        if finalized_rc != MPI_SUCCESS {
            eprintln!(
                "mpi: MPI_Finalized failed during owned shutdown ({source}, rc={finalized_rc})"
            );
            std::process::abort();
        }
        if finalized != 0 {
            OWNED_SHUTDOWN.store(true, Ordering::Release);
            return;
        }
        if self.init_thread != Some(std::thread::current().id()) {
            self.abort(
                "call sdpx_solver::mpi_finalize() on the MPI initialization thread before it exits",
            );
        }
        // The duplicated communicators are owned by this module. Freeing them
        // is collective on each communicator; every rank must call this hook
        // before finalizing its process.
        for &comm in self.comms.iter() {
            let mut owned = comm;
            let rc = unsafe { (self.fns.comm_free)(&mut owned) };
            if rc != MPI_SUCCESS {
                self.abort("MPI_Comm_free failed during owned shutdown");
            }
        }
        let rc = unsafe { (self.fns.finalize)() };
        if rc != MPI_SUCCESS {
            eprintln!("mpi: MPI_Finalize failed during owned shutdown ({source}, rc={rc})");
            std::process::abort();
        }
        OWNED_SHUTDOWN.store(true, Ordering::Release);
    }

    /// The `[begin, end)` range of `count` items owned by this rank.
    pub(crate) fn abort(&self, reason: &str) -> ! {
        eprintln!(
            "mpi: rank {}/{} fatal collective error: {reason}",
            self.rank, self.size
        );
        // A finalized MPI runtime must not receive MPI_Abort. Recheck here in
        // addition to `ensure_live` so a host finalizing concurrently cannot
        // turn an ordinary validation failure into an illegal MPI call.
        let mut finalized = 0;
        let finalized_rc = unsafe { (self.fns.finalized)(&mut finalized) };
        if finalized_rc != MPI_SUCCESS || finalized == 0 {
            if finalized_rc != MPI_SUCCESS {
                eprintln!(
                    "mpi: cannot query MPI_Finalized while aborting (rc={finalized_rc}); attempting MPI_Abort"
                );
            }
            let rc = unsafe { (self.fns.abort)(self.fns.comm_world, 1) };
            if rc != MPI_SUCCESS {
                eprintln!("mpi: MPI_Abort returned error code {rc}");
            }
        } else {
            eprintln!("mpi: MPI_Finalized is already set; skipping MPI_Abort");
        }
        // MPI_Abort is required to terminate the whole job. If an MPI
        // implementation returns from it, do not let this rank continue into
        // a mismatched collective.
        std::process::abort();
    }

    fn ensure_live(&self) {
        let mut finalized = 0;
        let rc = unsafe { (self.fns.finalized)(&mut finalized) };
        if rc != MPI_SUCCESS {
            self.abort("MPI_Finalized reports an inactive world");
        }
        if finalized != 0 {
            eprintln!(
                "mpi: rank {}/{} attempted a collective after MPI_Finalize",
                self.rank, self.size
            );
            std::process::abort();
        }
    }

    fn metadata_u64(bytes: &[u8], offset: usize) -> u64 {
        let mut value = [0u8; 8];
        value.copy_from_slice(&bytes[offset..offset + 8]);
        u64::from_le_bytes(value)
    }

    fn put_metadata_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    /// Check receive intervals before constructing MPI byte displacements.
    /// Empty ranges may repeat an offset, but two non-empty intervals must be
    /// disjoint: `MPI_Allgatherv` writes each received segment directly into
    /// `out`, so overlapping destinations would make the result undefined.
    ///
    /// The temporary interval list uses fallible reservation. A malformed or
    /// adversarially large range list therefore reports failure to the
    /// metadata handshake instead of panicking before peers can participate.
    fn ranges_nonoverlapping(ranges: &[(usize, usize)]) -> bool {
        let mut intervals = Vec::new();
        if intervals.try_reserve_exact(ranges.len()).is_err() {
            return false;
        }
        for &(offset, length) in ranges {
            if length == 0 {
                continue;
            }
            let Some(end) = offset.checked_add(length) else {
                return false;
            };
            intervals.push((offset, end));
        }
        intervals.sort_unstable_by_key(|&(offset, _)| offset);
        intervals
            .windows(2)
            .all(|window| window[0].1 <= window[1].0)
    }

    fn try_zeroed(length: usize) -> (Vec<u8>, bool) {
        let mut bytes = Vec::new();
        if bytes.try_reserve_exact(length).is_err() {
            return (bytes, false);
        }
        // The exact reservation above makes this resize infallible without an
        // additional allocation; the initialized bytes are required because
        // MPI writes only the received ranges and leaves gaps untouched.
        bytes.resize(length, 0);
        (bytes, true)
    }

    fn layout_hash(ranges: &[(usize, usize)], out_len: usize) -> Option<u64> {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let feed = |hash: &mut u64, value: u64| {
            *hash ^= value;
            *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        };
        feed(&mut hash, u64::try_from(out_len).ok()?);
        feed(&mut hash, u64::try_from(ranges.len()).ok()?);
        for &(offset, length) in ranges {
            feed(&mut hash, u64::try_from(offset).ok()?);
            feed(&mut hash, u64::try_from(length).ok()?);
        }
        Some(hash)
    }

    /// Exchange a fixed metadata record before sending payload bytes. A rank
    /// with an unsupported wire type, a malformed local segment, or a
    /// different layout still enters this handshake; all ranks then abort
    /// together instead of one rank returning before `MPI_Allgatherv`.
    fn exchange_wire_metadata(&self, site: usize, metadata: &[u8; WIRE_META_BYTES]) {
        let size = self.size as usize;
        let total = size
            .checked_mul(WIRE_META_BYTES)
            .filter(|&n| n <= MPI_INT_MAX)
            .unwrap_or_else(|| self.abort("wire metadata exceeds MPI int count"));
        let counts = vec![WIRE_META_BYTES as c_int; size];
        let displs: Vec<c_int> = (0..size)
            .map(|rank| (rank * WIRE_META_BYTES) as c_int)
            .collect();
        let mut all = vec![0u8; total];
        let rc = unsafe {
            (self.fns.allgatherv)(
                metadata.as_ptr().cast(),
                WIRE_META_BYTES as c_int,
                self.fns.byte,
                all.as_mut_ptr().cast(),
                counts.as_ptr(),
                displs.as_ptr(),
                self.fns.byte,
                self.comms[site],
            )
        };
        if rc != MPI_SUCCESS {
            self.abort("MPI_Allgatherv failed during wire metadata handshake");
        }
        let first = &all[..WIRE_META_BYTES];
        let expected_ok = Self::metadata_u64(first, 0) == WIRE_META_OK;
        for chunk in all.chunks_exact(WIRE_META_BYTES) {
            // Segment length at offset 48 is rank-local and may differ for a
            // deliberately uneven partition; all other fields describe the
            // shared wire/layout contract and must match exactly.
            if Self::metadata_u64(chunk, 0) != WIRE_META_OK
                || Self::metadata_u64(chunk, 8) != Self::metadata_u64(first, 8)
                || Self::metadata_u64(chunk, 16) != Self::metadata_u64(first, 16)
                || Self::metadata_u64(chunk, 24) != Self::metadata_u64(first, 24)
                || Self::metadata_u64(chunk, 32) != Self::metadata_u64(first, 32)
                || Self::metadata_u64(chunk, 40) != Self::metadata_u64(first, 40)
            {
                self.abort("wire type, precision, or gather layout differs across ranks");
            }
        }
        if !expected_ok {
            self.abort("local wire encoding or gather range validation failed");
        }
    }

    fn exchange_status(&self, site: usize, ok: bool) {
        let size = self.size as usize;
        let counts = vec![1i32; size];
        let displs: Vec<c_int> = (0..size).map(|rank| rank as c_int).collect();
        let local = [u8::from(ok)];
        let mut all = vec![0u8; size];
        let rc = unsafe {
            (self.fns.allgatherv)(
                local.as_ptr().cast(),
                1,
                self.fns.byte,
                all.as_mut_ptr().cast(),
                counts.as_ptr(),
                displs.as_ptr(),
                self.fns.byte,
                self.comms[site],
            )
        };
        if rc != MPI_SUCCESS {
            self.abort("MPI_Allgatherv failed during wire status handshake");
        }
        if all.iter().any(|&value| value == 0) {
            self.abort("wire encode/decode failed on at least one rank");
        }
    }

    /// Gather variable-length byte segments on the site's communicator.
    /// `ranges` lists every rank's `(offset, len)` position in `out`; `local`
    /// is this rank's segment. All range arithmetic is checked before this
    /// function is called.
    fn gather_bytes(&self, site: usize, local: &[u8], ranges: &[(i32, i32)], out: &mut [u8]) {
        let lens: Vec<i32> = ranges.iter().map(|&(_, length)| length).collect();
        let displs: Vec<i32> = ranges.iter().map(|&(offset, _)| offset).collect();
        let send_count = c_int::try_from(local.len())
            .unwrap_or_else(|_| self.abort("local MPI byte count exceeds INT_MAX"));
        // OpenMPI uses address 1 as MPI_IN_PLACE. An empty Vec<u8> has
        // that same dangling sentinel; use a real address for empty owners.
        let empty_send = 0u8;
        let send_buffer = if local.is_empty() {
            &empty_send as *const u8
        } else {
            local.as_ptr()
        };
        let t0 = std::time::Instant::now();
        let rc = unsafe {
            (self.fns.allgatherv)(
                send_buffer.cast(),
                send_count,
                self.fns.byte,
                out.as_mut_ptr().cast(),
                lens.as_ptr(),
                displs.as_ptr(),
                self.fns.byte,
                self.comms[site],
            )
        };
        let elapsed = t0.elapsed();
        if crate::receipt::profile_requested() {
            eprintln!("PHASE mpi.gather{site} {elapsed:?}");
        }
        crate::receipt::phase_record("mpi.gather", elapsed);
        if rc != MPI_SUCCESS {
            self.abort("MPI_Allgatherv failed");
        }
    }

    /// Gather variable-length byte segments only on `root`. The receive
    /// counts/displacements are still supplied in canonical rank order on
    /// every rank, but non-root ranks pass a dummy receive buffer and retain
    /// no assembled payload.
    fn gather_bytes_root(
        &self,
        site: usize,
        local: &[u8],
        ranges: &[(i32, i32)],
        root: usize,
        out: &mut [u8],
    ) {
        let lens: Vec<i32> = ranges.iter().map(|&(_, length)| length).collect();
        let displs: Vec<i32> = ranges.iter().map(|&(offset, _)| offset).collect();
        let send_count = c_int::try_from(local.len())
            .unwrap_or_else(|_| self.abort("local MPI byte count exceeds INT_MAX"));
        let empty_send = 0u8;
        let send_buffer = if local.is_empty() {
            &empty_send as *const u8
        } else {
            local.as_ptr()
        };
        let mut empty_receive = 0u8;
        let receive_buffer = if out.is_empty() {
            &mut empty_receive as *mut u8
        } else {
            out.as_mut_ptr()
        };
        let t0 = std::time::Instant::now();
        let rc = unsafe {
            (self.fns.gatherv)(
                send_buffer.cast(),
                send_count,
                self.fns.byte,
                receive_buffer.cast(),
                lens.as_ptr(),
                displs.as_ptr(),
                self.fns.byte,
                c_int::try_from(root).unwrap_or_else(|_| self.abort("MPI root exceeds INT_MAX")),
                self.comms[site],
            )
        };
        let elapsed = t0.elapsed();
        if crate::receipt::profile_requested() {
            eprintln!("PHASE mpi.gather_root{site} {elapsed:?}");
        }
        crate::receipt::phase_record("mpi.gather_root", elapsed);
        if rc != MPI_SUCCESS {
            self.abort("MPI_Gatherv failed");
        }
    }

    /// Elementwise gather through the Scalar wire codec. Raw Rust object
    /// bytes are never sent: the metadata handshake checks format, precision,
    /// element width, and the complete range layout on every rank.
    pub(crate) fn gather_slice<T: Scalar>(
        &self,
        site: usize,
        segment: &[T],
        ranges: &[(usize, usize)],
        out: &mut [T],
    ) {
        self.ensure_live();
        if site >= NSITES {
            self.abort("invalid MPI collective site");
        }
        let size = self.size as usize;
        let rank = self.rank as usize;
        let wire_size = T::wire_size();
        let layout_hash = Self::layout_hash(ranges, out.len());
        let mut valid = ranges.len() == size
            && rank < ranges.len()
            && wire_size.is_some_and(|n| n > 0)
            && T::wire_tag() != 0
            && layout_hash.is_some();
        let width = wire_size.unwrap_or(0);
        let expected = ranges.get(rank).copied().unwrap_or((0, 0));
        if segment.len() != expected.1 {
            valid = false;
        }
        let local_bytes_len = expected
            .1
            .checked_mul(width)
            .filter(|&length| length <= MPI_INT_MAX);
        let output_bytes_len = out
            .len()
            .checked_mul(width)
            .filter(|&length| length <= MPI_INT_MAX);
        if local_bytes_len.is_none() || output_bytes_len.is_none() {
            valid = false;
        }

        // Do not allocate payload buffers until every local range check has
        // passed. An invalid rank still sends a zero-length payload after the
        // metadata exchange, allowing all ranks to fail together without an
        // attacker-controlled range forcing a multi-gigabyte allocation.
        let mut byte_ranges = Vec::new();
        if valid && !Self::ranges_nonoverlapping(ranges) {
            valid = false;
        }
        if valid {
            if byte_ranges.try_reserve_exact(ranges.len()).is_err() {
                valid = false;
            } else {
                for &(offset, length) in ranges {
                    let Some(byte_offset) = offset.checked_mul(width) else {
                        valid = false;
                        break;
                    };
                    let Some(byte_length) = length.checked_mul(width) else {
                        valid = false;
                        break;
                    };
                    let in_output = offset
                        .checked_add(length)
                        .is_some_and(|end| end <= out.len());
                    let in_mpi_int = byte_offset <= MPI_INT_MAX
                        && byte_length <= MPI_INT_MAX
                        && byte_offset
                            .checked_add(byte_length)
                            .is_some_and(|end| end <= MPI_INT_MAX);
                    if !in_output || !in_mpi_int {
                        valid = false;
                        break;
                    }
                    byte_ranges.push((
                        c_int::try_from(byte_offset).unwrap_or(0),
                        c_int::try_from(byte_length).unwrap_or(0),
                    ));
                }
            }
        }

        let mut local_bytes = Vec::new();
        let mut output_bytes = Vec::new();
        if valid {
            let (bytes, allocated) = Self::try_zeroed(local_bytes_len.unwrap_or(0));
            local_bytes = bytes;
            valid &= allocated;
        }
        if valid {
            let (bytes, allocated) = Self::try_zeroed(output_bytes_len.unwrap_or(0));
            output_bytes = bytes;
            valid &= allocated;
        }
        if valid {
            for (index, value) in segment.iter().copied().enumerate() {
                let begin = index * width;
                if !value.write_wire(&mut local_bytes[begin..begin + width]) {
                    valid = false;
                }
            }
            for (index, value) in out.iter().copied().enumerate() {
                let begin = index * width;
                if !value.write_wire(&mut output_bytes[begin..begin + width]) {
                    valid = false;
                }
            }
        }
        let mut metadata = [0u8; WIRE_META_BYTES];
        Self::put_metadata_u64(
            &mut metadata,
            0,
            if valid { WIRE_META_OK } else { WIRE_META_BAD },
        );
        Self::put_metadata_u64(
            &mut metadata,
            8,
            wire_size.and_then(|n| u64::try_from(n).ok()).unwrap_or(0),
        );
        Self::put_metadata_u64(&mut metadata, 16, T::wire_tag());
        Self::put_metadata_u64(&mut metadata, 24, u64::try_from(out.len()).unwrap_or(0));
        Self::put_metadata_u64(&mut metadata, 32, u64::try_from(ranges.len()).unwrap_or(0));
        Self::put_metadata_u64(&mut metadata, 40, layout_hash.unwrap_or(0));
        Self::put_metadata_u64(&mut metadata, 48, u64::try_from(segment.len()).unwrap_or(0));
        self.exchange_wire_metadata(site, &metadata);
        // The checked metadata proves every rank has zero payload here.
        // Empty Vec<u8> uses address 1, which OpenMPI interprets as
        // MPI_IN_PLACE and rejects for a receive buffer even at count zero.
        // Keep the common handshake, then omit that vacuous payload exchange.
        if out.is_empty() {
            return;
        }
        self.gather_bytes(site, &local_bytes, &byte_ranges, &mut output_bytes);

        let mut decode_ok = width > 0 && output_bytes.len() % width == 0;
        if decode_ok {
            for (value, bytes) in out.iter_mut().zip(output_bytes.chunks_exact(width)) {
                match T::read_wire(bytes) {
                    Some(decoded) => *value = decoded,
                    None => {
                        decode_ok = false;
                        break;
                    }
                }
            }
        }
        self.exchange_status(site, decode_ok);
    }

    /// Gather variable-length Scalar segments only on `root` through the
    /// canonical wire codec. Non-root ranks allocate no output payload and
    /// return `None`.
    pub(crate) fn gather_slice_root<T: Scalar>(
        &self,
        site: usize,
        segment: &[T],
        ranges: &[(usize, usize)],
        root: usize,
        out_len: usize,
    ) -> Option<Vec<T>> {
        self.ensure_live();
        if site >= NSITES {
            self.abort("invalid MPI collective site");
        }
        let size = self.size as usize;
        let rank = self.rank as usize;
        if root >= size {
            self.abort("invalid MPI root rank");
        }
        let wire_size = T::wire_size();
        let layout_hash = Self::layout_hash(ranges, out_len);
        let mut valid = ranges.len() == size
            && rank < ranges.len()
            && wire_size.is_some_and(|n| n > 0)
            && T::wire_tag() != 0
            && layout_hash.is_some();
        let width = wire_size.unwrap_or(0);
        let expected = ranges.get(rank).copied().unwrap_or((0, 0));
        if segment.len() != expected.1 {
            valid = false;
        }
        let local_bytes_len = expected
            .1
            .checked_mul(width)
            .filter(|&length| length <= MPI_INT_MAX);
        let output_bytes_len = if rank == root {
            out_len
                .checked_mul(width)
                .filter(|&length| length <= MPI_INT_MAX)
        } else {
            Some(0)
        };
        if local_bytes_len.is_none() || output_bytes_len.is_none() {
            valid = false;
        }
        let mut byte_ranges = Vec::new();
        if valid && !Self::ranges_nonoverlapping(ranges) {
            valid = false;
        }
        if valid {
            if byte_ranges.try_reserve_exact(ranges.len()).is_err() {
                valid = false;
            } else {
                for &(offset, length) in ranges {
                    let Some(byte_offset) = offset.checked_mul(width) else {
                        valid = false;
                        break;
                    };
                    let Some(byte_length) = length.checked_mul(width) else {
                        valid = false;
                        break;
                    };
                    let in_output = offset.checked_add(length).is_some_and(|end| end <= out_len);
                    let in_mpi_int = byte_offset <= MPI_INT_MAX
                        && byte_length <= MPI_INT_MAX
                        && byte_offset
                            .checked_add(byte_length)
                            .is_some_and(|end| end <= MPI_INT_MAX);
                    if !in_output || !in_mpi_int {
                        valid = false;
                        break;
                    }
                    byte_ranges.push((
                        c_int::try_from(byte_offset).unwrap_or(0),
                        c_int::try_from(byte_length).unwrap_or(0),
                    ));
                }
            }
        }
        let mut local_bytes = Vec::new();
        let mut output_bytes = Vec::new();
        if valid {
            let (bytes, allocated) = Self::try_zeroed(local_bytes_len.unwrap_or(0));
            local_bytes = bytes;
            valid &= allocated;
        }
        if valid && rank == root {
            let (bytes, allocated) = Self::try_zeroed(output_bytes_len.unwrap_or(0));
            output_bytes = bytes;
            valid &= allocated;
        }
        if valid {
            for (index, value) in segment.iter().copied().enumerate() {
                let begin = index * width;
                if !value.write_wire(&mut local_bytes[begin..begin + width]) {
                    valid = false;
                }
            }
        }
        let mut metadata = [0u8; WIRE_META_BYTES];
        Self::put_metadata_u64(
            &mut metadata,
            0,
            if valid { WIRE_META_OK } else { WIRE_META_BAD },
        );
        Self::put_metadata_u64(
            &mut metadata,
            8,
            wire_size.and_then(|n| u64::try_from(n).ok()).unwrap_or(0),
        );
        Self::put_metadata_u64(&mut metadata, 16, T::wire_tag());
        Self::put_metadata_u64(&mut metadata, 24, u64::try_from(out_len).unwrap_or(0));
        Self::put_metadata_u64(&mut metadata, 32, u64::try_from(ranges.len()).unwrap_or(0));
        Self::put_metadata_u64(&mut metadata, 40, layout_hash.unwrap_or(0));
        Self::put_metadata_u64(&mut metadata, 48, u64::try_from(segment.len()).unwrap_or(0));
        self.exchange_wire_metadata(site, &metadata);
        if out_len == 0 {
            return (rank == root).then(Vec::new);
        }
        self.gather_bytes_root(site, &local_bytes, &byte_ranges, root, &mut output_bytes);
        let mut decoded = Vec::new();
        let mut decode_ok = rank != root || (width > 0 && output_bytes.len() % width == 0);
        if decode_ok && rank == root && decoded.try_reserve_exact(out_len).is_err() {
            decode_ok = false;
        }
        if decode_ok && rank == root {
            for bytes in output_bytes.chunks_exact(width) {
                match T::read_wire(bytes) {
                    Some(value) => decoded.push(value),
                    None => {
                        decode_ok = false;
                        break;
                    }
                }
            }
        }
        self.exchange_status(
            site,
            decode_ok && (rank != root || decoded.len() == out_len),
        );
        (rank == root).then_some(decoded)
    }

    /// Maximum of `v` over all ranks on the world communicator. Order-free
    /// and deterministic; callers must not invoke it concurrently with other
    /// collectives on the same stream.
    /// Elementwise maximum of `values` over all ranks, in place. One
    /// collective for several small agreement values.
    pub(crate) fn allreduce_max_f64_slice(&self, values: &mut [f64]) {
        self.ensure_collective_thread();
        self.ensure_live();
        if values.is_empty() {
            return;
        }
        let send = values.to_vec();
        let t0 = std::time::Instant::now();
        let rc = unsafe {
            (self.fns.allreduce)(
                send.as_ptr().cast::<c_void>(),
                values.as_mut_ptr().cast::<c_void>(),
                c_int::try_from(values.len()).unwrap_or(c_int::MAX),
                self.fns.double,
                self.fns.max,
                self.fns.comm_world,
            )
        };
        crate::receipt::phase_record("mpi.allreduce", t0.elapsed());
        if rc != MPI_SUCCESS {
            self.abort("MPI_Allreduce failed");
        }
    }

    pub(crate) fn allreduce_max_f64(&self, v: f64) -> f64 {
        // Every collective shares this world stream; one issued from a pool
        // worker concurrently with the control thread's would deadlock or
        // corrupt MPI state, so fail loudly instead.
        self.ensure_collective_thread();
        self.ensure_live();
        let mut out = 0.0f64;
        let t0 = std::time::Instant::now();
        let rc = unsafe {
            (self.fns.allreduce)(
                (&v as *const f64).cast::<c_void>(),
                (&mut out as *mut f64).cast::<c_void>(),
                1,
                self.fns.double,
                self.fns.max,
                self.fns.comm_world,
            )
        };
        // Includes waiting for the slowest rank: the first decision after
        // unbalanced work absorbs the imbalance.
        crate::receipt::phase_record("mpi.allreduce", t0.elapsed());
        if rc != MPI_SUCCESS {
            self.abort("MPI_Allreduce failed");
        }
        out
    }
}

/// Partition `count` items into `size` contiguous `(offset, len)` ranges.
pub(crate) fn ranges(count: usize, size: usize) -> Vec<(usize, usize)> {
    let base = count / size.max(1);
    let rem = count % size.max(1);
    (0..size)
        .map(|rank| {
            let begin = rank * base + rank.min(rem);
            (begin, base + usize::from(rank < rem))
        })
        .collect()
}

/// Partition `costs` into `size` contiguous `(offset, len)` ranges of
/// near-equal total cost. Boundaries sit at the prefix sum closest to each
/// `rank * total / size` mark, so equal-cost items split exactly like
/// [`ranges`]. Deterministic for a fixed cost array — every rank computes
/// the identical partition. Empty ranges are allowed.
pub(crate) fn cost_ranges(costs: &[u64], size: usize) -> Vec<(usize, usize)> {
    let n = costs.len();
    let size = size.max(1);
    let mut prefix = Vec::with_capacity(n + 1);
    prefix.push(0u128);
    for &c in costs {
        prefix.push(prefix.last().unwrap().saturating_add(c as u128));
    }
    let total = prefix[n];
    if total == 0 {
        return ranges(n, size);
    }
    let mut out = Vec::with_capacity(size);
    let mut begin = 0usize;
    for rank in 1..size {
        // Avoid `total * rank` overflow even for a synthetic cost vector
        // whose saturated prefix reaches u128::MAX.
        let q = total / size as u128;
        let r = total % size as u128;
        let target = q.saturating_mul(rank as u128) + r.saturating_mul(rank as u128) / size as u128;
        let mut end = begin;
        while end < n && prefix[end + 1] <= target {
            end += 1;
        }
        // prefix[end] <= target < prefix[end+1]; take the closer boundary.
        if end < n && (prefix[end + 1] - target) < (target - prefix[end]) {
            end += 1;
        }
        out.push((begin, end - begin));
        begin = end;
    }
    out.push((begin, n - begin));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Zero;
    use sdpx_arithmetic::Bits768;

    #[test]
    fn partition_covers_once() {
        for count in [0, 1, 5, 27, 28, 100] {
            for size in 1..=8usize {
                let mut seen = 0usize;
                for (begin, len) in ranges(count, size) {
                    assert_eq!(begin, seen, "count={count} size={size}");
                    seen += len;
                }
                assert_eq!(seen, count, "count={count} size={size}");
            }
        }
        assert_eq!(
            ranges(28, 8),
            [
                (0, 4),
                (4, 4),
                (8, 4),
                (12, 4),
                (16, 3),
                (19, 3),
                (22, 3),
                (25, 3)
            ]
        );
    }

    #[test]
    fn cost_partition_covers_once_and_balances() {
        for count in [0, 1, 5, 27, 28, 100] {
            for size in 1..=8usize {
                let costs = vec![1u64; count];
                let mut seen = 0usize;
                for (begin, len) in cost_ranges(&costs, size) {
                    assert_eq!(begin, seen, "count={count} size={size}");
                    seen += len;
                }
                assert_eq!(seen, count);
            }
        }
        // Prefix boundary closest to total/2: [100,1,1]=102 vs [1,1,100]=102.
        let costs = [100u64, 1, 1, 1, 1, 100];
        let r = cost_ranges(&costs, 2);
        assert_eq!(r, [(0, 3), (3, 3)]);
        // Degenerate: all-zero costs split by count.
        assert_eq!(cost_ranges(&[0; 6], 3), [(0, 2), (2, 2), (4, 2)]);
    }

    #[test]
    fn receive_ranges_reject_overlap_but_allow_empty_and_gaps() {
        assert!(World::ranges_nonoverlapping(&[
            (0, 0),
            (2, 1),
            (5, 0),
            (5, 2)
        ]));
        assert!(World::ranges_nonoverlapping(&[(0, 0), (0, 0)]));
        assert!(!World::ranges_nonoverlapping(&[(0, 2), (1, 1)]));
        assert!(!World::ranges_nonoverlapping(&[(usize::MAX, 2)]));
    }

    #[test]
    fn single_process_world() {
        // No MPI environment: the world is absent and callers take the serial
        // path.
        assert!(World::get().is_none());
    }

    /// Run with `mpiexec -n 2 cargo test -p sdpx-solver --lib
    /// mpi::tests::mpi_probe_two_rank_wire_roundtrip -- --ignored --exact`.
    /// The ordinary test suite never starts MPI because this probe is ignored.
    #[test]
    #[ignore]
    fn mpi_probe_two_rank_wire_roundtrip() {
        let world = World::get().expect("MPI probe requires an active MPI world");
        assert_eq!(world.size(), 2, "probe requires exactly two MPI ranks");
        let context = MpiContext::initialize();
        assert!(context.agree_signature([7; 32]));
        assert!(!context.agree_signature([world.rank() as u8; 32]));
        assert!(!context.all_succeeded(world.rank() == 0));
        assert!(context.all_succeeded(true));

        // Uneven non-empty ownership exercises byte displacements and keeps
        // the two-rank probe small enough to run directly on a test binary.
        let f64_layout = [(0usize, 1usize), (1usize, 2usize)];
        let local_f64: Vec<f64> = if world.rank() == 0 {
            vec![1.25]
        } else {
            vec![-3.5, 8.75]
        };
        let mut all_f64 = [0.0f64; 3];
        world.gather_slice(SITE_FORWARD, &local_f64, &f64_layout, &mut all_f64);
        assert_eq!(all_f64, [1.25, -3.5, 8.75]);

        // The MPFR wire path must preserve decimal values and tiny exponents
        // that do not round through Float64. Rank zero intentionally owns an
        // empty segment for this gather.
        let expected_mpfr = [
            "1.234567890123456789012345678901234567890123456789e-123"
                .parse::<Bits768>()
                .unwrap(),
            "-9.876543210987654321098765432109876543210987654321e+77"
                .parse::<Bits768>()
                .unwrap(),
            "3.1415926535897932384626433832795028841971693993751e-222"
                .parse::<Bits768>()
                .unwrap(),
        ];
        let mpfr_layout = [(0usize, 0usize), (0usize, 3usize)];
        let local_mpfr: Vec<Bits768> = if world.rank() == 0 {
            Vec::new()
        } else {
            expected_mpfr.to_vec()
        };
        let mut all_mpfr = vec![Bits768::zero(); 3];
        world.gather_slice(SITE_ADJOINT, &local_mpfr, &mpfr_layout, &mut all_mpfr);
        assert_eq!(all_mpfr, expected_mpfr);
        world.gather_slice::<f64>(SITE_GRAM, &[], &[(0, 0), (0, 0)], &mut []);
        world.gather_slice::<Bits768>(SITE_GRAM, &[], &[(0, 0), (0, 0)], &mut []);
        finalize_owned();
    }

    /// A rank-local elapsed clock or callback must stop every rank before
    /// any peer enters another numerical collective.
    #[test]
    #[ignore]
    fn mpi_probe_two_rank_stopping() {
        use crate::algebra::CscMatrix;
        use crate::solver::*;
        let context = MpiContext::initialize();
        assert_eq!(context.size(), 2);
        let p = CscMatrix::<f64>::zeros((2, 2));
        let a = CscMatrix::from(&[[-1.0, -1.0], [-1.0, 0.0], [0.0, -1.0]]);
        let q = [1.0, 1.0];
        let b = [-1.0, 0.0, 0.0];
        let cones = [NonnegativeConeT(3)];
        let mut settings = DefaultSettings::default();
        settings.verbose = false;
        settings.max_threads = 1;
        settings.time_limit = 0.25;
        let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings).unwrap();
        let timers = solver.timers.as_mut().unwrap();
        *timers = crate::timers::Timers::default();
        if context.rank() == 0 {
            timers.start_setup();
            std::thread::sleep(std::time::Duration::from_millis(500));
            timers.stop_setup();
        }
        assert!(context
            .all_succeeded((timers.total_time().as_secs_f64() > 0.25) == (context.rank() == 0)));
        solver.solve();
        assert_eq!(solver.solution.status, SolverStatus::MaxTime);
        assert_eq!(solver.solution.iterations, 0);
        assert!(solver.solution.solve_time >= 0.5);

        let mut settings = DefaultSettings::default();
        settings.verbose = false;
        settings.max_threads = 1;
        let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings).unwrap();
        solver.set_termination_callback(move |_: &DefaultInfo<f64>| context.rank() == 0);
        solver.solve();
        assert_eq!(solver.solution.status, SolverStatus::CallbackTerminated);
        assert_eq!(solver.solution.iterations, 0);
        context.finish();
    }

    /// This probe is expected to terminate the two-rank launcher collectively
    /// with a nonzero status. Rank one mutates the global range layout; the
    /// wire metadata handshake must reject the mismatch before payload gather.
    /// Run with `mpiexec -n 2 cargo test -p
    /// sdpx-solver --lib mpi::tests::mpi_probe_two_rank_mismatched_layout
    /// -- --ignored --exact`; do not include it in a passing test batch.
    #[test]
    #[ignore]
    fn mpi_probe_two_rank_mismatched_layout() {
        let world = World::get().expect("MPI mismatch probe requires an active MPI world");
        assert_eq!(world.size(), 2, "probe requires exactly two MPI ranks");
        let mut layout = ranges(2, world.size());
        if world.rank() == 1 {
            layout[0].1 = layout[0].1.saturating_add(1);
        }
        let local = [world.rank() as f64];
        let mut all = [0.0f64; 2];
        world.gather_slice(SITE_FORWARD, &local, &layout, &mut all);
    }
}
