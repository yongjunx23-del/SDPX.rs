//! Optional MPI world for cross-node block sharding.
//!
//! Loaded lazily through `dlopen` so the dependency stays optional: no MPI
//! runtime, an initialization failure, and a single-process world all fall
//! back to the ordinary serial path. The world activates only when the
//! environment advertises more than one rank, so single-process solves never
//! touch MPI.
//!
//! The shard discipline is rank-ordered block partitioning: every block is
//! evaluated on exactly one rank with identical arithmetic, partial outputs
//! are exchanged with `MPI_Allgatherv`, and scalar reductions use `MPI_MAX`
//! (order-free), so results are bitwise identical for any rank count.

#![allow(non_snake_case)]

use std::ffi::{c_char, c_int, c_void, CString};
use std::sync::OnceLock;

type MpiComm = *mut c_void;
type MpiDatatype = *mut c_void;
type MpiOp = *mut c_void;

extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}
const RTLD_NOW: c_int = 2;
// RTLD_LOCAL is 0 on Linux/macOS; the value 4 is RTLD_NOLOAD, which returns
// null for not-yet-loaded libraries without setting dlerror.
const RTLD_LOCAL: c_int = 0;

/// Shard sites run collectives on dedicated communicators so concurrent
/// gathers from different solver phases never interleave on one stream.
pub(crate) const SITE_FORWARD: usize = 0;
pub(crate) const SITE_ADJOINT: usize = 1;
pub(crate) const SITE_SCALING: usize = 2;
pub(crate) const SITE_GRAM: usize = 3;
pub(crate) const SITE_RX: usize = 4;
pub(crate) const SITE_RZ: usize = 5;
pub(crate) const SITE_CONES: usize = 6;
const NSITES: usize = 8;

#[derive(Clone, Copy)]
struct Fns {
    init_thread: unsafe extern "C" fn(*mut c_int, *mut *mut *mut u8, c_int, *mut c_int) -> c_int,
    initialized: unsafe extern "C" fn(*mut c_int) -> c_int,
    comm_size: unsafe extern "C" fn(MpiComm, *mut c_int) -> c_int,
    comm_rank: unsafe extern "C" fn(MpiComm, *mut c_int) -> c_int,
    comm_dup: unsafe extern "C" fn(MpiComm, *mut MpiComm) -> c_int,
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    double: MpiDatatype,
    #[allow(dead_code)]
    max: MpiOp,
}

// OpenMPI handles are pointers to exported objects; MPICH-style builds use
// small integers instead. Resolve whichever convention the library exports.
unsafe fn resolve_handle(lib: *mut c_void, variable: &str, fallback: usize) -> *mut c_void {
    let name = CString::new(variable).unwrap();
    let symbol = unsafe { dlsym(lib, name.as_ptr()) };
    // OpenMPI predefined handles are the *addresses* of global struct
    // instances (MPI_COMM_WORLD = &ompi_mpi_comm_world), so the dlsym result
    // is already the handle. MPICH-style integer constants remain the
    // fallback when the OpenMPI globals are absent.
    if symbol.is_null() {
        fallback as *mut c_void
    } else {
        symbol
    }
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
    let mut candidates: Vec<CString> = NAMES
        .iter()
        .map(|n| CString::new(*n).unwrap())
        .collect();
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
            for n in NAMES {
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
                eprintln!(
                    "mpi: dlopen({}) failed: {msg}",
                    name.to_string_lossy()
                );
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
            return Some(Fns {
                init_thread: sym!(lib, "MPI_Init_thread"),
                initialized: sym!(lib, "MPI_Initialized"),
                comm_size: sym!(lib, "MPI_Comm_size"),
                comm_rank: sym!(lib, "MPI_Comm_rank"),
                comm_dup: sym!(lib, "MPI_Comm_dup"),
                allgatherv: sym!(lib, "MPI_Allgatherv"),
                allreduce: sym!(lib, "MPI_Allreduce"),
                comm_world: resolve_handle(lib, "ompi_mpi_comm_world", 0x4400_0000),
                byte: resolve_handle(lib, "ompi_mpi_byte", 0x4c00_000d),
                double: resolve_handle(lib, "ompi_mpi_double", 0x4c00_0011),
                max: resolve_handle(lib, "ompi_mpi_op_max", 0x5800_0001),
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
}

unsafe impl Send for World {}
unsafe impl Sync for World {}
unsafe impl Sync for Fns {}

static WORLD: OnceLock<Option<World>> = OnceLock::new();

impl World {
    /// The shared world, initialized once. `None` unless the environment
    /// advertises more than one rank and MPI initializes successfully.
    pub(crate) fn get() -> Option<Self> {
        *WORLD.get_or_init(Self::init)
    }

    fn init() -> Option<Self> {
        let debug = std::env::var_os("SDPX_MPI_DEBUG").is_some();
        if advertised_size() <= 1 && std::env::var_os("SDPX_MPI").is_none() {
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
                return None;
            }
        };
        if debug {
            eprintln!("mpi: library loaded");
        }
        let mut flag = 0;
        if unsafe { (fns.initialized)(&mut flag) } != 0 {
            return None;
        }
        if flag == 0 {
            // MPI_THREAD_MULTIPLE: gathers may be issued from rayon worker
            // threads on their dedicated per-site communicators.
            let mut provided = 0;
            let rc = unsafe {
                (fns.init_thread)(std::ptr::null_mut(), std::ptr::null_mut(), 3, &mut provided)
            };
            if debug {
                eprintln!("mpi: init_thread rc={rc} provided={provided}");
            }
            if rc != 0 || provided < 3 {
                return None;
            }
        }
        let (mut size, mut rank) = (0, 0);
        unsafe {
            (fns.comm_size)(fns.comm_world, &mut size);
            (fns.comm_rank)(fns.comm_world, &mut rank);
        }
        if debug {
            eprintln!("mpi: size={size} rank={rank}");
        }
        if size <= 1 {
            return None;
        }
        let comms: &'static mut [MpiComm; NSITES] =
            Box::leak(Box::new([std::ptr::null_mut(); NSITES]));
        for c in comms.iter_mut() {
            if unsafe { (fns.comm_dup)(fns.comm_world, c) } != 0 {
                return None;
            }
        }
        Some(Self {
            fns: Box::leak(Box::new(fns)),
            comms,
            rank,
            size,
        })
    }

    pub(crate) fn size(&self) -> usize {
        self.size as usize
    }

    pub(crate) fn rank(&self) -> usize {
        self.rank as usize
    }

    /// The `[begin, end)` range of `count` items owned by this rank.
    pub(crate) fn range(&self, count: usize) -> std::ops::Range<usize> {
        let r = ranges(count, self.size as usize)[self.rank as usize];
        r.0..r.0 + r.1
    }

    /// Gather variable-length byte segments on the site's communicator.
    /// `ranges` lists every rank's `(offset, len)` position in `out`; `local`
    /// is this rank's segment. All ranks must agree on `ranges` and `out`.
    fn gather(&self, site: usize, local: &[u8], ranges: &[(i32, i32)], out: &mut [u8]) {
        assert_eq!(ranges.len(), self.size as usize);
        let lens: Vec<i32> = ranges.iter().map(|&(_, l)| l).collect();
        let displs: Vec<i32> = ranges.iter().map(|&(o, _)| o).collect();
        let t0 = std::env::var_os("SDPX_PROFILE").map(|_| std::time::Instant::now());
        let rc = unsafe {
            (self.fns.allgatherv)(
                local.as_ptr().cast(),
                local.len() as c_int,
                self.fns.byte,
                out.as_mut_ptr().cast(),
                lens.as_ptr(),
                displs.as_ptr(),
                self.fns.byte,
                self.comms[site],
            )
        };
        if let Some(t0) = t0 {
            eprintln!("PHASE mpi.gather{site} {:?}", t0.elapsed());
        }
        assert_eq!(rc, 0, "MPI_Allgatherv failed");
    }

    /// Elementwise gather of a `Copy` buffer sharing its layout on every
    /// rank. `segment` is this rank's contribution; `ranges` are element
    /// `(offset, len)` positions in `out`.
    pub(crate) fn gather_slice<T: Copy>(
        &self,
        site: usize,
        segment: &[T],
        ranges: &[(usize, usize)],
        out: &mut [T],
    ) {
        let byte_ranges: Vec<(i32, i32)> = ranges
            .iter()
            .map(|&(o, l)| {
                (
                    (o * std::mem::size_of::<T>()) as i32,
                    (l * std::mem::size_of::<T>()) as i32,
                )
            })
            .collect();
        let local = unsafe {
            std::slice::from_raw_parts(
                segment.as_ptr().cast::<u8>(),
                byte_ranges[self.rank as usize].1 as usize,
            )
        };
        let out_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                out.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(out),
            )
        };
        self.gather(site, local, &byte_ranges, out_bytes);
    }

    /// Maximum of `v` over all ranks on the world communicator. Order-free
    /// and deterministic; callers must not invoke it concurrently with other
    /// collectives on the same stream.
    pub(crate) fn allreduce_max_f64(&self, v: f64) -> f64 {
        let mut out = 0.0f64;
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
        assert_eq!(rc, 0, "MPI_Allreduce failed");
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

#[cfg(test)]
mod tests {
    use super::*;

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
            [(0, 4), (4, 4), (8, 4), (12, 4), (16, 3), (19, 3), (22, 3), (25, 3)]
        );
    }

    #[test]
    fn single_process_world() {
        // No MPI environment: the world is absent and callers take the serial
        // path.
        assert!(World::get().is_none());
    }
}
