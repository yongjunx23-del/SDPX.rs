//! Lossless KKT/RHS snapshots and LDL-level replay (plan PR-02).
//!
//! `SDPX_SNAPSHOT=<dir>` captures each `DirectLDLKKTSolver::solve` call as a
//! `snap-NNNN.bin` plus a `snap-NNNN.json` manifest: the authoritative CSC
//! matrix, diagonal signs, the right-hand side as presented, the post-solve
//! `x`, and a sha256 checksum. `SDPX_SNAPSHOT_AT=0,12,…` restricts capture to
//! those call indices; failing solves are always captured. Values serialize
//! through `Scalar::scalar_exact_encode` — sign/kind, exponent and limbs —
//! with no pointers, text or f64 intermediates.

use crate::algebra::{AsFloatT, CscMatrix, FloatT, VectorMath};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const MAGIC: &[u8; 8] = b"SDPXSNP2";
const MAGIC_V1: &[u8; 8] = b"SDPXSNP1";

/// One captured KKT solve: matrix, signs, RHS as presented, and the recorded
/// solution (empty when the solve failed).
pub struct KktSnapshot<T> {
    /// Scalar precision in bits.
    pub precision_bits: u32,
    /// KKT order.
    pub n: usize,
    /// CSC column pointers (length n+1).
    pub colptr: Vec<u64>,
    /// CSC row indices (length nnz).
    pub rowval: Vec<u64>,
    /// CSC values.
    pub nzval: Vec<T>,
    /// Per-variable diagonal sign (+1/-1).
    pub dsigns: Vec<i8>,
    /// Right-hand side as presented to the LDL solve.
    pub rhs: Vec<T>,
    /// Recorded post-solve vector, empty when `solved` is false.
    pub x: Vec<T>,
    /// Whether the recorded solve succeeded.
    pub solved: bool,
    /// Static regularization escalation level active at capture time.
    /// V1 snapshots carry no level and decode as 0.
    pub reg_boost: usize,
}

fn wanted_indices() -> &'static Option<Vec<u64>> {
    static AT: OnceLock<Option<Vec<u64>>> = OnceLock::new();
    AT.get_or_init(|| {
        std::env::var("SDPX_SNAPSHOT_AT").ok().map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<u64>().ok())
                .collect()
        })
    })
}

/// Whether `SDPX_SNAPSHOT` is set.
pub fn enabled() -> bool {
    capture_enabled()
}

fn capture_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("SDPX_SNAPSHOT").is_some())
}

/// True when call `idx` should be captured: with no `SDPX_SNAPSHOT_AT` list
/// every call is captured; otherwise only listed indices. Failing solves are
/// always captured.
pub fn wanted(idx: u64, failed: bool) -> bool {
    capture_enabled() && (failed || wanted_indices().as_ref().map_or(true, |v| v.contains(&idx)))
}

fn enc<T: FloatT>(w: &mut impl Write, v: &T, nlimbs: usize) -> io::Result<()> {
    let (k, e, limbs) = v.scalar_exact_encode().ok_or_else(|| {
        io::Error::new(io::ErrorKind::Unsupported, "scalar has no exact encoding")
    })?;
    if limbs.len() != nlimbs {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "inconsistent limb count",
        ));
    }
    w.write_all(&k.to_le_bytes())?;
    w.write_all(&e.to_le_bytes())?;
    for l in &limbs {
        w.write_all(&l.to_le_bytes())?;
    }
    Ok(())
}

fn dec<T: FloatT>(r: &mut impl Read, nlimbs: usize) -> io::Result<T> {
    let mut k = [0u8; 4];
    let mut e = [0u8; 8];
    r.read_exact(&mut k)?;
    r.read_exact(&mut e)?;
    let mut limbs = vec![0u64; nlimbs];
    for l in &mut limbs {
        let mut b = [0u8; 8];
        r.read_exact(&mut b)?;
        *l = u64::from_le_bytes(b);
    }
    T::scalar_exact_decode(i32::from_le_bytes(k), i64::from_le_bytes(e), &limbs)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad scalar encoding"))
}

fn fnv1a64(data: &[u8]) -> String {
    let mut h = 0xcbf29ce484222325u64;
    for &b in data {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

/// Serialize one snapshot and its manifest sidecar. Returns `None` when the
/// scalar type has no exact encoding.
pub fn write<T: FloatT>(
    dir: &Path,
    idx: u64,
    kkt: &CscMatrix<T>,
    dsigns: &[i8],
    rhs: &[T],
    x: &[T],
    solved: bool,
    backend: &str,
    reg_boost: usize,
) -> io::Result<Option<PathBuf>> {
    let nlimbs = if let Some(v) = kkt.nzval.first().or(rhs.first()) {
        match v.scalar_exact_encode() {
            Some((_, _, l)) => l.len(),
            None => return Ok(None),
        }
    } else {
        0
    };
    let mut buf = Vec::new();
    buf.write_all(MAGIC)?;
    let prec = T::precision_bits() as u64;
    for v in [
        prec,
        kkt.n as u64,
        kkt.nzval.len() as u64,
        rhs.len() as u64,
        x.len() as u64,
        nlimbs as u64,
        solved as u64,
        reg_boost as u64,
    ] {
        buf.write_all(&v.to_le_bytes())?;
    }
    for &p in &kkt.colptr {
        buf.write_all(&(p as u64).to_le_bytes())?;
    }
    for &r in &kkt.rowval {
        buf.write_all(&(r as u64).to_le_bytes())?;
    }
    for &s in dsigns {
        buf.write_all(&[s as u8])?;
    }
    for v in kkt.nzval.iter().chain(rhs).chain(x) {
        enc(&mut buf, v, nlimbs)?;
    }
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("snap-{idx:04}.bin"));
    let checksum = fnv1a64(&buf);
    std::fs::write(&path, &buf)?;
    let manifest = format!(
        concat!(
            "{{\n",
            "  \"schema_version\": 2,\n",
            "  \"file\": {:?},\n",
            "  \"fnv1a64\": {:?},\n",
            "  \"call_index\": {},\n",
            "  \"solved\": {},\n",
            "  \"precision_bits\": {},\n",
            "  \"reg_boost\": {},\n",
            "  \"shape\": {{\"n\": {}, \"nnz\": {}, \"rhs\": {}}},\n",
            "  \"ordering\": \"csc upper-triangular, sorted rowval per column\",\n",
            "  \"backend\": {:?},\n",
            "  \"git_hash\": {:?}\n",
            "}}\n"
        ),
        path.file_name().unwrap().to_string_lossy(),
        checksum,
        idx,
        solved,
        prec,
        reg_boost,
        kkt.n,
        kkt.nzval.len(),
        rhs.len(),
        backend,
        option_env!("SDPX_GIT_HASH")
            .map(|h| format!("\"{h}\""))
            .unwrap_or_else(|| "null".to_string()),
    );
    std::fs::write(dir.join(format!("snap-{idx:04}.json")), manifest)?;
    Ok(Some(path))
}

/// Load a snapshot file. `T` must match the recorded `precision_bits`.
pub fn load<T: FloatT>(path: &Path) -> io::Result<KktSnapshot<T>> {
    let data = std::fs::read(path)?;
    let mut r = &data[..];
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    let v1 = &magic == MAGIC_V1;
    if !v1 && &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }
    let u64s = |r: &mut &[u8]| -> io::Result<u64> {
        let mut b = [0u8; 8];
        r.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    };
    let prec = u64s(&mut r)?;
    if prec != T::precision_bits() as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "precision {prec} does not match {} bits",
                T::precision_bits()
            ),
        ));
    }
    let (n, nnz, nrhs, nx, nlimbs, solved) = (
        u64s(&mut r)? as usize,
        u64s(&mut r)? as usize,
        u64s(&mut r)? as usize,
        u64s(&mut r)? as usize,
        u64s(&mut r)? as usize,
        u64s(&mut r)? != 0,
    );
    let reg_boost = if v1 { 0 } else { u64s(&mut r)? as usize };
    let mut colptr = Vec::with_capacity(n + 1);
    for _ in 0..=n {
        colptr.push(u64s(&mut r)?);
    }
    let mut rowval = Vec::with_capacity(nnz);
    for _ in 0..nnz {
        rowval.push(u64s(&mut r)?);
    }
    let mut dsigns = vec![0i8; n];
    for s in &mut dsigns {
        let mut b = [0u8; 1];
        r.read_exact(&mut b)?;
        *s = b[0] as i8;
    }
    let mut nzval = Vec::with_capacity(nnz);
    for _ in 0..nnz {
        nzval.push(dec(&mut r, nlimbs)?);
    }
    let mut rhs = Vec::with_capacity(nrhs);
    for _ in 0..nrhs {
        rhs.push(dec(&mut r, nlimbs)?);
    }
    let mut x = Vec::with_capacity(nx);
    for _ in 0..nx {
        x.push(dec(&mut r, nlimbs)?);
    }
    Ok(KktSnapshot {
        precision_bits: prec as u32,
        n,
        colptr,
        rowval,
        nzval,
        dsigns,
        rhs,
        x,
        solved,
        reg_boost,
    })
}

/// Result of replaying one snapshot through an LDL backend.
pub struct ReplayReport {
    /// Backend name reported by the solver.
    pub backend: String,
    /// Whether `refactor` accepted the matrix.
    pub refactored: bool,
    /// Whether the replayed solution vector is finite.
    pub solved: bool,
    /// `||Kx - b|| / (||K|| ||x|| + ||b||)` in f64; NaN when unavailable.
    pub normwise_residual: f64,
    /// `max |Kx - b|` in f64; NaN when unavailable.
    pub max_abs_residual: f64,
    /// Bitwise comparison against the recorded `x` (`None` when none was
    /// recorded).
    pub bitwise_match: Option<bool>,
    /// Factorization wall time.
    pub factor_secs: f64,
    /// Solve wall time.
    pub solve_secs: f64,
}

/// Apply `regularize_and_refactor`'s static diagonal shift to a snapshot's
/// stored values, mirroring production: `diag += sign * eps` with
/// `eps = constant + proportional * max|diag|`.
fn regularized_nzval<T: FloatT>(
    snap: &KktSnapshot<T>,
    settings: &crate::solver::CoreSettings<T>,
    reg_boost: usize,
) -> Vec<T> {
    let mut nzval = snap.nzval.clone();
    if !settings.static_regularization_enable {
        return nzval;
    }
    let mut diagpos = vec![usize::MAX; snap.n];
    for j in 0..snap.n {
        for p in snap.colptr[j] as usize..snap.colptr[j + 1] as usize {
            if snap.rowval[p] as usize == j {
                diagpos[j] = p;
                break;
            }
        }
    }
    let maxdiag = diagpos
        .iter()
        .filter(|&&p| p != usize::MAX)
        .map(|&p| snap.nzval[p])
        .fold(T::zero(), |a, b| a.max(b.abs()));
    let mut eps = settings.static_regularization_constant
        + settings.static_regularization_proportional * maxdiag;
    // Mirror the production escalation: each level multiplies the shift
    // by 100 (see `DirectLDLKKTSolver::escalate_regularization`).
    if reg_boost > 0 {
        eps = eps * 100f64.powi(reg_boost as i32).as_T();
    }
    for (j, &p) in diagpos.iter().enumerate() {
        if p != usize::MAX {
            nzval[p] += eps * T::from_i8(snap.dsigns[j]).unwrap();
        }
    }
    nzval
}

/// Rebuild the captured CSC matrix, run `refactor` + `solve` through the
/// backend selected by `settings.direct_solve_method`, and audit the result
/// against the recorded solution. This exercises the same constructor the
/// production KKT solver uses.
pub fn replay<T: FloatT>(
    snap: &KktSnapshot<T>,
    settings: &crate::solver::CoreSettings<T>,
) -> Result<ReplayReport, String> {
    // The snapshot stores the authoritative unregularized KKT (production
    // restores it after refactoring). Reproduce `regularize_and_refactor`'s
    // static shift at the captured escalation level so the factorization sees
    // the same matrix.
    let nzval = regularized_nzval(snap, settings, snap.reg_boost);
    let kkt = CscMatrix::new(
        snap.n,
        snap.n,
        snap.colptr.iter().map(|&v| v as usize).collect(),
        snap.rowval.iter().map(|&v| v as usize).collect(),
        nzval,
    );
    let (_shape, ctor) = T::get_ldlsolver_config(settings);
    let mut solver = ctor(&kkt, &snap.dsigns, settings, None);
    let backend = solver.linear_solver_info().name.clone();

    let t0 = std::time::Instant::now();
    let refactored = solver.refactor(&kkt);
    let factor_secs = t0.elapsed().as_secs_f64();

    let mut x = vec![T::zero(); snap.n];
    let mut b = snap.rhs.clone();
    let t1 = std::time::Instant::now();
    let solved = refactored && {
        solver.solve(&kkt, &mut x, &mut b);
        x.is_finite()
    };
    let solve_secs = t1.elapsed().as_secs_f64();

    // Residual of the symmetric matrix from its stored triangle.
    let mut r = snap.rhs.clone();
    r.negate();
    for j in 0..snap.n {
        let xj = x[j];
        for p in snap.colptr[j] as usize..snap.colptr[j + 1] as usize {
            let i = snap.rowval[p] as usize;
            let v = snap.nzval[p];
            r[i] += v * xj;
            if i != j {
                r[j] += v * x[i];
            }
        }
    }
    let f64_or_nan = |v: &T| v.to_f64().unwrap_or(f64::NAN);
    let rnorm = r.iter().map(|v| f64_or_nan(v).abs()).collect::<Vec<_>>();
    let max_abs_residual = rnorm.iter().cloned().fold(0.0, f64::max);
    let knorm = snap
        .nzval
        .iter()
        .map(|v| f64_or_nan(v).abs())
        .fold(0.0, f64::max);
    let xnorm = x.iter().map(|v| f64_or_nan(v).abs()).fold(0.0, f64::max);
    let bnorm = snap
        .rhs
        .iter()
        .map(|v| f64_or_nan(v).abs())
        .fold(0.0, f64::max);
    let normwise_residual = max_abs_residual / (knorm * xnorm + bnorm);

    let bitwise_match = if snap.x.len() == snap.n {
        Some(
            x.iter()
                .zip(&snap.x)
                .all(|(a, b)| a.scalar_exact_encode() == b.scalar_exact_encode()),
        )
    } else {
        None
    };

    Ok(ReplayReport {
        backend,
        refactored,
        solved,
        normwise_residual,
        max_abs_residual,
        bitwise_match,
        factor_secs,
        solve_secs,
    })
}

const MAGIC_DENSE: &[u8; 8] = b"SDPXDEN1";

/// Serialize a dense column-major matrix (e.g. a cone's SVD input) with the
/// same exact scalar encoding, plus a manifest sidecar.
pub fn write_dense<T: FloatT>(
    dir: &Path,
    name: &str,
    rows: usize,
    cols: usize,
    data: &[T],
) -> io::Result<Option<PathBuf>> {
    let nlimbs = match data.first().and_then(|v| v.scalar_exact_encode()) {
        Some((_, _, l)) => l.len(),
        None => return Ok(None),
    };
    let mut buf = Vec::new();
    buf.write_all(MAGIC_DENSE)?;
    for v in [
        T::precision_bits() as u64,
        rows as u64,
        cols as u64,
        data.len() as u64,
        nlimbs as u64,
    ] {
        buf.write_all(&v.to_le_bytes())?;
    }
    for v in data {
        enc(&mut buf, v, nlimbs)?;
    }
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{name}.bin"));
    let manifest = format!(
        "{{\n  \"schema_version\": 1,\n  \"file\": {:?},\n  \"fnv1a64\": {:?},\n  \"precision_bits\": {},\n  \"shape\": {{\"rows\": {}, \"cols\": {}}},\n  \"layout\": \"column-major\"\n}}\n",
        path.file_name().unwrap().to_string_lossy(),
        fnv1a64(&buf),
        T::precision_bits(),
        rows,
        cols,
    );
    std::fs::write(&path, &buf)?;
    std::fs::write(dir.join(format!("{name}.json")), manifest)?;
    Ok(Some(path))
}

/// Load a dense matrix snapshot written by [`write_dense`].
pub fn load_dense<T: FloatT>(path: &Path) -> io::Result<(usize, usize, Vec<T>)> {
    let data = std::fs::read(path)?;
    let mut r = &data[..];
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC_DENSE {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }
    let u64s = |r: &mut &[u8]| -> io::Result<u64> {
        let mut b = [0u8; 8];
        r.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    };
    let prec = u64s(&mut r)?;
    if prec != T::precision_bits() as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "precision mismatch",
        ));
    }
    let (rows, cols, len, nlimbs) = (
        u64s(&mut r)? as usize,
        u64s(&mut r)? as usize,
        u64s(&mut r)? as usize,
        u64s(&mut r)? as usize,
    );
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(dec(&mut r, nlimbs)?);
    }
    Ok((rows, cols, out))
}

/// Read the recorded scalar precision from a snapshot header without loading
/// the payload, so a caller can dispatch on the concrete `T`.
pub fn header_precision(path: &Path) -> io::Result<u64> {
    let mut f = std::fs::File::open(path)?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }
    let mut b = [0u8; 8];
    f.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::core::kktsolvers::direct::ldlsolvers::qdldl::QDLDLDirectLDLSolver;
    use crate::solver::core::kktsolvers::direct::DirectLDLSolver;
    use crate::solver::CoreSettings;
    use sdpx_arithmetic::MpFloat;

    // Quasidefinite KKT: [diag(+2)  A^T; A  diag(-1)] with a coupling column.
    fn kkt<T: FloatT>() -> (CscMatrix<T>, Vec<i8>, Vec<T>) {
        // upper triangle, column-sorted
        let colptr = vec![0, 1, 2, 4, 5];
        let rowval = vec![0, 1, 0, 2, 3];
        let nzval = vec![
            T::from_f64(2.0).unwrap(),
            T::from_f64(2.0).unwrap(),
            T::from_f64(1.0).unwrap(),
            T::from_f64(-1.0).unwrap(),
            T::from_f64(-1.0).unwrap(),
        ];
        let m = CscMatrix::new(4, 4, colptr, rowval, nzval);
        let dsigns = vec![1, 1, -1, -1];
        let rhs = vec![
            T::from_f64(3.0).unwrap(),
            T::from_f64(-1.0).unwrap(),
            T::from_f64(0.5).unwrap(),
            T::from_f64(2.0).unwrap(),
        ];
        (m, dsigns, rhs)
    }

    fn run<T: FloatT>(tag: &str) {
        let (m, dsigns, rhs) = kkt::<T>();
        let settings = CoreSettings::<T>::default();
        // Record a real solution: production factors the statically
        // regularized matrix, then restores the stored KKT.
        let snap_view = KktSnapshot {
            precision_bits: T::precision_bits() as u32,
            n: 4,
            colptr: m.colptr.iter().map(|&v| v as u64).collect(),
            rowval: m.rowval.iter().map(|&v| v as u64).collect(),
            nzval: m.nzval.clone(),
            dsigns: dsigns.clone(),
            rhs: rhs.clone(),
            x: Vec::new(),
            solved: false,
            reg_boost: 0,
        };
        let reg = CscMatrix::new(
            4,
            4,
            m.colptr.clone(),
            m.rowval.clone(),
            regularized_nzval(&snap_view, &settings, 0),
        );
        let mut solver = QDLDLDirectLDLSolver::new(&reg, &dsigns, &settings, None);
        assert!(solver.refactor(&reg));
        let mut x = vec![T::zero(); 4];
        let mut b = rhs.clone();
        solver.solve(&reg, &mut x, &mut b);

        let dir = std::env::temp_dir().join(format!("sdpx-snap-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = write(&dir, 7, &m, &dsigns, &rhs, &x, true, "qdldl", 0)
            .unwrap()
            .expect("exact encoding available");
        let snap = load::<T>(&path).unwrap();
        assert_eq!(snap.n, 4);
        assert_eq!(
            snap.nzval
                .iter()
                .map(|v| v.scalar_exact_encode())
                .collect::<Vec<_>>(),
            m.nzval
                .iter()
                .map(|v| v.scalar_exact_encode())
                .collect::<Vec<_>>()
        );

        let rep = replay(&snap, &settings).unwrap();
        assert!(rep.refactored && rep.solved);
        assert_eq!(rep.bitwise_match, Some(true));
        // Residual vs the unregularized KKT is bounded by the
        // static regularization shift, not machine epsilon.
        assert!(rep.normwise_residual.is_finite() && rep.normwise_residual < 1e-4);

        let dpath = write_dense(&dir, "dense-0", 2, 2, &rhs).unwrap().unwrap();
        let (r2, c2, data2) = load_dense::<T>(&dpath).unwrap();
        assert_eq!((r2, c2), (2, 2));
        assert_eq!(
            data2
                .iter()
                .map(|v| v.scalar_exact_encode())
                .collect::<Vec<_>>(),
            rhs.iter()
                .map(|v| v.scalar_exact_encode())
                .collect::<Vec<_>>()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn snapshot_roundtrip_and_replay() {
        run::<f64>("f64");
        run::<MpFloat<4>>("mpfr256");
    }
}
