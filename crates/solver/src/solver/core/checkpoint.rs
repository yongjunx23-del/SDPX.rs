//! Iterate checkpoint/restart for the replicated solver.
//!
//! A file holds an accepted iterate (x, s, z, τ, κ) in internal coordinates
//! together with the writer's equilibration (d, e, c), in the pointer-free
//! scalar wire format. It is identified by two fingerprints of the internal
//! problem: `structure` (dimensions, sparsity patterns, cone list, sampled
//! block shapes) must match to load at all; `values` (all coefficients)
//! decides between an exact continuation and a hot start from a nearby
//! problem, which maps the iterate through original coordinates into the new
//! problem's scaling.
use crate::algebra::FloatT;
use std::io::{self, Error, ErrorKind};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"SDPXCKP2";
const HEADER: usize = 8 + 8 + 8 + 8 + 4 + 8 + 8;

/// Checkpoint and restart requests attached to a solver.
#[derive(Debug, Clone, Default)]
pub(crate) struct CheckpointConfig {
    /// Write the accepted iterate here every `every` iterations.
    pub path: Option<PathBuf>,
    pub every: u32,
    /// Start from this file instead of the default starting point.
    pub restart: Option<PathBuf>,
}

/// Fingerprints of the internal problem a checkpoint belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Identity {
    pub structure: u64,
    pub values: u64,
}

/// FNV-1a over integers and scalar wire encodings.
pub(crate) struct Fnv(u64);
impl Fnv {
    pub fn new() -> Self {
        Self(0xcbf29ce484222325)
    }
    pub fn word(&mut self, w: u64) {
        for byte in w.to_le_bytes() {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    pub fn words(&mut self, ws: &[usize]) {
        self.word(ws.len() as u64);
        for &w in ws {
            self.word(w as u64);
        }
    }
    pub fn scalars<T: FloatT>(&mut self, vs: &[T]) {
        self.word(vs.len() as u64);
        let mut buffer = vec![0u8; T::wire_size().unwrap_or(0)];
        for &v in vs {
            if v.write_wire(&mut buffer) {
                for &byte in &buffer {
                    self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x100000001b3);
                }
            }
        }
    }
    pub fn finish(&self) -> u64 {
        self.0
    }
}

/// An iterate read from a checkpoint, with the writer's equilibration.
pub(crate) struct Loaded<T> {
    pub values: u64,
    pub tau: T,
    pub kappa: T,
    pub c: T,
    pub x: Vec<T>,
    pub s: Vec<T>,
    pub z: Vec<T>,
    pub d: Vec<T>,
    pub e: Vec<T>,
}

fn bad(message: &str) -> Error {
    Error::new(ErrorKind::InvalidData, message.to_string())
}

fn wire_size<T: FloatT>() -> io::Result<usize> {
    T::wire_size().ok_or_else(|| bad("scalar has no checkpoint encoding"))
}

/// Write atomically (temporary file, then rename). `vectors` are
/// `[x, s, z, d, e]`; `scalars` are `[τ, κ, c]`.
pub(crate) fn write<T: FloatT>(
    path: &Path,
    id: Identity,
    iter: u32,
    scalars: [T; 3],
    vectors: [&[T]; 5],
) -> io::Result<()> {
    let size = wire_size::<T>()?;
    let (n, m) = (vectors[0].len(), vectors[1].len());
    let count = 3 + vectors.iter().map(|v| v.len()).sum::<usize>();
    let mut out = Vec::with_capacity(HEADER + size * count);
    out.extend_from_slice(MAGIC);
    for w in [T::wire_tag(), id.structure, id.values] {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out.extend_from_slice(&iter.to_le_bytes());
    out.extend_from_slice(&(n as u64).to_le_bytes());
    out.extend_from_slice(&(m as u64).to_le_bytes());
    for &v in scalars.iter().chain(vectors.into_iter().flatten()) {
        let at = out.len();
        out.resize(at + size, 0);
        if !v.write_wire(&mut out[at..]) {
            return Err(bad("scalar has no checkpoint encoding"));
        }
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".part");
    std::fs::write(&tmp, out)?;
    std::fs::rename(&tmp, path)
}

/// Read a checkpoint written for a problem of the same structure.
pub(crate) fn read<T: FloatT>(
    path: &Path,
    structure: u64,
    n: usize,
    m: usize,
) -> io::Result<Loaded<T>> {
    let size = wire_size::<T>()?;
    let bytes = std::fs::read(path)?;
    if bytes.len() < HEADER || &bytes[..8] != MAGIC {
        return Err(bad("not an SDPX checkpoint"));
    }
    let u64_at = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
    if u64_at(8) != T::wire_tag() {
        return Err(bad("checkpoint precision differs from the solver's"));
    }
    if u64_at(16) != structure || u64_at(36) != n as u64 || u64_at(44) != m as u64 {
        return Err(bad(
            "checkpoint was written for a problem of different structure",
        ));
    }
    let count = 3 + 2 * n + 3 * m;
    if bytes.len() != HEADER + count * size {
        return Err(bad("checkpoint has the wrong length"));
    }
    let mut values = bytes[HEADER..]
        .chunks_exact(size)
        .map(|chunk| T::read_wire(chunk).ok_or_else(|| bad("corrupt checkpoint scalar")));
    let mut take = |len: usize| {
        (0..len)
            .map(|_| values.next().unwrap())
            .collect::<io::Result<Vec<T>>>()
    };
    let head = take(3)?;
    Ok(Loaded {
        values: u64_at(24),
        tau: head[0],
        kappa: head[1],
        c: head[2],
        x: take(n)?,
        s: take(m)?,
        z: take(m)?,
        d: take(n)?,
        e: take(m)?,
    })
}
