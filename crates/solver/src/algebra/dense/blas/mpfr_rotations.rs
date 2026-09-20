//! Deferred Givens-rotation log for the bidiagonal MPFR SVD.
//!
//! `demmel_kahan`/`shifted_qr` sweeps record (column, c, s) pairs instead of
//! writing U/V immediately; a flush replays them in order. Serial replay runs
//! the identical `rotate` body per entry, so results are bitwise identical to
//! immediate application. Tiled replay partitions rows into disjoint tiles
//! built with `chunks_mut` (no raw pointers), and each tile replays the whole
//! log serially — the same per-element arithmetic sequence, also bitwise
//! identical. The mode is an internal experiment control; `immediate` is the
//! default and reproduces the pre-log behavior exactly.

use super::*;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RotMode {
    Immediate,
    Serial,
    Tiled,
}

fn env_mode() -> RotMode {
    static MODE: OnceLock<RotMode> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("SDPX_SVD_ROT").ok().as_deref() {
        Some("serial") => RotMode::Serial,
        Some("tiled") => RotMode::Tiled,
        _ => RotMode::Immediate,
    })
}

// One recorded rotation: columns p and p+1, coefficients as passed to
// `svd_rotate` (replays apply the same -s flip).
struct Rot<const N: usize> {
    p: u32,
    c: F<N>,
    s: F<N>,
}

// Bounded log; ~2048 entries per side costs ~0.5 MiB at 768-bit precision.
const LOG_CAP: usize = 2048;
// Below this count a flush stays serial: tile construction would dominate.
const MIN_TILED: usize = 16;

pub(super) struct RotLog<const N: usize> {
    u: Vec<Rot<N>>,
    v: Vec<Rot<N>>,
    mode: RotMode,
}

impl<const N: usize> RotLog<N> {
    pub(super) fn new() -> Self {
        Self::with_mode(env_mode())
    }
    pub(super) fn with_mode(mode: RotMode) -> Self {
        Self {
            u: Vec::new(),
            v: Vec::new(),
            mode,
        }
    }

    pub(super) fn rotate_u(&mut self, u: &mut [F<N>], rows: usize, p: usize, c: F<N>, s: F<N>) {
        if u.is_empty() {
            return;
        }
        if self.mode == RotMode::Immediate {
            svd_rotate(u, rows, p, c, s);
            return;
        }
        self.u.push(Rot {
            p: p as u32,
            c,
            s,
        });
        if self.u.len() >= LOG_CAP {
            self.flush_u(u, rows);
        }
    }

    pub(super) fn rotate_v(&mut self, v: &mut [F<N>], rows: usize, p: usize, c: F<N>, s: F<N>) {
        if v.is_empty() {
            return;
        }
        if self.mode == RotMode::Immediate {
            svd_rotate(v, rows, p, c, s);
            return;
        }
        self.v.push(Rot {
            p: p as u32,
            c,
            s,
        });
        if self.v.len() >= LOG_CAP {
            self.flush_v(v, rows);
        }
    }

    pub(super) fn flush(&mut self, u: &mut [F<N>], m: usize, v: &mut [F<N>], n: usize) {
        self.flush_u(u, m);
        self.flush_v(v, n);
    }

    fn flush_u(&mut self, u: &mut [F<N>], rows: usize) {
        replay(self.mode, u, rows, std::mem::take(&mut self.u));
    }

    fn flush_v(&mut self, v: &mut [F<N>], rows: usize) {
        replay(self.mode, v, rows, std::mem::take(&mut self.v));
    }
}

fn replay<const N: usize>(mode: RotMode, a: &mut [F<N>], rows: usize, log: Vec<Rot<N>>) {
    if log.is_empty() {
        return;
    }
    match mode {
        RotMode::Serial => serial(a, rows, &log),
        RotMode::Tiled => tiled(a, rows, &log),
        RotMode::Immediate => unreachable!(),
    }
}

fn serial<const N: usize>(a: &mut [F<N>], rows: usize, log: &[Rot<N>]) {
    for r in log {
        svd_rotate(a, rows, r.p as usize, r.c, r.s);
    }
}

// Row-tiled replay: disjoint row ranges replay the full log serially, so the
// arithmetic applied to each element is the same sequence as `serial`.
fn tiled<const N: usize>(a: &mut [F<N>], rows: usize, log: &[Rot<N>]) {
    let workers = rayon::current_num_threads().max(1);
    let ntiles = (2 * workers).min(rows);
    if !inner_par() || ntiles < 2 || log.len() < MIN_TILED {
        serial(a, rows, log);
        return;
    }
    let w = rows.div_ceil(ntiles);
    let cols = a.len() / rows;
    // ceil(rows/w) tiles actually emerge, which can be fewer than `ntiles`.
    let mut tiles: Vec<Vec<&mut [F<N>]>> = Vec::new();
    for col in a.chunks_mut(rows) {
        for (t, seg) in col.chunks_mut(w).enumerate() {
            if tiles.len() <= t {
                tiles.push(Vec::with_capacity(cols));
            }
            tiles[t].push(seg);
        }
    }
    if tiles.len() < 2 {
        // Release the borrowed column segments before re-borrowing `a`.
        drop(tiles);
        serial(a, rows, log);
        return;
    }
    tiles.par_iter_mut().for_each(|tile| {
        for r in log {
            let p = r.p as usize;
            let (left, right) = tile.split_at_mut(p + 1);
            let (x, y) = (&mut *left[p], &mut *right[0]);
            let c = r.c;
            let ns = -r.s;
            for i in 0..x.len() {
                let xi = x[i];
                let yi = y[i];
                x[i] = r.s.mul_add(yi, c * xi);
                y[i] = ns.mul_add(xi, c * yi);
            }
        }
    });
}
