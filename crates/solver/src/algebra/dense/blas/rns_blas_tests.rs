use super::*;
use num_traits::FromPrimitive;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / (1u64 << 53) as f64
    }
}

// Full-mantissa values with signs, zeros and an exponent spread of ~2^±90.
fn random<const N: usize>(rng: &mut Lcg, len: usize, spread: i64) -> Vec<F<N>> {
    (0..len)
        .map(|_| {
            if rng.next() < 0.05 {
                return F::zero();
            }
            let num = F::<N>::from_f64(rng.next() * 2.0 - 1.0).unwrap();
            let den = F::<N>::from_f64(rng.next() + 0.5).unwrap();
            let shift = (rng.next() * (2 * spread + 1) as f64) as i64 - spread;
            (num / den).scale_pow2(shift)
        })
        .collect()
}

fn reference<const N: usize>(
    ta: u8,
    tb: u8,
    m: usize,
    n: usize,
    k: usize,
    a: &[F<N>],
    lda: usize,
    b: &[F<N>],
    ldb: usize,
) -> Vec<F<N>> {
    let at = |i: usize, p: usize| {
        if ta == b'N' {
            &a[i + p * lda]
        } else {
            &a[p + i * lda]
        }
    };
    let bt = |p: usize, j: usize| {
        if tb == b'N' {
            &b[p + j * ldb]
        } else {
            &b[j + p * ldb]
        }
    };
    let mut c = vec![F::zero(); m * n];
    for j in 0..n {
        for i in 0..m {
            c[i + j * m] = F::dot_fma((0..k).map(|p| (at(i, p), bt(p, j))));
        }
    }
    c
}

fn check<const N: usize>(m: usize, n: usize, k: usize, spread: i64, seed: u64) {
    let mut rng = Lcg(seed);
    for (ta, tb) in [(b'N', b'N'), (b'T', b'N'), (b'N', b'T'), (b'T', b'T')] {
        let (ar, ac) = if ta == b'N' { (m, k) } else { (k, m) };
        let (br, bc) = if tb == b'N' { (k, n) } else { (n, k) };
        let a = random::<N>(&mut rng, ar * ac, spread);
        let b = random::<N>(&mut rng, br * bc, spread);
        let expect = reference(ta, tb, m, n, k, &a, ar, &b, br);
        let mut got = vec![F::zero(); m * n];
        assert!(gemm(
            ta, tb, m, n, k, &a, ar, &b, br, false, None, &mut got, None
        ));
        // A cached constant operand gives the same bits, on a miss and on a hit.
        let mut cache = ResidueCache::default();
        for _ in 0..2 {
            let mut cached = vec![F::zero(); m * n];
            assert!(gemm(
                ta,
                tb,
                m,
                n,
                k,
                &a,
                ar,
                &b,
                br,
                false,
                None,
                &mut cached,
                Some(&mut cache)
            ));
            assert!(cached
                .iter()
                .zip(&got)
                .all(|(x, y)| x == y || (x.is_zero() && y.is_zero())));
        }
        for (x, y) in got.iter().zip(&expect) {
            assert!(
                x == y || (x.is_zero() && y.is_zero()),
                "{} {}: {x:?} != {y:?}",
                ta as char,
                tb as char
            );
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        let mut upper = vec![F::zero(); m * n];
        assert!(gemm(
            ta,
            tb,
            m,
            n,
            k,
            &a,
            ar,
            &b,
            br,
            true,
            Some(&pool),
            &mut upper,
            None
        ));
        for j in 0..n {
            for i in 0..=j.min(m - 1) {
                assert!(upper[i + j * m] == expect[i + j * m] || expect[i + j * m].is_zero());
            }
        }
    }
}

#[test]
fn residue_blas_gemm_matches_exact_dots_bitwise() {
    check::<12>(7, 5, 45, 90, 1);
    check::<12>(20, 20, 20, 3, 2);
    check::<8>(3, 4, 1, 40, 3);
    check::<4>(9, 6, 300, 60, 4);
    check::<2>(5, 5, 33, 10, 5);
    check::<16>(35, 37, 45, 90, 7); // packed upper tiles across the 32-column boundary
}

/// Right operands with zero inner runs (triangular, banded, whole zero
/// tiles): the inner reach skips only exact zeros, and pooled column panels
/// of one-prime groups give the serial bits.
#[test]
fn residue_blas_gemm_skips_zero_inner_terms() {
    fn shaped<const N: usize>(rng: &mut Lcg, tb: u8, n: usize, k: usize, band: usize) -> Vec<F<N>> {
        let br = if tb == b'N' { k } else { n };
        let mut b = random::<N>(rng, k * n, 20);
        for j in 0..n {
            for p in 0..k {
                if p < j || p > j + band || (32..64).contains(&j) {
                    b[if tb == b'N' { p + j * br } else { j + p * br }] = F::zero();
                }
            }
        }
        b
    }
    fn run<const N: usize>(m: usize, n: usize, k: usize, seed: u64) {
        let mut rng = Lcg(seed);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        for (ta, tb) in [(b'N', b'N'), (b'T', b'N'), (b'N', b'T'), (b'T', b'T')] {
            let ar = if ta == b'N' { m } else { k };
            let br = if tb == b'N' { k } else { n };
            let a = random::<N>(&mut rng, m * k, 20);
            let b = shaped::<N>(&mut rng, tb, n, k, k / 3);
            let expect = reference(ta, tb, m, n, k, &a, ar, &b, br);
            for upper in [false, true] {
                for pool in [None, Some(&pool)] {
                    let mut got = vec![F::zero(); m * n];
                    assert!(gemm(
                        ta, tb, m, n, k, &a, ar, &b, br, upper, pool, &mut got, None
                    ));
                    for j in 0..n {
                        for i in 0..if upper { (j + 1).min(m) } else { m } {
                            let (x, y) = (&got[i + j * m], &expect[i + j * m]);
                            assert!(x == y || (x.is_zero() && y.is_zero()), "{i} {j}");
                        }
                    }
                }
            }
        }
    }
    run::<4>(37, 100, 90, 11);
    run::<2>(20, 70, 130, 12);
    // Operands above the residue-scratch cap: one-prime groups, so the
    // pool splits each prime's columns.
    let (n, mut rng) = (520, Lcg(13));
    let a = random::<2>(&mut rng, n * n, 20);
    let b = shaped::<2>(&mut rng, b'N', n, n, n);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let product = |upper: bool, pool: Option<&rayon::ThreadPool>| {
        let mut c = vec![F::zero(); n * n];
        assert!(gemm(
            b'T', b'N', n, n, n, &a, n, &b, n, upper, pool, &mut c, None
        ));
        c
    };
    for upper in [false, true] {
        let (serial, split) = (product(upper, None), product(upper, Some(&pool)));
        assert!(serial
            .iter()
            .zip(&split)
            .all(|(x, y)| x == y || (x.is_zero() && y.is_zero())));
        for (i, j) in (0..64).map(|t| ((t * 97) % n, (t * 131 + 7) % n)) {
            let i = if upper { i.min(j) } else { i };
            let at = |p: usize| &a[p + i * n];
            let y = F::dot_fma((0..n).map(|p| (at(p), &b[p + j * n])));
            let x = &serial[i + j * n];
            assert!(*x == y || (x.is_zero() && y.is_zero()), "{i} {j}");
        }
    }
}

#[test]
fn residue_blas_gemm_chunks_long_inner_dimension() {
    // A long inner dimension selects narrower primes.
    check::<4>(2, 3, 70_000, 4, 6);
}

#[test]
#[ignore]
fn residue_blas_gemm_timing() {
    fn run<const N: usize>(n: usize) {
        let mut rng = Lcg(9);
        let a = random::<N>(&mut rng, n * n, 60);
        let b = random::<N>(&mut rng, n * n, 60);
        let reps = 20;
        let t = std::time::Instant::now();
        for _ in 0..reps {
            std::hint::black_box(reference(b'N', b'N', n, n, n, &a, n, &b, n));
        }
        let exact = t.elapsed().as_secs_f64() / reps as f64;
        let mut c = vec![F::zero(); n * n];
        let t = std::time::Instant::now();
        for _ in 0..reps {
            gemm(b'N', b'N', n, n, n, &a, n, &b, n, false, None, &mut c, None);
        }
        let blas = t.elapsed().as_secs_f64() / reps as f64;
        eprintln!(
            "bits={} n={n}: exactdot {:.2} ms, residue-blas {:.2} ms, x{:.1}",
            N * 64,
            exact * 1e3,
            blas * 1e3,
            exact / blas
        );
    }
    for n in [16, 32, 45, 90] {
        run::<4>(n);
        run::<8>(n);
        run::<12>(n);
    }
}

fn widen<const N: usize, const M: usize>(v: &F<N>) -> F<M> {
    let (kind, exponent, limbs) = v.exact_encode();
    let mut wide = [0u64; M];
    wide[M - N..].copy_from_slice(limbs);
    F::<M>::exact_decode(kind, exponent, wide)
}

fn narrow<const N: usize, const M: usize>(v: &F<M>) -> F<N> {
    if v.is_zero() {
        return F::zero();
    }
    let (kind, exponent, limbs) = v.exact_encode();
    F::<N>::from_scaled_integer(kind < 0, limbs, exponent - (64 * M) as i64)
}

// Reference: both products at 4x precision are exact for these spreads, so
// one final rounding gives the correctly rounded congruence.
fn check_congruence<const N: usize, const M: usize>(m: usize, k: usize, spread: i64, seed: u64) {
    let mut rng = Lcg(seed);
    for ta in [b'N', b'T'] {
        let (ar, ac) = if ta == b'N' { (m, k) } else { (k, m) };
        let a = random::<N>(&mut rng, ar * ac, spread);
        let mut x = random::<N>(&mut rng, k * k, spread);
        for j in 0..k {
            for i in j + 1..k {
                x[i + j * k] = x[j + i * k];
            }
        }
        let aw: Vec<F<M>> = a.iter().map(widen::<N, M>).collect();
        let xw: Vec<F<M>> = x.iter().map(widen::<N, M>).collect();
        let tb = if ta == b'N' { b'T' } else { b'N' };
        let t = reference(b'N', tb, k, m, k, &xw, k, &aw, ar);
        let c = reference(ta, b'N', m, m, k, &aw, ar, &t, k);
        let mut got = vec![F::zero(); m * m];
        assert!(congruence(
            ta, m, k, &a, ar, &x, k, false, None, &mut got, None
        ));
        let mut cache = ResidueCache::default();
        for _ in 0..2 {
            let mut cached = vec![F::zero(); m * m];
            assert!(congruence(
                ta,
                m,
                k,
                &a,
                ar,
                &x,
                k,
                false,
                None,
                &mut cached,
                Some(&mut cache)
            ));
            assert!(cached
                .iter()
                .zip(&got)
                .all(|(p, q)| p == q || (p.is_zero() && q.is_zero())));
        }
        // Split calls (a pool, or ways inside a pool worker) share the primes
        // across workers and must give the serial bits.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        let mut pooled = vec![F::zero(); m * m];
        assert!(congruence(
            ta,
            m,
            k,
            &a,
            ar,
            &x,
            k,
            false,
            Some(&pool),
            &mut pooled,
            None
        ));
        assert!(pooled
            .iter()
            .zip(&got)
            .all(|(p, q)| p == q || (p.is_zero() && q.is_zero())));
        let mut ways = vec![F::zero(); m * m];
        pool.install(|| {
            with_split_hint(3, || {
                let mut cache = ResidueCache::default();
                assert!(congruence(
                    ta,
                    m,
                    k,
                    &a,
                    ar,
                    &x,
                    k,
                    false,
                    None,
                    &mut ways,
                    Some(&mut cache)
                ));
            })
        });
        assert!(ways
            .iter()
            .zip(&got)
            .all(|(p, q)| p == q || (p.is_zero() && q.is_zero())));
        // Upper-triangle outputs split by columns: identical upper entries,
        // and the strict lower triangle is left untouched.
        let mut upper = vec![F::zero(); m * m];
        assert!(congruence(
            ta, m, k, &a, ar, &x, k, true, None, &mut upper, None
        ));
        let mut upper_split = vec![<F<N> as num_traits::One>::one(); m * m];
        assert!(congruence(
            ta,
            m,
            k,
            &a,
            ar,
            &x,
            k,
            true,
            Some(&pool),
            &mut upper_split,
            None
        ));
        for j in 0..m {
            for i in 0..m {
                let (p, q) = (upper_split[i + j * m], upper[i + j * m]);
                if i <= j {
                    assert!(p == q || (p.is_zero() && q.is_zero()));
                } else {
                    assert!(p == <F<N> as num_traits::One>::one());
                }
            }
        }
        for (o, (g, e)) in got.iter().zip(&c).enumerate() {
            let e = narrow::<N, M>(e);
            assert!(
                *g == e || (g.is_zero() && e.is_zero()),
                "{} entry {o}: {g:?} != {e:?}",
                ta as char
            );
        }
    }
}

#[test]
fn residue_blas_congruence_is_exact_rounded_once() {
    check_congruence::<4, 16>(9, 45, 30, 11);
    check_congruence::<12, 48>(45, 45, 40, 12);
    check_congruence::<8, 32>(5, 60, 20, 13);
}

#[test]
fn residue_split_encode_matches_serial() {
    let mut rng = Lcg(31);
    let (rows, cols) = (50, 45);
    let data = random::<12>(&mut rng, rows * cols, 60);
    // Leading dimension above `rows`: padding entries must stay zero.
    let ld = rows + 3;
    let mut padded = vec![F::zero(); ld * cols];
    for j in 0..cols {
        padded[j * ld..j * ld + rows].clone_from_slice(&data[j * rows..(j + 1) * rows]);
    }
    let x = View {
        data: &padded,
        rows,
        cols,
        ld,
    };
    let (lo, hi) = x.exponent_range().unwrap();
    let plan = Plan::new(cols, 2.0 * 768.0 + (hi - lo) as f64 + 16.0).unwrap();
    let serial = chunk_matrix(x, lo, hi - lo, &plan, &Split::Serial);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for split in [Split::Pool(&pool), Split::Ways(3)] {
        let parallel = pool.install(|| chunk_matrix(x, lo, hi - lo, &plan, &split));
        assert_eq!(
            (parallel.chunks, parallel.width, parallel.len),
            (serial.chunks, serial.width, serial.len)
        );
        assert!(parallel
            .e
            .iter()
            .zip(&serial.e)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    for bits in [24, 25] {
        let plan = Plan::with_count(bits, plan.count());
        let chunks = chunk_matrix(x, lo, hi - lo, &plan, &Split::Serial);
        let serial = CachedResidues::from_chunks(&chunks, &plan, &Split::Serial);
        for split in [Split::Pool(&pool), Split::Ways(3)] {
            let parallel = pool.install(|| CachedResidues::from_chunks(&chunks, &plan, &split));
            match (&serial, &parallel) {
                (CachedResidues::Narrow(a, c), CachedResidues::Narrow(b, d)) => {
                    assert_eq!((a, c), (b, d))
                }
                (CachedResidues::Wide(a), CachedResidues::Wide(b)) => {
                    assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
                }
                _ => panic!("cache widths differ"),
            }
        }
    }
}

#[test]
fn residue_block_gemm_declines_rounded_product_overflow() {
    let y = vec![F::<2>::max_value(); 2];
    let d = vec![F::from_f64(2.0).unwrap()];
    assert!((y[0] * d[0]).is_infinite());
    let columns = [[0, 1], [1, 2]];
    let blocks = [
        (y.as_slice(), d.as_slice(), columns[0].as_slice()),
        (y.as_slice(), d.as_slice(), columns[1].as_slice()),
    ];
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let sentinel = F::from_f64(7.0).unwrap();
    for pool in [None, Some(&pool)] {
        let mut out = vec![sentinel; 9];
        assert!(!gemm_blocks_upper(3, &blocks, &mut out, pool));
        assert!(out.iter().all(|&value| value == sentinel));
    }
}

#[test]
#[ignore]
fn residue_blas_concurrency() {
    let mut rng = Lcg(21);
    let n = 45;
    let a = random::<12>(&mut rng, n * n, 60);
    let b = random::<12>(&mut rng, n * n, 60);
    for threads in [1usize, 8, 16, 32] {
        let reps = 40;
        let t = std::time::Instant::now();
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let mut c = vec![F::zero(); n * n];
                    for _ in 0..reps {
                        gemm(b'N', b'N', n, n, n, &a, n, &b, n, true, None, &mut c, None);
                    }
                });
            }
        });
        let per = t.elapsed().as_secs_f64() / reps as f64;
        eprintln!(
            "threads={threads}: wall per round {:.2} ms (ideal = single-thread time)",
            per * 1e3
        );
    }
}

fn check_svec_quadratic<const N: usize, const M: usize>(
    h: usize,
    kmax: usize,
    spread: i64,
    seed: u64,
) {
    let mut rng = Lcg(seed);
    let trih = h * (h + 1) / 2;
    let q = random::<N>(&mut rng, h * kmax, spread);
    let x = random::<N>(&mut rng, trih, spread);
    let sqrt2 = <F<N> as num_traits::FloatConst>::SQRT_2();
    let mut got = vec![F::zero(); kmax];
    assert!(svec_quadratic(
        h, kmax, &q, false, &x, sqrt2, None, &mut got, None
    ));
    let mut cache = ResidueCache::default();
    for _ in 0..2 {
        let mut cached = vec![F::zero(); kmax];
        assert!(svec_quadratic(
            h,
            kmax,
            &q,
            false,
            &x,
            sqrt2,
            None,
            &mut cached,
            Some(&mut cache)
        ));
        assert!(cached
            .iter()
            .zip(&got)
            .all(|(p, r)| p == r || (p.is_zero() && r.is_zero())));
    }
    // |q| through q's cached residues equals the form on an explicit |q|.
    let qa: Vec<F<N>> = q.iter().map(|&v| sdpx_arithmetic::Scalar::abs(v)).collect();
    let xa: Vec<F<N>> = x.iter().map(|&v| sdpx_arithmetic::Scalar::abs(v)).collect();
    let mut plain = vec![F::zero(); kmax];
    let mut folded = vec![F::zero(); kmax];
    assert!(svec_quadratic(
        h, kmax, &qa, false, &xa, sqrt2, None, &mut plain, None
    ));
    assert!(svec_quadratic(
        h,
        kmax,
        &q,
        true,
        &xa,
        sqrt2,
        None,
        &mut folded,
        Some(&mut cache)
    ));
    assert_eq!(plain, folded);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let mut pooled = vec![F::zero(); kmax];
    assert!(svec_quadratic(
        h,
        kmax,
        &q,
        false,
        &x,
        sqrt2,
        Some(&pool),
        &mut pooled,
        None
    ));
    assert!(pooled
        .iter()
        .zip(&got)
        .all(|(p, r)| p == r || (p.is_zero() && r.is_zero())));
    let mut ways = vec![F::zero(); kmax];
    pool.install(|| {
        with_split_hint(3, || {
            assert!(svec_quadratic(
                h, kmax, &q, false, &x, sqrt2, None, &mut ways, None
            ));
        })
    });
    assert!(ways
        .iter()
        .zip(&got)
        .all(|(p, r)| p == r || (p.is_zero() && r.is_zero())));
    let (qw, xw): (Vec<F<M>>, Vec<F<M>>) = (
        q.iter().map(widen::<N, M>).collect(),
        x.iter().map(widen::<N, M>).collect(),
    );
    let sw = widen::<N, M>(&sqrt2);
    for k in 0..kmax {
        let mut left = Vec::new();
        let mut right = Vec::new();
        for j in 0..h {
            for i in 0..=j {
                let c = if i == j {
                    <F<M> as num_traits::One>::one()
                } else {
                    sw
                };
                left.push(c * xw[j * (j + 1) / 2 + i]);
                right.push(qw[i + k * h] * qw[j + k * h]);
            }
        }
        let e = narrow::<N, M>(&F::<M>::dot_fma(left.iter().zip(&right)));
        assert!(
            got[k] == e || (got[k].is_zero() && e.is_zero()),
            "k={k}: {:?} != {e:?}",
            got[k]
        );
    }
}

#[test]
fn residue_svec_quadratic_is_exact_rounded_once() {
    check_svec_quadratic::<4, 24>(12, 20, 20, 31);
    check_svec_quadratic::<12, 72>(45, 90, 30, 32);
}

#[test]
#[ignore]
fn residue_kernels_concurrency() {
    let mut rng = Lcg(41);
    let (h, kmax) = (45, 90);
    let a = random::<12>(&mut rng, h * h, 60);
    let mut x = random::<12>(&mut rng, h * h, 60);
    for j in 0..h {
        for i in j + 1..h {
            x[i + j * h] = x[j + i * h];
        }
    }
    let q = random::<12>(&mut rng, h * kmax, 60);
    let xs = random::<12>(&mut rng, h * (h + 1) / 2, 60);
    let sqrt2 = <F<12> as num_traits::FloatConst>::SQRT_2();
    for (label, which) in [("congruence", 0), ("svec_quadratic", 1)] {
        for threads in [1usize, 8, 32] {
            let reps = 20;
            let t = std::time::Instant::now();
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        let mut c = vec![F::zero(); h * h.max(kmax)];
                        for _ in 0..reps {
                            if which == 0 {
                                congruence(b'N', h, h, &a, h, &x, h, true, None, &mut c, None);
                            } else {
                                svec_quadratic(h, kmax, &q, false, &xs, sqrt2, None, &mut c, None);
                            }
                        }
                    });
                }
            });
            eprintln!(
                "{label} threads={threads}: {:.2} ms per call",
                t.elapsed().as_secs_f64() / reps as f64 * 1e3
            );
        }
    }
}

// Reference for Aᵀ·diag(d)·A: d_r·a_rj is exact at 4x precision, and the
// outer sum of exact products rounds once there; narrowing then rounds once.
fn check_diag_congruence<const N: usize, const M: usize>(
    m: usize,
    k: usize,
    spread: i64,
    seed: u64,
) {
    let mut rng = Lcg(seed);
    let a = random::<N>(&mut rng, k * m, spread);
    // A wide scaling spread, as near interior-point convergence.
    let d = random::<N>(&mut rng, k, 3 * spread);
    let aw: Vec<F<M>> = a.iter().map(widen::<N, M>).collect();
    let dw: Vec<F<M>> = d.iter().map(widen::<N, M>).collect();
    let mut t = vec![F::<M>::zero(); k * m];
    for j in 0..m {
        for r in 0..k {
            t[r + j * k] = aw[r + j * k] * dw[r];
        }
    }
    let c = reference(b'T', b'N', m, m, k, &aw, k, &t, k);
    let mut cache = ResidueCache::default();
    let mut got = vec![F::zero(); m * m];
    assert!(diag_congruence(
        m, k, &a, &d, false, None, &mut got, &mut cache
    ));
    for (o, (g, e)) in got.iter().zip(&c).enumerate() {
        let e = narrow::<N, M>(e);
        assert!(
            *g == e || (g.is_zero() && e.is_zero()),
            "entry {o}: {g:?} != {e:?}"
        );
    }
    let same = |x: &[F<N>], upper: bool| {
        (0..m).all(|j| {
            (0..if upper { j + 1 } else { m }).all(|i| {
                let (p, q) = (&x[i + j * m], &got[i + j * m]);
                p == q || (p.is_zero() && q.is_zero())
            })
        })
    };
    // Cached residues, a pool and upper-only output give the same bits.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    for (upper, p) in [(false, None), (true, None), (true, Some(&pool))] {
        let mut again = vec![F::zero(); m * m];
        assert!(diag_congruence(
            m, k, &a, &d, upper, p, &mut again, &mut cache
        ));
        assert!(same(&again, upper));
    }
}

#[test]
fn residue_diag_congruence_is_exact_rounded_once() {
    check_diag_congruence::<2, 8>(5, 200, 30, 21);
    check_diag_congruence::<4, 16>(7, 300, 40, 22);
    // Several row blocks.
    check_diag_congruence::<2, 8>(3, 5000, 20, 23);
    // Primes sized by the block are wider than all 9000 rows allow.
    check_diag_congruence::<2, 8>(3, 9000, 20, 24);
}

#[test]
fn prime_group_congruence_matches_unsplit() {
    fn run<const N: usize>(m: usize, seed: u64) {
        let mut rng = Lcg(seed);
        let a = random::<N>(&mut rng, m * m, 40);
        let mut x = random::<N>(&mut rng, m * m, 40);
        for j in 0..m {
            for i in j + 1..m {
                x[i + j * m] = x[j + i * m];
            }
        }
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for upper_only in [false, true] {
            let mut serial = vec![F::<N>::zero(); m * m];
            assert!(congruence(b'N', m, m, &a, m, &x, m, upper_only, None, &mut serial, None));
            let av = View { data: &a[..], rows: m, cols: m, ld: m };
            let xv = View { data: &x[..], rows: m, cols: m, ld: m };
            let (lo_a, hi_a) = av.exponent_range().unwrap();
            let (lo_x, hi_x) = xv.exponent_range().unwrap();
            let (da, dx) = (hi_a - lo_a, hi_x - lo_x);
            let needed = 3.0 * F::<N>::PRECISION_BITS as f64
                + (2 * da + dx) as f64
                + 2.0 * (m as f64).log2()
                + 3.0;
            let plan = Plan::new(m, needed).unwrap();
            for ways in 1..=5 {
                for columns in [false, true] {
                    let mut got = vec![<F<N> as num_traits::One>::one(); m * m];
                    pool.install(|| {
                        let split = Split::Ways(ways);
                        let ca = Operand::Chunks(chunk_matrix(av, lo_a, da, &plan, &split));
                        let cx = chunk_matrix(xv, lo_x, dx, &plan, &split);
                        let shape = CongruenceShape {
                            ta: b'N',
                            ta_n: true,
                            m,
                            k: m,
                            lda: m,
                            ldx: m,
                            len_a: m * m,
                            len_x: m * m,
                            upper_only,
                        };
                        let solve = if columns {
                            congruence_columns::<N>
                        } else {
                            congruence_prime_groups::<N>
                        };
                        assert!(solve(
                            shape, &plan, &ca, &cx, &split, ways, 2 * lo_a + lo_x, &mut got
                        ));
                    });
                    for j in 0..m {
                        for i in 0..m {
                            let o = i + j * m;
                            if upper_only && i > j {
                                assert_eq!(got[o], <F<N> as num_traits::One>::one());
                            } else {
                                assert!(got[o] == serial[o] || (got[o].is_zero() && serial[o].is_zero()));
                            }
                        }
                    }
                }
            }
        }
    }
    run::<4>(9, 71);
    run::<12>(53, 72);
}

/// Cost of granting ways to one congruence when nobody steals them (a single
/// worker runs every way): the per-way overhead `planned_ways` must model.
/// `cargo test --release … congruence_way_overhead -- --ignored --nocapture`
#[test]
#[ignore]
fn congruence_way_overhead() {
    fn bench<const N: usize>(n: usize) {
        let full = |i: usize, j: usize| {
            F::<N>::from_usize(i + 1).unwrap() / F::<N>::from_usize(j + 7).unwrap()
                - F::<N>::from_f64(0.5).unwrap()
        };
        let a: Vec<F<N>> = (0..n * n).map(|t| full(t % n, t / n + 3)).collect();
        let x: Vec<F<N>> = (0..n * n)
            .map(|t| {
                let (i, j) = (t % n, t / n);
                full(i.min(j), i.max(j))
            })
            .collect();
        let mut out = vec![F::<N>::zero(); n * n];
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let mut base = 0.0;
        for ways in [1usize, 2, 3, 4, 8] {
            let reps = 20;
            let secs = pool.install(|| {
                with_split_hint(ways, || {
                    assert!(congruence::<N>(b'N', n, n, &a, n, &x, n, true, None, &mut out, None));
                    let t = std::time::Instant::now();
                    for _ in 0..reps {
                        congruence::<N>(b'N', n, n, &a, n, &x, n, true, None, &mut out, None);
                    }
                    t.elapsed().as_secs_f64() / reps as f64
                })
            });
            if ways == 1 {
                base = secs;
            }
            println!(
                "bits {} n {n} ways {ways}: {:.2} ms ({:+.1}% vs 1 way)",
                N * 64,
                secs * 1e3,
                100.0 * (secs / base - 1.0)
            );
        }
    }
    for n in [24, 48, 57, 96, 160] {
        bench::<16>(n);
    }
    bench::<8>(57);
    bench::<12>(57);
}

/// Wall time of one split congruence on a pool of `ways` threads (speedup,
/// not overhead): `SDPX_BENCH_THREADS` lists the thread counts.
#[test]
#[ignore]
fn congruence_way_speedup() {
    fn bench<const N: usize>(n: usize, threads: &[usize]) {
        let full = |i: usize, j: usize| {
            F::<N>::from_usize(i + 1).unwrap() / F::<N>::from_usize(j + 7).unwrap()
                - F::<N>::from_f64(0.5).unwrap()
        };
        let a: Vec<F<N>> = (0..n * n).map(|t| full(t % n, t / n + 3)).collect();
        let x: Vec<F<N>> = (0..n * n)
            .map(|t| {
                let (i, j) = (t % n, t / n);
                full(i.min(j), i.max(j))
            })
            .collect();
        let mut out = vec![F::<N>::zero(); n * n];
        let mut base = 0.0;
        for &ways in threads {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(ways).build().unwrap();
            let reps = 20;
            let secs = pool.install(|| {
                with_split_hint(ways, || {
                    assert!(congruence::<N>(b'N', n, n, &a, n, &x, n, true, None, &mut out, None));
                    let t = std::time::Instant::now();
                    for _ in 0..reps {
                        congruence::<N>(b'N', n, n, &a, n, &x, n, true, None, &mut out, None);
                    }
                    t.elapsed().as_secs_f64() / reps as f64
                })
            });
            if base == 0.0 {
                base = secs;
            }
            println!(
                "speedup bits {} n {n} threads {ways}: {:.2} ms (x{:.2})",
                N * 64,
                secs * 1e3,
                base / secs
            );
        }
    }
    let threads: Vec<usize> = std::env::var("SDPX_BENCH_THREADS")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4]);
    for n in [48, 63, 96] {
        bench::<16>(n, &threads);
    }
}

/// The output-tiled CRT reconstructs every output exactly as the streamed
/// accumulator does: full and partial tiles, contiguous and upper selections.
#[test]
fn crt_tiled_matches_streamed() {
    let mut rng = Lcg(0x5eed);
    for (k, needed, m, upper) in [(57, 3300.0, 57, true), (8, 700.0, 13, false), (300, 5200.0, 9, true)] {
        let plan = Plan::new(k, needed).unwrap();
        let (count, outputs) = (plan.count(), m * m);
        let prod: Vec<f64> = (0..count * outputs)
            .map(|t| {
                let p = plan.primes[t / outputs];
                symmetric((rng.next() * p as f64) as u64 % p, p)
            })
            .collect();
        let selected = selection(m, m, upper);
        let n = if upper { selected.len() } else { outputs };
        let mut streamed = vec![F::<16>::zero(); outputs];
        let mut acc = CrtAccumulator::new(&plan, n, &selected);
        for q0 in (0..count).step_by(STREAM_GROUP) {
            let q1 = (q0 + STREAM_GROUP).min(count);
            let mut group = prod[q0 * outputs..q1 * outputs].to_vec();
            acc.add(&plan, q0, q1, &mut group, outputs, &Split::Serial);
        }
        assert!(acc.finish(&plan, -37, &mut streamed, &Split::Serial));
        let mut tiled = vec![F::<16>::zero(); outputs];
        let total = if upper { selected.len() } else { outputs };
        let cells: Vec<std::sync::Mutex<F<16>>> = (0..outputs).map(|_| Default::default()).collect();
        let write = |i: usize, v: F<16>| {
            *cells[if upper { selected[i] } else { i }].lock().unwrap() = v;
        };
        assert!(crt_tiled(&plan, &prod, outputs, &selected, 0..total, -37, &write));
        for (dst, cell) in tiled.iter_mut().zip(cells) {
            *dst = cell.into_inner().unwrap();
        }
        assert!(streamed.iter().zip(&tiled).all(|(a, b)| a.exact_encode() == b.exact_encode()));
        assert!(tiled.iter().any(|v| !v.is_zero()));
    }
}

/// Cost split of one tiled CRT: residue products, digit `dgemm` and
/// pack/round, for a congruence-sized output set.
/// `cargo test --release … crt_tiled_timing -- --ignored --nocapture`
#[test]
#[ignore]
fn crt_tiled_timing() {
    let mut rng = Lcg(7);
    for (k, needed, m) in [(57usize, 3100.0, 57usize), (96, 3120.0, 96)] {
        let plan = Plan::new(k, needed).unwrap();
        let (count, outputs) = (plan.count(), m * m);
        let prod: Vec<f64> = (0..count * outputs)
            .map(|t| {
                let p = plan.primes[t / outputs];
                symmetric((rng.next() * p as f64) as u64 % p, p)
            })
            .collect();
        let selected = selection(m, m, true);
        let crt = crt_constants(&plan);
        let reps = 20;
        let sink = std::sync::Mutex::new(0usize);
        let write = |_: usize, v: F<16>| {
            if v.is_zero() {
                *sink.lock().unwrap() += 1;
            }
        };
        let t = std::time::Instant::now();
        for _ in 0..reps {
            assert!(crt_tiled(&plan, &prod, outputs, &selected, 0..selected.len(), 0, &write));
        }
        let total = t.elapsed().as_secs_f64() / reps as f64;
        // The digit dgemm alone, at the same tile shape.
        let (kp, rows, chunks) = (count, count + 1, crt.chunks);
        let local = vec![1.0f64; CRT_TILE * rows];
        let mut y = vec![0.0f64; CRT_TILE * chunks];
        let (t_, table) = (CRT_TILE, &crt.table);
        let t = std::time::Instant::now();
        for _ in 0..reps {
            for _ in (0..selected.len()).step_by(CRT_TILE) {
                int_gemm(b'T', b'T', chunks, t_, rows, table, rows, &local, t_, &mut y, chunks);
            }
        }
        let dgemm = t.elapsed().as_secs_f64() / reps as f64;
        // The residue gather and scaling alone, same tiles.
        let mut local = vec![0.0f64; CRT_TILE * kp];
        let mut frac = vec![0.0f64; CRT_TILE];
        let t = std::time::Instant::now();
        for _ in 0..reps {
            for t0 in (0..selected.len()).step_by(CRT_TILE) {
                let tt = (selected.len() - t0).min(CRT_TILE);
                frac[..tt].fill(0.0);
                for q in 0..kp {
                    let (p, pinv, u) = (plan.p[q], plan.pinv[q], crt.u[q]);
                    let row = &prod[q * outputs..(q + 1) * outputs];
                    let dst = &mut local[q * tt..(q + 1) * tt];
                    for (d, &r) in dst.iter_mut().zip(&selected[t0..t0 + tt]) {
                        *d = row[r];
                    }
                    for (d, f) in dst.iter_mut().zip(&mut frac[..tt]) {
                        let v = residue_product(*d, u, p, pinv);
                        *d = v;
                        *f += v * pinv;
                    }
                }
            }
        }
        let residues = t.elapsed().as_secs_f64() / reps as f64;
        println!("  residues {:.2} ms ({} sink)", residues * 1e3, frac[0] + local[7]);
        // pack and rounding alone, on realistic digit sums.
        let mut digits: Vec<Vec<f64>> = Vec::new();
        for i in 0..selected.len().min(256) {
            let mut acc = vec![0.0f64; chunks];
            for (q, &r) in prod.iter().skip(selected[i]).step_by(outputs).enumerate() {
                let v = residue_product(r, crt.u[q], plan.p[q], plan.pinv[q]);
                for (l, a) in acc.iter_mut().enumerate() {
                    *a += v * crt.table[q + l * rows];
                }
            }
            digits.push(acc);
        }
        let mut mag = Vec::new();
        let (mut packs, mut rounds) = (0.0, 0.0);
        let mut nonzero = 0usize;
        for _ in 0..reps {
            for d in &digits {
                let t = std::time::Instant::now();
                let neg = pack(d.iter().copied(), crt.width, chunks, &mut mag).unwrap();
                packs += t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                let v = F::<16>::from_scaled_integer(neg, &mag, 0);
                rounds += t.elapsed().as_secs_f64();
                nonzero += (!v.is_zero()) as usize;
            }
        }
        let per = (reps * digits.len()) as f64;
        println!(
            "  per output: pack {:.0} ns, round {:.0} ns ({} limbs, {nonzero} nonzero)",
            packs / per * 1e9,
            rounds / per * 1e9,
            mag.len()
        );
        println!(
            "m {m} primes {count} chunks {chunks} outputs {}: crt {:.2} ms, digit dgemm {:.2} ms",
            selected.len(),
            total * 1e3,
            dgemm * 1e3
        );
    }
}

#[test]
#[ignore]
fn leaf_contribution_timing() {
    // Arrow leaf contribution YᵀZ (upper), Λ35-rs shapes at 1024 bits.
    fn run<const N: usize>(q: usize, g: usize) {
        let mut rng = Lcg(5);
        let y = random::<N>(&mut rng, g * q, 30);
        let z = random::<N>(&mut rng, g * q, 30);
        let reps = 5;
        let t = std::time::Instant::now();
        let mut out = vec![F::<N>::zero(); q * (q + 1) / 2];
        for _ in 0..reps {
            for c in 0..q {
                for r in 0..=c {
                    out[r + c * (c + 1) / 2] = F::dot_fma(
                        y[r * g..(r + 1) * g].iter().zip(&z[c * g..(c + 1) * g]),
                    );
                }
            }
        }
        let exact = t.elapsed().as_secs_f64() / reps as f64;
        let mut c = vec![F::zero(); q * (q + 1) / 2];
        let t = std::time::Instant::now();
        for _ in 0..reps {
            assert!(gemm(b'T', b'N', q, q, g, &y, g, &z, g, true, None, &mut c, None));
        }
        let blas = t.elapsed().as_secs_f64() / reps as f64;
        assert_eq!(c, out);
        eprintln!(
            "bits={} q={q} g={g}: exactdot {:.1} ms, residue {:.1} ms, x{:.1}",
            N * 64,
            exact * 1e3,
            blas * 1e3,
            exact / blas
        );
    }
    run::<16>(170, 114);
    run::<16>(170, 96);
    run::<12>(170, 114);
}
