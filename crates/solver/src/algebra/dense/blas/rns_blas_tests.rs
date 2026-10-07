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
    // Longer than one 4096-row block.
    check_diag_congruence::<2, 8>(3, 5000, 20, 23);
}
