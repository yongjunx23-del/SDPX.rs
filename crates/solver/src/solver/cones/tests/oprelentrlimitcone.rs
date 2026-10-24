use super::*;
use crate::solver::cones::oprelentrcone::Oracle as OreOracle;
use sdpx_arithmetic::MpFloat;

type Mp2 = MpFloat<2>;

const SHAPES: [(usize, usize); 6] = [(1, 0), (2, 1), (3, 1), (3, 2), (4, 1), (5, 2)];

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    }
}

// Deterministic interior point: A, C = I + small symmetric perturbations,
// B small, T' with a diagonal well above G.
fn point(n: usize, m: usize, seed: u64) -> Vec<f64> {
    let k = n - m;
    let (tn, mk, tk) = (triangular_number(n), m * k, triangular_number(k));
    let mut rnd = lcg(seed);
    let mut s = vec![0.0; tn + mk + 2 * tk];
    for j in 0..n {
        for i in 0..=j {
            s[tri(i, j)] = if i == j {
                3.0 + 0.3 * rnd()
            } else {
                0.2 * rnd()
            };
        }
    }
    for e in 0..mk {
        s[tn + e] = 0.6 * rnd();
    }
    for part in 0..2 {
        let off = tn + mk + part * tk;
        for j in 0..k {
            for i in 0..=j {
                let v = 0.3 * rnd();
                s[off + tri(i, j)] = if i == j {
                    1.0 + 0.4 * (part as f64) * (j as f64 + 1.0) / k as f64 + v
                } else {
                    v
                };
            }
        }
    }
    s
}

fn direction(len: usize, seed: u64) -> Vec<f64> {
    let mut rnd = lcg(seed);
    (0..len).map(|_| rnd()).collect()
}

fn cast<T: FloatT>(v: &[f64]) -> Vec<T> {
    v.iter().map(|&x| x.as_T()).collect()
}

fn max_abs<T: FloatT>(v: &[T]) -> T {
    v.iter().fold(T::zero(), |a, &x| a.max(x.abs()))
}

/// max |a − b| / max(1, max |b|)
fn rel<T: FloatT>(a: &[T], b: &[T]) -> T {
    let d = a
        .iter()
        .zip(b)
        .fold(T::zero(), |m, (&x, &y)| m.max((x - y).abs()));
    d / max_abs(b).max(T::one())
}

/// A value of one precision in another (through its decimal form).
fn to<U: FloatT, V: FloatT>(v: V) -> U {
    U::from_str_radix(&v.decimal_string(), 10).ok().unwrap()
}

#[test]
fn sizes_and_degree() {
    let c = OpRelEntropyLimitCone::<f64>::new(5, 2);
    assert_eq!(c.numel(), 15 + 6 + 12);
    assert_eq!(c.degree(), 5 + 2 * 3);
    let t = SupportedConeT::<f64>::OpRelEntropyLimitConeT(5, 2);
    assert_eq!(t.nvars(), c.numel());
    assert_eq!(
        OpRelEntropyLimitCone::<f64>::new(4, 0).numel(),
        3 * triangular_number(4)
    );
}

#[test]
fn gradient_and_hessian_match_differences() {
    for (n, m) in SHAPES {
        let s = point(n, m, 7 + n as u64 + 10 * m as u64);
        let dim = s.len();
        let d = direction(dim, 101 + n as u64 + 10 * m as u64);
        let mut o = Oracle::new(n, m, &s).expect("interior");
        let mut g = vec![0.0; dim];
        o.gradient(&mut g);
        let h = 1e-6;
        let at = |t: f64| {
            let p: Vec<f64> = s.iter().zip(&d).map(|(a, b)| a + t * b).collect();
            Oracle::new(n, m, &p).unwrap()
        };
        let fd = (at(h).barrier - at(-h).barrier) / (2.0 * h);
        let an: f64 = g.iter().zip(&d).map(|(a, b)| a * b).sum();
        assert!(
            (fd - an).abs() <= 1e-6 * (1.0 + an.abs()),
            "({n},{m}) grad {fd} vs {an}"
        );
        let mut hd = vec![0.0; dim];
        o.hess_prod(&d, &mut hd);
        let (mut gp, mut gm) = (vec![0.0; dim], vec![0.0; dim]);
        at(h).gradient(&mut gp);
        at(-h).gradient(&mut gm);
        let scale = max_abs(&hd);
        for k in 0..dim {
            let fd = (gp[k] - gm[k]) / (2.0 * h);
            assert!(
                (fd - hd[k]).abs() <= 1e-5 * (1.0 + scale),
                "({n},{m}) k={k} {fd} vs {}",
                hd[k]
            );
        }
    }
}

#[test]
fn log_homogeneity() {
    fn check<T: FloatT>() {
        let tol = T::epsilon() * (1e5).as_T();
        for (n, m) in SHAPES {
            let s: Vec<T> = cast(&point(n, m, 17 + n as u64 + 10 * m as u64));
            let dim = s.len();
            let mut o = Oracle::new(n, m, &s).unwrap();
            let mut g = vec![T::zero(); dim];
            o.gradient(&mut g);
            // ⟨g, s⟩ = −ν and ∇²F s = −g
            let nu: T = (n + 2 * (n - m)).as_T();
            let gs = g.dot(&s);
            assert!((gs + nu).abs() <= tol * nu, "({n},{m}) <g,s> {gs}");
            let mut hs = vec![T::zero(); dim];
            o.hess_prod(&s, &mut hs);
            hs.iter_mut().for_each(|v| *v = -*v);
            assert!(rel(&hs, &g) <= tol, "({n},{m}) Hs+g {}", rel(&hs, &g));
        }
    }
    check::<f64>();
    check::<Mp2>();
}

#[test]
fn factored_scaling_inverts_hessian() {
    fn check<T: FloatT>() {
        let tol = T::epsilon() * (1e6).as_T();
        for (n, m) in SHAPES {
            let s: Vec<T> = cast(&point(n, m, 31 + n as u64 + 10 * m as u64));
            let mut o = Oracle::new(n, m, &s).unwrap();
            let dim = s.len();
            let mu: T = (0.37).as_T();
            let mut cone = OpRelEntropyLimitCone::<T>::new(n, m);
            assert!(cone.update_scaling(&s, &s, mu, ScalingStrategy::Dual));
            let (mut e, mut col, mut back) = (
                vec![T::zero(); dim],
                vec![T::zero(); dim],
                vec![T::zero(); dim],
            );
            for j in 0..dim {
                e.fill(T::zero());
                e[j] = T::one();
                o.hess_prod(&e, &mut col);
                col.iter_mut().for_each(|v| *v *= mu);
                cone.mul_Hs(&mut back, &col, &mut []);
                assert!(rel(&back, &e) <= tol, "({n},{m}) column {j}");
            }
            // N⁻¹·Hs·N⁻ᵀ is the packed KKT block D
            let mut packed = vec![T::zero(); dim * (dim + 1) / 2];
            cone.get_Hs(&mut packed);
            let rows = cone.rows().clone();
            let scale = max_abs(&packed);
            for j in 0..dim {
                e.fill(T::zero());
                e[j] = T::one();
                rows.from_eigen(&mut e);
                cone.mul_Hs(&mut back, &e, &mut []);
                rows.to_eigen(&mut back);
                for i in 0..dim {
                    let (a, b) = (i.min(j), i.max(j));
                    let want = packed[b * (b + 1) / 2 + a];
                    assert!(
                        (back[i] - want).abs() <= tol * scale,
                        "({n},{m}) D({i},{j})"
                    );
                }
            }
        }
    }
    check::<f64>();
    check::<Mp2>();
}

#[test]
fn unit_schur_columns_match_dense() {
    for (n, m) in SHAPES {
        let s = point(n, m, 71 + n as u64 + 10 * m as u64);
        let mut o = Oracle::new(n, m, &s).unwrap();
        let s0 = o.schur_matrix();
        let k = n - m;
        let (mk, tk, kk) = (m * k, triangular_number(k), k * k);
        let ny = mk + 2 * tk;
        let scale = max_abs(&s0);
        let (mut a, mut c) = (vec![0.0; kk], vec![0.0; kk]);
        let (mut oh, mut oa, mut oc) = (vec![0.0; mk], vec![0.0; kk], vec![0.0; kk]);
        let (mut col, mut ws) = (vec![0.0; ny], Vec::new());
        for j in 0..ny {
            let mut e = vec![0.0; ny];
            e[j] = 1.0;
            smat(k, &e[mk..mk + tk], &mut a);
            smat(k, &e[mk + tk..], &mut c);
            o.schur_apply((&e[..mk], &a, &c), &mut oh, &mut oa, &mut oc, &mut ws);
            col[..mk].copy_from_slice(&oh);
            svec(k, &oa, &mut col[mk..mk + tk]);
            svec(k, &oc, &mut col[mk + tk..]);
            for i in 0..ny {
                assert!(
                    (col[i] - s0[i + j * ny]).abs() <= 1e-13 * scale,
                    "({n},{m}) S0({i},{j}) {} vs {}",
                    col[i],
                    s0[i + j * ny]
                );
            }
        }
    }
}

#[test]
fn unit_initialization_is_central() {
    fn check<T: FloatT>() {
        for (n, m) in SHAPES {
            let cone = OpRelEntropyLimitCone::<T>::new(n, m);
            let dim = cone.numel();
            let (mut z, mut s) = (vec![T::zero(); dim], vec![T::zero(); dim]);
            cone.unit_initialization(&mut z, &mut s);
            let o = Oracle::new(n, m, &s).unwrap();
            let mut g = vec![T::zero(); dim];
            o.gradient(&mut g);
            g.iter_mut().for_each(|v| *v = -*v);
            assert!(rel(&g, &s) <= T::epsilon() * (1e3).as_T(), "({n},{m})");
        }
    }
    check::<f64>();
    check::<Mp2>();
}

#[test]
fn dual_form_matches_hs() {
    fn check<T: FloatT>() {
        let tol = T::epsilon() * (1e6).as_T();
        for (n, m) in SHAPES {
            let s: Vec<T> = cast(&point(n, m, 57 + n as u64 + 10 * m as u64));
            let dim = s.len();
            let mut o = Oracle::new(n, m, &s).unwrap();
            let mut cone = OpRelEntropyLimitCone::<T>::new(n, m);
            assert!(cone.update_scaling(&s, &s, T::one(), ScalingStrategy::Dual));
            let mut hz = vec![T::zero(); dim];
            let mut quad = |z: &[T]| -> T {
                cone.mul_Hs(&mut hz, z, &mut []);
                hz.dot(z)
            };
            // on the central path: z = −2∇F(s), ⟨z, ∇²F⁻¹z⟩ = 4ν
            let mut g = vec![T::zero(); dim];
            o.gradient(&mut g);
            let z: Vec<T> = g.iter().map(|&v| -(v + v)).collect();
            let (e, q) = (o.dual_form(&z).unwrap(), quad(&z));
            let nu: T = (n + 2 * (n - m)).as_T();
            assert!((e - q).abs() <= tol * q, "({n},{m}) central {e} vs {q}");
            let four: T = (4.0).as_T();
            assert!((q - four * nu).abs() <= tol * q);
            for seed in 0..3u64 {
                let z: Vec<T> = cast(&direction(dim, 900 + seed));
                let (e, q) = (o.dual_form(&z).unwrap(), quad(&z));
                assert!((e - q).abs() <= tol * q, "({n},{m}) exact {e} vs {q}");
            }
        }
    }
    check::<f64>();
    check::<Mp2>();
}

#[test]
fn dual_ok_matches_factor_near_threshold() {
    fn check<T: FloatT>() {
        let separation = T::epsilon().sqrt().sqrt();
        for (n, m) in [(3usize, 1usize), (5, 2)] {
            let s: Vec<T> = cast(&point(n, m, 63 + n as u64));
            let z: Vec<T> = cast(&direction(s.len(), 911 + n as u64));
            let exact = Oracle::new(n, m, &s).unwrap().dual_form(&z).unwrap();
            for sign in [-T::one(), T::one()] {
                let limit = exact * (T::one() + sign * separation);
                let mut o = Oracle::new(n, m, &s).unwrap();
                assert_eq!(
                    o.dual_ok(&z, |q| q < limit),
                    exact < limit,
                    "({n},{m}) sign {sign}"
                );
            }
            // the T part alone rejects without S₀'s factor
            let (t, _) = Oracle::new(n, m, &s).unwrap().dual_parts(&z);
            let mut o = Oracle::new(n, m, &s).unwrap();
            assert!(!o.dual_ok(&z, |q| q < t));
            assert!(o.s0l.is_none() && o.k3.is_empty());
        }
    }
    check::<f64>();
    check::<Mp2>();
}

#[test]
fn third_order_matches_hessian_differences() {
    for (n, m) in [(2usize, 1usize), (3, 1), (4, 2)] {
        let s = point(n, m, 211 + n as u64 + 10 * m as u64);
        let h = direction(s.len(), 307 + n as u64);
        let mut cone = OpRelEntropyLimitCone::<f64>::new(n, m);
        assert!(cone.update_scaling(&s, &s, 1.0, ScalingStrategy::Dual));
        let v = cone.third_order(&h).expect("interior");
        let e = 1e-5;
        let hp = |t: f64| {
            let p: Vec<f64> = s.iter().zip(&h).map(|(a, b)| a + t * b).collect();
            let mut o = Oracle::new(n, m, &p).unwrap();
            let mut out = vec![0.0; s.len()];
            o.hess_prod(&h, &mut out);
            out
        };
        let (a, b) = (hp(e), hp(-e));
        let scale = max_abs(&v);
        let mixed = cone.third_mixed(&h, &h).expect("interior");
        let g = direction(s.len(), 401 + n as u64);
        let (ab, ba) = (
            cone.third_mixed(&h, &g).unwrap(),
            cone.third_mixed(&g, &h).unwrap(),
        );
        for k in 0..s.len() {
            assert!(
                (mixed[k] - v[k]).abs() <= 1e-6 * (1.0 + scale),
                "({n},{m}) k={k} mixed"
            );
            assert!(
                (ab[k] - ba[k]).abs() <= 1e-6 * (1.0 + ab[k].abs()),
                "({n},{m}) k={k} symmetric"
            );
            let fd = (a[k] - b[k]) / (2.0 * e);
            assert!(
                (fd - v[k]).abs() <= 1e-5 * (1.0 + scale),
                "({n},{m}) k={k} fd {fd} vs {}",
                v[k]
            );
        }
    }
}

#[test]
fn rank_zero_is_ore() {
    fn check<T: FloatT>() {
        let tol = T::epsilon() * (1e3).as_T();
        for n in [1usize, 2, 4] {
            let s: Vec<T> = cast(&point(n, 0, 81 + n as u64));
            let dim = s.len();
            let d: Vec<T> = cast(&direction(dim, 83 + n as u64));
            let mut lim = Oracle::new(n, 0, &s).unwrap();
            let mut ore = OreOracle::new(n, &s).unwrap();
            let f = ore.barrier.abs().max(T::one());
            assert!((lim.barrier - ore.barrier).abs() <= tol * f, "n={n} F");
            let (mut a, mut b) = (vec![T::zero(); dim], vec![T::zero(); dim]);
            lim.gradient(&mut a);
            ore.gradient(&mut b);
            assert!(rel(&a, &b) <= tol, "n={n} gradient");
            lim.hess_prod(&d, &mut a);
            ore.hess_prod(&d, &mut b);
            assert!(rel(&a, &b) <= tol, "n={n} Hessian");
            let (il, io) = (lim.schur_inverse().unwrap(), ore.schur_inverse().unwrap());
            assert!(rel(&il, &io) <= tol, "n={n} S0 inverse");
            let mu: T = (0.61).as_T();
            let mut cl = OpRelEntropyLimitCone::<T>::new(n, 0);
            let mut co = OpRelEntropyCone::<T>::new(n);
            assert!(cl.update_scaling(&s, &s, mu, ScalingStrategy::Dual));
            assert!(co.update_scaling(&s, &s, mu, ScalingStrategy::Dual));
            cl.mul_Hs(&mut a, &d, &mut []);
            co.mul_Hs(&mut b, &d, &mut []);
            assert!(rel(&a, &b) <= tol, "n={n} Hs");
            let mut pl = vec![T::zero(); dim * (dim + 1) / 2];
            let mut po = pl.clone();
            cl.get_Hs(&mut pl);
            co.get_Hs(&mut po);
            assert!(rel(&pl, &po) <= tol, "n={n} D");
            for forward in [true, false] {
                a.copy_from_slice(&d);
                b.copy_from_slice(&d);
                if forward {
                    cl.rows().to_eigen(&mut a);
                    co.rows().to_eigen(&mut b);
                } else {
                    cl.rows().from_eigen(&mut a);
                    co.rows().from_eigen(&mut b);
                }
                assert!(rel(&a, &b) <= tol, "n={n} rows (to_eigen {forward})");
            }
            assert_eq!(cl.degree(), co.degree());
            let (mut zl, mut sl) = (vec![T::zero(); dim], vec![T::zero(); dim]);
            let (mut zo, mut so) = (vec![T::zero(); dim], vec![T::zero(); dim]);
            cl.unit_initialization(&mut zl, &mut sl);
            co.unit_initialization(&mut zo, &mut so);
            assert_eq!(sl, so);
        }
    }
    check::<f64>();
    check::<Mp2>();
}

// Node sets across the Taylor threshold (hi − lo ≤ (hi + lo)/8), with
// coincident and widely spread eigenvalues (ascending, as `sym_eig` sorts).
const LAMBDAS: [f64; 10] = [
    1e-3, 0.3, 0.88, 0.9999999, 1.0, 1.0000001, 1.13, 1.14, 4.0, 250.0,
];

/// g = −log, φ = x log x, χ = x² log x divided differences by the Lagrange
/// sum at 1024 bits, coincident nodes split by multiples of 1e-60.
fn lagrange<R: FloatT>(x: &[f64]) -> [R; 3] {
    let delta = R::from_str_radix("1e-60", 10).ok().unwrap();
    let x: Vec<R> = x
        .iter()
        .enumerate()
        .map(|(t, &v)| {
            let (v, t): (R, R) = (v.as_T(), t.as_T());
            v + delta * t
        })
        .collect();
    let mut out = [R::zero(); 3];
    for (i, &xi) in x.iter().enumerate() {
        let mut p = R::one();
        for (j, &xj) in x.iter().enumerate() {
            if j != i {
                p *= xi - xj;
            }
        }
        let l = xi.ln();
        out[0] += -l / p;
        out[1] += xi * l / p;
        out[2] += xi * xi * l / p;
    }
    out
}

#[test]
fn divided_differences_match_lagrange() {
    fn check<T: FloatT>(ulps: f64) {
        let lam: Vec<T> = cast(&LAMBDAS);
        let k = lam.len();
        let nodes = Nodes::new(&lam);
        let ell = |a: usize| if a < k { LAMBDAS[a] } else { 1.0 };
        let tol: T = ulps.as_T();
        let mut worst = (T::zero(), Vec::new(), 0);
        for p in 0..k {
            for r in p..k {
                for q in r..k {
                    for ids in [
                        vec![p, k, k],
                        vec![p, r, k],
                        vec![p, r, k, k],
                        vec![p, r, q, k],
                        vec![p, r, q, k, k],
                    ] {
                        // φ needs order ≥ 2, χ order ≥ 3
                        let want = (ids.len() - 1).min(3);
                        let got = nodes.divided(&ids, want);
                        let x: Vec<f64> = ids.iter().map(|&a| ell(a)).collect();
                        let reference = lagrange::<MpFloat<16>>(&x);
                        for f in 0..want {
                            let r: T = to(reference[f]);
                            let e = (got[f] - r).abs() / r.abs() / T::epsilon();
                            assert!(e <= tol, "{ids:?} f={f}: {} vs {r} ({e} ulp)", got[f]);
                            if e > worst.0 {
                                worst = (e, x.clone(), f);
                            }
                        }
                    }
                }
            }
        }
        eprintln!(
            "divided differences, {} bits: worst {} ulp at {:?} (f = {})",
            T::precision_bits(),
            worst.0,
            worst.1,
            worst.2
        );
    }
    // just outside the series radius, (0.88, 0.88, 1, 1.14), the recurrence
    // loses 6.5 bits at 53 bits and 8.5 at 128
    check::<f64>(1024.0);
    check::<Mp2>(1024.0);
}

#[test]
fn kernel_identities() {
    // Leibniz with each node first: c̃ = −(ñᵢ + ñⱼ)/2 − (λᵢ + λⱼ)/2·ã,
    // K_AC = −λₚK_CC − ã_rq, −K_AA = λₚK_AC + c̃_rq, L₁ = −λₚL₀ − eᶜ_rq; and the
    // confluent values at λ = 1.
    fn check<T: FloatT>() {
        let tol = T::epsilon() * (256.0).as_T();
        // a − Σ terms vanishes to the rounding of its terms
        let close = |a: T, terms: &[T], what: &str| {
            let sum = terms.iter().fold(T::zero(), |s, &t| s + t);
            let scale = terms.iter().fold(a.abs(), |s, &t| s + t.abs());
            assert!((a - sum).abs() <= tol * scale, "{what}: {a} vs {sum}");
        };
        // A = I, C = diag(LAMBDAS), B = 1: the eigenvalues are LAMBDAS
        let (n, m) = (LAMBDAS.len() + 1, 1usize);
        let k = n - m;
        let (tn, mk, tk) = (triangular_number(n), m * k, triangular_number(k));
        let mut s = vec![T::zero(); tn + mk + 2 * tk];
        for i in 0..n {
            s[tri(i, i)] = (1e4).as_T();
        }
        for e in 0..mk {
            s[tn + e] = T::one();
        }
        for i in 0..k {
            s[tn + mk + tri(i, i)] = T::one();
            s[tn + mk + tk + tri(i, i)] = LAMBDAS[i].as_T();
        }
        let mut o = Oracle::new(n, m, &s).expect("interior");
        o.ensure_k3();
        let (rows, lam) = (&o.rows, &o.vv.lam);
        let half: T = (0.5).as_T();
        for j in 0..k {
            for i in 0..k {
                let ij = i + j * k;
                let (ni, nj) = (-half * rows.nt[i], -half * rows.nt[j]);
                let (ai, aj) = (-half * lam[i] * rows.at[ij], -half * lam[j] * rows.at[ij]);
                close(rows.ct[ij], &[ni, nj, ai, aj], "c~");
                close(rows.ea[ij], &[-lam[j] * rows.ec[ij]], "eA");
            }
        }
        for p in 0..k {
            for r in 0..k {
                for q in 0..k {
                    let kv = &o.k3[5 * (p + k * (r + k * q))..][..5];
                    let rq = r + q * k;
                    close(kv[1], &[-lam[p] * kv[2], -rows.at[rq]], "K_AC");
                    close(-kv[0], &[lam[p] * kv[1], rows.ct[rq]], "K_AA");
                    close(kv[4], &[-lam[p] * kv[3], -rows.ec[rq]], "L1");
                }
            }
        }
        // λ = 1: g[1,1,1] = 1/2, g[1,1,1,1] = −1/3, φ[1,1,1,1] = −1/6, −g[1,1] = 1
        let one = 4;
        close(lam[one], &[T::one()], "lambda(1)");
        let (n2, n3, n6): (T, T, T) = ((2.0).as_T(), (3.0).as_T(), (6.0).as_T());
        close(rows.nt[one], &[n2.recip()], "n~(1)");
        close(rows.at[one + one * k], &[-n3.recip()], "a~(1,1)");
        close(rows.ct[one + one * k], &[-n6.recip()], "c~(1,1)");
        close(rows.mg[one], &[T::one()], "m~(1)");
    }
    check::<f64>();
    check::<Mp2>();
}

/// The prototype's values at its points (`limit_cone_barrier.py`, 50 digits).
struct Fixture {
    n: usize,
    m: usize,
    kind: String,
    x: Vec<f64>,
    f: String,
    g: Vec<String>,
    hh: Vec<(Vec<f64>, Vec<String>)>,
}

fn fixtures() -> Vec<Fixture> {
    let f64s = |w: &[&str]| -> Vec<f64> { w.iter().map(|v| v.parse().unwrap()).collect() };
    let strs = |w: &[&str]| -> Vec<String> { w.iter().map(|v| v.to_string()).collect() };
    let mut out: Vec<Fixture> = Vec::new();
    for line in include_str!("oprelentrlimitcone.dat").lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        let Some((&tag, rest)) = words.split_first() else {
            continue;
        };
        match tag {
            "point" => out.push(Fixture {
                n: rest[0].parse().unwrap(),
                m: rest[1].parse().unwrap(),
                kind: rest[2].to_string(),
                x: Vec::new(),
                f: String::new(),
                g: Vec::new(),
                hh: Vec::new(),
            }),
            "x" => out.last_mut().unwrap().x = f64s(rest),
            "F" => out.last_mut().unwrap().f = rest[0].to_string(),
            "g" => out.last_mut().unwrap().g = strs(rest),
            "h" => out.last_mut().unwrap().hh.push((f64s(rest), Vec::new())),
            "Hh" => out.last_mut().unwrap().hh.last_mut().unwrap().1 = strs(rest),
            _ => {}
        }
    }
    out
}

#[test]
fn matches_prototype_fixtures() {
    fn check<T: FloatT>(ulps: impl Fn(&str) -> f64) {
        let parse = |v: &[String]| -> Vec<T> {
            v.iter()
                .map(|s| T::from_str_radix(s, 10).ok().unwrap())
                .collect()
        };
        let all = fixtures();
        assert_eq!(all.len(), 8);
        let mut bad = 0;
        for fx in all {
            let (n, m) = (fx.n, fx.m);
            let x: Vec<T> = cast(&fx.x);
            assert_eq!(x.len(), OpRelEntropyLimitCone::<T>::size(n, m));
            let mut o = Oracle::new(n, m, &x).expect("fixture point interior");
            let f = parse(&[fx.f.clone()])[0];
            let ef = (o.barrier - f).abs() / f.abs().max(T::one()) / T::epsilon();
            let mut g = vec![T::zero(); x.len()];
            o.gradient(&mut g);
            let eg = rel(&g, &parse(&fx.g)) / T::epsilon();
            let mut eh = T::zero();
            for (h, want) in &fx.hh {
                let mut out = vec![T::zero(); x.len()];
                o.hess_prod(&cast(h), &mut out);
                eh = eh.max(rel(&out, &parse(want)) / T::epsilon());
            }
            eprintln!(
                "{} bits ({n},{m}) {}: F {ef} g {eg} Hh {eh} ulp; lambda {:?}",
                T::precision_bits(),
                fx.kind,
                o.vv.lam
                    .iter()
                    .map(|v| v.to_f64().unwrap())
                    .collect::<Vec<_>>()
            );
            let tol: T = ulps(&fx.kind).as_T();
            bad += usize::from(!(ef <= tol && eg <= tol && eh <= tol));
        }
        assert_eq!(bad, 0);
    }
    // Rounding grows with the conditioning the prototype builds into its
    // points, alike at both precisions: λ(T' − G) down to 1e-12 (boundary;
    // 128 bits: 6e12 ulp), cond A = 1e10 (illA: 3e9 ulp at 53 bits) and λ
    // over 1e-3..1e3 (spread: 6e4 ulp, as the ORE's eigenbasis); ≤ 200 ulp
    // at the others.
    let ulps = |kind: &str| match kind {
        "boundary" => 1e14,
        "illA" => 1e11,
        "spread" => 1e6,
        _ => 1e3,
    };
    check::<f64>(ulps);
    check::<Mp2>(ulps);
}
