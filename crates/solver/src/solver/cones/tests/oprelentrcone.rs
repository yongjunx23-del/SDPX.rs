use super::*;

// Deterministic interior point: X, Y = I + small symmetric perturbations,
// T = D(X, Y) + margin·I (built through the oracle's own Ẑ shift).
fn point(n: usize, seed: u64) -> Vec<f64> {
    let m = n * (n + 1) / 2;
    let mut state = seed;
    let mut rnd = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let mut s = vec![0.0; 3 * m];
    for part in 1..3 {
        for j in 0..n {
            for i in 0..=j {
                let v = 0.3 * rnd();
                s[part * m + tri(i, j)] = if i == j { 1.0 + 0.4 * (part as f64) * (j as f64 + 1.0) / n as f64 + v } else { v };
            }
        }
    }
    // T: large enough diagonal that T − D ≻ 0
    for j in 0..n {
        for i in 0..=j {
            s[tri(i, j)] = if i == j { 2.5 + 0.2 * rnd() } else { 0.2 * rnd() };
        }
    }
    s
}

fn direction(n: usize, seed: u64) -> Vec<f64> {
    let mut d = point(n, seed);
    d.iter_mut().for_each(|v| *v -= 0.7);
    d
}

#[test]
fn gradient_and_hessian_match_differences() {
    for n in [1usize, 2, 4, 6] {
        let s = point(n, 7 + n as u64);
        let d = direction(n, 101 + n as u64);
        let mut o = Oracle::new(n, &s).expect("interior");
        let dim = s.len();
        let mut g = vec![0.0; dim];
        o.gradient(&mut g);
        let h = 1e-6;
        let at = |t: f64| {
            let p: Vec<f64> = s.iter().zip(&d).map(|(a, b)| a + t * b).collect();
            Oracle::new(n, &p).unwrap()
        };
        let fd = (at(h).barrier - at(-h).barrier) / (2.0 * h);
        let an: f64 = g.iter().zip(&d).map(|(a, b)| a * b).sum();
        assert!((fd - an).abs() <= 1e-6 * (1.0 + an.abs()), "n={n} grad {fd} vs {an}");
        // Hessian product against the gradient difference
        let mut hd = vec![0.0; dim];
        o.hess_prod(&d, &mut hd);
        let (mut gp, mut gm) = (vec![0.0; dim], vec![0.0; dim]);
        at(h).gradient(&mut gp);
        at(-h).gradient(&mut gm);
        let scale = hd.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        for k in 0..dim {
            let fd = (gp[k] - gm[k]) / (2.0 * h);
            assert!((fd - hd[k]).abs() <= 1e-5 * (1.0 + scale), "n={n} k={k} {fd} vs {}", hd[k]);
        }
        // log-homogeneity: ⟨g, s⟩ = −ν and H s = −g
        let gs: f64 = g.iter().zip(&s).map(|(a, b)| a * b).sum();
        assert!((gs + 3.0 * n as f64).abs() < 1e-9, "n={n} <g,s> {gs}");
        let mut hs = vec![0.0; dim];
        o.hess_prod(&s, &mut hs);
        for k in 0..dim {
            assert!((hs[k] + g[k]).abs() < 1e-8 * (1.0 + g[k].abs()), "n={n} Hs+g at {k}");
        }
    }
}

#[test]
fn factored_scaling_inverts_hessian() {
    for n in [1usize, 2, 3, 5] {
        let s = point(n, 31 + n as u64);
        let mut o = Oracle::new(n, &s).unwrap();
        let dim = s.len();
        let mu = 0.37;
        let mut cone = OpRelEntropyCone::<f64>::new(n);
        assert!(cone.update_scaling(&s, &s, mu, ScalingStrategy::Dual));
        let (mut e, mut col, mut back) = (vec![0.0; dim], vec![0.0; dim], vec![0.0; dim]);
        for j in 0..dim {
            e.fill(0.0);
            e[j] = 1.0;
            o.hess_prod(&e, &mut col);
            col.iter_mut().for_each(|v| *v *= mu);
            cone.mul_Hs(&mut back, &col, &mut []);
            for i in 0..dim {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((back[i] - want).abs() < 1e-8, "n={n} ({i},{j}) {}", back[i]);
            }
        }
        // N⁻¹·Hs·N⁻ᵀ is the packed KKT block D
        let mut packed = vec![0.0; dim * (dim + 1) / 2];
        cone.get_Hs(&mut packed);
        let rows = cone.rows().clone();
        let mut v = vec![0.0; dim];
        for j in 0..dim {
            v.fill(0.0);
            v[j] = 1.0;
            rows.from_eigen(&mut v);
            cone.mul_Hs(&mut back, &v, &mut []);
            rows.to_eigen(&mut back);
            for i in 0..dim {
                let (a, b) = (i.min(j), i.max(j));
                let want = packed[b * (b + 1) / 2 + a];
                assert!((back[i] - want).abs() < 1e-9 * (1.0 + want.abs()), "n={n} D({i},{j})");
            }
        }
    }
}

#[test]
fn dual_bound_is_upper_and_tight_on_central_path() {
    for n in [1usize, 2, 4] {
        let s = point(n, 57 + n as u64);
        let mut o = Oracle::new(n, &s).unwrap();
        let mut cone = OpRelEntropyCone::<f64>::new(n);
        assert!(cone.update_scaling(&s, &s, 1.0, ScalingStrategy::Dual));
        let mut hz = vec![0.0; s.len()];
        let mut quad = |z: &[f64]| -> f64 {
            cone.mul_Hs(&mut hz, z, &mut []);
            hz.iter().zip(z).map(|(a, b)| a * b).sum()
        };
        let mut g = vec![0.0; s.len()];
        o.gradient(&mut g);
        let z: Vec<f64> = g.iter().map(|v| -2.0 * v).collect();
        let (b, q) = (o.dual_form(&z, false).unwrap(), quad(&z));
        assert!((b - q).abs() < 1e-8 * q, "n={n} central {b} vs {q}");
        assert!((q - 4.0 * 3.0 * n as f64).abs() < 1e-8 * q);
        for seed in 0..5u64 {
            let z = direction(n, 900 + seed);
            let (b, q) = (o.dual_form(&z, false).unwrap(), quad(&z));
            assert!(b >= q * (1.0 - 1e-10), "n={n} bound {b} < {q}");
            let e = o.dual_form(&z, true).unwrap();
            assert!((e - q).abs() < 1e-9 * q, "n={n} exact {e} vs {q}");
        }
    }
}

#[test]
fn unit_initialization_is_central() {
    for n in [1usize, 3] {
        let cone = OpRelEntropyCone::<f64>::new(n);
        let dim = cone.numel();
        let (mut z, mut s) = (vec![0.0; dim], vec![0.0; dim]);
        cone.unit_initialization(&mut z, &mut s);
        let o = Oracle::new(n, &s).unwrap();
        let mut g = vec![0.0; dim];
        o.gradient(&mut g);
        for k in 0..dim {
            assert!((s[k] + g[k]).abs() < 1e-12, "n={n} s=-g at {k}: {} {}", s[k], g[k]);
        }
    }
}

#[test]
fn divided_differences_near_coincident_nodes() {
    // g = −log: g[a,b] → −1/a, g[a,b,c] → 1/(2a²)
    let a = 0.7f64;
    let b = a * (1.0 + 1e-9);
    let d1 = divided1(a, b, a.ln(), b.ln());
    assert!((d1 + 1.0 / (a * b).sqrt()).abs() < 1e-14);
    let v = [a, a * (1.0 + 1e-7), a * (1.0 - 2e-7)];
    let lns: Vec<f64> = v.iter().map(|x| x.ln()).collect();
    let d1f = |p: usize, q: usize| {
        if p == q {
            -1.0 / v[p]
        } else {
            divided1(v[p], v[q], lns[p], lns[q])
        }
    };
    let d2 = divided2(v, d1f);
    assert!((d2 - 0.5 / (a * a)).abs() < 1e-6 * d2);
    // well-separated nodes agree with the closed form 1/(abc)·(...) via recursion
    let w = [0.2f64, 1.3, 4.0];
    let lw: Vec<f64> = w.iter().map(|x| x.ln()).collect();
    let f1 = |p: usize, q: usize| -(lw[p] - lw[q]) / (w[p] - w[q]);
    let want = (f1(0, 1) - f1(1, 2)) / (w[0] - w[2]);
    let got = divided2(w, |p, q| if p == q { -1.0 / w[p] } else { f1(p, q) });
    assert!((got - want).abs() < 1e-14 * want.abs());
}

#[test]
fn third_order_matches_hessian_differences() {
    for n in [1usize, 2, 3] {
        let s = point(n, 211 + n as u64);
        let h = direction(n, 307 + n as u64);
        let mut cone = OpRelEntropyCone::<f64>::new(n);
        assert!(cone.update_scaling(&s, &s, 1.0, ScalingStrategy::Dual));
        let v = cone.third_order(&h).expect("interior");
        let e = 1e-5;
        let hp = |t: f64| {
            let p: Vec<f64> = s.iter().zip(&h).map(|(a, b)| a + t * b).collect();
            let mut o = Oracle::new(n, &p).unwrap();
            let mut out = vec![0.0; s.len()];
            o.hess_prod(&h, &mut out);
            out
        };
        let (a, b) = (hp(e), hp(-e));
        let scale = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let mixed = cone.third_mixed(&h, &h).expect("interior");
        let g = direction(n, 401 + n as u64);
        let (ab, ba) = (cone.third_mixed(&h, &g).unwrap(), cone.third_mixed(&g, &h).unwrap());
        for k in 0..s.len() {
            assert!((mixed[k] - v[k]).abs() <= 1e-6 * (1.0 + scale), "n={n} k={k} mixed");
            assert!((ab[k] - ba[k]).abs() <= 1e-6 * (1.0 + ab[k].abs()), "n={n} k={k} symmetric");
        }
        for k in 0..s.len() {
            let fd = (a[k] - b[k]) / (2.0 * e);
            assert!((fd - v[k]).abs() <= 1e-5 * (1.0 + scale), "n={n} k={k} fd {fd} toa {}", v[k]);
        }
    }
}

#[test]
fn unit_schur_columns_match_dense() {
    for n in [1usize, 3, 5] {
        let s = point(n, 71 + n as u64);
        let mut o = Oracle::new(n, &s).unwrap();
        o.ensure_d2();
        let (nn, m) = (n * n, n * (n + 1) / 2);
        let mut e = vec![0.0; m];
        let (mut h, mut k) = (vec![0.0; nn], vec![0.0; nn]);
        let (mut a1, mut a2, mut b1, mut b2) = (vec![0.0; nn], vec![0.0; nn], vec![0.0; nn], vec![0.0; nn]);
        let mut ws = Vec::new();
        for j in 0..n {
            for i in 0..=j {
                for part in 0..2 {
                    e.fill(0.0);
                    e[tri(i, j)] = 1.0;
                    if part == 0 {
                        smat(n, &e, &mut h);
                        k.fill(0.0);
                    } else {
                        h.fill(0.0);
                        smat(n, &e, &mut k);
                    }
                    o.schur_apply(&h, &k, &mut a1, &mut a2, &mut ws, None);
                    o.schur_apply(&h, &k, &mut b1, &mut b2, &mut ws, Some((i, j)));
                    assert_eq!((&a1, &a2), (&b1, &b2), "n={n} ({i},{j}) part {part}");
                }
            }
        }
    }
}
