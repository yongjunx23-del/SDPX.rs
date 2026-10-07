//! Eliminate SOC3 leaves around a small border of shared primal variables.
use super::*;
use crate::solver::cones::{CompositeCone, Cone, SupportedCone};

impl<T: FloatT> ArrowLDLSolver<T> {
    pub(crate) fn try_shared_soc(
        k: &CscMatrix<T>,
        signs: &[i8],
        a: &CscMatrix<T>,
        cones: &CompositeCone<T>,
        settings: &CoreSettings<T>,
    ) -> Option<Self> {
        let n = a.n;
        debug_assert_eq!(k.n, n + a.m);
        let mut row_owner = vec![usize::MAX; a.m];
        let mut groups = Vec::new();
        let mut trunk = Vec::new();
        for (cone, rows) in cones.iter().zip(&cones.rng_cones) {
            match cone {
                SupportedCone::ZeroCone(_) => trunk.extend(rows.clone().map(|r| n + r)),
                SupportedCone::SecondOrderCone(c) if c.numel() == 3 => {
                    row_owner[rows.clone()].fill(groups.len());
                    groups.push(rows.clone().map(|r| n + r).collect::<Vec<_>>());
                }
                _ => return None,
            }
        }
        if groups.len() < 8 {
            return None;
        }
        // A variable touching exactly one SOC belongs to that leaf. All other
        // variables stay in the border; structural zeros count as edges.
        for col in 0..n {
            let mut touched = a.rowval[a.colptr[col]..a.colptr[col + 1]]
                .iter()
                .map(|&r| row_owner[r])
                .filter(|&g| g != usize::MAX);
            match touched.next() {
                Some(g) if touched.all(|other| other == g) => groups[g].push(col),
                _ => trunk.push(col),
            }
        }
        if trunk.is_empty() || !trunk.iter().any(|&id| id < n) {
            return None;
        }
        let t = trunk.len() as u128;
        let cells = groups
            .iter()
            .map(|g| {
                let w = g.len() as u128;
                2 * w * w + 3 * w * t + 4 * w
            })
            .sum::<u128>()
            + 4 * t * t
            + 4 * k.n as u128;
        // The dense border must not be much larger than the couplings (the
        // variables' A entries) that fill it.
        let coupling = a.colptr[n];
        if cells * std::mem::size_of::<T>() as u128 > ARROW_MAX_BYTES
            || t * t > 8 * coupling as u128
        {
            return None;
        }
        let mut owner = vec![usize::MAX; k.n];
        for (group, ids) in groups.iter().enumerate() {
            for &id in ids {
                owner[id] = group;
            }
        }
        // In particular, P (or an orthant Schur contribution) must not link
        // primal variables assigned to different leaves.
        for j in 0..k.n {
            for &i in &k.rowval[k.colptr[j]..k.colptr[j + 1]] {
                if owner[i] != usize::MAX && owner[j] != usize::MAX && owner[i] != owner[j] {
                    return None;
                }
            }
        }
        Some(Self::from_groups(
            k,
            signs,
            settings,
            groups,
            trunk,
            Some(LocalStructure::SharedSoc),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::SupportedConeT::*;
    use num_traits::{FromPrimitive, One, Zero};
    use sdpx_arithmetic::{Bits512, Scalar};

    fn shared_soc_parity<T: FloatT>() {
        // Four SOC-only leaves, four SOC+primal leaves, and a mixed-sign
        // border (two shared primals and one equality multiplier).
        let (n, m) = (6, 25);
        let mut ai = Vec::new();
        let mut aj = Vec::new();
        let mut av = Vec::new();
        for g in 0..8 {
            for (r, c, v) in [(3 * g, 0, 0.25), (3 * g + 1, 1, -0.5)] {
                ai.push(r);
                aj.push(c);
                av.push(T::from_f64(v).unwrap());
            }
            if g < 4 {
                ai.push(3 * g + 2);
                aj.push(g + 2);
                av.push(T::one());
            }
        }
        ai.push(24);
        aj.push(2);
        av.push(T::from_f64(0.125).unwrap());
        let a = CscMatrix::new_from_triplets(m, n, ai, aj, av);
        let mut types = vec![SecondOrderConeT(3); 8];
        types.push(ZeroConeT(1));
        let cones = CompositeCone::new(&types);
        let signs: Vec<i8> = (0..n + m).map(|i| if i < n { 1 } else { -1 }).collect();
        let mut ki = Vec::new();
        let mut kj = Vec::new();
        let mut kv = Vec::new();
        for j in 0..n + m {
            ki.push(j);
            kj.push(j);
            kv.push(T::from_f64(5. * signs[j] as f64).unwrap());
        }
        for j in 0..n {
            for p in a.colptr[j]..a.colptr[j + 1] {
                ki.push(j);
                kj.push(n + a.rowval[p]);
                kv.push(a.nzval[p]);
            }
        }
        for g in 0..8 {
            ki.push(n + 3 * g);
            kj.push(n + 3 * g + 2);
            kv.push(T::from_f64(0.25).unwrap());
        }
        // A shared/local P coupling is legal.
        ki.push(0);
        kj.push(2);
        kv.push(T::from_f64(0.125).unwrap());
        let k = CscMatrix::new_from_triplets(n + m, n + m, ki.clone(), kj.clone(), kv.clone());
        let settings = CoreSettings::default();
        let mut solver = ArrowLDLSolver::try_shared_soc(&k, &signs, &a, &cones, &settings).unwrap();
        assert_eq!(solver.trunk, vec![30, 0, 1]);
        assert_eq!(
            solver
                .leaves
                .iter()
                .map(|l| l.ids.len())
                .collect::<Vec<_>>(),
            vec![4, 4, 4, 4, 3, 3, 3, 3]
        );
        let rhs: Vec<T> = (0..3 * k.n)
            .map(|i| T::from_f64((i as f64 - 10.) / 16.).unwrap())
            .collect();
        let mut reference = None;
        for threads in [1, 4] {
            solver.set_pool(Some(Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap(),
            )));
            assert!(solver.refactor(&k));
            assert!(solver.use_arrow);
            let mut actual = vec![T::zero(); rhs.len()];
            solver.solve_many(&k, &mut actual, &mut rhs.clone(), 3);
            for c in 0..3 {
                let span = c * k.n..(c + 1) * k.n;
                let mut residual = rhs[span.clone()].to_vec();
                k.sym_up()
                    .symv(&mut residual, &actual[span], -T::one(), T::one());
                assert!(residual.norm_inf() < T::epsilon() * T::from_f64(1024.).unwrap());
            }
            if let Some(expected) = &reference {
                assert_eq!(&actual, expected);
            } else {
                reference = Some(actual);
            }
        }
        // Cross-leaf P edges must decline the decomposition, including zeros.
        ki.push(2);
        kj.push(3);
        kv.push(T::zero());
        let crossed = CscMatrix::new_from_triplets(n + m, n + m, ki, kj, kv);
        assert!(ArrowLDLSolver::try_shared_soc(&crossed, &signs, &a, &cones, &settings).is_none());
    }

    #[test]
    fn shared_soc_float64() {
        shared_soc_parity::<f64>();
    }
    #[test]
    fn shared_soc_mpfr512() {
        shared_soc_parity::<Bits512>();
    }
    #[test]
    fn shared_soc_augmented_solve_mpfr512() {
        use crate::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus};
        type T = Bits512;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        let mut b = vec![T::zero(); 48];
        for g in 0..8 {
            rows.extend([3 * g, 3 * g + 1]);
            cols.extend([g + 1, 0]);
            vals.extend([-T::one(), -T::one()]);
            b[3 * g + 1] = T::from_f64(if g % 2 == 0 { 1. } else { -1. }).unwrap();
            b[3 * g + 2] = T::one();
        }
        // SOC-only leaves constrain the same shared variable.
        for g in 8..16 {
            rows.push(3 * g + 1);
            cols.push(0);
            vals.push(T::one());
            b[3 * g] = T::from_f64(2.).unwrap();
        }
        let a = CscMatrix::new_from_triplets(48, 9, rows, cols, vals);
        let mut q = vec![T::one(); 9];
        q[0] = T::zero();
        let p = CscMatrix::zeros((9, 9));
        let cones = vec![SecondOrderConeT(3); 16];
        let mut settings = DefaultSettings::<T>::default();
        // The condensed form eliminates SOC3 rows instead; the augmented
        // KKT of SOC3-only cones is the shared arrow.
        settings.kkt_form = "augmented".into();
        settings.verbose = false;
        settings.tol_feas = T::from_f64(1e-40).unwrap();
        settings.tol_gap_abs = settings.tol_feas;
        settings.tol_gap_rel = settings.tol_feas;
        let expected = T::from_f64(8.).unwrap() * T::from_f64(2.).unwrap().sqrt();
        let mut reference = None;
        for (method, threads) in [("auto", 1), ("auto", 4), ("qdldl", 1)] {
            settings.direct_solve_method = method.into();
            settings.max_threads = threads;
            let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings.clone()).unwrap();
            solver.solve();
            assert_eq!(solver.solution.status, SolverStatus::Solved);
            assert!((solver.solution.obj_val - expected).abs() < T::from_f64(1e-35).unwrap());
            if method == "auto" {
                assert_eq!(solver.info.linsolver.name, "shared_soc_arrow");
                if let Some(point) = &reference {
                    assert_eq!(&solver.solution.x, point);
                } else {
                    reference = Some(solver.solution.x.clone());
                }
            }
        }
    }
}
