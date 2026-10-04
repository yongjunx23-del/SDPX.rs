use super::*;
use sdpx_arithmetic::MpFloat;

fn problem<T: FloatT>(border: usize) -> DefaultProblemData<T> {
    let n = 6;
    let mut ptr = vec![0];
    let mut rows = Vec::new();
    let mut values = Vec::new();
    for j in 0..n {
        for i in 0..border {
            rows.push(i);
            // Include stored zeros, both signs, and cancellation in the Gram.
            values.push(T::from_i32(((i + 2 * j) % 5) as i32 - 2).unwrap());
        }
        rows.push(border + j);
        values.push(T::one());
        ptr.push(rows.len());
    }
    let mut cones = Vec::new();
    if border != 0 {
        cones.push(SupportedConeT::ZeroConeT(border));
    }
    cones.push(SupportedConeT::NonnegativeConeT(n));
    DefaultProblemData::new(
        &CscMatrix::identity(n),
        &vec![T::one(); n],
        &CscMatrix::new(border + n, n, ptr, rows, values),
        &vec![T::one(); border + n],
        &cones,
        &DefaultSettings {
            presolve_enable: false,
            equilibrate_enable: false,
            chordal_decomposition_enable: false,
            ..DefaultSettings::default()
        },
    )
}

// Original owner-first reduction, independent of the column scheduling path.
fn original<T: FloatT>(kkt: &OwnedKkt<T>) -> Vec<T> {
    let border = kkt.border_matrix.n;
    let mut result = vec![T::zero(); kkt.border_matrix.nnz()];
    let mut scratch = vec![T::zero(); border];
    for j in 0..border {
        result[kkt.border_matrix.colptr[j] + j] = kkt.shift;
    }
    for local in &kkt.locals {
        let d = local.kernel.interior_dimension();
        for j in 0..border {
            local.coupling.gemv(
                &mut scratch,
                &local.response[j * d..j * d + local.n],
                T::one(),
                T::zero(),
            );
            for i in 0..=j {
                result[kkt.border_matrix.colptr[j] + i] += scratch[i];
            }
        }
    }
    result
}

fn reset<T: FloatT>(kkt: &mut OwnedKkt<T>) {
    kkt.border_matrix.nzval.fill(T::zero());
    for j in 0..kkt.border_matrix.n {
        kkt.border_matrix.nzval[kkt.border_matrix.colptr[j] + j] = kkt.shift;
    }
    // GEMV beta=0 must discard any previous scratch content.
    kkt.border_rhs.fill(T::nan());
    for work in &mut kkt.border_work {
        work.fill(T::nan());
    }
}

fn check<T: FloatT>() {
    let settings = CoreSettings::<T>::default();
    for border in [0, 1, 7] {
        for owners in [1, 4, 8] {
            for workers in [1, 2, 4, 8] {
                let mut state = OwnedState::new(problem::<T>(border), owners).unwrap();
                for owner in &mut state.owners {
                    owner.cones.set_identity_scaling();
                }
                let pool = if workers == 1 {
                    None
                } else {
                    Some(Arc::new(
                        rayon::ThreadPoolBuilder::new()
                            .num_threads(workers)
                            .build()
                            .unwrap(),
                    ))
                };
                let mut kkt = OwnedKkt::new_with_pool(
                    &state.layout,
                    state.owners.iter().map(|o| (&o.data, &o.cones)),
                    &settings,
                    pool,
                    false,
                );
                let cones: Vec<_> = state.owners.into_iter().map(|o| o.cones).collect();
                let buffers: Vec<_> = kkt.border_work.iter().map(|w| w.as_ptr()).collect();
                assert_eq!(buffers.len(), workers.min(border));
                assert!(kkt.update_local(&cones, &settings));
                let expected = original(&kkt);
                assert_eq!(kkt.border_matrix.nzval, expected);
                for _ in 0..2 {
                    reset(&mut kkt);
                    kkt.assemble_border();
                    assert_eq!(kkt.border_matrix.nzval, expected);
                }
                // A nonfinite response is still visible to the border factor.
                if border != 0 {
                    let local = kkt.locals.iter_mut().find(|l| l.n != 0).unwrap();
                    let saved = local.response[0];
                    local.response[0] = T::nan();
                    let expected_bad = original(&kkt);
                    reset(&mut kkt);
                    kkt.assemble_border();
                    assert!(expected_bad.iter().any(|x| x.is_nan()));
                    for (&a, &b) in kkt.border_matrix.nzval.iter().zip(&expected_bad) {
                        assert!(a == b || (a.is_nan() && b.is_nan()));
                    }
                    kkt.locals.iter_mut().find(|l| l.n != 0).unwrap().response[0] = saved;
                    reset(&mut kkt);
                    kkt.assemble_border();
                    assert_eq!(kkt.border_matrix.nzval, expected);
                }
                assert_eq!(
                    buffers,
                    kkt.border_work
                        .iter()
                        .map(|w| w.as_ptr())
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn owned_border_parallel_f64() {
    check::<f64>();
}
#[test]
fn owned_border_parallel_mpfr256() {
    check::<MpFloat<4>>();
}
#[test]
fn owned_border_parallel_mpfr512() {
    check::<MpFloat<8>>();
}
