use super::*;
fn arr(v: &[f64]) -> Scalars {
    Scalars {
        kind: 0,
        reserved: 0,
        count: v.len() as u64,
        f64: v.as_ptr(),
        decimal: ptr::null(),
    }
}
#[test]
fn ffi_preserves_core_ktratio_defaults() {
    fn check<T: Scalar>(bits: u32) {
        let mut input = defaults();
        input.precision_bits = bits;
        let core = DefaultSettings::<T>::default();
        let parsed = unsafe { settings::<T>(&input).unwrap() };
        assert_eq!(parsed.tol_ktratio, core.tol_ktratio);
        assert_eq!(parsed.reduced_tol_ktratio, core.reduced_tol_ktratio);
        let custom = std::ffi::CString::new("1e-5").unwrap();
        input.tol_feas = custom.as_ptr();
        let parsed = unsafe { settings::<T>(&input).unwrap() };
        assert_eq!(parsed.tol_feas, T::decimal("1e-5").unwrap());
        assert_eq!(parsed.tol_ktratio, core.tol_ktratio);
    }
    check::<f64>(53);
    check::<MpFloat<2>>(128);
    check::<MpFloat<4>>(256);
    check::<MpFloat<8>>(512);
    check::<MpFloat<12>>(768);
    check::<MpFloat<16>>(1024);
    check::<MpFloat<32>>(2048);
}
fn lp() -> *mut Handle {
    lp_settings(defaults())
}
fn lp_settings(settings: Settings) -> *mut Handle {
    unsafe {
        let p = Csc {
            rows: 1,
            cols: 1,
            nnz: 0,
            colptr: [0, 0].as_ptr(),
            rowval: ptr::null(),
            values: arr(&[]),
        };
        let a = Csc {
            rows: 1,
            cols: 1,
            nnz: 1,
            colptr: [0, 1].as_ptr(),
            rowval: [0].as_ptr(),
            values: arr(&[-1.0]),
        };
        let c = Cone {
            kind: 1,
            reserved: 0,
            dim: 1,
            alpha: arr(&[]),
        };
        let mut h = ptr::null_mut();
        assert_eq!(
            sdpx_prepare(
                &p,
                &arr(&[1.0]),
                &a,
                &arr(&[-1.0]),
                &c,
                1,
                &settings,
                &mut h
            ),
            0
        );
        h
    }
}
#[test]
fn sampled_descriptor_validation() {
    unsafe {
        assert_eq!(std::mem::size_of::<SampledBlock>(), 104);
        let mut block = SampledBlock {
            row_start: 0,
            column_start: 0,
            dim: 1,
            basis_rows: 1,
            basis_cols: 1,
            basis: arr(&[1.0]),
            weights: arr(&[-1.0]),
        };
        let parsed = sampled_blocks::<f64>(std::slice::from_ref(&block)).unwrap();
        assert_eq!(parsed[0].basis, vec![1.0]);
        assert_eq!(parsed[0].weights, vec![-1.0]);
        block.weights.count = 2;
        assert!(sampled_blocks::<f64>(std::slice::from_ref(&block)).is_err());
        block.weights.count = 1;
        block.dim = u64::MAX;
        assert!(sampled_blocks::<f64>(std::slice::from_ref(&block)).is_err());
        block.dim = 1;
        block.basis_cols = 0;
        block.basis = arr(&[]);
        block.weights = arr(&[]);
        let empty = sampled_blocks::<f64>(std::slice::from_ref(&block)).unwrap();
        assert!(empty[0].basis.is_empty());
        assert!(empty[0].weights.is_empty());
        let mut out = 1usize as *mut Handle;
        assert_eq!(
            sdpx_prepare_sampled(
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                1,
                &mut out
            ),
            1
        );
        assert!(out.is_null());
    }
}
#[test]
fn sampled_public_abi_scalar_psd() {
    unsafe {
        for bits in [53, 512] {
            let mut settings = defaults();
            settings.precision_bits = bits;
            let p = Csc {
                rows: 1,
                cols: 1,
                nnz: 0,
                colptr: [0, 0].as_ptr(),
                rowval: ptr::null(),
                values: arr(&[]),
            };
            let a = Csc {
                rows: 1,
                cols: 1,
                nnz: 0,
                colptr: [0, 0].as_ptr(),
                rowval: ptr::null(),
                values: arr(&[]),
            };
            let cone = Cone {
                kind: 3,
                reserved: 0,
                dim: 1,
                alpha: arr(&[]),
            };
            let block = SampledBlock {
                row_start: 0,
                column_start: 0,
                dim: 1,
                basis_rows: 1,
                basis_cols: 1,
                basis: arr(&[1.0]),
                weights: arr(&[-1.0]),
            };
            let mut h = ptr::null_mut();
            assert_eq!(
                sdpx_prepare_sampled(
                    &p,
                    &arr(&[1.0]),
                    &a,
                    &arr(&[-1.0]),
                    &cone,
                    1,
                    &settings,
                    &block,
                    1,
                    &mut h
                ),
                0
            );
            assert!(!h.is_null());
            assert_eq!(sdpx_solve(h), 0);
            let mut info = std::mem::MaybeUninit::<Info>::uninit();
            assert_eq!(sdpx_get_info(h, info.as_mut_ptr()), 0);
            let info = info.assume_init();
            assert_eq!(info.status, 1);
            assert_eq!(info.working_bits, bits);
            // High precision is checked without f64 conversion in Julia tests.
            assert!((info.objective - 1.0).abs() <= 2e-8);
            assert_eq!(sdpx_destroy(h), 0);
        }
    }
}
#[test]
fn null_and_settings_errors() {
    unsafe {
        assert_eq!(sdpx_solve(ptr::null_mut()), 1);
        assert_eq!(sdpx_default_settings(ptr::null_mut()), 1);
        assert_eq!(sdpx_destroy(ptr::null_mut()), 0);
        let mut s = std::mem::MaybeUninit::<Settings>::uninit();
        assert_eq!(sdpx_default_settings(s.as_mut_ptr()), 0);
        let s = s.assume_init();
        assert_eq!(s.abi_version, 4);
        assert_eq!(s.preprocessing_flags, 7);
        assert_eq!(s.kkt_form, 0);
        assert_eq!(std::mem::size_of::<Settings>(), 88);
        assert_eq!(std::mem::size_of::<Info>(), 112);
    }
}
#[test]
fn abi_version_and_kkt_form_validation() {
    unsafe {
        let mut s = defaults();
        for version in [0, 1, 2, 3, 5] {
            s.abi_version = version;
            let error = settings::<f64>(&s).err().unwrap();
            assert_eq!(error.0, 1);
            assert!(error.1.contains("requires ABI 4"));
        }
        s.abi_version = ABI_VERSION;
        s.kkt_form = 3;
        assert!(settings::<f64>(&s).is_err());
        for (code, name) in [(0, "auto"), (1, "augmented"), (2, "condensed")] {
            s.kkt_form = code;
            assert_eq!(settings::<f64>(&s).unwrap().kkt_form, name);
        }
        s.reserved_0 = 2;
        assert!(settings::<f64>(&s).is_err());
        s.reserved_0 = 1;
        assert!(settings::<f64>(&s).is_err());
        s.reserved_0 = 0;
        s.max_threads = 4;
        let f64_settings = settings::<f64>(&s).unwrap();
        assert_eq!(f64_settings.max_threads, 4);
        assert_eq!(f64_settings.direct_solve_method, "auto");
        s.precision_bits = 128;
        let hp_settings = settings::<MpFloat<2>>(&s).unwrap();
        assert_eq!(hp_settings.max_threads, 4);
        assert_eq!(hp_settings.direct_solve_method, "auto");
    }
}
fn preprocessing_mapping<T: Scalar>(bits: u32) {
    let core = DefaultSettings::<T>::default();
    assert!(core.equilibrate_enable);
    assert!(core.presolve_enable);
    assert!(core.chordal_decomposition_enable);
    unsafe {
        for flags in 0..=PREPROCESS_ALL {
            let mut s = defaults();
            s.precision_bits = bits;
            s.preprocessing_flags = flags;
            let mapped = settings::<T>(&s).unwrap();
            assert_eq!(mapped.equilibrate_enable, flags & PREPROCESS_RUIZ != 0);
            assert_eq!(mapped.presolve_enable, flags & PREPROCESS_PRESOLVE != 0);
            assert_eq!(
                mapped.chordal_decomposition_enable,
                flags & PREPROCESS_CHORDAL != 0
            );
            assert!(!mapped.input_sparse_dropzeros);
            assert_eq!(mapped.tol_feas, core.tol_feas);
            assert_eq!(
                mapped.static_regularization_constant,
                core.static_regularization_constant
            );
            assert_eq!(mapped.max_threads, 1);
        }
    }
}

#[test]
fn preprocessing_flags_map_all_modes() {
    preprocessing_mapping::<f64>(53);
    preprocessing_mapping::<MpFloat<2>>(128);
    preprocessing_mapping::<MpFloat<4>>(256);
    preprocessing_mapping::<MpFloat<8>>(512);
    preprocessing_mapping::<MpFloat<12>>(768);
    preprocessing_mapping::<MpFloat<16>>(1024);
    preprocessing_mapping::<MpFloat<32>>(2048);
}

#[test]
fn preprocessing_unknown_bits_are_rejected() {
    unsafe {
        for flags in [8, PREPROCESS_ALL | 8, 1 << 31, u32::MAX] {
            let mut s = defaults();
            s.preprocessing_flags = flags;
            assert_eq!(settings::<f64>(&s).err().unwrap().0, 1);
            s.precision_bits = 512;
            assert_eq!(settings::<MpFloat<8>>(&s).err().unwrap().0, 1);
        }
    }
}

#[test]
fn prepared_solver_receives_preprocessing_flags() {
    unsafe {
        for bits in [53, 512] {
            for flags in 0..=PREPROCESS_ALL {
                let mut s = defaults();
                s.precision_bits = bits;
                s.preprocessing_flags = flags;
                let h = lp_settings(s);
                operate(h, |engine| {
                    dispatch!(engine, typed, {
                        let actual = typed.solver.settings();
                        assert_eq!(actual.equilibrate_enable, flags & PREPROCESS_RUIZ != 0);
                        assert_eq!(actual.presolve_enable, flags & PREPROCESS_PRESOLVE != 0);
                        assert_eq!(
                            actual.chordal_decomposition_enable,
                            flags & PREPROCESS_CHORDAL != 0
                        );
                        assert!(!actual.input_sparse_dropzeros);
                        Ok(())
                    })
                })
                .unwrap();
                assert_eq!(sdpx_destroy(h), 0);
            }
        }
    }
}

#[test]
fn solver_name_query_and_actual_info() {
    unsafe {
        let h = lp();
        let mut n = 0;
        assert_eq!(sdpx_get_solver_name(h, ptr::null_mut(), 0, &mut n), 0);
        assert_eq!(n, 6); // qdldl plus NUL
        let mut short = [0x55_u8; 2];
        assert_eq!(
            sdpx_get_solver_name(h, short.as_mut_ptr().cast(), 2, &mut n),
            6
        );
        assert_eq!(short, [0x55; 2]);
        let mut name = vec![0_u8; n as usize];
        assert_eq!(
            sdpx_get_solver_name(h, name.as_mut_ptr().cast(), n, &mut n),
            0
        );
        assert_eq!(name, b"qdldl\0");
        assert_eq!(
            sdpx_get_solver_name(h, ptr::null_mut(), 0, ptr::null_mut()),
            1
        );
        {
            let _guard = (*h).state.lock().unwrap();
            assert_eq!(sdpx_get_solver_name(h, ptr::null_mut(), 0, &mut n), 3);
        }
        let mut info = std::mem::MaybeUninit::<Info>::uninit();
        assert_eq!(sdpx_get_info(h, info.as_mut_ptr()), 0);
        let info = info.assume_init();
        assert_eq!(info.abi_version, 4);
        assert_eq!(info.kkt_form, 1);
        assert_eq!(info.backend_threads, 1);
        assert_eq!(info.cone_threads, 1);
        assert_eq!(sdpx_destroy(h), 0);
    }
}
#[test]
fn reject_unsorted_csc() {
    unsafe {
        let c = Csc {
            rows: 2,
            cols: 1,
            nnz: 2,
            colptr: [0, 2].as_ptr(),
            rowval: [1, 0].as_ptr(),
            values: arr(&[1.0, 1.0]),
        };
        assert!(matrix::<f64>(&c, false).is_err());
    }
}
#[test]
fn zero_time_limit_returns_max_time() {
    unsafe {
        let mut settings = defaults();
        settings.time_limit = 0.0;
        let h = lp_settings(settings);
        assert_eq!(sdpx_solve(h), 0);
        let mut info = std::mem::MaybeUninit::<Info>::uninit();
        assert_eq!(sdpx_get_info(h, info.as_mut_ptr()), 0);
        assert_eq!(info.assume_init().status, 8);
        assert_eq!(sdpx_destroy(h), 0);
    }
}
#[test]
fn lifecycle_and_update() {
    unsafe {
        // Structural preprocessing can invalidate reusable updates. Keep
        // Ruiz enabled while explicitly disabling presolve and chordal.
        let mut settings = defaults();
        settings.preprocessing_flags = PREPROCESS_RUIZ;
        let h = lp_settings(settings);
        let mut n = 0;
        assert_eq!(sdpx_result_f64(h, ptr::null_mut(), 0, &mut n), 7);
        assert_eq!(sdpx_solve(h), 0);
        assert_eq!(sdpx_result_f64(h, ptr::null_mut(), 0, &mut n), 0);
        let mut v = vec![0.0; n as usize];
        assert_eq!(sdpx_result_f64(h, v.as_mut_ptr(), n, &mut n), 0);
        assert!((v[0] - 1.0).abs() < 1e-7);
        assert_eq!(sdpx_update(h, &arr(&[f64::NAN]), &arr(&[-2.0])), 1);
        assert_eq!(sdpx_result_f64(h, ptr::null_mut(), 0, &mut n), 0);
        assert_eq!(sdpx_update(h, &arr(&[1.0]), &arr(&[-2.0])), 0);
        assert_eq!(sdpx_result_f64(h, ptr::null_mut(), 0, &mut n), 7);
        assert_eq!(sdpx_solve(h), 0);
        assert_eq!(sdpx_result_f64(h, v.as_mut_ptr(), n, &mut n), 0);
        assert!((v[0] - 2.0).abs() < 1e-7);
        assert_eq!(sdpx_destroy(h), 0);
    }
}
#[test]
fn panic_poisons_and_busy_rejects() {
    unsafe {
        let h = lp();
        {
            let _guard = (*h).state.lock().unwrap();
            assert_eq!(sdpx_solve(h), 3);
        }
        assert_eq!(boundary(|| operate(h, |_| panic!("test"))), 5);
        assert_eq!(sdpx_solve(h), 4);
        assert_eq!(sdpx_destroy(h), 0);
    }
}
