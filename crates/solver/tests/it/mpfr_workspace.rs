//! Caller-owned MPFR decomposition scratch: queries, refresh, and Rust allocations.
// This test imports the whole provider but exercises only selected operations.
#[allow(dead_code)]
#[path = "../../src/algebra/dense/blas/traits.rs"]
mod provider;
use num_traits::{FromPrimitive, One, ToPrimitive, Zero};
use provider::{XgesvdScalar, XsyevrScalar};
use sdpx_arithmetic::{MpFloat, Scalar};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

// Count this test thread only: included provider tests run concurrently.
// Count Rust scratch allocations;
// MPFR's own native arithmetic allocation is outside this counter's scope.
struct Allocator;
thread_local! {
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}
fn count_allocation() {
    let _ = ALLOCATIONS.try_with(|count| {
        if let Some(n) = count.get() {
            count.set(Some(n + 1));
        }
    });
}
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count_allocation();
        System.realloc(ptr, layout, size)
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;
fn without_rust_allocations(f: impl FnOnce()) {
    ALLOCATIONS.with(|count| count.set(Some(0)));
    f();
    let allocations = ALLOCATIONS.with(|count| count.replace(None));
    assert_eq!(allocations, Some(0));
}
type F<const N: usize> = MpFloat<N>;
fn f<const N: usize>(i: usize) -> F<N> {
    F::from_usize(i).unwrap()
}

fn run<const N: usize>() {
    let (zero, one, pad) = (F::<N>::zero(), F::<N>::one(), f::<N>(777));
    for (m, n) in [(4usize, 3usize), (3, 4), (0, 0), (0, 3), (3, 0)] {
        let r = m.min(n);
        let (lda, ldu, ldvt) = (m + 2, m + 1, n + 1);
        let mut original = vec![pad; lda * n];
        for j in 0..n {
            for i in 0..m {
                original[i + j * lda] = if i == j { f(i + 2) } else { one / f(i + j + 2) };
            }
        }
        for ju in [b'A', b'S', b'N', b'O'] {
            for jv in [b'A', b'S', b'N', b'O'] {
                if ju == b'O' && jv == b'O' {
                    continue;
                }
                let mut a = original.clone();
                let mut s = vec![pad; r];
                let mut u = vec![pad; ldu * m];
                let mut vt = vec![pad; ldvt * n];
                let mut work = vec![zero];
                let mut info = 0;
                F::xgesvd(
                    ju,
                    jv,
                    m as i32,
                    n as i32,
                    &mut a,
                    lda as i32,
                    &mut s,
                    &mut u,
                    ldu as i32,
                    &mut vt,
                    ldvt as i32,
                    &mut work,
                    -1,
                    &mut info,
                );
                assert_eq!(info, 0);
                assert_eq!(a, original);
                assert!(s.iter().chain(&u).chain(&vt).all(|&v| v == pad));
                let lwork = work[0].to_i32().unwrap();
                assert!(lwork >= 1);
                if lwork > 1 {
                    F::xgesvd(
                        ju,
                        jv,
                        m as i32,
                        n as i32,
                        &mut a,
                        lda as i32,
                        &mut s,
                        &mut u,
                        ldu as i32,
                        &mut vt,
                        ldvt as i32,
                        &mut work,
                        1,
                        &mut info,
                    );
                    assert_eq!(info, -13);
                    assert_eq!(a, original);
                }
                work.resize(lwork as usize, zero);
                F::xgesvd(
                    ju,
                    jv,
                    m as i32,
                    n as i32,
                    &mut a,
                    lda as i32,
                    &mut s,
                    &mut u,
                    ldu as i32,
                    &mut vt,
                    ldvt as i32,
                    &mut work,
                    lwork,
                    &mut info,
                );
                assert_eq!(info, 0);
                let expected = (a.clone(), s.clone(), u.clone(), vt.clone());
                let identity = (work.as_ptr(), work.capacity());
                a.copy_from_slice(&original);
                s.fill(pad);
                u.fill(pad);
                vt.fill(pad);
                work.fill(F::nan());
                without_rust_allocations(|| {
                    F::xgesvd(
                        ju,
                        jv,
                        m as i32,
                        n as i32,
                        &mut a,
                        lda as i32,
                        &mut s,
                        &mut u,
                        ldu as i32,
                        &mut vt,
                        ldvt as i32,
                        &mut work,
                        lwork,
                        &mut info,
                    );
                });
                assert_eq!(info, 0);
                assert_eq!((a, s, u, vt), expected);
                assert_eq!((work.as_ptr(), work.capacity()), identity);
                assert_eq!(work[0].to_i32(), Some(lwork));
            }
        }
    }
    // Reuse the same buffers while toggling vector requests and selected ranges.
    let mut work = vec![zero];
    let mut iwork = vec![0];
    for job in [b'V', b'N', b'V'] {
        for range in [b'A', b'I', b'V'] {
            let n = 4;
            let mut a = vec![pad; 6 * n];
            for j in 0..n {
                for i in 0..=j {
                    a[i + j * 6] = if i == j { f(i + 1) } else { zero };
                }
            }
            let original = a.clone();
            let mut w = vec![pad; n];
            let mut z = vec![pad; 6 * n];
            let mut support = vec![-1; 2 * n];
            let (mut count, mut info) = (0, 0);
            F::xsyevr(
                job,
                range,
                b'U',
                4,
                &mut a,
                6,
                one,
                f(3),
                2,
                3,
                zero,
                &mut count,
                &mut w,
                &mut z,
                6,
                &mut support,
                &mut work,
                -1,
                &mut iwork,
                -1,
                &mut info,
            );
            assert_eq!(info, 0);
            assert_eq!(a, original);
            let (lwork, liwork) = (work[0].to_i32().unwrap(), iwork[0]);
            work.resize(lwork as usize, zero);
            iwork.resize(liwork as usize, 0);
            let identity = (
                work.as_ptr(),
                work.capacity(),
                iwork.as_ptr(),
                iwork.capacity(),
            );
            work.fill(F::nan());
            iwork.fill(-1);
            without_rust_allocations(|| {
                F::xsyevr(
                    job,
                    range,
                    b'U',
                    4,
                    &mut a,
                    6,
                    one,
                    f(3),
                    2,
                    3,
                    zero,
                    &mut count,
                    &mut w,
                    &mut z,
                    6,
                    &mut support,
                    &mut work,
                    lwork,
                    &mut iwork,
                    liwork,
                    &mut info,
                );
            });
            assert_eq!(info, 0);
            assert_eq!(a, original);
            let expected = if range == b'A' {
                vec![one, f(2), f(3), f(4)]
            } else {
                vec![f(2), f(3)]
            };
            assert_eq!(&w[..count as usize], expected);
            for j in 0..count as usize {
                for i in 0..6 {
                    if job == b'N' || i >= 4 {
                        assert_eq!(z[i + j * 6], pad);
                    } else {
                        assert_eq!(f::<N>(i + 1) * z[i + j * 6], w[j] * z[i + j * 6]);
                    }
                }
            }
            assert_eq!(
                (
                    work.as_ptr(),
                    work.capacity(),
                    iwork.as_ptr(),
                    iwork.capacity()
                ),
                identity
            );
            assert_eq!(work[0].to_i32(), Some(lwork));
            assert_eq!(iwork[0], liwork);
        }
    }
}

#[test]
fn all_precisions_reuse_decomposition_workspace() {
    run::<2>();
    run::<4>();
    run::<8>();
    run::<12>();
    run::<16>();
    run::<32>();
}
