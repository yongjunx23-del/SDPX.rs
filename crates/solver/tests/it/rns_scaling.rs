//! Thread scaling of one exact residue-kernel product (benchmark, ignored by
//! default): `cargo test --release --features sdp,... --test rns_scaling --
//! --ignored --nocapture`.
#[allow(dead_code)]
#[path = "../../src/algebra/dense/blas/traits.rs"]
mod provider;
use num_traits::{FromPrimitive, One};
use provider::*;
use sdpx_arithmetic::MpFloat;
use std::time::Instant;

type F = MpFloat<12>;

// Full-mantissa values in (-1, 1): ratios of small integers.
fn value(i: usize) -> F {
    let num = F::from_i64((i as i64 * 7919) % 1999 - 999).unwrap();
    num / F::from_i64(1000 + (i as i64 % 37)).unwrap()
        + F::one() / F::from_i64(3 + i as i64).unwrap()
}

#[test]
#[ignore]
fn rns_congruence_thread_scaling() {
    for n in [45usize, 53] {
        let a: Vec<F> = (0..n * n).map(value).collect();
        let mut x: Vec<F> = (0..n * n).map(|i| value(i + 17)).collect();
        for j in 0..n {
            for i in 0..j {
                x[j + i * n] = x[i + j * n];
            }
        }
        let mut c = vec![F::from_i64(0).unwrap(); n * n];
        for threads in [1usize, 2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            assert!(F::xcongruence_exact(
                b'N',
                n,
                n,
                &a,
                n,
                &x,
                n,
                &mut c,
                true,
                Some(&pool),
                None
            ));
            let reps = 20;
            let t = Instant::now();
            for _ in 0..reps {
                F::xcongruence_exact(b'N', n, n, &a, n, &x, n, &mut c, true, Some(&pool), None);
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            eprintln!("RNS congruence n={n} threads={threads} {ms:.3} ms");
        }
    }
}
