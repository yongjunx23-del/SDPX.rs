//! Disposable private test module: call the production MPFR BLAS provider.
use crate::algebra::{XgemmScalar, XsyrkScalar};
use sdpx_arithmetic::MpFloat;
use std::{
    fs::OpenOptions,
    io::Write,
    path::Path,
    time::{Duration, Instant},
};
type F<const N: usize> = MpFloat<N>;
fn integer<const N: usize>(v: usize) -> F<N> {
    v.to_string().parse().unwrap()
}
fn dump<const N: usize>(path: &Path, values: &[F<N>]) {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    for value in values {
        writeln!(file, "{}", value.to_decimal(None)).unwrap();
    }
}
fn call<const N: usize>(
    op: usize,
    n: usize,
    a: &[F<N>],
    b: &[F<N>],
    c: &mut [F<N>],
    constants: &[F<N>; 4],
) {
    let [one, zero, alpha, beta] = *constants;
    let d = n as i32;
    match op {
        0 => F::<N>::xgemm(b'N', b'N', d, d, d, one, a, d, b, d, zero, c, d),
        1 => F::<N>::xgemm(b'N', b'T', d, d, d, one, a, d, b, d, zero, c, d),
        2 => F::<N>::xsyrk(b'U', b'N', d, d, one, a, d, zero, c, d),
        3 => F::<N>::xsyrk(b'L', b'T', d, d, one, a, d, zero, c, d),
        4 => F::<N>::xgemm(b'T', b'N', d, d, d, alpha, a, d, b, d, beta, c, d),
        _ => unreachable!(),
    }
}
fn precision<const N: usize>(out: &Path, repeats: usize) {
    for n in [12, 16, 64] {
        let setup = Instant::now();
        // Non-dyadic rational inputs are rounded directly at N*64 bits. No
        // binary64 input or arithmetic intermediate is involved.
        let a: Vec<_> = (0..n * n)
            .map(|i| integer::<N>((i * 17 + 3) % 101 + 1) / integer::<N>(103))
            .collect();
        let b: Vec<_> = (0..n * n)
            .map(|i| integer::<N>((i * 29 + 7) % 107 + 1) / integer::<N>(109))
            .collect();
        let seed: Vec<_> = (0..n * n)
            .map(|i| integer::<N>((i * 11 + 5) % 31 + 1) / integer::<N>(37))
            .collect();
        let constants = [
            integer::<N>(1),
            integer::<N>(0),
            integer::<N>(3) / integer::<N>(7),
            integer::<N>(2) / integer::<N>(11),
        ];
        let mut c = seed.clone();
        let setup_ns = setup.elapsed().as_nanos();
        let prefix = format!("p{}_n{}", N * 64, n);
        dump(&out.join(format!("{prefix}_a.txt")), &a);
        dump(&out.join(format!("{prefix}_b.txt")), &b);
        dump(&out.join(format!("{prefix}_seed.txt")), &seed);
        for (op, name) in [
            "gemm_nn",
            "gemm_nt",
            "syrk_un",
            "syrk_lt",
            "gemm_tn_general",
        ]
        .iter()
        .enumerate()
        {
            c.copy_from_slice(&seed);
            let start = Instant::now();
            call(op, n, &a, &b, &mut c, &constants);
            let first = start.elapsed().as_nanos();
            let expected = c.clone();
            // Sub-millisecond samples can mostly measure transient scheduling
            // and CPU frequency. Use a fixed amount of arithmetic per sample
            // and warm this exact kernel before collecting measured calls.
            // Both candidate and baseline use the same batch size.
            let batch = (64usize.pow(3)).div_ceil(n.pow(3));
            let warmup = Instant::now();
            while warmup.elapsed() < Duration::from_millis(150) {
                c.copy_from_slice(&seed);
                call(op, n, &a, &b, &mut c, &constants);
                std::hint::black_box(&c);
            }
            let mut times = Vec::with_capacity(repeats);
            let mut batch_times = Vec::with_capacity(repeats);
            for _ in 0..repeats {
                let mut elapsed = 0;
                for _ in 0..batch {
                    c.copy_from_slice(&seed); // reset outside every call's timer
                    let start = Instant::now();
                    call(op, n, &a, &b, &mut c, &constants);
                    elapsed += start.elapsed().as_nanos();
                    std::hint::black_box(&c);
                }
                batch_times.push(elapsed);
                times.push(elapsed / batch as u128);
                assert_eq!(c, expected, "non-deterministic provider output");
                std::hint::black_box(&c);
            }
            dump(&out.join(format!("{prefix}_{name}_output.txt")), &expected);
            let raw = times
                .iter()
                .map(u128::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let raw_batches = batch_times
                .iter()
                .map(u128::to_string)
                .collect::<Vec<_>>()
                .join(",");
            times.sort_unstable();
            println!("MPFR_DENSE {{\"bits\":{},\"n\":{n},\"kernel\":\"{name}\",\"setup_ns\":{setup_ns},\"first_ns\":{first},\"batch\":{batch},\"warm_batch_ns\":[{raw_batches}],\"warm_ns\":[{raw}],\"median_ns\":{}}}",N*64,times[times.len()/2]);
        }
    }
}
#[test]
#[ignore = "explicit microbenchmark only; never part of normal acceptance"]
fn dense_microbenchmark() {
    let out =
        std::env::var("SDPX_MPFR_OUTPUT").expect("explicit external output directory required");
    let out = Path::new(&out);
    assert!(out.is_dir());
    let repeats: usize = std::env::var("SDPX_MPFR_REPEATS")
        .unwrap_or_else(|_| "9".into())
        .parse()
        .unwrap();
    assert!(
        repeats >= 7 && repeats % 2 == 1,
        "use an odd repetition count >=7"
    );
    precision::<2>(out, repeats);
    precision::<4>(out, repeats);
    precision::<8>(out, repeats);
    precision::<12>(out, repeats);
    precision::<16>(out, repeats);
    precision::<32>(out, repeats);
}
