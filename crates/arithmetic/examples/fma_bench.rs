// Quick microbench: dot_fma cost per FMA at 512-bit
use sdpx_arithmetic::{Bits512, Scalar};
use num_traits::{FromPrimitive, ToPrimitive};
use std::time::Instant;

fn main() {
    let n = 200000usize;
    let a: Vec<Bits512> = (0..n).map(|i| Bits512::from_f64(((i * 2654435761) % 1000000) as f64 * 1e-6 + 0.5).unwrap()).collect();
    let b: Vec<Bits512> = (0..n).map(|i| Bits512::from_f64(((i * 40503 + 7) % 999983) as f64 * 1e-6 - 0.3).unwrap()).collect();
    // warmup
    let mut s = Bits512::dot_fma(a.iter().take(1000).zip(b.iter().take(1000)));
    let t0 = Instant::now();
    for _ in 0..5 {
        s = s + Bits512::dot_fma(a.iter().zip(b.iter()));
    }
    let dt = t0.elapsed();
    println!("{:?} total, {:.1} ns/fma, sink={:?}", dt, dt.as_nanos() as f64 / (n * 5) as f64, s.to_f64());
    // raw mul cost
    let t0 = Instant::now();
    for _ in 0..5 {
        for i in 0..n { s = s + a[i] * b[i]; }
    }
    let dt = t0.elapsed();
    println!("mul+add {:.1} ns/op-pair", dt.as_nanos() as f64 / (n * 5) as f64);
    println!("sink {:?}", s.to_f64());
}
