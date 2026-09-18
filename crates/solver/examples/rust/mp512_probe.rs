#![allow(non_snake_case)]
use sdpx_solver::algebra::*;
use sdpx_solver::solver::*;

fn main() {
    // same 2x2 SOCP as example_socp.rs, but in 512-bit arithmetic
    let P = CscMatrix::<Bits512>::from(&[
        [Bits512::from(0.), Bits512::from(0.)],
        [Bits512::from(0.), Bits512::from(2.)],
    ]);
    let q = vec![Bits512::from(0.), Bits512::from(0.)];
    let A = CscMatrix::<Bits512>::from(&[
        [Bits512::from(0.), Bits512::from(0.)],
        [Bits512::from(-2.), Bits512::from(0.)],
        [Bits512::from(0.), Bits512::from(-1.)],
    ]);
    let b = vec![Bits512::from(1.), Bits512::from(-2.), Bits512::from(-2.)];
    let cones = [SecondOrderConeT(3)];
    let settings = DefaultSettings::<Bits512>::default();
    let mut solver = DefaultSolver::<Bits512>::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();
    println!("status = {:?}", solver.solution.status);
    for x in solver.solution.x.iter() { println!("  x = {}", x); }
}
