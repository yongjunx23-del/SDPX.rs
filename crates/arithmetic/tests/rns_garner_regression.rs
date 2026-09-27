// Add to crates/arithmetic/tests/rns_garner_regression.rs in an experiment worktree.
// Tests target the pinned source d79fae0827e990a637fc141f1f0d99a067e13e1b.
use num_traits::One;
use sdpx_arithmetic::{EncodeSide, MpFloat, RnsPlan};

#[test]
fn garner_descending_modulus_boundary() {
    type F = MpFloat<2>;
    let p = [
        1152921504606846883u64,
        1152921504606846869,
        1152921504606846803,
        1152921504606846797,
        1152921504606846719,
    ];
    let s = 94944856841779688819459117394129014u128;
    // shift_a=shift_b=0: the reconstructed integer is the output value itself.
    let plan = RnsPlan::for_ranges::<2>((0, 0), (0, 0), 1).unwrap();
    assert_eq!(plan.primes(), p.len());
    let residues: Vec<u64> = p.iter().map(|&q| (s % q as u128) as u64).collect();
    assert_eq!(residues[0], p[0] - 1);
    assert_eq!(residues[1], 0);
    let expected: F = s.to_string().parse().unwrap();
    assert_eq!(plan.reconstruct::<2>(&residues), expected);
    let neg: Vec<u64> = residues
        .iter()
        .zip(p)
        .map(|(&r, q)| if r == 0 { 0 } else { q - r })
        .collect();
    assert_eq!(plan.reconstruct::<2>(&neg), -expected);
}

#[test]
fn real_encoded_dot_hits_same_boundary() {
    type F = MpFloat<2>;
    // Both are exactly representable 128-bit integers, so their encoding shifts
    // are zero. Their product is -1 mod p0 and 0 mod p1.
    let a: F = "170154311670604377805668618416223575362".parse().unwrap();
    let b: F = "170141183460469231731687303715884105728".parse().unwrap();
    let aa = [a];
    let bb = [b];
    let plan = RnsPlan::for_pair(&aa, &bb, 1).unwrap();
    let ra = plan.encode(&aa, EncodeSide::A).unwrap();
    let rb = plan.encode(&bb, EncodeSide::B).unwrap();
    assert_eq!(plan.dot::<2>(&ra, 0, 1, &rb, 0, 1, 1), a * b);
    // A harmless ordinary case remains covered too.
    let one = [F::one()];
    let plan = RnsPlan::for_pair(&one, &one, 1).unwrap();
    let r = plan.encode(&one, EncodeSide::A).unwrap();
    assert_eq!(plan.dot::<2>(&r, 0, 1, &r, 0, 1, 1), F::one());
}
