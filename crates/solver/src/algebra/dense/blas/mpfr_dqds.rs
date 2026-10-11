// Singular values of an upper bidiagonal by dqds: a port of LAPACK 3.12
// reference `dlasq1`–`dlasq6` and `dlas2` (BSD licensed, Univ. of Tennessee,
// Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd.).
// The control flow, shift strategy and constants follow the Fortran; the
// qd array keeps its 1-based layout. Differences: MPFR's exponent range makes
// `dlasq1`'s safe-minimum scaling unnecessary, so it is dropped and the
// `SAFMIN` guards reduce to positivity tests; `dlasq5` uses the IEEE
// branch's one-division update with the non-IEEE branch's stop at the first
// negative pivot (a failed shift then costs a partial transform); every
// LAPACK failure (`INFO` ≠ 0) and any non-finite result is an error, for the
// caller's QR fallback.
use super::*;

struct Z<'a, const N: usize>(&'a mut Vec<F<N>>);
impl<const N: usize> std::ops::Index<isize> for Z<'_, N> {
    type Output = F<N>;
    fn index(&self, i: isize) -> &F<N> {
        &self.0[i as usize]
    }
}
impl<const N: usize> std::ops::IndexMut<isize> for Z<'_, N> {
    fn index_mut(&mut self, i: isize) -> &mut F<N> {
        &mut self.0[i as usize]
    }
}
impl<const N: usize> Z<'_, N> {
    fn swap(&mut self, a: isize, b: isize) {
        self.0.swap(a as usize, b as usize);
    }
}

fn c<const N: usize>(x: f64) -> F<N> {
    <F<N> as num_traits::FromPrimitive>::from_f64(x).unwrap()
}
fn min<const N: usize>(a: F<N>, b: F<N>) -> F<N> {
    if b < a {
        b
    } else {
        a
    }
}
fn max<const N: usize>(a: F<N>, b: F<N>) -> F<N> {
    if b > a {
        b
    } else {
        a
    }
}

/// Shift state carried between `dlasq3` calls.
struct State<const N: usize> {
    dmin: F<N>,
    sigma: F<N>,
    desig: F<N>,
    qmax: F<N>,
    nfail: usize,
    iter: usize,
    ndiv: isize,
    ttype: i32,
    dmin1: F<N>,
    dmin2: F<N>,
    dn: F<N>,
    dn1: F<N>,
    dn2: F<N>,
    g: F<N>,
    tau: F<N>,
}

/// Singular values of the upper bidiagonal `(d, e)` (orders `n`, `n − 1`),
/// returned in `d` in descending order (LAPACK `dlasq1`). `work` is reused
/// scratch for the qd array.
pub(super) fn dqds_values<const N: usize>(
    d: &mut [F<N>],
    e: &[F<N>],
    n: usize,
    work: &mut Vec<F<N>>,
) -> Result<(), i32> {
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        d[0] = d[0].abs();
        return Ok(());
    }
    if n == 2 {
        let (small, large) = las2(d[0], e[0], d[1]);
        d[0] = large;
        d[1] = small;
        return Ok(());
    }
    let mut sigmx = F::zero();
    for i in 0..n - 1 {
        d[i] = d[i].abs();
        sigmx = max(sigmx, e[i].abs());
    }
    d[n - 1] = d[n - 1].abs();
    if sigmx == F::zero() {
        d[..n].sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        return Ok(());
    }
    work.clear();
    work.resize(4 * n + 6, F::zero());
    let mut z = Z(work);
    for i in 0..n {
        z.0[2 * i + 1] = d[i] * d[i];
        if i + 1 < n {
            z.0[2 * i + 2] = e[i] * e[i];
        }
    }
    z.0[2 * n] = F::zero();
    lasq2(n as isize, &mut z)?;
    for i in 0..n {
        let q = z.0[i + 1];
        if !q.is_finite() || q < F::zero() {
            return Err(4);
        }
        d[i] = q.sqrt();
    }
    Ok(())
}

/// `dlas2`: singular values `(small, large)` of `[f g; 0 h]`.
fn las2<const N: usize>(f: F<N>, g: F<N>, h: F<N>) -> (F<N>, F<N>) {
    let (one, two) = (F::<N>::one(), c::<N>(2.0));
    let (fa, ga, ha) = (f.abs(), g.abs(), h.abs());
    let (fhmn, fhmx) = (min(fa, ha), max(fa, ha));
    if fhmn == F::zero() {
        let large = if fhmx == F::zero() {
            ga
        } else {
            let (lo, hi) = (min(fhmx, ga), max(fhmx, ga));
            let r = lo / hi;
            hi * (one + r * r).sqrt()
        };
        (F::zero(), large)
    } else if ga < fhmx {
        let as_ = one + fhmn / fhmx;
        let at = (fhmx - fhmn) / fhmx;
        let au = (ga / fhmx) * (ga / fhmx);
        let cc = two / ((as_ * as_ + au).sqrt() + (at * at + au).sqrt());
        (fhmn * cc, fhmx / cc)
    } else {
        let au = fhmx / ga;
        if au == F::zero() {
            ((fhmn * fhmx) / ga, ga)
        } else {
            let as_ = one + fhmn / fhmx;
            let at = (fhmx - fhmn) / fhmx;
            let cc = one / ((one + (as_ * au) * (as_ * au)).sqrt() + (one + (at * au) * (at * au)).sqrt());
            let small = (fhmn * cc) * au;
            (small + small, ga / (cc + cc))
        }
    }
}

/// `dlasq2` for `n ≥ 3` on the squared, interleaved qd array.
fn lasq2<const N: usize>(n: isize, z: &mut Z<'_, N>) -> Result<(), i32> {
    let zero = F::<N>::zero();
    let (half, four, cbias) = (c::<N>(0.5), c::<N>(4.0), c::<N>(1.5));
    let eps = F::<N>::epsilon();
    let tol = eps * c::<N>(100.0);
    let tol2 = tol * tol;
    z[2 * n] = zero;
    let (mut d, mut e) = (zero, zero);
    let mut k = 1;
    while k <= 2 * (n - 1) {
        if z[k] < zero || z[k + 1] < zero {
            return Err(-1);
        }
        d += z[k];
        e += z[k + 1];
        k += 2;
    }
    if z[2 * n - 1] < zero {
        return Err(-1);
    }
    d += z[2 * n - 1];
    if e == zero {
        for k in 2..=n {
            z[k] = z[2 * k - 1];
        }
        z.0[1..=n as usize].sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        return Ok(());
    }
    let trace = d + e;
    if trace == zero {
        return Ok(());
    }
    let mut k = 2 * n;
    while k >= 2 {
        z[2 * k] = zero;
        z[2 * k - 1] = z[k];
        z[2 * k - 2] = zero;
        z[2 * k - 3] = z[k - 1];
        k -= 2;
    }
    let mut i0: isize = 1;
    let mut n0: isize = n;
    if cbias * z[4 * i0 - 3] < z[4 * n0 - 3] {
        let ipn4 = 4 * (i0 + n0);
        let mut i4 = 4 * i0;
        while i4 <= 2 * (i0 + n0 - 1) {
            z.swap(i4 - 3, ipn4 - i4 - 3);
            z.swap(i4 - 1, ipn4 - i4 - 5);
            i4 += 4;
        }
    }
    let mut pp: isize = 0;
    for _ in 0..2 {
        d = z[4 * n0 + pp - 3];
        let mut i4 = 4 * (n0 - 1) + pp;
        while i4 >= 4 * i0 + pp {
            if z[i4 - 1] <= tol2 * d {
                z[i4 - 1] = -zero;
                d = z[i4 - 3];
            } else {
                d = z[i4 - 3] * (d / (d + z[i4 - 1]));
            }
            i4 -= 4;
        }
        d = z[4 * i0 + pp - 3];
        let mut i4 = 4 * i0 + pp;
        while i4 <= 4 * (n0 - 1) + pp {
            z[i4 - 2 * pp - 2] = d + z[i4 - 1];
            if z[i4 - 1] <= tol2 * d {
                z[i4 - 1] = -zero;
                z[i4 - 2 * pp - 2] = d;
                z[i4 - 2 * pp] = zero;
                d = z[i4 + 1];
            } else if zero < z[i4 + 1] && zero < z[i4 - 2 * pp - 2] {
                let temp = z[i4 + 1] / z[i4 - 2 * pp - 2];
                z[i4 - 2 * pp] = z[i4 - 1] * temp;
                d *= temp;
            } else {
                z[i4 - 2 * pp] = z[i4 + 1] * (z[i4 - 1] / z[i4 - 2 * pp - 2]);
                d = z[i4 + 1] * (d / z[i4 - 2 * pp - 2]);
            }
            i4 += 4;
        }
        z[4 * n0 - pp - 2] = d;
        pp = 1 - pp;
    }
    let mut st = State {
        dmin: zero,
        sigma: zero,
        desig: zero,
        qmax: zero,
        nfail: 0,
        iter: 2,
        ndiv: 2 * (n0 - i0),
        ttype: 0,
        dmin1: zero,
        dmin2: zero,
        dn: zero,
        dn1: zero,
        dn2: zero,
        g: zero,
        tau: zero,
    };
    for _ in 1..=n + 1 {
        if n0 < 1 {
            break;
        }
        st.desig = zero;
        st.sigma = if n0 == n { zero } else { -z[4 * n0 - 1] };
        if st.sigma < zero {
            return Err(1);
        }
        let mut emax = zero;
        let mut qmin = z[4 * n0 - 3];
        st.qmax = qmin;
        let mut i4 = 4 * n0;
        let mut found = false;
        while i4 >= 8 {
            if z[i4 - 5] <= zero {
                found = true;
                break;
            }
            if qmin >= four * emax {
                qmin = min(qmin, z[i4 - 3]);
                emax = max(emax, z[i4 - 5]);
            }
            st.qmax = max(st.qmax, z[i4 - 7] + z[i4 - 5]);
            i4 -= 4;
        }
        if !found {
            i4 = 4;
        }
        i0 = i4 / 4;
        pp = 0;
        if n0 - i0 > 1 {
            let mut dee = z[4 * i0 - 3];
            let mut deemin = dee;
            let mut kmin = i0;
            let mut i4 = 4 * i0 + 1;
            while i4 <= 4 * n0 - 3 {
                dee = z[i4] * (dee / (dee + z[i4 - 2]));
                if dee <= deemin {
                    deemin = dee;
                    kmin = (i4 + 3) / 4;
                }
                i4 += 4;
            }
            if (kmin - i0) * 2 < n0 - kmin && deemin <= half * z[4 * n0 - 3] {
                let ipn4 = 4 * (i0 + n0);
                pp = 2;
                let mut i4 = 4 * i0;
                while i4 <= 2 * (i0 + n0 - 1) {
                    z.swap(i4 - 3, ipn4 - i4 - 3);
                    z.swap(i4 - 2, ipn4 - i4 - 2);
                    z.swap(i4 - 1, ipn4 - i4 - 5);
                    z.swap(i4, ipn4 - i4 - 4);
                    i4 += 4;
                }
            }
        }
        let floor = qmin - c::<N>(2.0) * qmin.sqrt() * emax.sqrt();
        st.dmin = -max(zero, floor);
        let nbig = 100 * (n0 - i0 + 1);
        let mut done = false;
        for _ in 1..=nbig {
            if i0 > n0 {
                done = true;
                break;
            }
            lasq3(i0, &mut n0, z, &mut pp, &mut st)?;
            pp = 1 - pp;
            if pp == 0 && n0 - i0 >= 3 && (z[4 * n0] <= tol2 * st.qmax || z[4 * n0 - 1] <= tol2 * st.sigma) {
                let mut splt = i0 - 1;
                st.qmax = z[4 * i0 - 3];
                let mut emin = z[4 * i0 - 1];
                let mut oldemn = z[4 * i0];
                let mut i4 = 4 * i0;
                while i4 <= 4 * (n0 - 3) {
                    if z[i4] <= tol2 * z[i4 - 3] || z[i4 - 1] <= tol2 * st.sigma {
                        z[i4 - 1] = -st.sigma;
                        splt = i4 / 4;
                        st.qmax = zero;
                        emin = z[i4 + 3];
                        oldemn = z[i4 + 4];
                    } else {
                        st.qmax = max(st.qmax, z[i4 + 1]);
                        emin = min(emin, z[i4 - 1]);
                        oldemn = min(oldemn, z[i4]);
                    }
                    i4 += 4;
                }
                z[4 * n0 - 1] = emin;
                z[4 * n0] = oldemn;
                i0 = splt + 1;
            }
        }
        if !done {
            return Err(2);
        }
    }
    if n0 >= 1 {
        return Err(3);
    }
    for k in 2..=n {
        z[k] = z[4 * k - 3];
    }
    z.0[1..=n as usize].sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
    Ok(())
}

/// `dlasq3`: deflation checks, then one dqds transform with a shift.
fn lasq3<const N: usize>(i0: isize, n0: &mut isize, z: &mut Z<'_, N>, pp: &mut isize, st: &mut State<N>) -> Result<(), i32> {
    let zero = F::<N>::zero();
    let (one, half, qurtr, two, cbias) = (F::<N>::one(), c::<N>(0.5), c::<N>(0.25), c::<N>(2.0), c::<N>(1.5));
    let n0in = *n0;
    let eps = F::<N>::epsilon();
    let tol = eps * c::<N>(100.0);
    let tol2 = tol * tol;
    loop {
        if *n0 < i0 {
            return Ok(());
        }
        let nn = 4 * *n0 + *pp;
        // Otherwise the bottom 2×2 deflates.
        let mut deflate_one = *n0 == i0;
        if !deflate_one && *n0 != i0 + 1 {
            if z[nn - 5] > tol2 * (st.sigma + z[nn - 3]) && z[nn - 2 * *pp - 4] > tol2 * z[nn - 7] {
                if z[nn - 9] > tol2 * st.sigma && z[nn - 2 * *pp - 8] > tol2 * z[nn - 11] {
                    break;
                }
            } else {
                deflate_one = true;
            }
        }
        if deflate_one {
            z[4 * *n0 - 3] = z[4 * *n0 + *pp - 3] + st.sigma;
            *n0 -= 1;
            continue;
        }
        if z[nn - 3] > z[nn - 7] {
            z.swap(nn - 3, nn - 7);
        }
        let mut t = half * ((z[nn - 7] - z[nn - 3]) + z[nn - 5]);
        if z[nn - 5] > z[nn - 3] * tol2 && t != zero {
            let mut s = z[nn - 3] * (z[nn - 5] / t);
            if s <= t {
                s = z[nn - 3] * (z[nn - 5] / (t * (one + (one + s / t).sqrt())));
            } else {
                s = z[nn - 3] * (z[nn - 5] / (t + t.sqrt() * (t + s).sqrt()));
            }
            t = z[nn - 7] + (s + z[nn - 5]);
            z[nn - 3] = z[nn - 3] * (z[nn - 7] / t);
            z[nn - 7] = t;
        }
        z[4 * *n0 - 7] = z[nn - 7] + st.sigma;
        z[4 * *n0 - 3] = z[nn - 3] + st.sigma;
        *n0 -= 2;
    }
    let n0 = *n0;
    if *pp == 2 {
        *pp = 0;
    }
    let pp = *pp;
    if st.dmin <= zero || n0 < n0in {
        if cbias * z[4 * i0 + pp - 3] < z[4 * n0 + pp - 3] {
            let ipn4 = 4 * (i0 + n0);
            let mut j4 = 4 * i0;
            while j4 <= 2 * (i0 + n0 - 1) {
                z.swap(j4 - 3, ipn4 - j4 - 3);
                z.swap(j4 - 2, ipn4 - j4 - 2);
                z.swap(j4 - 1, ipn4 - j4 - 5);
                z.swap(j4, ipn4 - j4 - 4);
                j4 += 4;
            }
            if n0 - i0 <= 4 {
                z[4 * n0 + pp - 1] = z[4 * i0 + pp - 1];
                z[4 * n0 - pp] = z[4 * i0 - pp];
            }
            st.dmin2 = min(st.dmin2, z[4 * n0 + pp - 1]);
            z[4 * n0 + pp - 1] = min(min(z[4 * n0 + pp - 1], z[4 * i0 + pp - 1]), z[4 * i0 + pp + 3]);
            z[4 * n0 - pp] = min(min(z[4 * n0 - pp], z[4 * i0 - pp]), z[4 * i0 - pp + 4]);
            st.qmax = max(max(st.qmax, z[4 * i0 + pp - 3]), z[4 * i0 + pp + 1]);
            st.dmin = -zero;
        }
    }
    lasq4(i0, n0, z, pp, n0in, st);
    let zero_shift = loop {
        lasq5(i0, n0, z, pp, st, eps);
        st.ndiv += n0 - i0 + 2;
        st.iter += 1;
        if st.dmin >= zero && st.dmin1 >= zero {
            break false;
        } else if st.dmin < zero
            && st.dmin1 > zero
            && z[4 * (n0 - 1) - pp] < tol * (st.sigma + st.dn1)
            && st.dn.abs() < tol * st.sigma
        {
            z[4 * (n0 - 1) - pp + 2] = zero;
            st.dmin = zero;
            break false;
        } else if st.dmin < zero {
            st.nfail += 1;
            if st.ttype < -22 {
                st.tau = zero;
            } else if st.dmin1 > zero {
                st.tau = (st.tau + st.dmin) * (one - two * eps);
                st.ttype -= 11;
            } else {
                st.tau = qurtr * st.tau;
                st.ttype -= 12;
            }
        } else if st.dmin.is_nan() {
            if st.tau == zero {
                break true;
            }
            st.tau = zero;
        } else {
            break true;
        }
        if st.nfail > 64 * (n0 - i0 + 2) as usize {
            return Err(5);
        }
    };
    if zero_shift {
        lasq6(i0, n0, z, pp, st);
        st.ndiv += n0 - i0 + 2;
        st.iter += 1;
        st.tau = zero;
    }
    let t;
    if st.tau < st.sigma {
        st.desig += st.tau;
        t = st.sigma + st.desig;
        st.desig -= t - st.sigma;
    } else {
        t = st.sigma + st.tau;
        st.desig = st.sigma - (t - st.tau) + st.desig;
    }
    st.sigma = t;
    Ok(())
}

/// `dlasq4`: the shift. Early returns keep the previous `tau`, as in LAPACK.
fn lasq4<const N: usize>(i0: isize, n0: isize, z: &Z<'_, N>, pp: isize, n0in: isize, st: &mut State<N>) {
    let zero = F::<N>::zero();
    let one = F::<N>::one();
    let (cnst1, cnst2, cnst3) = (c::<N>(0.563), c::<N>(1.01), c::<N>(1.05));
    let (qurtr, third, half, two, hundrd) = (c::<N>(0.25), c::<N>(0.333), c::<N>(0.5), c::<N>(2.0), c::<N>(100.0));
    let (dmin, dmin1, dmin2, dn, dn1, dn2) = (st.dmin, st.dmin1, st.dmin2, st.dn, st.dn1, st.dn2);
    if dmin <= zero {
        st.tau = -dmin;
        st.ttype = -1;
        return;
    }
    let nn = 4 * n0 + pp;
    let s;
    if n0in == n0 {
        if dmin == dn || dmin == dn1 {
            let mut b1 = z[nn - 3].sqrt() * z[nn - 5].sqrt();
            let mut b2 = z[nn - 7].sqrt() * z[nn - 9].sqrt();
            let mut a2 = z[nn - 7] + z[nn - 5];
            if dmin == dn && dmin1 == dn1 {
                let gap2 = dmin2 - a2 - dmin2 * qurtr;
                let gap1 = if gap2 > zero && gap2 > b2 {
                    a2 - dn - (b2 / gap2) * b2
                } else {
                    a2 - dn - (b1 + b2)
                };
                if gap1 > zero && gap1 > b1 {
                    s = max(dn - (b1 / gap1) * b1, half * dmin);
                    st.ttype = -2;
                } else {
                    let mut t = zero;
                    if dn > b1 {
                        t = dn - b1;
                    }
                    if a2 > b1 + b2 {
                        t = min(t, a2 - (b1 + b2));
                    }
                    s = max(t, third * dmin);
                    st.ttype = -3;
                }
            } else {
                st.ttype = -4;
                let mut t = qurtr * dmin;
                let gam;
                let mut np;
                if dmin == dn {
                    gam = dn;
                    a2 = zero;
                    if z[nn - 5] > z[nn - 7] {
                        return;
                    }
                    b2 = z[nn - 5] / z[nn - 7];
                    np = nn - 9;
                } else {
                    np = nn - 2 * pp;
                    gam = dn1;
                    if z[np - 4] > z[np - 2] {
                        return;
                    }
                    a2 = z[np - 4] / z[np - 2];
                    if z[nn - 9] > z[nn - 11] {
                        return;
                    }
                    b2 = z[nn - 9] / z[nn - 11];
                    np = nn - 13;
                }
                a2 += b2;
                let mut i4 = np;
                while i4 >= 4 * i0 - 1 + pp {
                    if b2 == zero {
                        break;
                    }
                    b1 = b2;
                    if z[i4] > z[i4 - 2] {
                        return;
                    }
                    b2 *= z[i4] / z[i4 - 2];
                    a2 += b2;
                    if hundrd * max(b2, b1) < a2 || cnst1 < a2 {
                        break;
                    }
                    i4 -= 4;
                }
                a2 = cnst3 * a2;
                if a2 < cnst1 {
                    t = gam * (one - a2.sqrt()) / (one + a2);
                }
                s = t;
            }
        } else if dmin == dn2 {
            st.ttype = -5;
            let mut t = qurtr * dmin;
            let np = nn - 2 * pp;
            let mut b1 = z[np - 2];
            let mut b2 = z[np - 6];
            let gam = dn2;
            if z[np - 8] > b2 || z[np - 4] > b1 {
                return;
            }
            let mut a2 = (z[np - 8] / b2) * (one + z[np - 4] / b1);
            if n0 - i0 > 2 {
                b2 = z[nn - 13] / z[nn - 15];
                a2 += b2;
                let mut i4 = nn - 17;
                while i4 >= 4 * i0 - 1 + pp {
                    if b2 == zero {
                        break;
                    }
                    b1 = b2;
                    if z[i4] > z[i4 - 2] {
                        return;
                    }
                    b2 *= z[i4] / z[i4 - 2];
                    a2 += b2;
                    if hundrd * max(b2, b1) < a2 || cnst1 < a2 {
                        break;
                    }
                    i4 -= 4;
                }
                a2 = cnst3 * a2;
            }
            if a2 < cnst1 {
                t = gam * (one - a2.sqrt()) / (one + a2);
            }
            s = t;
        } else {
            if st.ttype == -6 {
                st.g += third * (one - st.g);
            } else if st.ttype == -18 {
                st.g = qurtr * third;
            } else {
                st.g = qurtr;
            }
            s = st.g * dmin;
            st.ttype = -6;
        }
    } else if n0in == n0 + 1 {
        if dmin1 == dn1 && dmin2 == dn2 {
            st.ttype = -7;
            let mut t = third * dmin1;
            if z[nn - 5] > z[nn - 7] {
                return;
            }
            let mut b1 = z[nn - 5] / z[nn - 7];
            let mut b2 = b1;
            if b2 != zero {
                let mut i4 = 4 * n0 - 9 + pp;
                while i4 >= 4 * i0 - 1 + pp {
                    let a2 = b1;
                    if z[i4] > z[i4 - 2] {
                        return;
                    }
                    b1 *= z[i4] / z[i4 - 2];
                    b2 += b1;
                    if hundrd * max(b1, a2) < b2 {
                        break;
                    }
                    i4 -= 4;
                }
            }
            b2 = (cnst3 * b2).sqrt();
            let a2 = dmin1 / (one + b2 * b2);
            let gap2 = half * dmin2 - a2;
            if gap2 > zero && gap2 > b2 * a2 {
                t = max(t, a2 * (one - cnst2 * a2 * (b2 / gap2) * b2));
            } else {
                t = max(t, a2 * (one - cnst2 * b2));
                st.ttype = -8;
            }
            s = t;
        } else {
            let mut t = qurtr * dmin1;
            if dmin1 == dn1 {
                t = half * dmin1;
            }
            s = t;
            st.ttype = -9;
        }
    } else if n0in == n0 + 2 {
        if dmin2 == dn2 && two * z[nn - 5] < z[nn - 7] {
            st.ttype = -10;
            let mut t = third * dmin2;
            if z[nn - 5] > z[nn - 7] {
                return;
            }
            let mut b1 = z[nn - 5] / z[nn - 7];
            let mut b2 = b1;
            if b2 != zero {
                let mut i4 = 4 * n0 - 9 + pp;
                while i4 >= 4 * i0 - 1 + pp {
                    if z[i4] > z[i4 - 2] {
                        return;
                    }
                    b1 *= z[i4] / z[i4 - 2];
                    b2 += b1;
                    if hundrd * b1 < b2 {
                        break;
                    }
                    i4 -= 4;
                }
            }
            b2 = (cnst3 * b2).sqrt();
            let a2 = dmin2 / (one + b2 * b2);
            let gap2 = z[nn - 7] + z[nn - 9] - z[nn - 11].sqrt() * z[nn - 9].sqrt() - a2;
            if gap2 > zero && gap2 > b2 * a2 {
                t = max(t, a2 * (one - cnst2 * a2 * (b2 / gap2) * b2));
            } else {
                t = max(t, a2 * (one - cnst2 * b2));
            }
            s = t;
        } else {
            s = qurtr * dmin2;
            st.ttype = -11;
        }
    } else if n0in > n0 + 2 {
        s = zero;
        st.ttype = -12;
    } else {
        return;
    }
    st.tau = s;
}

/// `dlasq5`: one dqds transform with shift `tau`.
fn lasq5<const N: usize>(i0: isize, n0: isize, z: &mut Z<'_, N>, pp: isize, st: &mut State<N>, eps: F<N>) {
    let zero = F::<N>::zero();
    if n0 - i0 - 1 <= 0 {
        return;
    }
    let dthresh = eps * (st.sigma + st.tau);
    if st.tau < dthresh * c::<N>(0.5) {
        st.tau = zero;
    }
    let tau = st.tau;
    let flush = tau == zero;
    let mut j4 = 4 * i0 + pp - 3;
    let mut emin = z[j4 + 4];
    let mut d = z[j4] - tau;
    st.dmin = d;
    st.dmin1 = -z[j4];
    // (q, e) of the source and destination halves of the ping-pong array.
    let (qa, ea, qb, eb) = if pp == 0 { (1, -1, -2, 0) } else { (2, 0, -3, -1) };
    j4 = 4 * i0;
    while j4 <= 4 * (n0 - 3) {
        z[j4 + qb] = d + z[j4 + ea];
        if d < zero {
            return;
        }
        let temp = z[j4 + qa] / z[j4 + qb];
        d = d * temp - tau;
        z[j4 + eb] = z[j4 + ea] * temp;
        if flush && d < dthresh {
            d = zero;
        }
        st.dmin = min(st.dmin, d);
        emin = min(emin, z[j4 + eb]);
        j4 += 4;
    }
    st.dn2 = d;
    st.dmin2 = st.dmin;
    j4 = 4 * (n0 - 2) - pp;
    let mut j4p2 = j4 + 2 * pp - 1;
    z[j4 - 2] = st.dn2 + z[j4p2];
    if st.dn2 < zero {
        return;
    }
    let temp = z[j4p2 + 2] / z[j4 - 2];
    z[j4] = z[j4p2] * temp;
    st.dn1 = st.dn2 * temp - tau;
    st.dmin = min(st.dmin, st.dn1);
    st.dmin1 = st.dmin;
    j4 += 4;
    j4p2 = j4 + 2 * pp - 1;
    z[j4 - 2] = st.dn1 + z[j4p2];
    if st.dn1 < zero {
        return;
    }
    let temp = z[j4p2 + 2] / z[j4 - 2];
    z[j4] = z[j4p2] * temp;
    st.dn = st.dn1 * temp - tau;
    st.dmin = min(st.dmin, st.dn);
    z[j4 + 2] = st.dn;
    z[4 * n0 - pp] = emin;
}

/// `dlasq6`: one dqd transform without shift.
fn lasq6<const N: usize>(i0: isize, n0: isize, z: &mut Z<'_, N>, pp: isize, st: &mut State<N>) {
    let zero = F::<N>::zero();
    if n0 - i0 - 1 <= 0 {
        return;
    }
    let mut j4 = 4 * i0 + pp - 3;
    let mut emin = z[j4 + 4];
    let mut d = z[j4];
    st.dmin = d;
    let (qa, ea, qb, eb) = if pp == 0 { (1, -1, -2, 0) } else { (2, 0, -3, -1) };
    j4 = 4 * i0;
    while j4 <= 4 * (n0 - 3) {
        z[j4 + qb] = d + z[j4 + ea];
        if z[j4 + qb] == zero {
            z[j4 + eb] = zero;
            d = z[j4 + qa];
            st.dmin = d;
            emin = zero;
        } else if zero < z[j4 + qa] && zero < z[j4 + qb] {
            let temp = z[j4 + qa] / z[j4 + qb];
            z[j4 + eb] = z[j4 + ea] * temp;
            d *= temp;
        } else {
            z[j4 + eb] = z[j4 + qa] * (z[j4 + ea] / z[j4 + qb]);
            d = z[j4 + qa] * (d / z[j4 + qb]);
        }
        st.dmin = min(st.dmin, d);
        emin = min(emin, z[j4 + eb]);
        j4 += 4;
    }
    st.dn2 = d;
    st.dmin2 = st.dmin;
    let mut dn1 = zero;
    for step in 0..2 {
        j4 = 4 * (n0 - 2) - pp + 4 * step;
        let j4p2 = j4 + 2 * pp - 1;
        let prev = if step == 0 { st.dn2 } else { dn1 };
        z[j4 - 2] = prev + z[j4p2];
        let next;
        if z[j4 - 2] == zero {
            z[j4] = zero;
            next = z[j4p2 + 2];
            st.dmin = next;
            emin = zero;
        } else if zero < z[j4p2 + 2] && zero < z[j4 - 2] {
            let temp = z[j4p2 + 2] / z[j4 - 2];
            z[j4] = z[j4p2] * temp;
            next = prev * temp;
        } else {
            z[j4] = z[j4p2 + 2] * (z[j4p2] / z[j4 - 2]);
            next = z[j4p2 + 2] * (prev / z[j4 - 2]);
        }
        st.dmin = min(st.dmin, next);
        if step == 0 {
            dn1 = next;
            st.dn1 = next;
            st.dmin1 = st.dmin;
        } else {
            st.dn = next;
        }
    }
    z[j4 + 2] = st.dn;
    z[4 * n0 - pp] = emin;
}
