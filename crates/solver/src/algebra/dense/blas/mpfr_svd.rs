use super::*;

pub(super) fn num<const N: usize>(v: usize) -> F<N> {
    <F<N> as num_traits::FromPrimitive>::from_usize(v).unwrap()
}
pub(super) fn hypot<const N: usize>(x: F<N>, y: F<N>) -> F<N> {
    let x = x.abs();
    let y = y.abs();
    let h = if x > y { x } else { y };
    if h == F::zero() {
        h
    } else {
        let x = x / h;
        let y = y / h;
        h * (x * x + y * y).sqrt()
    }
}
pub(super) fn norm<const N: usize>(a: &[F<N>]) -> F<N> {
    a.iter().fold(F::zero(), |v, &x| hypot(v, x))
}
pub(super) fn rotate<const N: usize>(a: &mut [F<N>], rows: usize, p: usize, q: usize, c: F<N>, s: F<N>) {
    let ns = -s;
    for i in 0..rows {
        let x = a[i + p * rows];
        let y = a[i + q * rows];
        a[i + p * rows] = ns.mul_add(y, c * x);
        a[i + q * rows] = s.mul_add(x, c * y);
    }
}
// Work is owned and retained by the decomposition engine, as in COSMO's
// PsdBlasWorkspace lifecycle: query once, resize, then reuse across factors.
pub(super) fn take_work<'a, const N: usize>(work: &mut &'a mut [F<N>], len: usize) -> &'a mut [F<N>] {
    let (head, tail) = std::mem::take(work).split_at_mut(len);
    *work = tail;
    head
}
pub(super) fn identity<const N: usize>(v: &mut [F<N>], n: usize) {
    v.fill(F::zero());
    for i in 0..n {
        v[i + i * n] = F::one();
    }
}
// Bidiagonal reduction, Demmel--Kahan iteration, shifted QR, and reflector
// reconstruction adapted from GenericLinearAlgebra.jl v0.4.0 src/svd.jl and
// SDPX.jl/src/core/utils/bigfloat_svd.jl (the latter maps v0.4.1 upstream).
//
// The MIT License (MIT)
// Copyright (c) 2014-2018 Andreas Noack
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Real Givens: [c s; -s c] [x;y] = [r;0]. Stable hypot avoids squares of
// unscaled entries; all signs, norms, and divisions use the declared precision.
pub(super) fn givens<const N: usize>(x: F<N>, y: F<N>) -> (F<N>, F<N>, F<N>) {
    let r = hypot(x, y);
    if r == F::zero() {
        (F::one(), F::zero(), r)
    } else {
        (x / r, y / r, r)
    }
}
pub(super) fn svd_rotate<const N: usize>(u: &mut [F<N>], rows: usize, p: usize, c: F<N>, s: F<N>) {
    if !u.is_empty() {
        rotate(u, rows, p, p + 1, c, -s);
    }
}
// Construct H=I-tau*v*v', with v[0]=1 and H*x=beta*e_1.
pub(super) fn reflector<const N: usize>(x: &mut [F<N>]) -> (F<N>, F<N>) {
    let magnitude = norm(x);
    if magnitude == F::zero() {
        return (F::zero(), F::zero());
    }
    let beta = if x[0] >= F::zero() {
        -magnitude
    } else {
        magnitude
    };
    let denominator = x[0] - beta;
    let tau = (beta - x[0]) / beta;
    for value in &mut x[1..] {
        *value /= denominator;
    }
    x[0] = F::one();
    (beta, tau)
}
pub(super) fn reduce_bidiagonal<const N: usize>(
    a: &mut [F<N>],
    m: usize,
    n: usize,
    d: &mut [F<N>],
    e: &mut [F<N>],
    left: &mut [F<N>],
    right: &mut [F<N>],
    scratch: &mut [F<N>],
) {
    for k in 0..n {
        let x = &mut scratch[..m - k];
        x.copy_from_slice(&a[k + k * m..(k + 1) * m]);
        let (beta, tau) = reflector(x);
        d[k] = beta;
        left[k] = tau;
        for j in k + 1..n {
            let mut dot = F::dot_exact(
                x.iter().zip(&a[k + j * m..m + j * m]),
            );
            dot *= tau;
            let ndot = -dot;
            for i in k..m {
                a[i + j * m] = x[i - k].mul_add(ndot, a[i + j * m]);
            }
        }
        a[k + k * m] = beta;
        for i in k + 1..m {
            a[i + k * m] = x[i - k];
        }
        if k + 1 < n {
            let x = &mut scratch[..n - k - 1];
            for j in k + 1..n {
                x[j - k - 1] = a[k + j * m];
            }
            let (beta, tau) = reflector(x);
            e[k] = beta;
            right[k] = tau;
            for i in k + 1..m {
                let mut dot = F::dot_exact(
                    (k + 1..n).map(|j| (&a[i + j * m], &x[j - k - 1])),
                );
                dot *= tau;
                let ndot = -dot;
                for j in k + 1..n {
                    a[i + j * m] = x[j - k - 1].mul_add(ndot, a[i + j * m]);
                }
            }
            a[k + (k + 1) * m] = beta;
            for j in k + 2..n {
                a[k + j * m] = x[j - k - 1];
            }
        }
    }
}
pub(super) fn demmel_kahan<const N: usize>(
    d: &mut [F<N>],
    e: &mut [F<N>],
    lo: usize,
    hi: usize,
    u: &mut [F<N>],
    m: usize,
    v: &mut [F<N>],
    n: usize,
) {
    let (mut c, mut oldc, mut olds) = (F::one(), F::one(), F::zero());
    for i in lo..hi {
        let (nc, s, r) = givens(d[i] * c, e[i]);
        c = nc;
        svd_rotate(v, n, i, c, s);
        if i > lo {
            e[i - 1] = olds * r;
        }
        let (nc, ns, r) = givens(oldc * r, d[i + 1] * s);
        oldc = nc;
        olds = ns;
        d[i] = r;
        svd_rotate(u, m, i, oldc, olds);
    }
    let h = d[hi] * c;
    e[hi - 1] = h * olds;
    d[hi] = h * oldc;
}
pub(super) fn shifted_qr<const N: usize>(
    d: &mut [F<N>],
    e: &mut [F<N>],
    lo: usize,
    hi: usize,
    shift: F<N>,
    u: &mut [F<N>],
    m: usize,
    v: &mut [F<N>],
    n: usize,
) {
    let (c, s, _) = givens(d[lo] - shift * (shift / d[lo]), e[lo]);
    svd_rotate(v, n, lo, c, s);
    let mut di = d[lo] * c + e[lo] * s;
    let mut ei1 = -d[lo] * s + e[lo] * c;
    let mut di1 = d[lo + 1] * c;
    let mut bulge = d[lo + 1] * s;
    for i in lo..hi - 1 {
        let (c, s, _) = givens(di, bulge);
        svd_rotate(u, m, i, c, s);
        d[i] = c * di + s * bulge;
        let ei = c * ei1 + s * di1;
        di1 = -s * ei1 + c * di1;
        ei1 = e[i + 1] * c;
        bulge = s * e[i + 1];
        let (c, s, _) = givens(ei, bulge);
        svd_rotate(v, n, i + 1, c, s);
        e[i] = ei * c + bulge * s;
        di = di1 * c + ei1 * s;
        ei1 = -di1 * s + ei1 * c;
        bulge = d[i + 2] * s;
        di1 = d[i + 2] * c;
    }
    let (c, s, _) = givens(di, bulge);
    svd_rotate(u, m, hi - 1, c, s);
    d[hi - 1] = c * di + s * bulge;
    e[hi - 1] = c * ei1 + s * di1;
    d[hi] = -s * ei1 + c * di1;
}
// LAWN 3 Lemma 1 recurrences. Deflate off-diagonals relative to the estimated
// smallest singular value, never by an absolute cutoff on diagonal values.
pub(super) fn estimate_smallest<const N: usize>(
    d: &[F<N>],
    e: &mut [F<N>],
    lo: usize,
    hi: usize,
    tol: F<N>,
) -> F<N> {
    let mut mu = d[lo].abs();
    let mut b1 = mu;
    for j in lo..hi {
        let denominator = mu + e[j].abs();
        mu = if denominator == F::zero() {
            F::zero()
        } else {
            d[j + 1].abs() * (mu / denominator)
        };
        if e[j].abs() < tol * mu {
            e[j] = F::zero();
        }
        if mu < b1 {
            b1 = mu;
        }
    }
    let mut lambda = d[hi].abs();
    let mut binf = lambda;
    for j in (lo..hi).rev() {
        let denominator = lambda + e[j].abs();
        lambda = if denominator == F::zero() {
            F::zero()
        } else {
            d[j].abs() * (lambda / denominator)
        };
        if e[j].abs() < tol * lambda {
            e[j] = F::zero();
        }
        if lambda < binf {
            binf = lambda;
        }
    }
    if binf < b1 {
        binf
    } else {
        b1
    }
}
// Smaller singular value of a 2x2 upper bidiagonal. Scaling and the product
// formula avoid squaring a tiny singular value when constructing the shift.
pub(super) fn small_shift<const N: usize>(d1: F<N>, d2: F<N>, e: F<N>) -> F<N> {
    let mut h = d1.abs();
    if d2.abs() > h {
        h = d2.abs();
    }
    if e.abs() > h {
        h = e.abs();
    }
    if h == F::zero() {
        return h;
    }
    let a = d1 / h;
    let b = d2 / h;
    let c = e / h;
    let aa = a * a;
    let bb = b * b;
    let cc = c * c;
    let discriminant = hypot(
        (a + b) * (a - b),
        c * hypot(num::<N>(2).sqrt() * hypot(a, b), c),
    );
    let largest = ((aa + bb + cc + discriminant) / num::<N>(2)).sqrt();
    if d1.abs() < d2.abs() {
        (d1.abs() / largest) * (d2.abs() / h)
    } else {
        (d2.abs() / largest) * (d1.abs() / h)
    }
}
pub(super) fn bidiagonal_svd<const N: usize>(
    d: &mut [F<N>],
    e: &mut [F<N>],
    u: &mut [F<N>],
    m: usize,
    v: &mut [F<N>],
    n: usize,
) -> Result<(), i32> {
    if n < 2 {
        return Ok(());
    }
    // Upstream GenericLinearAlgebra `__svd!` default tol = 100*eps(T); LAPACK
    // dbdsqr TOLMUL is also 100 at every MPFR precision >= 128 bits.
    let tol = F::epsilon() * num::<N>(100);
    let mut hi = n - 1;
    // Unlike upstream's unbounded controller, exhaustion is an explicit error.
    for _ in 0..(64 + 128 * N) * n {
        while hi > 0 && e[hi - 1] == F::zero() {
            hi -= 1;
        }
        if hi == 0 {
            return Ok(());
        }
        let mut lo = hi - 1;
        while lo > 0 && e[lo - 1] != F::zero() {
            lo -= 1;
        }
        if d[lo..=hi].iter().any(|&x| x == F::zero()) {
            demmel_kahan(d, e, lo, hi, u, m, v, n);
            continue;
        }
        let smallest = estimate_smallest(d, e, lo, hi, tol);
        let threshold = tol * smallest;
        let mut split = false;
        for x in &mut e[lo..hi] {
            if x.abs() <= threshold {
                *x = F::zero();
                split = true;
            }
        }
        if split {
            continue;
        }
        let mut largest = F::zero();
        for x in d[lo..=hi].iter().chain(e[lo..hi].iter()) {
            if x.abs() > largest {
                largest = x.abs();
            }
        }
        // Upstream `__svd!` shift guard at its default tolerance:
        // fudge * tol * sigma^- <= eps * sigma^+.
        if num::<N>(hi - lo + 1) * tol * smallest <= F::epsilon() * largest {
            demmel_kahan(d, e, lo, hi, u, m, v, n);
        } else {
            let shift = small_shift(d[hi - 1], d[hi], e[hi - 1]);
            if (shift / d[lo]).abs() < F::epsilon().sqrt() {
                demmel_kahan(d, e, lo, hi, u, m, v, n);
            } else {
                shifted_qr(d, e, lo, hi, shift, u, m, v, n);
            }
        }
        if d.iter().chain(e.iter()).any(|x| !x.is_finite()) {
            return Err(1);
        }
    }
    Err(1)
}
// Packed Householder reconstruction, applied backwards as in SDPX's upstream
// reflector routines. Initial identity columns also supply all null vectors.
pub(super) fn apply_reflectors<const N: usize>(
    a: &[F<N>],
    m: usize,
    n: usize,
    left: &[F<N>],
    right: &[F<N>],
    u: &mut [F<N>],
    uc: usize,
    v: &mut [F<N>],
) {
    for k in (0..n).rev() {
        for j in 0..uc {
            let mut dot = F::dot_exact(
                (k + 1..m).map(|i| (&a[i + k * m], &u[i + j * m])),
            ) + u[k + j * m];
            dot *= left[k];
            let ndot = -dot;
            u[k + j * m] += ndot;
            for i in k + 1..m {
                u[i + j * m] = a[i + k * m].mul_add(ndot, u[i + j * m]);
            }
        }
    }
    if !v.is_empty() {
        for k in (0..n.saturating_sub(1)).rev() {
            for j in 0..n {
                let mut dot = F::dot_exact(
                    (k + 2..n).map(|i| (&a[k + i * m], &v[i + j * n])),
                ) + v[k + 1 + j * n];
                dot *= right[k];
                let ndot = -dot;
                v[k + 1 + j * n] += ndot;
                for i in k + 2..n {
                    v[i + j * n] = a[k + i * m].mul_add(ndot, v[i + j * n]);
                }
            }
        }
    }
}

// Scale only if every nonzero input remains representable. Returning a
// bounded failure is preferable to silently replacing a small singular value
// by zero at MPFR's exponent boundary.
pub(super) fn scale_checked<const N: usize>(a: &mut [F<N>], scale: F<N>) -> Result<(), i32> {
    if scale == F::zero() {
        return Ok(());
    }
    for x in a {
        let old = *x;
        *x /= scale;
        if !x.is_finite() || (old != F::zero() && *x == F::zero()) {
            return Err(1);
        }
    }
    Ok(())
}
// Work layout for the tall factorization; wide matrices transpose both the
// input and requested vector counts. GESVD has no integer work argument, so
// exact integer indices occupy scalar work cells during output ordering.
pub(super) fn svd_work_len(m: usize, n: usize, uc: usize, vr: usize) -> Option<usize> {
    let (m, n, uc, vr) = if m < n {
        (n, m, vr, uc)
    } else {
        (m, n, uc, vr)
    };
    let mut size = 0usize;
    for cells in [
        m.checked_mul(n)?,
        n,
        n.saturating_sub(1),
        n,
        n.saturating_sub(1),
        m,
        m.checked_mul(uc)?,
        if vr > 0 { n.checked_mul(n)? } else { 0 },
        n,
        m.checked_mul(uc)?,
        vr.checked_mul(n)?,
        n,
    ] {
        size = size.checked_add(cells)?;
    }
    Some(size.max(1))
}
// The returned vectors are in the tall orientation. The caller transposes
// their indexing for a wide input, without allocating intermediate factors.
pub(super) fn svd<'a, const N: usize>(
    a: &[F<N>],
    lda: usize,
    m: usize,
    n: usize,
    uc: usize,
    vr: usize,
    work: &'a mut [F<N>],
) -> Result<(&'a [F<N>], &'a [F<N>], &'a [F<N>]), i32> {
    let wide = m < n;
    let (m, n, uc, vr) = if wide { (n, m, vr, uc) } else { (m, n, uc, vr) };
    work.fill(F::zero());
    let mut work = work;
    let b = take_work(&mut work, m * n);
    let d = take_work(&mut work, n);
    let e = take_work(&mut work, n.saturating_sub(1));
    let left = take_work(&mut work, n);
    let right = take_work(&mut work, n.saturating_sub(1));
    let scratch = take_work(&mut work, m);
    let u = take_work(&mut work, m * uc);
    let v = take_work(&mut work, if vr > 0 { n * n } else { 0 });
    let order = take_work(&mut work, n);
    let sorted_u = take_work(&mut work, m * uc);
    let vt = take_work(&mut work, vr * n);
    let ss = take_work(&mut work, n);
    for j in 0..n {
        for i in 0..m {
            b[i + j * m] = if wide { a[j + i * lda] } else { a[i + j * lda] };
        }
    }
    let mut scale = F::zero();
    for &x in b.iter() {
        if !x.is_finite() {
            return Err(1);
        }
        if x.abs() > scale {
            scale = x.abs();
        }
    }
    scale_checked(b, scale)?;
    reduce_bidiagonal(b, m, n, d, e, left, right, scratch);
    if b.iter()
        .chain(d.iter())
        .chain(e.iter())
        .any(|x| !x.is_finite())
    {
        return Err(1);
    }
    for j in 0..uc {
        u[j + j * m] = F::one();
    }
    if vr > 0 {
        identity(v, n);
    }
    bidiagonal_svd(d, e, u, m, v, n)?;
    apply_reflectors(b, m, n, left, right, u, uc, v);
    for (i, cell) in order.iter_mut().enumerate() {
        *cell = num::<N>(i);
    }
    // Explicit index ties reproduce the original stable ordering without
    // sort_by's temporary heap buffer.
    order.sort_unstable_by(|i, j| {
        let (i, j) = (i.to_usize().unwrap(), j.to_usize().unwrap());
        d[j].abs().partial_cmp(&d[i].abs()).unwrap().then(i.cmp(&j))
    });
    for j in 0..n {
        let p = order[j].to_usize().unwrap();
        ss[j] = d[p].abs() * scale;
        if !ss[j].is_finite() || (d[p] != F::zero() && ss[j] == F::zero()) {
            return Err(1);
        }
        if j < uc {
            let sign = if d[p] < F::zero() {
                -F::one()
            } else {
                F::one()
            };
            for i in 0..m {
                sorted_u[i + j * m] = sign * u[i + p * m];
            }
        }
        if j < vr {
            for i in 0..n {
                vt[j + i * vr] = v[i + p * n];
            }
        }
    }
    for j in n..uc {
        for i in 0..m {
            sorted_u[i + j * m] = u[i + j * m];
        }
    }
    Ok((ss, sorted_u, vt))
}

impl<const N: usize> XgesvdScalar for F<N> {
    fn xgesvd(
        ju: u8,
        jv: u8,
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        s: &mut [Self],
        u: &mut [Self],
        ldu: i32,
        vt: &mut [Self],
        ldvt: i32,
        work: &mut [Self],
        lwork: i32,
        info: &mut i32,
    ) {
        let ju = upper(ju);
        let jv = upper(jv);
        let r = m.min(n).max(0);
        let uc = if ju == b'A' {
            m
        } else if ju == b'S' {
            r
        } else {
            0
        };
        let vr = if jv == b'A' {
            n
        } else if jv == b'S' {
            r
        } else {
            0
        };
        *info = if !matches!(ju, b'A' | b'S' | b'O' | b'N') {
            -1
        } else if !matches!(jv, b'A' | b'S' | b'O' | b'N') || (ju == b'O' && jv == b'O') {
            -2
        } else if m < 0 {
            -3
        } else if n < 0 {
            -4
        } else if lda < m.max(1) {
            -6
        } else if ldu < 1 || (uc > 0 && ldu < m) {
            -9
        } else if ldvt < vr.max(1) {
            -11
        } else if work.is_empty() || lwork < -1 || (lwork != -1 && lwork < 1) {
            -13
        } else {
            0
        };
        if *info != 0 {
            return;
        }
        let out_uc = if ju == b'O' { r } else { uc } as usize;
        let out_vr = if jv == b'O' { r } else { vr } as usize;
        let Some(required) = svd_work_len(m as usize, n as usize, out_uc, out_vr)
            .filter(|&size| size <= i32::MAX as usize)
        else {
            *info = -13;
            return;
        };
        work[0] = num::<N>(required);
        if lwork == -1 {
            return;
        }
        if lwork < required as i32 || work.len() < lwork as usize {
            *info = -13;
            return;
        }
        if !valid(a.len(), m, n, lda) {
            *info = -5;
            return;
        }
        if s.len() < r as usize {
            *info = -7;
            return;
        }
        if uc > 0 && !valid(u.len(), m, uc, ldu) {
            *info = -8;
            return;
        }
        if vr > 0 && !valid(vt.len(), vr, n, ldvt) {
            *info = -10;
            return;
        }
        let mm = m as usize;
        let nn = n as usize;
        match svd(
            a,
            lda as usize,
            mm,
            nn,
            out_uc,
            out_vr,
            &mut work[..required],
        ) {
            Err(e) => *info = e,
            Ok((ss, uu, vv)) => {
                s[..r as usize].copy_from_slice(ss);
                let get_u = |i, j| {
                    if mm < nn {
                        vv[j + i * out_uc]
                    } else {
                        uu[i + j * mm]
                    }
                };
                let get_vt = |i, j| {
                    if mm < nn {
                        uu[j + i * nn]
                    } else {
                        vv[i + j * out_vr]
                    }
                };
                for j in 0..uc as usize {
                    for i in 0..mm {
                        u[i + j * ldu as usize] = get_u(i, j);
                    }
                }
                for j in 0..nn {
                    for i in 0..vr as usize {
                        vt[i + j * ldvt as usize] = get_vt(i, j);
                    }
                }
                if ju == b'O' {
                    for j in 0..r as usize {
                        for i in 0..mm {
                            a[i + j * lda as usize] = get_u(i, j);
                        }
                    }
                }
                if jv == b'O' {
                    for j in 0..nn {
                        for i in 0..r as usize {
                            a[i + j * lda as usize] = get_vt(i, j);
                        }
                    }
                }
            }
        }
        work[0] = num::<N>(required);
    }
}
impl<const N: usize> XgesddScalar for F<N> {
    fn xgesdd(
        j: u8,
        m: i32,
        n: i32,
        a: &mut [Self],
        lda: i32,
        s: &mut [Self],
        u: &mut [Self],
        ldu: i32,
        vt: &mut [Self],
        ldvt: i32,
        work: &mut [Self],
        lwork: i32,
        _iwork: &mut [i32],
        info: &mut i32,
    ) {
        let j = upper(j);
        if !matches!(j, b'A' | b'S' | b'O' | b'N') {
            *info = -1;
            return;
        }
        let (ju, jv) = if j == b'O' {
            if m >= n {
                (b'O', b'A')
            } else {
                (b'A', b'O')
            }
        } else {
            (j, j)
        };
        Self::xgesvd(ju, jv, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);
        // GESDD has one fewer JOB argument; otherwise positions are identical.
        if *info < -2 {
            *info += 1;
        } else if *info == -2 {
            *info = -1;
        }
    }
}

