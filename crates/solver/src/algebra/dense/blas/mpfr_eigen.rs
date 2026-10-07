use super::*;

// Householder tridiagonalization of the scaled symmetric matrix in `b`
// (n×n, column-major, both triangles filled). Produces the diagonal `d`,
// the subdiagonal `e` (e[k] = T[k+1,k]). With nonempty `taus`, packs each
// reflector's v[1..] into b[k+2..n, k] for backward vector accumulation.
pub(super) fn tridiagonalize<const N: usize>(
    b: &mut [F<N>],
    n: usize,
    d: &mut [F<N>],
    e: &mut [F<N>],
    taus: &mut [F<N>],
    scratch: &mut [F<N>],
) {
    for k in 0..n.saturating_sub(2) {
        let len = n - k - 1;
        let (v, p) = scratch.split_at_mut(n);
        let v = &mut v[..len];
        for i in 0..len {
            v[i] = b[k + 1 + i + k * n];
        }
        let (beta, tau) = reflector(v);
        e[k] = beta;
        if !taus.is_empty() {
            taus[k] = tau;
        }
        if tau != F::zero() {
            // Two-sided update of the trailing block B = b[k+1..n, k+1..n]:
            // p = tau * B v; w = p - (tau/2)(p·v) v; B -= v wᵀ + w vᵀ.
            // The symv rows and the rank-2 column updates are independent,
            // so they join the ambient pool under the inner-parallel gate.
            let parts = inner_tasks::<N>(2 * len * len, len);
            let p = &mut p[..len];
            if parts > 1 {
                let tail: &[F<N>] = &b[(k + 1) * n..n * n];
                p.par_iter_mut().with_min_len(len.div_ceil(parts)).enumerate().for_each(|(i, pi)| {
                    let acc = F::dot_fma((0..len).map(|j| (&tail[k + 1 + i + j * n], &v[j])));
                    *pi = tau * acc;
                });
            } else {
                for i in 0..len {
                    let acc =
                        F::dot_fma((0..len).map(|j| (&b[k + 1 + i + (k + 1 + j) * n], &v[j])));
                    p[i] = tau * acc;
                }
            }
            let mut dot = F::zero();
            for i in 0..len {
                dot += p[i] * v[i];
            }
            let alpha = (tau / num::<N>(2)) * dot;
            for i in 0..len {
                p[i] -= alpha * v[i];
            }
            if parts > 1 {
                b[(k + 1) * n..n * n]
                    .par_chunks_mut(n)
                    .with_min_len(len.div_ceil(parts))
                    .enumerate()
                    .for_each(|(j, col)| {
                        for i in 0..=j {
                            col[k + 1 + i] -= v[i] * p[j] + p[i] * v[j];
                        }
                    });
            } else {
                for j in 0..len {
                    for i in 0..=j {
                        b[k + 1 + i + (k + 1 + j) * n] -= v[i] * p[j] + p[i] * v[j];
                    }
                }
            }
            // The update is symmetric: copy the rounded upper value instead
            // of repeating both MPFR products for its lower counterpart.
            for j in 0..len {
                for i in 0..j {
                    b[k + 1 + j + (k + 1 + i) * n] = b[k + 1 + i + (k + 1 + j) * n];
                }
            }
        }
        // v[0] == 1 is implicit; only vector requests need the packed tail.
        if !taus.is_empty() {
            for i in 1..len {
                b[k + 1 + i + k * n] = v[i];
            }
        }
    }
    if n >= 2 {
        e[n - 2] = b[n - 1 + (n - 2) * n];
    }
    for i in 0..n {
        d[i] = b[i + i * n];
    }
}
// Accumulate the packed reflectors backwards into the identity, matching the
// apply_reflectors pattern: Q = H_0 ... H_{n-3} with H_k acting on k+1..n.
pub(super) fn form_q<const N: usize>(b: &[F<N>], n: usize, taus: &[F<N>], q: &mut [F<N>]) {
    identity(q, n);
    // Reflector applications to independent columns join the ambient pool
    // under the inner-parallel gate; the reflector sequence stays serial.
    for k in (0..n.saturating_sub(2)).rev() {
        let parts = inner_tasks::<N>(2 * n * (n - k), n);
        if parts > 1 {
            q[..n * n].par_chunks_mut(n).with_min_len(n.div_ceil(parts)).for_each(|qj| {
                let mut dot = qj[k + 1];
                for i in k + 2..n {
                    dot += b[i + k * n] * qj[i];
                }
                dot *= taus[k];
                qj[k + 1] -= dot;
                for i in k + 2..n {
                    qj[i] -= b[i + k * n] * dot;
                }
            });
        } else {
            for j in 0..n {
                let mut dot = q[k + 1 + j * n];
                for i in k + 2..n {
                    dot += b[i + k * n] * q[i + j * n];
                }
                dot *= taus[k];
                q[k + 1 + j * n] -= dot;
                for i in k + 2..n {
                    q[i + j * n] -= b[i + k * n] * dot;
                }
            }
        }
    }
}
// Implicit QL iteration with Wilkinson shift on the tridiagonal (d, e),
// following EISPACK tql2 / Numerical Recipes tqli. When `q` is nonempty its
// columns accumulate the rotations; the scalar sequence on (d, e) is
// identical either way, so 'N' and 'V' eigenvalues agree bit-for-bit.
pub(super) fn tridiagonal_ql<const N: usize>(
    d: &mut [F<N>],
    e: &mut [F<N>],
    q: &mut [F<N>],
    n: usize,
) -> Result<(), i32> {
    let eps = F::epsilon();
    if n == 0 {
        return Ok(());
    }
    e[n - 1] = F::zero();
    let mut budget = (64 + 2 * N * 64) * n;
    for l in 0..n {
        loop {
            // Locate a small subdiagonal element.
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= eps * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            budget -= 1;
            if budget == 0 {
                return Err(1);
            }
            let mut g = (d[l + 1] - d[l]) / (num::<N>(2) * e[l]);
            let r = hypot(g, F::one());
            g = d[m] - d[l] + e[l] / (g + if g >= F::zero() { r } else { -r });
            let (mut s, mut c, mut p) = (F::one(), F::one(), F::zero());
            let mut underflow = false;
            for i in (l..m).rev() {
                let f = s * e[i];
                let bb = c * e[i];
                let r = hypot(f, g);
                e[i + 1] = r;
                if r == F::zero() {
                    // Recover from underflow.
                    d[i + 1] -= p;
                    e[m] = F::zero();
                    underflow = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                let r = (d[i] - g) * s + num::<N>(2) * c * bb;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - bb;
                if !q.is_empty() {
                    for k in 0..n {
                        let f = q[k + (i + 1) * n];
                        q[k + (i + 1) * n] = s * q[k + i * n] + c * f;
                        q[k + i * n] = c * q[k + i * n] - s * f;
                    }
                }
            }
            if underflow {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = F::zero();
        }
    }
    Ok(())
}

// Number of eigenvalues strictly below x on the tridiagonal (d, e), via
// the Sturm sequence of T - xI. Use the limit from below in x: an exactly
// zero pivot is not counted and is continued as a tiny positive pivot.
// Scale that continuation with the matrix; disconnected blocks require no
// division and hence no continuation of the preceding block's zero pivot.
pub(super) fn sturm_count<const N: usize>(d: &[F<N>], e: &[F<N>], n: usize, x: F<N>) -> usize {
    let mut scale = F::zero();
    for value in d[..n].iter().chain(e[..n.saturating_sub(1)].iter()) {
        if value.abs() > scale {
            scale = value.abs();
        }
    }
    let tiny = (scale * F::epsilon()).max(F::min_positive_value());
    let mut count = 0usize;
    let mut q = d[0] - x;
    if q < F::zero() {
        count += 1;
    }
    for i in 1..n {
        q = if e[i - 1] == F::zero() {
            d[i] - x
        } else {
            if q == F::zero() {
                q = tiny;
            }
            d[i] - x - (e[i - 1] * e[i - 1]) / q
        };
        if q < F::zero() {
            count += 1;
        }
    }
    count
}

// Solve (T - sI) x = b on the tridiagonal (d, e) by Gaussian elimination
// with partial pivoting (LAPACK dgtsv/dgttrs), fusing factor and solve.
// `dd`,`du`,`du2`,`dl` are n-sized scratch.  Returns false on an exactly
// singular pivot — at a nearly exact eigenvalue shift the caller treats
// that as convergence, not failure.
#[allow(clippy::too_many_arguments)]
pub(super) fn tridiag_solve<const N: usize>(
    d: &[F<N>],
    e: &[F<N>],
    n: usize,
    s: F<N>,
    b: &mut [F<N>],
    dl: &mut [F<N>],
    dd: &mut [F<N>],
    du: &mut [F<N>],
    du2: &mut [F<N>],
) -> bool {
    for i in 0..n {
        dd[i] = d[i] - s;
    }
    for i in 0..n - 1 {
        dl[i] = e[i];
        du[i] = e[i];
    }
    for i in 0..n.saturating_sub(2) {
        du2[i] = F::zero();
    }
    for i in 0..n - 1 {
        if dd[i].abs() >= dl[i].abs() {
            if dd[i] != F::zero() {
                let fact = dl[i] / dd[i];
                dd[i + 1] -= fact * du[i];
                if i + 1 < n - 1 {
                    // For tridiagonal input du2[i] is zero on entry to this
                    // step; retain the general banded elimination identity.
                    du[i + 1] -= fact * du2[i];
                }
                b[i + 1] -= fact * b[i];
                dl[i] = fact;
            } else {
                dl[i] = F::zero();
            }
        } else {
            let fact = dd[i] / dl[i];
            dd[i] = dl[i];
            let temp = dd[i + 1];
            dd[i + 1] = du[i] - fact * temp;
            if i + 1 < n - 1 {
                let next = du[i + 1];
                du[i + 1] = du2[i] - fact * next;
                du2[i] = next;
            } else {
                du2[i] = F::zero();
            }
            du[i] = temp;
            dl[i] = fact;
            b.swap(i, i + 1);
            b[i + 1] -= dl[i] * b[i];
        }
    }
    if (0..n).any(|i| dd[i] == F::zero()) {
        return false;
    }
    b[n - 1] /= dd[n - 1];
    if n >= 2 {
        b[n - 2] = (b[n - 2] - du[n - 2] * b[n - 1]) / dd[n - 2];
        for i in (0..n - 2).rev() {
            b[i] = (b[i] - du[i] * b[i + 1] - du2[i] * b[i + 2]) / dd[i];
        }
    }
    true
}

#[cfg(test)]
std::thread_local! {
    pub(super) static EIGVAL_STEPS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

// The k-th smallest eigenvalue (0-based) of the tridiagonal (d, e).
// Sturm-sequence bisection first isolates a bracket containing only that
// eigenvalue, then Rayleigh quotient iteration on the same tridiagonal
// polishes it.  Sturm counts verify the result, so failure is detected
// and the caller can fall back to the full QL sweep.  `scratch` needs
// at least 6n cells and is left clobbered.
pub(super) fn tridiag_eigval_at<const N: usize>(
    d: &[F<N>],
    e: &[F<N>],
    n: usize,
    k: usize,
    scratch: &mut [F<N>],
) -> Option<F<N>> {
    #[cfg(test)]
    EIGVAL_STEPS.with(|v| v.set((0, 0)));
    if n == 1 {
        return (k == 0).then_some(d[0]);
    }
    // Gershgorin radius of the tridiagonal.
    let mut ger = d[0].abs() + e[0].abs();
    for i in 1..n {
        let mut r = d[i].abs() + e[i - 1].abs();
        if i + 1 < n {
            r += e[i].abs();
        }
        if r > ger {
            ger = r;
        }
    }
    if ger == F::zero() {
        return Some(F::zero());
    }
    let (mut lo, mut hi) = (-ger, ger);
    // Isolate: keep count(lo) <= k and count(hi) >= k+1 until the bracket
    // holds exactly one eigenvalue.
    // This is a fast path: spend at most 32 halvings on isolation (a
    // 2^-32 fraction of the initial bracket). More tightly clustered or
    // repeated roots use the full QL sweep, at the original precision.
    let max_bisect = 32;
    for _ in 0..max_bisect {
        if sturm_count(d, e, n, hi) - sturm_count(d, e, n, lo) <= 1 {
            break;
        }
        #[cfg(test)]
        EIGVAL_STEPS.with(|v| {
            let (b, r) = v.get();
            v.set((b + 1, r));
        });
        let mid = (lo + hi) / num::<N>(2);
        if mid == lo || mid == hi {
            break;
        }
        if sturm_count(d, e, n, mid) > k {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    if sturm_count(d, e, n, hi) - sturm_count(d, e, n, lo) != 1 {
        return None;
    }
    // Rayleigh quotient iteration: solve (T - μI)ŷ = y, μ += ŷᵀy/ŷᵀŷ.
    let (y, rest) = scratch.split_at_mut(n);
    let (b, rest) = rest.split_at_mut(n);
    let (dl, rest) = rest.split_at_mut(n);
    let (dd, rest) = rest.split_at_mut(n);
    let (du, du2) = rest.split_at_mut(n);
    let mut mu = (lo + hi) / num::<N>(2);
    let inv_n = F::one() / num::<N>(n).sqrt();
    y.fill(inv_n);
    let eps = F::epsilon();
    for _ in 0..32 {
        #[cfg(test)]
        EIGVAL_STEPS.with(|v| {
            let (b, r) = v.get();
            v.set((b, r + 1));
        });
        b.copy_from_slice(y);
        if !tridiag_solve(d, e, n, mu, b, dl, dd, du, du2) {
            break; // exact singular pivot: μ is already an eigenvalue
        }
        let mut numer = F::zero();
        let mut denom = F::zero();
        for i in 0..n {
            numer += b[i] * y[i];
            denom += b[i] * b[i];
        }
        if denom == F::zero() {
            return None;
        }
        // Keep the isolating bracket: unconstrained RQI can converge to
        // a different eigenvalue. An escaping trial uses bisection instead.
        if sturm_count(d, e, n, mu) > k {
            hi = hi.min(mu);
        } else {
            lo = lo.max(mu);
        }
        let step = numer / denom;
        let trial = mu + step;
        let safeguarded = trial < lo || trial > hi;
        mu = if safeguarded {
            (lo + hi) / num::<N>(2)
        } else {
            trial
        };
        let inv = F::one() / denom.sqrt();
        for i in 0..n {
            y[i] = b[i] * inv;
        }
        if !safeguarded && step.abs() <= eps * (mu.abs() + ger) {
            break;
        }
    }
    // Verify: exactly k eigenvalues below and k+1 at-or-below the answer
    // within a few ulps of the Gershgorin scale.
    let delta = eps * ger * num::<N>(8);
    if sturm_count(d, e, n, mu - delta) == k && sturm_count(d, e, n, mu + delta) == k + 1 {
        Some(mu)
    } else {
        None
    }
}

impl<const N: usize> XsyevrScalar for F<N> {
    fn xsyevr(
        job: u8,
        range: u8,
        uplo: u8,
        n: i32,
        a: &mut [Self],
        lda: i32,
        vl: Self,
        vu: Self,
        il: i32,
        iu: i32,
        abstol: Self,
        m: &mut i32,
        w: &mut [Self],
        z: &mut [Self],
        ldz: i32,
        isuppz: &mut [i32],
        work: &mut [Self],
        lwork: i32,
        iwork: &mut [i32],
        liwork: i32,
        info: &mut i32,
    ) {
        let job = upper(job);
        let range = upper(range);
        *m = 0;
        *info = if !matches!(job, b'N' | b'V') {
            -1
        } else if !matches!(range, b'A' | b'V' | b'I') {
            -2
        } else if !tri(uplo) {
            -3
        } else if n < 0 {
            -4
        } else if lda < n.max(1) {
            -6
        } else if range == b'V' && n > 0 && vu <= vl {
            -8
        } else if range == b'I' && (il < 1 || il > n.max(1)) {
            -9
        } else if range == b'I' && (iu < n.min(il) || iu > n) {
            -10
        } else if ldz < 1 || (job == b'V' && ldz < n) {
            -15
        } else if work.is_empty() || lwork < -1 || (lwork != -1 && liwork != -1 && lwork < 1) {
            -18
        } else if iwork.is_empty() || liwork < -1 || (lwork != -1 && liwork != -1 && liwork < 1) {
            -20
        } else {
            0
        };
        if *info != 0 {
            return;
        }
        let nn = n as usize;
        // Owned scratch: b (n²) + d/e (2n) + v/p (2n), plus taus (n)
        // and Q (n²) for vectors, or a 7n single-index block (Sturm + RQI).
        let indexed = job == b'N' && range == b'I' && il == iu;
        let Some(required) = nn
            .checked_mul(nn)
            .and_then(|cells| cells.checked_mul(if job == b'V' { 2 } else { 1 }))
            .and_then(|cells| {
                cells.checked_add(nn.checked_mul(if job == b'V' {
                    5
                } else if indexed {
                    11
                } else {
                    4
                })?)
            })
            .map(|cells| cells.max(1))
            .filter(|&cells| cells <= i32::MAX as usize)
        else {
            *info = -18;
            return;
        };
        let integer_required = nn.max(1);
        work[0] = num::<N>(required);
        iwork[0] = integer_required as i32;
        if lwork == -1 || liwork == -1 {
            return;
        }
        if lwork < required as i32 || work.len() < lwork as usize {
            *info = -18;
            return;
        }
        if liwork < integer_required as i32 || iwork.len() < liwork as usize {
            *info = -20;
            return;
        }
        if !valid(a.len(), n, n, lda) {
            *info = -5;
            return;
        }
        if w.len() < n as usize {
            *info = -13;
            return;
        }
        let mut tail = &mut work[..required];
        let b = take_work(&mut tail, nn * nn);
        let d = take_work(&mut tail, nn);
        let e = take_work(&mut tail, nn);
        let scratch = take_work(&mut tail, 2 * nn);
        let taus = take_work(&mut tail, if job == b'V' { nn } else { 0 });
        let idx = take_work(&mut tail, if indexed { 7 * nn } else { 0 });
        let q = if job == b'V' {
            take_work(&mut tail, nn * nn)
        } else {
            &mut tail[..0]
        };
        let mut scale = Self::zero();
        for j in 0..nn {
            for i in 0..nn {
                let x = sym(a, lda as usize, i, j, uplo);
                if !x.is_finite() {
                    *info = 1;
                    return;
                }
                b[i + j * nn] = x;
                if x.abs() > scale {
                    scale = x.abs();
                }
            }
        }
        if scale_checked(b, scale).is_err() {
            *info = 1;
            return;
        }
        tridiagonalize(b, nn, d, e, taus, scratch);
        if job == b'V' {
            form_q(b, nn, taus, q);
        }
        if indexed {
            // Single eigenvalue by index: Sturm isolation plus RQI polish,
            // verified by Sturm counts.  (d, e) are read-only here, so a
            // failure can fall through to the full QL sweep below.
            match tridiag_eigval_at(d, e, nn, (il - 1) as usize, idx) {
                Some(v) => {
                    w[0] = v * scale;
                    *m = 1;
                    work[0] = num::<N>(required);
                    iwork[0] = integer_required as i32;
                    return;
                }
                None => {}
            }
        }
        // ABSTOL is accepted for LAPACK signature compatibility but must not
        // loosen the eps-based deflation criterion inside the QL iteration.
        let _ = abstol;
        if tridiagonal_ql(d, e, q, nn).is_err() {
            *info = 1;
            return;
        }
        let order = &mut iwork[..nn];
        for (i, p) in order.iter_mut().enumerate() {
            *p = i as i32;
        }
        order.sort_unstable_by(|&i, &j| {
            let (i, j) = (i as usize, j as usize);
            d[i].partial_cmp(&d[j]).unwrap().then(i.cmp(&j))
        });
        let selected = |k: usize, p: usize| {
            range == b'A'
                || (range == b'I' && k + 1 >= il as usize && k + 1 <= iu as usize)
                || (range == b'V' && d[p] * scale > vl && d[p] * scale <= vu)
        };
        let count = order
            .iter()
            .enumerate()
            .filter(|&(k, &p)| selected(k, p as usize))
            .count();
        if job == b'V' && (!valid(z.len(), n, count as i32, ldz) || isuppz.len() < 2 * count) {
            *info = -14;
            return;
        }
        let mut j = 0;
        for (k, &p) in order.iter().enumerate() {
            let p = p as usize;
            if !selected(k, p) {
                continue;
            }
            w[j] = d[p] * scale;
            if job == b'V' {
                for i in 0..nn {
                    z[i + j * ldz as usize] = q[i + p * nn];
                }
                isuppz[2 * j] = 1;
                isuppz[2 * j + 1] = n;
            }
            j += 1;
        }
        *m = count as i32;
        work[0] = num::<N>(required);
        iwork[0] = integer_required as i32;
    }
}
