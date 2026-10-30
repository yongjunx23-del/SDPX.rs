#![allow(non_snake_case)]
//! Limit of the operator relative entropy cone along a recession ray
//!
//! K = cl(K_ore + R₊(0, −W, −W)) has a lineality space, so it is passed in
//! reduced coordinates (n, m = rank W, k = n − m; the caller applies the
//! basis [U V], V = ker W):
//! K = cl{(T', B, A, C) : A, C ≻ 0, T' ⪰ G}, G = Δ' + E·N(A, C)·Eᵀ,
//! Δ' = [0 B; Bᵀ A − C], E = [B; A − C], N(A, C) = ∫₀¹ (1−s)((1−s)C + sA)⁻¹ds,
//! for T' ∈ Sⁿ, B ∈ R^{m×k}, A, C ∈ Sᵏ, stacked as
//! svec T' | vec B (column-major, (i, j) at j·m + i) | svec A | svec C.
//! m = 0 is the ORE cone (`OpRelEntropyConeT(n)`, X = A, Y = C).
//!
//! Barrier F = −logdet(T' − G) − logdet A − logdet C, ν = n + 2k, used as
//! the ORE cone uses its own: Hs = (μ∇²F(s))⁻¹ and a Dikin dual test.
//!
//! Everything is evaluated in the ORE's eigenbasis of (A, C): A = LLᵀ,
//! L⁻¹CL⁻ᵀ = UΛUᵀ, Q = L⁻ᵀU, P = LU = Q⁻ᵀ, Π = I_m ⊕ Q, B̂ = BQ. There
//! N = Q·diag(ñ)·Qᵀ, ñ = g[λ, 1, 1] (divided differences of g = −log), and
//! Ĝ = ΠᵀGΠ = [B̂ñB̂ᵀ B̂·diag(m̃); · g(Λ)], m̃ = −g[λ, 1], Ŝ = ΠᵀT'Π − Ĝ,
//! F = −logdet Ŝ − 3 logdet A + Σ g(λᵢ). The VV block is the ORE's: an ORE
//! oracle with W = Ŵ_VV (`vv`) gives its kernels and S₀ terms. The U border
//! carries single divided differences of g, φ = x log x and χ = x² log x with
//! nodes 1, 1 appended (`Nodes`), each of fixed sign:
//!   DĜ: VV c∘Ã + g⁽¹⁾∘C̃ (the ORE's), UV −Ĥ_B·diag(g[λ,1]) + B̂(eᴬ∘Ã + eᶜ∘C̃),
//!       UU Ĥ_BñB̂ᵀ + B̂ñĤ_Bᵀ + B̂(c̃∘Ã + ã∘C̃)B̂ᵀ, with ã = g[λᵢ,λⱼ,1,1],
//!       c̃ = φ[λᵢ,λⱼ,1,1], eᶜ = −g[λᵢ,λⱼ,1], eᴬᵢⱼ = λⱼ·g[λᵢ,λⱼ,1];
//!   D²N: K_CC = g[λᵢ,λₗ,λⱼ,1,1], K_AC = φ[…], K_AA = −χ[…];
//!   the second-order term of N(A − C) (the UV block) along (Ã, C̃):
//!       T₂ₚ_q = Σᵣ (L₀C̃ₚᵣ + L₁Ãₚᵣ)(C̃ᵣ_q − λ_qÃᵣ_q), L₀ = −g[λₚ,λᵣ,λ_q,1],
//!       L₁ = −φ[λₚ,λᵣ,λ_q,1].
//! ∇²F = Lᵀ·diag(Ŵ⊗Ŵ, S₀)·L with L = [I −DĜ; 0 I] as for the ORE; S₀ has
//! order mk + 2t(k). Its B block has no cheap bound (the log-dets give it no
//! curvature), so the dual test takes S₀'s factor.

use super::oprelentrcone::{
    atanh_ratio, central_point, cholesky, congruence, gemm, heavy, join_if, lower_inverse, pieces,
    product, search, smat, spd_inverse, svec, sym_eig, symmetrize, tri, zeros, OpRelEntropyRows,
    Oracle as Ore,
};
use super::*;
use crate::algebra::*;
use rayon::prelude::*;

/// `op(a)·op(b)` (`rows × cols`, inner `k`; zero when a dimension is).
fn mul<T: FloatT>(
    t: (u8, u8),
    (rows, cols, k): (usize, usize, usize),
    a: (&[T], usize),
    b: (&[T], usize),
) -> Vec<T> {
    let mut c = vec![T::zero(); rows * cols];
    if rows * cols * k > 0 {
        product(t, (rows, cols, k), a, b, &mut c);
    }
    c
}

/// ΠᵀXΠ (`left_t`) or ΠXΠᵀ for Π = I_m ⊕ B (B k×k, X n×n symmetric).
fn pi_congruence<T: FloatT>(n: usize, m: usize, b: &[T], x: &[T], left_t: bool, out: &mut [T]) {
    let k = n - m;
    let mut xvv = vec![T::zero(); k * k];
    for j in 0..k {
        for i in 0..k {
            xvv[i + j * k] = x[(m + i) + (m + j) * n];
        }
    }
    let mut tmp = vec![T::zero(); k * k];
    let mut ovv = vec![T::zero(); k * k];
    congruence(k, b, &xvv, left_t, &mut tmp, &mut ovv);
    for j in 0..k {
        for i in 0..k {
            out[(m + i) + (m + j) * n] = ovv[i + j * k];
        }
    }
    if m == 0 {
        return;
    }
    for j in 0..m {
        for i in 0..m {
            out[i + j * n] = x[i + j * n];
        }
    }
    let xuv: Vec<T> = (0..m * k).map(|e| x[e % m + (m + e / m) * n]).collect();
    let tb = if left_t { b'N' } else { b'T' };
    let uv = mul((b'N', tb), (m, k, k), (&xuv, m), (b, k));
    for j in 0..k {
        for i in 0..m {
            out[i + (m + j) * n] = uv[i + j * m];
            out[(m + j) + i * n] = uv[i + j * m];
        }
    }
}

fn clustered<T: FloatT>(lo: T, hi: T) -> bool {
    (hi - lo) * (8.0).as_T() <= hi + lo
}

/// Taylor weights 1/(r+j), 1/((r+j)(r+j−1)), 1/((r+j)(r+j−1)(r+j−2)) of
/// the divided differences of order r (`taylor`).
#[derive(Clone)]
struct Weights<T>(Vec<Vec<[T; 3]>>);

impl<T: FloatT> Weights<T> {
    fn new() -> Self {
        // |v| ≤ 1/8: about three bits a term
        let len = T::precision_bits() as usize / 2 + 64;
        Self(
            (0..5)
                .map(|r| (0..len).map(|j| Self::at(r, j)).collect())
                .collect(),
        )
    }

    fn at(r: usize, j: usize) -> [T; 3] {
        let q = r + j;
        let w0 = if q > 0 {
            T::one() / q.as_T()
        } else {
            T::zero()
        };
        let w1 = if q > 1 {
            w0 / (q - 1).as_T()
        } else {
            T::zero()
        };
        let w2 = if q > 2 {
            w1 / (q - 2).as_T()
        } else {
            T::zero()
        };
        [w0, w1, w2]
    }

    fn get(&self, r: usize, j: usize) -> [T; 3] {
        self.0[r].get(j).copied().unwrap_or_else(|| Self::at(r, j))
    }
}

/// Divided differences of g = −log, φ = x log x and χ = x² log x (the first
/// `want`; φ needs order r ≥ 2, χ r ≥ 3) at the sorted nodes x (order
/// r = len − 1) with hi − lo ≤ (hi + lo)/8, by their Taylor series about
/// c = (lo + hi)/2: f[x] = Σⱼ f⁽ʳ⁺ʲ⁾(c)/(r+j)!·hⱼ(x − c), hⱼ the complete
/// homogeneous symmetric polynomial. With v = (x − c)/c,
/// g[x] = (−1)ʳc⁻ʳ·Σⱼ(−1)ʲhⱼ(v)/(r+j), φ[x] = (−1)ʳc¹⁻ʳ·Σⱼ(−1)ʲhⱼ(v)/((r+j)(r+j−1)),
/// χ[x] = 2(−1)ʳ⁻¹c²⁻ʳ·Σⱼ(−1)ʲhⱼ(v)/((r+j)(r+j−1)(r+j−2)).
fn taylor<T: FloatT>(x: &[T], want: usize, w: &Weights<T>) -> [T; 3] {
    let r = x.len() - 1;
    debug_assert!(r >= 1 && (want < 2 || r >= 2) && (want < 3 || r >= 3));
    let (lo, hi) = (x[0], x[r]);
    let two = T::one() + T::one();
    let c = (lo + hi) / two;
    let mut v = [T::zero(); 5];
    for (v, &x) in v.iter_mut().zip(x) {
        *v = (x - c) / c;
    }
    let vmax = (hi - lo) / (hi + lo);
    let mut e = [T::one(); 5];
    let mut sum = [T::zero(); 3];
    // bound on |hⱼ(v)|: C(j+r, r)·vmaxʲ
    let mut bound = T::one();
    let eps = T::epsilon();
    let mut j = 0usize;
    loop {
        if j > 0 {
            e[0] *= v[0];
            for i in 1..=r {
                e[i] = v[i] * e[i] + e[i - 1];
            }
            bound *= vmax * (j + r).as_T() / j.as_T();
        }
        let wj = w.get(r, j);
        for f in 0..want {
            let t = e[r] * wj[f];
            if j % 2 == 1 {
                sum[f] -= t;
            } else {
                sum[f] += t;
            }
        }
        if bound * wj[0] <= eps * sum[0].abs() || j > 4096 {
            break;
        }
        j += 1;
    }
    let mut cr = T::one();
    for _ in 0..r {
        cr *= c;
    }
    let sgn = if r % 2 == 1 { -T::one() } else { T::one() };
    [
        sgn * sum[0] / cr,
        sgn * sum[1] * c / cr,
        -two * sgn * sum[2] * (c * c) / cr,
    ]
}

/// The nodes λ₀, …, λ_{k−1} and 1 (index k) of the border kernels, with
/// the first divided differences of g between every pair.
#[derive(Clone)]
struct Nodes<T> {
    lam: Vec<T>,
    /// g[ℓₐ, ℓ_b] ((k+1)×(k+1)), from ln(ℓₐ/ℓ_b) or, for close nodes, atanh
    pair: Vec<T>,
    w: Weights<T>,
}

impl<T: FloatT> Nodes<T> {
    fn new(lam: &[T]) -> Self {
        let k = lam.len();
        let kp = k + 1;
        let ell = |a: usize| if a < k { lam[a] } else { T::one() };
        let mut pair = vec![T::zero(); kp * kp];
        let two = T::one() + T::one();
        for b in 0..kp {
            for a in 0..=b {
                let (x, y) = (ell(a), ell(b));
                let s = x + y;
                let r = (x - y) / s;
                let v = if r.abs() <= (0.125).as_T() {
                    -two / s * atanh_ratio(r)
                } else {
                    -(x / y).ln() / (x - y)
                };
                pair[a + b * kp] = v;
                pair[b + a * kp] = v;
            }
        }
        Self {
            lam: lam.to_vec(),
            pair,
            w: Weights::new(),
        }
    }

    fn ell(&self, a: usize) -> T {
        self.lam.get(a).copied().unwrap_or(T::one())
    }

    /// g, φ and χ divided differences (the first `want`) at the nodes `ids`
    /// (3 to 5 of them). Clustered nodes take the Taylor series; otherwise
    /// g comes from the recurrence on the sorted nodes (`entry`) and
    /// φ[x₀..xᵣ] = −x₀g[x₀..xᵣ] − g[x₁..xᵣ], χ[x₀..xᵣ] = x₀φ[x₀..xᵣ] + φ[x₁..xᵣ]
    /// (Leibniz with x₀ the least node: the subtracted share is at most
    /// (r−1)/r, so at most log₂ r bits cancel).
    fn divided(&self, ids: &[usize], want: usize) -> [T; 3] {
        let len = ids.len();
        let mut id = [0usize; 5];
        id[..len].copy_from_slice(ids);
        for i in 1..len {
            let mut j = i;
            while j > 0 && self.ell(id[j - 1]) > self.ell(id[j]) {
                id.swap(j - 1, j);
                j -= 1;
            }
        }
        let mut x = [T::zero(); 5];
        for i in 0..len {
            x[i] = self.ell(id[i]);
        }
        let (x, id) = (&x[..len], &id[..len]);
        let r = len - 1;
        if clustered(x[0], x[r]) {
            return taylor(x, want, &self.w);
        }
        let mut memo = [[None; 5]; 5];
        let g0 = self.entry(x, id, (0, r), &mut memo);
        let mut out = [g0, T::zero(), T::zero()];
        if want > 1 {
            let g1 = self.entry(x, id, (1, r), &mut memo);
            out[1] = -x[0] * g0 - g1;
            if want > 2 {
                let g2 = self.entry(x, id, (2, r), &mut memo);
                out[2] = x[0] * out[1] - x[1] * g1 - g2;
            }
        }
        out
    }

    /// g[xᵢ..xⱼ] at the sorted nodes x (indices `id`): the Taylor series
    /// where they cluster, else the recurrence on the widest gap, whose two
    /// terms share their sign (order-r divided differences of g are
    /// (−1)ʳ∫₀^∞ dt/Π(xᵢ + t)) and differ by a share the gap bounds below.
    fn entry(
        &self,
        x: &[T],
        id: &[usize],
        (i, j): (usize, usize),
        memo: &mut [[Option<T>; 5]; 5],
    ) -> T {
        if j == i + 1 {
            return self.pair[id[i] + id[j] * (self.lam.len() + 1)];
        }
        if let Some(v) = memo[i][j] {
            return v;
        }
        let v = if clustered(x[i], x[j]) {
            taylor(&x[i..=j], 1, &self.w)[0]
        } else {
            (self.entry(x, id, (i + 1, j), memo) - self.entry(x, id, (i, j - 1), memo))
                / (x[j] - x[i])
        };
        memo[i][j] = Some(v);
        v
    }
}

/// The cone's change of KKT row coordinates, as `OpRelEntropyRows`: with 𝒬
/// the map of primal directions to eigen coordinates (T ↦ ΠᵀTΠ, B ↦ BQ,
/// A ↦ QᵀAQ, C ↦ QᵀCQ), (μ∇²F)⁻¹ = N·D·Nᵀ, N = 𝒬⁻¹·[I DĜ; 0 I],
/// D = diag(Ŝ⊗Ŝ, S₀⁻¹)/μ. Carries DĜ's kernels.
#[derive(Clone, Default)]
pub(crate) struct OpRelEntropyLimitRows<T> {
    n: usize,
    m: usize,
    q: Vec<T>,
    /// B̂ = BQ (m×k)
    bh: Vec<T>,
    /// ñ = g[λ, 1, 1] and m̃ = −g[λ, 1] (k)
    nt: Vec<T>,
    mg: Vec<T>,
    /// ã, c̃, eᴬ, eᶜ and the ORE's c, g⁽¹⁾ (k×k)
    at: Vec<T>,
    ct: Vec<T>,
    ea: Vec<T>,
    ec: Vec<T>,
    c: Vec<T>,
    d1: Vec<T>,
}

type Parts<T> = (Vec<T>, Vec<T>, Vec<T>, Vec<T>);

impl<T: FloatT> OpRelEntropyLimitRows<T> {
    fn k(&self) -> usize {
        self.n - self.m
    }

    /// (T n×n, B m×k, A k×k, C k×k) of a cone vector.
    fn split(&self, v: &[T]) -> Parts<T> {
        let (n, m, k) = (self.n, self.m, self.k());
        let (tn, mk, tk) = (triangular_number(n), m * k, triangular_number(k));
        let mut t = vec![T::zero(); n * n];
        smat(n, &v[..tn], &mut t);
        let h = v[tn..tn + mk].to_vec();
        let mut a = vec![T::zero(); k * k];
        smat(k, &v[tn + mk..tn + mk + tk], &mut a);
        let mut c = vec![T::zero(); k * k];
        smat(k, &v[tn + mk + tk..], &mut c);
        (t, h, a, c)
    }

    fn join(&self, (t, h, a, c): (&[T], &[T], &[T], &[T]), v: &mut [T]) {
        let (n, m, k) = (self.n, self.m, self.k());
        let (tn, mk, tk) = (triangular_number(n), m * k, triangular_number(k));
        svec(n, t, &mut v[..tn]);
        v[tn..tn + mk].copy_from_slice(h);
        svec(k, a, &mut v[tn + mk..tn + mk + tk]);
        svec(k, c, &mut v[tn + mk + tk..]);
    }

    /// T ↦ ΠᵀTΠ, B ↦ BM, A ↦ MᵀAM (`left_t`, Π = I ⊕ M) or T ↦ ΠTΠᵀ,
    /// B ↦ BMᵀ, A ↦ MAMᵀ: with M = Q primal directions to eigen
    /// coordinates and eigen covectors back; with M = P covectors to eigen
    /// coordinates and eigen directions back.
    fn map(
        &self,
        b: &[T],
        left_t: bool,
        (t, h, a, c): (&mut [T], &mut Vec<T>, &mut [T], &mut [T]),
    ) {
        let (n, m, k) = (self.n, self.m, self.k());
        let mut out = vec![T::zero(); n * n];
        pi_congruence(n, m, b, t, left_t, &mut out);
        t.copy_from_slice(&out);
        if m > 0 {
            let tb = if left_t { b'N' } else { b'T' };
            *h = mul((b'N', tb), (m, k, k), (&h[..], m), (b, k));
        }
        let mut tmp = vec![T::zero(); k * k];
        let mut o = vec![T::zero(); k * k];
        for x in [a, c] {
            congruence(k, b, x, left_t, &mut tmp, &mut o);
            x.copy_from_slice(&o);
        }
    }

    /// DĜ[H, A, C] (eigen coordinates, n×n).
    fn dg(&self, h: &[T], a: &[T], c: &[T], out: &mut [T]) {
        let (n, m, k) = (self.n, self.m, self.k());
        for j in 0..k {
            for i in 0..k {
                let ij = i + j * k;
                out[(m + i) + (m + j) * n] = self.c[ij] * a[ij] + self.d1[ij] * c[ij];
            }
        }
        if m == 0 {
            return;
        }
        let kk = k * k;
        let (mut n1, mut t1) = (vec![T::zero(); kk], vec![T::zero(); kk]);
        for ij in 0..kk {
            n1[ij] = self.ct[ij] * a[ij] + self.at[ij] * c[ij];
            t1[ij] = self.ea[ij] * a[ij] + self.ec[ij] * c[ij];
        }
        let bh = &self.bh;
        let hn: Vec<T> = (0..m * k).map(|e| h[e] * self.nt[e / m]).collect();
        let x = mul((b'N', b'T'), (m, m, k), (&hn, m), (bh, m));
        let y = mul((b'N', b'N'), (m, k, k), (bh, m), (&n1, k));
        let z = mul((b'N', b'T'), (m, m, k), (&y, m), (bh, m));
        let bt = mul((b'N', b'N'), (m, k, k), (bh, m), (&t1, k));
        let half: T = (0.5).as_T();
        for j in 0..m {
            for i in 0..m {
                out[i + j * n] = x[i + j * m] + x[j + i * m] + half * (z[i + j * m] + z[j + i * m]);
            }
        }
        for j in 0..k {
            for i in 0..m {
                let v = h[i + j * m] * self.mg[j] + bt[i + j * m];
                out[i + (m + j) * n] = v;
                out[(m + j) + i * n] = v;
            }
        }
    }

    /// DĜᵀ[R] for a symmetric n×n R: the eigen covector (H, A, C).
    fn dg_adj(&self, r: &[T], oh: &mut [T], oa: &mut [T], oc: &mut [T]) {
        let (n, m, k) = (self.n, self.m, self.k());
        for j in 0..k {
            for i in 0..k {
                let ij = i + j * k;
                let v = r[(m + i) + (m + j) * n];
                oa[ij] = self.c[ij] * v;
                oc[ij] = self.d1[ij] * v;
            }
        }
        if m == 0 {
            return;
        }
        let bh = &self.bh;
        let ruu: Vec<T> = (0..m * m).map(|e| r[e % m + e / m * n]).collect();
        let ruv: Vec<T> = (0..m * k).map(|e| r[e % m + (m + e / m) * n]).collect();
        let rb = mul((b'N', b'N'), (m, k, m), (&ruu, m), (bh, m));
        let two = T::one() + T::one();
        for e in 0..m * k {
            oh[e] = two * (rb[e] * self.nt[e / m] + ruv[e] * self.mg[e / m]);
        }
        let mm = mul((b'T', b'N'), (k, k, m), (bh, m), (&ruv, m));
        let om = mul((b'T', b'N'), (k, k, m), (bh, m), (&rb, m));
        let half: T = (0.5).as_T();
        for j in 0..k {
            for i in 0..k {
                let (ij, ji) = (i + j * k, j + i * k);
                let s = half * (om[ij] + om[ji]);
                oa[ij] += self.ea[ij] * mm[ij] + self.ea[ji] * mm[ji] + self.ct[ij] * s;
                oc[ij] += self.ec[ij] * mm[ij] + self.ec[ji] * mm[ji] + self.at[ij] * s;
            }
        }
    }

    /// v ← N⁻¹v = [I −DĜ; 0 I]·𝒬v
    pub(crate) fn to_eigen(&self, v: &mut [T]) {
        let (mut t, mut h, mut a, mut c) = self.split(v);
        self.map(&self.q, true, (&mut t, &mut h, &mut a, &mut c));
        let mut d = vec![T::zero(); t.len()];
        self.dg(&h, &a, &c, &mut d);
        t.iter_mut().zip(&d).for_each(|(t, &d)| *t -= d);
        self.join((&t, &h, &a, &c), v);
    }

    /// v ← N⁻ᵀv = 𝒬ᵀ·[I 0; −DĜᵀ I]v
    pub(crate) fn from_eigen(&self, v: &mut [T]) {
        let (mut t, mut h, mut a, mut c) = self.split(v);
        let k = self.k();
        let (mut oh, mut oa, mut oc) = (
            vec![T::zero(); h.len()],
            vec![T::zero(); k * k],
            vec![T::zero(); k * k],
        );
        self.dg_adj(&t, &mut oh, &mut oa, &mut oc);
        h.iter_mut().zip(&oh).for_each(|(x, &d)| *x -= d);
        a.iter_mut().zip(&oa).for_each(|(x, &d)| *x -= d);
        c.iter_mut().zip(&oc).for_each(|(x, &d)| *x -= d);
        self.map(&self.q, false, (&mut t, &mut h, &mut a, &mut c));
        self.join((&t, &h, &a, &c), v);
    }
}

/// The KKT row transform of an operator relative entropy cone or its limit.
#[derive(Clone)]
pub(crate) enum EigenRows<T> {
    Ore(OpRelEntropyRows<T>),
    Limit(OpRelEntropyLimitRows<T>),
}

impl<T: FloatT> Default for EigenRows<T> {
    fn default() -> Self {
        Self::Ore(OpRelEntropyRows::default())
    }
}

impl<T: FloatT> EigenRows<T> {
    /// The rows of `cone`'s current scaling.
    pub(crate) fn update(&mut self, cone: &SupportedCone<T>) {
        match cone {
            SupportedCone::OpRelEntropyCone(c) => match self {
                Self::Ore(t) => t.clone_from(c.rows()),
                s => *s = Self::Ore(c.rows().clone()),
            },
            SupportedCone::OpRelEntropyLimitCone(c) => match self {
                Self::Limit(t) => t.clone_from(c.rows()),
                s => *s = Self::Limit(c.rows().clone()),
            },
            _ => {}
        }
    }

    pub(crate) fn to_eigen(&self, v: &mut [T]) {
        match self {
            Self::Ore(t) => t.to_eigen(v),
            Self::Limit(t) => t.to_eigen(v),
        }
    }

    pub(crate) fn from_eigen(&self, v: &mut [T]) {
        match self {
            Self::Ore(t) => t.from_eigen(v),
            Self::Limit(t) => t.from_eigen(v),
        }
    }
}

/// S₀'s border data: Ŵ_UU, Ŵ_UV, Ŵ_UU·B̂, Ω = B̂ᵀŴ_UUB̂, Z = B̂ᵀŴ_UV.
#[derive(Clone, Default)]
struct Border<T> {
    wuu: Vec<T>,
    wuv: Vec<T>,
    wb: Vec<T>,
    om: Vec<T>,
    z: Vec<T>,
}

/// Barrier oracle at an interior point, in the eigenbasis of (A, C).
#[derive(Clone)]
pub(crate) struct Oracle<T> {
    rows: OpRelEntropyLimitRows<T>,
    p: Vec<T>,
    /// the ORE oracle of the VV block: λ, g, g⁽¹⁾, c and W = Ŵ_VV
    vv: Ore<T>,
    nodes: Nodes<T>,
    sh: Vec<T>,
    ls: Vec<T>,
    w: Vec<T>,
    /// (K_AA, K_AC, K_CC, L₀, L₁) per ordered index triple (built on demand)
    k3: Vec<T>,
    border: Border<T>,
    /// Lower Cholesky factor of S₀ (built on demand; None if not ≻ 0).
    s0l: Option<Option<Vec<T>>>,
    /// F(s) (the tests check the gradient against it)
    #[cfg_attr(not(test), allow(dead_code))]
    pub barrier: T,
}

impl<T: FloatT> Oracle<T> {
    /// None unless s is interior.
    pub(crate) fn new(n: usize, m: usize, s: &[T]) -> Option<Self> {
        let k = n - m;
        let (tn, mk, tk, kk) = (triangular_number(n), m * k, triangular_number(k), k * k);
        let (st, rest) = s.split_at(tn);
        let (sb, rest) = rest.split_at(mk);
        let (sa, sc) = rest.split_at(tk);
        let mut la = vec![T::zero(); kk];
        smat(k, sa, &mut la);
        if !cholesky(k, &mut la) {
            return None;
        }
        let mut li = vec![T::zero(); kk];
        lower_inverse(k, &la, &mut li);
        let mut cm = vec![T::zero(); kk];
        smat(k, sc, &mut cm);
        let mut tmp = vec![T::zero(); kk];
        let mut psi = vec![T::zero(); kk];
        congruence(k, &li, &cm, false, &mut tmp, &mut psi);
        let mut lam = vec![T::zero(); k];
        let mut u = vec![T::zero(); kk];
        if !sym_eig(k, &mut psi, &mut lam, &mut u) {
            return None;
        }
        if !lam.iter().all(|&l| l.is_finite() && l > T::zero()) {
            return None;
        }
        let mut q = vec![T::zero(); kk];
        gemm(true, false, k, &li, &u, &mut q);
        let mut p = vec![T::zero(); kk];
        gemm(false, false, k, &la, &u, &mut p);
        let g: Vec<T> = lam.iter().map(|&l| -l.ln()).collect();
        let nodes = Nodes::new(&lam);
        let mut rows = OpRelEntropyLimitRows {
            n,
            m,
            ..Default::default()
        };
        if m > 0 {
            rows.bh = mul((b'N', b'N'), (m, k, k), (sb, m), (&q, k));
            rows.nt = (0..k).map(|i| nodes.divided(&[i, k, k], 1)[0]).collect();
            rows.mg = (0..k).map(|i| -nodes.pair[i + k * (k + 1)]).collect();
        }
        let mut tm = vec![T::zero(); n * n];
        smat(n, st, &mut tm);
        let mut sh = vec![T::zero(); n * n];
        pi_congruence(n, m, &q, &tm, true, &mut sh);
        for i in 0..k {
            sh[(m + i) + (m + i) * n] -= g[i];
        }
        if m > 0 {
            let bh = &rows.bh;
            let hn: Vec<T> = (0..mk).map(|e| bh[e] * rows.nt[e / m]).collect();
            let bnb = mul((b'N', b'T'), (m, m, k), (&hn, m), (bh, m));
            let half: T = (0.5).as_T();
            for j in 0..m {
                for i in 0..m {
                    sh[i + j * n] -= half * (bnb[i + j * m] + bnb[j + i * m]);
                }
            }
            for j in 0..k {
                for i in 0..m {
                    let v = bh[i + j * m] * rows.mg[j];
                    sh[i + (m + j) * n] -= v;
                    sh[(m + j) + i * n] -= v;
                }
            }
        }
        let mut ls = sh.clone();
        if !cholesky(n, &mut ls) {
            return None;
        }
        let mut w = vec![T::zero(); n * n];
        if !spd_inverse(n, &ls, &mut w) {
            return None;
        }
        let two = T::one() + T::one();
        let mut logdet_s = T::zero();
        let mut logdet_a = T::zero();
        for i in 0..n {
            logdet_s += two * ls[i + i * n].ln();
        }
        for i in 0..k {
            logdet_a += two * la[i + i * k].ln();
        }
        let three: T = (3.0).as_T();
        let barrier = -logdet_s - three * logdet_a + g.iter().fold(T::zero(), |a, &v| a + v);
        let mut wvv = vec![T::zero(); kk];
        for j in 0..k {
            for i in 0..k {
                wvv[i + j * k] = w[(m + i) + (m + j) * n];
            }
        }
        let vv = Ore::eigen_part(lam, wvv);
        rows.q = q;
        rows.c = vv.c.clone();
        rows.d1 = vv.d1.clone();
        if m > 0 {
            for v in [&mut rows.at, &mut rows.ct, &mut rows.ea, &mut rows.ec] {
                v.resize(kk, T::zero());
            }
            let lam = &vv.lam;
            for j in 0..k {
                for i in 0..=j {
                    let [a4, c4, _] = nodes.divided(&[i, j, k, k], 2);
                    let g3 = nodes.divided(&[i, j, k], 1)[0];
                    for (ij, lj) in [(i + j * k, lam[j]), (j + i * k, lam[i])] {
                        rows.at[ij] = a4;
                        rows.ct[ij] = c4;
                        rows.ec[ij] = -g3;
                        rows.ea[ij] = lj * g3;
                    }
                }
            }
        }
        Some(Self {
            rows,
            p,
            vv,
            nodes,
            sh,
            ls,
            w,
            k3: Vec::new(),
            border: Border::default(),
            s0l: None,
            barrier,
        })
    }

    /// The second-order kernels and border data of S₀.
    fn ensure_k3(&mut self) {
        self.vv.ensure_d2();
        let (n, m, k) = (self.rows.n, self.rows.m, self.rows.k());
        if m == 0 || !self.k3.is_empty() {
            return;
        }
        let ids: Vec<[usize; 3]> = (0..k)
            .flat_map(|a| (a..k).flat_map(move |b| (b..k).map(move |c| [a, b, c])))
            .collect();
        let mut x = vec![T::zero(); 5 * ids.len()];
        let nodes = &self.nodes;
        pieces::<T>(&mut x, 5 * k, 1024 * ids.len(), |piece, out| {
            for (o, t) in out.chunks_exact_mut(5).zip(&ids[piece * k..]) {
                let [g5, p5, c5] = nodes.divided(&[t[0], t[1], t[2], k, k], 3);
                let [g4, p4, _] = nodes.divided(&[t[0], t[1], t[2], k], 2);
                o.copy_from_slice(&[-c5, p5, g5, -g4, -p4]);
            }
        });
        let mut k3 = vec![T::zero(); 5 * k * k * k];
        for (&[a, b, c], v) in ids.iter().zip(x.chunks_exact(5)) {
            for (i, j, l) in [
                (a, b, c),
                (a, c, b),
                (b, a, c),
                (b, c, a),
                (c, a, b),
                (c, b, a),
            ] {
                let at = 5 * (i + k * (j + k * l));
                k3[at..at + 5].copy_from_slice(v);
            }
        }
        self.k3 = k3;
        let (w, bh) = (&self.w, &self.rows.bh);
        let wuu: Vec<T> = (0..m * m).map(|e| w[e % m + e / m * n]).collect();
        let wuv: Vec<T> = (0..m * k).map(|e| w[e % m + (m + e / m) * n]).collect();
        let wb = mul((b'N', b'N'), (m, k, m), (&wuu, m), (bh, m));
        let mut om = mul((b'T', b'N'), (k, k, m), (bh, m), (&wb, m));
        symmetrize(k, &mut om);
        let z = mul((b'T', b'N'), (k, k, m), (bh, m), (&wuv, m));
        self.border = Border {
            wuu,
            wuv,
            wb,
            om,
            z,
        };
    }

    /// Primal gradient ∇F(s).
    pub(crate) fn gradient(&self, out: &mut [T]) {
        let rows = &self.rows;
        let (m, k) = (rows.m, rows.k());
        let (mut gh, mut ga, mut gc) = (
            vec![T::zero(); m * k],
            vec![T::zero(); k * k],
            vec![T::zero(); k * k],
        );
        rows.dg_adj(&self.w, &mut gh, &mut ga, &mut gc);
        for i in 0..k {
            ga[i + i * k] -= T::one();
            gc[i + i * k] -= self.vv.lam[i].recip();
        }
        let mut gt: Vec<T> = self.w.iter().map(|&v| -v).collect();
        rows.map(&rows.q, false, (&mut gt, &mut gh, &mut ga, &mut gc));
        rows.join((&gt, &gh, &ga, &gc), out);
    }

    /// The border part of S₀y, y = (H, A, C) in eigen coordinates (A, C
    /// full k×k): adds to `oh` and to the unsymmetrized `ra`, `rc`. Reads only
    /// the columns `hcols` of H and the entries `pairs` of A, C (both
    /// orders), so a unit column of S₀ costs O(k). The terms are the
    /// gradients of ½⟨Ŵ, D²Ĝ[y, y]⟩ outside Ŵ_VV:
    /// ⟨Ŵ_UU, HñHᵀ⟩ + 2⟨HᵀŴ_UUB̂, N₁⟩ + ½⟨Ω, D²N[y, y]⟩ + 2⟨HᵀŴ_UV, T₁⟩
    /// + 2⟨Z, T₂⟩, N₁ = c̃∘A + ã∘C, T₁ = eᴬ∘A + eᶜ∘C and T₂ the second-order
    /// term of N(A − C) (module docs).
    #[allow(clippy::too_many_arguments)]
    fn schur_border(
        &self,
        (h, a, c): (&[T], &[T], &[T]),
        hcols: &[usize],
        pairs: &[(usize, usize)],
        oh: &mut [T],
        ra: &mut [T],
        rc: &mut [T],
    ) {
        let rows = &self.rows;
        let (m, k) = (rows.m, rows.k());
        let Border {
            wuu,
            wuv,
            wb,
            om,
            z,
        } = &self.border;
        let (nt, ct, at, ea, ec, lam) = (
            &rows.nt,
            &rows.ct,
            &rows.at,
            &rows.ea,
            &rows.ec,
            &self.vv.lam,
        );
        let two = T::one() + T::one();
        for &l in hcols {
            // ⟨Ŵ_UU, HñHᵀ⟩
            for b in 0..m {
                let hb = h[b + l * m];
                if hb == T::zero() {
                    continue;
                }
                let f = two * nt[l] * hb;
                for i in 0..m {
                    oh[i + l * m] += f * wuu[i + b * m];
                }
            }
            // (HᵀŴ_UUB̂)ₗ_q and (HᵀŴ_UV)ₗ_q
            for qq in 0..k {
                let (mut xw, mut x4) = (T::zero(), T::zero());
                for i in 0..m {
                    xw += h[i + l * m] * wb[i + qq * m];
                    x4 += h[i + l * m] * wuv[i + qq * m];
                }
                let (lq, ql) = (l + qq * k, qq + l * k);
                ra[lq] += ct[lq] * xw + two * ea[lq] * x4;
                ra[ql] += ct[ql] * xw;
                rc[lq] += at[lq] * xw + two * ec[lq] * x4;
                rc[ql] += at[ql] * xw;
            }
        }
        for &(x, y) in pairs {
            let xy = x + y * k;
            let (ax, cx) = (a[xy], c[xy]);
            let n1 = ct[xy] * ax + at[xy] * cx;
            let t1 = ea[xy] * ax + ec[xy] * cx;
            for i in 0..m {
                oh[i + y * m] += two * wb[i + x * m] * n1;
                oh[i + x * m] += two * wuv[i + y * m] * t1;
            }
            let d = cx - lam[y] * ax;
            for zz in 0..k {
                let kv = &self.k3[5 * (x + k * (y + k * zz))..][..5];
                // ½⟨Ω, D²N⟩
                let (ua, uc) = (kv[0] * ax + kv[1] * cx, kv[1] * ax + kv[2] * cx);
                let (oy, ox) = (om[y + zz * k], om[zz + x * k]);
                ra[x + zz * k] += ua * oy;
                rc[x + zz * k] += uc * oy;
                ra[zz + y * k] += ua * ox;
                rc[zz + y * k] += uc * ox;
                // 2⟨Z, T₂⟩: through the first factor (index p = zz) ...
                let f = two * z[zz + y * k] * d;
                rc[zz + x * k] += kv[3] * f;
                ra[zz + x * k] += kv[4] * f;
                // ... and the second (index q = zz)
                let s = two * z[x + zz * k] * (kv[3] * cx + kv[4] * ax);
                rc[y + zz * k] += s;
                ra[y + zz * k] -= lam[zz] * s;
            }
        }
    }

    /// S₀y for y = (H, A, C) (eigen coordinates, full matrices): the ORE's
    /// VV terms and log-dets plus the border.
    #[allow(clippy::too_many_arguments)]
    fn schur_apply(
        &self,
        y: (&[T], &[T], &[T]),
        oh: &mut [T],
        oa: &mut [T],
        oc: &mut [T],
        ws: &mut Vec<T>,
    ) {
        let (m, k) = (self.rows.m, self.rows.k());
        self.vv.schur_apply(y.1, y.2, oa, oc, ws, None);
        oh.fill(T::zero());
        if m == 0 {
            return;
        }
        let kk = k * k;
        let (mut ra, mut rc) = (vec![T::zero(); kk], vec![T::zero(); kk]);
        let cols: Vec<usize> = (0..k).collect();
        let pairs: Vec<(usize, usize)> = (0..k).flat_map(|j| (0..k).map(move |i| (i, j))).collect();
        self.schur_border(y, &cols, &pairs, oh, &mut ra, &mut rc);
        let half: T = (0.5).as_T();
        for j in 0..k {
            for i in 0..k {
                let (ij, ji) = (i + j * k, j + i * k);
                oa[ij] += half * (ra[ij] + ra[ji]);
                oc[ij] += half * (rc[ij] + rc[ji]);
            }
        }
    }

    /// Hessian product ∇²F(s)[d].
    pub(crate) fn hess_prod(&mut self, d: &[T], out: &mut [T]) {
        self.ensure_k3();
        let rows = &self.rows;
        let (n, m, k) = (rows.n, rows.m, rows.k());
        let nn = n * n;
        let (mut t, mut h, mut a, mut c) = rows.split(d);
        rows.map(&rows.q, true, (&mut t, &mut h, &mut a, &mut c));
        let mut dd = vec![T::zero(); nn];
        rows.dg(&h, &a, &c, &mut dd);
        for e in 0..nn {
            dd[e] = t[e] - dd[e];
        }
        // R = Ŵ(Ĥ_T − DĜ[y])Ŵ; outputs (R, S₀y − DĜᵀR).
        let mut r = vec![T::zero(); nn];
        congruence(n, &self.w, &dd, false, &mut t, &mut r);
        let (mut oh, mut oa, mut oc) = (
            vec![T::zero(); m * k],
            vec![T::zero(); k * k],
            vec![T::zero(); k * k],
        );
        let mut ws = Vec::new();
        self.schur_apply((&h, &a, &c), &mut oh, &mut oa, &mut oc, &mut ws);
        rows.dg_adj(&r, &mut h, &mut a, &mut c);
        oh.iter_mut().zip(&h).for_each(|(o, &v)| *o -= v);
        oa.iter_mut().zip(&a).for_each(|(o, &v)| *o -= v);
        oc.iter_mut().zip(&c).for_each(|(o, &v)| *o -= v);
        rows.map(&rows.q, false, (&mut r, &mut oh, &mut oa, &mut oc));
        rows.join((&r, &oh, &oa, &oc), out);
    }

    /// S₀ (column-major, order mk + 2t(k)) in eigen coordinates
    /// (vec H | svec A | svec C): the ORE's block of (A, C) plus the
    /// border's unit columns.
    fn schur_matrix(&mut self) -> Vec<T> {
        self.ensure_k3();
        let (m, k) = (self.rows.m, self.rows.k());
        let core = self.vv.schur_matrix();
        if m == 0 {
            return core;
        }
        let (mk, tk, kk) = (m * k, triangular_number(k), k * k);
        let (nc, ny) = (2 * tk, mk + 2 * tk);
        let mut s0 = zeros::<T>(ny * ny);
        for j in 0..nc {
            s0[(mk + j) * ny + mk..][..nc].copy_from_slice(&core[j * nc..(j + 1) * nc]);
        }
        drop(core);
        let upper: Vec<(usize, usize)> =
            (0..k).flat_map(|j| (0..=j).map(move |i| (i, j))).collect();
        let this = &*self;
        pieces::<T>(&mut s0, k * ny, 32 * k * ny, |piece, cols| {
            let (mut h, mut oh) = (vec![T::zero(); mk], vec![T::zero(); mk]);
            let (mut a, mut c) = (vec![T::zero(); kk], vec![T::zero(); kk]);
            let (mut ra, mut rc) = (vec![T::zero(); kk], vec![T::zero(); kk]);
            let mut v = vec![T::zero(); tk];
            for (cidx, column) in cols.chunks_exact_mut(ny).enumerate() {
                let col = piece * k + cidx;
                for x in [&mut h, &mut oh, &mut a, &mut c, &mut ra, &mut rc] {
                    x.fill(T::zero());
                }
                let (hcols, pairs) = if col < mk {
                    h[col] = T::one();
                    (vec![col / m], Vec::new())
                } else {
                    let e = col - mk;
                    let (i, j) = upper[e % tk];
                    let x = if e < tk { &mut a } else { &mut c };
                    let u = if i == j { T::one() } else { T::FRAC_1_SQRT_2() };
                    x[i + j * k] = u;
                    x[j + i * k] = u;
                    let pairs = if i == j {
                        vec![(i, i)]
                    } else {
                        vec![(i, j), (j, i)]
                    };
                    (Vec::new(), pairs)
                };
                this.schur_border((&h, &a, &c), &hcols, &pairs, &mut oh, &mut ra, &mut rc);
                column[..mk].iter_mut().zip(&oh).for_each(|(o, &x)| *o += x);
                // svec symmetrizes the raw partials
                for (part, r) in [(0, &ra), (1, &rc)] {
                    svec(k, r, &mut v);
                    column[mk + part * tk..][..tk]
                        .iter_mut()
                        .zip(&v)
                        .for_each(|(o, &x)| *o += x);
                }
            }
        });
        symmetrize(ny, &mut s0);
        s0
    }

    /// Lower Cholesky factor of S₀.
    fn schur_factor(&mut self) -> Option<Vec<T>> {
        let mut s0 = self.schur_matrix();
        let ny = self.ny();
        let timer = crate::receipt::start();
        let ok = cholesky(ny, &mut s0);
        crate::receipt::finish("ore.schur_cholesky", timer);
        ok.then_some(s0)
    }

    fn ny(&self) -> usize {
        let k = self.rows.k();
        self.rows.m * k + 2 * triangular_number(k)
    }

    /// S₀⁻¹ (column-major) in eigen coordinates.
    pub(crate) fn schur_inverse(&mut self) -> Option<Vec<T>> {
        let ny = self.ny();
        let l = self.schur_factor()?;
        let mut s0i = zeros::<T>(ny * ny);
        let timer = crate::receipt::start();
        let ok = spd_inverse(ny, &l, &mut s0i) && s0i.iter().all(|v| v.is_finite());
        crate::receipt::finish("ore.schur_inverse", timer);
        ok.then_some(s0i)
    }

    /// ⟨z, (∇²F)⁻¹z⟩ = t + vᵀS₀⁻¹v with t = tr(ŜR̂ŜR̂) = ‖L_ŜᵀR̂L_Ŝ‖²,
    /// (R̂, ŷ) = 𝒬⁻ᵀz and v = ŷ + DĜᵀ[R̂]. Returns (t, v).
    fn dual_parts(&self, z: &[T]) -> (T, Vec<T>) {
        let rows = &self.rows;
        let (n, m, k) = (rows.n, rows.m, rows.k());
        let (mk, tk) = (m * k, triangular_number(k));
        let (mut t, mut h, mut a, mut c) = rows.split(z);
        rows.map(&self.p, true, (&mut t, &mut h, &mut a, &mut c));
        let (mut oh, mut oa, mut oc) = (
            vec![T::zero(); mk],
            vec![T::zero(); k * k],
            vec![T::zero(); k * k],
        );
        rows.dg_adj(&t, &mut oh, &mut oa, &mut oc);
        let mut x = vec![T::zero(); n * n];
        let mut tmp = vec![T::zero(); n * n];
        congruence(n, &self.ls, &t, true, &mut tmp, &mut x);
        let tt = x.iter().fold(T::zero(), |a, &t| a + t * t);
        let mut v = vec![T::zero(); mk + 2 * tk];
        for e in 0..mk {
            v[e] = h[e] + oh[e];
        }
        a.iter_mut().zip(&oa).for_each(|(a, &d)| *a += d);
        c.iter_mut().zip(&oc).for_each(|(c, &d)| *c += d);
        svec(k, &a, &mut v[mk..mk + tk]);
        svec(k, &c, &mut v[mk + tk..]);
        (tt, v)
    }

    /// ⟨z, (∇²F)⁻¹z⟩ through S₀'s factor.
    pub(crate) fn dual_form(&mut self, z: &[T]) -> Option<T> {
        let (t, v) = self.dual_parts(z);
        Some(t + self.schur_norm(v)?)
    }

    /// wᵀS₀⁻¹w = ‖L₀⁻¹w‖² by forward substitution (S₀'s factor built once
    /// per oracle).
    fn schur_norm(&mut self, mut w: Vec<T>) -> Option<T> {
        if self.s0l.is_none() {
            self.s0l = Some(self.schur_factor());
        }
        let l = self.s0l.as_ref().unwrap().as_ref()?;
        let ny = w.len();
        let mut q = T::zero();
        for j in 0..ny {
            let wj = w[j] / l[j + j * ny];
            q += wj * wj;
            for i in j + 1..ny {
                w[i] -= l[i + j * ny] * wj;
            }
        }
        Some(q)
    }

    /// `ok(⟨z, (∇²F)⁻¹z⟩)` for a monotone `ok` (true below a limit): the
    /// T part t is a lower bound, else S₀'s factor decides.
    pub(crate) fn dual_ok(&mut self, z: &[T], ok: impl Fn(T) -> bool) -> bool {
        let (t, v) = self.dual_parts(z);
        ok(t) && self.schur_norm(v).is_some_and(|q| ok(t + q))
    }
}

#[derive(Clone)]
pub struct OpRelEntropyLimitConeData<T> {
    μ: T,
    s: Vec<T>,
    z: Vec<T>,
    /// Hs = N·D·Nᵀ (`OpRelEntropyLimitRows`): N's parts, P, Ŝ and S₀⁻¹
    rows: OpRelEntropyLimitRows<T>,
    p: Vec<T>,
    sh: Vec<T>,
    s0i: Vec<T>,
    work: Vec<T>,
    /// The scaling's oracle at s (kernels built), for the corrector's ∇²F(s).
    at_s: Option<Oracle<T>>,
}

#[derive(Clone)]
pub struct OpRelEntropyLimitCone<T> {
    pub n: usize,
    pub m: usize,
    data: Box<OpRelEntropyLimitConeData<T>>,
}

impl<T: FloatT> OpRelEntropyLimitCone<T> {
    pub fn new(n: usize, m: usize) -> Self {
        let d = Self::size(n, m);
        Self {
            n,
            m,
            data: Box::new(OpRelEntropyLimitConeData {
                μ: T::one(),
                s: vec![T::zero(); d],
                z: vec![T::zero(); d],
                rows: OpRelEntropyLimitRows::default(),
                p: Vec::new(),
                sh: Vec::new(),
                s0i: Vec::new(),
                work: vec![T::zero(); d],
                at_s: None,
            }),
        }
    }

    /// t(n) + m(n − m) + 2t(n − m)
    pub(crate) fn size(n: usize, m: usize) -> usize {
        let k = n - m;
        triangular_number(n) + m * k + 2 * triangular_number(k)
    }

    fn nu(&self) -> T {
        (3 * self.n - 2 * self.m).as_T()
    }

    /// ∇³F(s)[h, h] by central differences of the gradient along h/‖h‖ₛ.
    #[cfg(test)]
    fn third_order(&self, h: &[T]) -> Option<Vec<T>> {
        let s = &self.data.s;
        let mut o = Oracle::new(self.n, self.m, s)?;
        let mut hh = vec![T::zero(); h.len()];
        o.hess_prod(h, &mut hh);
        let norm2 = hh.dot(h);
        if !(norm2 > T::zero()) || !norm2.is_finite() {
            return None;
        }
        let eps = T::epsilon().sqrt().sqrt().min((0.25).as_T());
        let step = eps / norm2.sqrt();
        let mut g0 = vec![T::zero(); h.len()];
        o.gradient(&mut g0);
        let mut out = vec![T::zero(); h.len()];
        let mut w = vec![T::zero(); h.len()];
        for sign in [T::one(), -T::one()] {
            w.waxpby(T::one(), s, sign * step, h);
            let p = Oracle::new(self.n, self.m, &w)?;
            p.gradient(&mut hh);
            out.iter_mut().zip(&hh).for_each(|(o, &g)| *o += g);
        }
        let scale = norm2 / (eps * eps);
        out.iter_mut()
            .zip(&g0)
            .for_each(|(o, &g)| *o = (*o - g - g) * scale);
        out.iter().all(|v| v.is_finite()).then_some(out)
    }

    #[cfg(test)]
    fn third_mixed(&self, a: &[T], b: &[T]) -> Option<Vec<T>> {
        self.third_mixed_at(self.mixed_points(None, a, false)?, b, false)
    }

    /// The points s ± ta of [`Self::third_mixed_at`] with ‖ta‖ₛ = ε_mach^¼,
    /// as for the ORE cone.
    #[allow(clippy::type_complexity)]
    fn mixed_points(
        &self,
        at_s: Option<&mut Oracle<T>>,
        a: &[T],
        paired: bool,
    ) -> Option<(T, Oracle<T>, Oracle<T>)> {
        let s = &self.data.s;
        let mut fresh = None;
        let o = match at_s {
            Some(o) => o,
            None => fresh.insert(Oracle::new(self.n, self.m, s)?),
        };
        let mut ha = vec![T::zero(); a.len()];
        o.hess_prod(a, &mut ha);
        let norm2 = ha.dot(a);
        if !(norm2 > T::zero()) || !norm2.is_finite() {
            return None;
        }
        let eps = T::epsilon().sqrt().sqrt().min((0.1).as_T());
        let t = eps / norm2.sqrt();
        let point = |sign: T| {
            let mut w = vec![T::zero(); a.len()];
            w.waxpby(T::one(), s, sign * t, a);
            let mut p = Oracle::new(self.n, self.m, &w)?;
            p.ensure_k3();
            Some(p)
        };
        let (plus, minus) = join_if(paired, || point(T::one()), || point(-T::one()));
        Some((t, plus?, minus?))
    }

    /// ∇³F(s)[a, b] = d/dt ∇²F(s + ta)b by central differences.
    fn third_mixed_at(
        &self,
        (t, mut plus, mut minus): (T, Oracle<T>, Oracle<T>),
        b: &[T],
        paired: bool,
    ) -> Option<Vec<T>> {
        let prod = |o: &mut Oracle<T>| {
            let mut h = vec![T::zero(); b.len()];
            o.hess_prod(b, &mut h);
            h
        };
        let (hp, hm) = join_if(paired, || prod(&mut plus), || prod(&mut minus));
        let mut out = vec![T::zero(); b.len()];
        for (sign, h) in [(T::one(), &hp), (-T::one(), &hm)] {
            out.iter_mut().zip(h).for_each(|(o, &h)| *o += sign * h);
        }
        let scale = (t + t).recip();
        out.iter_mut().for_each(|o| *o *= scale);
        out.iter().all(|v| v.is_finite()).then_some(out)
    }

    /// y = Hs x = N·D·Nᵀx.
    fn hs_prod(&self, y: &mut [T], x: &[T]) {
        let data = &self.data;
        let rows = &data.rows;
        let (n, m, k) = (self.n, self.m, self.n - self.m);
        let (mk, tk, kk) = (m * k, triangular_number(k), k * k);
        let ny = mk + 2 * tk;
        let (mut t, mut h, mut a, mut c) = rows.split(x);
        rows.map(&data.p, true, (&mut t, &mut h, &mut a, &mut c));
        let (mut oh, mut oa, mut oc) = (
            vec![T::zero(); mk],
            vec![T::zero(); kk],
            vec![T::zero(); kk],
        );
        rows.dg_adj(&t, &mut oh, &mut oa, &mut oc);
        let mut v = vec![T::zero(); ny];
        for e in 0..mk {
            v[e] = h[e] + oh[e];
        }
        a.iter_mut().zip(&oa).for_each(|(a, &d)| *a += d);
        c.iter_mut().zip(&oc).for_each(|(c, &d)| *c += d);
        svec(k, &a, &mut v[mk..mk + tk]);
        svec(k, &c, &mut v[mk + tk..]);
        let inv = data.μ.recip();
        // S₀⁻¹v in row pieces: each entry sums its columns in order
        let mut u = vec![T::zero(); ny];
        let s0i = &data.s0i;
        pieces::<T>(&mut u, n, ny * ny, |piece, out| {
            let i0 = piece * n;
            out.fill(T::zero());
            for (j, &vj) in v.iter().enumerate() {
                let col = &s0i[j * ny + i0..][..out.len()];
                for (y, &c) in out.iter_mut().zip(col) {
                    *y += c * vj;
                }
            }
        });
        u.iter_mut().for_each(|u| *u *= inv);
        let (mut tmp, mut b) = (vec![T::zero(); n * n], vec![T::zero(); n * n]);
        congruence(n, &data.sh, &t, false, &mut tmp, &mut b);
        b.iter_mut().for_each(|b| *b *= inv);
        let mut uh = u[..mk].to_vec();
        smat(k, &u[mk..mk + tk], &mut a);
        smat(k, &u[mk + tk..], &mut c);
        let mut d = vec![T::zero(); n * n];
        rows.dg(&uh, &a, &c, &mut d);
        b.iter_mut().zip(&d).for_each(|(b, &d)| *b += d);
        rows.map(&data.p, false, (&mut b, &mut uh, &mut a, &mut c));
        rows.join((&b, &uh, &a, &c), y);
    }

    pub(crate) fn rows(&self) -> &OpRelEntropyLimitRows<T> {
        &self.data.rows
    }

    /// z passes the Dikin test at the primal point of `oracle`:
    /// (ν−1)⟨z, ∇²F(s)⁻¹z⟩ < ⟨s, z⟩² certifies z ∈ int K*.
    fn dual_ok(&self, oracle: &mut Oracle<T>, s: &[T], z: &[T]) -> bool {
        let sz = s.dot(z);
        if !(sz > T::zero()) || !z.iter().all(|v| v.is_finite()) {
            return false;
        }
        let nu1 = self.nu() - T::one();
        oracle.dual_ok(z, |q| nu1 * q < sz * sz)
    }
}

impl<T> Cone<T> for OpRelEntropyLimitCone<T>
where
    T: FloatT,
{
    fn degree(&self) -> usize {
        3 * self.n - 2 * self.m
    }

    fn numel(&self) -> usize {
        Self::size(self.n, self.m)
    }

    fn is_symmetric(&self) -> bool {
        false
    }

    fn is_sparse_expandable(&self) -> bool {
        false
    }

    fn allows_primal_dual_scaling(&self) -> bool {
        false
    }

    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool {
        // one scalar for the whole cone: (T', B, A, C) ↦ c(T', B, A, C) keeps K
        δ.copy_from(e).recip().scale(e.mean());
        true
    }

    fn margins(&mut self, _z: &mut [T], _pd: PrimalOrDualCone) -> (T, T) {
        // nonsymmetric problems always use unit_initialization
        unreachable!();
    }

    fn scaled_unit_shift(&self, _z: &mut [T], _α: T, _pd: PrimalOrDualCone) {
        unreachable!();
    }

    fn unit_initialization(&self, z: &mut [T], s: &mut [T]) {
        // (I_m ⊕ tI_k, 0, xI_k, yI_k) with the ORE's central (t, x, y)
        let (n, m, k) = (self.n, self.m, self.n - self.m);
        let (tn, mk, tk) = (triangular_number(n), m * k, triangular_number(k));
        let [t, x, y] = central_point::<T>();
        s.set(T::zero());
        for i in 0..n {
            s[tri(i, i)] = if i < m { T::one() } else { t };
        }
        for i in 0..k {
            s[tn + mk + tri(i, i)] = x;
            s[tn + mk + tk + tri(i, i)] = y;
        }
        z.copy_from(s);
    }

    fn set_identity_scaling(&mut self) {
        unreachable!();
    }

    fn update_scaling(
        &mut self,
        s: &[T],
        z: &[T],
        μ: T,
        _scaling_strategy: ScalingStrategy,
    ) -> bool {
        let Some(mut oracle) = Oracle::new(self.n, self.m, s) else {
            return false;
        };
        let Some(s0i) = oracle.schur_inverse() else {
            return false;
        };
        let data = &mut self.data;
        data.rows.clone_from(&oracle.rows);
        data.p.clone_from(&oracle.p);
        data.sh.clone_from(&oracle.sh);
        data.at_s = Some(oracle);
        data.s0i = s0i;
        data.μ = μ;
        data.s.copy_from(s);
        data.z.copy_from(z);
        true
    }

    fn Hs_is_diagonal(&self) -> bool {
        false
    }

    fn get_Hs(&self, Hsblock: &mut [T]) {
        // D in the KKT's eigen row coordinates (`OpRelEntropyLimitRows`)
        let n = self.n;
        let (sh, s0i) = (&self.data.sh, &self.data.s0i);
        let inv = self.data.μ.recip();
        let sqrt2 = T::SQRT_2();
        let z = |i: usize, j: usize| sh[i + j * n];
        let mut idx = 0;
        for l in 0..n {
            for k in 0..=l {
                for j in 0..=l {
                    for i in 0..=j {
                        if tri(i, j) > tri(k, l) {
                            break;
                        }
                        Hsblock[idx] = inv
                            * match (i == j, k == l) {
                                (false, false) => z(i, k) * z(j, l) + z(i, l) * z(j, k),
                                (true, false) => sqrt2 * z(j, l) * z(j, k),
                                (false, true) => sqrt2 * z(i, l) * z(j, k),
                                (true, true) => z(j, l) * z(j, l),
                            };
                        idx += 1;
                    }
                }
            }
        }
        let tn = triangular_number(n);
        let ny = self.numel() - tn;
        for j in 0..ny {
            Hsblock[idx..idx + tn].fill(T::zero());
            idx += tn;
            for i in 0..=j {
                Hsblock[idx] = inv * s0i[i + j * ny];
                idx += 1;
            }
        }
    }

    fn mul_Hs(&mut self, y: &mut [T], x: &[T], _work: &mut [T]) {
        self.hs_prod(y, x);
    }

    fn affine_ds(&self, ds: &mut [T], s: &[T]) {
        ds.copy_from(s);
    }

    fn combined_ds_shift(&mut self, shift: &mut [T], step_z: &mut [T], step_s: &mut [T], σμ: T) {
        // As for the ORE cone: ds = Hs z − (σμ/μ)s (the caller adds
        // affine_ds = s), less the weighted third-order term
        // ½ Hs ∇³F(s)[Δs, H⁻¹Δz].
        let paired = sdpx_arithmetic::inner_parallel::paired();
        let mut hz = std::mem::take(&mut self.data.work);
        let mut at_s = self.data.at_s.take();
        let (this, len, step_z) = (&*self, shift.len(), &*step_z);
        let mu = this.data.μ;
        let ((_, w), points) = join_if(
            paired,
            || {
                join_if(
                    paired,
                    || this.hs_prod(&mut hz, &this.data.z),
                    || {
                        let mut w = vec![T::zero(); len];
                        this.hs_prod(&mut w, step_z);
                        w.iter_mut().for_each(|v| *v *= mu);
                        w
                    },
                )
            },
            || this.mixed_points(at_s.as_mut(), step_s, paired),
        );
        let c = T::one() + σμ / mu;
        for ((o, &h), &s) in shift.iter_mut().zip(&hz).zip(&self.data.s) {
            *o = h - c * s;
        }
        self.data.work = hz;
        self.data.at_s = at_s;
        if let Some(v) = points.and_then(|p| self.third_mixed_at(p, &w, paired)) {
            let mut u = vec![T::zero(); v.len()];
            self.hs_prod(&mut u, &v);
            let σ = (σμ / mu).max(T::zero()).min(T::one());
            let half: T = (0.5).as_T();
            let w = half * (T::one() - σ.cbrt());
            shift.iter_mut().zip(&u).for_each(|(o, &u)| *o -= w * u);
        }
    }

    fn Δs_from_Δz_offset(&mut self, out: &mut [T], ds: &[T], _work: &mut [T], _z: &[T]) {
        out.copy_from(ds);
    }

    fn step_length(
        &mut self,
        dz: &[T],
        ds: &[T],
        z: &[T],
        s: &[T],
        settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T) {
        let step = settings.linesearch_backtrack_step;
        let αmin = settings.min_terminate_step_length;
        let (n, m) = (self.n, self.m);
        let mut work = std::mem::take(&mut self.data.work);
        let mut wz = vec![T::zero(); work.len()];
        // primal: s + α ds interior; the oracle at αs serves the dual test
        let mut at = None;
        let αs = search(αmax, αmin, step, |α| {
            work.waxpby(T::one(), s, α, ds);
            at = Oracle::new(n, m, &work).map(|o| (α, o));
            at.is_some()
        });
        // dual: z + α dz certified at the trial primal point s + min(α, αs) ds,
        // trials below αs batched on idle pool workers (as for the ORE cone),
        // within the ORE cone's 512 MiB allowance: most trials factor S₀
        let ways = rayon::current_thread_index().map_or(1, |_| rayon::current_num_threads());
        let k = n - m;
        let ny = m * k + k * (k + 1);
        let trial_bytes = 2 * ny * ny * std::mem::size_of::<T>();
        let ways = ways.min(((512 << 20) / trial_bytes).max(1));
        let split = heavy::<T>(n);
        let this = &*self;
        let mut α = αmax;
        let mut αz = T::zero();
        while α > αmin {
            if α >= αs || ways == 1 {
                wz.waxpby(T::one(), z, α, dz);
                let a = α.min(αs);
                if at.as_ref().is_none_or(|(b, _)| *b != a) {
                    work.waxpby(T::one(), s, a, ds);
                    at = Oracle::new(n, m, &work).map(|o| (a, o));
                }
                if at
                    .as_mut()
                    .is_some_and(|(_, o)| this.dual_ok(o, &work, &wz))
                {
                    αz = α;
                    break;
                }
                α *= step;
                continue;
            }
            let mut batch = Vec::with_capacity(ways);
            while batch.len() < ways && α > αmin {
                batch.push(α);
                α *= step;
            }
            let trials: Vec<bool> = batch
                .par_iter()
                .map(|&a| {
                    let _inner = split.then(sdpx_arithmetic::inner_parallel::Guard::enter);
                    let mut p = vec![T::zero(); s.len()];
                    p.waxpby(T::one(), s, a, ds);
                    let mut q = vec![T::zero(); z.len()];
                    q.waxpby(T::one(), z, a, dz);
                    Oracle::new(n, m, &p).is_some_and(|mut o| this.dual_ok(&mut o, &p, &q))
                })
                .collect();
            if let Some(i) = trials.iter().position(|&t| t) {
                αz = batch[i];
                break;
            }
        }
        self.data.work = work;
        (αz, αs)
    }

    fn compute_barrier(&mut self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        // f(s) + f*(z) ≈ −ν(1 + ln μ̃) + ½(⟨z, ∇²F(s)⁻¹z⟩/μ̃² − ν), as for
        // the ORE cone with the exact form (no cheap bound here).
        let mut ws = std::mem::take(&mut self.data.work);
        ws.waxpby(T::one(), s, α, ds);
        let mut wz = vec![T::zero(); ws.len()];
        wz.waxpby(T::one(), z, α, dz);
        let nu = self.nu();
        let out = Oracle::new(self.n, self.m, &ws)
            .and_then(|mut o| {
                let mu = ws.dot(&wz) / nu;
                let q = o.dual_form(&wz)?;
                let half: T = (0.5).as_T();
                (mu > T::zero()).then(|| -nu * (T::one() + mu.ln()) + half * (q / (mu * mu) - nu))
            })
            .unwrap_or(T::infinity());
        self.data.work = ws;
        out
    }
}

#[cfg(test)]
#[path = "tests/oprelentrlimitcone.rs"]
mod tests;
