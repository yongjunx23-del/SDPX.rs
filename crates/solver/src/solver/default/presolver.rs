#![allow(non_snake_case)]
use super::*;
use crate::algebra::*;
use crate::solver::SupportedConeT;

// ---------------
// Data type for default problem presolver
// ---------------

// PJG: updates required here
#[derive(Debug)]
pub(crate) struct PresolverRowReductionIndex {
    // vector of length = original RHS.   Entries are false
    // for those rows that should be eliminated before solve
    pub keep_logical: Vec<bool>,
}

/// Presolver data for the standard solver implementation

#[derive(Debug)]
pub(crate) struct Presolver<T> {
    // original cones of the problem
    pub(crate) _init_cones: Vec<SupportedConeT<T>>,

    // Original-row map for exact equality and infinite NN reductions
    pub(crate) reduce_map: Option<PresolverRowReductionIndex>,

    // size of original and reduced RHS, respectively
    pub(crate) mfull: usize,
    pub(crate) mreduced: usize,

    // inf bound that was taken from the module level
    // and should be applied throughout.   Held here so
    // that any subsequent change to the module's state
    // won't mess up our solver mid-solve
    pub(crate) infbound: f64,
}

impl<T> Presolver<T>
where
    T: FloatT,
{
    /// create a new presolver object
    pub(crate) fn new(
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        _settings: &DefaultSettings<T>,
    ) -> Self {
        let infbound = crate::get_infinity();

        // make copy of cones to protect from user interference
        let init_cones = cones.to_vec();
        let mfull = b.len();

        let (reduce_map, mreduced) = make_reduction_map(A, cones, b, infbound.as_T());

        Self {
            _init_cones: init_cones,
            reduce_map,
            mfull,
            mreduced,
            infbound,
        }
    }

    /// true if the presolver has reduced the problem
    pub(crate) fn is_reduced(&self) -> bool {
        self.reduce_map.is_some()
    }
    /// returns number of constraints eliminated
    pub(crate) fn count_reduced(&self) -> usize {
        self.mfull - self.mreduced
    }

    pub(crate) fn presolve(
        &self,
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
    ) -> (CscMatrix<T>, Vec<T>, Vec<SupportedConeT<T>>) {
        let (A_new, b_new) = self.reduce_A_b(A, b);
        let cones_new = self.reduce_cones(cones);

        (A_new, b_new, cones_new)
    }

    fn reduce_A_b(&self, A: &CscMatrix<T>, b: &[T]) -> (CscMatrix<T>, Vec<T>) {
        assert!(self.reduce_map.is_some());
        let map = self.reduce_map.as_ref().unwrap();

        let A = A.select_rows(&map.keep_logical);
        let b = b.select(&map.keep_logical);

        (A, b)
    }

    fn reduce_cones(&self, cones: &[SupportedConeT<T>]) -> Vec<SupportedConeT<T>> {
        assert!(self.reduce_map.is_some());
        let map = self.reduce_map.as_ref().unwrap();

        // assume that we will end up with the same
        // number of cones, despite small possibility
        // that some will be completely eliminated

        let mut cones_new = Vec::with_capacity(cones.len());
        let mut keep_iter = map.keep_logical.iter();

        for cone in cones {
            let numel_cone = cone.nvars();
            let markers = keep_iter.by_ref().take(numel_cone);

            if matches!(
                cone,
                SupportedConeT::NonnegativeConeT(_) | SupportedConeT::ZeroConeT(_)
            ) {
                let nkeep = markers.filter(|&b| *b).count();
                if nkeep > 0 {
                    cones_new.push(if matches!(cone, SupportedConeT::ZeroConeT(_)) {
                        SupportedConeT::ZeroConeT(nkeep)
                    } else {
                        SupportedConeT::NonnegativeConeT(nkeep)
                    });
                }
            } else {
                //NB: take() is lazy, so must consume this block
                //to force keep_iter to advance to the next cone

                // this clippy lint is a false positive
                #[allow(unknown_lints)] // suppress error in old versions
                #[allow(clippy::double_ended_iterator_last)]
                markers.last(); // skip this cone
                cones_new.push(cone.clone());
            }
        }

        cones_new
    }

    pub(crate) fn reverse_presolve(
        &self,
        solution: &mut DefaultSolution<T>,
        variables: &DefaultVariables<T>,
    ) {
        solution.x.copy_from(&variables.x);

        let map = self.reduce_map.as_ref().unwrap();
        let mut ctr = 0;

        let zero_rows = self._init_cones.iter().flat_map(|c| {
            std::iter::repeat_n(matches!(c, SupportedConeT::ZeroConeT(_)), c.nvars())
        });
        for ((idx, &keep), zero) in map.keep_logical.iter().enumerate().zip(zero_rows) {
            if keep {
                solution.s[idx] = variables.s[ctr];
                solution.z[idx] = variables.z[ctr];
                ctr += 1;
            } else {
                solution.s[idx] = if zero {
                    T::zero()
                } else {
                    self.infbound.as_T()
                };
                solution.z[idx] = T::zero();
            }
        }
    }
}

fn make_reduction_map<T>(
    A: &CscMatrix<T>,
    cones: &[SupportedConeT<T>],
    b: &[T],
    infbound: T,
) -> (Option<PresolverRowReductionIndex>, usize)
where
    T: FloatT,
{
    //assume we keep everything initially
    let mut keep_logical = vec![true; b.len()];
    let mut mreduced = b.len();

    // only try to reduce nn cones.  Make a slight contraction
    // so that we are firmly "less than" here
    let infbound = (T::one() - T::epsilon() * (10.).as_T()) * infbound;

    // we loop through b and remove any entries that are both infinite
    // and in a nonnegative cone

    let mut idx = 0; // index into the b vector

    for cone in cones {
        let numel_cone = cone.nvars();

        if matches!(cone, SupportedConeT::NonnegativeConeT(_)) {
            for _ in 0..numel_cone {
                if b[idx] > infbound {
                    keep_logical[idx] = false;
                    mreduced -= 1;
                }
                idx += 1;
            }
        } else {
            // skip this cone
            idx += numel_cone;
        }
    }

    if let Some(redundant) = redundant_equalities(A, b, cones) {
        for row in redundant {
            keep_logical[row] = false;
            mreduced -= 1;
        }
    }

    let outoption = {
        if mreduced < b.len() {
            Some(PresolverRowReductionIndex { keep_logical })
        } else {
            None
        }
    };

    (outoption, mreduced)
}

// Prove dependence over exact input values, including RHS. Retained rows are
// unchanged, so zero multipliers restore a valid dual in original coordinates.
// The elimination has no operation budget: it runs to completion, and only the
// per-coefficient size guard (`Exact::bounded`) can decline a degenerate set.
fn redundant_equalities<T: FloatT>(
    A: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
) -> Option<Vec<usize>> {
    use sdpx_arithmetic::Exact;
    use std::collections::BTreeMap;
    let mut ids = Vec::new();
    let mut start = 0;
    for cone in cones {
        if matches!(cone, SupportedConeT::ZeroConeT(_)) {
            ids.extend(start..start + cone.nvars());
        }
        start += cone.nvars();
    }
    if ids.is_empty() {
        return None;
    }
    let mut lookup = vec![usize::MAX; b.len()];
    let mut rows: Vec<BTreeMap<usize, Exact>> = (0..ids.len()).map(|_| BTreeMap::new()).collect();
    for (i, &r) in ids.iter().enumerate() {
        lookup[r] = i;
        if b[r] != T::zero() {
            rows[i].insert(A.n, b[r].exact()?);
        }
    }
    for c in 0..A.n {
        for k in A.colptr[c]..A.colptr[c + 1] {
            let i = lookup[A.rowval[k]];
            if i != usize::MAX && A.nzval[k] != T::zero() {
                // Do not prove rank from overwritten noncanonical CSC entries.
                if rows[i].insert(c, A.nzval[k].exact()?).is_some() {
                    return None;
                }
            }
        }
    }
    // A full row rank image proves that no exact row can be removed. A
    // singular image proves nothing: preserve the rational fallback below.
    if modular_full_row_rank(&rows) {
        return Some(Vec::new());
    }
    let mut basis: BTreeMap<usize, BTreeMap<usize, Exact>> = BTreeMap::new();
    let mut redundant = Vec::new();
    for (id, mut row) in ids.into_iter().zip(rows) {
        loop {
            let Some((&pivot, value)) = row.first_key_value() else {
                redundant.push(id);
                break;
            };
            let factor = value.clone();
            if let Some(previous) = basis.get(&pivot) {
                for (&c, value) in previous {
                    let entry = row.entry(c).or_default();
                    entry.subtract_product(&factor, value);
                    if !entry.bounded() {
                        return None;
                    }
                    if entry.is_zero() {
                        row.remove(&c);
                    }
                }
            } else {
                for value in row.values_mut() {
                    value.divide(&factor);
                    if !value.bounded() {
                        return None;
                    }
                }
                basis.insert(pivot, row);
                break;
            }
        }
    }
    Some(redundant)
}

fn modular_full_row_rank(
    rows: &[std::collections::BTreeMap<usize, sdpx_arithmetic::Exact>],
) -> bool {
    use std::collections::BTreeMap;
    const P: u64 = (1 << 31) - 1;
    let mut basis: BTreeMap<usize, BTreeMap<usize, u64>> = BTreeMap::new();
    for source in rows {
        let mut row = BTreeMap::new();
        for (&column, value) in source {
            let Some(value) = value.modulo_mersenne31() else {
                return false;
            };
            if value != 0 {
                row.insert(column, u64::from(value));
            }
        }
        loop {
            let Some((&pivot, &factor)) = row.first_key_value() else {
                return false;
            };
            if let Some(previous) = basis.get(&pivot) {
                for (&column, &value) in previous {
                    let entry = row.entry(column).or_default();
                    *entry = (*entry + P - factor * value % P) % P;
                    if *entry == 0 {
                        row.remove(&column);
                    }
                }
            } else {
                let (mut power, mut exponent, mut inverse) = (factor, P - 2, 1u64);
                while exponent != 0 {
                    if exponent & 1 != 0 {
                        inverse = inverse * power % P;
                    }
                    power = power * power % P;
                    exponent >>= 1;
                }
                for value in row.values_mut() {
                    *value = *value * inverse % P;
                }
                basis.insert(pivot, row);
                break;
            }
        }
    }
    true
}

#[cfg(test)]
mod exact_tests {
    use super::*;
    fn check<T: FloatT>() {
        let one = T::one();
        let two = one + one;
        let zero = T::zero();
        // Last row differs below f64 resolution in MPFR; it must be retained.
        let delta = T::epsilon();
        let a = CscMatrix::new(
            5,
            2,
            vec![0, 4, 7],
            vec![0, 1, 2, 4, 0, 1, 4],
            vec![one, two, one, one, one, two, one + delta],
        );
        let b = vec![two, two + two, zero, zero, two];
        let cones = vec![SupportedConeT::ZeroConeT(5)];
        assert_eq!(redundant_equalities(&a, &b, &cones).unwrap(), vec![1, 3]);
        let mut inconsistent = b.clone();
        inconsistent[1] += delta * two * two * two;
        assert!(!redundant_equalities(&a, &inconsistent, &cones)
            .unwrap()
            .contains(&1));
    }
    fn peeled_chain<T: FloatT>() {
        // Rows 0..31 form a singleton cascade; the separate two-row component
        // has one exact dependence. A nonzero RHS alone must not peel a row.
        let n = 34;
        let mut i = Vec::new();
        let mut j = Vec::new();
        let mut v = Vec::new();
        for r in 0..32 {
            i.push(r);
            j.push(r);
            v.push(T::one());
            if r > 0 {
                i.push(r);
                j.push(r - 1);
                v.push(T::one());
            }
        }
        for r in 32..34 {
            for c in 32..34 {
                i.push(r);
                j.push(c);
                v.push(T::one());
            }
        }
        let a = CscMatrix::new_from_triplets(n, n, i, j, v);
        let mut b = vec![T::one(); n];
        let cones = [SupportedConeT::ZeroConeT(n)];
        assert_eq!(redundant_equalities(&a, &b, &cones).unwrap(), vec![33]);
        b[33] += T::one();
        assert!(redundant_equalities(&a, &b, &cones).unwrap().is_empty());
    }
    #[test]
    fn singleton_cascade_f64() {
        peeled_chain::<f64>();
    }
    #[test]
    fn singleton_cascade_256() {
        peeled_chain::<sdpx_arithmetic::Bits256>();
    }
    #[test]
    fn singleton_cascade_512() {
        peeled_chain::<sdpx_arithmetic::Bits512>();
    }
    #[test]
    fn noncanonical_duplicates_skip_exact_reduction() {
        let a = CscMatrix::new(2, 1, vec![0, 3], vec![0, 0, 1], vec![1., 2., 2.]);
        assert!(redundant_equalities(&a, &[2., 2.], &[SupportedConeT::ZeroConeT(2)]).is_none());
    }
    #[test]
    fn exact_rows_f64() {
        check::<f64>();
    }
    #[test]
    fn exact_rows_256() {
        check::<sdpx_arithmetic::Bits256>();
    }
    #[test]
    fn exact_rows_512() {
        check::<sdpx_arithmetic::Bits512>();
    }
}
