#![allow(non_snake_case)]
use crate::algebra::*;

pub(crate) struct EigBlasWorkVectors<T> {
    isuppz: Vec<i32>,
    work: Vec<T>,
    iwork: Vec<i32>,
}

impl<T> EigBlasWorkVectors<T>
where
    T: FloatT,
{
    fn new(n: usize) -> Self {
        let isuppz = vec![0; 2 * n];
        // must be at least 1 element because the
        // requiring work size is written into the
        // first element
        let work = vec![T::one()];
        let iwork = vec![1];
        Self {
            isuppz,
            work,
            iwork,
        }
    }
}

pub(crate) struct EigEngine<T> {
    /// Computed eigenvalues in ascending order
    pub λ: Vec<T>,

    // BLAS workspace (allocated vecs only)
    pub blas: Option<EigBlasWorkVectors<T>>,
}

impl<T> EigEngine<T>
where
    T: FloatT,
{
    pub fn new(n: usize) -> Self {
        let λ = vec![T::zero(); n];

        match n {
            1..=3 => Self { λ, blas: None },
            _ => {
                let blas = Some(EigBlasWorkVectors::new(n));
                Self { λ, blas }
            }
        }
    }

    pub fn n(&self) -> usize {
        self.λ.len()
    }

    fn checkdim<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        if !A.is_square() || A.nrows() != self.n() {
            Err(DenseFactorizationError::IncompatibleDimension)
        } else {
            Ok(())
        }
    }
}

impl<T> EigEngine<T>
where
    T: FloatT,
{
    pub(crate) fn eigvals<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        self.checkdim(A)?;
        match self.n() {
            1 => self.eigvals1(A),
            2 => self.eigvals2(A),
            3 => self.eigvals3(A),
            _ => self.syevr(A),
        }
    }
}

impl<T> EigEngine<T>
where
    T: FloatT,
{
    /// Smallest eigenvalue only.  Sizes n <= 3 reuse the closed forms and
    /// Float64 keeps the proven full-spectrum syevr path; at higher
    /// precisions index 1 is requested via syevr's 'I' range, which MPFR
    /// resolves by Sturm isolation plus RQI.
    pub(crate) fn eigval_min<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
    ) -> Result<T, DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        self.checkdim(A)?;
        match self.n() {
            1 => {
                self.eigvals1(A)?;
            }
            2 => {
                self.eigvals2(A)?;
            }
            3 => {
                self.eigvals3(A)?;
            }
            _ if T::precision_bits() > 53 => {
                self.syevr_range(A, b'I', 1, 1)?;
                return Ok(self.λ[0]);
            }
            _ => {
                // Float64 keeps the proven full-spectrum path; an ulp-level
                // change in λmin can flip a borderline step length.
                self.syevr(A)?;
            }
        }
        Ok(self.λ.minimum())
    }
}

// trivial implementation for 1x1 matrices

impl<T> EigEngine<T>
where
    T: FloatT,
{
    fn eigvals1<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        self.λ[0] = A[(0, 0)];
        Ok(())
    }
}

// implementation for 2x2 matrices

impl<T> EigEngine<T>
where
    T: FloatT,
{
    fn eigvals2<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        // symmetric 2x2, stack allocated
        let mut As = DenseMatrixSym2::<T>::from(A.sym_up());
        let e = As.eigvals();
        self.λ.copy_from_slice(&e);
        Ok(())
    }
}

// implementation for 3x3 matrices

impl<T> EigEngine<T>
where
    T: FloatT,
{
    fn eigvals3<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        // symmetric 3x3, stack allocated
        let mut As = DenseMatrixSym3::<T>::from(A.sym_up());
        let e = As.eigvals();
        self.λ.copy_from_slice(&e);
        Ok(())
    }
}

// implementation for arbitrary size matrices

impl<T> EigEngine<T>
where
    T: FloatT,
{
    fn syevr<S>(&mut self, A: &mut DenseStorageMatrix<S, T>) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        self.syevr_range(A, b'A', 0, 0)
    }

    fn syevr_range<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
        range: u8,
        il: i32,
        iu: i32,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        let An = self.n();

        // unwrap or populate on the first call
        let blaswork = self.blas.get_or_insert_with(|| EigBlasWorkVectors::new(An));

        // standard BLAS ?syevr arguments for computing a full set of eigenvalues.

        let uplo = MatrixTriangle::Triu.as_blas_char(); // we always assume triu form
        let n = An.try_into().unwrap();
        let a = A.data_mut();
        let lda = n;
        let vl = T::zero(); // eig value lb (range = A => not used)
        let vu = T::zero(); // eig value ub (range = A => not used)
        let abstol = -T::one(); // forces default tolerance
        let m = &mut 0_i32; // returns # of computed eigenvalues
        let w = &mut self.λ; // eigenvalues go here
        let ldz = n; // leading dim of eigenvector matrix
        let isuppz = &mut blaswork.isuppz;
        let work = &mut blaswork.work;
        let mut lwork = -1_i32; // -1 => config to request required work size
        let iwork = &mut blaswork.iwork;
        let mut liwork = -1_i32; // -1 => config to request required work size
        let info = &mut 0_i32; // output info

        // The eigenvalue-only LAPACK call does not reference z.
        let mut z = [T::zero()];

        for i in 0..2 {
            T::xsyevr(
                b'N', range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, w, &mut z, ldz, isuppz,
                work, lwork, iwork, liwork, info,
            );
            if *info != 0 {
                return Err(DenseFactorizationError::Eigen(*info));
            }
            // resize work vectors and reset lengths
            if i == 0 {
                lwork = work[0].to_i32().unwrap();
                liwork = iwork[0];
                work.resize(lwork as usize, T::zero());
                iwork.resize(liwork as usize, 0);
            }
        }
        Ok(())
    }
}

macro_rules! generate_test_eigen {
    ($fxx:ty, $test_name:ident) => {
        #[test]
        fn $test_name() {
            use crate::algebra::VectorMath;

            // has to be 4x4 to avoid the special case
            let mut S = Matrix::<$fxx>::from(&[
                [3., 2., 4., 0.], //
                [2., 0., 2., 0.], //
                [4., 2., 3., 0.], //
                [0., 0., 0., 9.], //
            ]);

            let mut eng = EigEngine::<$fxx>::new(4);
            assert!(eng.eigvals(&mut S).is_ok());
            let sol = [-1.0, -1.0, 8., 9.];
            assert!(eng.λ.norm_inf_diff(&sol) < 1e-6);
        }
    };
}

generate_test_eigen!(f32, test_eigen_f32);
generate_test_eigen!(f64, test_eigen_f64);
