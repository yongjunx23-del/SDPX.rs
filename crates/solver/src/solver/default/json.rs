use crate::{algebra::*, solver::*};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::io::Write;
use std::{fs::File, io, io::Read};

/// Portable conic input using Clarabel's CSC JSON layout.
///
/// MPFR scalars use decimal strings; Float64 uses JSON numbers. With `sdp`,
/// optional sampled blocks define PSD coefficients and `A` stores only the
/// ordinary linear entries, just as in [`DefaultSolver::new_sampled`].
#[derive(Serialize, Deserialize)]
#[serde(bound = "T: Serialize + DeserializeOwned", deny_unknown_fields)]
#[allow(non_snake_case)]
pub struct JsonProblem<T: FloatT> {
    /// Upper-triangle quadratic objective matrix.
    pub P: CscMatrix<T>,
    /// Linear objective.
    pub q: Vec<T>,
    /// Constraint matrix (ordinary entries only when sampled blocks are used).
    pub A: CscMatrix<T>,
    /// Constraint right-hand side.
    pub b: Vec<T>,
    /// Ordered cone descriptors.
    pub cones: Vec<SupportedConeT<T>>,
    /// Solver settings; absent fields retain core defaults.
    #[serde(default, serialize_with = "serialize_settings")]
    pub settings: DefaultSettings<T>,
    /// Factor-authoritative PSD input, without a materialized copy in JSON.
    #[cfg(feature = "sdp")]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sampled: Vec<SampledBlock<T>>,
}

impl<T: FloatT + Serialize + DeserializeOwned> JsonProblem<T> {
    /// Read input without setting up or solving it. Settings can be edited
    /// before calling [`Self::into_solver`].
    pub fn read(reader: impl Read) -> Result<Self, SolverError> {
        let mut data: Self = serde_json::from_reader(reader)?;
        desanitize_settings(&mut data.settings);
        Ok(data)
    }

    /// Write the original input and factors; no scaling or materialization.
    pub fn write(&self, writer: impl Write) -> Result<(), SolverError> {
        Ok(serde_json::to_writer(writer, self)?)
    }

    /// Build the shared Rust solver, validating serialized dimensions first.
    pub fn into_solver(self) -> Result<DefaultSolver<T>, SolverError> {
        DefaultSolver::from_prepared(self.into_prepared()?)
    }

    /// Build the partitioned backend through the same validated preparation.
    /// An active MPI world selects the rank-local owner transport internally;
    /// an explicit count must equal the MPI world size in that mode.
    #[cfg(feature = "sdp")]
    pub fn into_partitioned_solver(
        self,
        partitions: usize,
    ) -> Result<PartitionedSolver<T>, SolverError> {
        PartitionedSolver::from_prepared(self.into_prepared()?, partitions)
    }

    /// Import optional historical owner costs while constructing an explicit
    /// partition plan.  Metadata is checked against the exact prepared input.
    #[cfg(all(feature = "sdp", feature = "serde"))]
    pub fn into_partitioned_solver_with_cost_history(
        self,
        partitions: usize,
        options: CostHistoryOptions,
    ) -> Result<PartitionedSolver<T>, SolverError> {
        // Keep the exact-input hash opt-in.  A caller can use this common
        // constructor with the default options for an ordinary partitioned
        // solve; only importing history or recording a training run needs the
        // identity payload.
        let with_identity = options.history.is_some() || options.record;
        PartitionedSolver::from_prepared_with_cost_history(
            self.into_prepared_with_identity(with_identity)?,
            partitions,
            options,
        )
    }

    /// Choose structural tasks for the requested thread budget, limiting
    /// replicated equality storage. In MPI this selects one global owner per
    /// rank and the returned solver reports the global owner count.
    #[cfg(feature = "sdp")]
    pub fn into_auto_partitioned_solver(self) -> Result<PartitionedSolver<T>, SolverError> {
        PartitionedSolver::from_prepared_auto(self.into_prepared()?)
    }

    /// Choose bounded structural tasks, optionally using validated historical
    /// owner costs and opt-in training timers.
    #[cfg(all(feature = "sdp", feature = "serde"))]
    pub fn into_auto_partitioned_solver_with_cost_history(
        self,
        options: CostHistoryOptions,
    ) -> Result<PartitionedSolver<T>, SolverError> {
        let with_identity = options.history.is_some() || options.record;
        PartitionedSolver::from_prepared_auto_with_cost_history(
            self.into_prepared_with_identity(with_identity)?,
            options,
        )
    }

    pub(crate) fn into_prepared(self) -> Result<super::PreparedProblem<T>, SolverError> {
        self.into_prepared_with_identity(false)
    }

    fn into_prepared_with_identity(
        self,
        with_cost_identity: bool,
    ) -> Result<super::PreparedProblem<T>, SolverError> {
        for matrix in [&self.P, &self.A] {
            if matrix.n.checked_add(1) != Some(matrix.colptr.len())
                || matrix.colptr.first() != Some(&0)
                || matrix.check_format().is_err()
            {
                return Err(SolverError::BadInputData("invalid CSC matrix"));
            }
        }
        let mut rows = 0usize;
        for cone in &self.cones {
            let invalid = || SolverError::BadInputData("invalid cone dimensions or parameters");
            let size = match cone {
                SupportedConeT::ZeroConeT(n)
                | SupportedConeT::NonnegativeConeT(n)
                | SupportedConeT::SecondOrderConeT(n) => *n,
                SupportedConeT::ExponentialConeT() => 3,
                SupportedConeT::PowerConeT(alpha) => {
                    if !alpha.is_finite() || *alpha <= T::zero() || *alpha >= T::one() {
                        return Err(invalid());
                    }
                    3
                }
                SupportedConeT::GenPowerConeT(alpha, dim2) => {
                    // Match the constructor's conditions without letting bad
                    // external data trigger its assertions.
                    if !alpha.iter().all(|a| a.is_finite() && *a > T::zero())
                        || (T::one() - alpha.as_slice().sum()).abs()
                            >= T::epsilon() * alpha.len().as_T() * (0.5).as_T()
                    {
                        return Err(invalid());
                    }
                    alpha.len().checked_add(*dim2).ok_or_else(invalid)?
                }
                #[cfg(feature = "sdp")]
                SupportedConeT::PSDTriangleConeT(n) => n
                    .checked_add(1)
                    .and_then(|next| n.checked_mul(next))
                    .map(|twice| twice / 2)
                    .ok_or_else(invalid)?,
            };
            rows = rows.checked_add(size).ok_or_else(invalid)?;
        }
        if rows != self.b.len() {
            return Err(SolverError::BadInputData("cone dimensions do not match b"));
        }
        #[cfg(feature = "sdp")]
        if !self.sampled.is_empty() {
            if with_cost_identity {
                #[cfg(feature = "serde")]
                return super::PreparedProblem::new_sampled_with_cost_identity(
                    &self.P,
                    &self.q,
                    &self.A,
                    &self.b,
                    &self.cones,
                    self.sampled,
                    self.settings,
                    true,
                );
            }
            return super::PreparedProblem::new_sampled(
                &self.P,
                &self.q,
                &self.A,
                &self.b,
                &self.cones,
                self.sampled,
                self.settings,
            );
        }
        if with_cost_identity {
            #[cfg(feature = "serde")]
            return super::PreparedProblem::new_with_cost_identity(
                &self.P,
                &self.q,
                &self.A,
                &self.b,
                &self.cones,
                self.settings,
            );
        }
        super::PreparedProblem::new(
            &self.P,
            &self.q,
            &self.A,
            &self.b,
            &self.cones,
            self.settings,
        )
    }
}

fn serialize_settings<T, S>(settings: &DefaultSettings<T>, serializer: S) -> Result<S::Ok, S::Error>
where
    T: FloatT + Serialize + DeserializeOwned,
    S: serde::Serializer,
{
    let mut settings = settings.clone();
    sanitize_settings(&mut settings);
    settings.serialize(serializer)
}

impl<T> SolverJSONReadWrite<T> for DefaultSolver<T>
where
    T: FloatT + DeserializeOwned + Serialize,
{
    fn save_to_file(&self, file: &mut File) -> Result<(), io::Error> {
        let mut json_data = JsonProblem {
            P: self.data.P.clone(),
            q: self.data.q.clone(),
            A: self.data.A.clone(),
            b: self.data.b.clone(),
            cones: self.data.cones.clone(),
            settings: self.settings.clone(),
            #[cfg(feature = "sdp")]
            sampled: Vec::new(),
        };

        // restore scaling to original
        let dinv = &self.data.equilibration.dinv;
        let einv = &self.data.equilibration.einv;
        let c = &self.data.equilibration.c;

        json_data.P.lrscale(dinv, dinv);
        json_data.q.hadamard(dinv);
        json_data.P.scale(c.recip());
        json_data.q.scale(c.recip());

        json_data.A.lrscale(einv, dinv);
        json_data.b.hadamard(einv);

        // sanitize settings to remove values that
        // can't be serialized, i.e. infs
        sanitize_settings(&mut json_data.settings);

        // write to file
        let json = serde_json::to_string(&json_data)?;
        file.write_all(json.as_bytes())?;

        Ok(())
    }

    fn load_from_file(
        file: &mut File,
        settings: Option<DefaultSettings<T>>,
    ) -> Result<Self, SolverError> {
        let mut data = JsonProblem::read(file)?;
        if let Some(settings) = settings {
            data.settings = settings;
        }
        data.into_solver()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem() -> JsonProblem<f64> {
        JsonProblem {
            P: CscMatrix::identity(1),
            q: vec![1.0],
            A: CscMatrix::identity(1),
            b: vec![0.0],
            cones: vec![SupportedConeT::NonnegativeConeT(1)],
            settings: DefaultSettings {
                presolve_enable: false,
                equilibrate_enable: false,
                ..DefaultSettings::default()
            },
            #[cfg(feature = "sdp")]
            sampled: Vec::new(),
        }
    }

    #[test]
    fn cost_identity_is_opt_in() {
        let without_identity = problem().into_prepared_with_identity(false).unwrap();
        assert!(without_identity.cost_input_fingerprint.is_none());

        let with_identity = problem().into_prepared_with_identity(true).unwrap();
        assert!(with_identity.cost_input_fingerprint.is_some());
    }
}

fn sanitize_settings<T: FloatT>(settings: &mut DefaultSettings<T>) {
    if settings.time_limit == f64::INFINITY {
        settings.time_limit = f64::MAX;
    }
}

fn desanitize_settings<T: FloatT>(settings: &mut DefaultSettings<T>) {
    if settings.time_limit == f64::MAX {
        settings.time_limit = f64::INFINITY;
    }
}
