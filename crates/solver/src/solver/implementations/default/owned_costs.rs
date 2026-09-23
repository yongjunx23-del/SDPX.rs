//! Optional historical owner-cost metadata for deterministic block allocation.
//!
//! Cost histories never participate in numerical arithmetic.  They select a
//! fixed owner assignment before the HSD loop starts; all reductions continue
//! to use the assignment's canonical owner order.

#[cfg(feature = "serde")]
use std::io::{self, Read, Write};

/// Versioned metadata exchanged by the optional cost-history route.
pub const COST_HISTORY_SCHEMA_VERSION: u32 = 2;

/// One structurally identified connected component and its positive work cost.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CostComponent {
    /// Stable hash of the component's structural members and incident edges.
    pub identity: u64,
    /// Deterministic structural work proxy used for grouped timing apportionment.
    pub structural_weight: u64,
    /// Historical positive finite cost used by LPT assignment.
    pub cost: f64,
}

/// Timings observed for one owner during an opt-in training solve.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CostOwnerSample {
    /// Zero-based owner index used by the training plan.
    pub owner: usize,
    /// Local `CondensedKKTSolver::update_partition` elapsed nanoseconds.
    pub local_assemble_ns: f64,
    /// Local factor/interior-response elapsed nanoseconds.
    pub factor_response_ns: f64,
    /// Component identities assigned to this owner for this fixed plan.
    pub components: Vec<u64>,
}

/// Portable, explicitly validated historical cost data.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CostHistory {
    /// Serialization schema version for this history record.
    pub schema_version: u32,
    /// SHA-256 over the authoritative numeric input before preprocessing.
    pub input_fingerprint: [u8; 32],
    /// Scalar precision used by the training solve.
    pub precision_bits: usize,
    /// Resolved `max_threads` budget used to collect this history.
    pub thread_budget: usize,
    /// Prepared variable count used by the training solve.
    pub n: usize,
    /// Prepared row count used by the training solve.
    pub m: usize,
    /// Historical costs for every structural component.
    pub components: Vec<CostComponent>,
    /// Per-owner timings and component membership observed during training.
    pub owner_samples: Vec<CostOwnerSample>,
    /// Documents the grouped-owner attribution policy in exported files.
    pub apportionment: String,
    /// Direct linear-solver method used while training.
    pub direct_solve_method: String,
    /// KKT formulation used while training.
    pub kkt_form: String,
    /// Compile-time linear-algebra provider provenance.
    pub provider: String,
}

/// Optional allocation controls.  `record` is false unless the caller
/// explicitly requests training/export, so ordinary solves pay no timer cost.
#[derive(Clone, Debug, Default)]
pub struct CostHistoryOptions {
    /// Optional validated history used to choose owner assignments.
    pub history: Option<CostHistory>,
    /// Enable per-owner timing collection for a training/export solve.
    pub record: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct CostRuntimeConfig {
    pub input_fingerprint: Option<[u8; 32]>,
    pub thread_budget: usize,
    pub record: bool,
    pub direct_solve_method: String,
    pub kkt_form: String,
    pub provider: String,
}

impl CostRuntimeConfig {
    /// Inert cost history, used by the test-only state constructors.
    #[cfg(test)]
    pub(crate) fn disabled() -> Self {
        Self {
            input_fingerprint: None,
            thread_budget: 1,
            record: false,
            direct_solve_method: String::new(),
            kkt_form: String::new(),
            provider: String::new(),
        }
    }
}

pub(crate) fn provider_tag() -> String {
    let provider = if cfg!(feature = "sdp-mkl") {
        "mkl"
    } else if cfg!(feature = "sdp-openblas") {
        "openblas"
    } else if cfg!(feature = "sdp-netlib") {
        "netlib"
    } else if cfg!(feature = "sdp-accelerate") {
        "accelerate"
    } else {
        "unspecified"
    };
    let build = option_env!("SDPX_GIT_HASH").unwrap_or("unknown");
    format!(
        "{}:solver-{}:{}",
        provider,
        env!("CARGO_PKG_VERSION"),
        build
    )
}

impl CostHistoryOptions {
    /// Use an existing history without collecting new timings.
    pub fn with_history(history: CostHistory) -> Self {
        Self {
            history: Some(history),
            record: false,
        }
    }

    /// Collect timings for a new history without importing one.
    pub fn training() -> Self {
        Self {
            history: None,
            record: true,
        }
    }

    /// Use an existing history and collect timings for a refreshed export.
    pub fn with_history_and_training(history: CostHistory) -> Self {
        Self {
            history: Some(history),
            record: true,
        }
    }
}

impl CostHistory {
    /// Parse a serialized history.  Numeric and identity checks happen when it
    /// is attached to a prepared problem, so malformed values are rejected by
    /// the solver rather than silently relabeled here.
    #[cfg(feature = "serde")]
    pub fn read(reader: impl Read) -> Result<Self, serde_json::Error> {
        serde_json::from_reader(reader)
    }

    /// Write this history as pretty-printed JSON followed by a newline.
    #[cfg(feature = "serde")]
    pub fn write(&self, mut writer: impl Write) -> io::Result<()> {
        serde_json::to_writer_pretty(&mut writer, self).map_err(io::Error::other)?;
        writer.write_all(b"\n")
    }

    /// Validate metadata and return costs aligned with `identities`.
    pub(crate) fn validate(
        &self,
        input_fingerprint: Option<[u8; 32]>,
        precision_bits: usize,
        thread_budget: usize,
        n: usize,
        m: usize,
        identities: &[u64],
        structural_weights: &[u64],
    ) -> Result<Vec<f64>, String> {
        if self.schema_version != COST_HISTORY_SCHEMA_VERSION {
            return Err(format!(
                "unsupported cost-history schema {}; expected {}",
                self.schema_version, COST_HISTORY_SCHEMA_VERSION
            ));
        }
        if self.apportionment != "owner_elapsed_apportioned_by_structural_weight" {
            return Err("unsupported cost-history apportionment policy".into());
        }
        let Some(input_fingerprint) = input_fingerprint else {
            return Err("cost history requires an authoritative prepared-input fingerprint".into());
        };
        if self.input_fingerprint != input_fingerprint {
            return Err(
                "cost history input fingerprint does not match the prepared problem".into(),
            );
        }
        if self.precision_bits != precision_bits {
            return Err(format!(
                "cost history precision {} does not match {}",
                self.precision_bits, precision_bits
            ));
        }
        if self.thread_budget != thread_budget {
            return Err(format!(
                "cost history thread budget {} does not match {}",
                self.thread_budget, thread_budget
            ));
        }
        if self.n != n || self.m != m {
            return Err(format!(
                "cost history dimensions ({}, {}) do not match ({}, {})",
                self.n, self.m, n, m
            ));
        }
        if identities.len() != structural_weights.len() || self.components.len() != identities.len()
        {
            return Err("cost history component count does not match the problem".into());
        }
        let mut costs = vec![0.0; identities.len()];
        let mut seen = std::collections::BTreeSet::new();
        for component in &self.components {
            if !seen.insert(component.identity) {
                return Err("cost history contains duplicate component identities".into());
            }
            let Some(index) = identities
                .iter()
                .position(|&identity| identity == component.identity)
            else {
                return Err(format!(
                    "cost history component {:016x} is not present in the problem",
                    component.identity
                ));
            };
            if structural_weights[index] != component.structural_weight {
                return Err(format!(
                    "cost history structural weight for component {:016x} does not match",
                    component.identity
                ));
            }
            if !component.cost.is_finite() || component.cost <= 0.0 {
                return Err(format!(
                    "cost history component {:016x} has a non-positive or non-finite cost",
                    component.identity
                ));
            }
            costs[index] = component.cost;
        }
        if seen.len() != identities.len() {
            return Err("cost history is missing one or more structural components".into());
        }
        for sample in &self.owner_samples {
            if !sample.local_assemble_ns.is_finite()
                || !sample.factor_response_ns.is_finite()
                || sample.local_assemble_ns < 0.0
                || sample.factor_response_ns < 0.0
            {
                return Err("cost history contains an invalid owner timing sample".into());
            }
            let mut sample_ids = std::collections::BTreeSet::new();
            for &identity in &sample.components {
                if !sample_ids.insert(identity) || !seen.contains(&identity) {
                    return Err(
                        "cost history owner sample has an invalid component identity".into(),
                    );
                }
            }
        }
        Ok(costs)
    }

    pub(crate) fn validate_runtime(
        &self,
        direct_solve_method: &str,
        kkt_form: &str,
        provider: &str,
    ) -> Result<(), String> {
        if self.direct_solve_method != direct_solve_method
            || self.kkt_form != kkt_form
        {
            return Err("cost history solver algorithm settings do not match".into());
        }
        if self.provider != provider {
            return Err("cost history linear-algebra provider/build does not match".into());
        }
        Ok(())
    }

    /// Build an export from owner-local phase measurements.  Grouped owners
    /// are apportioned by structural weight and explicitly labeled as such.
    pub(crate) fn from_measurements(
        input_fingerprint: Option<[u8; 32]>,
        precision_bits: usize,
        thread_budget: usize,
        n: usize,
        m: usize,
        components: &[(u64, u64, usize)],
        owner_samples: Vec<CostOwnerSample>,
        direct_solve_method: String,
        kkt_form: String,
        provider: String,
    ) -> Result<Self, String> {
        let input_fingerprint =
            input_fingerprint.ok_or("cost history requires an authoritative input fingerprint")?;
        let owner_count = components
            .iter()
            .map(|&(_, _, owner)| owner)
            .max()
            .map_or(0, |owner| owner + 1);
        let mut elapsed = vec![0.0; owner_count];
        for sample in &owner_samples {
            if sample.owner >= owner_count
                || !sample.local_assemble_ns.is_finite()
                || !sample.factor_response_ns.is_finite()
                || sample.local_assemble_ns < 0.0
                || sample.factor_response_ns < 0.0
            {
                return Err("invalid owner timing sample".into());
            }
            elapsed[sample.owner] += sample.local_assemble_ns + sample.factor_response_ns;
        }
        let mut owner_weight = vec![0u128; owner_count];
        for &(_, weight, owner) in components {
            owner_weight[owner] = owner_weight[owner]
                .checked_add(weight as u128)
                .ok_or("owner structural weight overflow")?;
        }
        let mut exported = Vec::with_capacity(components.len());
        for &(identity, weight, owner) in components {
            if owner_weight[owner] == 0 {
                return Err("owner has no structural weight".into());
            }
            if elapsed[owner] <= 0.0 || !elapsed[owner].is_finite() {
                return Err("training produced no positive owner timing".into());
            }
            let cost = elapsed[owner] * (weight as f64) / (owner_weight[owner] as f64);
            if !cost.is_finite() || cost <= 0.0 {
                return Err("apportioned component timing is not positive and finite".into());
            }
            exported.push(CostComponent {
                identity,
                structural_weight: weight,
                cost,
            });
        }
        exported.sort_by_key(|component| component.identity);
        Ok(Self {
            schema_version: COST_HISTORY_SCHEMA_VERSION,
            input_fingerprint,
            precision_bits,
            thread_budget,
            n,
            m,
            components: exported,
            owner_samples,
            apportionment: "owner_elapsed_apportioned_by_structural_weight".into(),
            direct_solve_method,
            kkt_form,
            provider,
        })
    }
}

#[cfg(all(feature = "serde", feature = "sdp"))]
use sha2::{Digest, Sha256};

#[cfg(all(feature = "serde", feature = "sdp"))]
fn feed_u64(hash: &mut Sha256, value: u64) {
    hash.update(value.to_le_bytes());
}

#[cfg(all(feature = "serde", feature = "sdp"))]
fn feed_bytes(hash: &mut Sha256, bytes: &[u8]) {
    feed_u64(hash, bytes.len() as u64);
    hash.update(bytes);
}

#[cfg(all(feature = "serde", feature = "sdp"))]
fn feed_scalar<T: crate::algebra::FloatT>(hash: &mut Sha256, value: T) {
    feed_u64(hash, T::precision_bits() as u64);
    feed_u64(hash, T::wire_tag());
    if let Some(size) = T::wire_size() {
        let mut bytes = vec![0u8; size];
        if T::write_wire(value, &mut bytes) {
            feed_bytes(hash, &bytes);
            return;
        }
    }
    feed_bytes(hash, value.to_string().as_bytes());
}

#[cfg(all(feature = "serde", feature = "sdp"))]
fn feed_matrix<T: crate::algebra::FloatT>(
    hash: &mut Sha256,
    matrix: &crate::algebra::CscMatrix<T>,
) {
    feed_u64(hash, matrix.m as u64);
    feed_u64(hash, matrix.n as u64);
    for &value in &matrix.colptr {
        feed_u64(hash, value as u64);
    }
    for &value in &matrix.rowval {
        feed_u64(hash, value as u64);
    }
    for &value in &matrix.nzval {
        feed_scalar(hash, value);
    }
}

#[cfg(all(feature = "serde", feature = "sdp"))]
fn feed_cones<T: crate::algebra::FloatT>(
    hash: &mut Sha256,
    cones: &[crate::solver::SupportedConeT<T>],
) {
    feed_u64(hash, cones.len() as u64);
    for cone in cones {
        use crate::solver::SupportedConeT;
        match cone {
            SupportedConeT::ZeroConeT(n) => {
                feed_u64(hash, 0);
                feed_u64(hash, *n as u64);
            }
            SupportedConeT::NonnegativeConeT(n) => {
                feed_u64(hash, 1);
                feed_u64(hash, *n as u64);
            }
            SupportedConeT::SecondOrderConeT(n) => {
                feed_u64(hash, 2);
                feed_u64(hash, *n as u64);
            }
            SupportedConeT::ExponentialConeT() => feed_u64(hash, 3),
            SupportedConeT::PowerConeT(alpha) => {
                feed_u64(hash, 4);
                feed_scalar(hash, *alpha);
            }
            SupportedConeT::GenPowerConeT(alpha, dim2) => {
                feed_u64(hash, 5);
                feed_u64(hash, *dim2 as u64);
                for &value in alpha {
                    feed_scalar(hash, value);
                }
            }
            SupportedConeT::PSDTriangleConeT(n) => {
                feed_u64(hash, 6);
                feed_u64(hash, *n as u64);
            }
        }
    }
}

/// Exact numeric identity used by the cost-history validator.  It hashes raw
/// input values before preprocessing; MPFR values use their canonical wire
/// payload rather than a rounded `f64` conversion.
#[cfg(all(feature = "serde", feature = "sdp"))]
pub(crate) fn input_fingerprint<T: crate::algebra::FloatT>(
    p: &crate::algebra::CscMatrix<T>,
    q: &[T],
    a: &crate::algebra::CscMatrix<T>,
    b: &[T],
    cones: &[crate::solver::SupportedConeT<T>],
    sampled: Option<&[crate::solver::implementations::default::SampledBlock<T>]>,
) -> [u8; 32] {
    let mut hash = Sha256::new();
    feed_bytes(&mut hash, b"SDPX-cost-input-v1");
    feed_matrix(&mut hash, p);
    feed_bytes(&mut hash, b"q");
    for &value in q {
        feed_scalar(&mut hash, value);
    }
    feed_matrix(&mut hash, a);
    feed_bytes(&mut hash, b"b");
    for &value in b {
        feed_scalar(&mut hash, value);
    }
    feed_cones(&mut hash, cones);
    if let Some(blocks) = sampled {
        feed_u64(&mut hash, blocks.len() as u64);
        for block in blocks {
            feed_u64(&mut hash, block.row_start as u64);
            feed_u64(&mut hash, block.column_start as u64);
            feed_u64(&mut hash, block.dim as u64);
            feed_u64(&mut hash, block.basis_rows as u64);
            feed_u64(&mut hash, block.basis_cols as u64);
            for &value in &block.basis {
                feed_scalar(&mut hash, value);
            }
            for &value in &block.weights {
                feed_scalar(&mut hash, value);
            }
        }
    } else {
        feed_u64(&mut hash, 0);
    }
    hash.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_owner_measurements_use_documented_structural_shares() {
        let history = CostHistory::from_measurements(
            Some([7; 32]),
            53,
            2,
            4,
            3,
            &[(1, 2, 0), (2, 1, 0), (3, 1, 1)],
            vec![
                CostOwnerSample {
                    owner: 0,
                    local_assemble_ns: 6.0,
                    factor_response_ns: 4.0,
                    components: vec![1, 2],
                },
                CostOwnerSample {
                    owner: 1,
                    local_assemble_ns: 5.0,
                    factor_response_ns: 0.0,
                    components: vec![3],
                },
            ],
            "auto".into(),
            "auto".into(),
            "provider".into(),
        )
        .unwrap();
        let costs: Vec<_> = history
            .components
            .iter()
            .map(|component| component.cost)
            .collect();
        assert_eq!(costs.len(), 3);
        assert!((costs[0] - 20.0 / 3.0).abs() < 1e-12);
        assert!((costs[1] - 10.0 / 3.0).abs() < 1e-12);
        assert_eq!(costs[2], 5.0);
        assert!(history
            .validate(Some([7; 32]), 53, 2, 4, 3, &[1, 2, 3], &[2, 1, 1])
            .is_ok());
    }
}
