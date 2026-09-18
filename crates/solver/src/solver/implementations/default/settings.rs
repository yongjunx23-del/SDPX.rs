use crate::solver::core::ffi::*;
use crate::solver::core::traits::Settings;
use crate::{algebra::*, solver::core::SettingsError};
use derive_builder::Builder;

#[cfg(feature = "serde")]
use serde::{de::DeserializeOwned, Deserialize, Serialize};

// PJG: Serialization is required for file in/out, but is also used to pass
// settings structures between Rust and Julia (and possibly Python)
// Passing to Julia should be done using the new FFI interface types
// implemented in https://github.com/oxfordcontrol/Clarabel.rs/pull/176

/// Standard-form solver type implementing the [`Settings`](crate::solver::core::traits::Settings) trait

#[derive(Builder, Debug, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: Serialize + DeserializeOwned"))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct DefaultSettings<T: FloatT> {
    ///maximum number of iterations
    #[builder(default = "200")]
    pub max_iter: u32,

    ///maximum run time (seconds)
    #[builder(default = "f64::INFINITY")]
    pub time_limit: f64,

    ///verbose printing
    #[builder(default = "true")]
    pub verbose: bool,

    ///maximum interior point step length
    #[builder(default = "(0.99).as_T()")]
    pub max_step_fraction: T,

    ///absolute duality gap tolerance
    #[builder(default = "accuracy_default::<T>(1e-8)")]
    pub tol_gap_abs: T,

    ///relative duality gap tolerance
    #[builder(default = "accuracy_default::<T>(1e-8)")]
    pub tol_gap_rel: T,

    ///feasibility check tolerance (primal and dual)
    #[builder(default = "accuracy_default::<T>(1e-8)")]
    pub tol_feas: T,

    ///absolute infeasibility tolerance (primal and dual)
    #[builder(default = "accuracy_default::<T>(1e-8)")]
    pub tol_infeas_abs: T,

    ///relative infeasibility tolerance (primal and dual)
    #[builder(default = "accuracy_default::<T>(1e-8)")]
    pub tol_infeas_rel: T,

    ///κ/τ tolerance
    #[builder(default = "accuracy_default::<T>(1e-6)")]
    pub tol_ktratio: T,

    ///reduced absolute duality gap tolerance
    // NB: reduced_tol_infeas_abs is *smaller* when relaxed, since
    // we are checking that we are this far into the interior of
    // an inequality when checking.   Smaller for this value means
    // "less margin required"
    #[builder(default = "reduced_accuracy_default::<T>(5e-5)")]
    pub reduced_tol_gap_abs: T,

    ///reduced relative duality gap tolerance
    #[builder(default = "reduced_accuracy_default::<T>(5e-5)")]
    pub reduced_tol_gap_rel: T,

    ///reduced feasibility check tolerance (primal and dual)
    #[builder(default = "reduced_accuracy_default::<T>(1e-4)")]
    pub reduced_tol_feas: T,

    ///reduced absolute infeasibility tolerance (primal and dual)
    #[builder(default = "reduced_margin_default::<T>(5e-12)")]
    pub reduced_tol_infeas_abs: T,

    ///reduced relative infeasibility tolerance (primal and dual)
    #[builder(default = "reduced_accuracy_default::<T>(5e-5)")]
    pub reduced_tol_infeas_rel: T,

    ///reduced κ/τ tolerance
    #[builder(default = "reduced_accuracy_default::<T>(1e-4)")]
    pub reduced_tol_ktratio: T,

    ///enable data equilibration pre-scaling
    #[builder(default = "true")]
    pub equilibrate_enable: bool,

    /// maximum equilibration scaling iterations
    #[builder(default = "10")]
    pub equilibrate_max_iter: u32,

    ///minimum equilibration scaling allowed
    #[builder(default = "(1e-4).as_T()")]
    pub equilibrate_min_scaling: T,

    ///maximum equilibration scaling allowed
    #[builder(default = "(1e+4).as_T()")]
    pub equilibrate_max_scaling: T,

    ///line search backtracking
    #[builder(default = "(0.8).as_T()")]
    pub linesearch_backtrack_step: T,

    ///minimum step size allowed for asymmetric cones with PrimalDual scaling
    #[builder(default = "(1e-1).as_T()")]
    pub min_switch_step_length: T,

    ///minimum step size allowed for symmetric cones & asymmetric cones with Dual scaling
    #[builder(default = "(1e-4).as_T()")]
    pub min_terminate_step_length: T,

    ///maximum worker budget for cone phases and multithreaded KKT solvers
    ///choosing 0 lets the solver choose for itself
    #[builder(default = "0")]
    pub max_threads: u32,

    ///use a direct linear solver method (required true)
    #[builder(default = "true")]
    pub direct_kkt_solver: bool,

    ///direct linear solver method(e.g. "faer", "qdldl", "auto")
    #[builder(default = r#""auto".to_string()"#)]
    pub direct_solve_method: String,

    ///KKT formulation: structural automatic selection, augmented, or condensed
    #[builder(default = r#""auto".to_string()"#)]
    pub kkt_form: String,

    ///enable KKT static regularization
    #[builder(default = "true")]
    pub static_regularization_enable: bool,

    ///KKT static regularization parameter
    #[builder(default = "linear_default::<T>(1e-8)")]
    pub static_regularization_constant: T,

    ///additional regularization parameter w.r.t. the maximum abs diagonal term
    #[builder(default = "T::epsilon()*T::epsilon()")]
    pub static_regularization_proportional: T,

    ///enable KKT dynamic regularization
    #[builder(default = "true")]
    pub dynamic_regularization_enable: bool,

    ///KKT dynamic regularization threshold
    #[builder(default = "linear_default::<T>(1e-13)")]
    pub dynamic_regularization_eps: T,

    ///KKT dynamic regularization shift
    #[builder(default = "accuracy_default::<T>(2e-7)")]
    pub dynamic_regularization_delta: T,

    ///KKT direct solve with iterative refinement
    #[builder(default = "true")]
    pub iterative_refinement_enable: bool,

    ///iterative refinement relative tolerance
    #[builder(default = "linear_default::<T>(1e-13)")]
    pub iterative_refinement_reltol: T,

    ///iterative refinement absolute tolerance
    #[builder(default = "linear_default::<T>(1e-12)")]
    pub iterative_refinement_abstol: T,

    ///iterative refinement maximum iterations
    #[builder(default = "10")]
    pub iterative_refinement_max_iter: u32,

    ///iterative refinement stalling tolerance
    #[builder(default = "(5.0).as_T()")]
    pub iterative_refinement_stop_ratio: T,

    ///enable presolve constraint reduction
    #[builder(default = "true")]
    pub presolve_enable: bool,

    ///explicitly drop structural zeros from sparse data inputs
    ///Caution: this will disable parametric updating functionality
    ///See also ['dropzeros'][crate::algebra::CscMatrix::dropzeros]
    ///for dropping structural zeros before passing to the solver
    ///
    #[builder(default = "false")]
    pub input_sparse_dropzeros: bool,

    /// enable chordal decomposition.
    /// [requires "sdp" feature.]
    #[cfg(feature = "sdp")]
    #[builder(default = "true")]
    pub chordal_decomposition_enable: bool,

    ///chordal decomposition merge method ("none", "parent_child" or "clique_graph").
    /// [requires "sdp" feature.]
    #[cfg(feature = "sdp")]
    #[builder(default = r#""clique_graph".to_string()"#)]
    pub chordal_decomposition_merge_method: String,

    ///assemble decomposed system in "compact" form
    ///[requires "sdp" feature.]
    #[cfg(feature = "sdp")]
    #[builder(default = "true")]
    pub chordal_decomposition_compact: bool,

    ///complete PSD dual variables after decomposition
    /// [requires "sdp" feature.]
    #[cfg(feature = "sdp")]
    #[builder(default = "true")]
    pub chordal_decomposition_complete_dual: bool,
}

// Same precision policy as SDPX.jl/src/settings.jl. Primitive defaults retain
// the upstream values; MPFR defaults use the scalar's own epsilon, with the
// reduced family kept strictly looser than the strict one (see
// `reduced_accuracy_default` and `reduced_margin_default`).
fn is_primitive<T: FloatT>() -> bool {
    use std::any::TypeId;
    TypeId::of::<T>() == TypeId::of::<f64>() || TypeId::of::<T>() == TypeId::of::<f32>()
}
fn accuracy_default<T: FloatT>(primitive: f64) -> T {
    if is_primitive::<T>() {
        primitive.as_T()
    } else {
        T::epsilon().sqrt()
    }
}
/// Reduced ("almost") counterpart of [`accuracy_default`].
///
/// Primitive types keep upstream Clarabel's reduced values. MPFR cannot reuse
/// `accuracy_default` here: it returns `sqrt(eps)` for *every* non-primitive
/// type, so a reduced tolerance built from it is bit-identical to the strict
/// one and `check_convergence_almost` can never admit a point that the strict
/// gate already rejected, leaving the almost-statuses unreachable above 53
/// bits. `eps^(1/4)` is exactly one half-order looser than the strict
/// `eps^(1/2)`, and still far tighter than any binary64 tolerance.
fn reduced_accuracy_default<T: FloatT>(primitive: f64) -> T {
    if is_primitive::<T>() {
        primitive.as_T()
    } else {
        T::epsilon().sqrt().sqrt()
    }
}
/// Reduced infeasibility *margin* default.
///
/// The infeasibility trigger keeps upstream's direction: a relaxed margin is
/// *smaller* than the strict one (`5e-12 < 1e-8` in binary64), because the test
/// asks how far inside the interior the iterate is. MPFR therefore uses
/// `eps^(3/4)`, one half-order below the strict `eps^(1/2)`.
fn reduced_margin_default<T: FloatT>(primitive: f64) -> T {
    if is_primitive::<T>() {
        primitive.as_T()
    } else {
        let root = T::epsilon().sqrt();
        root * root.sqrt()
    }
}
fn linear_default<T: FloatT>(primitive: f64) -> T {
    if is_primitive::<T>() {
        primitive.as_T()
    } else {
        let root = T::epsilon().sqrt();
        root * root.sqrt()
    }
}

impl<T> Default for DefaultSettings<T>
where
    T: FloatT,
{
    fn default() -> DefaultSettings<T> {
        DefaultSettingsBuilder::<T>::default().build().unwrap()
    }
}

macro_rules! check_immutable_setting {
    ($self:expr, $prev:expr, $field:ident) => {
        if $self.$field != $prev.$field {
            return Err(SettingsError::ImmutableSetting(stringify!($field)));
        }
    };
}

impl<T> Settings<T> for DefaultSettings<T>
where
    T: FloatT,
{
    //NB: CoreSettings is typedef'd to DefaultSettings
    fn core(&self) -> &DefaultSettings<T> {
        self
    }
    fn core_mut(&mut self) -> &mut DefaultSettings<T> {
        self
    }

    /// Checks that the settings are valid.  This only ensures that fields specified
    /// by strings contain valid options.   It does not sanity check numerical values
    fn validate(&self) -> Result<(), SettingsError> {
        // this direct check avoids an internal panic since indirect
        // solvers are not yet available at all
        if !self.direct_kkt_solver {
            return Err(SettingsError::BadFieldValue("direct_kkt_solver"));
        }

        //check that the choice of LDL solver (string) is valid
        validate_direct_solve_method(&self.direct_solve_method)?;
        validate_kkt_form(&self.kkt_form)?;

        // check that the chordal decomposition merge method (string) is valid
        #[cfg(feature = "sdp")]
        validate_chordal_decomposition_merge_method(&self.chordal_decomposition_merge_method)?;

        Ok(())
    }

    /// check that a settings object is valid as an updated collection
    /// of settings for a solver that has already been initialized.   This
    /// should reject changed to parameters that are only applicable during
    /// solver initialization.  Calls `validate()` internally to check
    /// that values are also legal.
    fn validate_as_update(&self, prev: &Self) -> Result<(), SettingsError> {
        self.validate()?;

        check_immutable_setting!(self, prev, equilibrate_enable);
        check_immutable_setting!(self, prev, equilibrate_max_iter);
        check_immutable_setting!(self, prev, equilibrate_min_scaling);
        check_immutable_setting!(self, prev, equilibrate_max_scaling);
        check_immutable_setting!(self, prev, max_threads);
        check_immutable_setting!(self, prev, direct_kkt_solver);
        check_immutable_setting!(self, prev, direct_solve_method);
        check_immutable_setting!(self, prev, kkt_form);
        check_immutable_setting!(self, prev, presolve_enable);
        check_immutable_setting!(self, prev, input_sparse_dropzeros);

        #[cfg(feature = "sdp")]
        {
            check_immutable_setting!(self, prev, chordal_decomposition_enable);
            check_immutable_setting!(self, prev, chordal_decomposition_merge_method);
            check_immutable_setting!(self, prev, chordal_decomposition_compact);
            check_immutable_setting!(self, prev, chordal_decomposition_complete_dual);
        }

        Ok(())
    }
}

impl<T: FloatT> SolverFFI<Self> for DefaultSettings<T> {
    type FFI = super::ffi::DefaultSettingsFFI<T>;
}

// pre build checker (for auto-validation when using the builder)

impl From<SettingsError> for DefaultSettingsBuilderError {
    fn from(e: SettingsError) -> Self {
        DefaultSettingsBuilderError::ValidationError(e.to_string())
    }
}

/// Automatic pre-build settings validation
impl<T> DefaultSettingsBuilder<T>
where
    T: FloatT,
{
    /// check that the specified direct_solve_method is valid
    pub fn validate(&self) -> Result<(), SettingsError> {
        if let Some(ref direct_solve_method) = self.direct_solve_method {
            validate_direct_solve_method(direct_solve_method)?;
        }
        if let Some(ref kkt_form) = self.kkt_form {
            validate_kkt_form(kkt_form)?;
        }

        // check that the chordal decomposition merge method is valid
        #[cfg(feature = "sdp")]
        if let Some(ref chordal_decomposition_merge_method) =
            self.chordal_decomposition_merge_method
        {
            validate_chordal_decomposition_merge_method(chordal_decomposition_merge_method)?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------
// individual validation functions go here
// ---------------------------------------------------------

fn validate_kkt_form(form: &str) -> Result<(), SettingsError> {
    match form {
        "auto" | "augmented" => Ok(()),
        "condensed" if cfg!(feature = "sdp") => Ok(()),
        _ => Err(SettingsError::BadFieldValue("kkt_form")),
    }
}

fn validate_direct_solve_method(direct_solve_method: &str) -> Result<(), SettingsError> {
    match direct_solve_method {
        "auto" => Ok(()),
        "qdldl" => Ok(()),
        #[cfg(feature = "faer-sparse")]
        "faer" => Ok(()),
        _ => Err(SettingsError::BadFieldValue("direct_solve_method")),
    }
}

#[cfg(feature = "sdp")]
fn validate_chordal_decomposition_merge_method(
    chordal_decomposition_merge_method: &str,
) -> Result<(), SettingsError> {
    match chordal_decomposition_merge_method {
        "none" => Ok(()),
        "parent_child" => Ok(()),
        "clique_graph" => Ok(()),
        _ => Err(SettingsError::BadFieldValue(
            "chordal_decomposition_merge_method",
        )),
    }
}

#[test]
fn test_settings_validate() {
    // all standard settings
    DefaultSettingsBuilder::<f64>::default().build().unwrap();

    // fail on unknown direct solve method
    assert!(DefaultSettingsBuilder::<f64>::default()
        .direct_solve_method("foo".to_string())
        .build()
        .is_err());

    // fail on solve options in disabled feature
    let builder = DefaultSettingsBuilder::<f64>::default()
        .direct_solve_method("faer".to_string())
        .build();
    cfg_if::cfg_if! {
        if #[cfg(feature = "faer-sparse")] {
            assert!(builder.is_ok());
        }
        else {
            assert!(builder.is_err());
        }
    }
    #[cfg(feature = "sdp")]
    // fail on unknown chordal decomposition merge method
    assert!(DefaultSettingsBuilder::<f64>::default()
        .chordal_decomposition_merge_method("foo".to_string())
        .build()
        .is_err());

    // directly construct a bad DefaultSettings and manually check
    let settings = DefaultSettings::<f64> {
        direct_solve_method: "foo".to_string(),
        ..DefaultSettings::default()
    };
    assert!(settings.validate().is_err());

    // try to overlay prohibited update values
    let oldsettings = DefaultSettings::<f64> {
        presolve_enable: false,
        ..DefaultSettings::default()
    };

    let newsettings = DefaultSettings::<f64> {
        presolve_enable: true,
        ..DefaultSettings::default()
    };
    assert!(newsettings.validate_as_update(&oldsettings).is_err());

    // try to overlay allowed update values
    let oldsettings = DefaultSettings::<f64> {
        max_iter: 10,
        ..DefaultSettings::default()
    };

    let newsettings = DefaultSettings::<f64> {
        max_iter: 11,
        ..DefaultSettings::default()
    };
    assert!(newsettings.validate_as_update(&oldsettings).is_ok());
}

/// The reduced ("almost") gates must be *ordered* against the strict ones at
/// every precision. A reduced value that equals its strict counterpart makes
/// `check_convergence_almost` unreachable, which is how the MPFR defaults
/// silently lost the almost-statuses before `reduced_accuracy_default` existed.
#[test]
fn reduced_tolerances_stay_distinct_from_full_accuracy_tolerances() {
    fn check<T: FloatT>() {
        let s = DefaultSettings::<T>::default();
        let bits = T::precision_bits();
        assert!(s.reduced_tol_gap_abs > s.tol_gap_abs, "gap_abs@{bits}");
        assert!(s.reduced_tol_gap_rel > s.tol_gap_rel, "gap_rel@{bits}");
        assert!(s.reduced_tol_feas > s.tol_feas, "feas@{bits}");
        assert!(
            s.reduced_tol_infeas_rel > s.tol_infeas_rel,
            "infeas_rel@{bits}"
        );
        assert!(s.reduced_tol_ktratio > s.tol_ktratio, "ktratio@{bits}");
        // A relaxed infeasibility margin is *smaller*: the test asks how far
        // inside the interior the iterate is.
        assert!(
            s.reduced_tol_infeas_abs < s.tol_infeas_abs,
            "infeas_abs@{bits}"
        );
        // Operands must stay strictly positive in the working arithmetic.
        for value in [
            s.reduced_tol_gap_abs,
            s.reduced_tol_gap_rel,
            s.reduced_tol_feas,
            s.reduced_tol_infeas_abs,
            s.reduced_tol_infeas_rel,
            s.reduced_tol_ktratio,
        ] {
            assert!(value > T::zero() && value.is_finite());
        }
    }
    check::<f64>();
    check::<sdpx_arithmetic::Bits256>();
    check::<sdpx_arithmetic::Bits512>();
}
