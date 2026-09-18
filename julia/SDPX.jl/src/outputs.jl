const _RETENTION_NONE = :none
const _RETENTION_ALL = :all
const _DIAGNOSTICS_LEVELS = (:none, :summary, :full)

"""
    Outputs

Typed public retention policy for result fields.

For the three raw-field groups (`primal`, `constraint_dual`,
`dual_slack`) the value is either

- `:all`   — retain every component of the group;
- `:none`  — retain nothing from the group;
- a concrete component vector, retaining only those components:
  `primal` and `dual_slack` take `Vector{VariableRef}` (raw primal
  variable / dual-slack components), while `constraint_dual` takes
  `Vector{ConstraintRef}` (affine-cone constraint components).  The
  groups are semantically typed: a constraint vector is rejected for
  `primal`/`dual_slack`, and a variable vector is rejected for
  `constraint_dual`.

`objectives::Bool` retains primal/dual objective values.
`diagnostics::Symbol` is `:none`,
`:summary`, or `:full` (`:full` keeps the detailed planning/diagnostic
payload).

This policy controls what is retained in the returned `Result`; it does not
change the solver workspace or guarantee a lower peak allocation during the
solve.

All fields are validated on construction and normalized by the public
`normalize_outputs` entry point. `history` and `trace` remain accepted for
source compatibility with the SDPX.jl-derived API, but requesting them is an
error: the Rust SDPX core publishes no per-iteration history or performance
trace across the C ABI, so the fields can never be populated.
"""
struct Outputs
    primal::Union{Symbol,Vector{VariableRef}}
    constraint_dual::Union{Symbol,Vector{ConstraintRef}}
    dual_slack::Union{Symbol,Vector{VariableRef}}
    objectives::Bool
    diagnostics::Symbol
    history::Bool
    trace::Bool

    function Outputs(
        primal::Union{Symbol,Vector{VariableRef}},
        constraint_dual::Union{Symbol,Vector{ConstraintRef}},
        dual_slack::Union{Symbol,Vector{VariableRef}},
        objectives::Bool,
            diagnostics::Symbol,
        history::Bool,
        trace::Bool,
    )
        _validate_retention_spec(primal, :primal, :variables)
        _validate_retention_spec(constraint_dual, :constraint_dual, :constraints)
        _validate_retention_spec(dual_slack, :dual_slack, :variables)
        diagnostics in _DIAGNOSTICS_LEVELS || throw(ArgumentError(
            "diagnostics must be one of $_DIAGNOSTICS_LEVELS, got $(repr(diagnostics))",
        ))
        return new(primal, constraint_dual, dual_slack, objectives, diagnostics, history, trace)
    end
end

function _validate_retention_spec(value, field::Symbol, kind::Symbol)
    value === _RETENTION_ALL && return nothing
    value === _RETENTION_NONE && return nothing
    kind === :variables && value isa Vector{VariableRef} && return nothing
    kind === :constraints && value isa Vector{ConstraintRef} && return nothing
    allowed = kind === :variables ?
              "Vector{VariableRef}" : "Vector{ConstraintRef}"
    throw(ArgumentError(
        "$field retention must be :all, :none, or a concrete $allowed, got $(repr(value))",
    ))
end

"""
    ResultFieldNotRetained

Stable, inspectable error raised when a result field was not retained.
`field` is the policy-meaningful field name (`:primal`, `:constraint_dual`,
`:dual_slack`, `:objectives`, `:diagnostics`, `:history`,
`:trace`); `message` is a self-contained human description that records
the originating output policy.
"""
struct ResultFieldNotRetained <: Exception
    field::Symbol
    message::String
end

ResultFieldNotRetained(field::Symbol) = ResultFieldNotRetained(
    field,
    "requested result field `$field` was not retained by the output policy",
)

Base.showerror(io::IO, error::ResultFieldNotRetained) =
    print(io, "ResultFieldNotRetained: ", error.message)

"""
    normalize_outputs(outputs::Outputs) -> Outputs

Return the canonical, validated retention policy.  The policy struct is
already validated and typed at construction, so normalization is the pure,
deterministic identity documented for integration code that wants an
explicit stable entry point before forwarding the policy to the result
layer.
"""
normalize_outputs(outputs::Outputs) = _normalize_outputs(outputs)

Base.:(==)(left::Outputs, right::Outputs) =
    isequal(left.primal, right.primal) &&
    isequal(left.constraint_dual, right.constraint_dual) &&
    isequal(left.dual_slack, right.dual_slack) &&
    left.objectives == right.objectives &&
    left.diagnostics == right.diagnostics &&
    left.history == right.history &&
    left.trace == right.trace

function _normalize_outputs(outputs::Outputs)
    return Outputs(
        _normalized_group(outputs.primal),
        _normalized_group(outputs.constraint_dual),
        _normalized_group(outputs.dual_slack),
        outputs.objectives,
        outputs.diagnostics,
        outputs.history,
        outputs.trace,
    )
end

_normalized_group(value::Symbol) = value
_normalized_group(value::Vector{VariableRef}) = copy(value)
_normalized_group(value::Vector{ConstraintRef}) = copy(value)

"""Validated output policies have no cross-field certification requirement."""
outputs_conflict(::Outputs) = nothing

function Outputs(
    primal::Union{Symbol,Vector{VariableRef}}=:all,
    constraint_dual::Union{Symbol,Vector{ConstraintRef}}=:all,
    dual_slack::Union{Symbol,Vector{VariableRef}}=:all;
    objectives::Bool=true,
    certificate::Union{Nothing,Symbol}=nothing,
    diagnostics::Symbol=:summary,
    history::Bool=false,
    trace::Bool=false,
)
    certificate in (nothing, :none) || throw(ArgumentError(
        "certificate output was removed; use residual/status accessors and external validation",
    ))
    history === false || throw(ArgumentError(
        "iteration history is not published by the Rust SDPX core; use `diagnostics` and the per-solve metrics",
    ))
    trace === false || throw(ArgumentError(
        "performance traces are not published by the Rust SDPX core; use `diagnostics` and `execution_plan`",
    ))
    validated = Outputs(
        primal,
        constraint_dual,
        dual_slack,
        objectives,
        diagnostics,
        history,
        trace,
    )
    return _normalize_outputs(validated)
end

function Base.show(io::IO, outputs::Outputs)
    print(
        io,
        "Outputs(",
        "primal=", _show_retention(outputs.primal),
        ", constraint_dual=", _show_retention(outputs.constraint_dual),
        ", dual_slack=", _show_retention(outputs.dual_slack),
        ", objectives=", outputs.objectives,
        ", diagnostics=", outputs.diagnostics,
        ", history=", outputs.history,
        ", trace=", outputs.trace,
        ")",
    )
end

_show_retention(value::Symbol) = value
_show_retention(value::Vector{VariableRef}) = "[:variables...]"
_show_retention(value::Vector{ConstraintRef}) = "[:constraints...]"
