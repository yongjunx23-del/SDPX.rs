# Public statuses retain their established names; relaxed accuracy is never Optimal.
@enum SolveStatus begin
    NotStarted
    Optimal
    FeasibleCert
    InfeasibleCert
    Stalled
    IterLimit
    TimeLimit
    NumericalBreakdown
    MaxRestartsExceeded
    UserStopped
    AlmostOptimal
    InsufficientPrecision
    NumericalFailure
    PrimalInfeasible
    DualInfeasible
end

struct RawResult{T<:AbstractFloat,I}
    status::SolveStatus
    x::Vector{T}
    y::Vector{T}
    s::Vector{T}
    primal_objective::T
    dual_objective::T
    primal_residual::T
    dual_residual::T
    relative_gap::T
    iterations::Int
    solve_time::Float64
    info::I
end
function Base.getproperty(r::RawResult,k::Symbol)
    k===:z && return getfield(r,:y)
    k===:objective && return getfield(r,:primal_objective)
    k===:pObj && return getfield(r,:primal_objective)
    k===:dObj && return getfield(r,:dual_objective)
    k===:p_res && return getfield(r,:primal_residual)
    k===:d_res && return getfield(r,:dual_residual)
    k===:gap_rel && return getfield(r,:relative_gap)
    getfield(r,k)
end
Base.propertynames(r::RawResult,private::Bool=false)=(fieldnames(typeof(r))...,:z,:objective,:pObj,:dObj,:p_res,:d_res,:gap_rel)
const _STATUS_SYMBOLS=(:not_started,:optimal,:feasible_certificate,:infeasible_certificate,
    :stalled,:iteration_limit,:time_limit,:numerical_breakdown,:max_restarts_exceeded,
    :user_stopped,:almost_optimal,:insufficient_precision,:numerical_failure,
    :primal_infeasible,:dual_infeasible)
_status_symbol(s::SolveStatus)=_STATUS_SYMBOLS[Int(s)+1]
_status_symbol(s::Symbol)=s
status(r::RawResult)=_status_symbol(r.status)
primal_objective(r::RawResult)=r.primal_objective
dual_objective(r::RawResult)=r.dual_objective
primal_residual(r::RawResult)=r.primal_residual
dual_residual(r::RawResult)=r.dual_residual
relative_gap(r::RawResult)=r.relative_gap
iterations(r::RawResult)=r.iterations
solve_time(r::RawResult)=r.solve_time
value(r::RawResult)=deepcopy(r.x)
dual(r::RawResult)=deepcopy(r.y)
dual_slack(r::RawResult)=deepcopy(r.s)
objective_value(r)=primal_objective(r)
dual_objective_value(r)=dual_objective(r)
is_optimal(r)=_status_symbol(status(r))===:optimal
is_primal_infeasible(r)=_status_symbol(status(r))===:primal_infeasible
is_dual_infeasible(r)=_status_symbol(status(r))===:dual_infeasible
termination_status(r)=status(r)
primal_status(r)=is_optimal(r) ? :feasible_point : _status_symbol(status(r))===:almost_optimal ? :nearly_feasible_point : is_dual_infeasible(r) ? :infeasibility_ray : :unknown
dual_status(r)=is_optimal(r) ? :feasible_point : _status_symbol(status(r))===:almost_optimal ? :nearly_feasible_point : is_primal_infeasible(r) ? :infeasibility_ray : :unknown
function diagnostics(r::RawResult)
    r.info.diagnostics_level===:none && throw(ResultFieldNotRetained(:diagnostics))
    deepcopy(r.info)
end
termination(r::RawResult)=(status=r.status,reason=r.info.core_status,stage=:solver,message=string(r.info.core_status))
execution_plan(r::RawResult)=(engine=:rust,kkt_form=r.info.kkt_form,
    provider=r.info.provider,factorization=r.info.factorization,precision_bits=r.info.precision_bits,
    backend_threads=r.info.backend_threads,cone_threads=r.info.cone_threads)
