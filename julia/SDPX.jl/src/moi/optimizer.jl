"""SDPX conic optimizer, usable through JuMP or raw MOI."""
mutable struct Optimizer <: MOI.AbstractOptimizer
    model::MOI.Utilities.Model{Float64}
    result::Union{Nothing,RawResult{Float64}}
    solve_time_sec::Float64
    silent::Bool
    time_limit::Float64
    num_threads::Union{Nothing,Int}
    options::Dict{String,Any}
    objective_sign::Float64
    objective_constant::Float64
    # Variable mapping: MOI VariableIndex → SDPX variable index (1-based).
    var_idx::Vector{MOI.VariableIndex}
    var_to_col::Dict{MOI.VariableIndex,Int}
    constraint_rows::Dict{MOI.ConstraintIndex,UnitRange{Int}}

    function Optimizer()
        return new(
            MOI.Utilities.Model{Float64}(),
            nothing, 0.0, false, Inf, nothing, Dict{String,Any}(), 1.0, 0.0,
            MOI.VariableIndex[], Dict{MOI.VariableIndex,Int}(),
            Dict{MOI.ConstraintIndex,UnitRange{Int}}(),
        )
    end
end

# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------

MOI.supports_incremental_interface(::Optimizer) = false
MOI.is_empty(o::Optimizer) = MOI.is_empty(o.model)
MOI.empty!(o::Optimizer) = begin
    MOI.empty!(o.model)
    o.result = nothing
    o.solve_time_sec = 0.0
    empty!(o.var_idx)
    empty!(o.var_to_col)
    empty!(o.constraint_rows)
    return
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    MOI.empty!(dest)
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        MOI.supports_constraint(dest, F, S) ||
            throw(MOI.UnsupportedConstraint{F,S}())
    end
    index_map = MOI.copy_to(dest.model, src)
    # Store the destination model's ordered variable list for primal extraction.
    dest.var_idx = MOI.get(dest.model, MOI.ListOfVariableIndices())
    for (col, v) in enumerate(dest.var_idx)
        dest.var_to_col[v] = col
    end
    return index_map
end

MOI.add_variable(::Optimizer) = throw(MOI.AddVariableNotAllowed())
MOI.add_variables(::Optimizer, n) = throw(MOI.AddVariableNotAllowed())

# ---------------------------------------------------------------------------
# Constraint support
# ---------------------------------------------------------------------------

const _MOI_SCALAR_SETS = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

const _MOI_VECTOR_SETS = Union{
    MOI.Nonnegatives,
    MOI.Zeros,
    MOI.SecondOrderCone,
    MOI.PositiveSemidefiniteConeTriangle,
    MOI.ExponentialCone,
    MOI.PowerCone{Float64},
}

MOI.supports_constraint(::Optimizer, ::Type{MOI.VariableIndex}, ::Type{<:_MOI_SCALAR_SETS}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{<:_MOI_SCALAR_SETS}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.VectorOfVariables}, ::Type{<:_MOI_VECTOR_SETS}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.VectorAffineFunction{Float64}}, ::Type{<:_MOI_VECTOR_SETS}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.EqualTo{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{<:MOI.AbstractFunction}, ::Type{<:MOI.AbstractSet}) = false

# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------

MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.get(o::Optimizer, ::MOI.Silent) = o.silent
MOI.set(o::Optimizer, ::MOI.Silent, v::Bool) = (o.silent = v; nothing)

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
MOI.get(o::Optimizer, ::MOI.TimeLimitSec) = o.time_limit
function MOI.set(o::Optimizer, ::MOI.TimeLimitSec, v)
    v === nothing || (v >= 0 || throw(ArgumentError("TimeLimitSec must be nonnegative")))
    o.time_limit = v === nothing ? Inf : Float64(v)
    return nothing
end

MOI.supports(::Optimizer, ::MOI.NumberOfThreads) = true
MOI.get(o::Optimizer, ::MOI.NumberOfThreads) = o.num_threads
function MOI.set(o::Optimizer, ::MOI.NumberOfThreads, v)
    if v === nothing
        o.num_threads = nothing
        delete!(o.options, "max_threads")
        return nothing
    end
    threads = Limits(threads=v).threads
    o.num_threads = threads
    delete!(o.options, "max_threads")
    return nothing
end

# Grouped constructor/read-only properties (limits, tolerances) are not setters.
_moi_setting_key(k::Symbol) = k === :threads ? :max_threads : k === :verbosity ? :verbose : k
MOI.supports(::Optimizer, a::MOI.RawOptimizerAttribute) =
    hasfield(Settings{Float64}, _moi_setting_key(Symbol(a.name)))
function MOI.set(o::Optimizer, a::MOI.RawOptimizerAttribute, v)
    MOI.supports(o, a) || throw(MOI.UnsupportedAttribute(a))
    settings = Settings(Float64)
    setproperty!(settings, Symbol(a.name), v) # validate before storing
    key = _moi_setting_key(Symbol(a.name))
    o.options[string(key)] = deepcopy(getproperty(settings, key))
    key === :verbose && (o.silent = !settings.verbose)
    key === :time_limit && (o.time_limit = settings.time_limit)
    key === :max_threads && (o.num_threads = settings.max_threads)
    return
end
function MOI.get(o::Optimizer, a::MOI.RawOptimizerAttribute)
    MOI.supports(o, a) || throw(MOI.UnsupportedAttribute(a))
    key = _moi_setting_key(Symbol(a.name))
    val = if key === :verbose
        !o.silent
    elseif key === :time_limit
        o.time_limit
    elseif key === :max_threads
        something(o.num_threads, Settings(Float64).max_threads)
    else
        get(o.options, string(key), getproperty(Settings(Float64), key))
    end
    return a.name == "verbosity" ? Int(val) : deepcopy(val)
end
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.VariableIndex}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}) = true
MOI.get(o::Optimizer, ::MOI.ResultCount) = o.result === nothing ? 0 : 1
MOI.get(o::Optimizer, ::MOI.RawStatusString) = o.result === nothing ? "NotStarted" : string(o.result.status)

MOI.get(o::Optimizer, ::MOI.SolverName) = "SDPX"
MOI.get(o::Optimizer, ::MOI.SolverVersion) = "0.6.1"

MOI.get(o::Optimizer, ::MOI.NumberOfVariables) =
    MOI.get(o.model, MOI.NumberOfVariables())

MOI.get(o::Optimizer, ::MOI.NumberOfConstraints{F,S}) where {F,S} =
    MOI.get(o.model, MOI.NumberOfConstraints{F,S}())

function MOI.get(o::Optimizer, ::MOI.ListOfConstraintTypesPresent)
    return MOI.get(o.model, MOI.ListOfConstraintTypesPresent())
end

# ---------------------------------------------------------------------------
# optimize! — convert MOI model to conic form and solve
# ---------------------------------------------------------------------------

function _moi_extract_row_terms(f::MOI.ScalarAffineFunction{Float64})
    return f.terms, f.constant
end

function _moi_extract_row_terms(v::MOI.VariableIndex)
    return [MOI.ScalarAffineTerm(1.0, v)], 0.0
end

function _moi_scalar_row(terms, var_to_col)
    row = Tuple{Int,Float64}[]
    for term in terms
        push!(row, (var_to_col[term.variable], term.coefficient))
    end
    return row
end

function _moi_convert_to_conic(o::Optimizer)
    model = o.model
    nvars = MOI.get(model, MOI.NumberOfVariables())
    var_list = MOI.get(model, MOI.ListOfVariableIndices())
    var_to_col = Dict{MOI.VariableIndex,Int}()
    for (col, v) in enumerate(var_list)
        var_to_col[v] = col
    end

    a_rows = Int[]
    a_cols = Int[]
    a_vals = Float64[]
    b = Float64[]
    cones = Any[]
    row = 0

    function emit!(coeffs, constant)
        for (col, val) in coeffs
            iszero(val) && continue
            push!(a_rows, row + 1)
            push!(a_cols, col)
            push!(a_vals, val)
        end
        push!(b, constant)
        row += 1
    end

    empty!(o.constraint_rows)
    # Iterate over constraint types in a deterministic order.
    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        cis = MOI.get(model, MOI.ListOfConstraintIndices{F,S}())
        for ci in cis
            func = MOI.get(model, MOI.ConstraintFunction(), ci)
            set = MOI.get(model, MOI.ConstraintSet(), ci)
            first_row = row + 1
            _moi_emit_constraint!(emit!, cones, func, set, var_to_col)
            o.constraint_rows[ci] = first_row:row
        end
    end

    # Objective — handle ScalarAffineFunction and VariableIndex.
    sense = MOI.get(model, MOI.ObjectiveSense())
    q = zeros(nvars)
    obj_constant = 0.0
    obj_type = MOI.get(model, MOI.ObjectiveFunctionType())
    pi = Int[]; pj = Int[]; pv = Float64[]
    if sense === MOI.FEASIBILITY_SENSE
        # Ignore any stored objective, including a quadratic one.
    elseif obj_type === MOI.ScalarQuadraticFunction{Float64}
        f = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}())
        for term in f.affine_terms
            q[var_to_col[term.variable]] += term.coefficient
        end
        for term in f.quadratic_terms
            i, j = minmax(var_to_col[term.variable_1], var_to_col[term.variable_2])
            # MOI diagonal terms already use the one-half convention.
            push!(pi, i); push!(pj, j); push!(pv, term.coefficient)
        end
        obj_constant = f.constant
    elseif obj_type === MOI.ScalarAffineFunction{Float64}
        obj_func = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
        for term in obj_func.terms
            q[var_to_col[term.variable]] += term.coefficient
        end
        obj_constant = obj_func.constant
    elseif obj_type === MOI.VariableIndex
        obj_func = MOI.get(model, MOI.ObjectiveFunction{MOI.VariableIndex}())
        q[var_to_col[obj_func]] += 1.0
    elseif obj_type !== Nothing
        error("unsupported MOI objective type: $obj_type")
    end

    A = sparse(a_rows, a_cols, a_vals, row, nvars)
    P = sparse(pi, pj, pv, nvars, nvars)
    return P, A, b, cones, q, sense, obj_constant
end

function _moi_emit_constraint!(emit!, cones, func, set, var_to_col)
    if func isa MOI.VariableIndex
        _moi_emit_variable_bound!(emit!, cones, func, set, var_to_col)
    elseif func isa MOI.ScalarAffineFunction{Float64}
        _moi_emit_scalar_affine!(emit!, cones, func, set, var_to_col)
    elseif func isa MOI.VectorOfVariables
        _moi_emit_vector_vars!(emit!, cones, func, set, var_to_col)
    elseif func isa MOI.VectorAffineFunction{Float64}
        _moi_emit_vector_affine!(emit!, cones, func, set, var_to_col)
    else
        error("unsupported MOI function type: $(typeof(func))")
    end
end

function _moi_emit_variable_bound!(emit!, cones, v::MOI.VariableIndex, set, var_to_col)
    col = var_to_col[v]
    if set isa MOI.GreaterThan{Float64}
        emit!([(col, -1.0)], -set.lower)
        push!(cones, NonnegativeConeT(1))
    elseif set isa MOI.LessThan{Float64}
        emit!([(col, 1.0)], set.upper)
        push!(cones, NonnegativeConeT(1))
    elseif set isa MOI.EqualTo{Float64}
        emit!([(col, 1.0)], set.value)
        push!(cones, ZeroConeT(1))
    elseif set isa MOI.Interval{Float64}
        emit!([(col, -1.0)], -set.lower)
        emit!([(col, 1.0)], set.upper)
        push!(cones, NonnegativeConeT(2))
    end
end

function _moi_emit_scalar_affine!(emit!, cones, f::MOI.ScalarAffineFunction{Float64}, set, var_to_col)
    terms = _moi_scalar_row(f.terms, var_to_col)
    if set isa MOI.GreaterThan{Float64}
        emit!([(c, -v) for (c, v) in terms], f.constant - set.lower)
        push!(cones, NonnegativeConeT(1))
    elseif set isa MOI.LessThan{Float64}
        emit!(terms, set.upper - f.constant)
        push!(cones, NonnegativeConeT(1))
    elseif set isa MOI.EqualTo{Float64}
        emit!(terms, set.value - f.constant)
        push!(cones, ZeroConeT(1))
    elseif set isa MOI.Interval{Float64}
        emit!([(c, -v) for (c, v) in terms], f.constant - set.lower)
        emit!(terms, set.upper - f.constant)
        push!(cones, NonnegativeConeT(2))
    end
end

function _moi_row_scale(set, index)
    set isa MOI.PositiveSemidefiniteConeTriangle || return 1.0
    k = set.side_dimension
    1 <= index <= widemul(k,k+1) ÷ 2 || error("invalid PSD row")
    # Diagonal positions are triangular numbers. Avoid scanning every earlier
    # entry separately for each PSD row during large MOI model conversion.
    discriminant = widemul(8,index) + 1
    root = isqrt(discriminant)
    return root*root == discriminant ? 1.0 : sqrt(2.0)
end

function _moi_emit_vector_vars!(emit!, cones, f::MOI.VectorOfVariables, set, var_to_col)
    for i in 1:MOI.dimension(set)
        col = var_to_col[f.variables[i]]
        set isa MOI.Zeros ?
            emit!([(col, 1.0)], 0.0) :
            emit!([(col, -_moi_row_scale(set, i))], 0.0)
    end
    push!(cones, _moi_cone_tag(set))
end

function _moi_emit_vector_affine!(emit!, cones, f::MOI.VectorAffineFunction{Float64}, set, var_to_col)
    k = MOI.dimension(set)
    component_terms = [Tuple{Int,Float64}[] for _ in 1:k]
    for term in f.terms
        push!(component_terms[term.output_index],
              (var_to_col[term.scalar_term.variable], term.scalar_term.coefficient))
    end
    for i in 1:k
        if set isa MOI.Zeros
            emit!(component_terms[i], -f.constants[i])
        else
            emit!([(c, -v * _moi_row_scale(set, i)) for (c, v) in component_terms[i]], f.constants[i] * _moi_row_scale(set, i))
        end
    end
    push!(cones, _moi_cone_tag(set))
end

function _moi_cone_tag(set)
    if set isa MOI.Nonnegatives
        return NonnegativeConeT(Int(set.dimension))
    elseif set isa MOI.Zeros
        return ZeroConeT(Int(set.dimension))
    elseif set isa MOI.SecondOrderCone
        return SecondOrderConeT(Int(set.dimension))
    elseif set isa MOI.PositiveSemidefiniteConeTriangle
        return PSDTriangleConeT(Int(set.side_dimension))
    elseif set isa MOI.ExponentialCone
        return ExponentialConeT()
    elseif set isa MOI.PowerCone{Float64}
        return PowerConeT(set.exponent)
    else
        error("unsupported MOI set: $(typeof(set))")
    end
end

function MOI.optimize!(o::Optimizer)
    o.result = nothing
    o.solve_time_sec = 0.0
    P, A, b, cones, q, sense, obj_constant = _moi_convert_to_conic(o)
    if sense === MOI.MAX_SENSE
        P = -P
        q = -q
    end

    sense === MOI.FEASIBILITY_SENSE && fill!(q, 0.0)
    o.objective_sign = sense === MOI.MAX_SENSE ? -1.0 : 1.0
    o.objective_constant = sense === MOI.FEASIBILITY_SENSE ? 0.0 : obj_constant
    settings = Settings(Float64)
    for (key, val) in o.options
        setproperty!(settings, Symbol(key), val)
    end
    settings.verbose = !o.silent
    settings.time_limit = o.time_limit
    o.num_threads === nothing || (settings.threads = o.num_threads)
    o.result = nothing
    t0 = time()
    o.result = solve(P, q, A, b, cones; settings=settings)
    o.solve_time_sec = time() - t0
    return
end

# ---------------------------------------------------------------------------
# Result accessors
# ---------------------------------------------------------------------------

function MOI.get(o::Optimizer, ::MOI.TerminationStatus)
    o.result === nothing && return MOI.OPTIMIZE_NOT_CALLED
    s = status(o.result)
    s in (Optimal, :optimal) && return MOI.OPTIMAL
    s in (AlmostOptimal, :almost_optimal) && return MOI.ALMOST_OPTIMAL
    s in (PrimalInfeasible, :primal_infeasible) && return MOI.INFEASIBLE
    s in (DualInfeasible, :dual_infeasible) && return MOI.DUAL_INFEASIBLE
    s in (IterLimit, :iteration_limit) && return MOI.ITERATION_LIMIT
    s in (TimeLimit, :time_limit) && return MOI.TIME_LIMIT
    s in (NumericalBreakdown, :numerical_breakdown) && return MOI.NUMERICAL_ERROR
    s in (NumericalFailure, :numerical_failure) && return MOI.NUMERICAL_ERROR
    s in (Stalled, :stalled) && return MOI.SLOW_PROGRESS
    s in (InsufficientPrecision, :insufficient_precision) && return MOI.NUMERICAL_ERROR
    return MOI.OTHER_ERROR
end

function MOI.get(o::Optimizer, a::MOI.PrimalStatus)
    (o.result === nothing || a.result_index != 1) && return MOI.NO_SOLUTION
    s = status(o.result)
    s in (Optimal, :optimal) && return MOI.FEASIBLE_POINT
    s in (AlmostOptimal, :almost_optimal) && return MOI.NEARLY_FEASIBLE_POINT
    s in (DualInfeasible, :dual_infeasible) && return MOI.INFEASIBILITY_CERTIFICATE
    return MOI.NO_SOLUTION
end

function MOI.get(o::Optimizer, a::MOI.DualStatus)
    (o.result === nothing || a.result_index != 1) && return MOI.NO_SOLUTION
    s = status(o.result)
    s in (Optimal, :optimal) && return MOI.FEASIBLE_POINT
    s in (AlmostOptimal, :almost_optimal) && return MOI.NEARLY_FEASIBLE_POINT
    s in (PrimalInfeasible, :primal_infeasible) && return MOI.INFEASIBILITY_CERTIFICATE
    return MOI.NO_SOLUTION
end

function MOI.get(o::Optimizer, a::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(o, a)
    return o.objective_sign * primal_objective(o.result) + o.objective_constant
end

function MOI.get(o::Optimizer, a::MOI.DualObjectiveValue)
    MOI.check_result_index_bounds(o, a)
    return o.objective_sign * dual_objective(o.result) + o.objective_constant
end

function MOI.get(o::Optimizer, a::MOI.VariablePrimal, v::MOI.VariableIndex)
    MOI.check_result_index_bounds(o, a)
    col = get(o.var_to_col, v, 0)
    col == 0 && throw(MOI.InvalidIndex(v))
    return value(o.result)[col]
end

function MOI.get(o::Optimizer, a::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex)
    MOI.check_result_index_bounds(o, a)
    MOI.is_valid(o.model, ci) || throw(MOI.InvalidIndex(ci))
    f = MOI.get(o.model, MOI.ConstraintFunction(), ci)
    x(v) = o.result.x[o.var_to_col[v]]
    # Certificates are directions, not points: omit affine constants.
    ray = MOI.get(o, MOI.PrimalStatus(a.result_index)) == MOI.INFEASIBILITY_CERTIFICATE
    if f isa MOI.VariableIndex
        return x(f)
    elseif f isa MOI.ScalarAffineFunction
        return sum(t.coefficient * x(t.variable) for t in f.terms; init=0.0) +
               (ray ? 0.0 : f.constant)
    elseif f isa MOI.VectorOfVariables
        return x.(f.variables)
    else
        out = ray ? zeros(length(f.constants)) : copy(f.constants)
        for t in f.terms
            out[t.output_index] += t.scalar_term.coefficient * x(t.scalar_term.variable)
        end
        return out
    end
end

function MOI.get(o::Optimizer, a::MOI.ConstraintDual, ci::MOI.ConstraintIndex)
    MOI.check_result_index_bounds(o, a)
    MOI.is_valid(o.model, ci) || throw(MOI.InvalidIndex(ci))
    set = MOI.get(o.model, MOI.ConstraintSet(), ci)
    z = o.result.z[o.constraint_rows[ci]]
    # MOI dual signs depend on the constraint, not on the objective sense.
    if set isa MOI.GreaterThan
        return only(z)
    elseif set isa Union{MOI.LessThan,MOI.EqualTo}
        return -only(z)
    elseif set isa MOI.Interval
        return z[1] - z[2]
    elseif set isa MOI.Zeros
        return -z
    elseif set isa MOI.PositiveSemidefiniteConeTriangle
        # Convert svec duals to the ordinary triangle with trace inner product.
        return [z[i] / _moi_row_scale(set, i) for i in eachindex(z)]
    end
    return z
end

MOI.get(o::Optimizer, ::MOI.SolveTimeSec) = o.solve_time_sec
MOI.get(o::Optimizer, ::MOI.SimplexIterations) = 0
MOI.get(o::Optimizer, ::MOI.BarrierIterations) =
    o.result === nothing ? 0 : o.result.iterations
