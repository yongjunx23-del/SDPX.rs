"""Arithmetic contract of a model. BigFloat precision is fixed explicitly."""
struct ArithmeticSpec{T<:AbstractFloat}
    precision_bits::Int
    supports_multifloat::Bool
end
ArithmeticSpec(::Type{Float64}) = ArithmeticSpec{Float64}(53, false)
function ArithmeticSpec(::Type{BigFloat}; precision_bits::Int=256)
    _check_precision(BigFloat, precision_bits)
    ArithmeticSpec{BigFloat}(precision_bits, false)
end
ArithmeticSpec(::Type{T}) where {T<:AbstractFloat} = throw(ArgumentError("unsupported arithmetic $T; use Float64 or BigFloat"))

"""Internal, typed record for one native variable block."""
struct VariableBlockRecord{T<:AbstractFloat}
    name::Symbol
    domain::ProductConeDomain
    shape::Int
    offset::Int
    length::Int
    primal_start::Union{Nothing,Vector{T}}
    dual_slack_start::Union{Nothing,Vector{T}}
end

"""A scalar affine expression in the model's global packed variable order."""
struct ScalarAffine{T<:AbstractFloat}
    model::UInt64
    precision_bits::Int
    indices::Vector{Int}
    coefficients::Vector{T}
    constant::T
end

"""Internal, typed record for one affine-in-cone constraint block."""
struct AffineConstraintRecord{T<:AbstractFloat}
    name::Symbol
    domain::ProductConeDomain
    shape::Int
    expressions::Vector{ScalarAffine{T}}
    refs::Vector{ConstraintRef}
    dual_start::Union{Nothing,Vector{T}}
end

"""Internal, typed record for the single scalar affine objective."""
struct ObjectiveRecord{T<:AbstractFloat}
    sense::Union{Minimize,Maximize}
    expression::ScalarAffine{T}
end

"""
    SDPX.Model{T<:AbstractFloat}

User-facing mathematical model for the model frontend. `T` is the
arithmetic type of every number that enters the model; the immutable
`arithmetic::ArithmeticSpec{T}` records and validates the precision
contract.

Constructors
- `Model(Float64)` — IEEE binary64 arithmetic.
- `Model(BigFloat; precision_bits=256)` — arbitrary-precision arithmetic;
  supported precisions are 128, 256, 512, 768, 1024, and 2048 bits.


The mutable fields are the minimal builder state for B1: 1-based id
counters, concrete `Vector{VariableRef}` / `Vector{ConstraintRef}`
registries, and the typed variable-block records owned by the model
(`VariableBlockRecord{T}`; see modeling/model.jl). Constraint and
objective data, dualization state, and solver choices never live here.
The block registry uses concrete typed vectors and a `Dict{Symbol,Int}`
name→record map; no `Any`-typed state is stored.
"""
mutable struct Model{T<:AbstractFloat}
    arithmetic::ArithmeticSpec{T}
    identity::UInt64
    name::String
    next_variable_id::Int
    next_constraint_id::Int
    next_block_id::Int
    variables::Vector{VariableRef}
    constraints::Vector{ConstraintRef}
    variable_blocks::Vector{VariableBlockRecord{T}}
    block_names::Dict{Symbol,Int}
    constraint_blocks::Vector{AffineConstraintRecord{T}}
    constraint_names::Dict{Symbol,Int}
    objective::Union{Nothing,ObjectiveRecord{T}}
end

# A model identity must remain unique even after the garbage collector reuses
# an object address.  The process-local monotone counter is thread-safe and
# references still store only the resulting value, never a model pointer.
const _MODEL_ID_COUNTER = Base.Threads.Atomic{UInt64}(0)

function _next_model_identity()
    identity = Base.Threads.atomic_add!(_MODEL_ID_COUNTER, UInt64(1)) + UInt64(1)
    identity == 0 && throw(OverflowError("SDPX Model identity counter exhausted"))
    return identity
end

Model(::Type{Float64}; name::AbstractString="") =
    Model{Float64}(
        ArithmeticSpec(Float64),
        _next_model_identity(),
        String(name),
        1,
        1,
        1,
        VariableRef[],
        ConstraintRef[],
        VariableBlockRecord{Float64}[],
        Dict{Symbol,Int}(),
        AffineConstraintRecord{Float64}[],
        Dict{Symbol,Int}(),
        nothing,
    )

function Model(::Type{BigFloat}; precision_bits::Int=256, name::AbstractString="")
    spec = ArithmeticSpec(BigFloat; precision_bits=precision_bits)
    return Model{BigFloat}(
        spec,
        _next_model_identity(),
        String(name),
        1,
        1,
        1,
        VariableRef[],
        ConstraintRef[],
        VariableBlockRecord{BigFloat}[],
        Dict{Symbol,Int}(),
        AffineConstraintRecord{BigFloat}[],
        Dict{Symbol,Int}(),
        nothing,
    )
end

function Model(::Type{T}; name::AbstractString="") where {T<:AbstractFloat}
    return Model{T}(
        ArithmeticSpec(T),
        _next_model_identity(),
        String(name),
        1,
        1,
        1,
        VariableRef[],
        ConstraintRef[],
        VariableBlockRecord{T}[],
        Dict{Symbol,Int}(),
        AffineConstraintRecord{T}[],
        Dict{Symbol,Int}(),
        nothing,
    )
end

Base.eltype(::Type{Model{T}}) where {T} = T
Base.eltype(::Model{T}) where {T} = T

"""
    model_identity(model) -> UInt64

Opaque monotone identity of `model` used by every `VariableRef` /
`ConstraintRef` it owns. It is never a pointer to the model and is never
exposed to the solver layer as a handle.
"""
model_identity(model::Model) = model.identity

"""
    arithmetic(model) -> ArithmeticSpec{T}

Immutable arithmetic/precision metadata of `model`.
"""
arithmetic(model::Model) = model.arithmetic

"""
    precision_bits(model) -> Int

Nominal precision of `model`'s arithmetic (`53` for `Float64`, the
requested `precision_bits >= 2` for `BigFloat`, mantissa estimate for
MultiFloat types).
"""
precision_bits(model::Model) = model.arithmetic.precision_bits

"""
    num_variables(model), num_constraints(model)

Number of frontend variables / constraints registered so far. Both
counts are maintained by the B1 builder through the 1-based id
counters; the foundation only provides the accessors.
"""
num_variables(model::Model) = length(model.variables)
num_constraints(model::Model) = length(model.constraints)

function Base.show(io::IO, model::Model{T}) where {T}
    print(io, "Model{", T, "}(precision_bits=", model.arithmetic.precision_bits,
          ", name=", repr(model.name),
          ", variables=", num_variables(model), ", constraints=", num_constraints(model),
          ", blocks=", length(model.variable_blocks), ")")
end

@inline function _mature_domain_label(domain::ProductConeDomain)
    domain isa PowerCone && return "PowerCone"
    return string(typeof(domain))
end

function _mature_scalar_breakdown(records, kind::Symbol)
    counts = Dict{String,Int}()
    for record in records
        label = _mature_domain_label(record.domain)
        if kind === :variable
            counts[label] = get(counts, label, 0) + record.length
        else
            counts[label] = get(counts, label, 0) + length(record.refs)
        end
    end
    isempty(counts) && return ""
    ordered = sort!(collect(keys(counts)))
    return join((string(label, ": ", counts[label]) for label in ordered), ", ")
end

function Base.show(io::IO, ::MIME"text/plain", model::Model{T}) where {T}
    if T === BigFloat
        println(io, "SDPX Model{BigFloat} (", model.arithmetic.precision_bits, " bits)")
    else
        println(io, "SDPX Model{", T, "}")
    end
    if model.objective === nothing
        println(io, "Objective: none")
    else
        sense = model.objective.sense isa Minimize ? "Minimize" : "Maximize"
        constant = model.objective.expression.constant
        if iszero(constant)
            println(io, "Objective: ", sense)
        else
            println(io, "Objective: ", sense, " (constant=", constant, ")")
        end
    end
    nblocks = length(model.variable_blocks)
    if nblocks == 0
        println(io, "Variables: ", num_variables(model))
    else
        breakdown = _mature_scalar_breakdown(model.variable_blocks, :variable)
        println(io, "Variables: ", num_variables(model), " in ", nblocks, " blocks (", breakdown, ")")
    end
    cblocks = length(model.constraint_blocks)
    if cblocks == 0
        return print(io, "Constraints: ", num_constraints(model))
    end
    breakdown = _mature_scalar_breakdown(model.constraint_blocks, :constraint)
    return print(
        io,
        "Constraints: ", num_constraints(model),
        " in ", cblocks, " blocks (", breakdown, ")",
    )
end
