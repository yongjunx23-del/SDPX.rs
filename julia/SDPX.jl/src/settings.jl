struct Tolerances{T<:AbstractFloat}
    primal::Union{Nothing,T}
    dual::Union{Nothing,T}
    gap::Union{Nothing,T}
end
function Tolerances(::Type{T}=Float64; primal=nothing, dual=nothing, gap=nothing) where T<:AbstractFloat
    vals=map((primal,dual,gap)) do x
        x === nothing && return nothing
        v=deepcopy(T(x))
        isfinite(v) && v>0 || throw(ArgumentError("tolerances must be finite and positive"))
        v
    end
    Tolerances{T}(vals...)
end
Tolerances{T}(;kwargs...) where T<:AbstractFloat = Tolerances(T;kwargs...)
struct Limits
    iterations::Int
    time::Float64
    threads::Int
end
function Limits(;iterations=200,time=Inf,threads=1)
    0<=iterations<=typemax(UInt32) && !isnan(time) && time>=0 && 1<=threads<=typemax(UInt32) ||
        throw(ArgumentError("invalid iteration, time or thread limit"))
    Limits(Int(iterations),Float64(time),Int(threads))
end
"""Settings for the Rust SDPX engine. Precision is fixed for the prepared handle."""
mutable struct Settings{T<:AbstractFloat}
    precision_bits::Int
    max_iter::Int
    time_limit::Float64
    max_threads::Int
    verbose::Bool
    equilibration::Symbol
    presolve_enable::Bool
    chordal_decomposition_enable::Bool
    kkt_form::Symbol
    tol_gap_abs::Union{Nothing,T}
    tol_gap_rel::Union{Nothing,T}
    tol_feas::Union{Nothing,T}
    tol_infeas_abs::Union{Nothing,T}
    tol_infeas_rel::Union{Nothing,T}
    outputs::Outputs
end
function Settings(::Type{T}=Float64; precision_bits=precision(T), limits=Limits(),
        tolerances=Tolerances(T), outputs=Outputs(), verbose=false, verbosity=nothing,
        equilibration=:ruiz, presolve_enable=true, chordal_decomposition_enable=true,
        kkt_form=:auto, working_precision_policy=:fixed, provider=:auto, kwargs...) where T<:AbstractFloat
    _check_precision(T,precision_bits)
    working_precision_policy in (:fixed,:auto) || throw(ArgumentError("precision policy must be :fixed or :auto"))
    provider===:auto || throw(ArgumentError("backend selection belongs to the Rust SDPX engine"))
    tolerances isa Tolerances{T} || throw(ArgumentError("tolerance arithmetic must match settings"))
    tolerances.primal !== nothing && tolerances.dual !== nothing && tolerances.primal != tolerances.dual &&
        throw(ArgumentError("the C ABI currently requires equal primal and dual tolerances"))
    feasibility=tolerances.primal === nothing ? tolerances.dual : tolerances.primal
    s=Settings{T}(Int(precision_bits),limits.iterations,limits.time,limits.threads,
        verbosity===nothing ? Bool(verbose) : verbosity>0,equilibration,
        presolve_enable,chordal_decomposition_enable,kkt_form,
        deepcopy(tolerances.gap),deepcopy(tolerances.gap),deepcopy(feasibility),nothing,nothing,normalize_outputs(outputs))
    for (key,val) in kwargs
        setproperty!(s,key,val)
    end
    _validate_settings(s)
end
function Base.getproperty(s::Settings,k::Symbol)
    k===:threads && return getfield(s,:max_threads)
    k===:verbosity && return Int(getfield(s,:verbose))
    k===:limits && return Limits(iterations=s.max_iter,time=s.time_limit,threads=s.max_threads)
    k===:tolerances && return Tolerances{typeof(s).parameters[1]}(deepcopy(s.tol_feas),deepcopy(s.tol_feas),deepcopy(s.tol_gap_abs))
    getfield(s,k)
end
function Base.setproperty!(s::Settings{T},k::Symbol,v) where T
    k===:threads && (k=:max_threads)
    k===:verbosity && (k=:verbose; v=v>0)
    hasfield(typeof(s),k) || throw(ArgumentError("unsupported setting $k"))
    converted=_owned_arithmetic_scope(T,s.precision_bits) do
        v === nothing ? nothing : deepcopy(convert(fieldtype(typeof(s),k),v))
    end
    # Validate a candidate first, preserving settings on rejected mutation.
    candidate=deepcopy(s)
    setfield!(candidate,k,converted)
    _validate_settings(candidate)
    setfield!(s,k,converted)
end
Base.propertynames(s::Settings,private::Bool=false)=(fieldnames(typeof(s))...,:threads,:verbosity,:limits,:tolerances)
function _validate_settings(s::Settings{T}) where T
    _check_precision(T,s.precision_bits)
    Limits(iterations=s.max_iter,time=s.time_limit,threads=s.max_threads)
    s.equilibration in (:off,:none,:ruiz) || throw(ArgumentError("equilibration must be :off or :ruiz"))
    s.kkt_form in (:auto,:augmented,:condensed) || throw(ArgumentError("kkt_form must be :auto, :augmented or :condensed"))
    for key in (:tol_gap_abs,:tol_gap_rel,:tol_feas,:tol_infeas_abs,:tol_infeas_rel)
        v=getfield(s,key)
        v===nothing || (isfinite(v) && v>0) || throw(ArgumentError("$key must be finite and positive"))
    end
    s
end
