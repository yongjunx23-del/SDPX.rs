function optimize!(model::Model{T}; settings=nothing, outputs::Union{Nothing,Outputs}=nothing, warm_start=nothing) where {T}
    warm_start === nothing || throw(ArgumentError("the unified solver does not support warm starts"))
    p=_frontend_compile(model,settings)
    outputs=_frontend_outputs(settings,outputs)
    _frontend_validate_outputs(_frontend_layout(p),outputs)
    raw=_solve_program(p;settings=settings)
    recovery=_frontend_recovery_program(model,p,raw)
    return recover_result(recovery,raw;outputs=outputs)
end
solve(model::Model;kwargs...)=optimize!(model;kwargs...)

struct PreparedStructureMismatch <: Exception
    reason::Symbol
    message::String
end
Base.showerror(io::IO,e::PreparedStructureMismatch)=print(io,e.message)
mutable struct PreparedFrontend{T,H,S}
    program::FrontendProgram{T}
    handle::H
    settings::S
    outputs::Outputs
    equality_rhs_only::Bool
end
function prepare(model::Model;settings=nothing,outputs::Union{Nothing,Outputs}=nothing)
    return _prepare_frontend(_frontend_compile(model,settings),settings,_frontend_outputs(settings,outputs),false)
end
function _prepare_frontend(p,settings,outputs,equality_rhs_only)
    _frontend_validate_outputs(_frontend_layout(p),outputs)
    handle=_prepare_program(p;settings=settings)
    return PreparedFrontend(p,handle,settings,normalize_outputs(outputs),equality_rhs_only)
end
function _frontend_same_structure(a,b)
    a.precision_bits == b.precision_bits && a.P == b.P && a.A == b.A || return false
    a.A.colptr==b.A.colptr && a.A.rowval==b.A.rowval && a.P.colptr==b.P.colptr && a.P.rowval==b.P.rowval || return false
    length(a.cones)==length(b.cones) || return false
    all(typeof(x)===typeof(y) && all(isequal(getfield(x,k),getfield(y,k)) for k in 1:fieldcount(typeof(x))) for (x,y) in zip(a.cones,b.cones)) || return false
    # Updates may change numbers, never block shape, domain, or variable order.
    for (ab,bb) in ((a.variable_blocks,b.variable_blocks),(a.constraint_blocks,b.constraint_blocks))
        length(ab)==length(bb) || return false
        all(_frontend_domain_equal(x.domain,y.domain) && x.shape==y.shape && x.offset==y.offset && x.length==y.length for (x,y) in zip(ab,bb)) || return false
    end
    return true
end
function _frontend_updated(p::FrontendProgram{T};objective=nothing,rhs=nothing,equality_rhs_only=false) where {T}
    return _owned_arithmetic_scope(T,p.precision_bits) do
        own(v)=owned_arithmetic_copy(T,v;precision_bits=p.precision_bits)
        q=owned_vector_copy(T,p.q;precision_bits=p.precision_bits)
        b=owned_vector_copy(T,p.b;precision_bits=p.precision_bits)
        if objective !== nothing
            length(objective)==length(q) || throw(DimensionMismatch("replacement objective length must be $(length(q))"))
            all(isfinite,objective) || throw(ArgumentError("replacement objective must be finite"))
            for i in eachindex(q)
                q[i]=own(p.objective_sign*own(objective[i]))
            end
        end
        if rhs !== nothing
            blocks=equality_rhs_only ? filter(x -> x.domain isa ZeroCone,p.constraint_blocks) : p.constraint_blocks
            expected=sum(x.length for x in blocks;init=0)
            length(rhs)==expected || throw(DimensionMismatch("replacement RHS length must be $expected"))
            all(isfinite,rhs) || throw(ArgumentError("replacement RHS must be finite"))
            offset=1
            for block in blocks
                vals=owned_vector_copy(T,rhs[offset:offset+block.length-1];precision_bits=p.precision_bits)
                b[block.rows]=-block.transform*vals
                offset+=block.length
            end
        end
        all(isfinite,q) && all(isfinite,b) || throw(ArgumentError("updated data is not finite in working arithmetic"))
        return FrontendProgram(p.P,q,p.A,b,p.cones,p.identity,p.precision_bits,p.objective_sign,
            deepcopy(p.objective_constant),p.variable_blocks,p.constraint_blocks)
    end
end
function _solve_frontend!(prepared::PreparedFrontend;objective=nothing,rhs=nothing,warm_start=:previous)
    # :previous is the historical prepared default, meaning retain the
    # prepared symbolic workspace. The public engine has never accepted a
    # supplied primal/dual warm-start point.
    warm_start in (nothing,:previous,:none) || throw(ArgumentError("explicit warm starts are unsupported"))
    p=_frontend_updated(prepared.program;objective,rhs,equality_rhs_only=prepared.equality_rhs_only)
    raw=_solve_prepared_program!(prepared.handle,p;settings=prepared.settings)
    prepared.program=p
    return recover_result(p,raw;outputs=prepared.outputs)
end
function _solve_frontend!(prepared::PreparedFrontend,model::Model;objective=nothing,rhs=nothing,warm_start=:previous)
    # Match the precision of the prepared numerical workspace, including when
    # an explicit setting exceeds the builder's nominal arithmetic metadata.
    _frontend_working_bits(model,prepared.settings)
    p=compile_model(model,eltype(model),prepared.program.precision_bits)
    _frontend_same_structure(prepared.program,p) || throw(PreparedStructureMismatch(:structure_changed,"prepared problem coefficient structure or values changed"))
    # Keep the original problem on failure, including invalid updates.
    p=_frontend_updated(p;objective,rhs,equality_rhs_only=prepared.equality_rhs_only)
    warm_start in (nothing,:previous,:none) || throw(ArgumentError("explicit warm starts are unsupported"))
    _frontend_validate_outputs(_frontend_layout(p),prepared.outputs)
    raw=_solve_prepared_program!(prepared.handle,p;settings=prepared.settings)
    prepared.program=p
    return recover_result(p,raw;outputs=prepared.outputs)
end

_frontend_domain_equal(a,b)=typeof(a)===typeof(b) && (!(a isa PowerCone) || a.alpha==b.alpha)

_frontend_outputs(settings,outputs)=outputs===nothing ? (settings===nothing ? Outputs() : settings.outputs) : outputs


function _frontend_working_bits(model::Model{T},settings) where {T}
    settings===nothing && return precision_bits(model)
    settings isa Settings{T} || throw(ArgumentError("settings arithmetic must match model arithmetic $T"))
    return settings.precision_bits
end
_frontend_compile(model::Model,settings)=compile_model(model,eltype(model),_frontend_working_bits(model,settings))
function _frontend_recovery_program(model::Model,p::FrontendProgram,raw)
    bits=hasproperty(raw,:info) && hasproperty(raw.info,:precision_bits) ? raw.info.precision_bits : p.precision_bits
    bits==p.precision_bits && return p
    # A precision retry must rebuild coordinate maps and the objective offset
    # from original model data, never lift an already rounded canonical map.
    return compile_model(model,eltype(model),bits)
end

function solve!(prepared::PreparedFrontend,args...;kwargs...)
    lock(prepared.handle.lock) do
        _require_open(prepared.handle)
        _solve_frontend!(prepared,args...;kwargs...)
    end
end
