# Results own their solved layout and retained coordinates, independently of
# subsequent model mutation and prepared workspace reuse.
struct FrontendLayout
    identity::UInt64
    variable_blocks::Tuple
    constraint_blocks::Tuple
end
struct FrontendData{T}
    values::Vector{T}
    retained::BitVector
end
struct FrontendResult{T,M}
    layout::FrontendLayout
    metrics::M
    outputs::Outputs
    primal_data::Union{Nothing,FrontendData{T}}
    constraint_dual_data::Union{Nothing,FrontendData{T}}
    dual_slack_data::Union{Nothing,FrontendData{T}}
    primal_objective_data::Union{Nothing,T}
    dual_objective_data::Union{Nothing,T}
    precision_bits::Int
end

function _frontend_layout(p::FrontendProgram)
    snapshot(b)=(domain=deepcopy(b.domain),shape=b.shape,offset=b.offset,length=b.length)
    return FrontendLayout(p.identity,Tuple(snapshot(b) for b in p.variable_blocks),Tuple(snapshot(b) for b in p.constraint_blocks))
end
function _frontend_index(layout, ref::Union{VariableRef,ConstraintRef})
    ref.model == layout.identity || throw(ArgumentError("reference belongs to a different model"))
    blocks=ref isa VariableRef ? layout.variable_blocks : layout.constraint_blocks
    1 <= ref.block <= length(blocks) || throw(ArgumentError("reference block is absent from solved layout"))
    block=blocks[ref.block]
    1 <= ref.index <= block.length || throw(ArgumentError("reference coordinate is absent from solved layout"))
    return block.offset+ref.index-1
end
function _frontend_validate_outputs(layout, outputs)
    for spec in (outputs.primal,outputs.constraint_dual,outputs.dual_slack)
        spec isa Symbol && continue
        for ref in spec
            _frontend_index(layout,ref)
        end
    end
    return nothing
end
function _frontend_retain(values::Vector{T}, spec, layout, bits) where {T}
    spec === :none && return nothing
    mask=falses(length(values))
    spec === :all ? fill!(mask,true) : foreach(ref -> mask[_frontend_index(layout,ref)]=true,spec)
    # Unselected values are discarded, including their mutable MPFR limbs.
    data=[owned_arithmetic_copy(T,mask[i] ? values[i] : 0;precision_bits=bits) for i in eachindex(values)]
    return FrontendData(data,mask)
end

"""Recover the raw-coordinate primal and dual covectors from one core result.
`raw` provides x and z plus ordinary public status/metric accessors. Packed
PSD dual accessors return covectors (off-diagonal = twice matrix entry),
while block accessors reconstruct the symmetric dual matrix.
"""
function recover_result(p::FrontendProgram{T},raw;outputs::Outputs=Outputs()) where {T}
    return _owned_arithmetic_scope(T,p.precision_bits) do
        layout=_frontend_layout(p)
        policy=normalize_outputs(outputs)
        _frontend_validate_outputs(layout,policy)
        own(v)=owned_arithmetic_copy(T,v;precision_bits=p.precision_bits)
        ds=[own(0) for _ in p.q]
        y=[own(0) for _ in 1:sum(b.length for b in p.constraint_blocks;init=0)]
        for (blocks,destination) in ((p.variable_blocks,ds),(p.constraint_blocks,y))
            for block in blocks
                isempty(block.rows) && continue
                vals=transpose(block.transform)*view(raw.z,block.rows)
                destination[block.offset:block.offset+block.length-1]=vals
            end
        end
        # Metrics are a small snapshot, never a retained raw solver result.
        core_status=hasproperty(raw,:status) ? raw.status : status(raw)
        terminal=hasmethod(termination,Tuple{typeof(raw)}) ? deepcopy(termination(raw)) :
            (status=core_status,reason=_status_symbol(core_status),stage=:solver,message=string(_status_symbol(core_status)))
        execution=raw isa RawResult ? execution_plan(raw) :
            (engine=:rust,kkt_form=:unknown,provider=:unknown,precision_bits=p.precision_bits)
        metrics=(status=core_status,termination=terminal,execution=execution,primal_residual=own(primal_residual(raw)),
            dual_residual=own(dual_residual(raw)),relative_gap=own(relative_gap(raw)),
            iterations=iterations(raw),solve_time=solve_time(raw))
        po=policy.objectives ? own(p.objective_sign*primal_objective(raw)+p.objective_constant) : nothing
        dob=policy.objectives ? own(p.objective_sign*dual_objective(raw)+p.objective_constant) : nothing
        return FrontendResult(layout,metrics,policy,
            _frontend_retain(raw.x,policy.primal,layout,p.precision_bits),
            _frontend_retain(y,policy.constraint_dual,layout,p.precision_bits),
            _frontend_retain(ds,policy.dual_slack,layout,p.precision_bits),po,dob,p.precision_bits)
    end
end

function _frontend_data(result,field)
    data=field === :primal ? result.primal_data : field === :constraint_dual ? result.constraint_dual_data : result.dual_slack_data
    data === nothing && throw(ResultFieldNotRetained(field))
    return data
end
function _frontend_values(result,field,indices)
    data=_frontend_data(result,field)
    all(i -> data.retained[i],indices) || throw(ResultFieldNotRetained(field))
    return owned_vector_copy(eltype(data.values),data.values[indices];precision_bits=result.precision_bits)
end
function _frontend_get(result,field,ref)
    i=_frontend_index(result.layout,ref)
    return only(_frontend_values(result,field,i:i))
end
function _frontend_block(result,field,block)
    ref=block isa VariableBlockRef ? VariableRef(model_identity(block.model),block.block,1) : ConstraintRef(model_identity(block.model),block.block,1)
    _frontend_index(result.layout,ref)
    blocks=block isa VariableBlockRef ? result.layout.variable_blocks : result.layout.constraint_blocks
    record=blocks[block.block]
    values=_frontend_values(result,field,record.offset:record.offset+record.length-1)
    record.domain isa PSDCone || return values
    T=eltype(values)
    return _owned_arithmetic_scope(T,result.precision_bits) do
        matrix=Matrix{T}(undef,record.shape,record.shape)
        for (k,(i,j)) in enumerate(psd_packed_pairs(record.shape))
            v=values[k]/(field !== :primal && i != j ? T(2) : one(T))
            matrix[i,j]=owned_arithmetic_copy(T,v;precision_bits=result.precision_bits)
            matrix[j,i]=owned_arithmetic_copy(T,v;precision_bits=result.precision_bits)
        end
        matrix
    end
end
for (fn,field,ref,entry,block) in ((:value,:primal,:VariableRef,:VariableEntry,:VariableBlockRef),
    (:dual,:constraint_dual,:ConstraintRef,:ConstraintEntry,:ConstraintBlockRef),
    (:dual_slack,:dual_slack,:VariableRef,:VariableEntry,:VariableBlockRef))
    @eval begin
        $fn(r::FrontendResult) = _frontend_values(r,$(QuoteNode(field)),eachindex(_frontend_data(r,$(QuoteNode(field))).values))
        $fn(r::FrontendResult,ref::$ref) = _frontend_get(r,$(QuoteNode(field)),ref)
        $fn(r::FrontendResult,e::$entry) = $fn(r,e.ref)
        $fn(r::FrontendResult,b::$block) = _frontend_block(r,$(QuoteNode(field)),b)
    end
end
for name in (:primal_residual,:dual_residual,:relative_gap,:iterations,:solve_time)
    @eval $name(r::FrontendResult)=getproperty(r.metrics,$(QuoteNode(name)))
end
function primal_objective(r::FrontendResult)
    r.primal_objective_data === nothing && throw(ResultFieldNotRetained(:objectives))
    return deepcopy(r.primal_objective_data)
end
function dual_objective(r::FrontendResult)
    r.dual_objective_data === nothing && throw(ResultFieldNotRetained(:objectives))
    return deepcopy(r.dual_objective_data)
end
objective_value(r::FrontendResult)=primal_objective(r)
dual_objective_value(r::FrontendResult)=dual_objective(r)
function diagnostics(r::FrontendResult)
    r.outputs.diagnostics === :none && throw(ResultFieldNotRetained(:diagnostics))
    return deepcopy(r.metrics)
end

execution_plan(r::FrontendResult{T}) where {T}=merge(r.metrics.execution,(arithmetic=T,))


# Public getters retain the stable symbolic API; the property exposes the
# typed solver status, as the original Result contract did.
status(r::FrontendResult)=_status_symbol(r.metrics.status)
termination(r::FrontendResult)=deepcopy(r.metrics.termination)
function Base.getproperty(r::FrontendResult,name::Symbol)
    name===:status && return getfield(r,:metrics).status
    name===:termination && return deepcopy(getfield(r,:metrics).termination)
    return getfield(r,name)
end
Base.propertynames(r::FrontendResult,private::Bool=false)=(fieldnames(typeof(r))...,:status,:termination)
