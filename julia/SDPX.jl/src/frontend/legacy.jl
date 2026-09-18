# Legacy geometric data is only an ingestion/recovery adapter. All numerical
# work, including repeated updates, uses the same standard-form core solver.
struct _LegacyLP{D}
    data::D
end
struct _LegacySOC{D}
    data::D
end
struct _LegacyPSD{D}
    data::D
end

struct SDPProblem{T,S}
    program::FrontendProgram{T}
    dims::NamedTuple{(:L,:m,:n,:k),Tuple{Int,Int,Int,Vector{Int}}}
    source::S
end
Base.eltype(::SDPProblem{T}) where {T}=T
function _frontend_from_rows(c,F,rhs,domains,::Type{T};bits=_legacy_bits(T,c,F,rhs)) where {T}
    return _owned_arithmetic_scope(T,bits) do
        size(F)==(length(rhs),length(c)) || throw(DimensionMismatch("coefficient matrix dimensions disagree"))
        all(isfinite,c) && all(isfinite,rhs) && all(isfinite,F) || throw(ArgumentError("input data must be finite"))
        own(v)=owned_arithmetic_copy(T,v;precision_bits=bits)
        blocks=FrontendBlock{T}[]; maps=SparseMatrixCSC{T,Int}[]
        cones=SupportedCone[]; offset=1; row=1
        for (domain,shape,name) in domains
            shape>0 || throw(ArgumentError("cone dimension must be positive"))
            R=_frontend_transform(domain,shape,T); len=variable_length(domain,shape)
            push!(maps,R)
            push!(blocks,FrontendBlock(name,deepcopy(domain),shape,offset,len,row:row+size(R,1)-1,R))
            size(R,1)>0 && push!(cones,_frontend_cone(domain,shape,T))
            offset+=len; row+=size(R,1)
        end
        offset-1==length(rhs) || throw(DimensionMismatch("cone dimensions disagree with row count"))
        R=isempty(maps) ? spzeros(T,0,0) : blockdiag(maps...)
        f=owned_sparse_copy(T,SparseArrays.sparse(F);precision_bits=bits)
        q=owned_vector_copy(T,c;precision_bits=bits)
        b=-(R*owned_vector_copy(T,rhs;precision_bits=bits))
        n=length(c)
        vars=n==0 ? FrontendBlock{T}[] : [FrontendBlock(:free_variables,Reals(),n,1,n,1:0,spzeros(T,0,n))]
        p=FrontendProgram(spzeros(T,n,n),q,-R*f,b,cones,_next_model_identity(),bits,own(1),own(0),vars,blocks)
        all(isfinite,p.q) && all(isfinite,p.b) && all(isfinite,p.A) || throw(ArgumentError("converted data must be finite"))
        return p
    end
end
function _legacy_type(T,arrays...)
    ET=T===nothing ? float(promote_type(map(eltype,arrays)...)) : T
    ET === BigFloat ? ArithmeticSpec(BigFloat;precision_bits=precision(BigFloat)) : ArithmeticSpec(ET)
    return ET
end
function _legacy_policy(sparse)
    sparse in (true,false,:auto,:dense,:sparse) || throw(ArgumentError("invalid sparse storage policy"))
end
function linear_program(c::AbstractVector,G::AbstractMatrix,h::AbstractVector;
    Aeq=nothing,beq=nothing,T=nothing,sparse=:auto,validate=true,verbosity=1)
    _legacy_policy(sparse)
    length(c)==size(G,2) && length(h)==size(G,1) || throw(DimensionMismatch("LP input dimensions disagree"))
    E=Aeq===nothing ? spzeros(eltype(G),0,length(c)) : Aeq
    d=beq===nothing ? eltype(h)[] : beq
    size(E)==(length(d),length(c)) || throw(DimensionMismatch("equality dimensions disagree"))
    ET=_legacy_type(T,c,G,h,E,d)
    domains=Tuple{ProductConeDomain,Int,Symbol}[]
    size(G,1)>0 && push!(domains,(Nonnegative(),size(G,1),:inequalities))
    length(d)>0 && push!(domains,(ZeroCone(),length(d),:equalities))
    source=_LegacyLP(deepcopy((c,G,h,E,d,domains)))
    p=_legacy_build(source,ET,_legacy_bits(ET,c,G,h,E,d))
    return SDPProblem(p,(L=size(G,1),m=length(c),n=length(d),k=ones(Int,size(G,1))),source)
end

_coefficient_count(block::AbstractArray{<:Any,3})=size(block,1)
_coefficient_count(block::AbstractVector)=length(block)
_coefficient_matrix(block::AbstractArray{<:Any,3},i)=view(block,i,:,:)
_coefficient_matrix(block::AbstractVector,i)=block[i]
function _legacy_symmetric(M,::Type{T},bits,symmetrize,sym_tol;validate=true,verbosity=1) where {T}
    n=size(M,1)
    size(M,2)==n || throw(DimensionMismatch("PSD coefficient must be square"))
    all(isfinite,M) || throw(ArgumentError("PSD coefficient must be finite"))
    own(v)=owned_arithmetic_copy(T,v;precision_bits=bits)
    if !validate
        return M isa SparseMatrixCSC ? owned_sparse_copy(T,M;precision_bits=bits) : map(own,M)
    end
    if M isa SparseMatrixCSC
        matrix=owned_sparse_copy(T,M;precision_bits=bits)
        difference=matrix-transpose(matrix)
        scale=maximum(abs,matrix;init=zero(T))
        !symmetrize && maximum(abs,difference;init=zero(T))>own(sym_tol)*scale &&
            throw(ArgumentError("PSD coefficient is not symmetric within sym_tol; use symmetrize=true"))
        if symmetrize
            if verbosity>=1 && maximum(abs,difference;init=zero(T))>own(sym_tol)*scale
                @warn "PSD coefficient exceeds symmetry tolerance; averaging both triangles"
            end
            matrix=(matrix+transpose(matrix))/T(2)
            for i in 1:n
                matrix[i,i]=own(M[i,i])
            end
        end
        return owned_sparse_copy(T,matrix;precision_bits=bits)
    end
    out=Matrix{T}(undef,n,n)
    scale=maximum(abs,M;init=zero(T))
    if symmetrize && verbosity>=1 && maximum(abs,M-transpose(M);init=zero(T))>own(sym_tol)*scale
        @warn "PSD coefficient exceeds symmetry tolerance; averaging both triangles"
    end
    for j in 1:n,i in j:n
        a=own(M[i,j]); b=own(M[j,i])
        !symmetrize && abs(a-b)>own(sym_tol)*scale && throw(ArgumentError("PSD coefficient is not symmetric within sym_tol; use symmetrize=true"))
        v=symmetrize && i!=j ? (a+b)/T(2) : a
        out[i,j]=own(v);out[j,i]=own(v)
    end
    return out
end
function ingest(c,A::AbstractVector,C,B,b;T=nothing,sparse=:auto,validate=true,
    symmetrize=true,sym_tol=1e-8,verbosity=1)
    _legacy_policy(sparse)
    sym_tol>=0 && isfinite(sym_tol) || throw(ArgumentError("sym_tol must be finite and nonnegative"))
    length(A)==length(C)>0 || throw(DimensionMismatch("A and C need the same positive block count"))
    size(B)==(length(c),length(b)) || throw(DimensionMismatch("B must have variables × equalities shape"))
    source_types=Type[eltype(c),eltype(B),eltype(b)]
    for l in eachindex(A,C)
        push!(source_types,eltype(C[l]))
        _coefficient_count(A[l])==length(c) || throw(DimensionMismatch("wrong number of PSD coefficients"))
        for i in 1:length(c)
            push!(source_types,eltype(_coefficient_matrix(A[l],i)))
        end
    end
    ET=T===nothing ? float(promote_type(source_types...)) : T
    bits=_legacy_bits(ET,c,A,C,B,b)
    ET===BigFloat ? ArithmeticSpec(ET;precision_bits=bits) : ArithmeticSpec(ET)
    # Keep the original triangles: averaging is arithmetic and must be repeated
    # at the requested precision, including after an automatic retry.
    source=_LegacyPSD(deepcopy((c,A,C,B,b,symmetrize,sym_tol,validate)))
    p=_legacy_build(source,ET,bits;verbosity)
    dims=[size(block,1) for block in C]
    return SDPProblem(p,(L=length(A),m=length(c),n=length(b),k=dims),source)
end

function _legacy_build(source::_LegacyLP,::Type{T},bits) where {T}
    c,G,h,E,d,domains=source.data
    return _owned_arithmetic_scope(T,bits) do
        _frontend_from_rows(c,vcat(G,E),vcat(h,d),domains,T;bits)
    end
end
function _legacy_build(source::_LegacySOC,::Type{T},bits) where {T}
    c,matrices,offsets,E,d,domains=source.data
    return _owned_arithmetic_scope(T,bits) do
        matrix=vcat(matrices...,E)
        rhs=vcat((-offset for offset in offsets)...,d)
        _frontend_from_rows(c,matrix,rhs,domains,T;bits)
    end
end
function _legacy_build(source::_LegacyPSD,::Type{ET},bits;verbosity=0) where {ET}
    c,A,C,B,b,symmetrize,sym_tol,validate=source.data
    return _owned_arithmetic_scope(ET,bits) do
        I=Int[];J=Int[];V=ET[];rhs=ET[];dims=Int[]
        domains=Tuple{ProductConeDomain,Int,Symbol}[]
        offset=0
        for l in eachindex(A,C)
            constant=_legacy_symmetric(C[l],ET,bits,symmetrize,sym_tol;validate,verbosity)
            n=size(constant,1);n>0 || throw(ArgumentError("PSD block must be nonempty"))
            push!(dims,n);push!(domains,(PSDCone(),n,Symbol(:psd_block_,l)))
            for (i,j) in psd_packed_pairs(n)
                push!(rhs,deepcopy(constant[i,j]))
            end
            for variable in 1:length(c)
                coef=_coefficient_matrix(A[l],variable)
                size(coef)==(n,n) || throw(DimensionMismatch("PSD coefficient shape disagrees"))
                M=_legacy_symmetric(coef,ET,bits,symmetrize,sym_tol;validate,verbosity)
                if M isa SparseMatrixCSC
                    rows,cols,vals=findnz(M)
                    for k in eachindex(vals)
                        rows[k]>=cols[k] && !iszero(vals[k]) || continue
                        push!(I,offset+psd_packed_index(rows[k],cols[k],n))
                        push!(J,variable);push!(V,deepcopy(vals[k]))
                    end
                else
                    for (k,(i,j)) in enumerate(psd_packed_pairs(n))
                        iszero(M[i,j]) && continue
                        push!(I,offset+k);push!(J,variable);push!(V,deepcopy(M[i,j]))
                    end
                end
            end
            offset+=psd_packed_length(n)
        end
        append!(rhs,owned_vector_copy(ET,b;precision_bits=bits))
        if B isa SparseMatrixCSC
            cols,rows,vals=findnz(B)
            for k in eachindex(vals)
                push!(I,offset+rows[k]);push!(J,cols[k]);push!(V,owned_arithmetic_copy(ET,vals[k];precision_bits=bits))
            end
        else
            for row in eachindex(b),col in eachindex(c)
                iszero(B[col,row]) && continue
                push!(I,offset+row);push!(J,col);push!(V,owned_arithmetic_copy(ET,B[col,row];precision_bits=bits))
            end
        end
        !isempty(b) && push!(domains,(ZeroCone(),length(b),:equalities))
        p=_frontend_from_rows(c,SparseArrays.sparse(I,J,V,length(rhs),length(c)),rhs,domains,ET;bits)
        return p
    end
end

struct SOCConstraint{T}
    A::SparseMatrixCSC{T,Int}
    b::Vector{T}
end
function SOCConstraint(A::AbstractMatrix,b::AbstractVector;T=nothing)
    size(A,1)==length(b)>0 || throw(DimensionMismatch("SOC row count must equal positive offset length"))
    ET=_legacy_type(T,A,b);bits=_legacy_bits(ET,A,b)
    all(isfinite,A) && all(isfinite,b) || throw(ArgumentError("SOC data must be finite"))
    return SOCConstraint(owned_sparse_copy(ET,SparseArrays.sparse(A);precision_bits=bits),owned_vector_copy(ET,b;precision_bits=bits))
end
struct ConicProblem{T,S}
    program::FrontendProgram{T}
    source::S
end
Base.eltype(::ConicProblem{T}) where {T}=T
function second_order_program(c::AbstractVector,cones::AbstractVector{<:SOCConstraint};Aeq=nothing,beq=nothing,T=nothing)
    isempty(cones) && throw(ArgumentError("at least one SOC block is required"))
    E=Aeq===nothing ? spzeros(eltype(c),0,length(c)) : Aeq
    d=beq===nothing ? eltype(c)[] : beq
    size(E)==(length(d),length(c)) || throw(DimensionMismatch("equality dimensions disagree"))
    all(size(k.A,2)==length(c) for k in cones) || throw(DimensionMismatch("SOC column count disagrees"))
    ET=_legacy_type(T,c,E,d,(k.A for k in cones)...,(k.b for k in cones)...)
    domains=Tuple{ProductConeDomain,Int,Symbol}[(LorentzCone(),length(k.b),Symbol(:soc_,i)) for (i,k) in enumerate(cones)]
    !isempty(d) && push!(domains,(ZeroCone(),length(d),:equalities))
    # Select precision from the original offsets before negation: BigFloat
    # unary minus also rounds at ambient precision, even for owned inputs.
    bits=_legacy_bits(ET,c,E,d,(k.A for k in cones)...,(k.b for k in cones)...)
    source=_LegacySOC(deepcopy((c,[k.A for k in cones],[k.b for k in cones],E,d,domains)))
    return ConicProblem(_legacy_build(source,ET,bits),source)
end
function second_order_program(c::AbstractVector,G::AbstractMatrix,h::AbstractVector;cone_dims=[size(G,1)],kwargs...)
    sum(cone_dims)==size(G,1)==length(h) && all(>(0),cone_dims) || throw(DimensionMismatch("invalid SOC dimensions"))
    cones=SOCConstraint[];offset=0
    for dim in cone_dims
        push!(cones,SOCConstraint(G[offset+1:offset+dim,:],h[offset+1:offset+dim]))
        offset+=dim
    end
    return second_order_program(c,cones;kwargs...)
end

struct LegacyResult{R,T}
    result::R
    x::Vector{T}
    X::Vector{Matrix{T}}
    y::Vector{T}
    Y::Vector{Matrix{T}}
    slack::Vector{Vector{T}}
    dual::Vector{Vector{T}}
end
function _legacy_result(p,raw)
    level=hasproperty(raw,:info) && hasproperty(raw.info,:diagnostics_level) ? raw.info.diagnostics_level : :summary
    result=recover_result(p,raw;outputs=Outputs(diagnostics=level))
    T=eltype(p)
    return _owned_arithmetic_scope(T,p.precision_bits) do
        X=Matrix{T}[];Y=Matrix{T}[];y=T[];slack=Vector{T}[];duals=Vector{T}[]
        for block in p.constraint_blocks
            if block.domain isa PSDCone
                primal=transpose(block.transform)*view(raw.s,block.rows)
                dualvec=transpose(block.transform)*view(raw.z,block.rows)
                xm=Matrix{T}(undef,block.shape,block.shape);ym=similar(xm)
                for (k,(i,j)) in enumerate(psd_packed_pairs(block.shape))
                    # R'R doubles off-diagonals. Both physical primal and
                    # dual matrices divide their packed covectors by two.
                    d=i==j ? one(T) : T(2)
                    xm[i,j]=deepcopy(primal[k]/d);xm[j,i]=deepcopy(xm[i,j])
                    ym[i,j]=deepcopy(dualvec[k]/d);ym[j,i]=deepcopy(ym[i,j])
                end
                push!(X,xm);push!(Y,ym)
            elseif block.domain isa Nonnegative
                # Historical LP interface represents scalar inequalities as
                # 1×1 PSD blocks in X/Y.
                for k in block.rows
                    push!(X,reshape([deepcopy(raw.s[k])],1,1))
                    push!(Y,reshape([deepcopy(raw.z[k])],1,1))
                end
            elseif block.domain isa LorentzCone
                push!(slack,owned_vector_copy(T,raw.s[block.rows];precision_bits=p.precision_bits))
                push!(duals,owned_vector_copy(T,raw.z[block.rows];precision_bits=p.precision_bits))
            elseif block.domain isa ZeroCone
                append!(y,owned_vector_copy(T,raw.z[block.rows];precision_bits=p.precision_bits))
            end
        end
        LegacyResult(result,value(result),X,y,Y,slack,duals)
    end
end
function Base.getproperty(r::LegacyResult,s::Symbol)
    s in fieldnames(typeof(r)) && return getfield(r,s)
    s===:equality_dual && return getfield(r,:y)
    s===:status && return getfield(r,:result).status
    s===:pObj && return primal_objective(getfield(r,:result))
    s===:dObj && return dual_objective(getfield(r,:result))
    s===:gap_rel && return relative_gap(getfield(r,:result))
    s===:p_res && return primal_residual(getfield(r,:result))
    s===:d_res && return dual_residual(getfield(r,:result))
    s===:iterations && return iterations(getfield(r,:result))
    return getproperty(getfield(r,:result),s)
end
for fn in (:status,:termination,:primal_objective,:dual_objective,:objective_value,:dual_objective_value,:primal_residual,:dual_residual,:relative_gap,:iterations,:solve_time,:diagnostics)
    @eval $fn(r::LegacyResult)=$fn(r.result)
end
value(r::LegacyResult)=deepcopy(r.x)
function solve(problem::Union{SDPProblem,ConicProblem};settings=nothing)
    p=_legacy_compile(problem,settings)
    raw=_solve_program(p;settings)
    return _legacy_result(_legacy_recovery_program(problem,p,raw),raw)
end
optimize!(problem::Union{SDPProblem,ConicProblem};kwargs...)=solve(problem;kwargs...)
solve_lp(c,G,h;settings=nothing,kwargs...)=solve(linear_program(c,G,h;kwargs...);settings)
solve_socp(problem::ConicProblem;kwargs...)=solve(problem;kwargs...)
struct PreparedLegacy{F}
    frontend::F
end
function prepare(problem::Union{SDPProblem,ConicProblem};settings=nothing)
    return PreparedLegacy(_prepare_frontend(_legacy_compile(problem,settings),settings,Outputs(),true))
end
function _solve_legacy!(prepared::PreparedLegacy;objective=nothing,rhs=nothing,warm_start=:previous)
    f=prepared.frontend
    warm_start in (nothing,:previous,:none) || throw(ArgumentError("explicit warm starts are unsupported"))
    p=_frontend_updated(f.program;objective,rhs,equality_rhs_only=true)
    raw=_solve_prepared_program!(f.handle,p;settings=f.settings)
    f.program=p
    return _legacy_result(p,raw)
end
function _solve_legacy!(prepared::PreparedLegacy,problem::Union{SDPProblem,ConicProblem};objective=nothing,rhs=nothing,warm_start=:previous)
    f=prepared.frontend
    _legacy_working_bits(problem,f.settings)
    p=_legacy_build(problem.source,eltype(problem),f.program.precision_bits)
    _frontend_same_structure(f.program,p) || throw(PreparedStructureMismatch(:structure_changed,"prepared problem coefficient structure or values changed"))
    warm_start in (nothing,:previous,:none) || throw(ArgumentError("explicit warm starts are unsupported"))
    p=_frontend_updated(p;objective,rhs,equality_rhs_only=true)
    raw=_solve_prepared_program!(f.handle,p;settings=f.settings)
    f.program=p
    return _legacy_result(p,raw)
end

Base.propertynames(r::LegacyResult,private::Bool=false)=(fieldnames(typeof(r))...,:status,:pObj,:dObj,:gap_rel,:p_res,:d_res,:iterations,:equality_dual)

_legacy_source_bits(x::BigFloat)=precision(x)
_legacy_source_bits(x::Number)=0
_legacy_source_bits(x::AbstractArray)=maximum(_legacy_source_bits,x;init=0)
function _legacy_bits(::Type{T},arrays...) where {T}
    T===BigFloat || return precision(T)
    return lock(_precision_lock) do
        max(precision(BigFloat),maximum(_legacy_source_bits,arrays;init=0))
    end
end

function _legacy_working_bits(problem::Union{SDPProblem{T},ConicProblem{T}},settings) where {T}
    settings===nothing && return problem.program.precision_bits
    settings isa Settings{T} || throw(ArgumentError("settings arithmetic must match problem arithmetic $T"))
    return settings.precision_bits
end
function _legacy_compile(problem,settings)
    bits=_legacy_working_bits(problem,settings)
    return _legacy_build(problem.source,eltype(problem),bits)
end
function _legacy_recovery_program(problem,p,raw)
    bits=hasproperty(raw,:info) && hasproperty(raw.info,:precision_bits) ? raw.info.precision_bits : p.precision_bits
    return bits==p.precision_bits ? p : _legacy_build(problem.source,eltype(problem),bits)
end

function solve!(prepared::PreparedLegacy,args...;kwargs...)
    lock(prepared.frontend.handle.lock) do
        _require_open(prepared.frontend.handle)
        _solve_legacy!(prepared,args...;kwargs...)
    end
end
