# The solver variables retain the frontend's raw lower-packed coordinates.
# Only cone rows are transformed: A*x+s=b, s=R*(F*x+g). Thus A=-R*F,
# b=R*g and the original dual covector is R'*z. For PSD rows, R permutes
# lower packing to upper-column svec and scales off-diagonals by sqrt(2).
struct FrontendBlock{T}
    name::Symbol
    domain::ProductConeDomain
    shape::Int
    offset::Int
    length::Int
    rows::UnitRange{Int}
    transform::SparseMatrixCSC{T,Int}
end
struct FrontendProgram{T}
    P::SparseMatrixCSC{T,Int}
    q::Vector{T}
    A::SparseMatrixCSC{T,Int}
    b::Vector{T}
    cones::Vector{SupportedCone}
    identity::UInt64
    precision_bits::Int
    objective_sign::T
    objective_constant::T
    variable_blocks::Vector{FrontendBlock{T}}
    constraint_blocks::Vector{FrontendBlock{T}}
end
Base.eltype(::FrontendProgram{T}) where {T} = T

function _frontend_cone(domain, n, ::Type{T}) where {T}
    domain isa ZeroCone && return ZeroConeT(n)
    domain isa Union{Nonnegative,Nonpositive} && return NonnegativeConeT(n)
    domain isa Union{LorentzCone,RotatedLorentzCone} && return SecondOrderConeT(n)
    domain isa PSDCone && return PSDTriangleConeT(n)
    domain isa ExponentialCone && return ExponentialConeT()
    domain isa PowerCone && return PowerConeT(owned_arithmetic_copy(T,domain.alpha;precision_bits=precision(T)))
    throw(ArgumentError("unsupported cone domain $domain"))
end

function _frontend_transform(domain, shape, ::Type{T}) where {T}
    n = variable_length(domain, shape)
    domain isa Reals && return spzeros(T, 0, n)
    if domain isa PSDCone
        I = Int[]; J = Int[]; V = T[]
        k = 0
        for j in 1:shape, i in 1:j
            k += 1
            push!(I,k); push!(J,psd_packed_index(j,i,shape))
            push!(V, i == j ? one(T) : sqrt(T(2)))
        end
        return sparse(I,J,V,n,n)
    elseif domain isa RotatedLorentzCone
        # Orthogonal map (u,v,w) -> ((u+v)/sqrt(2),(u-v)/sqrt(2),w).
        a = inv(sqrt(T(2)))
        I = [1,1,2,2]; J = [1,2,1,2]; V = T[deepcopy(a),deepcopy(a),deepcopy(a),-a]
        for i in 3:n
            push!(I,i); push!(J,i); push!(V,one(T))
        end
        return sparse(I,J,V,n,n)
    end
    sign = domain isa Nonpositive ? -one(T) : one(T)
    return spdiagm(0 => [deepcopy(sign) for _ in 1:n])
end

function compile_model(model::Model, ::Type{T}=eltype(model), bits::Int=precision_bits(model)) where {T<:AbstractFloat}
    bits >= 2 || throw(ArgumentError("precision_bits must be at least 2"))
    return _owned_arithmetic_scope(T,bits) do
        _compile_frontend(model,T,bits)
    end
end
function _owned_arithmetic_scope(f, ::Type{T}, bits) where {T}
    T === BigFloat || return f()
    return lock(_precision_lock) do
        setprecision(f,BigFloat,bits)
    end
end

function _compile_frontend(model, ::Type{T}, bits) where {T}
    n = num_variables(model)
    function own(v)
        value=owned_arithmetic_copy(T,v;precision_bits=bits)
        isfinite(value) || throw(ArgumentError("coefficient is not finite in working arithmetic"))
        return value
    end
    I=Int[]; J=Int[]; V=T[]; b=T[]
    cones=SupportedCone[]
    vb=FrontendBlock{T}[]; cb=FrontendBlock{T}[]
    function append_block(record, offset, expressions, variable)
        R = _frontend_transform(record.domain,record.shape,T)
        firstrow = length(b)+1
        count = size(R,1)
        g = variable ? [own(0) for _ in 1:record.length] : [own(e.constant) for e in expressions]
        append!(b, R*g)
        ri, rj, rv = findnz(R)
        for k in eachindex(rv)
            row=firstrow+ri[k]-1
            if variable
                push!(I,row); push!(J,offset+rj[k]-1); push!(V,own(-rv[k]))
            else
                e=expressions[rj[k]]
                for pos in eachindex(e.indices)
                    1 <= e.indices[pos] <= n || throw(ArgumentError("affine variable index out of bounds"))
                    push!(I,row); push!(J,e.indices[pos]); push!(V,own(-rv[k]*own(e.coefficients[pos])))
                end
            end
        end
        count > 0 && push!(cones,_frontend_cone(record.domain,record.shape,T))
        len = variable ? record.length : length(expressions)
        return FrontendBlock(record.name,deepcopy(record.domain),record.shape,offset,len,firstrow:firstrow+count-1,R)
    end
    for record in model.variable_blocks
        (record.primal_start === nothing && record.dual_slack_start === nothing) ||
            throw(ArgumentError("the unified solver does not support warm starts"))
        push!(vb,append_block(record,record.offset,nothing,true))
    end
    offset=1
    for record in model.constraint_blocks
        record.dual_start === nothing || throw(ArgumentError("the unified solver does not support warm starts"))
        push!(cb,append_block(record,offset,record.expressions,false))
        offset += length(record.expressions)
    end
    sign = own(model.objective !== nothing && model.objective.sense isa Maximize ? -1 : 1)
    q=[own(0) for _ in 1:n]
    constant=own(0)
    if model.objective !== nothing
        e=model.objective.expression
        constant=own(e.constant)
        for k in eachindex(e.indices)
            q[e.indices[k]] += sign*own(e.coefficients[k])
        end
    end
    return FrontendProgram(spzeros(T,n,n),q,sparse(I,J,V,length(b),n),owned_vector_copy(T,b;precision_bits=bits),cones,
        model_identity(model),bits,sign,constant,vb,cb)
end
