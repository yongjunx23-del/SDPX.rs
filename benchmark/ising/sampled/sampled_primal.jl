module SDPBSampledPrimal

using LinearAlgebra, SparseArrays, SHA
using ..SDPBSampledInput
import ..SDPBSampledInput: sdpx_cones
const Input = SDPBSampledInput

export SampledPrimalConic, compile_sampled_primal, sdpx_cones,
    pack_sampled_primal_solution, unpack_sampled_primal_solution,
    sampled_primal_matrices, sampled_primal_mapping_audit

const PRIMAL_MAPPING_SCHEMA = """
SDPB sampled primal; original sampled B,c,b,Q unchanged; no rank reduction.
lambda order: block, matrix column, matrix row<=column, sample innermost.
minimize c'lambda; B'lambda=b; X_j=sum_p lambda_p A_jp PSD.
A_jp=E_rs tensor Q[:,k]Q[:,k]', E_rr=1 and E_rs=E_sr=1/2 for r<s.
conic x=lambda; A=[B';-svec(A_jp)]; rhs=[b;0]; q=concat(c).
PSD order: active block/parity; composite index=matrix_index*basis_rows+basis_index.
conic primal PSD slack=svec(X); dual equality=-SDPB y; dual PSD=svec(Y).
original primal objective=constant+c'lambda; original dual=constant+b'y.
Both equal constant+conic objective at optimality. Old Gram formulation minimizes -b'y.
Off-diagonal svec=sqrt(2)*matrix entry. Matrix inputs must be symmetric.
Audit excludes cone membership and status: caller checks both independently.
"""
const PRIMAL_MAPPING_SHA256 = bytes2hex(sha256(PRIMAL_MAPPING_SCHEMA))

struct PrimalGramLayout
    block_index::Int
    parity::Int
    basis_rows::Int
    dimension::Int
    slack_rows::UnitRange{Int}
end

struct SampledPrimalConic{T}
    P::SparseMatrixCSC{T,Int}
    q::Vector{T}
    A::SparseMatrixCSC{T,Int}
    b::Vector{T}
    cones::Vector{Input.SampledCone}
    grams::Vector{PrimalGramLayout}
    block_columns::Vector{UnitRange{Int}}
    num_equalities::Int
    sampled::SampledSDP{T}
    mapping_sha256::String
end

function compile_sampled_primal(sampled::SampledSDP{T}) where {T}
    Input._with_precision(T,sampled.bits) do
        N = length(sampled.b)
        columns = UnitRange{Int}[]
        n = 0
        for block in sampled.blocks
            last = Base.checked_add(n,length(block.c))
            push!(columns,n+1:last)
            n = last
        end
        grams = PrimalGramLayout[]
        m = N
        for block in sampled.blocks, parity in 0:1
            h = size(block.bases[parity+1],1)
            h == 0 && continue
            dim = Base.checked_mul(block.dim,h)
            last = Base.checked_add(m,Input._triangular(dim))
            push!(grams,PrimalGramLayout(block.block_index,parity,h,dim,m+1:last))
            m = last
        end
        q,rhs = Input._zeros(T,n),Input._zeros(T,m)
        for i in 1:N
            rhs[i] = Input._owned(sampled.b[i])
        end
        I,J,V = Int[],Int[],T[]
        root2 = sqrt(T(2))
        gi = 1
        for (bi,block) in enumerate(sampled.blocks)
            offset = first(columns[bi])-1
            for p in eachindex(block.c)
                q[offset+p] = Input._owned(block.c[p])
                for i in 1:N
                    value = block.B[p,i]
                    iszero(value) && continue
                    push!(I,i); push!(J,offset+p); push!(V,Input._owned(value))
                end
            end
            for Q in block.bases
                h = size(Q,1)
                h == 0 && continue
                layout = grams[gi]
                p = 0
                for s in 1:block.dim, r in 1:s, k in 1:block.num_points
                    p += 1
                    # Transpose the original trace coefficient map, with the
                    # orthonormal svec weight. Ordered basis pairs combine only
                    # when their packed positions genuinely coincide (r=s).
                    for b in 1:h, a in 1:h
                        value = Q[a,k]*Q[b,k]
                        isfinite(value) || throw(ArgumentError("sample coefficient overflow"))
                        iszero(value) && !iszero(Q[a,k]) && !iszero(Q[b,k]) &&
                            throw(ArgumentError("sample coefficient underflow"))
                        iszero(value) && continue
                        i,j = (r-1)*h+a,(s-1)*h+b
                        value = i == j ? -value : -value/root2
                        isfinite(value) && !iszero(value) || throw(ArgumentError("svec coefficient range exceeded"))
                        push!(I,first(layout.slack_rows)+Input._packed(i,j)-1)
                        push!(J,offset+p); push!(V,value)
                    end
                end
                gi += 1
            end
        end
        A = sparse(I,J,V,m,n,+)
        for i in eachindex(A.nzval)
            isfinite(A.nzval[i]) || throw(ArgumentError("assembled sampled coefficient overflow"))
            A.nzval[i] = Input._owned(A.nzval[i])
        end
        cones = Input.SampledCone[]
        N > 0 && push!(cones,Input.SampledCone(:zero,N))
        append!(cones,[Input.SampledCone(:psd_triangle,g.dimension) for g in grams])
        SampledPrimalConic{T}(spzeros(T,n,n),q,A,rhs,cones,grams,columns,N,sampled,PRIMAL_MAPPING_SHA256)
    end
end

sdpx_cones(conic::SampledPrimalConic, solver_module) = [
    c.kind === :zero ? solver_module.ZeroConeT(c.dim) : solver_module.PSDTriangleConeT(c.dim)
    for c in conic.cones]

function _put_matrix!(v,rows,M,n)
    size(M) == (n,n) || throw(DimensionMismatch("sampled matrix dimension"))
    issymmetric(M) || throw(ArgumentError("sampled matrices must be symmetric; audit any symmetrization externally"))
    root2 = sqrt(eltype(v)(2))
    for j in 1:n, i in 1:j
        v[first(rows)+Input._packed(i,j)-1] = Input._owned(i == j ? M[i,j] : root2*M[i,j])
    end
end

function _get_matrix(v,rows,n)
    T = eltype(v)
    M = Matrix{T}(undef,n,n)
    root2 = sqrt(T(2))
    for j in 1:n, i in 1:j
        value = v[first(rows)+Input._packed(i,j)-1]
        value = i == j ? value : value/root2
        M[i,j] = Input._owned(value)
        i != j && (M[j,i] = Input._owned(value))
    end
    M
end

function pack_sampled_primal_solution(conic::SampledPrimalConic{T},lambda::AbstractVector{T},
        y::AbstractVector{T},primal_matrices,dual_matrices) where {T}
    Input._with_precision(T,conic.sampled.bits) do
        length(lambda) == length(conic.q) || throw(DimensionMismatch("sample lambda count"))
        length(y) == conic.num_equalities || throw(DimensionMismatch("sample free variable count"))
        length(primal_matrices) == length(dual_matrices) == length(conic.grams) ||
            throw(DimensionMismatch("sample matrix block count"))
        x = T[Input._owned(v) for v in lambda]
        s,z = Input._zeros(T,length(conic.b)),Input._zeros(T,length(conic.b))
        for i in eachindex(y)
            z[i] = -y[i]
        end
        for (g,X,Y) in zip(conic.grams,primal_matrices,dual_matrices)
            _put_matrix!(s,g.slack_rows,X,g.dimension)
            _put_matrix!(z,g.slack_rows,Y,g.dimension)
        end
        (x=x,s=s,z=z)
    end
end

function unpack_sampled_primal_solution(conic::SampledPrimalConic{T},x::AbstractVector{T},
        s::AbstractVector{T},z::AbstractVector{T}) where {T}
    Input._with_precision(T,conic.sampled.bits) do
        length(x) == length(conic.q) && length(s) == length(z) == length(conic.b) ||
            throw(DimensionMismatch("sampled primal conic point dimensions"))
        (lambda=T[Input._owned(v) for v in x],
         sample_x=[T[Input._owned(x[i]) for i in rows] for rows in conic.block_columns],
         y=T[-z[i] for i in 1:conic.num_equalities],
         primal_matrices=[_get_matrix(s,g.slack_rows,g.dimension) for g in conic.grams],
         dual_matrices=[_get_matrix(z,g.slack_rows,g.dimension) for g in conic.grams])
    end
end

"""Independently reconstruct SDPB X=sum lambda A from original Q, without compiled A."""
function sampled_primal_matrices(sampled::SampledSDP{T},lambda::AbstractVector{T}) where {T}
    Input._with_precision(T,sampled.bits) do
        length(lambda) == sum(length(b.c) for b in sampled.blocks;init=0) ||
            throw(DimensionMismatch("sample lambda count"))
        matrices = Matrix{T}[]
        offset = 0
        for block in sampled.blocks
            for Q in block.bases
                h = size(Q,1)
                h == 0 && continue
                dim = block.dim*h
                X = reshape(Input._zeros(T,dim*dim),dim,dim)
                p = offset
                for s in 1:block.dim, r in 1:s, k in 1:block.num_points
                    p += 1
                    weight = r == s ? lambda[p] : lambda[p]/2
                    for b in 1:h, a in 1:h
                        i,j = (r-1)*h+a,(s-1)*h+b
                        term = weight*(Q[a,k]*Q[b,k])
                        X[i,j] += term
                        r != s && (X[j,i] += term)
                    end
                end
                push!(matrices,X)
            end
            offset += length(block.c)
        end
        matrices
    end
end

_maxabs(v) = isempty(v) ? zero(eltype(v)) : maximum(abs,v)
_scaled_difference(a,b) = _maxabs(a-b)/max(one(eltype(a)),_maxabs(a),_maxabs(b))

"""External original-equation audit; PSD membership and solver status are separate gates."""
function sampled_primal_mapping_audit(conic::SampledPrimalConic{T},x::AbstractVector{T},
        s::AbstractVector{T},z::AbstractVector{T}) where {T}
    Input._with_precision(T,conic.sampled.bits) do
        point = unpack_sampled_primal_solution(conic,x,s,z)
        sampled = conic.sampled
        lhs = Input._zeros(T,conic.num_equalities)
        for (block,rows) in zip(sampled.blocks,conic.block_columns)
            lhs .+= block.B'*point.lambda[rows]
        end
        reconstructed = sampled_primal_matrices(sampled,point.lambda)
        link = zero(T)
        for (X,expected) in zip(point.primal_matrices,reconstructed)
            link = max(link,_scaled_difference(X,expected))
        end
        dual = sampled_residual(sampled,point.y,point.dual_matrices)
        pobj = Input._owned(sampled.constant)
        for (block,rows) in zip(sampled.blocks,conic.block_columns)
            pobj += dot(block.c,point.lambda[rows])
        end
        dobj = sampled_objective(sampled,point.y)
        (mapping_sha256=conic.mapping_sha256,input_sha256=sampled.input_sha256,
         sampled_primal_affine=_scaled_difference(lhs,sampled.b),
         sampled_primal_psd_link=link,zero_slack=_maxabs(view(s,1:conic.num_equalities)),
         sampled_dual_relative=dual.max_relative,sampled_dual_absolute=dual.max_absolute,
         gap=abs(pobj-dobj)/max(one(T),abs(pobj),abs(dobj)),
         primal_objective=pobj,dual_objective=dobj,
         finite=all(isfinite,x)&&all(isfinite,s)&&all(isfinite,z))
    end
end

end # module SDPBSampledPrimal
