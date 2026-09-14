"""
    SampledBlock(row_start, column_start, dim, basis, weights)

Factor-authoritative sampled PSD operator. Starts are Julia 1-based indices.
`basis` has L rows and K columns; the block occupies one PSD cone of order
`dim*L`. Its `dim*(dim+1)/2*K` consecutive variable columns are ordered by
`s=1:dim, r=1:s, k=1:K`. Column p is `weights[p]` times
`svec(sym(e_r*e_s') ⊗ (basis[:,k]*basis[:,k]'))`.
The linear CSC operator must be zero on this block's PSD rows. L must be positive;
K may be zero. Basis and weight element types are preserved until `sampled_program`
converts them at the selected precision. Each factor value is copied independently.
"""
struct SampledBlock{TQ<:Real,TW<:Real}
    row_start::Int
    column_start::Int
    dim::Int
    basis::Matrix{TQ}
    weights::Vector{TW}
    function SampledBlock(row_start::Integer,column_start::Integer,dim::Integer,
            basis::AbstractMatrix,weights::AbstractVector)
        row_start>=1 && column_start>=1 && dim>=1 ||
            throw(ArgumentError("sampled starts and dimension must be positive"))
        d=Int(dim)
        L,K=size(basis)
        L>0 || throw(ArgumentError("sampled basis must have positive row count"))
        count=Base.checked_mul(div(Base.checked_mul(d,Base.checked_add(d,1)),2),K)
        length(weights)==count || throw(DimensionMismatch("sampled weights require dim*(dim+1)/2*basis_cols entries"))
        TQ,TW=eltype(basis),eltype(weights)
        TQ<:Real && TW<:Real || throw(ArgumentError("sampled factors must be real"))
        # No promotion here: exact integers/rationals must survive until the
        # program chooses its arithmetic and explicit precision.
        Q=Matrix{TQ}(undef,L,K); w=Vector{TW}(undef,count)
        for i in eachindex(Q); Q[i]=deepcopy(basis[i]); end
        for i in eachindex(w); w[i]=deepcopy(weights[i]); end
        all(isfinite,Q) && all(isfinite,w) || throw(ArgumentError("sampled factors must be finite"))
        new{TQ,TW}(Int(row_start),Int(column_start),d,Q,w)
    end
end

"""Owned standard-form data plus authoritative sampled PSD factors."""
struct SampledProgram{T<:AbstractFloat}
    P::SparseMatrixCSC{T,Int}
    q::Vector{T}
    A::SparseMatrixCSC{T,Int}
    b::Vector{T}
    cones::Vector{SupportedCone}
    precision_bits::Int
    blocks::Vector{SampledBlock{T,T}}
end

"""
    sampled_program([P,] q, A_linear, b, cones, blocks; settings=nothing, T=nothing)

Copy input into an owned sampled program at the explicitly selected arithmetic.
The Rust constructor validates block/cone alignment and structural compatibility.
Use `solve(program)`, `solve_conic(program)` or `prepare(program)` normally.
"""
function sampled_program(P::AbstractMatrix,q::AbstractVector,A::AbstractMatrix,b::AbstractVector,
        cones,blocks;settings=nothing,T=nothing)
    blocks=collect(blocks)
    all(block->block isa SampledBlock,blocks) || throw(ArgumentError("blocks must be SampledBlock values"))
    factors=[a for block in blocks for a in (block.basis,block.weights)]
    arithmetic,bits=_direct_arithmetic(settings,T,P,q,A,b,factors...)
    size(P)==(length(q),length(q)) && size(A)==(length(b),length(q)) ||
        throw(DimensionMismatch("inconsistent sampled program dimensions"))
    owned_blocks=SampledBlock{arithmetic,arithmetic}[]
    for block in blocks
        basis=reshape(owned_vector_copy(arithmetic,vec(block.basis);precision_bits=bits),size(block.basis))
        weights=owned_vector_copy(arithmetic,block.weights;precision_bits=bits)
        push!(owned_blocks,SampledBlock(block.row_start,block.column_start,block.dim,basis,weights))
    end
    SampledProgram(_working_copy(arithmetic,P,bits),_working_copy(arithmetic,q,bits),
        _working_copy(arithmetic,A,bits),_working_copy(arithmetic,b,bits),
        SupportedCone[deepcopy(c) for c in cones],bits,owned_blocks)
end
function sampled_program(q::AbstractVector,A::AbstractMatrix,b::AbstractVector,cones,blocks;kwargs...)
    # Empty P has no scalar payload and contributes only q's arithmetic to inference.
    sampled_program(spzeros(eltype(q),length(q),length(q)),q,A,b,cones,blocks;kwargs...)
end
function _program_sampled_storage(p::SampledProgram)
    anchors=[]; descriptors=CSampledBlock[]
    for block in p.blocks
        basis=_scalar_storage(vec(block.basis)); weights=_scalar_storage(block.weights)
        push!(anchors,(basis,weights))
        push!(descriptors,CSampledBlock(block.row_start-1,block.column_start-1,block.dim,
            size(block.basis,1),size(block.basis,2),basis.descriptor,weights.descriptor))
    end
    (descriptors=descriptors,anchors=anchors)
end
solve(p::SampledProgram;settings=nothing)=_solve_program(p;settings)
function solve_conic(p::SampledProgram;return_result=false,kwargs...)
    r=solve(p;kwargs...)
    return_result ? r : (status(r),value(r),primal_objective(r))
end
prepare(p::SampledProgram;settings=nothing)=PreparedProblem(p,_prepare_program(p;settings))
_updated_program(p::SampledProgram,q,b)=SampledProgram(p.P,q,p.A,b,p.cones,p.precision_bits,p.blocks)
