module SDPXSampledAdapter

using SparseArrays
import SDPX
using ..SDPBSampledPrimal: SampledPrimalConic, sdpx_cones

export sdpx_sampled_program

"""
    sdpx_sampled_program(conic::SampledPrimalConic; settings)

Adapt the unchanged sampled-primal mapping to SDPX's factor-authoritative input.
Only equality rows are retained in A_linear. Every active Gram block supplies
its original Q and weights -1, in the original (matrix column, row, sample)
variable order. The original compiled conic remains available for external audit.
"""
function sdpx_sampled_program(conic::SampledPrimalConic{T};settings) where T
    sampled=conic.sampled
    settings.precision_bits==sampled.bits || throw(ArgumentError("sampled precision mismatch"))
    length(sampled.blocks)==length(conic.block_columns) ||
        throw(DimensionMismatch("sampled block-column mapping"))
    lookup=Dict(block.block_index=>(block,columns)
        for (block,columns) in zip(sampled.blocks,conic.block_columns))
    length(lookup)==length(sampled.blocks) || throw(ArgumentError("duplicate sampled block index"))
    factors=SDPX.SampledBlock{T,T}[]
    for gram in conic.grams
        block,columns=lookup[gram.block_index]
        gram.parity in (0,1) || throw(ArgumentError("sampled parity must be even or odd"))
        Q=block.bases[gram.parity+1]
        size(Q)==(gram.basis_rows,block.num_points) &&
            gram.dimension==Base.checked_mul(block.dim,gram.basis_rows) ||
            throw(DimensionMismatch("sampled Gram basis layout"))
        count=Base.checked_mul(div(Base.checked_mul(block.dim,Base.checked_add(block.dim,1)),2),block.num_points)
        length(columns)==count || throw(DimensionMismatch("sampled primitive column count"))
        # Explicit precision and distinct MPFR storage, without an ambient mutation.
        weights=T[T===BigFloat ? BigFloat(-1;precision=sampled.bits) : T(-1) for _ in 1:count]
        push!(factors,SDPX.SampledBlock(first(gram.slack_rows),first(columns),block.dim,Q,weights))
    end
    # Preserve the compiled equality coefficients exactly; never keep duplicate
    # materialized PSD coefficients alongside their authoritative factors.
    rows=Int[]; values=T[]; colptr=Int[1]
    for column in axes(conic.A,2)
        for p in nzrange(conic.A,column)
            row=conic.A.rowval[p]
            row<=conic.num_equalities || continue
            push!(rows,row); push!(values,deepcopy(conic.A.nzval[p]))
        end
        push!(colptr,length(rows)+1)
    end
    linear=SparseMatrixCSC(size(conic.A)...,colptr,rows,values)
    SDPX.sampled_program(conic.P,conic.q,linear,conic.b,sdpx_cones(conic,SDPX),factors;settings)
end

end # module SDPXSampledAdapter
