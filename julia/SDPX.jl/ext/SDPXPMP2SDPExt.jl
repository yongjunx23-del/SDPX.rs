module SDPXPMP2SDPExt

using SDPX, PMP2SDP, SparseArrays

"""
    SDPX.sampled_program(c::PMP2SDP.SampledPrimalConic; settings=nothing, T=nothing)

Explicitly select factor-authoritative sampled input from a PMP2SDP primal conic.
Original basis descriptors define the PSD operator; the compiled CSC coefficients
are retained only outside those PSD rows. This opt-in adapter does not alter the
ordinary CSC API. It owns its data and preserves stored BigFloat precision unless
arithmetic or settings are explicitly overridden. The conic objective excludes
`c.objective_constant`; use PMP2SDP recovery for the original PMP objective.
"""
function SDPX.sampled_program(c::PMP2SDP.SampledPrimalConic;settings=nothing,T=nothing)
    arithmetic=T===nothing ? (settings===nothing ? eltype(c.q) : typeof(settings).parameters[1]) : T
    # PMP precision_bits includes basis construction precision even for Float64
    # output; only BigFloat storage carries that precision into this solver.
    bits=settings===nothing ? (arithmetic===BigFloat ? c.precision_bits : precision(arithmetic)) : settings.precision_bits
    options=settings===nothing ? SDPX.Settings(arithmetic;precision_bits=bits) : settings
    options isa SDPX.Settings{arithmetic} || throw(ArgumentError("settings arithmetic does not match T"))
    descriptors=PMP2SDP.sampled_schur_structure(c)
    length(descriptors)==length(c.gram_rows)==length(c.gram_dimensions) ||
        throw(DimensionMismatch("PMP sampled Gram metadata"))
    blocks=SDPX.SampledBlock[]
    sampled_rows=falses(length(c.b))
    for (descriptor,rows,dimension) in zip(descriptors,c.gram_rows,c.gram_dimensions)
        descriptor.rows==rows || throw(DimensionMismatch("PMP sampled row metadata"))
        dimension==Base.checked_mul(descriptor.dim,size(descriptor.basis,1)) ||
            throw(DimensionMismatch("PMP sampled PSD order"))
        length(rows)==div(Base.checked_mul(dimension,Base.checked_add(dimension,1)),2) ||
            throw(DimensionMismatch("PMP sampled PSD row count"))
        first(rows)>=1 && last(rows)<=length(c.b) || throw(DimensionMismatch("PMP sampled rows out of range"))
        any(view(sampled_rows,rows)) && throw(ArgumentError("overlapping PMP sampled rows"))
        sampled_rows[rows].=true
        weights=fill(-1,length(descriptor.columns))
        push!(blocks,SDPX.SampledBlock(first(rows),first(descriptor.columns),descriptor.dim,descriptor.basis,weights))
    end
    # Copy every non-sampled coefficient exactly, including all equality rows.
    rows=Int[]; values=eltype(c.A)[]; colptr=Int[1]
    for column in axes(c.A,2)
        for p in nzrange(c.A,column)
            row=c.A.rowval[p]
            sampled_rows[row] && continue
            push!(rows,row); push!(values,deepcopy(c.A.nzval[p]))
        end
        push!(colptr,length(rows)+1)
    end
    linear=SparseMatrixCSC(size(c.A)...,colptr,rows,values)
    cones=map(c.cones) do cone
        cone.kind===:zero && return SDPX.ZeroConeT(cone.dim)
        cone.kind===:psd_triangle && return SDPX.PSDTriangleConeT(cone.dim)
        throw(ArgumentError("unsupported PMP cone $(cone.kind)"))
    end
    SDPX.sampled_program(c.q,linear,c.b,cones,blocks;settings=options,T=arithmetic)
end

end # module SDPXPMP2SDPExt
