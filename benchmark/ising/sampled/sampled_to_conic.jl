module SDPBSampledInput

using JSON, SHA, SparseArrays, LinearAlgebra

export SampledBlock, SampledSDP, GramLayout, SampledConic, read_sampled_sdp,
    compile_sampled_sdp, sdpx_cones, pack_sampled_solution,
    unpack_sampled_solution, sampled_residual, sampled_objective

const UPSTREAM_REVISION = "67ebd5386daef4f7366b3e249a252a3159b06d07"
const SCHEMA_CONTRACT = """
SDPB sampled JSON, split block_info/block_data, revision=$UPSTREAM_REVISION
control.json: num_blocks integer>=0, optional command string
objectives.json: constant decimal string, b decimal-string vector length N
block_info_j.json: dim integer>0, num_points integer>0; j=0:num_blocks-1
block_data_j.json: bilinear_bases_even/odd rectangular decimal-string rows,
  each width num_points (empty outer array means zero basis rows);
  c length num_points*dim*(dim+1)/2; B same height, width N; all decimal strings
sample order: matrix column outer, row<=column, sample inner
Gram order: block j, parity even/odd; composite index=matrix_index*basis_rows+basis_index
Gram variables: unscaled symmetric upper column-major; PSD slack=sqrt(2)-svec
equations: B*y+sum_ordered_basis_pairs(Q[a,k]*Q[b,k]*Y[(r,a),(s,b)])=c
objective: minimize -b'y; recover SDPB maximum=constant+b'y
arithmetic: explicit bits, round to nearest
"""
const SCHEMA_SHA256 = bytes2hex(sha256(SCHEMA_CONTRACT))
const OFFICIAL_SCHEMA_HASHES = [
    "sdp_control_schema.json" => "dbd06f9cab09285462cbd709666d1bb9321cd0d6a86fbb80c9023dfad76fc06b",
    "sdp_objectives_schema.json" => "565c1eb4f836721c5d49824d3285ea57c733371e3b10a57696a816826f2c3146",
    "sdp_block_info_schema.json" => "40a4d6ec391feb371052cfdae11a64d38ee666205232cae55abf439429a55e65",
    "sdp_block_data_schema.json" => "a491be4bbb1fdd770f716a14a7f1cfa21c43b4f6ce0192e655a2010583c86791",
]
const _precision_lock = ReentrantLock()
const _decimal = r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"

struct SampledBlock{T}
    block_index::Int
    dim::Int
    num_points::Int
    bases::NTuple{2,Matrix{T}}
    c::Vector{T}
    B::Matrix{T}
end

struct SampledSDP{T}
    constant::T
    b::Vector{T}
    blocks::Vector{SampledBlock{T}}
    bits::Int
    num_free_variables::Int
    producer_command::String
    input_manifest::Vector{Pair{String,String}}
    input_sha256::String
    schema_sha256::String
    identity_sha256::String
end

struct GramLayout
    block_index::Int                 # SDPB block index, starting at zero
    parity::Int                      # 0=even, 1=odd
    basis_rows::Int
    dimension::Int
    columns::UnitRange{Int}
    slack_rows::UnitRange{Int}
end

struct SampledCone
    kind::Symbol
    dim::Int                         # PSD matrix order, not packed row count
end

struct SampledConic{T}
    P::SparseMatrixCSC{T,Int}
    q::Vector{T}
    A::SparseMatrixCSC{T,Int}
    b::Vector{T}
    cones::Vector{SampledCone}
    grams::Vector{GramLayout}
    y_columns::UnitRange{Int}
    block_rows::Vector{UnitRange{Int}}
    num_equalities::Int
    sampled::SampledSDP{T}
end

_owned(x) = x
_owned(x::BigFloat) = deepcopy(x)
_zeros(::Type{T}, n) where {T} = [zero(T) for _ in 1:n]
_triangular(n::Int) = Base.checked_mul(n, Base.checked_add(n,1)) ÷ 2
_packed(i,j) = (i <= j ? _triangular(j-1)+i : _triangular(i-1)+j)

function _with_precision(f, ::Type{T}, bits::Integer) where {T}
    if T === BigFloat
        bits >= 2 || throw(ArgumentError("BigFloat bits must be at least 2"))
        return lock(_precision_lock) do
            setprecision(BigFloat, bits) do
                setrounding(BigFloat,RoundNearest) do
                    f()
                end
            end
        end
    end
    T === Float64 && bits == 53 || throw(ArgumentError(
        "sampled importer supports BigFloat at explicit bits or Float64 at 53 bits"))
    return f()
end

function _object_keys(value, required, optional, path)
    value isa AbstractDict || throw(ArgumentError("$path must be a JSON object"))
    actual = Set(String.(keys(value)))
    issubset(Set(required),actual) || throw(ArgumentError("$path is missing required fields"))
    issubset(actual,Set(vcat(required,optional))) || throw(ArgumentError("$path has unknown fields"))
    return value
end

function _integer(value, path; positive=false)
    value isa Integer && !(value isa Bool) || throw(ArgumentError("$path must be an integer"))
    value >= (positive ? 1 : 0) && value <= typemax(Int) ||
        throw(ArgumentError("$path is outside the supported dimension range"))
    return Int(value)
end

function _number(::Type{T}, value, path) where {T}
    value isa AbstractString && occursin(_decimal,value) || throw(ArgumentError(
        "$path must be a finite decimal string; raw JSON numbers are not precision-safe"))
    result = parse(T,value)
    isfinite(result) || throw(ArgumentError("$path overflows the requested arithmetic"))
    mantissa = first(split(lowercase(value),'e'))
    iszero(result) && any(c -> '1' <= c <= '9',mantissa) &&
        throw(ArgumentError("$path underflows the requested arithmetic"))
    return result
end

function _vector(::Type{T}, values, path) where {T}
    values isa AbstractVector || throw(ArgumentError("$path must be an array"))
    return T[_number(T,v,"$path[$i]") for (i,v) in enumerate(values)]
end

function _matrix(::Type{T}, rows, width::Int, path; height=nothing) where {T}
    rows isa AbstractVector || throw(ArgumentError("$path must be an array of rows"))
    height === nothing || length(rows) == height || throw(DimensionMismatch("$path height"))
    out = Matrix{T}(undef,length(rows),width)
    for (i,row) in enumerate(rows)
        row isa AbstractVector && length(row) == width || throw(DimensionMismatch("$path row $i width"))
        for j in 1:width
            out[i,j] = _number(T,row[j],"$path[$i][$j]")
        end
    end
    return out
end

"""
Read official, uncompressed `pmp2sdp --outputFormat=json` sampled output.
Decimal strings are parsed directly at `bits`; producer-command precision is
recorded, never guessed to be the importer precision. Auxiliary normalization
and PMP metadata are hashed but are not applied again to the sampled equations.
"""
function read_sampled_sdp(directory::AbstractString; T::Type=BigFloat, bits::Integer)
    return _with_precision(T,bits) do
        isdir(directory) || throw(ArgumentError("sampled SDP directory does not exist: $directory"))
        manifest = Pair{String,String}[]
        function read_json(name)
            path = joinpath(directory,name)
            isfile(path) || throw(ArgumentError(
                "missing $name; generate uncompressed pmp2sdp --outputFormat=json output"))
            bytes = read(path)
            push!(manifest,name => bytes2hex(sha256(bytes)))
            return JSON.parse(String(bytes))
        end
        control = _object_keys(read_json("control.json"),["num_blocks"],["command"],"control.json")
        count = _integer(control["num_blocks"],"control.num_blocks")
        command = get(control,"command","")
        command isa AbstractString || throw(ArgumentError("control.command must be a string"))
        objectives = _object_keys(read_json("objectives.json"),["constant","b"],String[],"objectives.json")
        objective = _vector(T,objectives["b"],"objectives.b")
        constant = _number(T,objectives["constant"],"objectives.constant")
        blocks = SampledBlock{T}[]
        for j in 0:count-1
            info = _object_keys(read_json("block_info_$j.json"),["dim","num_points"],String[],"block_info_$j.json")
            dim = _integer(info["dim"],"block $j dim";positive=true)
            points = _integer(info["num_points"],"block $j num_points";positive=true)
            rows = Base.checked_mul(points,_triangular(dim))
            d = _object_keys(read_json("block_data_$j.json"),
                ["bilinear_bases_even","bilinear_bases_odd","c","B"],String[],"block_data_$j.json")
            bases = (_matrix(T,d["bilinear_bases_even"],points,"block $j even basis"),
                _matrix(T,d["bilinear_bases_odd"],points,"block $j odd basis"))
            c = _vector(T,d["c"],"block $j c")
            length(c) == rows || throw(DimensionMismatch("block $j c length must be $rows"))
            B = _matrix(T,d["B"],length(objective),"block $j B";height=rows)
            push!(blocks,SampledBlock{T}(j,dim,points,bases,c,B))
        end
        # Optional original-PMP metadata does not alter already normalized
        # sampled B,c,b,Q. Include it in the byte identity to retain provenance.
        for name in ("normalization.json","pmp_info.json")
            isfile(joinpath(directory,name)) && read_json(name)
        end
        expected = Set(first.(manifest))
        actual = Set(filter(name -> endswith(name,".json"),readdir(directory)))
        actual == expected || throw(ArgumentError("unexpected JSON files in sampled SDP directory"))
        sort!(manifest;by=first)
        input_hash = bytes2hex(sha256(join(("$name\0$hash\n" for (name,hash) in manifest))))
        identity = bytes2hex(sha256("$input_hash\n$SCHEMA_SHA256\n$T\n$bits\n"))
        return SampledSDP{T}(constant,objective,blocks,Int(bits),length(objective),
            String(command),manifest,input_hash,SCHEMA_SHA256,identity)
    end
end

"""Compile the sampled trace equations to `min q'x, A*x+s=b`.

Packed Gram decision variables are unscaled upper-triangle entries. Only
their PSD slack rows use svec scaling. Empty parity bases contribute no cone.
"""
function compile_sampled_sdp(sampled::SampledSDP{T}) where {T}
    return _with_precision(T,sampled.bits) do
        N = length(sampled.b)
        block_rows = UnitRange{Int}[]
        equations = 0
        for block in sampled.blocks
            last = Base.checked_add(equations,length(block.c))
            push!(block_rows,equations+1:last)
            equations = last
        end
        grams = GramLayout[]
        next_column, next_row = N+1,equations+1
        for block in sampled.blocks, parity in 0:1
            basis_rows = size(block.bases[parity+1],1)
            basis_rows == 0 && continue
            dim = Base.checked_mul(block.dim,basis_rows)
            packed = _triangular(dim)
            last_column = Base.checked_add(next_column,packed-1)
            last_row = Base.checked_add(next_row,packed-1)
            push!(grams,GramLayout(block.block_index,parity,basis_rows,dim,
                next_column:last_column,next_row:last_row))
            next_column = Base.checked_add(last_column,1)
            next_row = Base.checked_add(last_row,1)
        end
        variables, total_rows = next_column-1,next_row-1
        I,J,V = Int[],Int[],T[]
        rhs = _zeros(T,total_rows)
        gram_index = 1
        for (block_index,block) in enumerate(sampled.blocks)
            rows = block_rows[block_index]
            for p in eachindex(block.c)
                row = first(rows)+p-1
                rhs[row] = _owned(block.c[p])
                for n in 1:N
                    value = block.B[p,n]
                    iszero(value) && continue
                    push!(I,row); push!(J,n); push!(V,_owned(value))
                end
            end
            for parity in 0:1
                Q = block.bases[parity+1]
                h = size(Q,1)
                h == 0 && continue
                gram = grams[gram_index]
                p = 0
                for s in 1:block.dim, r in 1:s, k in 1:block.num_points
                    p += 1
                    row = first(rows)+p-1
                    # All ordered basis pairs are required. For r=s,
                    # sparse() adds the two truly identical packed entries.
                    # For r<s, (a,b) and (b,a) usually have different columns.
                    for b in 1:h, a in 1:h
                        value = Q[a,k]*Q[b,k]
                        isfinite(value) || throw(ArgumentError("Gram coefficient overflow; increase arithmetic range"))
                        iszero(value) && !iszero(Q[a,k]) && !iszero(Q[b,k]) &&
                            throw(ArgumentError("Gram coefficient underflow; increase arithmetic range"))
                        iszero(value) && continue
                        i,j = (r-1)*h+a,(s-1)*h+b
                        push!(I,row); push!(J,first(gram.columns)+_packed(i,j)-1)
                        push!(V,value)
                    end
                end
                gram_index += 1
            end
        end
        root2 = sqrt(T(2))
        for gram in grams, j in 1:gram.dimension, i in 1:j
            packed = _packed(i,j)
            push!(I,first(gram.slack_rows)+packed-1)
            push!(J,first(gram.columns)+packed-1)
            push!(V,i == j ? -one(T) : -root2)
        end
        A = sparse(I,J,V,total_rows,variables,+)
        # sparse duplicate accumulation may share references with triplets;
        # detach the retained CSC cells before mutable provider use.
        for i in eachindex(A.nzval)
            isfinite(A.nzval[i]) || throw(ArgumentError("assembled coefficient overflow"))
            A.nzval[i] = _owned(A.nzval[i])
        end
        q = _zeros(T,variables)
        for i in 1:N
            q[i] = _owned(-sampled.b[i])
        end
        cones = SampledCone[]
        equations > 0 && push!(cones,SampledCone(:zero,equations))
        append!(cones,[SampledCone(:psd_triangle,g.dimension) for g in grams])
        return SampledConic{T}(spzeros(T,variables,variables),q,A,rhs,cones,
            grams,1:N,block_rows,equations,sampled)
    end
end

"""Create SDPX public cone tags without loading a solver during import."""
function sdpx_cones(conic::SampledConic, solver_module)
    return [cone.kind === :zero ? solver_module.ZeroConeT(cone.dim) :
        solver_module.PSDTriangleConeT(cone.dim) for cone in conic.cones]
end

function unpack_sampled_solution(conic::SampledConic{T}, x::AbstractVector{T}) where {T}
    length(x) == length(conic.q) || throw(DimensionMismatch("conic solution length"))
    y = T[_owned(x[i]) for i in conic.y_columns]
    matrices = Matrix{T}[]
    for gram in conic.grams
        Y = Matrix{T}(undef,gram.dimension,gram.dimension)
        for j in 1:gram.dimension, i in 1:j
            value = x[first(gram.columns)+_packed(i,j)-1]
            Y[i,j] = _owned(value)
            i != j && (Y[j,i] = _owned(value))
        end
        push!(matrices,Y)
    end
    return (y=y,grams=matrices)
end

function pack_sampled_solution(conic::SampledConic{T}, y::AbstractVector{T}, grams) where {T}
    return _with_precision(T,conic.sampled.bits) do
        length(y) == length(conic.y_columns) || throw(DimensionMismatch("free variable count"))
        length(grams) == length(conic.grams) || throw(DimensionMismatch("Gram block count"))
        x = _zeros(T,length(conic.q))
        for i in eachindex(y)
            x[i] = _owned(y[i])
        end
        for (layout,Y) in zip(conic.grams,grams)
            size(Y) == (layout.dimension,layout.dimension) || throw(DimensionMismatch("Gram matrix size"))
            issymmetric(Y) || throw(ArgumentError("Gram matrices must be symmetric"))
            for j in 1:layout.dimension, i in 1:j
                x[first(layout.columns)+_packed(i,j)-1] = _owned(Y[i,j])
            end
        end
        return x
    end
end

function sampled_objective(sampled::SampledSDP{T}, y::AbstractVector{T}) where {T}
    length(y) == length(sampled.b) || throw(DimensionMismatch("free variable count"))
    return _with_precision(T,sampled.bits) do
        sampled.constant + dot(sampled.b,y)
    end
end

"""External residual of the original sampled equations, without compiled A.

`grams` follows active block/parity order. This reports affine residuals only;
the caller separately checks PSD membership, dual equations, gap and status.
"""
function sampled_residual(sampled::SampledSDP{T}, y::AbstractVector{T}, grams) where {T}
    return _with_precision(T,sampled.bits) do
        length(y) == length(sampled.b) || throw(DimensionMismatch("free variable count"))
        residual,work = T[],T[]
        gram_index = 1
        for block in sampled.blocks
            active = Tuple{Matrix{T},Any}[]
            for Q in block.bases
                h = size(Q,1)
                h == 0 && continue
                gram_index <= length(grams) || throw(DimensionMismatch("missing Gram matrix"))
                Y = grams[gram_index]
                size(Y) == (block.dim*h,block.dim*h) || throw(DimensionMismatch("Gram matrix size"))
                issymmetric(Y) || throw(ArgumentError("Gram matrices must be symmetric"))
                push!(active,(Q,Y))
                gram_index += 1
            end
            p = 0
            for s in 1:block.dim, r in 1:s, k in 1:block.num_points
                p += 1
                value,scale = -block.c[p],abs(block.c[p])
                for n in eachindex(y)
                    term = block.B[p,n]*y[n]
                    value += term
                    scale += abs(term)
                end
                for (Q,Y) in active
                    h = size(Q,1)
                    # Deliberately use the full matrix contraction, without
                    # packed-index helpers or compiled sparse coefficients.
                    for b in 1:h, a in 1:h
                        term = Q[a,k]*Y[(r-1)*h+a,(s-1)*h+b]*Q[b,k]
                        value += term
                        scale += abs(term)
                    end
                end
                push!(residual,_owned(value)); push!(work,_owned(scale))
            end
        end
        gram_index == length(grams)+1 || throw(DimensionMismatch("extra Gram matrices"))
        max_absolute = zero(T)
        max_relative = zero(T)
        for (r,w) in zip(residual,work)
            if !(isfinite(r) && isfinite(w))
                max_absolute = max_relative = T(Inf)
                break
            end
            max_absolute = max(max_absolute,abs(r))
            relative = iszero(w) ? (iszero(r) ? zero(T) : T(Inf)) : abs(r)/w
            max_relative = max(max_relative,relative)
        end
        return (residual=residual,work=work,max_absolute=max_absolute,max_relative=max_relative)
    end
end

end # module SDPBSampledInput
