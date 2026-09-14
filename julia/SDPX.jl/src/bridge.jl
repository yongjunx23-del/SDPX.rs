# Layouts mirror include/sdpx.h ABI 3. No Julia solver engine is loaded.
const _ABI_VERSION=UInt32(3)
struct CScalars
    kind::UInt32
    reserved::UInt32
    count::UInt64
    f64::Ptr{Cdouble}
    decimal::Ptr{Ptr{Cchar}}
end
struct CCsc
    rows::UInt64
    cols::UInt64
    nnz::UInt64
    colptr::Ptr{UInt64}
    rowval::Ptr{UInt64}
    values::CScalars
end
struct CCone
    kind::UInt32
    reserved::UInt32
    dim::UInt64
    alpha::CScalars
end
struct CSampledBlock
    row_start::UInt64
    column_start::UInt64
    dim::UInt64
    basis_rows::UInt64
    basis_cols::UInt64
    basis::CScalars
    weights::CScalars
end
struct CSettings
    abi_version::UInt32
    struct_size::UInt32
    precision_bits::UInt32
    max_iter::UInt32
    verbose::UInt32
    preprocessing_flags::UInt32
    max_threads::UInt32
    kkt_form::UInt32
    time_limit::Cdouble
    tol_gap_abs::Ptr{Cchar}
    tol_gap_rel::Ptr{Cchar}
    tol_feas::Ptr{Cchar}
    tol_infeas_abs::Ptr{Cchar}
    tol_infeas_rel::Ptr{Cchar}
end
struct CInfo
    abi_version::UInt32
    struct_size::UInt32
    status::UInt32
    iterations::UInt32
    working_bits::UInt32
    backend_threads::UInt32
    cone_threads::UInt32
    kkt_form::UInt32
    n::UInt64
    m::UInt64
    solve_time::Cdouble
    objective::Cdouble
    dual_objective::Cdouble
    primal_residual::Cdouble
    dual_residual::Cdouble
    gap_abs::Cdouble
    gap_rel::Cdouble
end
const _library_lock=ReentrantLock()
const _library=Ref{Ptr{Cvoid}}(C_NULL)
function _library_path()
    haskey(ENV,"SDPX_LIBRARY") && return abspath(ENV["SDPX_LIBRARY"])
    root=normpath(joinpath(@__DIR__,"..","..",".."))
    name=Sys.iswindows() ? "sdpx.dll" : "libsdpx." * Libdl.dlext
    target=get(ENV,"CARGO_TARGET_DIR",joinpath(root,"target"))
    isabspath(target) || (target=joinpath(root,target))
    for profile in ("release","debug")
        candidate=joinpath(target,profile,name)
        isfile(candidate) && return candidate
    end
    error("SDPX shared library not found; run Pkg.build(\"SDPX\") or set SDPX_LIBRARY")
end
function _check_abi(version,size,expected_size)
    version==_ABI_VERSION || error("SDPX ABI version mismatch: library returned ABI $version, frontend requires ABI $_ABI_VERSION")
    size==expected_size || error("SDPX ABI structure size mismatch: expected $expected_size, got $size")
    nothing
end
function _sym(name::Symbol)
    lock(_library_lock) do
        if _library[]==C_NULL
            library=Libdl.dlopen(_library_path())
            try
                settings=Ref{CSettings}()
                code=ccall(Libdl.dlsym(library,:sdpx_default_settings),Cint,(Ref{CSettings},),settings)
                code==0 || error("SDPX ABI default-settings query failed with code $code")
                _check_abi(settings[].abi_version,settings[].struct_size,sizeof(CSettings))
                _library[]=library
            catch
                Libdl.dlclose(library)
                rethrow()
            end
        end
        Libdl.dlsym(_library[],name)
    end
end
function _abi_error(code)
    code==0 && return nothing
    n=ccall(_sym(:sdpx_last_error),UInt64,(Ptr{UInt8},UInt64),C_NULL,0)
    buffer=Vector{UInt8}(undef,max(1,Int(n)))
    ccall(_sym(:sdpx_last_error),UInt64,(Ptr{UInt8},UInt64),buffer,length(buffer))
    msg=String(buffer[1:findfirst(iszero,buffer)-1])
    error("SDPX ABI error $code: $msg")
end
function _scalar_storage(values::Vector{Float64})
    (descriptor=CScalars(0,0,length(values),pointer(values),C_NULL), values=values)
end
function _scalar_storage(values::Vector{BigFloat})
    strings=string.(values)
    pointers=Ptr{Cchar}[pointer(x) for x in strings]
    (descriptor=CScalars(1,0,length(values),C_NULL,pointer(pointers)),values=values,strings=strings,pointers=pointers)
end
function _csc_storage(matrix::SparseMatrixCSC)
    colptr=UInt64.(matrix.colptr .- 1)
    rowval=UInt64.(matrix.rowval .- 1)
    values=_scalar_storage(matrix.nzval)
    descriptor=CCsc(size(matrix,1),size(matrix,2),nnz(matrix),pointer(colptr),pointer(rowval),values.descriptor)
    (descriptor=descriptor,colptr=colptr,rowval=rowval,values=values)
end
function _cone_storage(cones,::Type{T},bits) where T
    anchors=[]
    descriptors=CCone[]
    for cone in cones
        kind,dim,alpha=if cone isa ZeroConeT
            (0,cone.dim,T[])
        elseif cone isa NonnegativeConeT
            (1,cone.dim,T[])
        elseif cone isa SecondOrderConeT
            (2,cone.dim,T[])
        elseif cone isa PSDTriangleConeT
            (3,cone.dim,T[])
        elseif cone isa ExponentialConeT
            (4,3,T[])
        elseif cone isa PowerConeT
            (5,3,owned_vector_copy(T,[cone.α];precision_bits=bits))
        elseif cone isa GenPowerConeT
            (6,cone.dim2,owned_vector_copy(T,cone.α;precision_bits=bits))
        else
            throw(ArgumentError("unsupported cone $(typeof(cone))"))
        end
        dim>=0 || throw(ArgumentError("cone dimension must be nonnegative"))
        storage=_scalar_storage(alpha)
        push!(anchors,storage)
        push!(descriptors,CCone(kind,0,dim,storage.descriptor))
    end
    (descriptors=descriptors,anchors=anchors)
end
function _settings_storage(s::Settings)
    _validate_settings(s)
    strings=[v===nothing ? nothing : string(v) for v in
        (s.tol_gap_abs,s.tol_gap_rel,s.tol_feas,s.tol_infeas_abs,s.tol_infeas_rel)]
    pointers=Ptr{Cchar}[x===nothing ? C_NULL : pointer(x) for x in strings]
    form=s.kkt_form===:auto ? 0 : s.kkt_form===:augmented ? 1 : 2
    preprocessing=UInt32(s.equilibration===:ruiz) |
        (UInt32(s.presolve_enable)<<1) | (UInt32(s.chordal_decomposition_enable)<<2)
    descriptor=CSettings(_ABI_VERSION,sizeof(CSettings),s.precision_bits,s.max_iter,s.verbose,
        preprocessing,s.max_threads,form,s.time_limit,pointers...)
    (descriptor=descriptor,strings=strings,pointers=pointers)
end
mutable struct PreparedCore{T}
    pointer::Ptr{Cvoid}
    lock::ReentrantLock
    n::Int
    m::Int
    precision_bits::Int
    settings::Settings{T}
end
function Base.close(h::PreparedCore)
    lock(h.lock) do
        h.pointer==C_NULL && return nothing
        ptr=h.pointer
        # Consume ownership exactly once, including if destruction reports an error.
        h.pointer=C_NULL
        _abi_error(ccall(_sym(:sdpx_destroy),Cint,(Ptr{Cvoid},),ptr))
        nothing
    end
end
Base.isopen(h::PreparedCore)=lock(()->h.pointer!=C_NULL,h.lock)
_require_open(h)=h.pointer==C_NULL ? throw(ArgumentError("prepared SDPX handle is closed")) : nothing
function _prepare_program(p;settings=nothing,reusable=true)
    T=eltype(p.q)
    _check_precision(T,p.precision_bits)
    s=settings===nothing ? Settings(T;precision_bits=p.precision_bits) : deepcopy(settings)
    s isa Settings{T} && s.precision_bits==p.precision_bits || throw(ArgumentError("settings arithmetic/precision mismatch"))
    _validate_settings(s)
    # Reusable handles preserve the original matrix structure for q/b updates.
    # Direct solves use the requested preprocessing, including all defaults.
    if reusable
        s.presolve_enable=false
        s.chordal_decomposition_enable=false
    end
    size(p.P)==(length(p.q),length(p.q)) && size(p.A)==(length(p.b),length(p.q)) || throw(DimensionMismatch("inconsistent conic dimensions"))
    all(isfinite,p.q) && all(isfinite,p.b) && all(isfinite,p.P.nzval) && all(isfinite,p.A.nzval) || throw(ArgumentError("problem data must be finite"))
    P=_csc_storage(triu(p.P)); q=_scalar_storage(p.q)
    A=_csc_storage(p.A); b=_scalar_storage(p.b)
    cones=_cone_storage(p.cones,T,p.precision_bits); options=_settings_storage(s)
    out=Ref{Ptr{Cvoid}}(C_NULL)
    blocks=_program_sampled_storage(p)
    GC.@preserve P q A b cones options blocks begin
        if blocks===nothing
            _abi_error(ccall(_sym(:sdpx_prepare),Cint,
                (Ref{CCsc},Ref{CScalars},Ref{CCsc},Ref{CScalars},Ptr{CCone},UInt64,Ref{CSettings},Ref{Ptr{Cvoid}}),
                P.descriptor,q.descriptor,A.descriptor,b.descriptor,cones.descriptors,length(cones.descriptors),options.descriptor,out))
        else
            _abi_error(ccall(_sym(:sdpx_prepare_sampled),Cint,
                (Ref{CCsc},Ref{CScalars},Ref{CCsc},Ref{CScalars},Ptr{CCone},UInt64,Ref{CSettings},Ptr{CSampledBlock},UInt64,Ref{Ptr{Cvoid}}),
                P.descriptor,q.descriptor,A.descriptor,b.descriptor,cones.descriptors,length(cones.descriptors),options.descriptor,
                blocks.descriptors,length(blocks.descriptors),out))
        end
    end
    out[]!=C_NULL || error("SDPX returned a null prepared handle")
    h=PreparedCore(out[],ReentrantLock(),length(p.q),length(p.b),p.precision_bits,s)
    finalizer(h) do handle
        try close(handle) catch end
    end
    h
end
const _ABI_STATUSES=(NotStarted,Optimal,PrimalInfeasible,DualInfeasible,AlmostOptimal,
    Stalled,Stalled,IterLimit,TimeLimit,NumericalFailure,Stalled,UserStopped)
function _solver_name(h::PreparedCore)
    required=Ref{UInt64}(0)
    _abi_error(ccall(_sym(:sdpx_get_solver_name),Cint,
        (Ptr{Cvoid},Ptr{UInt8},UInt64,Ref{UInt64}),h.pointer,C_NULL,0,required))
    1<=required[]<=typemax(Int) || error("SDPX solver name size is invalid")
    bytes=Vector{UInt8}(undef,Int(required[]))
    _abi_error(ccall(_sym(:sdpx_get_solver_name),Cint,
        (Ptr{Cvoid},Ptr{UInt8},UInt64,Ref{UInt64}),h.pointer,bytes,length(bytes),required))
    required[]==length(bytes) && last(bytes)==0 || error("SDPX solver name is not NUL terminated")
    Symbol(String(bytes[1:end-1]))
end
function _copy_result(h::PreparedCore{T}) where T
    info=Ref{CInfo}()
    _abi_error(ccall(_sym(:sdpx_get_info),Cint,(Ptr{Cvoid},Ref{CInfo}),h.pointer,info))
    i=info[]
    _check_abi(i.abi_version,i.struct_size,sizeof(CInfo))
    Int(i.working_bits)==h.precision_bits || error("SDPX returned unexpected working precision")
    Int(i.n)==h.n && Int(i.m)==h.m || error("SDPX result dimensions changed")
    count=h.n+2h.m+6
    required=Ref{UInt64}(0)
    vals=if T===Float64
        data=Vector{Float64}(undef,count)
        _abi_error(ccall(_sym(:sdpx_result_f64),Cint,(Ptr{Cvoid},Ptr{Cdouble},UInt64,Ref{UInt64}),h.pointer,data,count,required))
        required[]==count || error("SDPX result size mismatch")
        data
    else
        _abi_error(ccall(_sym(:sdpx_result_decimal),Cint,(Ptr{Cvoid},Ptr{UInt8},UInt64,Ref{UInt64}),h.pointer,C_NULL,0,required))
        data=Vector{UInt8}(undef,Int(required[]))
        _abi_error(ccall(_sym(:sdpx_result_decimal),Cint,(Ptr{Cvoid},Ptr{UInt8},UInt64,Ref{UInt64}),h.pointer,data,length(data),required))
        strings=split(String(data),'\0';keepempty=true)
        length(strings)==count+1 && isempty(last(strings)) || error("SDPX decimal result size mismatch")
        [parse(BigFloat,x;precision=h.precision_bits) for x in strings[1:end-1]]
    end
    n=h.n; m=h.m; offset=n+2m
    metrics=vals[offset+1:end]
    state=Int(i.status)<length(_ABI_STATUSES) ? _ABI_STATUSES[Int(i.status)+1] : NumericalFailure
    i.kkt_form in (1,2) || error("SDPX returned an unknown KKT form")
    metadata=(core_status=Symbol("rust_status_" * string(i.status)),kkt_form=i.kkt_form==1 ? :augmented : :condensed,provider=T===Float64 ? :rust : :mpfr,
        factorization=_solver_name(h),precision_bits=h.precision_bits,backend_threads=Int(i.backend_threads),
        cone_threads=Int(i.cone_threads),diagnostics_level=h.settings.outputs.diagnostics,
        equilibration=h.settings.equilibration,presolve_enable=h.settings.presolve_enable,
        chordal_decomposition_enable=h.settings.chordal_decomposition_enable)
    RawResult(state,vals[1:n],vals[n+1:n+m],vals[n+m+1:offset],metrics[1],metrics[2],metrics[3],metrics[4],metrics[6],Int(i.iterations),i.solve_time,metadata)
end
function _solve_prepared_program!(h::PreparedCore{T},p;settings=nothing) where T
    lock(h.lock) do
        _require_open(h)
        length(p.q)==h.n && length(p.b)==h.m || throw(DimensionMismatch("prepared q/b dimensions are fixed"))
        qvalues=owned_vector_copy(T,p.q;precision_bits=h.precision_bits)
        bvalues=owned_vector_copy(T,p.b;precision_bits=h.precision_bits)
        all(isfinite,qvalues) && all(isfinite,bvalues) || throw(ArgumentError("updated data must be finite"))
        q=_scalar_storage(qvalues); b=_scalar_storage(bvalues)
        GC.@preserve q b begin
            _abi_error(ccall(_sym(:sdpx_update),Cint,(Ptr{Cvoid},Ref{CScalars},Ref{CScalars}),h.pointer,q.descriptor,b.descriptor))
        end
        _abi_error(ccall(_sym(:sdpx_solve),Cint,(Ptr{Cvoid},),h.pointer))
        _copy_result(h)
    end
end
function _solve_program(p;settings=nothing,recompile=nothing)
    h=_prepare_program(p;settings,reusable=false)
    try
        lock(h.lock) do
            _abi_error(ccall(_sym(:sdpx_solve),Cint,(Ptr{Cvoid},),h.pointer))
            _copy_result(h)
        end
    finally
        close(h)
    end
end
struct StandardProgram{T<:AbstractFloat}
    P::SparseMatrixCSC{T,Int}
    q::Vector{T}
    A::SparseMatrixCSC{T,Int}
    b::Vector{T}
    cones::Vector{SupportedCone}
    precision_bits::Int
end
function _working_copy(::Type{T},a::AbstractVector,bits) where T
    owned_vector_copy(T,a;precision_bits=bits)
end
function _working_copy(::Type{T},a::AbstractMatrix,bits) where T
    mat=sparse(a)
    SparseMatrixCSC(size(mat)...,Int.(mat.colptr),Int.(mat.rowval),_working_copy(T,mat.nzval,bits))
end
function _standard_program(P,q,A,b,cones,::Type{T},bits) where T
    _check_precision(T,bits)
    _owned_arithmetic_scope(T,bits) do
        StandardProgram(_working_copy(T,P,bits),_working_copy(T,q,bits),_working_copy(T,A,bits),
            _working_copy(T,b,bits),SupportedCone[deepcopy(c) for c in cones],bits)
    end
end
function _direct_arithmetic(settings,T,arrays...)
    arithmetic=T===nothing ? (settings===nothing ? promote_type(eltype.(arrays)...) : typeof(settings).parameters[1]) : T
    arithmetic<:AbstractFloat || (arithmetic=Float64)
    bits=if settings!==nothing
        settings.precision_bits
    elseif arithmetic===BigFloat
        max(precision(BigFloat),maximum(a->maximum(x->x isa BigFloat ? precision(x) : 0,a;init=0),arrays;init=0))
    else
        precision(arithmetic)
    end
    _check_precision(arithmetic,bits)
    arithmetic,bits
end
function solve(P::AbstractMatrix,q::AbstractVector,A::AbstractMatrix,b::AbstractVector,cones;settings=nothing,T=nothing)
    arithmetic,bits=_direct_arithmetic(settings,T,P,q,A,b)
    _solve_program(_standard_program(P,q,A,b,cones,arithmetic,bits);settings)
end
function solve(q::AbstractVector,A::AbstractMatrix,b::AbstractVector,cones;settings=nothing,T=nothing)
    arithmetic,bits=_direct_arithmetic(settings,T,q,A,b)
    _solve_program(_standard_program(spzeros(arithmetic,length(q),length(q)),q,A,b,cones,arithmetic,bits);settings)
end
function solve_conic(q,A,b,cones;return_result=false,kwargs...)
    r=solve(q,A,b,cones;kwargs...)
    return_result ? r : (status(r),value(r),primal_objective(r))
end
function solve_conic(P,q,A,b,cones;return_result=false,kwargs...)
    r=solve(P,q,A,b,cones;kwargs...)
    return_result ? r : (status(r),value(r),primal_objective(r))
end
_program_sampled_storage(p)=nothing
mutable struct PreparedProblem{T,P}
    program::P
    handle::PreparedCore{T}
end
function prepare(P::AbstractMatrix,q::AbstractVector,A::AbstractMatrix,b::AbstractVector,cones;settings=nothing,T=nothing)
    arithmetic,bits=_direct_arithmetic(settings,T,P,q,A,b)
    p=_standard_program(P,q,A,b,cones,arithmetic,bits)
    PreparedProblem(p,_prepare_program(p;settings))
end
function prepare(q::AbstractVector,A::AbstractMatrix,b::AbstractVector,cones;settings=nothing,T=nothing)
    arithmetic,bits=_direct_arithmetic(settings,T,q,A,b)
    p=_standard_program(spzeros(arithmetic,length(q),length(q)),q,A,b,cones,arithmetic,bits)
    PreparedProblem(p,_prepare_program(p;settings))
end
_updated_program(p::StandardProgram,q,b)=StandardProgram(p.P,q,p.A,b,p.cones,p.precision_bits)
function solve!(p::PreparedProblem{T};q=nothing,b=nothing) where T
    lock(p.handle.lock) do
        _require_open(p.handle)
        old=p.program
        updated=_updated_program(old,q===nothing ? old.q : _working_copy(T,q,old.precision_bits),
            b===nothing ? old.b : _working_copy(T,b,old.precision_bits))
        r=_solve_prepared_program!(p.handle,updated)
        p.program=updated
        r
    end
end
Base.close(p::PreparedProblem)=close(p.handle)
Base.isopen(p::PreparedProblem)=isopen(p.handle)
