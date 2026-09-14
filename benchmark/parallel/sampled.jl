#!/usr/bin/env julia
# Predeclared dense sampled PSD diagnostic; no fixture selection from results.
using SDPX, SparseArrays, LinearAlgebra, SHA, JSON
required(k)=isempty(get(ENV,k,"")) ? error("set $k explicitly") : ENV[k]
const source=realpath(required("SDPX_EXPECTED_SOURCE"))
realpath(dirname(dirname(pathof(SDPX))))==source || error("SDPX source mismatch")
const library=realpath(required("SDPX_LIBRARY"))
const identity=required("SDPX_SOURCE_ID")
const output=required("SDPX_OUTPUT")
ispath(output) && error("refusing to overwrite $output")
const bits=parse(Int,get(ENV,"SDPX_BITS","53"))
const width=parse(Int,get(ENV,"SDPX_THREADS","1"))
bits in (53,256,512) || error("SDPX_BITS must be 53, 256 or 512")
width in (1,2,4,8) || error("SDPX_THREADS must be 1, 2, 4 or 8")
const n=parse(Int,get(ENV,"SDPX_N",bits==53 ? "64" : "24"))
n>=2 || error("SDPX_N must be at least 2")
const m=div(Base.checked_mul(n,Base.checked_add(n,1)),2)
if bits!=53
    # Optional benchmark dependency, used only by the external PSD audit.
    import GenericLinearAlgebra
end
BLAS.set_num_threads(1)
filehash(p)=open(sha256,p) |> bytes2hex
safe(x::AbstractFloat)=string(x)
safe(x::Symbol)=string(x)
safe(::Nothing)=nothing
safe(x::Union{Bool,Integer,AbstractString})=x
safe(x::NamedTuple)=Dict(string(k)=>safe(v) for (k,v) in pairs(x))
safe(x::AbstractDict)=Dict(string(k)=>safe(v) for (k,v) in x)
safe(x::AbstractVector)=safe.(x)
hashconfig(d)=bytes2hex(sha256(join([k*"="*string(d[k]) for k in sort(collect(keys(d)))],"\n")))

function unpack(v,root2)
    M=Matrix{eltype(v)}(undef,n,n); p=0
    for j in 1:n, i in 1:j
        p+=1
        value=i==j ? v[p] : v[p]/root2
        M[i,j]=deepcopy(value); M[j,i]=deepcopy(value)
    end
    M
end
function psd_distance(M)
    # Full symmetric eigensolver, independent of native cone/factor kernels.
    max(zero(eltype(M)),-minimum(eigvals(Symmetric(M))))/(one(eltype(M))+opnorm(M,Inf))
end
function audit(r,q,A,b,tol,root2)
    x,z,s=SDPX.value(r),SDPX.dual(r),SDPX.dual_slack(r)
    finite=all(isfinite,x) && all(isfinite,z) && all(isfinite,s)
    finite || return (;pass=false,finite,status=string(SDPX.status(r)))
    affine_slack=b-A*x
    primal=norm(affine_slack-s,Inf)/(one(tol)+norm(b,Inf))
    dual=norm(q+A'*z,Inf)/(one(tol)+norm(q,Inf))
    primal_psd=psd_distance(unpack(affine_slack,root2))
    slack_psd=psd_distance(unpack(s,root2))
    dual_psd=psd_distance(unpack(z,root2))
    pobj=dot(q,x); dobj=-dot(b,z)
    gap=abs(pobj-dobj)/(one(tol)+abs(pobj))
    objective_error=abs(pobj-n)/(one(tol)+n)
    x_error=norm(x.-one(tol),Inf)
    reported_objective_error=max(abs(SDPX.primal_objective(r)-pobj),
        abs(SDPX.dual_objective(r)-dobj))/(one(tol)+abs(pobj))
    residuals=(primal,dual,primal_psd,slack_psd,dual_psd,gap,objective_error,reported_objective_error)
    pass=SDPX.is_optimal(r) && all(v->isfinite(v) && v<=tol,residuals)
    (;pass,finite,status=string(SDPX.status(r)),tol,primal,dual,primal_psd,
        slack_psd,dual_psd,gap,objective_error,x_error,reported_objective_error,pobj,dobj)
end
function fresh_solve(Q,weights,q,linear,b,settings)
    block=SDPX.SampledBlock(1,1,1,Q,weights)
    program=SDPX.sampled_program(q,linear,b,[SDPX.PSDTriangleConeT(n)],[block];settings)
    SDPX.solve_conic(program;settings,return_result=true)
end
function run(T)
    rho=T(1)/T(4); offset=rho/T(n)
    Q=[(i==j ? one(T) : zero(T))+offset for i in 1:n,j in 1:n]
    q=ones(T,n); weights=[-one(T) for _ in 1:n]
    b=[i==j ? -one(T) : zero(T) for j in 1:n for i in 1:j]
    linear=spzeros(T,m,n)
    root2=sqrt(T(2))
    # Materialize solely for external validation, using original factors. This
    # dense operator is never passed to the sampled solver or built in timing.
    A=Matrix{T}(undef,m,n)
    for k in 1:n
        p=0
        for j in 1:n, i in 1:j
            p+=1
            A[p,k]=-Q[i,k]*Q[j,k]*(i==j ? one(T) : root2)
        end
    end
    settings=SDPX.Settings(T;precision_bits=bits,max_threads=width,kkt_form=:condensed)
    tol=T===Float64 ? T(1e-8) : sqrt(eps(T))
    config=Dict(string(k)=>safe(getfield(settings,k)) for k in fieldnames(typeof(settings)) if k!=:outputs)
    config["outputs"]=string(settings.outputs)
    numerical=copy(config); delete!(numerical,"max_threads")
    io=IOBuffer(); println(io,"single PSD; dim=1; row_start=1; column_start=1; n=$n; m=$m; Alinear=0")
    for array in (vec(Q),weights,q,b)
        println(io,length(array)); foreach(v->println(io,v),array)
    end
    manifest=joinpath(dirname(Base.active_project()),"Manifest.toml")
    cargo_lock=joinpath(dirname(dirname(source)),"Cargo.lock")
    library_hash=filehash(library); driver_hash=filehash(@__FILE__); manifest_hash=filehash(manifest)
    metadata=Dict("kind"=>"predeclared synthetic dominant sampled PSD diagnostic",
        "source"=>source,"source_id"=>identity,"library"=>library,"library_sha256"=>library_hash,
        "driver_sha256"=>driver_hash,"project"=>Base.active_project(),"manifest_sha256"=>manifest_hash,
        "cargo_lock_sha256"=>(isfile(cargo_lock) ? filehash(cargo_lock) : nothing),
        "julia"=>string(VERSION),"julia_mpfr_version"=>string(Base.MPFR.version()),
        "sdpx_version"=>string(Base.pkgversion(SDPX)),"json_version"=>string(Base.pkgversion(JSON)),
        "audit_eigensolver"=>(T===Float64 ? "LinearAlgebra/LAPACK" : "GenericLinearAlgebra"),
        "genericlinearalgebra_version"=>(T===Float64 ? nothing : string(Base.pkgversion(GenericLinearAlgebra))),
        "input_sha256"=>bytes2hex(sha256(take!(io))),
        "input_formula"=>"Q=I+(1/4)*ones(n,n)/n; weights=-1; q=ones; b=-svec(I)",
        "known_objective"=>string(T(n)),"rho"=>string(rho),"basis_offset"=>string(offset),
        "n"=>n,"m"=>m,"psd_order"=>n,"precision_bits"=>bits,"requested_threads"=>width,
        "julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),
        "blas_config"=>string(BLAS.get_config()),"rayon_num_threads"=>get(ENV,"RAYON_NUM_THREADS","unset"),
        "veclib_maximum_threads"=>get(ENV,"VECLIB_MAXIMUM_THREADS","unset"),
        "settings"=>config,"config_sha256"=>hashconfig(config),
        "numerical_config_sha256"=>hashconfig(numerical),"external_tolerance"=>string(tol),
        "input_route"=>"sampled_factors","api_timing_scope"=>"fresh factor adaptation + public solve",
        "rss_source"=>"Sys.maxrss bytes; cumulative process high-water mark including prior audits",
        "runs"=>Any[])
    for i in 1:4
        record=Dict{String,Any}("run"=>i,"phase"=>(i==1 ? "cold" : "warm"),"pass"=>false)
        try
            record["max_rss_bytes_before_solve"]=Sys.maxrss()
            timed=@timed fresh_solve(Q,weights,q,linear,b,settings)
            r=timed.value
            record["max_rss_bytes_after_solve"]=Sys.maxrss()
            record["frontend_s"]=timed.time
            record["native_solve_s"]=SDPX.solve_time(r)
            record["native_setup_s"]=nothing
            record["julia_allocated_bytes"]=timed.bytes
            record["julia_gc_s"]=timed.gctime
            record["iterations"]=SDPX.iterations(r)
            record["execution"]=safe(r.info)
            validation=audit(r,q,A,b,tol,root2) # Outside API and native timers.
            record["validation"]=safe(validation)
            receipt_ok=r.info.factorization==:condensed_sampled_qdldl &&
                r.info.kkt_form==:condensed && r.info.precision_bits==bits &&
                r.info.backend_threads==1 && r.info.cone_threads==width && BLAS.get_num_threads()==1
            timings_ok=all(v->isfinite(v) && v>=0,(timed.time,timed.gctime,SDPX.solve_time(r)))
            record["receipt_ok"]=receipt_ok
            record["pass"]=validation.pass && receipt_ok && timings_ok
        catch err
            record["error"]=sprint(showerror,err)
        end
        record["max_rss_bytes_after_audit"]=Sys.maxrss()
        push!(metadata["runs"],record)
    end
    unchanged=filehash(library)==library_hash && filehash(@__FILE__)==driver_hash && filehash(manifest)==manifest_hash
    passed=unchanged && all(r->r["pass"],metadata["runs"])
    metadata["artifacts_unchanged"]=unchanged; metadata["pass"]=passed
    metadata["max_rss_bytes"]=Sys.maxrss()
    warm=metadata["runs"][2:4]
    metadata["warm_frontend_median_s"]=passed ? sort([r["frontend_s"] for r in warm])[2] : nothing
    metadata["warm_native_solve_median_s"]=passed ? sort([r["native_solve_s"] for r in warm])[2] : nothing
    stream=Base.Filesystem.open(output,Base.JL_O_WRONLY|Base.JL_O_CREAT|Base.JL_O_EXCL,0o600)
    try
        write(stream,JSON.json(metadata,2)*"\n")
    finally
        close(stream)
    end
    passed || exit(1)
end
bits==53 ? run(Float64) : setprecision(()->run(BigFloat),BigFloat,bits)
