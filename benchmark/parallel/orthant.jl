#!/usr/bin/env julia
using SDPX, SparseArrays, LinearAlgebra, SHA, JSON

# One process / width, one cold plus three warm fresh public solves.
# Required identities point to a reviewed, immutable source and library.
required(k) = isempty(get(ENV,k,"")) ? error("set $k explicitly") : ENV[k]
const source = realpath(required("SDPX_EXPECTED_SOURCE"))
realpath(dirname(dirname(pathof(SDPX)))) == source || error("SDPX source mismatch")
const library = realpath(required("SDPX_LIBRARY"))
const identity = required("SDPX_SOURCE_ID")
const output = required("SDPX_OUTPUT")
ispath(output) && error("refusing to overwrite $output")
const bits = parse(Int,get(ENV,"SDPX_BITS","53"))
const width = parse(Int,get(ENV,"SDPX_THREADS","1"))
const n = parse(Int,get(ENV,"SDPX_N","10000"))
const repeats = parse(Int,get(ENV,"SDPX_ROWS_PER_VAR","16"))
bits in (53,256,512) || error("SDPX_BITS must be 53, 256 or 512")
width in (1,2,4,8) || error("SDPX_THREADS must be 1, 2, 4 or 8")
n > 0 && repeats > 0 || error("dimensions must be positive")
const m = Base.checked_mul(n,repeats)
BLAS.set_num_threads(1)
filehash(p) = open(sha256,p) |> bytes2hex
# Stringify arithmetic receipts: never narrow BigFloat to Float64 for validation.
safe(x::AbstractFloat) = string(x)
safe(x::Symbol) = string(x)
safe(x::Nothing) = nothing
safe(x::Union{Bool,Integer,AbstractString}) = x
safe(x::NamedTuple) = Dict(string(k)=>safe(v) for (k,v) in pairs(x))
safe(x::AbstractDict) = Dict(string(k)=>safe(v) for (k,v) in x)
safe(x::AbstractVector) = safe.(x)

function audit(r,q,A,b,tol)
    x,z,s = SDPX.value(r),SDPX.dual(r),SDPX.dual_slack(r)
    finite = all(isfinite,x) && all(isfinite,z) && all(isfinite,s)
    slack = b-A*x
    pn = one(tol)+norm(b,Inf); dn = one(tol)+norm(q,Inf)
    primal = norm(slack-s,Inf)/pn
    primal_cone = max(zero(tol),-minimum(slack))/pn
    slack_cone = max(zero(tol),-minimum(s))/pn
    dual = norm(A'*z+q,Inf)/dn
    dual_cone = max(zero(tol),-minimum(z))/dn
    pobj = dot(q,x); dobj = -dot(b,z)
    gap = abs(pobj-dobj)/(one(tol)+abs(pobj))
    objective_error = abs(pobj-length(q))/(one(tol)+length(q))
    x_error = norm(x.-one(tol),Inf)
    reported_objective_error = max(abs(SDPX.primal_objective(r)-pobj),
        abs(SDPX.dual_objective(r)-dobj))/(one(tol)+abs(pobj))
    pass = SDPX.is_optimal(r) && finite &&
        all(v->v<=tol,(primal,primal_cone,slack_cone,dual,dual_cone,gap,
            objective_error,reported_objective_error))
    (;pass,finite,status=string(SDPX.status(r)),tol,primal,primal_cone,slack_cone,
        dual,dual_cone,gap,objective_error,x_error,reported_objective_error,pobj,dobj)
end

function run(T)
    # Rows (j-1)*repeats+k impose x[j] >= k/repeats; max bound=1.
    q = ones(T,n)
    b = [-T(k)/T(repeats) for j in 1:n for k in 1:repeats]
    A = sparse(collect(1:m),repeat(collect(1:n),inner=repeats),fill(-one(T),m),m,n)
    cones = [SDPX.NonnegativeConeT(m)]
    settings = SDPX.Settings(T;precision_bits=bits,max_threads=width,
        presolve_enable=false,kkt_form=:augmented)
    tol = T===Float64 ? T(1e-8) : sqrt(eps(T))
    config = Dict(string(k)=>safe(getfield(settings,k)) for k in fieldnames(typeof(settings)) if k != :outputs)
    config["outputs"] = string(settings.outputs)
    numerical = copy(config); delete!(numerical,"max_threads")
    # Stable ordered key/value hash; input hash includes all numerical CSC bytes as text.
    hashconfig(d) = bytes2hex(sha256(join([k*"="*string(d[k]) for k in sort(collect(keys(d)))],"\n")))
    input_io = IOBuffer()
    for a in (q,A.colptr,A.rowval,A.nzval,b)
        println(input_io,length(a)); foreach(v->println(input_io,v),a)
    end
    metadata = Dict("kind"=>"targeted synthetic single-orthant diagnostic",
        "source"=>source,"source_id"=>identity,"library"=>library,
        "library_sha256"=>filehash(library),"driver_sha256"=>filehash(@__FILE__),
        "project"=>Base.active_project(),"julia"=>string(VERSION),
        "sdpx_version"=>string(Base.pkgversion(SDPX)),"json_version"=>string(Base.pkgversion(JSON)),
        "manifest_sha256"=>filehash(joinpath(dirname(Base.active_project()),"Manifest.toml")),
        "input_sha256"=>bytes2hex(sha256(take!(input_io))),"input_formula"=>"q[j]=1; A[(j-1)*r+k,j]=-1; b=-k/r; x*=1",
        "n"=>n,"m"=>m,"rows_per_variable"=>repeats,"precision_bits"=>bits,
        "requested_threads"=>width,"julia_threads"=>Threads.nthreads(),
        "rayon_num_threads"=>get(ENV,"RAYON_NUM_THREADS","unset"),
        "blas_threads"=>BLAS.get_num_threads(),"blas_config"=>string(BLAS.get_config()),
        "settings"=>config,"config_sha256"=>hashconfig(config),
        "numerical_config_sha256"=>hashconfig(numerical),"external_tolerance"=>string(tol))
    records = Any[]
    for i in 1:4
        try
            timed = @timed SDPX.solve_conic(q,A,b,cones;settings=settings,return_result=true)
            r = timed.value
            validation = audit(r,q,A,b,tol) # Always outside timer.
            receipt_ok = r.info.factorization==:qdldl && r.info.kkt_form==:augmented &&
                r.info.precision_bits==bits && r.info.backend_threads==1 &&
                1<=r.info.cone_threads<=width && !r.info.presolve_enable
            push!(records,Dict("run"=>i,"phase"=>i==1 ? "cold" : "warm",
                "pass"=>validation.pass && receipt_ok,"validation"=>safe(validation),
                "receipt_ok"=>receipt_ok,"execution"=>safe(r.info),
                "iterations"=>SDPX.iterations(r),"frontend_s"=>timed.time,
                "native_solve_s"=>SDPX.solve_time(r),"native_setup_s"=>nothing,
                "julia_allocated_bytes"=>timed.bytes,"julia_gc_s"=>timed.gctime))
        catch err
            push!(records,Dict("run"=>i,"phase"=>i==1 ? "cold" : "warm",
                "pass"=>false,"error"=>sprint(showerror,err)))
        end
    end
    passed = all(r->r["pass"],records)
    metadata["runs"] = records; metadata["pass"] = passed
    # No timing credit from partial/failed runs.
    metadata["warm_frontend_median_s"] = passed ? sort([r["frontend_s"] for r in records[2:4]])[2] : nothing
    metadata["warm_native_solve_median_s"] = passed ? sort([r["native_solve_s"] for r in records[2:4]])[2] : nothing
    io = Base.Filesystem.open(output, Base.JL_O_WRONLY | Base.JL_O_CREAT | Base.JL_O_EXCL, 0o600)
    try
        write(io,JSON.json(metadata,2)*"\n")
    finally
        close(io)
    end
    passed || exit(1)
end
bits==53 ? run(Float64) : setprecision(()->run(BigFloat),BigFloat,bits)
