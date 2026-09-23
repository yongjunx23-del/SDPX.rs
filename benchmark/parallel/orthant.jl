#!/usr/bin/env julia
# Deterministic orthant input generator plus native CLI driver.
# Julia is used only for input construction and the independent audit; the
# numerical solve is one fresh Rust `sdpx` process per recorded sample.
using SparseArrays, LinearAlgebra, SHA, JSON

required(k) = isempty(get(ENV, k, "")) ? error("set $k explicitly") : ENV[k]
const cli = realpath(required("SDPX_CLI"))
isfile(cli) && Base.Filesystem.isexecutable(cli) || error("SDPX_CLI is not an executable file")
const identity = required("SDPX_SOURCE_ID")
const output = required("SDPX_OUTPUT")
ispath(output) && error("refusing to overwrite $output")
const bits = parse(Int, get(ENV, "SDPX_BITS", "53"))
const width = parse(Int, get(ENV, "SDPX_THREADS", "1"))
const n = parse(Int, get(ENV, "SDPX_N", "10000"))
const repeats = parse(Int, get(ENV, "SDPX_ROWS_PER_VAR", "16"))
bits in (53, 256, 512) || error("SDPX_BITS must be 53, 256 or 512")
width in (1, 2, 4, 8) || error("SDPX_THREADS must be 1, 2, 4 or 8")
n > 0 && repeats > 0 || error("dimensions must be positive")
bits == 53 || setprecision(BigFloat, bits)
const m = Base.checked_mul(n, repeats)

wire(x::Real) = bits == 53 ? Float64(x) : string(x)
wire(x) = x
filehash(path) = open(sha256, path) |> bytes2hex
safe(x::AbstractFloat) = string(x)
safe(x::Integer) = x
safe(x::Bool) = x
safe(x::AbstractString) = x
safe(x::Nothing) = nothing
safe(x::NamedTuple) = Dict(string(k) => safe(v) for (k, v) in pairs(x))
safe(x::AbstractDict) = Dict(string(k) => safe(v) for (k, v) in x)
safe(x::AbstractVector) = safe.(x)

function cone_distance(v, cones; dual=false)
    any(!isfinite, v) && return NaN
    off = 1
    worst = zero(eltype(v))
    for cone in cones
        tag, p = first(collect(cone))
        len = Int(p)
        blk = @view v[off:off + len - 1]
        d = tag == "ZeroConeT" ? (dual ? zero(eltype(v)) : maximum(abs, blk)) :
            tag == "NonnegativeConeT" ? max(zero(eltype(v)), -minimum(blk)) :
            tag == "SecondOrderConeT" ? max(zero(eltype(v)), norm(blk[2:end]) - blk[1]) :
            tag == "PSDTriangleConeT" ? error("orthant fixture has no PSD cone") : error("unsupported cone $tag")
        worst = max(worst, d)
        off += len
    end
    return worst
end

function audit(result, q, A, b, cones, tol)
    T = eltype(q)
    x = T[T(v) for v in result["x"]]
    z = T[T(v) for v in result["z"]]
    s = T[T(v) for v in result["s"]]
    finite = all(isfinite, x) && all(isfinite, z) && all(isfinite, s)
    finite || return (pass=false, finite=false, status=string(result["status"]))
    slack = b - A * x
    pn, dn = one(T) + norm(b, Inf), one(T) + norm(q, Inf)
    primal = norm(slack - s, Inf) / pn
    primal_cone = cone_distance(slack, cones) / pn
    slack_cone = cone_distance(s, cones) / pn
    dual = norm(A' * z + q, Inf) / dn
    dual_cone = cone_distance(z, cones; dual=true) / dn
    pobj, dobj = dot(q, x), -dot(b, z)
    gap = abs(pobj - dobj) / (one(T) + abs(pobj))
    objective_error = abs(pobj - T(n)) / (one(T) + T(n))
    reported_objective_error = max(abs(T(result["objective"]) - pobj),
        abs(T(result["dual_objective"]) - dobj)) / (one(T) + abs(pobj))
    status = string(result["status"])
    optimal = status in ("Solved", "Optimal", "solved", "optimal")
    values = (primal, primal_cone, slack_cone, dual, dual_cone, gap,
              objective_error, reported_objective_error)
    passed = optimal && all(v -> isfinite(v) && v <= tol, values)
    return (pass=passed, finite=true, status=status, tol=tol, primal=primal,
        primal_cone=primal_cone, slack_cone=slack_cone, dual=dual,
        dual_cone=dual_cone, gap=gap, objective_error=objective_error,
        reported_objective_error=reported_objective_error, pobj=pobj, dobj=dobj)
end

T = bits == 53 ? Float64 : BigFloat
q = T[one(T) for _ in 1:n]
b = T[-T(k) / T(repeats) for _ in 1:n for k in 1:repeats]
A = sparse(collect(1:m), repeat(collect(1:n), inner=repeats), fill(-one(T), m), m, n)
cones = [Dict("NonnegativeConeT" => m)]
problem = Dict("P" => Dict("m" => n, "n" => n, "colptr" => fill(0, n + 1),
    "rowval" => Int[], "nzval" => wire.(T[])),
    "q" => wire.(q),
    "A" => Dict("m" => m, "n" => n, "colptr" => Int[A.colptr .- 1;],
        "rowval" => Int[A.rowval .- 1;], "nzval" => wire.(T[A.nzval...])),
    "b" => wire.(b), "cones" => cones,
    "settings" => Dict("verbose" => false, "max_threads" => width,
        "presolve_enable" => false, "kkt_form" => "augmented"))
input = tempname() * ".json"
write(input, JSON.json(problem) * "\n")
input_sha = filehash(input)
tol = bits == 53 ? T(1e-8) : sqrt(eps(T))
records = Any[]
for i in 1:4
    result_path = tempname() * ".json"
    started = time()
    command = `$cli $input --precision $bits --threads $width --quiet --output $result_path`
    ok = true
    err = nothing
    try
        run(command)
    catch ex
        ok = false
        err = sprint(showerror, ex)
    end
    cli_seconds = time() - started
    result = ok && isfile(result_path) ? JSON.parsefile(result_path) : Dict{String,Any}()
    validation = if ok
        try
            audit(result, q, A, b, cones, tol)
        catch ex
            (pass=false, finite=false, status="Error", error=sprint(showerror, ex))
        end
    else
        (pass=false, finite=false, status="Error")
    end
    actual_kkt = get(result, "kkt_form", startswith(string(get(result, "linear_solver", "")), "condensed_") ? "condensed" : "augmented")
    receipt_ok = ok && get(result, "precision_bits", 0) == bits &&
        get(result, "threads_requested", 0) == width &&
        actual_kkt == "augmented" &&
        get(result, "linear_solver_threads", 0) == 1 &&
        1 <= get(result, "cone_threads", 0) <= width
    pass = ok && get(validation, :pass, false) && receipt_ok
    push!(records, Dict("run" => i, "phase" => i == 1 ? "cold" : "warm",
        "sample_scope" => "fresh_cli_process",
        "pass" => pass, "status" => get(result, "status", "Error"),
        "validation" => safe(validation), "receipt_ok" => receipt_ok,
        "native_seconds" => get(result, "native_seconds", nothing),
        "api_seconds" => get(result, "api_seconds", nothing),
        "load_seconds" => get(result, "load_seconds", nothing),
        "cli_e2e_seconds" => cli_seconds,
        "iterations" => get(result, "iterations", nothing),
        "execution" => Dict(k => get(result, k, nothing) for k in
            ("linear_solver", "linear_solver_threads", "cone_threads", "kkt_form")),
        "error" => err))
    rm(result_path, force=true)
end
rm(input, force=true)
metadata = Dict("kind" => "targeted synthetic single-orthant diagnostic",
    "cli" => cli, "cli_sha256" => filehash(cli), "source_id" => identity,
    "driver_sha256" => filehash(@__FILE__), "input_sha256" => input_sha,
    "input_formula" => "q[j]=1; A[(j-1)*r+k,j]=-1; b=-k/r; x*=1",
    "n" => n, "m" => m, "rows_per_variable" => repeats,
    "precision_bits" => bits, "requested_threads" => width,
    "external_tolerance" => string(tol), "timing_scope" =>
    "one fresh native CLI process per sample; API/native/load and CLI wall times separate",
    "runs" => records, "pass" => all(r["pass"] for r in records))
io = Base.Filesystem.open(output, Base.JL_O_WRONLY | Base.JL_O_CREAT | Base.JL_O_EXCL, 0o600)
try
    write(io, JSON.json(metadata, 2) * "\n")
finally
    close(io)
end
metadata["pass"] || exit(1)
