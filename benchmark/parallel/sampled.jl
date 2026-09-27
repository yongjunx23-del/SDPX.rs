#!/usr/bin/env julia
# Deterministic sampled-factor input generator plus native CLI driver.
# No SDPX Julia package is loaded: factors are sent directly to Rust JSON.
using SparseArrays, LinearAlgebra, SHA, JSON
if get(ENV, "SDPX_BITS", "53") != "53"
    try
        import GenericLinearAlgebra
    catch
        error("MPFR sampled audit requires GenericLinearAlgebra")
    end
end

required(k) = isempty(get(ENV, k, "")) ? error("set $k explicitly") : ENV[k]
const cli = realpath(required("SDPX_CLI"))
isfile(cli) && Base.Filesystem.isexecutable(cli) || error("SDPX_CLI is not an executable file")
const identity = required("SDPX_SOURCE_ID")
const output = required("SDPX_OUTPUT")
ispath(output) && error("refusing to overwrite $output")
const bits = parse(Int, get(ENV, "SDPX_BITS", "53"))
const width = parse(Int, get(ENV, "SDPX_THREADS", "1"))
const n = parse(Int, get(ENV, "SDPX_N", bits == 53 ? "64" : "24"))
bits in (53, 256, 512) || error("SDPX_BITS must be 53, 256 or 512")
width in (1, 2, 4, 8) || error("SDPX_THREADS must be 1, 2, 4 or 8")
n >= 2 || error("SDPX_N must be at least 2")
bits == 53 || setprecision(BigFloat, bits)
const m = div(Base.checked_mul(n, Base.checked_add(n, 1)), 2)
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

function unpack(v, root2)
    T = eltype(v)
    M = Matrix{T}(undef, n, n)
    p = 0
    for j in 1:n, i in 1:j
        p += 1
        value = i == j ? v[p] : v[p] / root2
        M[i, j] = value
        M[j, i] = value
    end
    M
end
psd_distance(v, root2) = max(zero(eltype(v)), -minimum(eigvals(Symmetric(unpack(v, root2))))) /
    (one(eltype(v)) + opnorm(unpack(v, root2), Inf))

function audit(result, q, A, b, cones, tol, root2)
    T = eltype(q)
    x = T[T(v) for v in result["x"]]
    z = T[T(v) for v in result["z"]]
    s = T[T(v) for v in result["s"]]
    finite = all(isfinite, x) && all(isfinite, z) && all(isfinite, s)
    finite || return (pass=false, finite=false, status=string(result["status"]))
    slack = b - A * x
    primal = norm(slack - s, Inf) / (one(T) + norm(b, Inf))
    dual = norm(q + A' * z, Inf) / (one(T) + norm(q, Inf))
    primal_psd = psd_distance(slack, root2)
    slack_psd = psd_distance(s, root2)
    dual_psd = psd_distance(z, root2)
    pobj, dobj = dot(q, x), -dot(b, z)
    gap = abs(pobj - dobj) / (one(T) + abs(pobj))
    objective_error = abs(pobj - T(n)) / (one(T) + T(n))
    reported_objective_error = max(abs(T(result["objective"]) - pobj),
        abs(T(result["dual_objective"]) - dobj)) / (one(T) + abs(pobj))
    status = string(result["status"])
    optimal = status in ("Solved", "Optimal", "solved", "optimal")
    values = (primal, dual, primal_psd, slack_psd, dual_psd, gap,
              objective_error, reported_objective_error)
    passed = optimal && all(v -> isfinite(v) && v <= tol, values)
    return (pass=passed, finite=true, status=status, tol=tol, primal=primal,
        dual=dual, primal_psd=primal_psd, slack_psd=slack_psd,
        dual_psd=dual_psd, gap=gap, objective_error=objective_error,
        reported_objective_error=reported_objective_error, pobj=pobj, dobj=dobj)
end

T = bits == 53 ? Float64 : BigFloat
rho = T(1) / T(4)
offset = rho / T(n)
Q = Matrix{T}(undef, n, n)
for j in 1:n, i in 1:n
    Q[i, j] = (i == j ? one(T) : zero(T)) + offset
end
q = T[one(T) for _ in 1:n]
weights = T[-one(T) for _ in 1:n]
b = T[i == j ? -one(T) : zero(T) for j in 1:n for i in 1:j]
A = Matrix{T}(undef, m, n)
for k in 1:n
    p = 0
    for j in 1:n, i in 1:j
        p += 1
        A[p, k] = -Q[i, k] * Q[j, k] * (i == j ? one(T) : sqrt(T(2)))
    end
end
sampled = [Dict("row_start" => 0, "column_start" => 0, "dim" => 1,
    "basis_rows" => n, "basis_cols" => n, "basis" => wire.(vec(Q)),
    "weights" => wire.(weights))]
problem = Dict("P" => Dict("m" => n, "n" => n, "colptr" => fill(0, n + 1),
    "rowval" => Int[], "nzval" => wire.(T[])), "q" => wire.(q),
    "A" => Dict("m" => m, "n" => n, "colptr" => fill(0, n + 1),
        "rowval" => Int[], "nzval" => wire.(T[])), "b" => wire.(b),
    "cones" => [Dict("PSDTriangleConeT" => n)], "sampled" => sampled,
    "settings" => Dict("verbose" => false, "max_threads" => width,
        "kkt_form" => "condensed"))
input = tempname() * ".json"
write(input, JSON.json(problem) * "\n")
input_sha = filehash(input)
tol = bits == 53 ? T(1e-8) : sqrt(eps(T))
root2 = sqrt(T(2))
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
            audit(result, q, A, b, [Dict("PSDTriangleConeT" => n)], tol, root2)
        catch ex
            (pass=false, finite=false, status="Error", error=sprint(showerror, ex))
        end
    else
        (pass=false, finite=false, status="Error")
    end
    actual_kkt = get(result, "kkt_form", startswith(string(get(result, "linear_solver", "")), "condensed_") ? "condensed" : "augmented")
    receipt_ok = ok && get(result, "precision_bits", 0) == bits &&
        get(result, "threads_requested", 0) == width &&
        actual_kkt == "condensed" &&
        get(result, "linear_solver_threads", 0) >= 1 &&
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
metadata = Dict("kind" => "predeclared synthetic sampled PSD diagnostic",
    "cli" => cli, "cli_sha256" => filehash(cli), "source_id" => identity,
    "driver_sha256" => filehash(@__FILE__), "input_sha256" => input_sha,
    "input_formula" => "Q=I+(1/4)*ones(n,n)/n; weights=-1; q=ones; b=-svec(I)",
    "n" => n, "m" => m, "psd_order" => n, "precision_bits" => bits,
    "requested_threads" => width, "external_tolerance" => string(tol),
    "input_route" => "sampled_factors", "timing_scope" =>
    "one fresh native CLI process per sample; API/native/load and CLI wall times separate",
    "runs" => records, "pass" => all(r["pass"] for r in records))
io = Base.Filesystem.open(output, Base.JL_O_WRONLY | Base.JL_O_CREAT | Base.JL_O_EXCL, 0o600)
try
    write(io, JSON.json(metadata, 2) * "\n")
finally
    close(io)
end
metadata["pass"] || exit(1)
