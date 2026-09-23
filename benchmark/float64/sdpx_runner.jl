#!/usr/bin/env julia
# Standalone native-CLI benchmark adapter.
#
# This file intentionally does not load the retired Julia solver package:
# Julia is only an independent
# original-coordinate oracle and JSON/input utility.  Every sample launches a
# fresh Rust `sdpx` process and retains native/API/load/CLI timings separately.
using SparseArrays, LinearAlgebra, SHA, JSON

required(k) = isempty(get(ENV, k, "")) ? error("set $k explicitly") : ENV[k]
const cli = realpath(get(ENV, "SDPX_CLI", required("SDPX_CLI")))
isfile(cli) && Base.Filesystem.isexecutable(cli) || error("SDPX_CLI is not an executable file")
const threads = parse(Int, get(ENV, "SDPX_BENCH_THREADS", "1"))
threads in (1, 2, 4, 8) || error("SDPX_BENCH_THREADS must be 1, 2, 4 or 8")
const blas = get(ENV, "SDPX_BENCH_BLAS", "default")
blas in ("default", "accelerate") || error("unsupported benchmark BLAS label")
const requested_runs = Ref(4)
const requested_tol = Ref(1e-6)
const requested_settings = Ref{Union{Nothing,String}}(nothing)

for arg in ARGS
    startswith(arg, "--runs=") && (requested_runs[] = parse(Int, split(arg, "=", limit=2)[2]))
    startswith(arg, "--tol=") && (requested_tol[] = parse(Float64, split(arg, "=", limit=2)[2]))
    startswith(arg, "--settings=") &&
        (requested_settings[] = split(arg, "=", limit=2)[2])
end
requested_runs[] > 0 || error("--runs must be positive")
requested_settings[] === nothing || isfile(requested_settings[]) ||
    error("--settings path is not a file")

_f(x) = try Float64(x) catch; NaN end
filehash(path) = open(sha256, path) |> bytes2hex
safe(x::AbstractFloat) = isfinite(x) ? x : nothing
safe(x::Integer) = x
safe(x::Bool) = x
safe(x::AbstractString) = x
safe(x::Nothing) = nothing
safe(x::NamedTuple) = Dict(string(k) => safe(v) for (k, v) in pairs(x))
safe(x::AbstractDict) = Dict(string(k) => safe(v) for (k, v) in x)
safe(x::AbstractVector) = safe.(x)

function csc(d)
    m, n = Int(d["m"]), Int(d["n"])
    colptr = Int[Int(v) + 1 for v in d["colptr"]]
    rowval = Int[Int(v) + 1 for v in d["rowval"]]
    nzval = Float64[_f(v) for v in d["nzval"]]
    SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

function block_length(tag, p)
    tag in ("ZeroConeT", "NonnegativeConeT", "SecondOrderConeT") && return Int(p)
    tag == "PSDTriangleConeT" && return div(Int(p) * (Int(p) + 1), 2)
    error("unsupported cone $tag")
end

function psd_matrix(v, side)
    M = zeros(Float64, side, side)
    p = 0
    for j in 1:side, i in 1:j
        p += 1
        M[i, j] = i == j ? v[p] : v[p] / sqrt(2.0)
        M[j, i] = M[i, j]
    end
    M
end

function cone_distance(v, cones; dual=false)
    any(!isfinite, v) && return NaN
    off, worst = 1, 0.0
    for cone in cones
        tag, parameter = first(collect(cone))
        len = block_length(tag, parameter)
        block = @view v[off:off + len - 1]
        distance = if tag == "ZeroConeT"
            dual ? 0.0 : maximum(abs, block)
        elseif tag == "NonnegativeConeT"
            max(0.0, -minimum(block))
        elseif tag == "SecondOrderConeT"
            max(0.0, norm(block[2:end]) - block[1])
        else
            max(0.0, -minimum(eigvals(Symmetric(psd_matrix(block, Int(parameter))))))
        end
        worst = max(worst, distance)
        off += len
    end
    worst
end

function audit(result, q, P, A, b, cones, tol)
    x, z, s = Float64.([_f(v) for v in result["x"]]), Float64.([_f(v) for v in result["z"]]),
        Float64.([_f(v) for v in result["s"]])
    finite = all(isfinite, x) && all(isfinite, z) && all(isfinite, s)
    finite || return (pass=false, finite=false, status=string(get(result, "status", "Error")))
    slack = b - A * x
    rp = norm(slack - s, Inf)
    dist_s = cone_distance(s, cones)
    rp_x = cone_distance(slack, cones)
    # P is serialized as an upper triangle.  Use it as the symmetric
    # quadratic operator for stationarity and both objective values.
    px = P * x
    rd = norm(px + A' * z + q, Inf)
    dist_z = cone_distance(z, cones; dual=true)
    pxx, qx, bz = dot(x, px), dot(q, x), dot(b, z)
    primal_objective = 0.5 * pxx + qx
    dual_objective = -0.5 * pxx - bz
    gap = abs(primal_objective - dual_objective) / (1.0 + abs(primal_objective))
    tol_feas, tol_dual = tol * (1.0 + norm(b, Inf)), tol * (1.0 + norm(q, Inf))
    status = string(get(result, "status", "Error"))
    optimal = status in ("Solved", "Optimal", "solved", "optimal")
    residuals_ok = all(isfinite(v) && v <= tol_feas for v in (rp, dist_s, rp_x)) &&
        all(isfinite(v) && v <= tol_dual for v in (rd, dist_z)) && isfinite(gap) && gap <= tol
    (pass=optimal && residuals_ok, finite=true, status=status, solver_optimal=optimal,
        r_p=rp, dist_K_s=dist_s, r_p_x=rp_x, r_d=rd, dist_Kstar_z=dist_z,
        gap=gap, tol_feas=tol_feas, tol_dual=tol_dual,
        independent_primal_objective=primal_objective,
        independent_dual_objective=dual_objective,
        primal_objective=_f(result["objective"]), dual_objective=_f(result["dual_objective"]),
        solver_primal_affine_residual=_f(result["primal_residual"]),
        solver_dual_affine_residual=_f(result["dual_residual"]))
end

function run_input(path)
    data = JSON.parsefile(path)
    q, b, P, A, cones = Float64.([_f(v) for v in data["q"]]), Float64.([_f(v) for v in data["b"]]),
        Symmetric(csc(data["P"]), :U), csc(data["A"]), data["cones"]
    records = Any[]
    for i in 1:requested_runs[]
        result_path = tempname() * ".json"
        started = time()
        ok, err = true, nothing
        try
            argv = [cli, path, "--precision", "53", "--threads", string(threads),
                "--quiet", "--output", result_path]
            if requested_settings[] !== nothing
                append!(argv, ["--settings", requested_settings[]])
            end
            run(Cmd(argv))
        catch ex
            ok, err = false, sprint(showerror, ex)
        end
        cli_seconds = time() - started
        result_available = isfile(result_path)
        result = result_available ? JSON.parsefile(result_path) : Dict{String,Any}()
        validation = if result_available
            try
                audit(result, q, P, A, b, cones, requested_tol[])
            catch ex
                (pass=false, finite=false, status="Error", error=sprint(showerror, ex))
            end
        else
            (pass=false, finite=false, status="Error")
        end
        receipt_ok = result_available && get(result, "precision_bits", 0) == 53 &&
            get(result, "threads_requested", 0) == threads && 1 <= get(result, "cone_threads", 0) <= threads
        pass = ok && get(validation, :pass, false) && receipt_ok
        push!(records, Dict("run" => i, "phase" => i == 1 ? "cold" : "warm",
            "sample_scope" => "fresh_cli_process",
            "ok" => ok, "result_available" => result_available, "pass" => pass,
            "status" => get(result, "status", "Error"),
            "termination_reason" => get(result, "status", "Error"),
            "validation" => safe(validation), "receipt_ok" => receipt_ok,
            "native_seconds" => get(result, "native_seconds", nothing),
            "api_seconds" => get(result, "api_seconds", nothing),
            "load_seconds" => get(result, "load_seconds", nothing),
            "cli_e2e_seconds" => cli_seconds, "iterations" => get(result, "iterations", nothing),
            "settings" => get(result, "settings", Dict()), "error" => err))
        rm(result_path, force=true)
    end
    result_records = [r for r in records if r["result_available"]]
    pass = length(result_records) == requested_runs[] && all(r["ok"] && r["pass"] for r in records)
    e2e = [r["cli_e2e_seconds"] for r in result_records]
    warm = length(e2e) > 1 ? sort(e2e[2:end])[cld(length(e2e[2:end]), 2)] : nothing
    Dict("instance" => splitext(basename(path))[1], "n" => length(q), "m" => length(b),
        "runs_n" => requested_runs[], "tol" => requested_tol[], "pass" => pass,
        "cli" => cli, "cli_sha256" => filehash(cli), "timing_scope" =>
        "one fresh native CLI process per sample; API/native/load and CLI wall times separate",
        "cold_cli_e2e_s" => isempty(e2e) ? nothing : e2e[1], "warm_cli_e2e_median_s" => warm,
        "runs" => records)
end

files = filter(x -> !startswith(x, "--"), ARGS)
isempty(files) && error("usage: sdpx_runner.jl INPUT.json [--runs=N] [--tol=X]")
println(JSON.json(Dict("impl" => "SDPX native CLI", "cli" => cli,
    "cli_sha256" => filehash(cli), "runs" => requested_runs[], "tol" => requested_tol[],
    "settings" => requested_settings[],
    "timing" => "one fresh native CLI process per run", "requested_native_threads" => threads,
    "blas_selection" => blas)))
all_pass = true
for file in files
    row = try run_input(file) catch ex
        Dict("instance" => splitext(basename(file))[1], "pass" => false,
             "error" => sprint(showerror, ex), "runs" => Any[])
    end
    global all_pass &= get(row, "pass", false)
    println(JSON.json(row))
end
all_pass || exit(1)
