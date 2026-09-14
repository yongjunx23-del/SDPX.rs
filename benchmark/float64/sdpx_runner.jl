#!/usr/bin/env julia
#
# SDPX Julia/Rust frontend leg. Adapted from the retained phase13 runner.
# External oracle and fresh-solve timing scopes are unchanged; see README.md.
#
# Consumes the *same* Clarabel JSON bytes as runner/rust-harness (the sha256 in
# fixtures/manifest.json identifies that shared input) and emits the same
# per-run oracle fields and PASS/FAIL gate.
#
# Per-run oracle:
#   1. finiteness of x, z, s and returned-slack affine/cone residuals
#   2. r_p_x = dist_K(b - Ac x)      primal feasibility from the primal point
#   3. r_d   = || Ac' z + q ||_inf   dual feasibility
#   4. dist_K*(z)                    dual cone membership
#   5. gap   = | q'x + b'z | / (1 + |q'x|)
# plus the ordinary solver status; independent residuals determine accuracy;
# residuals alone never mark a run as PASS.

using SparseArrays
using LinearAlgebra
using Printf
using JSON
using SDPX

if haskey(ENV, "SDPX_EXPECTED_SOURCE")
    realpath(dirname(dirname(pathof(SDPX)))) == realpath(ENV["SDPX_EXPECTED_SOURCE"]) ||
        error("SDPX environment does not resolve to the declared candidate")
end

# Benchmark-only selection: AppleAccelerate redirects libblastrampoline.
# Production dependencies and provider choices remain outside this driver.
const SDPX_BENCH_BLAS = get(ENV, "SDPX_BENCH_BLAS", "default")
const SDPX_BENCH_THREADS = parse(Int, get(ENV, "SDPX_BENCH_THREADS", "1"))
const SDPX_BENCH_PREPROCESSING = get(ENV, "SDPX_BENCH_PREPROCESSING", "default")
SDPX_BENCH_PREPROCESSING in ("default", "controlled") ||
    error("SDPX_BENCH_PREPROCESSING must be default or controlled")
SDPX_BENCH_THREADS in (1, 2, 4, 8) || error("SDPX_BENCH_THREADS must be 1, 2, 4 or 8")
SDPX_BENCH_BLAS in ("default", "accelerate") ||
    error("SDPX_BENCH_BLAS must be default or accelerate")
if SDPX_BENCH_BLAS == "accelerate"
    import AppleAccelerate
    AppleAccelerate.get_macos_version() >= v"15" ||
        error("the Accelerate benchmark thread receipt requires macOS 15 or later")
    AppleAccelerate.set_num_threads(1)
end
BLAS.set_num_threads(1)
benchmark_blas_threads() = SDPX_BENCH_BLAS == "accelerate" ?
    AppleAccelerate.get_num_threads() : BLAS.get_num_threads()
benchmark_blas_threads() == 1 || error("benchmark requires one BLAS thread")

# ---------------------------------------------------------------------------
# Common input parsing uses the benchmark environment's JSON package.
_f64(x) = Float64(x)

function _csc(d)
    m = Int(d["m"])
    n = Int(d["n"])
    colptr = Int[Int(c) for c in d["colptr"]] .+ 1
    rowval = Int[Int(c) for c in d["rowval"]] .+ 1
    nzval = Float64[_f64(v) for v in d["nzval"]]
    return SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

function read_problem_json(path::String)
    data = JSON.parsefile(path)
    q = Float64[_f64(v) for v in data["q"]]
    b = Float64[_f64(v) for v in data["b"]]
    A = _csc(data["A"])
    return q, A, b, data["cones"]
end

# ---------------------------------------------------------------------------
# JSON-safe serialisation
# ---------------------------------------------------------------------------

_jj(x::Bool) = x ? "true" : "false"
_jj(x::Real) = isfinite(Float64(x)) ? string(Float64(x)) : "null"
_jj(x::AbstractString) = JSON.json(x)

# Receipts are serialized outside setup/solve timing. Keep the actual core
# field names, including optional tolerance overrides and backend parameters.
_jj(::Nothing) = "null"
_jj(x::Symbol) = _jj(string(x))
_jj(x::Integer) = string(x)
_jj(x::AbstractVector) = "[" * join(_jj.(x), ",") * "]"
_setting_json(x) = x isa Real && !isfinite(x) ? _jj(string(x)) : _jj(x)
function settings_json(settings, result)
    result.info.precision_bits == 53 || error("Float64 precision receipt mismatch")
    settings.max_threads == SDPX_BENCH_THREADS || error("requested native thread receipt mismatch")
    for key in (:equilibration, :presolve_enable, :chordal_decomposition_enable)
        getproperty(settings,key) == getproperty(result.info,key) ||
            error("native preprocessing receipt mismatch: $key")
    end
    factorization = result.info.factorization
    factorization in (:qdldl, :condensed_qdldl, :faer, :condensed_faer) ||
        error("unsupported factorization thread receipt: $factorization")
    factor_threads = factorization in (:faer, :condensed_faer) ? SDPX_BENCH_THREADS : 1
    result.info.backend_threads == factor_threads || error("native factorization thread receipt mismatch")
    1 <= result.info.cone_threads <= SDPX_BENCH_THREADS || error("native cone thread budget exceeded")
    benchmark_blas_threads() == 1 || error("benchmark requires one BLAS thread")
    fields = [_jj(string(k)) * ":" * _setting_json(getfield(settings, k))
              for k in fieldnames(typeof(settings)) if k != :outputs]
    append!(fields, ["\"requested_native_threads\":" * _jj(SDPX_BENCH_THREADS),
        "\"actual_backend_threads\":" * _jj(result.info.backend_threads),
        "\"actual_cone_threads\":" * _jj(result.info.cone_threads),
        "\"actual_blas_threads\":" * _jj(benchmark_blas_threads()),
        "\"actual_factorization\":" * _jj(result.info.factorization),
        "\"actual_kkt_form\":" * _jj(result.info.kkt_form),
        "\"preprocessing_arm\":" * _jj(SDPX_BENCH_PREPROCESSING)])
    return "{" * join(fields, ",") * "}"
end

"""JSON-safe fixed/two-exponent formatting: non-finite becomes `null`."""
function _jf6(x::Real)
    v = Float64(x)
    return isfinite(v) ? @sprintf("%.6f", v) : "null"
end

function _jfe(x::Real)
    v = Float64(x)
    return isfinite(v) ? @sprintf("%.6e", v) : "null"
end

# ---------------------------------------------------------------------------
# Clarabel JSON cones -> SDPX native program
# ---------------------------------------------------------------------------

const _SUPPORTED = "ZeroConeT/NonnegativeConeT/SecondOrderConeT/PSDTriangleConeT"

"""Block length in the Clarabel slack vector for one JSON cone."""
function _block_len(tag::String, param)
    tag == "ZeroConeT" && return Int(param)
    tag == "NonnegativeConeT" && return Int(param)
    tag == "SecondOrderConeT" && return Int(param)
    if tag == "PSDTriangleConeT"
        k = Int(param)                       # JSON stores the matrix side
        return div(k * (k + 1), 2)
    end
    error("residual oracle supports $(_SUPPORTED) only; fixture contains $tag")
end

"""Clarabel svec index: upper triangle column-wise, diagonal per column."""
_cl_svec_index_c(row, col) = row + div(col * (col - 1), 2)

# Fresh public standard-form setup: cone construction belongs inside timing.
function build_sdpx_program(q, A, b, cones, T::Type=Float64)
    specs = [getproperty(SDPX, Symbol(first(keys(c))))(Int(first(values(c)))) for c in cones]
    enabled = SDPX_BENCH_PREPROCESSING == "default"
    return (T.(q), SparseMatrixCSC{T,Int}(A), T.(b), specs), SDPX.Settings(T;
        verbose=false, max_threads=SDPX_BENCH_THREADS, max_iter=200, tol_gap_abs=1e-8,
        tol_gap_rel=1e-8, tol_feas=1e-8, equilibration=enabled ? :ruiz : :off,
        presolve_enable=enabled, chordal_decomposition_enable=enabled)
end
function solve_full(data, settings, T::Type=Float64)
    return SDPX.solve_conic(data...; settings=settings, return_result=true)
end

# ---------------------------------------------------------------------------
# cone geometry
# ---------------------------------------------------------------------------

function _psd_mat(v, k::Int)
    M = zeros(Float64, k, k)
    idx = 1
    for j in 1:k, i in 1:j
        M[i, j] = i == j ? Float64(v[idx]) : Float64(v[idx]) / sqrt(2.0)
        M[j, i] = M[i, j]
        idx += 1
    end
    return M
end

"""Distance of `v` to the cone product; `dual=true` evaluates dual membership
(ZeroConeT's dual is free; the other cones here are self-dual)."""
function cone_distance(v, cones; dual::Bool = false)
    any(!isfinite, v) && return NaN
    worst = 0.0
    off = 0
    for cone in cones
        tag = first(keys(cone))
        p = cone[tag]
        L = _block_len(tag, p)
        blk = @view v[(off + 1):(off + L)]
        d = if tag == "ZeroConeT"
            dual ? 0.0 : maximum(abs, blk)
        elseif tag == "NonnegativeConeT"
            max(0.0, -minimum(blk))
        elseif tag == "SecondOrderConeT"
            max(0.0, norm(blk[2:end]) - blk[1])
        else
            max(0.0, -eigmin(Symmetric(_psd_mat(blk, Int(p)))))
        end
        worst = max(worst, d)
        off += L
    end
    return worst
end

# The direct public solve already returns original Clarabel row coordinates.
phase_timings(result) = nothing

"""Independent original-coordinate residuals plus the solver status and the external accuracy gate."""
function oracle(result, q::Vector{Float64}, A::AbstractMatrix, b::Vector{Float64},
    cones, tol::Float64; dual_sign::Float64 = 1.0)
    x = Float64.(SDPX.value(result))
    z = dual_sign .* Float64.(SDPX.dual(result))
    returned_s = Float64.(SDPX.dual_slack(result))

    finite = all(isfinite, x) && all(isfinite, z) && all(isfinite, returned_s)
    s = b .- A * x
    r_p = norm(returned_s .- s, Inf)
    dist_k_s = cone_distance(returned_s, cones)
    r_p_x = cone_distance(s, cones)
    r_d = norm(A' * z .+ q, Inf)
    dist_k_z = cone_distance(z, cones; dual = true)
    qx = dot(q, x)
    bz = dot(b, z)
    gap = abs(qx + bz) / (1.0 + abs(qx))

    tol_feas = tol * (1.0 + norm(b, Inf))
    tol_dual = tol * (1.0 + norm(q, Inf))
    residuals_ok = finite && r_p <= tol_feas && dist_k_s <= tol_feas &&
                   r_p_x <= tol_feas && r_d <= tol_dual &&
                   dist_k_z <= tol_dual && gap <= tol
    solver_optimal = SDPX.status(result) in (SDPX.Optimal, :optimal)
    return (;
        finite,
        status = string(SDPX.status(result)),
        termination_reason = string(SDPX.status(result)),
        termination_stage = "solver",
        termination_message = string(result.status),
        iterations = Int(SDPX.iterations(result)),
        solver_optimal,
        r_p,
        dist_k_s,
        r_p_x,
        r_d,
        dist_k_z,
        gap,
        residuals_ok,
        pass = solver_optimal && residuals_ok,
        tol_feas,
        tol_dual,
        primal_objective = Float64(SDPX.primal_objective(result)),
        dual_objective = Float64(SDPX.dual_objective(result)),
        solver_primal_affine_residual = Float64(SDPX.primal_residual(result)),
        solver_dual_affine_residual = Float64(SDPX.dual_residual(result)),
        solver_gap = Float64(SDPX.relative_gap(result)),
        timings = phase_timings(result),
    )
end

function _median(v::Vector{Float64})
    isempty(v) && return NaN
    w = sort(v)
    n = length(w)
    return isodd(n) ? w[(n + 1) ÷ 2] : 0.5 * (w[n ÷ 2] + w[n ÷ 2 + 1])
end

function _timings_json(t)
    t === nothing && return "null"
    return "{" * join(["\"$k\":" * _jj(v) for (k, v) in sort(collect(t))], ",") * "}"
end

function _run_json(rec)
    rec.ok || return string(
        "{\"ok\":false,\"error\":", _jj(rec.error),
        ",\"setup_s\":null,\"solve_s\":null,\"e2e_s\":null,\"pass\":false}")
    r = rec.oracle
    return string(
        "{\"ok\":true",
        ",\"setup_s\":", _jf6(rec.setup_s),
        ",\"solve_s\":", _jf6(rec.solve_s),
        ",\"e2e_s\":", _jf6(rec.setup_s + rec.solve_s),
        ",\"settings\":", rec.settings,
        ",\"status\":", _jj(r.status),
        ",\"termination_reason\":", _jj(r.termination_reason),
        ",\"termination_stage\":", _jj(r.termination_stage),
        ",\"termination_message\":", _jj(r.termination_message),
        ",\"iterations\":", r.iterations,
        ",\"solver_optimal\":", _jj(r.solver_optimal),
        ",\"finite\":", _jj(r.finite),
        ",\"r_p\":", _jfe(r.r_p),
        ",\"dist_K_s\":", _jfe(r.dist_k_s),
        ",\"r_p_x\":", _jfe(r.r_p_x),
        ",\"r_d\":", _jfe(r.r_d),
        ",\"dist_Kstar_z\":", _jfe(r.dist_k_z),
        ",\"gap\":", _jfe(r.gap),
        ",\"solver_primal_affine_residual\":", _jfe(r.solver_primal_affine_residual),
        ",\"solver_dual_affine_residual\":", _jfe(r.solver_dual_affine_residual),
        ",\"solver_gap\":", _jfe(r.solver_gap),
        ",\"timings\":", _timings_json(r.timings),
        ",\"pass\":", _jj(r.pass), "}",
    )
end

function main()
    files = String[]
    runs = 8
    tol = 1e-6
    dual_sign = 1.0
    for a in ARGS
        if startswith(a, "--runs=")
            runs = parse(Int, split(a, '=')[2])
        elseif startswith(a, "--tol=")
            tol = parse(Float64, split(a, '=')[2])
        elseif startswith(a, "--dual-sign=")
            dual_sign = parse(Float64, split(a, '=')[2])
        elseif startswith(a, "--")
            continue
        else
            push!(files, a)
        end
    end
    isempty(files) && error("usage: sdpx_runner.jl <problem.json>... [--runs=N] [--tol=X]")

    println("{\"impl\":\"SDPX Julia/Rust\",\"julia\":", _jj(string(VERSION)),
        ",\"runs\":", runs, ",\"tol\":", tol,
        ",\"timing\":\"fresh model build + solve per run (no reuse)\",",
        "\"blas_selection\":", _jj(SDPX_BENCH_BLAS),
        ",\"requested_native_threads\":", SDPX_BENCH_THREADS,
        ",\"rayon_num_threads\":", _jj(get(ENV, "RAYON_NUM_THREADS", "unset")),
        ",\"appleaccelerate_version\":", _jj(SDPX_BENCH_BLAS == "accelerate" ? string(Base.pkgversion(AppleAccelerate)) : nothing),
        ",\"blas_num_threads\":", BLAS.get_num_threads(),
        ",\"blas_config\":", _jj(string(BLAS.get_config())), "}")
    flush(stdout)

    all_pass = true
    for f in files
        name = replace(basename(f), ".json" => "")
        records = []
        t_json = NaN
        cone_str = ""
        n_vars = 0
        m_rows = 0

        try
            t = time()
            q, A, b, cones = read_problem_json(f)
            t_json = time() - t
            n_vars = length(q)
            m_rows = size(A, 1)
            cone_str = join(
                [string(first(keys(c)), ":", c[first(keys(c))]) for c in cones], ",")

            for _ in 1:runs
                try
                    t = time()
                    model, program = build_sdpx_program(q, A, b, cones)
                    setup_s = time() - t
                    t = time()
                    res = solve_full(model, program)
                    solve_s = time() - t
                    push!(records, (ok = true, setup_s = setup_s, solve_s = solve_s,
                        error = "", settings = settings_json(program, res),
                        oracle = oracle(res, q, A, b, cones, tol;
                            dual_sign = dual_sign)))
                catch err
                    push!(records, (ok = false, setup_s = NaN, solve_s = NaN,
                        error = sprint(showerror, err), oracle = nothing))
                end
            end
        catch err
            push!(records, (ok = false, setup_s = NaN, solve_s = NaN,
                error = sprint(showerror, err), oracle = nothing))
        end

        ok_recs = [r for r in records if r.ok]
        pass = !isempty(ok_recs) && all(r -> r.oracle.pass, ok_recs) &&
               length(ok_recs) == runs
        all_pass &= pass

        e2e = [r.setup_s + r.solve_s for r in ok_recs]
        cold_e2e = isempty(e2e) ? NaN : e2e[1]
        warm_e2e = length(e2e) > 1 ? _median(e2e[2:end]) : NaN
        setups = Float64[r.setup_s for r in ok_recs]
        solves = Float64[r.solve_s for r in ok_recs]
        cost = isempty(ok_recs) ? NaN : ok_recs[1].oracle.primal_objective
        tol_feas = isempty(ok_recs) ? NaN : ok_recs[1].oracle.tol_feas
        tol_dual = isempty(ok_recs) ? NaN : ok_recs[1].oracle.tol_dual

        print("{\"instance\":", _jj(name), ",\"n\":", n_vars, ",\"m\":", m_rows,
            ",\"cones\":", _jj(cone_str), ",\"runs_n\":", runs,
            ",\"t_json_s\":", _jf6(t_json),
            ",\"cold_e2e_s\":", _jf6(cold_e2e),
            ",\"warm_e2e_median_s\":", _jf6(warm_e2e),
            ",\"setup_median_s\":", _jf6(_median(setups)),
            ",\"solve_median_s\":", _jf6(_median(solves)),
            ",\"cost_primal\":", _jfe(cost),
            ",\"tol_feas\":", _jfe(tol_feas),
            ",\"tol_dual\":", _jfe(tol_dual),
            ",\"tol_gap\":", _jfe(tol),
            ",\"pass\":", _jj(pass), ",\"runs\":[")
        for (i, r) in enumerate(records)
            i > 1 && print(",")
            print(_run_json(r))
        end
        println("]}")
        flush(stdout)
    end
    all_pass || exit(1)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
