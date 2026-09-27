# Lightweight receipt gate for a native SDPX CLI result.
# This script intentionally loads only JSON; it never loads the retired SDPX.jl
# frontend and performs no solver work.  The original-coordinate numerical
# gate remains audit_point.jl and runs outside the solve timing.
using JSON

length(ARGS) == 3 || error("usage: gate.jl RAW_JSON BITS THREADS")
raw = JSON.parsefile(ARGS[1])
bits = parse(Int, ARGS[2])
threads = parse(Int, ARGS[3])
bits in (512, 768) || error("unsupported precision")
threads in (1, 2, 4, 8, 16, 32, 64) || error("unsupported thread budget")
get(raw, "status", "") == "Solved" || error("native status is not Solved")
get(raw, "precision_bits", 0) == bits || error("native precision mismatch")
get(raw, "threads_requested", threads) == threads || error("native thread request mismatch")
for key in ("native_seconds", "api_seconds", "load_seconds")
    value = Float64(get(raw, key, NaN))
    isfinite(value) && value >= 0 || error("invalid timing: $key")
end
Float64(raw["native_seconds"]) > 0 || error("native solver timing is not positive")
Float64(raw["api_seconds"]) > 0 || error("native API timing is not positive")
for key in ("primal_residual", "dual_residual")
    if haskey(raw, key)
        value = parse(BigFloat, string(raw[key]))
        isfinite(value) && value <= parse(BigFloat, "1e-42") ||
            error("native $key exceeds 1e-42")
    end
end
for key in ("x", "s", "z")
    get(raw, key, nothing) isa AbstractVector || error("missing native vector: $key")
end
solver = get(raw, "linear_solver", "")
solver isa AbstractString && !isempty(solver) || error("missing native linear solver")
raw["linear_solver_threads"] isa Integer || error("missing native linear solver thread receipt")
raw["cone_threads"] isa Integer || error("missing native cone thread receipt")
println("PASS native status=", raw["status"], " precision=", bits,
        " threads=", threads)
