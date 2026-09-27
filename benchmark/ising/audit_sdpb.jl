#!/usr/bin/env julia
"""Audit one SDPB sampled output against the original sampled equations.

This is deliberately a standalone oracle process.  It reads the input JSON
and SDPB's exported ``x_*.txt``, ``y.txt``, ``X_matrix_*.txt`` and
``Y_matrix_*.txt`` files, then reuses the independent original-coordinate
audit in ``audit_helpers.jl``.  It never loads the retired SDPX Julia
frontend, and it is outside any solver timing scope.

Usage:

    julia --startup-file=no audit_sdpb.jl INPUT SDPB_OUT OUTPUT_JSON BITS REFERENCE_AUDIT
"""

length(ARGS) == 5 || error(
    "usage: audit_sdpb.jl INPUT SDPB_OUT OUTPUT_JSON BITS REFERENCE_AUDIT")

const input_dir = ARGS[1]
const output_dir = ARGS[2]
const output_path = ARGS[3]
const bits_text = ARGS[4]
const reference_path = ARGS[5]
const bits = try
    parse(Int, bits_text)
catch
    error("BITS must be an integer")
end
bits in (512, 768) || error("unsupported Ising precision; expected 512 or 768")
isdir(input_dir) || error("sampled input directory does not exist: $input_dir")
isdir(output_dir) || error("SDPB output directory does not exist: $output_dir")
isfile(reference_path) || error("reference audit does not exist: $reference_path")

const tolerance_text = "1e-30"
include(joinpath(@__DIR__, "audit_helpers.jl"))
BLAS.set_num_threads(1)

"""Hash all regular SDPB output files by name and content."""
function output_sha256(directory)
    files = sort(filter(isfile, readdir(directory; join=true)))
    isempty(files) && error("SDPB output directory is empty: $directory")
    manifest = join((basename(path) * "\0" * bytes2hex(sha256(read(path))) * "\n"
                     for path in files))
    bytes2hex(sha256(manifest))
end

"""Return the status line used by SDPB's output log."""
function solver_is_optimal(path)
    isfile(path) || error("missing SDPB solver log: $path")
    text = lowercase(read(path, String))
    occursin("found primal-dual optimal solution", text)
end

setprecision(BigFloat, bits) do
    reference = JSON.parsefile(reference_path)
    get(reference, "accepted", false) === true ||
        error("reference audit was not accepted")

    sampled = read_sampled_sdp(input_dir; T=BigFloat, bits=bits)
    reference_hash = get(reference, "input_sha256",
        get(get(reference, "mapping", Dict()), "input_sha256", nothing))
    reference_hash == sampled.input_sha256 ||
        error("reference belongs to another sampled input")
    ref_objective = parse(BigFloat, string(reference["objective"]))
    isfinite(ref_objective) || error("reference objective is not finite")

    conic = compile_sampled_primal(sampled)
    # SDPB writes one x file per sampled block and names matrix files by the
    # original block/parity index.  read_matrix handles both row and column
    # vectors emitted by Elemental's text writer.
    lambda = zeros(BigFloat, length(conic.q))
    for (block, columns) in zip(sampled.blocks, conic.block_columns)
        path = joinpath(output_dir, "x_$(block.block_index).txt")
        lambda[columns] = vec(read_matrix(path))
    end
    y = vec(read_matrix(joinpath(output_dir, "y.txt")))
    primal_matrices = [
        read_symmetric_matrix(joinpath(output_dir,
            "X_matrix_$(2 * g.block_index + g.parity).txt"))
        for g in conic.grams
    ]
    dual_matrices = [
        read_symmetric_matrix(joinpath(output_dir,
            "Y_matrix_$(2 * g.block_index + g.parity).txt"))
        for g in conic.grams
    ]
    point = pack_sampled_primal_solution(conic, lambda, y,
                                         primal_matrices, dual_matrices)
    audit_result = audit(conic, point.x, point.s, point.z)
    audit_result["mapping_relative"] = mapping_gate_value(audit_result["mapping"])
    audit_result["reference_objective_agreement"] =
        abs(audit_result["objective"] - ref_objective) /
        max(one(BigFloat), abs(audit_result["objective"]), abs(ref_objective))
    audit_result["reference_comparison"] = true

    optimal = solver_is_optimal(joinpath(output_dir, "out.txt"))
    fields = ("primal", "dual", "gap", "primal_psd", "dual_psd",
              "mapping_relative", "reference_objective_agreement")
    accepted = optimal && all(isfinite(audit_result[key]) &&
                              audit_result[key] <= parse(BigFloat, tolerance_text)
                              for key in fields)
    audit_result["accepted"] = accepted
    audit_result["optimal"] = optimal
    audit_result["status"] = optimal ? "Solved" : "NotSolved"
    audit_result["precision_bits"] = bits
    audit_result["precision_source"] = "BITS argument; SDPB log precision is checked by the harness"
    audit_result["input_sha256"] = sampled.input_sha256
    audit_result["sdpb_output_sha256"] = output_sha256(output_dir)
    audit_result["reference_sha256"] = bytes2hex(sha256(read(reference_path)))
    audit_result["output_skew"] = maximum_output_skew[]

    open(output_path, "w") do io
        JSON.print(io, stringify(audit_result), 2)
    end
    println("accepted=", accepted, " status=", audit_result["status"],
            " precision_bits=", bits)
    accepted || error("SDPB external audit failed")
end
