#!/usr/bin/env julia
# Precision-aware original-coordinate oracle for native JSON receipts.
# The solver executable is never loaded; this script only parses points and
# checks the fixed research gates outside the native timing scope.
using LinearAlgebra, SparseArrays, JSON

length(ARGS) >= 3 || error("usage: native_oracle.jl INPUT.json RESULT.json BITS")
const input_path, result_path, bits = ARGS[1], ARGS[2], parse(Int, ARGS[3])
if bits != 53
    try
        import GenericLinearAlgebra
    catch
        error("MPFR oracle requires GenericLinearAlgebra")
    end
end
const T = bits == 53 ? Float64 : BigFloat
bits == 53 || setprecision(BigFloat, bits)
num(x) = T(x)
data = JSON.parsefile(input_path)
result = JSON.parsefile(result_path)

function csc(d)
    m, n = Int(d["m"]), Int(d["n"])
    colptr = Int[Int(v) + 1 for v in d["colptr"]]
    rowval = Int[Int(v) + 1 for v in d["rowval"]]
    nzval = T[num(v) for v in d["nzval"]]
    m, n, colptr, rowval, nzval
end

function sparse_csc(d)
    m, n, colptr, rowval, nzval = csc(d)
    SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

function sampled_dense(data)
    m, n, _, _, _ = csc(data["A"])
    A = zeros(T, m, n)
    for block in get(data, "sampled", Any[])
        row_start, col_start = Int(block["row_start"]), Int(block["column_start"])
        dim, rows, cols = Int(block["dim"]), Int(block["basis_rows"]), Int(block["basis_cols"])
        basis = T[num(v) for v in block["basis"]]
        weights = T[num(v) for v in block["weights"]]
        index = 0
        for s in 0:dim - 1, r in 0:s, k in 0:cols - 1
            weight = weights[index + 1]
            index += 1
            if s == r
                for a in 0:rows - 1, b in a:rows - 1
                    i, j = s * rows + a, r * rows + b
                    qa, qb = basis[a + rows * k + 1], basis[b + rows * k + 1]
                    value = weight * qa * qb * (a == b ? one(T) : sqrt(T(2)))
                    row = row_start + j * (j + 1) ÷ 2 + i + 1
                    A[row, col_start + index - 1 + 1] += value
                end
            else
                for a in 0:rows - 1, b in 0:rows - 1
                    i, j = s * rows + a, r * rows + b
                    lo, hi = min(i, j), max(i, j)
                    qa, qb = basis[a + rows * k + 1], basis[b + rows * k + 1]
                    row = row_start + hi * (hi + 1) ÷ 2 + lo + 1
                    A[row, col_start + index - 1 + 1] += weight * qa * qb * sqrt(T(2)) / 2
                end
            end
        end
    end
    A
end

P = sparse_csc(data["P"])
P_operator = Symmetric(P, :U)
A = sparse_csc(data["A"])
if !isempty(get(data, "sampled", Any[]))
    A = A + sparse(sampled_dense(data))
end
q = T[num(v) for v in data["q"]]
b = T[num(v) for v in data["b"]]
x = T[num(v) for v in result["x"]]
s = T[num(v) for v in result["s"]]
z = T[num(v) for v in result["z"]]

function cone_block_distance(values, cones; dual=false)
    any(!isfinite, values) && return T(NaN)
    offset, worst = 1, zero(T)
    for cone in cones
        tag, p = first(collect(cone))
        parameter = Int(p)
        length = tag == "PSDTriangleConeT" ? parameter * (parameter + 1) ÷ 2 : parameter
        block = @view values[offset:offset + length - 1]
        distance = if tag == "ZeroConeT"
            dual ? zero(T) : maximum(abs, block)
        elseif tag == "NonnegativeConeT"
            max(zero(T), -minimum(block))
        elseif tag == "SecondOrderConeT"
            max(zero(T), norm(block[2:end]) - block[1])
        elseif tag == "PSDTriangleConeT"
            M = zeros(T, parameter, parameter)
            pindex = 0
            for j in 1:parameter, i in 1:j
                pindex += 1
                M[i, j] = i == j ? block[pindex] : block[pindex] / sqrt(T(2))
                M[j, i] = M[i, j]
            end
            max(zero(T), -minimum(eigvals(Symmetric(M))))
        else
            error("unsupported cone $tag")
        end
        worst = max(worst, distance)
        offset += length
    end
    worst
end

finite = all(isfinite, x) && all(isfinite, s) && all(isfinite, z)
slack = b - A * x
rp = finite ? norm(slack - s, Inf) : T(NaN)
dist_s = finite ? cone_block_distance(s, data["cones"]) : T(NaN)
rp_x = finite ? cone_block_distance(slack, data["cones"]) : T(NaN)
rd = finite ? norm(P_operator * x + A' * z + q, Inf) : T(NaN)
dist_z = finite ? cone_block_distance(z, data["cones"]; dual=true) : T(NaN)
px = finite ? P_operator * x : T[]
objective = finite ? dot(q, x) + dot(x, px) / 2 : T(NaN)
dual_objective = finite ? -dot(b, z) - dot(x, px) / 2 : T(NaN)
gap = finite ? abs(objective - dual_objective) /
    (one(T) + abs(objective)) : T(NaN)
tol = sqrt(eps(T))
tol_feas = tol * (one(T) + norm(b, Inf))
tol_dual = tol * (one(T) + norm(q, Inf))
status = string(get(result, "status", "Error"))
optimal = status in ("Solved", "Optimal", "solved", "optimal")
passed = optimal && finite && all(v -> isfinite(v) && v <= tol_feas, (rp, dist_s, rp_x)) &&
    all(v -> isfinite(v) && v <= tol_dual, (rd, dist_z)) && isfinite(gap) && gap <= tol
safe(v) = v isa AbstractFloat ? (isfinite(v) ? string(v) : nothing) : v
println(JSON.json(Dict("pass" => passed, "finite" => finite, "status" => status,
    "solver_optimal" => optimal, "r_p" => safe(rp), "dist_K_s" => safe(dist_s),
    "r_p_x" => safe(rp_x), "r_d" => safe(rd), "dist_Kstar_z" => safe(dist_z),
    "gap" => safe(gap), "tol_feas" => safe(tol_feas), "tol_dual" => safe(tol_dual),
    "objective" => safe(objective), "dual_objective" => safe(dual_objective),
    "precision_bits" => bits, "oracle_precision_bits" => bits == 53 ? 53 : precision(one(T)))))
passed || exit(1)
