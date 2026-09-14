# Reuse the existing original-coordinate oracle and fresh-solve protocol.
include(joinpath(@__DIR__, "..", "float64", "sdpx_runner.jl"))
# Microsecond rounding can manufacture percentage gains on small instances.
# Only timing serialization changes; all mathematical gates remain identical.
_jf6(x::Real) = isfinite(Float64(x)) ? JSON.json(Float64(x)) : "null"
main()
