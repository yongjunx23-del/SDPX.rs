module SDPX
using LinearAlgebra, SparseArrays, Base.Threads, Libdl
const _precision_lock=ReentrantLock()
include("cones.jl")
for file in ("domains", "refs", "types", "storage", "model", "affine", "constraints", "starts")
    include("frontend/" * file * ".jl")
end
include("outputs.jl")
include("settings.jl")
include("results.jl")
include("bridge.jl")
include("sampled.jl")
for file in ("compile", "result", "api", "legacy")
    include("frontend/" * file * ".jl")
end
import MathOptInterface
const MOI=MathOptInterface
include("moi/optimizer.jl")
Base.close(p::PreparedFrontend)=close(p.handle)
Base.isopen(p::PreparedFrontend)=isopen(p.handle)
Base.close(p::PreparedLegacy)=close(p.frontend)
Base.isopen(p::PreparedLegacy)=isopen(p.frontend)
export Model, variable!, constraint!, objective!, Minimize, Maximize
export Settings, Limits, Tolerances, Outputs
export SampledBlock, SampledProgram, sampled_program
export ZeroCone, Nonnegative, Nonpositive, Reals, LorentzCone, RotatedLorentzCone
export PSDCone, ExponentialCone, PowerCone
export optimize!, solve, prepare, solve!, solve_conic, execution_plan
export status, value, dual, dual_slack, primal_objective, dual_objective
export objective_value, dual_objective_value, primal_residual, dual_residual, relative_gap
export iterations, solve_time, is_optimal, is_primal_infeasible, is_dual_infeasible
export primal_status, dual_status, termination_status, diagnostics, termination
export num_variables, num_constraints, variable_by_name, constraint_by_name
export variable_names, constraint_names
export ZeroConeT, NonnegativeConeT, SecondOrderConeT, PSDTriangleConeT
export ExponentialConeT, PowerConeT, GenPowerConeT, Optimizer
end
