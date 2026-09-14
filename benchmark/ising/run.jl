# Same sampled SDPB primal as the earlier Ising campaign; audits are external.
using SDPX
startswith(realpath(pathof(SDPX)),realpath(ENV["SDPX_FROZEN_ROOT"])*"/") || error("SDPX is outside frozen source")
length(ARGS)==5 || error("usage: run.jl INPUT MODE RUN_DIR THREADS WARMED_REPETITIONS")
const input_dir, mode, run_dir, threads_text, warmed_text = ARGS
const width=parse(Int,threads_text)
const warmed_repetitions=parse(Int,warmed_text)
width in (1,2,4,8) || error("unsupported benchmark thread budget")
0<=warmed_repetitions<=3 || error("unbounded repetitions")
const tolerance_text = "1e-30"
include("audit_helpers.jl")
const sampled_input_text=get(ENV,"SDPX_ISING_SAMPLED","0")
sampled_input_text in ("0","1") || error("SDPX_ISING_SAMPLED must be 0 or 1")
const use_sampled_input=sampled_input_text=="1" && mode=="sdpx"
const input_route=mode!="sdpx" ? "sdpb_output_audit" : use_sampled_input ? "sampled_factors" : "materialized_csc"
if use_sampled_input
    include("sampled_adapter.jl")
end
function save(name,x)
    open(joinpath(run_dir,name),"w") do io; JSON.print(io,stringify(x),2); end
end
setprecision(BigFloat,512) do
    expected=sort(parse.(Int,split(ENV["SDPX_RANK_CPUS"],',')))
    line=only(filter(x->startswith(x,"Cpus_allowed_list:"),readlines("/proc/self/status")))
    actual=Int[]
    for token in split(strip(split(line,':';limit=2)[2]),',')
        v=parse.(Int,split(token,'-')); append!(actual,first(v):last(v))
    end
    sort(actual)==expected || error("Julia affinity differs from reserved subset")
    Threads.nthreads()==width || error("Julia thread budget mismatch")
    BLAS.get_num_threads()==1 || error("BLAS oversubscription")
    reference=JSON.parsefile(ENV["SDPX_REFERENCE_AUDIT"])
    reference["accepted"]===true || error("reference audit was not accepted")
    ref_objective=parse(BigFloat,reference["objective"])
    isfinite(ref_objective) || error("nonfinite reference objective")
    t=time_ns(); sampled=read_sampled_sdp(input_dir;T=BigFloat,bits=512)
    conic=compile_sampled_primal(sampled)
    save("input.json",Dict("input_sha256"=>sampled.input_sha256,"identity_sha256"=>sampled.identity_sha256,
        "mapping_sha256"=>conic.mapping_sha256,"n"=>length(conic.q),"m"=>length(conic.b),
        "equalities"=>conic.num_equalities,"psd_dimensions"=>[g.dimension for g in conic.grams],
        "compile_seconds"=>(time_ns()-t)/1e9,"julia_version"=>string(VERSION),
        "julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),
        "affinity"=>actual,"sdpx_path"=>realpath(pathof(SDPX)),
        "input_route"=>input_route,"requested_factor_input"=>sampled_input_text=="1",
        "sampled_adapter_sha256"=>(use_sampled_input ? bytes2hex(sha256(read(joinpath(@__DIR__,"sampled_adapter.jl")))) : nothing)))
    metrics=("primal","dual","gap","primal_psd","dual_psd","mapping_relative")
    function checked(x,s,z)
        a=audit(conic,x,s,z); a["mapping_relative"]=mapping_gate_value(a["mapping"])
        a["reference_objective_agreement"]=abs(a["objective"]-ref_objective)/max(one(BigFloat),abs(a["objective"]),abs(ref_objective))
        a["accepted"]=all(isfinite(a[k]) && a[k]<=parse(BigFloat,tolerance_text) for k in metrics) &&
            isfinite(a["reference_objective_agreement"]) && a["reference_objective_agreement"]<=parse(BigFloat,tolerance_text)
        a
    end
    if mode=="sdpx"
        tol=parse(BigFloat,"1e-42")
        settings=SDPX.Settings(BigFloat;precision_bits=512,
            tolerances=SDPX.Tolerances(BigFloat;primal=tol,dual=tol,gap=tol),
            limits=SDPX.Limits(iterations=1000,time=850,threads=width),kkt_form=:auto,verbose=true)
        cones=sdpx_cones(conic,SDPX)
        for label in vcat(["first"],["warmed-$(i)" for i in 1:warmed_repetitions])
            println("SDPX $label start"); flush(stdout)
            run=@timed if use_sampled_input
                program=SDPXSampledAdapter.sdpx_sampled_program(conic;settings)
                SDPX.solve_conic(program;settings,return_result=true)
            else
                SDPX.solve_conic(conic.q,conic.A,conic.b,cones;settings,return_result=true)
            end
            r=run.value
            save("$label-raw.json",Dict("status"=>string(SDPX.status(r)),"x"=>r.x,"s"=>r.s,"z"=>r.y,
                "iterations"=>r.iterations,"api_seconds"=>run.time,"julia_bytes"=>run.bytes,
                "solver_seconds"=>r.solve_time,"info"=>r.info,"execution_plan"=>SDPX.execution_plan(r),
                "input_route"=>input_route,"requested_factor_input"=>use_sampled_input,
                "prepare_entrypoint"=>(use_sampled_input ? "sdpx_prepare_sampled" : "sdpx_prepare"),
                "actual_solver_name"=>string(r.info.factorization),
                "api_timing_scope"=>(use_sampled_input ? "fresh factor adaptation + public solve" : "fresh public CSC solve"),
                "compile_seconds"=>(hasproperty(run,:compile_time) ? run.compile_time : nothing),
                "recompile_seconds"=>(hasproperty(run,:recompile_time) ? run.recompile_time : nothing)))
            a=checked(r.x,r.s,r.y); a["optimal"]=SDPX.is_optimal(r); a["accepted"] &= a["optimal"]
            save("$label-audit.json",a)
            a["accepted"] || error("SDPX external audit failed")
            plan=SDPX.execution_plan(r)
            expected_factorization=use_sampled_input ? "condensed_sampled_qdldl" : "condensed_qdldl"
            plan.precision_bits==512 && plan.cone_threads==width && plan.backend_threads==1 &&
                plan.kkt_form==:condensed && string(plan.factorization)==expected_factorization ||
                error("actual solver plan differs from the qualified condensed MPFR route")
        end
    elseif mode=="sdpb"
        out=joinpath(run_dir,"sdpb-out")
        lambda=zeros(BigFloat,length(conic.q))
        for (block,cols) in zip(sampled.blocks,conic.block_columns)
            lambda[cols]=vec(read_matrix(joinpath(out,"x_$(block.block_index).txt")))
        end
        y=vec(read_matrix(joinpath(out,"y.txt")))
        Xmats=[read_symmetric_matrix(joinpath(out,"X_matrix_$(2*g.block_index+g.parity).txt")) for g in conic.grams]
        Ymats=[read_symmetric_matrix(joinpath(out,"Y_matrix_$(2*g.block_index+g.parity).txt")) for g in conic.grams]
        r=pack_sampled_primal_solution(conic,lambda,y,Xmats,Ymats)
        a=checked(r.x,r.s,r.z)
        a["optimal"]=occursin("found primal-dual optimal solution",read(joinpath(out,"out.txt"),String))
        a["accepted"] &= a["optimal"]; a["output_skew"]=maximum_output_skew[]
        save("audit.json",a)
        a["accepted"] || error("SDPB external audit failed")
    else
        error("unknown mode")
    end
end
