using Test, SDPX, SparseArrays, LinearAlgebra
import MathOptInterface as MOI

@testset "ABI layouts and precision contract" begin
    @test sizeof(SDPX.CScalars)==32
    @test sizeof(SDPX.CCsc)==72
    @test sizeof(SDPX.CCone)==48
    @test sizeof(SDPX.CSettings)==80
    @test sizeof(SDPX.CInfo)==104
    @test SDPX._settings_storage(Settings()).descriptor.abi_version==3
    @test_throws ErrorException SDPX._check_abi(1,80,80)
    @test_throws ErrorException SDPX._check_abi(2,80,80)
    @test_throws ErrorException SDPX._check_abi(3,72,80)
    @test_throws ArgumentError Settings(kkt_form=:unknown)
    for (form,code) in ((:auto,0),(:augmented,1),(:condensed,2))
        @test SDPX._settings_storage(Settings(kkt_form=form)).descriptor.kkt_form==code
    end
    settings=Settings(kkt_form=:augmented)
    @test_throws ArgumentError (settings.kkt_form=:unknown)
    @test settings.kkt_form==:augmented
    @test_throws ArgumentError Model(BigFloat;precision_bits=412)
    @test_throws ArgumentError Settings(Float64;precision_bits=128)
    @test_throws ArgumentError Settings(tolerances=Tolerances(primal=1e-9,dual=1e-8))
    @test !isdefined(SDPX,:SolverCore)
end

include("moi_contracts.jl")
include("preprocessing.jl")
include("sampled.jl")

@testset "Direct LP and reusable handle" begin
    q=[1.0]; A=sparse(reshape([-1.0],1,1)); b=[-1.0]; cones=[NonnegativeConeT(1)]
    r=solve_conic(q,A,b,cones;return_result=true)
    @test is_optimal(r)
    @test r.x≈[1.0] atol=1e-7
    @test norm(A*r.x+r.s-b,Inf)<1e-7
    @test norm(q+A'*r.z,Inf)<1e-7
    @test solve_time(r)>=0
    p=prepare(q,A,b,cones)
    r1=solve!(p;b=[-2.0])
    r2=solve!(p;q=[2.0])
    @test is_optimal(r1) && is_optimal(r2)
    @test r2.x≈[2.0] atol=1e-7
    @test primal_objective(r2)≈4.0 atol=1e-7
    @test_throws DimensionMismatch solve!(p;b=[1.0,2.0])
    @test is_optimal(solve!(p))
    @test r1.x≈[2.0] atol=1e-7
    close(p); close(p)
    @test !isopen(p)
    @test_throws ArgumentError solve!(p)
    @test_throws ArgumentError solve!(p;q=[3.0])
end

@testset "SOCP and PSD original coordinates" begin
    # min t subject to (t, 3, 4) in SOC.
    A=sparse(reshape([-1.0,0,0],3,1)); b=[0.0,3,4]
    r=solve([1.0],A,b,[SecondOrderConeT(3)])
    @test is_optimal(r)
    @test r.x[1]≈5.0 atol=1e-7
    @test norm(A*r.x+r.s-b,Inf)<1e-7
    model=Model(Float64)
    X=variable!(model,:X,2,2;domain=PSDCone())
    c=constraint!(model,:offdiag,X[1,2]-1,ZeroCone())
    objective!(model,Minimize(),X[1,1]+X[2,2])
    result=optimize!(model)
    @test is_optimal(result)
    @test value(result,X)≈ones(2,2) atol=2e-6
    @test abs(primal_objective(result)-2)<1e-7
    @test abs(dual(result,c)[1]-2)<1e-6
    @test eigmin(Symmetric(dual_slack(result,X)))>=-1e-7
    @test abs(dot(value(result,X),dual_slack(result,X)))<1e-6
end

@testset "Model preparation and objective recovery" begin
    model=Model(Float64)
    x=variable!(model,:x,1)
    constraint!(model,:bound,3-x[1],Nonnegative())
    objective!(model,Maximize(),2x[1]+7)
    p=prepare(model)
    r=solve!(p)
    @test is_optimal(r)
    @test primal_objective(r)≈13.0 atol=1e-7
    @test value(r,x)[1]≈3.0 atol=1e-7
    close(p)
    @test_throws ArgumentError solve!(p)
end

@testset "MOI LP and PSD conversion" begin
    model=MOI.Utilities.Model{Float64}()
    x=MOI.add_variable(model)
    MOI.add_constraint(model,x,MOI.GreaterThan(2.0))
    MOI.set(model,MOI.ObjectiveSense(),MOI.MIN_SENSE)
    MOI.set(model,MOI.ObjectiveFunction{MOI.VariableIndex}(),x)
    opt=Optimizer(); MOI.set(opt,MOI.Silent(),true)
    mapping=MOI.copy_to(opt,model); MOI.optimize!(opt)
    @test MOI.get(opt,MOI.TerminationStatus())==MOI.OPTIMAL
    @test MOI.get(opt,MOI.VariablePrimal(),mapping[x])≈2.0 atol=1e-7
    @test MOI.get(opt,MOI.ObjectiveValue())≈2.0 atol=1e-7
    MOI.empty!(model)
    xs=MOI.add_variables(model,3)
    MOI.add_constraint(model,MOI.VectorOfVariables(xs),MOI.PositiveSemidefiniteConeTriangle(2))
    MOI.add_constraint(model,xs[2],MOI.EqualTo(1.0))
    f=MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0,xs[1]),MOI.ScalarAffineTerm(1.0,xs[3])],0.0)
    MOI.set(model,MOI.ObjectiveSense(),MOI.MIN_SENSE)
    MOI.set(model,MOI.ObjectiveFunction{typeof(f)}(),f)
    mapping=MOI.copy_to(opt,model); MOI.optimize!(opt)
    @test MOI.get(opt,MOI.TerminationStatus())==MOI.OPTIMAL
    @test MOI.get(opt,MOI.ObjectiveValue())≈2.0 atol=1e-7
    @test MOI.get(opt,MOI.VariablePrimal(),mapping[xs[2]])≈1.0 atol=1e-7
end

# Every requested precision is a required acceptance leg, never a skip or f64 fallback.
@testset "BigFloat decimal transport at $bits bits" for bits in SDPX.SUPPORTED_PRECISIONS
    setprecision(BigFloat,bits) do
        target=1+BigFloat(2)^(-70)
        tol=BigFloat(2)^(-min(bits÷3,120))
        settings=Settings(BigFloat;precision_bits=bits,tolerances=Tolerances(BigFloat;primal=tol,dual=tol,gap=tol))
        p=prepare(BigFloat[0],sparse(reshape(BigFloat[1],1,1)),BigFloat[target],[ZeroConeT(1)];settings)
        r=solve!(p)
        @test is_optimal(r)
        @test precision(r.x[1])==bits
        @test abs(r.x[1]-target)<BigFloat(2)^(-min(bits÷2,150))
        @test r.x[1]!=BigFloat(Float64(r.x[1]))
        @test execution_plan(r).precision_bits==bits
        updated=target+BigFloat(2)^(-71)
        r2=solve!(p;b=BigFloat[updated])
        @test is_optimal(r2)
        @test abs(r2.x[1]-updated)<BigFloat(2)^(-min(bits÷2,150))
        @test r.x[1]!=r2.x[1]
        close(p)
        @test_throws ArgumentError solve!(p)
    end
end

@testset "Nonoptimal statuses and external ray checks" begin
    A=sparse(reshape([-1.0,1.0],2,1)); b=[-1.0,0.0]
    infeasible=solve([0.0],A,b,[NonnegativeConeT(2)])
    @test is_primal_infeasible(infeasible)
    @test !is_optimal(infeasible)
    @test minimum(infeasible.z)>=-1e-8
    @test norm(A'*infeasible.z,Inf)<1e-7
    @test dot(b,infeasible.z)<-1e-4
    A=sparse(reshape([-1.0],1,1)); b=[0.0]; q=[-1.0]
    unbounded=solve(q,A,b,[NonnegativeConeT(1)])
    @test is_dual_infeasible(unbounded)
    @test !is_optimal(unbounded)
    @test norm(A*unbounded.x+unbounded.s,Inf)<1e-7
    @test minimum(unbounded.s)>=-1e-8
    @test dot(q,unbounded.x)<-1e-4
    limited=solve([1.0],A,[-1.0],[NonnegativeConeT(1)];settings=Settings(limits=Limits(iterations=0)))
    @test status(limited)==:iteration_limit
    @test !is_optimal(limited)
    timed=solve([1.0],A,[-1.0],[NonnegativeConeT(1)];settings=Settings(limits=Limits(time=0)))
    @test status(timed)==:time_limit
    @test !is_optimal(timed)
end

@testset "GC ownership and prepared result snapshots" begin
    p=prepare([1.0],sparse(reshape([-1.0],1,1)),[-1.0],[NonnegativeConeT(1)])
    GC.gc(true)
    first=solve!(p)
    for k in 2:5
        GC.gc(true)
        result=solve!(p;b=[-Float64(k)])
        @test is_optimal(result)
        @test result.x[1]≈k atol=1e-7
        @test first.x[1]≈1.0 atol=1e-7
    end
    finalize(p.handle)
    @test !isopen(p)
    close(p)
    @test_throws ArgumentError solve!(p)
end

@testset "BigFloat SOC and SDP at $bits bits" for bits in SDPX.SUPPORTED_PRECISIONS
    setprecision(BigFloat,bits) do
        tol=BigFloat(2)^(-bits÷3)
        settings=Settings(BigFloat;precision_bits=bits,tolerances=Tolerances(BigFloat;primal=tol,dual=tol,gap=tol))
        A=sparse(reshape(BigFloat[-1,0,0],3,1)); b=BigFloat[0,3,4]
        soc=solve(BigFloat[1],A,b,[SecondOrderConeT(3)];settings)
        @test is_optimal(soc)
        @test abs(soc.x[1]-5)<100tol
        @test norm(A*soc.x+soc.s-b,Inf)<100tol
        @test norm(BigFloat[1]+A'*soc.z,Inf)<100tol
        @test soc.s[1]-norm(soc.s[2:end])>=-100tol
        @test all(v->precision(v)==bits,soc.x)
        model=Model(BigFloat;precision_bits=bits)
        X=variable!(model,:X,2,2;domain=PSDCone())
        constraint!(model,:offdiag,X[1,2]-1,ZeroCone())
        objective!(model,Minimize(),X[1,1]+X[2,2])
        result=optimize!(model;settings)
        @test is_optimal(result)
        matrix=value(result,X)
        @test abs(matrix[1,2]-1)<100tol
        @test abs(primal_objective(result)-2)<100tol
        @test matrix[1,1]>=-100tol && matrix[2,2]>=-100tol
        @test det(matrix)>=-100tol
        @test all(v->precision(v)==bits,matrix)
        @test primal_residual(result)<100tol
        @test dual_residual(result)<100tol
        @test relative_gap(result)<100tol
    end
end

@testset "Float64 backend thread budget $nt" for nt in (1,2,4,8)
    settings=Settings(limits=Limits(threads=nt))
    cases=(
        (ones(64),-sparse(I,64,64),ones(64),[NonnegativeConeT(64)],-64.0),
        ([1.0],sparse(reshape([-1.0,0,0],3,1)),[0.0,3,4],[SecondOrderConeT(3)],5.0),
        ([1.0],sparse(reshape([-1.0,0,-1.0],3,1)),[0.0,sqrt(2.0),0.0],[PSDTriangleConeT(2)],1.0),
    )
    for (q,A,b,cones,expected) in cases
        r=solve_conic(q,A,b,cones;settings,return_result=true)
        @test is_optimal(r)
        @test abs(primal_objective(r)-expected)<1e-6
        @test norm(A*r.x+r.s-b,Inf)<1e-7
        @test norm(q+A'*r.z,Inf)<1e-7
        # These tiny sparse systems select QDLDL even with a larger budget.
        @test r.info.backend_threads==1
        @test execution_plan(r).backend_threads==1
        @test execution_plan(r).cone_threads==r.info.cone_threads==1
        @test execution_plan(r).kkt_form in (:augmented,:condensed)
        @test execution_plan(r).factorization in (:qdldl,:condensed_qdldl)
    end
end

@testset "High precision cone workers and original-coordinate results" begin
    setprecision(BigFloat,128) do
        # Four independent PSD16 blocks exceed the structural cone-pool cutoff.
        # Each block is x_j*I with one off-diagonal target j, hence x_j >= j.
        order=16; blocks=4; packed=order*(order+1)÷2
        rows=Int[]; columns=Int[]
        for block in 1:blocks, col in 1:order
            push!(rows,(block-1)*packed+col*(col+1)÷2)
            push!(columns,block)
        end
        A=sparse(rows,columns,BigFloat[-1 for _ in rows],blocks*packed,blocks)
        b=BigFloat[0 for _ in 1:blocks*packed]
        for block in 1:blocks
            b[(block-1)*packed+2]=sqrt(BigFloat(2))*block
        end
        q=BigFloat[1 for _ in 1:blocks]
        cones=[PSDTriangleConeT(order) for _ in 1:blocks]
        tol=BigFloat(2)^(-42)
        results=map((1,4)) do nt
            # Keep the PSD16 blocks intact to exercise the cone worker pool.
            # Default chordal recovery is covered separately in preprocessing.jl.
            settings=Settings(BigFloat;precision_bits=128,kkt_form=:condensed,
                chordal_decomposition_enable=false,
                limits=Limits(threads=nt),tolerances=Tolerances(BigFloat;primal=tol,dual=tol,gap=tol))
            r=solve_conic(q,A,b,cones;settings,return_result=true)
            @test is_optimal(r)
            @test all(v->precision(v)==128,vcat(r.x,r.s,r.z))
            @test maximum(abs.(r.x.-BigFloat[1,2,3,4]))<100tol
            @test norm(A*r.x+r.s-b,Inf)<100tol
            @test norm(q+A'*r.z,Inf)<100tol
            @test abs(dot(q,r.x)+dot(b,r.z))/(1+abs(dot(q,r.x)))<100tol
            # Gershgorin lower bounds suffice to certify PSD for these blocks;
            # no Float64 conversion or optional BigFloat eigensolver is needed.
            for vector in (r.s,r.z), block in 1:blocks
                matrix=BigFloat[0 for _ in 1:order, _ in 1:order]
                index=(block-1)*packed
                for col in 1:order, row in 1:col
                    index+=1
                    v=row==col ? vector[index] : vector[index]/sqrt(BigFloat(2))
                    matrix[row,col]=deepcopy(v); matrix[col,row]=deepcopy(v)
                end
                @test minimum(matrix[row,row]-sum(abs(matrix[row,col]) for col in 1:order if col!=row)
                    for row in 1:order)>=-100tol
            end
            plan=execution_plan(r)
            @test plan.precision_bits==128
            @test plan.kkt_form==:condensed
            # The four independent PSD16 Schur blocks are arrow-eligible at
            # MPFR; qdldl remains the fallback when the structure declines.
            @test plan.factorization in (:condensed_qdldl,:condensed_arrow)
            @test plan.backend_threads==1
            @test plan.cone_threads==nt
            r
        end
        @test norm(results[1].x-results[2].x,Inf)<100tol
        @test norm(results[1].s-results[2].s,Inf)<100tol
        @test norm(results[1].z-results[2].z,Inf)<100tol
    end
end
