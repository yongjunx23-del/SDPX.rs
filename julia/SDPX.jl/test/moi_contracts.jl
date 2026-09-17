# End-to-end MOI contracts; independent analytic answers, not internal row order.
function moi_run(model)
    opt = Optimizer()
    MOI.set(opt, MOI.Silent(), true)
    mapping = MOI.copy_to(opt, model)
    MOI.optimize!(opt)
    return opt, mapping
end
function moi_objective!(m, f; sense=MOI.MIN_SENSE)
    MOI.set(m, MOI.ObjectiveSense(), sense)
    MOI.set(m, MOI.ObjectiveFunction{typeof(f)}(), f)
end

@testset "MOI scalar constraint values and sense-independent duals" begin
    for (set, sense, expected_x, expected_dual) in (
        (MOI.GreaterThan(2.0),MOI.MIN_SENSE,2.0,1.0),
        (MOI.LessThan(4.0),MOI.MAX_SENSE,4.0,-1.0),
        (MOI.EqualTo(3.0),MOI.MIN_SENSE,3.0,1.0),
        (MOI.EqualTo(3.0),MOI.MAX_SENSE,3.0,-1.0),
        (MOI.Interval(2.0,4.0),MOI.MIN_SENSE,2.0,1.0),
        (MOI.Interval(2.0,4.0),MOI.MAX_SENSE,4.0,-1.0))
        for affine in (false,true)
            m=MOI.Utilities.Model{Float64}(); x=MOI.add_variable(m)
            f=affine ? MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(2.0,x)],1.0) : x
            c=MOI.add_constraint(m,f,set)
            moi_objective!(m,f;sense)
            o,idx=moi_run(m)
            @test MOI.get(o,MOI.TerminationStatus())==MOI.OPTIMAL
            @test MOI.get(o,MOI.ConstraintPrimal(),idx[c])≈expected_x atol=1e-6
            @test MOI.get(o,MOI.ConstraintDual(),idx[c])≈expected_dual atol=1e-6
            @test_throws MOI.ResultIndexBoundsError MOI.get(o,MOI.ConstraintDual(2),idx[c])
            @test_throws MOI.InvalidIndex MOI.get(o,MOI.ConstraintPrimal(),typeof(idx[c])(99999))
            MOI.empty!(o)
            @test isempty(o.constraint_rows) && isempty(o.var_to_col)
            @test MOI.get(o,MOI.ResultCount())==0
        end
    end
end

@testset "MOI PSD dual trace scaling and vector affine constants" begin
    for affine in (false,true), sense in (MOI.MIN_SENSE,MOI.MAX_SENSE)
        m=MOI.Utilities.Model{Float64}(); x=MOI.add_variables(m,3)
        f=affine ? MOI.VectorAffineFunction([
            MOI.VectorAffineTerm(i,MOI.ScalarAffineTerm(1.0,x[i])) for i in 1:3], [0.0,1.0,0.0]) : MOI.VectorOfVariables(x)
        c=MOI.add_constraint(m,f,MOI.PositiveSemidefiniteConeTriangle(2))
        eq=MOI.add_constraint(m,x[2],MOI.EqualTo(affine ? 0.0 : 1.0))
        sign=sense==MOI.MIN_SENSE ? 1.0 : -1.0
        obj=MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(sign,x[1]),MOI.ScalarAffineTerm(sign,x[3])],0.0)
        moi_objective!(m,obj;sense)
        o,idx=moi_run(m)
        @test MOI.get(o,MOI.TerminationStatus())==MOI.OPTIMAL
        @test MOI.get(o,MOI.ConstraintPrimal(),idx[c])≈ones(3) atol=1e-6
        @test MOI.get(o,MOI.ConstraintDual(),idx[c])≈[1.0,-1.0,1.0] atol=1e-6
        @test MOI.get(o,MOI.ConstraintDual(),idx[eq])≈2.0 atol=1e-6
        d=MOI.get(o,MOI.ConstraintDual(),idx[c]); d[1]=999
        @test MOI.get(o,MOI.ConstraintDual(),idx[c])[1]≈1.0 atol=1e-6
    end
end

@testset "MOI vector zeros and infeasibility certificates" begin
    m=MOI.Utilities.Model{Float64}(); x=MOI.add_variable(m)
    f=MOI.VectorAffineFunction([MOI.VectorAffineTerm(1,MOI.ScalarAffineTerm(1.0,x))],[-2.0])
    c=MOI.add_constraint(m,f,MOI.Zeros(1)); moi_objective!(m,x)
    o,idx=moi_run(m)
    @test MOI.get(o,MOI.ConstraintPrimal(),idx[c])≈[0.0] atol=1e-6
    @test MOI.get(o,MOI.ConstraintDual(),idx[c])≈[1.0] atol=1e-6
    # A recession direction must omit the constant 5 from f(x).
    m=MOI.Utilities.Model{Float64}(); x=MOI.add_variable(m)
    f=MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0,x)],5.0)
    c=MOI.add_constraint(m,f,MOI.GreaterThan(0.0))
    moi_objective!(m,MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(-1.0,x)],0.0))
    o,idx=moi_run(m)
    @test MOI.get(o,MOI.PrimalStatus())==MOI.INFEASIBILITY_CERTIFICATE
    @test MOI.get(o,MOI.ConstraintPrimal(),idx[c])≈MOI.get(o,MOI.VariablePrimal(),idx[x])
    @test MOI.get(o,MOI.ConstraintPrimal(),idx[c])>0
    m=MOI.Utilities.Model{Float64}(); x=MOI.add_variable(m)
    lo=MOI.add_constraint(m,x,MOI.GreaterThan(2.0))
    hi=MOI.add_constraint(m,x,MOI.LessThan(1.0))
    moi_objective!(m,x)
    o,idx=moi_run(m)
    @test MOI.get(o,MOI.DualStatus())==MOI.INFEASIBILITY_CERTIFICATE
    dl=MOI.get(o,MOI.ConstraintDual(),idx[lo]); dh=MOI.get(o,MOI.ConstraintDual(),idx[hi])
    @test dl>0 && dh<0
    @test abs(dl+dh)<1e-6*max(abs(dl),abs(dh))
    @test 2dl+dh>0
end

@testset "MOI writable raw attributes and aliases" begin
    o=Optimizer()
    for (name,val) in (("limits",Limits()),("tolerances",Tolerances()))
        a=MOI.RawOptimizerAttribute(name)
        @test !MOI.supports(o,a)
        @test_throws MOI.UnsupportedAttribute MOI.set(o,a,val)
        @test_throws MOI.UnsupportedAttribute MOI.get(o,a)
    end
    for (name,val) in (("threads",2),("max_threads",3),("verbosity",0),("verbose",true))
        a=MOI.RawOptimizerAttribute(name)
        @test MOI.supports(o,a)
        MOI.set(o,a,val)
        @test MOI.get(o,a)==val
    end
    @test MOI.get(o,MOI.RawOptimizerAttribute("threads"))==3
    @test MOI.get(o,MOI.NumberOfThreads())==3
    MOI.set(o,MOI.Silent(),true)
    @test MOI.get(o,MOI.RawOptimizerAttribute("verbosity"))==0
    MOI.set(o,MOI.TimeLimitSec(),2.0)
    @test MOI.get(o,MOI.RawOptimizerAttribute("time_limit"))==2.0
    @test_throws ArgumentError MOI.set(o,MOI.RawOptimizerAttribute("threads"),0)
    @test MOI.get(o,MOI.NumberOfThreads())==3
    MOI.set(o,MOI.NumberOfThreads(),nothing)
    @test MOI.get(o,MOI.RawOptimizerAttribute("max_threads"))==1
    @test !haskey(o.options,"max_threads")
    MOI.set(o,MOI.NumberOfThreads(),2)
    @test MOI.get(o,MOI.RawOptimizerAttribute("threads"))==2
    @test_throws ArgumentError MOI.set(o,MOI.NumberOfThreads(),Int(typemax(UInt32))+1)
end

@testset "MOI quadratic objective convention, duplicate terms and senses" begin
    # H=[4 1;1 2], q=[-6,-5] => x=[1,2], objective with constant 9 is 1.
    for sense in (MOI.MIN_SENSE,MOI.MAX_SENSE)
        m=MOI.Utilities.Model{Float64}(); x=MOI.add_variables(m,2)
        s=sense==MOI.MIN_SENSE ? 1.0 : -1.0
        f=MOI.ScalarQuadraticFunction([
            MOI.ScalarQuadraticTerm(4s,x[1],x[1]),
            MOI.ScalarQuadraticTerm(2s,x[2],x[2]),
            MOI.ScalarQuadraticTerm(0.25s,x[1],x[2]),
            MOI.ScalarQuadraticTerm(0.75s,x[2],x[1])],
            [MOI.ScalarAffineTerm(-6s,x[1]),MOI.ScalarAffineTerm(-5s,x[2])],9s)
        moi_objective!(m,f;sense)
        o,idx=moi_run(m)
        @test MOI.supports(o,MOI.ObjectiveFunction{typeof(f)}())
        @test MOI.get(o,MOI.TerminationStatus())==MOI.OPTIMAL
        @test [MOI.get(o,MOI.VariablePrimal(),idx[v]) for v in x]≈[1.0,2.0] atol=1e-6
        @test MOI.get(o,MOI.ObjectiveValue())≈s atol=1e-6
        @test MOI.get(o,MOI.DualObjectiveValue())≈s atol=1e-6
        MOI.set(m,MOI.ObjectiveSense(),MOI.FEASIBILITY_SENSE)
        o,idx=moi_run(m)
        @test MOI.get(o,MOI.ObjectiveValue())≈0.0 atol=1e-8
    end
end

@testset "MOI exponential and power cone coverage" begin
    for affine in (false,true), power in (false,true)
        m=MOI.Utilities.Model{Float64}(); x=MOI.add_variables(m,3)
        set=power ? MOI.PowerCone(0.5) : MOI.ExponentialCone()
        f=affine ? MOI.VectorAffineFunction([
            MOI.VectorAffineTerm(i,MOI.ScalarAffineTerm(1.0,x[i])) for i in 1:3],zeros(3)) : MOI.VectorOfVariables(x)
        c=MOI.add_constraint(m,f,set)
        MOI.add_constraint(m,x[1],MOI.EqualTo(power ? 4.0 : 0.0))
        MOI.add_constraint(m,x[2],MOI.EqualTo(1.0))
        moi_objective!(m,x[3];sense=power ? MOI.MAX_SENSE : MOI.MIN_SENSE)
        o,idx=moi_run(m)
        @test MOI.get(o,MOI.TerminationStatus())==MOI.OPTIMAL
        @test MOI.get(o,MOI.ObjectiveValue())≈(power ? 2.0 : 1.0) atol=2e-6
        @test MOI.get(o,MOI.ConstraintPrimal(),idx[c])≈(power ? [4.0,1.0,2.0] : [0.0,1.0,1.0]) atol=2e-6
        @test length(MOI.get(o,MOI.ConstraintDual(),idx[c]))==3
    end
end
