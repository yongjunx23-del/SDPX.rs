# Preprocessing must preserve public coordinates and the prepared-update contract.
function preprocessing_diagonal_sdp(::Type{T}) where T
    # Four disconnected vertices force a proper chordal decomposition. Packed
    # upper-triangle diagonal positions are 1,3,6,10; min t with t >= 1,2,3,4.
    rows=[1,3,6,10]
    A=sparse(rows,ones(Int,4),fill(-one(T),4),10,1)
    b=zeros(T,10)
    b[rows]=-T.(1:4)
    (q=T[1],A=A,b=b,cones=[PSDTriangleConeT(4)])
end

function preprocessing_structure_changed(q,A,b,cones,settings,reason)
    bits=settings.precision_bits
    p=SDPX._standard_program(spzeros(eltype(q),length(q),length(q)),q,A,b,cones,eltype(q),bits)
    h=SDPX._prepare_program(p;settings,reusable=false)
    try
        # The existing update API rejects genuinely transformed structures.
        # No numerical solve is entered when this rejection occurs.
        @test_throws reason SDPX._solve_prepared_program!(h,p)
    finally
        close(h)
    end
end

@testset "Default preprocessing and explicit opt-out" begin
    defaults=Settings()
    @test defaults.equilibration===:ruiz
    @test defaults.presolve_enable && defaults.chordal_decomposition_enable
    @test SDPX._settings_storage(defaults).descriptor.preprocessing_flags==7
    native=Ref{SDPX.CSettings}()
    SDPX._abi_error(ccall(SDPX._sym(:sdpx_default_settings),Cint,(Ref{SDPX.CSettings},),native))
    @test native[].preprocessing_flags==7
    @test native[].abi_version==SDPX._ABI_VERSION
    off=Settings(equilibration=:off,presolve_enable=false,chordal_decomposition_enable=false)
    @test SDPX._settings_storage(off).descriptor.preprocessing_flags==0
    for s in (defaults,off)
        q=[1.0]; A=sparse(reshape([-1.0],1,1)); b=[-2.0]
        r=solve(q,A,b,[NonnegativeConeT(1)];settings=s)
        @test is_optimal(r)
        @test r.x≈[2.0] atol=1e-7
        @test r.info.equilibration===s.equilibration
        @test r.info.presolve_enable==s.presolve_enable
        @test r.info.chordal_decomposition_enable==s.chordal_decomposition_enable
        @test norm(A*r.x+r.s-b,Inf)<1e-7
        @test norm(q+A'*r.z,Inf)<1e-7
    end
end

@testset "Chordal recovery in original coordinates" begin
    for T in (Float64,BigFloat)
        setprecision(BigFloat,256) do
            p=preprocessing_diagonal_sdp(T)
            s=Settings(T)
            tol=T===Float64 ? T(1e-7) : parse(T,"1e-30")
            r=solve(p.q,p.A,p.b,p.cones;settings=s)
            @test is_optimal(r)
            @test length(r.x)==1 && length(r.s)==10 && length(r.z)==10
            @test abs(r.x[1]-T(4))<tol
            @test abs(primal_objective(r)-T(4))<tol
            @test norm(p.A*r.x+r.s-p.b,Inf)<tol
            @test norm(p.q+p.A'*r.z,Inf)<tol
            @test abs(dot(r.s,r.z))<tol
            @test all(r.s[[1,3,6,10]].>=-tol)
            @test norm(r.s[[2,4,5,7,8,9]],Inf)<tol
            preprocessing_structure_changed(p.q,p.A,p.b,p.cones,s,"chordal decomposition is active")
        end
    end
end

@testset "Finite presolve row recovery" begin
    # Equal to the native infinity bound, rather than exceeding it: upstream
    # caps larger b values. A zero coefficient row makes reconstructed slack
    # exactly match the original finite input, without a large residual scale.
    q=[1.0]; A=sparse([1],[1],[-1.0],2,1); b=[-2.0,1e20]
    cones=[NonnegativeConeT(2)]
    s=Settings(chordal_decomposition_enable=false)
    preprocessing_structure_changed(q,A,b,cones,s,"presolver is active")
    r=solve(q,A,b,cones;settings=s)
    @test is_optimal(r)
    @test length(r.x)==1 && length(r.s)==2 && length(r.z)==2
    @test abs(r.x[1]-2)<1e-7
    @test norm(A*r.x+r.s-b,Inf)<1e-7
    @test norm(q+A'*r.z,Inf)<1e-7
    @test r.s[2]==b[2] && iszero(r.z[2])
end

@testset "Prepared Ruiz updates retain caller settings" begin
    caller=Settings()
    q=[1.0]; A=sparse(reshape([-100.0],1,1)); b=[-200.0]
    p=prepare(q,A,b,[NonnegativeConeT(1)];settings=caller)
    try
        @test caller.equilibration===:ruiz
        @test caller.presolve_enable && caller.chordal_decomposition_enable
        @test p.handle.settings !== caller
        @test p.handle.settings.equilibration===:ruiz
        @test !p.handle.settings.presolve_enable && !p.handle.settings.chordal_decomposition_enable
        for (qnew,bnew,xexpected) in (([2.0],[-300.0],3.0),([3.0],[-400.0],4.0))
            r=solve!(p;q=qnew,b=bnew)
            @test is_optimal(r)
            @test abs(r.x[1]-xexpected)<1e-7
            @test abs(primal_objective(r)-qnew[1]*xexpected)<1e-7
            @test norm(A*r.x+r.s-bnew,Inf)<1e-6
            @test norm(qnew+A'*r.z,Inf)<1e-7
            @test r.info.equilibration===:ruiz
            @test !r.info.presolve_enable && !r.info.chordal_decomposition_enable
        end
        @test caller.presolve_enable && caller.chordal_decomposition_enable
    finally
        close(p)
    end
end
