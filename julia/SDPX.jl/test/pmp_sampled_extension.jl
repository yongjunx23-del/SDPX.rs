# Optional integration. Run in an environment with the candidate SDPX and
# PMP2SDP developed, after resolving extension metadata. Not in core runtests.
using Test, SDPX, PMP2SDP, LinearAlgebra, SparseArrays

@testset "PMP2SDP factor-authoritative extension" begin
    @test Base.get_extension(SDPX,:SDPXPMP2SDPExt)!==nothing
    for (T,bits) in ((Float64,53),(BigFloat,512))
        c=setprecision(BigFloat,512) do
            # Production compiler fixture: max y subject to 1-y >= 0.
            pmp=PMP2SDP.StrictPMP(PMP2SDP.monomial_basis(T,0),1)
            PMP2SDP.set_objective!(pmp,zero(T),T[1])
            PMP2SDP.add_block!(pmp,PMP2SDP.HalfLine(zero(T)),
                [reshape(T[1],1,1)],[[reshape(T[-1],1,1)]])
            PMP2SDP.compile_sampled_primal(pmp)
        end
        original_A=deepcopy(c.A)
        p=setprecision(BigFloat,256) do
            program=SDPX.sampled_program(c)
            @test precision(BigFloat)==256
            @test program.precision_bits==bits
            @test eltype(program.q)===T
            if T===BigFloat
                @test all(precision(v)==512 for b in program.blocks for v in b.basis)
                sources=[Q for g in c.groups for Q in g.bilinear_bases if size(Q,1)>0]
                for (block,Q) in zip(program.blocks,sources)
                    @test isempty(intersect(Set(v.d for v in block.basis),Set(v.d for v in Q)))
                    @test length(Set(v.d for v in block.basis))==length(block.basis)
                end
            end
            program
        end
        @test c.A==original_A
        @test p.A[1:c.num_decision_variables,:]==c.A[1:c.num_decision_variables,:]
        @test all(iszero,p.A[vcat(c.gram_rows...),:])
        @test length(p.blocks)==length(c.gram_rows)
        settings=SDPX.Settings(T;precision_bits=bits)
        explicit=SDPX.sampled_program(c;settings,T)
        @test explicit.precision_bits==bits && explicit.q==p.q
        setprecision(BigFloat,512) do
            result=SDPX.solve(p;settings)
            tol=T===Float64 ? T(1e-8) : sqrt(eps(BigFloat))
            @test SDPX.is_optimal(result)
            @test norm(c.A*result.x+result.s-c.b,Inf)/(1+norm(c.b,Inf))<=tol
            @test norm(c.q+c.A'*result.z,Inf)/(1+norm(c.q,Inf))<=tol
            @test abs(dot(c.q,result.x)+dot(c.b,result.z))/(1+abs(dot(c.q,result.x)))<=tol
            recovered=PMP2SDP.recover_pmp_solution(c,result.x,result.z;s=result.s)
            @test abs(recovered.pmp_objective-1)/2<=tol
            @test result.info.precision_bits==bits
        end
    end
end
