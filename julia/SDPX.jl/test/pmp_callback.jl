# Optional integration: run in an isolated environment with this new SDPX and
# the existing PMP2SDP checkout developed as dependencies. No old SDPX is used.
using Test, SDPX, PMP2SDP, LinearAlgebra
cone_map(c)=c.kind===:zero ? SDPX.ZeroConeT(c.dim) :
    c.kind===:psd_triangle ? SDPX.PSDTriangleConeT(c.dim) : error("unsupported PMP cone $(c.kind)")
@testset "PMP2SDP solver-neutral callback $T" for T in (Float64,BigFloat)
    setprecision(BigFloat,256) do
        pmp=PMP2SDP.StrictPMP(PMP2SDP.monomial_basis(T,0),1)
        PMP2SDP.set_objective!(pmp,zero(T),T[1])
        # max y subject to 1-y >= 0 on the half line.
        PMP2SDP.add_block!(pmp,PMP2SDP.HalfLine(zero(T)),[reshape(T[1],1,1)],[[reshape(T[-1],1,1)]])
        compiled=PMP2SDP.compile_sampled_to_sdp(pmp)
        conic=PMP2SDP.compile_to_conic(compiled)
        settings=SDPX.Settings(T)
        result=PMP2SDP.solve_pmp(compiled,SDPX.solve_conic;cone_map,settings,return_result=true)
        tolerance=T===Float64 ? T(1e-7) : BigFloat("1e-30")
        @test SDPX.is_optimal(result)
        @test abs(result.x[1]-1)<tolerance
        @test norm(conic.A*result.x+result.s-conic.b,Inf)<tolerance
        @test norm(conic.q+conic.A'*result.z,Inf)<tolerance
    end
end
