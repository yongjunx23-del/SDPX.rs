using Test, SDPX, SparseArrays, LinearAlgebra

@testset "Sampled ABI and owned factors" begin
    @test sizeof(SDPX.CSampledBlock)==104
    @test sizeof(SDPX.CSettings)==80 && sizeof(SDPX.CInfo)==104
    @test_throws ArgumentError SampledBlock(0,1,1,ones(1,1),[-1.0])
    @test_throws DimensionMismatch SampledBlock(1,1,2,ones(1,1),[-1.0])
    @test_throws ArgumentError SampledBlock(1,1,1,ones(1,1),[NaN])
    # Explicit precision must not change ambient precision or alias MPFR factors.
    ambient=precision(BigFloat)
    one512=BigFloat(1;precision=512)
    block=SampledBlock(1,1,1,reshape([one512],1,1),[-one512])
    settings=Settings(BigFloat;precision_bits=512)
    p=sampled_program([one512],spzeros(BigFloat,1,1),[-one512],
        [PSDTriangleConeT(1)],[block];settings)
    @test precision(BigFloat)==ambient
    @test precision(p.blocks[1].basis[1])==512
    @test p.blocks[1].basis[1] !== block.basis[1]
    @test p.blocks[1].basis[1] !== one512
    block.basis[1]=BigFloat(2;precision=512)
    @test p.blocks[1].basis[1]==1
end

@testset "Sampled factor conversion waits for explicit arithmetic" begin
    setprecision(BigFloat,256) do
        settings=Settings(BigFloat;precision_bits=512)
        exact=big(2)^300+1
        one512=BigFloat(1;precision=512)
        mixed=SampledBlock(1,1,1,reshape([one512],1,1),[exact])
        @test eltype(mixed.basis)===BigFloat && eltype(mixed.weights)===BigInt
        @test mixed.weights[1]==exact
        p=sampled_program([one512],spzeros(BigFloat,1,1),[one512],
            [PSDTriangleConeT(1)],[mixed];settings)
        @test precision(BigFloat)==256
        @test precision(p.blocks[1].weights[1])==512
        @test p.blocks[1].weights[1]==BigFloat(exact;precision=512)
        @test BigInt(p.blocks[1].weights[1])==exact
        @test p.blocks[1].basis[1] !== mixed.basis[1]
        rational=SampledBlock(1,1,1,ones(1,1),[3//7])
        @test eltype(rational.basis)===Float64 && eltype(rational.weights)===Rational{Int}
        p2=sampled_program([1.0],spzeros(1,1),[1.0],
            [PSDTriangleConeT(1)],[rational];settings)
        @test p2.blocks[1].weights[1]==BigFloat(3//7;precision=512)
        @test precision(p2.blocks[1].weights[1])==512
        @test precision(BigFloat)==256
    end
end

@testset "Sampled zero-column factors" begin
    block=SampledBlock(1,1,1,Matrix{Float64}(undef,2,0),Float64[])
    @test size(block.basis)==(2,0) && isempty(block.weights)
    p=sampled_program(Float64[],spzeros(3,0),zeros(3),[PSDTriangleConeT(2)],[block])
    descriptor=only(SDPX._program_sampled_storage(p).descriptors)
    @test descriptor.basis_rows==2 && descriptor.basis_cols==0
    @test descriptor.basis.count==0 && descriptor.weights.count==0
end

@testset "Sampled public direct and prepared solve" begin
    for (T,bits) in ((Float64,53),(BigFloat,512))
        setprecision(BigFloat,bits==53 ? 256 : bits) do
            tol=T===Float64 ? T(1e-8) : sqrt(eps(T))
            settings=Settings(T;precision_bits=bits)
            q=T[1]; b=T[-1]; A=spzeros(T,1,1)
            block=SampledBlock(1,1,1,reshape(T[1],1,1),T[-1])
            p=sampled_program(q,A,b,[PSDTriangleConeT(1)],[block];settings)
            r=solve_conic(p;settings,return_result=true)
            @test is_optimal(r)
            @test abs(r.x[1]-1)/(1+abs(r.x[1]))<=tol
            @test abs(primal_objective(r)-1)/2<=tol
            @test abs(-r.x[1]+r.s[1]+1)/2<=tol
            @test abs(1-r.z[1])/2<=tol
            @test min(r.s[1],r.z[1])>=-2tol
            @test abs(primal_objective(r)-dual_objective(r))/(1+abs(primal_objective(r)))<=tol
            @test r.info.precision_bits==bits
            prepared=prepare(p;settings)
            try
                r2=solve!(prepared)
                @test is_optimal(r2)
                @test abs(r2.x[1]-1)/(1+abs(r2.x[1]))<=tol
                r3=solve!(prepared;b=T[-2])
                @test is_optimal(r3)
                @test abs(r3.x[1]-2)/(1+abs(r3.x[1]))<=tol
            finally
                close(prepared)
            end
            @test !isopen(prepared)
            # Full-rank PSD2 factors exercise an untransformed sampled Schur block.
            s2=Settings(T;precision_bits=bits,kkt_form=:condensed,
                presolve_enable=false,chordal_decomposition_enable=false)
            blocks=[SampledBlock(1,1,1,Matrix{T}(I,2,2),T[-1,-1])]
            p2=sampled_program(T[1,1],spzeros(T,3,2),T[-1,0,-1],
                [PSDTriangleConeT(2)],blocks;settings=s2)
            r2=solve(p2;settings=s2)
            @test is_optimal(r2)
            @test r2.info.kkt_form==:condensed
            @test norm(r2.x.-1,Inf)/(1+norm(r2.x,Inf))<=tol
            @test abs(primal_objective(r2)-2)/3<=tol
            @test norm(T[-r2.x[1],0,-r2.x[2]]+r2.s-p2.b,Inf)/2<=tol
            @test norm(T[1-r2.z[1],1-r2.z[3]],Inf)/2<=tol
            @test min(r2.s[1],r2.s[3],r2.z[1],r2.z[3])>=-2tol
            # Valid descriptor payload with incompatible cone/row metadata is rejected.
            bad=sampled_program(T[1,1],spzeros(T,3,2),T[-1,0,-1],
                [PSDTriangleConeT(2)],[SampledBlock(2,1,1,Matrix{T}(I,2,2),T[-1,-1])];settings=s2)
            @test_throws ErrorException prepare(bad;settings=s2)
            nonzero=sampled_program(T[1,1],sparse([1],[1],T[1],3,2),T[-1,0,-1],
                [PSDTriangleConeT(2)],blocks;settings=s2)
            @test_throws ErrorException prepare(nonzero;settings=s2)
        end
    end
end
