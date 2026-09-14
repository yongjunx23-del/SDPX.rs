using SDPX, SparseArrays, LinearAlgebra
startswith(realpath(pathof(SDPX)),realpath(ENV["SDPX_FROZEN_ROOT"])*"/") || error("SDPX is outside frozen source")
for (T,bits,toltext) in ((Float64,53,"1e-8"),(BigFloat,512,"1e-42"))
    setprecision(BigFloat,512) do
        tol=parse(T,toltext)
        q=T[1]; A=sparse(reshape(T[-1,0,-1],3,1)); b=T[0,sqrt(T(2)),0]
        settings=SDPX.Settings(T;precision_bits=bits,tolerances=SDPX.Tolerances(T;primal=tol,dual=tol,gap=tol))
        r=SDPX.solve_conic(q,A,b,[SDPX.PSDTriangleConeT(2)];settings,return_result=true)
        @assert SDPX.is_optimal(r)
        accept=T===Float64 ? T(1e-6) : parse(T,"1e-30")
        @assert abs(r.x[1]-1)<=accept
        @assert maximum(abs,A*r.x+r.s-b)<=accept
        @assert maximum(abs,q+transpose(A)*r.y)<=accept
        @assert abs(dot(q,r.x)+dot(b,r.y))<=accept
        @assert r.info.precision_bits==bits
        println("PASS ",T," ",bits," ",r.info)
    end
end
