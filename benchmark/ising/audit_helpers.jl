using JSON, LinearAlgebra, SparseArrays, GenericLinearAlgebra, SHA
include(joinpath(@__DIR__, "sampled", "sampled_to_conic.jl"))
using .SDPBSampledInput
include(joinpath(@__DIR__, "sampled", "sampled_primal.jl"))
using .SDPBSampledPrimal
function read_matrix(path)
    tokens=split(read(path,String)); length(tokens)>=2 || error("missing matrix header: $path")
    m,n=parse.(Int,tokens[1:2]); length(tokens)==2+m*n || error("wrong matrix size: $path")
    # Elemental text writer emits one row per line.
    permutedims(reshape(parse.(BigFloat,tokens[3:end]),n,m))
end
const maximum_output_skew = Ref(BigFloat(0))
function read_symmetric_matrix(path)
    M=read_matrix(path)
    size(M,1)==size(M,2) || error("matrix output not square: $path")
    skew=opnorm(M-transpose(M),Inf)/max(one(BigFloat),opnorm(M,Inf))
    maximum_output_skew[]=max(maximum_output_skew[],skew)
    skew<=parse(BigFloat,tolerance_text) || error("matrix output symmetry tolerance failed: $path")
    # The antisymmetric part has zero contraction with each symmetric SDP
    # coefficient. Record its magnitude and audit the symmetric part explicitly.
    (M+transpose(M))/BigFloat(2)
end
function svec(M)
    [r==c ? M[r,c] : sqrt(BigFloat(2))*M[r,c] for c in axes(M,2) for r in 1:c]
end
function from_svec(v,n)
    M=zeros(BigFloat,n,n); k=0
    for c=1:n,r=1:c
        k+=1; M[r,c]=M[c,r]=r==c ? v[k] : v[k]/sqrt(BigFloat(2))
    end
    M
end
maxabs(v)=isempty(v) ? zero(BigFloat) : maximum(abs,v)
function cone_violation(M)
    # Independent arbitrary-precision symmetric eigensolver, outside solve timer.
    max(zero(BigFloat),-minimum(eigvals(Symmetric(M))))/max(one(BigFloat),opnorm(M,Inf))
end
function audit(conic,x,s,z)
    rp=conic.A*x+s-conic.b; rd=conic.q+transpose(conic.A)*z
    primal=maxabs(rp)/max(one(BigFloat),maxabs(conic.A*x),maxabs(s),maxabs(conic.b))
    dual=maxabs(rd)/max(one(BigFloat),maxabs(conic.q),maxabs(transpose(conic.A)*z))
    pobj=dot(conic.q,x); dobj=-dot(conic.b,z)
    gap=abs(pobj-dobj)/max(one(BigFloat),abs(pobj),abs(dobj))
    pv=zero(BigFloat); dv=zero(BigFloat)
    for g in conic.grams
        pv=max(pv,cone_violation(from_svec(s[g.slack_rows],g.dimension)))
        dv=max(dv,cone_violation(from_svec(z[g.slack_rows],g.dimension)))
    end
    primal=max(primal,maxabs(s[1:conic.num_equalities]))
    mapping=sampled_primal_mapping_audit(conic,x,s,z)
    Dict("primal"=>primal,"dual"=>dual,"gap"=>gap,"primal_psd"=>pv,"dual_psd"=>dv,
         "mapping"=>mapping,"objective"=>conic.sampled.constant+pobj)

end
function mapping_gate_value(a)
    a.finite || return BigFloat(Inf)
    max(a.sampled_primal_affine,a.sampled_primal_psd_link,a.zero_slack,a.sampled_dual_relative,a.gap)
end

stringify(v::AbstractFloat)=string(v)
stringify(v::AbstractArray)=map(stringify,v)
stringify(v::NamedTuple)=Dict(string(k)=>stringify(x) for (k,x) in pairs(v))
stringify(v::AbstractDict)=Dict(string(k)=>stringify(x) for (k,x) in v)
stringify(v)=v
