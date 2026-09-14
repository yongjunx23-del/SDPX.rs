"""Conic descriptors use upper-column svec coordinates for PSD rows."""
abstract type SupportedCone end
for name in (:ZeroConeT, :NonnegativeConeT, :SecondOrderConeT, :PSDTriangleConeT)
    @eval struct $name <: SupportedCone
        dim::Int
    end
end
struct ExponentialConeT <: SupportedCone end
struct PowerConeT{T<:Real} <: SupportedCone
    α::T
end
struct GenPowerConeT{T<:Real} <: SupportedCone
    α::Vector{T}
    dim2::Int
end
const SUPPORTED_PRECISIONS = (128, 256, 512, 768, 1024, 2048)
function _check_precision(::Type{T}, bits) where T
    T === Float64 && bits == 53 && return bits
    T === BigFloat && bits in SUPPORTED_PRECISIONS && return bits
    throw(ArgumentError("use Float64 at 53 bits or BigFloat at one of $SUPPORTED_PRECISIONS bits"))
end
