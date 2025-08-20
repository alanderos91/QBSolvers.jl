###
### Symmetric Diagonal Scaling
###
### J = S * M * S
###
### M is an arbitrary matrix-like object implementing mul!()
### S is a diagonal matrix
###

struct SymDiagScale{T,matT,vecT} <: AbstractMatrix{T}
  M::matT
  s::vecT
  tmp::vecT
end

function SymDiagScale(M::matT, s::vecT, tmp::vecT = similar(s)) where {matT, vecT}
  T = eltype(M)
  if size(M, 1) != size(M, 2)
    throw(DimensionMismatch("Matrix `M` must be square; got $(size(M))."))
  end
  if !issymmetric(M)
    throw(ErrorException("Matrix `M` is not symmetric."))
  end
  if length(s) != size(M, 1)
    throw(DimensionMismatch("Scale factor `s` does not match dimension of `M`; got $(length(s)) and $(size(M, 1)), respectively."))
  end
  if length(tmp) != size(M, 1)
    throw(DimensionMismatch("Buffer `tmp` must match dimension of `M` and `s`."))
  end
  return SymDiagScale{T,matT,vecT}(M, s, tmp)
end

function Base.getindex(J::SymDiagScale, i, j)
  M, s = J.M, J.s
  return s[i]*M[i,j]*s[j]
end

LinearAlgebra.issymmetric(::SymDiagScale) = true
Base.size(J::SymDiagScale) = size(J.M)
Base.eltype(::SymDiagScale{T}) where T = T

#
# y = J*x = S*M*S*x
#
function LinearAlgebra.mul!(y::AbstractVector, J::SymDiagScale, x::AbstractVector)
  M, s, Sx = J.M, J.s, J.tmp
  @. Sx = s * x
  mul!(y, M, Sx)
  @. y = s * y
  return y
end

# HACK! We should avoid this
Symmetric(J::SymDiagScale) = J
