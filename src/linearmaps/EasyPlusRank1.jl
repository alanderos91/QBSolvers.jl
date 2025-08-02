###
### Easy + Symmetric Rank-1
###
### H = αJ + βuvᵀ
###
### J is a square matrix; should be a specialized type that supports ldiv!()
### u, v should be vectors
### α, β should be scalars
###

struct EasyPlusRank1{T,matT,vecT} <: AbstractMatrix{T}
  J::matT
  u::vecT
  v::vecT
  aJu::vecT
  aJv::vecT
  alpha::T
  beta::T
end

function EasyPlusRank1(J::matT, u::vecT, v::vecT, alpha, beta) where {matT, vecT}
  T = eltype(J)
  if length(u) != length(v)
    throw(DimensionMismatch("Factors `u` and `v` do not match dimension; got $(length(u)) and $(length(v)), respectively."))
  end
  if size(J, 1) != size(J, 2)
    throw(DimensionMismatch("Matrix `J` must be square; got $(size(J))."))
  end
  if size(J, 1) != length(u)
    throw(DimensionMismatch("Matrix `J` is incompatible with rank-1 matrix `uvᵀ`."))
  end
  aJu = alpha*copy(u); ldiv!(J, aJu)
  aJv = alpha*copy(v); ldiv!(J, aJv)
  return EasyPlusRank1(J, u, v, aJu, aJv, T(alpha), T(beta))
end

function EasyPlusRank1(J::matT, u::vecT, alpha, beta) where {matT, vecT}
  T = eltype(J)
  if size(J, 1) != size(J, 2)
    throw(DimensionMismatch("Matrix `J` must be square; got $(size(J))."))
  end
  if size(J, 1) != length(u)
    throw(DimensionMismatch("Matrix `J` is incompatible with rank-1 matrix `uuᵀ`."))
  end
  aJu = alpha*copy(u); ldiv!(J, aJu)
  return EasyPlusRank1(J, u, u, aJu, aJu, T(alpha), T(beta))
end

function Base.getindex(H::EasyPlusRank1, i, j)
  J, u, v, alpha, beta = H.J, H.u, H.v, H.alpha, H.beta
  return alpha * J[i,j] + beta * u[i]*transpose(v[j])
end

LinearAlgebra.issymmetric(H::EasyPlusRank1) = issymmetric(H.J) && H.u === H.v
Base.size(H::EasyPlusRank1) = size(H.J)
Base.eltype(::EasyPlusRank1{T}) where T = T

#
# y = H*x = αJ*x + β*(vᵀx)*u
#
function LinearAlgebra.mul!(y::AbstractVector, H::EasyPlusRank1, x::AbstractVector)
  T = eltype(H)
  J, u, v, alpha, beta = H.J, H.u, H.v, H.alpha, H.beta
  @. y = u
  mul!(y, J, x, alpha, beta*dot(v, x))
  return y
end

#
# Sherman-Morrison:
#
# H⁻¹ = (αJ + βuvᵀ)⁻¹ = (αJ)⁻¹ - β (αJ)⁻¹u [(αJ)⁻¹v]ᵀ / (1 + βvᵀ(αJ)⁻¹u)
#
function LinearAlgebra.ldiv!(y::AbstractVector, H::EasyPlusRank1, x::AbstractVector)
  T = eltype(H)
  J, v, aJu, aJv, alpha, beta = H.J, H.v, H.aJu, H.aJv, H.alpha, H.beta
  top = beta * dot(aJv, x)
  bot = 1 + beta * dot(v, aJu)
  ldiv!(y, J, x)
  @. y = T(inv(alpha))*y - T(top/bot)*aJu
  return y
end

