###
### Block Diagonal + Symmetric Rank-1
###
### H = αJ + βuuᵀ
###
### This object should be used as an approximation to AᵀA where A is n_obs × n_var.
### J is (block) diagonal.
###   This should be a specialized type like Diagonal or BlkDiagHessian.
### u is a vector
### α, β should be scalars
###

struct BlkDiagPlusRank1{T,matT,vecT} <: AbstractMatrix{T}
  n_obs::Int
  n_var::Int
  J::matT
  u::vecT
  w::vecT   # (αJ)⁻¹u
  alpha::T
  beta::T
end

function BlkDiagPlusRank1(n_obs, n_var, J, u, alpha, beta)
  T = eltype(J)
  w = alpha*copy(u)
  ldiv!(J, w)
  return BlkDiagPlusRank1(n_obs, n_var, J, u, w, alpha, beta)
end

function Base.getindex(H::BlkDiagPlusRank1, i, j)
  J, u, alpha, beta = H.J, H.u, H.alpha, H.beta
  return alpha * J[i,j] + beta * u[i]*transpose(u[j])
end

LinearAlgebra.issymmetric(::BlkDiagPlusRank1) = true
Base.size(H::BlkDiagPlusRank1) = (H.n_var, H.n_var)
Base.eltype(::BlkDiagPlusRank1{T}) where T = T

#
# y = H*x = αJ*x + β*(uᵀx)*u
#
function LinearAlgebra.mul!(y::AbstractVector, H::BlkDiagPlusRank1, x::AbstractVector)
  T = eltype(H)
  J, u, alpha, beta = H.J, H.u, H.alpha, H.beta
  @. y = x
  mul!(y, J, x, T(alpha), T(beta*dot(u, x)))
  return y
end

#
# Sherman-Morrison:
#
# H⁻¹ = (αJ + βuuᵀ)⁻¹ = (αJ)⁻¹ - wwᵀ / (β⁻¹ + uᵀw); w = (αJ)⁻¹u
#
function LinearAlgebra.ldiv!(H::BlkDiagPlusRank1, x::AbstractVector)
  T = eltype(H)
  J, u, w, alpha, beta = H.J, H.u, H.w, H.alpha, H.beta
  top = dot(w, x)
  bot = inv(beta) + dot(u, w)
  ldiv!(J, x)
  @. x = T(inv(alpha))*x + T(top/bot)*w
  return x
end
