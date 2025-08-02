###
### Gram plus Diagonal
###
### αAᵀA+βD
###

struct GramPlusDiag{T,matT1,matT2,matT3,vecT} <: AbstractMatrix{T}
  A::matT1
  AtA::matT2
  D::matT3
  n_obs::Int
  n_var::Int
  tmp::vecT
  alpha::T
  beta::T
end

function GramPlusDiag(A::AbstractMatrix{T}, D::Union{UniformScaling{T},Diagonal{T}} = one(T)*I;
  alpha::Real = one(T),
  beta::Real  = zero(T),
  gram::Bool  = false
) where T
  #
  n_obs, n_var = size(A)
  if gram
    AtA = transpose(A) * A
    tmp = similar(A, 0)
  else
    AtA = similar(A, 0, 0)
    tmp = similar(A, n_obs)
  end
  return GramPlusDiag(A, AtA, D, n_obs, n_var, tmp, T(alpha), T(beta))
end

function GramPlusDiag(gpd::GramPlusDiag{T}, alpha, beta) where T
  return GramPlusDiag(gpd.A, gpd.AtA, gpd.D, gpd.n_obs, gpd.n_var, gpd.tmp, T(alpha), T(beta))
end

function GramPlusDiag(gpd::GramPlusDiag, D::Diagonal)
  return GramPlusDiag(gpd.A, gpd.AtA, D, gpd.n_obs, gpd.n_var, gpd.tmp, gpd.alpha, gpd.beta)
end

function GramPlusDiag(gpd::GramPlusDiag, D::UniformScaling)
  return GramPlusDiag(gpd.A, gpd.AtA, D, gpd.n_obs, gpd.n_var, gpd.tmp, gpd.alpha, gpd.beta)
end

function Base.getindex(gpd::GramPlusDiag, i, j)
  alpha, beta = gpd.alpha, gpd.beta
  if length(gpd.AtA) > 0
    AtA_ij = _gpd_getindex_(gpd.A, gpd, i, j)
    alpha * AtA_ij + beta * gpd.D[i,j]
  else
    @views begin
      alpha * dot(gpd.A[:,i], gpd.A[:,j]) + beta * gpd.D[i,j]
    end
  end
end

function _gpd_getindex_(::AbstractMatrix, gpd, i, j)
  gpd.AtA[i,j]
end

LinearAlgebra.issymmetric(::GramPlusDiag) = true
Base.size(gpd::GramPlusDiag) = (gpd.n_var, gpd.n_var)
Base.eltype(::GramPlusDiag{T}) where T = T

function LinearAlgebra.mul!(y::AbstractVector, gpd::GramPlusDiag, x::AbstractVector)
  _gpd_mul_(gpd.A, y, gpd, x) # dispatch to handle AtA*x
  if iszero(gpd.beta)
    @. y = gpd.alpha * y
  else
    # α*AᵀAx + β*D*x
    _gpd_mul_diag_(y, gpd.D, x, gpd.beta, gpd.alpha)
  end
  return y
end

function _gpd_mul_(::AbstractMatrix, y, gpd, x)
  if size(gpd.AtA, 1) > 0
    mul!(y, Symmetric(gpd.AtA), x)
  else
    mul!(gpd.tmp, gpd.A, x)
    mul!(y, transpose(gpd.A), gpd.tmp)
  end
  return y
end

function _gpd_mul_diag_(y, D::Diagonal, x, alpha, beta)
  T = eltype(y)
  @. y = T(beta)*y + T(alpha) * D.diag * x
  return y
end

function _gpd_mul_diag_(y, D::UniformScaling, x, alpha, beta)
  T = eltype(y)
  @. y = T(beta)*y + T(alpha*D.λ)*x
  return y
end

