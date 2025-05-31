###
### Gram plus Diagonal
###
### αAᵀA+βI
###

struct GramPlusDiag{T,matT1,matT2,vecT} <: AbstractMatrix{T}
  A::matT1
  AtA::matT2
  n_obs::Int
  n_var::Int
  tmp::vecT
  alpha::T
  beta::T
end

function GramPlusDiag(A::AbstractMatrix{T};
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
  return GramPlusDiag(A, AtA, n_obs, n_var, tmp, T(alpha), T(beta))
end

function GramPlusDiag(gpd::GramPlusDiag{T}, alpha, beta) where T
  return GramPlusDiag(gpd.A, gpd.AtA, gpd.n_obs, gpd.n_var, gpd.tmp, T(alpha), T(beta))
end

function Base.getindex(gpd::GramPlusDiag, i, j)
  alpha, beta = gpd.alpha, gpd.beta
  if length(gpd.AtA) > 0
    AtA_ij = _gpd_getindex_(gpd.A, gpd, i, j)
    alpha * AtA_ij + (i == j)*beta
  else
    @views begin
      alpha * dot(gpd.A[:,i], gpd.A[:,j]) + (i == j)*beta
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
    axpby!(gpd.beta, x, gpd.alpha, y)
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
