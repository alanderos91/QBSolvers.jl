###
### Block Diagonal + Rank-1
###
### H = S(D - n(S⁻¹x̄)(S⁻¹x̄)ᵀ)S = SDS - nx̄x̄ᵀ
###
### D is (block) diagonal with entries in X̂ᵀX̂+βI for some scalar β.
###   This should be a specialized type like Diagonal or BlkDiagHessian.
### x̄ is a shift vector so that (X - 1x̄ᵀ) is centered at 0.
### S is a diagonal matrix so that columns of X̂ = (X - 1x̄ᵀ)S⁻¹ have unit norm.
### X̂ᵀX̂ is a correlation matrix.
###

struct BlkDiagPlusRank1{T,matT1,matT2,vecT} <: AbstractMatrix{T}
  n_obs::Int
  n_var::Int
  D::matT1
  xbar::vecT
  S::matT2
  w::vecT
end

function BlkDiagPlusRank1(X, D)
  n_obs, n_var = size(X)
  shift = similar(X, n_var); mean!(shift, transpose(X))
  scale = similar(X, n_var); @. scale = 1
  u = similar(X, n_obs) # temporary
  for j in axes(X, 2)
    if !iszero(shift[j])
      @views u .= X[:, j]
      @. u = u - shift[j]
      scale[j] = norm(u)
    end
  end
  S = Diagonal(scale)
  w = copy(shift)
  ldiv!(S, w)
  ldiv!(D, w)
  ldiv!(S, w)
  T, matT1, matT2, vecT = eltype(D), typeof(D), typeof(S), typeof(w)
  return BlkDiagPlusRank1{T,matT1,matT2,vecT}(n_obs, n_var, D, shift, S, w)
end

function Base.getindex(H::BlkDiagPlusRank1, i, j)
  n, D, S, xbar = H.n_obs, H.D, H.S, H.xbar
  return (S[i,i]*D[i,j]*S[j,j] + n*xbar[i]*xbar[j])
end

LinearAlgebra.issymmetric(::BlkDiagPlusRank1) = true
Base.size(H::BlkDiagPlusRank1) = (H.n_var, H.n_var)
Base.eltype(::BlkDiagPlusRank1{T}) where T = T

function LinearAlgebra.mul!(v::AbstractVector, H::BlkDiagPlusRank1, u::AbstractVector)
  T = eltype(H)
  n, D, S, xbar = H.n_obs, H.D, H.S, H.xbar
  c = T(dot(xbar, u))
  # v = S*D*S*u
  @. v = u
  mul!(S, v)
  mul!(D, v)
  mul!(S, v)
  # v = S*D*S*u + nx̄x̄ᵀu
  @. v = v + n*c*xbar
  return v
end

#
# Sherman-Morrison:
#
# H⁻¹ = (SDS)⁻¹ - wwᵀ / (1/n + x̄ᵀw); w = (SDS)⁻¹x̄
#
function LinearAlgebra.ldiv!(H::BlkDiagPlusRank1, u::AbstractVector)
  T = eltype(H)
  n, D, S, xbar, w = H.n_obs, H.D, H.S, H.xbar, H.w
  top = dot(w, u)
  bot = (1/n + dot(xbar, w))
  # out = (SDS)⁻¹u
  ldiv!(S, u)
  ldiv!(D, u)
  ldiv!(S, u)
  # out = (SDS)⁻¹u - (wᵀu)/(1/n+x̄ᵀw)*w
  @. u = u - T(top/bot)*w
  return u
end
