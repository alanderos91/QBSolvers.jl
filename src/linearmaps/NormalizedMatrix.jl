###
### Normalized Matrix
###
### C = (A - 1āᵀ)S⁻¹
###
### so that CᵀC is a correlation matrix.
###

struct NormalizedMatrix{T,matT,vecT1,vecT2} <: AbstractMatrix{T}
  A::matT
  n_obs::Int
  n_var::Int
  shift::vecT2
  scale::vecT2
  u::vecT1    # length n_obs
  v::vecT2    # length n_var
end

function NormalizedMatrix(A::AbstractMatrix{T};
  u::AbstractVector{T} = similar(A, size(A, 1)),
  v::AbstractVector{T} = similar(A, size(A, 2))
  ) where T
  #
  n_obs, n_var = size(A)
  shift = similar(A, n_var); mean!(shift, transpose(A))
  scale = std(A, dims=1) |> vec
  @. scale = scale * sqrt(n_obs - 1)
  matT, vecT1, vecT2 = typeof(A), typeof(u), typeof(v)
  return NormalizedMatrix{T,matT,vecT1,vecT2}(A, n_obs, n_var, shift, scale, u, v)
end

Base.getindex(C::NormalizedMatrix, i, j) = (C.A[i,j] - C.shift[j]) / C.scale[j]

LinearAlgebra.issymmetric(C::NormalizedMatrix) = issymmetric(C.A)
Base.size(C::NormalizedMatrix) = (C.n_obs, C.n_var)
Base.eltype(::NormalizedMatrix{T}) where T = T

function Base.view(C::NormalizedMatrix, I, J)
  T = eltype(C)
  Aview = view(C.A, I, J)
  n_obs, n_var = size(Aview, 1), length(J)
  shift = view(C.shift, J)
  scale = view(C.scale, J)
  u = view(C.u, I)
  v = view(C.v, J)
  matT, vecT1, vecT2 = typeof(Aview), typeof(u), typeof(v)
  return NormalizedMatrix{T,matT,vecT1,vecT2}(Aview, n_obs, n_var, shift, scale, u, v)
end

function LinearAlgebra.mul!(y::AbstractVector, C::NormalizedMatrix, x::AbstractVector)
  T = eltype(C)
  v = C.v
  ldiv!(v, Diagonal(C.scale), x)
  c = dot(C.shift, v)
  mul!(y, C.A, v)
  @. y = y - T(c)
  return y
end

function LinearAlgebra.mul!(x::AbstractVector, _C::Transpose{<:Any,<:NormalizedMatrix}, y::AbstractVector)
  C = parent(_C)
  T = eltype(C)
  γ = sum(y)
  @. x = C.shift
  mul!(x, transpose(C.A), y, one(T), -T(γ))
  ldiv!(Diagonal(C.scale), x)
  return x
end

function LinearAlgebra.mul!(AtB::AbstractMatrix, CAt::Transpose{<:Any,<:NormalizedMatrix}, CB::NormalizedMatrix)
  T = eltype(AtB)
  CA = parent(CAt)
  mul!(AtB, transpose(CA.A), CB.A)
  alpha = T(CA.n_obs) # == CB.n_obs

  # we should use dispatch here; for now assume CA.A and CB.A are dense matrices on CPU
  BLAS.ger!(-alpha, CA.shift, CB.shift, AtB)

  ldiv!(Diagonal(CA.scale), AtB)
  rdiv!(AtB, Diagonal(CB.scale))

  return AtB
end

#
# Extensions to GramPlusDiag
#
function GramPlusDiag(A_::NormalizedMatrix{T}, D::Union{UniformScaling{T},Diagonal{T}} = one(T)*I; kwargs...) where T
  # do not compute AtA under normalization
  gpd = GramPlusDiag(A_.A, D; kwargs...)
  # now pass to constructor using the possibly cached AtA
  return GramPlusDiag(A_, gpd.AtA, gpd.D, gpd.n_obs, gpd.n_var, gpd.tmp, gpd.alpha, gpd.beta)
end

function NormalizedGramPlusDiag(gpd::GramPlusDiag)
  A_ = NormalizedMatrix(gpd.A)
  return GramPlusDiag(A_, gpd.AtA, gpd.D, gpd.n_obs, gpd.n_var, gpd.tmp, gpd.alpha, gpd.beta)
end

function NormalizedGramPlusDiag(gpd::GramPlusDiag, D::AbstractMatrix)
  A_ = NormalizedMatrix(gpd.A)
  return GramPlusDiag(A_, gpd.AtA, D, gpd.n_obs, gpd.n_var, gpd.tmp, gpd.alpha, gpd.beta)
end

# make sure getindex works correctly when AtA is cached
function _gpd_getindex_(::NormalizedMatrix, gpd, i, j)
  n, S, u = gpd.n_obs, Diagonal(gpd.A.scale), gpd.A.shift
  1/S[i,i] * (gpd.AtA[i,j] - n*u[i]*u[j]) * 1/S[j,j]
end

# make sure mul! works correctly when AtA is cached
function _gpd_mul_(::NormalizedMatrix, y, gpd, x)
  T = eltype(gpd)
  n, S, avg, v = gpd.n_obs, Diagonal(gpd.A.scale), gpd.A.shift, gpd.A.v
  AtA = GramPlusDiag(gpd.A.A, gpd.AtA, gpd.D, gpd.n_obs, gpd.n_var, gpd.tmp, one(T), zero(T))
  ldiv!(v, S, x)
  c = dot(avg, v)
  mul!(y, AtA, v)
  @. y = y - n*c*avg
  ldiv!(S, y)
  return y
end
