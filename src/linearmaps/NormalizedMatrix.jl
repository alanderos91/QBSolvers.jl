###
### Normalized Matrix
###
### C = (A - 1āᵀ)S⁻¹
###
### where C is a centered and scaled to be a correlation matrix.
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
  scale = similar(A, n_var); @. scale = 1
  
  for j in axes(A, 2)
    if !iszero(shift[j])
      @views u .= A[:, j]
      @. u = u - shift[j]
      scale[j] = norm(u)
    end
  end
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
  @. v = x
  ldiv!(Diagonal(C.scale), v)
  γ = dot(C.shift, v)
  fill!(y, γ)
  mul!(y, C.A, v, one(T), -one(T))
  return y
end

function LinearAlgebra.mul!(x::AbstractVector, _C::Transpose{<:Any,<:NormalizedMatrix}, y::AbstractVector)
  C = parent(_C)
  T = eltype(C)
  v = C.v
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

function form_AtApI!(AtApI::AbstractMatrix{T}, C::NormalizedMatrix{T}, alpha, beta) where T
  BLAS.syrk!('L', 'T', T(alpha), C.A, zero(T), AtApI) # AtApI enters as I
  BLAS.syr!('L', -T(C.n_obs), C.shift, AtApI)
  ldiv!(Diagonal(C.scale), AtApI)
  rdiv!(AtApI, Diagonal(C.scale))
  @views begin
    AtApI[diagind(AtApI)] .+= T(beta)
  end
  return AtApI
end
