###
### GramMinusBlkDiag
###
### AtA - D
###
struct GramMinusBlkDiag{T,gpdT,bdhT,vecT} <: AbstractMatrix{T}
  AtA::gpdT
  D::bdhT
  tmp::vecT
end

function GramMinusBlkDiag(AtA::AbstractMatrix{T}, D::AbstractMatrix{T}) where T
  tmp = zeros(size(AtA, 1))
  gpdT = typeof(AtA)
  bdhT = typeof(D)
  vecT = typeof(tmp)
  GramMinusBlkDiag{T,gpdT,bdhT,vecT}(AtA, D, tmp)
end

function Base.getindex(gmbd::GramMinusBlkDiag, i, j)
  getindex(gmbd.AtA, i, j) - getindex(gmbd.D, i, j)
end

LinearAlgebra.issymmetric(::GramMinusBlkDiag) = true
Base.size(gmbd::GramMinusBlkDiag) = size(gmbd.AtA)
Base.eltype(::GramMinusBlkDiag{T}) where T = T

function LinearAlgebra.mul!(y::AbstractVector, gmbd::GramMinusBlkDiag, x::AbstractVector)
  # y = A' * (A * x)
  mul!(y, gmbd.AtA, x)

  # block substraction
  mul!(gmbd.tmp, gmbd.D, x)
  @. y = y - gmbd.tmp

  return y
end