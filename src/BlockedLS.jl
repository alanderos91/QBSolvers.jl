#
# Storage for LS problem; name outdated
#
mutable struct BlockedLS{T <: AbstractFloat}
  # data
  X    :: AbstractMatrix{T}
  y    :: AbstractVector{T}
  # parameter
  β    :: AbstractVector{T}
  # working arrays
  xtx  :: AbstractMatrix{T} # only relaized when p <= n
  xty  :: AbstractVector{T}
  # storaging arrays
  grad :: AbstractVector{T}
  storage_n_1 :: AbstractVector{T}
  storage_n_2 :: AbstractVector{T}
  storage_p_1 :: AbstractVector{T}
  storage_p_2 :: AbstractVector{T}
  direction   :: AbstractVector{T}
end

# constructor
function BlockedLS(
  X         :: AbstractMatrix{T}, 
  y         :: AbstractVector{T};
  covmatrix :: Bool = false
  ) where T <: AbstractFloat
  # dimensions
  n, p = size(X)
  
  if p > n || !covmatrix
    xtx = Matrix{T}(undef, 0, 0) # empty matrix
  else
    xtx = Matrix{T}(undef, p, p)
    mul!(xtx, transpose(X), X)
  end
  xty = Vector{T}(undef, p)
  mul!(xty, transpose(X), y)
  
  # parameter
  β    = Vector{T}(undef, p)
  grad = Vector{T}(undef, p)
  storage_n_1 = Vector{T}(undef, n)
  storage_n_2 = Vector{T}(undef, n)
  storage_p_1 = Vector{T}(undef, p)
  storage_p_2 = Vector{T}(undef, p)
  direction   = Vector{T}(undef, p)
  
  fill!(β, 0)
  copyto!(grad, β)
  copyto!(storage_p_1, β)
  copyto!(storage_p_2, β)
  copyto!(direction, β)
  copyto!(storage_n_1, y)
  copyto!(storage_n_2, y)
  
  BlockedLS(
  X, y, 
  β, 
  xtx, xty,
  grad, 
  storage_n_1, 
  storage_n_2, 
  storage_p_1,
  storage_p_2,
  direction
  )
end

eltype(x::BlockedLS{T}) where {T} = T