###
### Block Diagonal Hessian 
###
### | αA₁ᵀA₁+βD                         |
### |           αA₂ᵀA₂+βD               |
### |                     ⋱            |
### |                        αAₘᵀAₘ+βD  |
###

struct BlkDiagHessian{T,matT1,matT2,blkmatT,gpdT} <: AbstractMatrix{T}
  A::matT1
  n_obs::Int
  n_var::Int
  n_blk::Int
  A_blk::blkmatT
  diag::Vector{gpdT}
  chol::Vector{Cholesky{T,matT2}}
  alpha::T
  beta::T
end

function BlkDiagHessian(A::AbstractMatrix{T}, D, n_blk::Int;
  alpha::Real   = one(T),
  beta::Real    = zero(T),
  factor::Bool  = true,
  gram::Bool    = false,
) where T
  #
  n_obs, n_var = size(A)
  var_per_blk = cld(n_var, n_blk)
  A_blk = BlockedArray(A, [n_obs], repeat([var_per_blk], n_blk))

  blk1 = make_AtApD_blk(A_blk, D, 1, alpha, beta, gram)
  diag = Vector{typeof(blk1)}(undef, n_blk)
  chol = Vector{Cholesky{T,Matrix{T}}}(undef, n_blk)

  diag[1] = blk1
  if factor
    chol[1] = factor_AtApD_blk(diag[1], alpha, beta)
  end

  for k in 2:n_blk
    diag[k] = make_AtApD_blk(A_blk, D, k, alpha, beta, gram)
    if factor
      chol[k] = factor_AtApD_blk(diag[k], alpha, beta)
    end
  end

  return BlkDiagHessian(A, n_obs, n_var, n_blk, A_blk, diag, chol, T(alpha), T(beta))
end

function BlkDiagHessian(AtApD::GramPlusDiag{T}, n_blk::Int;
  alpha::Real   = one(T),
  beta::Real    = zero(T),
  factor::Bool  = true,
) where T
  #
  A, AtA, D = AtApD.A, AtApD.AtA, AtApD.D
  @assert size(AtA, 1) > 0 # raise an error if we never cached the full AtA

  n_obs, n_var = size(A)
  var_per_blk = cld(n_var, n_blk)
  blocksizes = repeat([var_per_blk], n_blk)
  A_blk = BlockedArray(A, [n_obs], blocksizes)
  AtA_blk = BlockedArray(AtA, blocksizes, blocksizes)

  blk1 = make_AtApD_blk(A_blk, AtA_blk, D, 1, alpha, beta)
  diag = Vector{typeof(blk1)}(undef, n_blk)
  chol = Vector{Cholesky{T,Matrix{T}}}(undef, n_blk)

  diag[1] = blk1
  if factor
    chol[1] = factor_AtApD_blk(diag[1], alpha, beta)
  end

  for k in 2:n_blk
    diag[k] = make_AtApD_blk(A_blk, AtA_blk, D, k, alpha, beta)
    if factor
      chol[k] = factor_AtApD_blk(diag[k], alpha, beta)
    end
  end

  return BlkDiagHessian(A, n_obs, n_var, n_blk, A_blk, diag, chol, T(alpha), T(beta))
end

function update_factors!(bdh::BlkDiagHessian, alpha, beta)
  T = eltype(bdh)
  for k in 1:bdh.n_blk
    bdh.diag[k] = GramPlusDiag(bdh.diag[k], alpha, beta)
    if isassigned(bdh.chol, k)
      bdh.chol[k] = factor_AtApD_blk!(bdh.chol[k].factors, bdh.diag[k], alpha, beta)
    else
      bdh.chol[k] = factor_AtApD_blk(bdh.diag[k], alpha, beta)
    end
  end
  return bdh
end

function update_factors!(bdh::BlkDiagHessian, A, D, alpha, beta)
  T = eltype(bdh)
  n_obs, n_var, n_blk = bdh.n_obs, bdh.n_var, bdh.n_blk
  var_per_blk = cld(n_var, n_blk)
  chol = bdh.chol
  A_blk = BlockedArray(A, [n_obs], repeat([var_per_blk], n_blk))
  blk1 = make_AtApD_blk(bdh, A_blk, D, 1, alpha, beta)
  diag = Vector{typeof(blk1)}(undef, n_blk)
  diag[1] = blk1

  if isassigned(chol, 1)
    chol[1] = factor_AtApD_blk!(chol[1].factors, diag[1], alpha, beta)
  else
    chol[1] = factor_AtApD_blk(diag[1], alpha, beta)
  end

  for k in 2:bdh.n_blk
    diag[k] = make_AtApD_blk(bdh, A_blk, D, k, alpha, beta)
    if isassigned(chol, k)
      chol[k] = factor_AtApD_blk!(chol[k].factors, diag[k], alpha, beta)
    else
      chol[k] = factor_AtApD_blk(diag[k], alpha, beta)
    end
  end
  return BlkDiagHessian(
    A, bdh.n_obs, bdh.n_var, bdh.n_blk,
    A_blk,
    diag, chol, T(alpha), T(beta)
  )
end

function Base.getindex(bdh::BlkDiagHessian{T}, i, j) where T
  blkrng = axes(bdh.A_blk, 2)
  blk_i = findblock(blkrng, i)
  blk_j = findblock(blkrng, j)
  if blk_i == blk_j
    k = first(blk_i.n)
    D_kk = bdh.diag[k]
    var_per_blk = size(D_kk, 1)
    return getindex(D_kk, mod1(i, var_per_blk), mod1(j, var_per_blk))
  else
    return zero(T)
  end
end

LinearAlgebra.issymmetric(::BlkDiagHessian) = true
Base.size(bdh::BlkDiagHessian) = (bdh.n_var, bdh.n_var)
Base.eltype(::BlkDiagHessian{T}) where T = T

function LinearAlgebra.mul!(y::AbstractVector, bdh::BlkDiagHessian, x::AbstractVector)
  baxes = axes(bdh.A_blk, 2)
  for k in 1:bdh.n_blk
    D_kk = bdh.diag[k]
    @views idx = baxes[Block(k)]
    @views mul!(y[idx], D_kk, x[idx])
  end
  return y
end

function LinearAlgebra.ldiv!(y::AbstractVector, bdh::BlkDiagHessian, x::AbstractVector)
  baxes = axes(bdh.A_blk, 2)
  for k in 1:bdh.n_blk
    chol_D_kk = bdh.chol[k]
    @views idx = baxes[Block(k)]
    @views ldiv!(y[idx], chol_D_kk, x[idx])
  end
  return y
end

function LinearAlgebra.ldiv!(bdh::BlkDiagHessian, x::AbstractVector)
  baxes = axes(bdh.A_blk, 2)
  for k in 1:bdh.n_blk
    chol_D_kk = bdh.chol[k]
    @views idx = baxes[Block(k)]
    @views ldiv!(chol_D_kk, x[idx])
  end
  return x
end
