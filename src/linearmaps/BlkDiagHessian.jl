###
### Block Diagonal Hessian 
###
### | αA₁ᵀA₁+βI                         |
### |           αA₂ᵀA₂+βI               |
### |                     ⋱            |
### |                        αAₘᵀAₘ+βI  |
###

struct BlkDiagHessian{T,matT,blkmatT,gpdT} <: AbstractMatrix{T}
  A::matT
  n_obs::Int
  n_var::Int
  n_blk::Int
  A_blk::blkmatT
  diag::Vector{gpdT}
  chol::Vector{Cholesky{T,matT}}
  alpha::T
  beta::T
end

function BlkDiagHessian(A::Matrix{T}, n_blk::Int;
  alpha::Real   = one(T),
  beta::Real    = zero(T),
  factor::Bool  = true,
  gram::Bool    = false,
) where T
  #
  n_obs, n_var = size(A)
  var_per_blk = cld(n_var, n_blk)
  A_blk = BlockedArray(A, [n_obs], repeat([var_per_blk], n_blk))

  A_11 = view(A_blk, Block(1))
  D_11 = GramPlusDiag(A_11; alpha=alpha, beta=beta, gram=gram)
  diag = Vector{typeof(D_11)}(undef, n_blk)
  chol = Vector{Cholesky{T,Matrix{T}}}(undef, n_blk)

  diag[1] = D_11
  if factor
    chol[1] = cholesky!(Symmetric(form_AtApI(A_11, alpha, beta), :L))
  end

  for k in 2:n_blk
    A_kk = view(A_blk, Block(k))
    diag[k] = GramPlusDiag(A_kk; alpha=alpha, beta=beta, gram=gram)
    if factor
      chol[k] = cholesky!(Symmetric(form_AtApI(A_kk, alpha, beta), :L))
    end
  end

  return BlkDiagHessian(A, n_obs, n_var, n_blk, A_blk, diag, chol, T(alpha), T(beta))
end

function BlkDiagHessian(AtApI::GramPlusDiag{T}, n_blk::Int;
  alpha::Real   = one(T),
  beta::Real    = zero(T),
  factor::Bool  = true,
) where T
  #
  A, AtA = AtApI.A, AtApI.AtA
  @assert size(AtA, 1) > 0 # raise an error if we never cached the full AtA

  n_obs, n_var = size(A)
  var_per_blk = cld(n_var, n_blk)
  blocksizes = repeat([var_per_blk], n_blk)
  A_blk = BlockedArray(A, [n_obs], blocksizes)
  AtA_blk = BlockedArray(AtA, blocksizes, blocksizes)

  A_1  = view(A_blk, Block(1))
  A_11 = view(AtA_blk, Block(1), Block(1))
  D_11 = GramPlusDiag(A_1, copy(A_11), n_obs, size(A_1, 2), similar(A, 0), T(alpha), T(beta))
  diag = Vector{typeof(D_11)}(undef, n_blk)
  chol = Vector{Cholesky{T,Matrix{T}}}(undef, n_blk)

  diag[1] = D_11
  if factor
    chol[1] = cholesky!(Symmetric(copy(A_11), :L))
  end

  for k in 2:n_blk
    A_k  = view(A_blk, Block(k))
    A_kk = view(AtA_blk, Block(k), Block(k))
    diag[k] = GramPlusDiag(A_k, copy(A_kk), n_obs, size(A_k, 2), similar(A, 0), T(alpha), T(beta))
    if factor
      chol[k] = cholesky!(Symmetric(copy(A_kk), :L))
    end
  end

  return BlkDiagHessian(A, n_obs, n_var, n_blk, A_blk, diag, chol, T(alpha), T(beta))
end

function form_AtApI(A::AbstractMatrix{T}, alpha, beta) where T
  p = size(A, 2)
  form_AtApI!(Matrix{T}(I, p, p), A, alpha, beta)
end

function form_AtApI!(AtApI::AbstractMatrix{T}, A::AbstractMatrix{T}, alpha, beta) where T
  BLAS.syrk!('L', 'T', T(alpha), A, T(beta), AtApI) # AtApI enters as I
end

function update_factors!(bdh::BlkDiagHessian{T}, alpha, beta) where T
  for k in 1:bdh.n_blk
    A_kk = view(bdh.A_blk, Block(k))
    bdh.diag[k] = GramPlusDiag(bdh.diag[k], alpha, beta)
    if isassigned(bdh.chol, k)
      AtApI = form_AtApI!(bdh.chol[k].factors, A_kk, alpha, beta)
    else
      AtApI = form_AtApI(A_kk, alpha, beta)
    end
    bdh.chol[k] = cholesky!(Symmetric(AtApI, :L))
  end
  return bdh
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
