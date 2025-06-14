#
# Spectral radius of AtA - J, where J is (block) diagonal Matrix based on AtA.
#
function estimate_spectral_radius(AtA, J; kwargs...)
  lambda, _, ch = powm!(GramMinusBlkDiag(AtA, J), ones(size(J, 1)); log=true, kwargs...)
  return abs(lambda)
end

#
# Efficient computation of AtA (block) diagonal
#
function compute_main_diagonal(A::AbstractMatrix, AtA)
  data = similar(A, size(A, 2))
  if size(AtA, 2) > 0
    idx = diagind(AtA)
    @views @. data = AtA[idx]
  else
    fdot(x) = dot(x, x)
    map!(fdot, data, eachcol(A))
  end
  return Diagonal(data)
end

function compute_main_diagonal(A::NormalizedMatrix, AtA)
  data = similar(A.A, size(A, 2))
  fill!(data, 1)
  return Diagonal(data)
end

function compute_block_diagonal(AtApD, n_blk; gram::Bool=false, kwargs...)
  if size(AtApD.AtA, 2) > 0
    J = BlkDiagHessian(AtApD, n_blk; kwargs...)
  else
    J = BlkDiagHessian(AtApD.A, AtApD.D, n_blk; gram=gram, kwargs...)
  end
  return J
end

#
# Allocation-free block extraction of Dⱼ in BlkDiagHessian αAⱼᵀAⱼ + βDⱼ
#
function get_diag_block(D::UniformScaling, baxes, blkI, blkJ)
  return D
end

function get_diag_block(D::Diagonal, baxes, blkI, blkJ)
  idxI, idxJ = baxes[blkI], baxes[blkJ]
  return view(D, idxI, idxJ)
end

function get_A_and_D_blk(A_blk, D, k)
  baxes = axes(A_blk, 2)
  A_k = view(A_blk, Block(k))
  D_kk = get_diag_block(D, baxes, Block(k), Block(k))
  return A_k, D_kk
end

function get_A_and_D_blk(A_blk, AtA_blk, D, k)
  baxes = axes(A_blk, 2)
  A_k = view(A_blk, Block(k))
  A_kk = view(AtA_blk, Block(k), Block(k))
  D_kk = get_diag_block(D, baxes, Block(k), Block(k))
  return A_k, A_kk, D_kk
end

function make_AtApD_blk(A_blk::BlockedArray, D, k::Integer, alpha::Real, beta::Real, gram::Bool)
  A_k, D_kk = get_A_and_D_blk(A_blk, D, k)
  return GramPlusDiag(A_k, D_kk; alpha=alpha, beta=beta, gram=gram)
end

function make_AtApD_blk(A_blk::BlockedArray, AtA_blk::BlockedArray, D, k::Integer, alpha::Real, beta::Real)
  T = eltype(A_blk)
  A_k, A_kk, D_kk = get_A_and_D_blk(A_blk, AtA_blk, D, k)
  return GramPlusDiag(A_k, copy(A_kk), D_kk, size(A_k, 1), size(A_k, 2), similar(parent(A_blk), 0), T(alpha), T(beta))
end

function make_AtApD_blk(bdh::BlkDiagHessian, A_blk::BlockedArray, D, k::Integer, alpha::Real, beta::Real)
  T = eltype(bdh)
  A_k, D_kk = get_A_and_D_blk(A_blk, D, k)
  return GramPlusDiag(A_k, bdh.diag[k].AtA, D_kk,
      bdh.diag[k].n_obs, bdh.diag[k].n_var, bdh.diag[k].tmp, alpha, beta)
end

#
# Gram + Diag computations
#
function factor_AtApD_blk(AtApD::GramPlusDiag, alpha, beta)
  T = eltype(AtApD)
  data = Matrix{T}(undef, size(AtApD))
  factor_AtApD_blk!(data, AtApD, alpha, beta)
end

function factor_AtApD_blk!(data, AtApD::GramPlusDiag, alpha, beta)
  A, AtA, D = AtApD.A, AtApD.AtA, AtApD.D
  if size(AtA, 2) > 0
    @. data = AtA
  else
    form_AtA!(data, A)
  end
  maybe_normalize!(data, A)
  @. data = alpha*data
  !iszero(beta) && make_AtApD_add_diag!(data, AtApD.D, beta)
  return cholesky!(Symmetric(data, :L))
end

function make_AtApD_add_diag!(data, D::UniformScaling, beta)
  T, idx = eltype(data), diagind(data)
  if !iszero(beta)
    for ii in idx
      data[ii] += T(beta*D.λ)
    end
  end
  data
end

function make_AtApD_add_diag!(data, D::Diagonal, beta)
  T, idx = eltype(data), diagind(data)
  if !iszero(beta)
    for (j, ii) in enumerate(idx)
      data[ii] += T(beta)*D.diag[j]
    end
  end
  data
end

function make_AtApD_add_diag!(data, D, beta) # D is a view of a Diagonal
  T, idx = eltype(data), diagind(data)
  if !iszero(beta)
    for (j, ii) in enumerate(idx)
      data[ii] += T(beta)*D[ii]
    end
  end
  data
end

function form_AtA!(data, A::AbstractMatrix)
  T = eltype(A)
  BLAS.syrk!('L', 'T', one(T), C.A, zero(T), data)
  return data
end

function form_AtA!(data, C::NormalizedMatrix)
  form_AtA!(data, C.A)
end

maybe_normalize!(data, A::AbstractMatrix) = data

function maybe_normalize!(data, C::NormalizedMatrix)
  T = eltype(C)
  BLAS.syr!('L', -T(C.n_obs), C.shift, data)
  ldiv!(Diagonal(C.scale), data)
  rdiv!(data, Diagonal(C.scale))
  return data # AtApD = S⁻¹(AᵀA - nāāᵀ)S⁻¹
end
