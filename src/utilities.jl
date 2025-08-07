#
# Spectral radius of AtA - J, where J is a block diagonal Matrix based on AtA.
#
function estimate_spectral_radius(AtA, J; kwargs...)
  M = GramMinusBlkDiag(AtA, J)
  v = ones(size(J, 1))
  lambda, _, ch = powm!(M, v; log=true, kwargs...)
  return abs(lambda)
end

function estimate_spectral_radius(G, J::Union{Diagonal,UniformScaling}; kwargs...)
  # M = AᵀA - J
  T = eltype(G)
  M = GramPlusDiag(
    G.A, G.AtA, J, G.n_obs, G.n_var, G.tmp, one(T), -one(T)
  )
  v = ones(size(G, 1))
  lambda, _, ch = powm!(M, v; log=true, kwargs...)
  return abs(lambda)
end

function run_power_method!(v, M; maxiter = maximum(size(M)))
  normalize!(v)
  tmp = similar(v)
  lambda = Inf
  lambda_prev = zero(lambda)
  iter = 0
  while iter < maxiter && abs(lambda - lambda_prev) > 1e-3
    iter += 1
    lambda_prev = lambda
    @. tmp = v
    mul!(v, M, tmp)
    lambda = dot(tmp, v) / dot(tmp, tmp)
    normalize!(v)
  end
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

# need to consider cases where A is centered and/or scaled
function compute_main_diagonal(A::NormalizedMatrix, AtA)
  return one(eltype(A))*I
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
# Rescale solutions whenever NormalizedMatrix is used for the design matrix
#
_apply_scaling_(op, x, A::NormalizedMatrix) = (@. x = op(x, A.scale))

maybe_rescale!(x, A) = nothing
maybe_unscale!(x, A) = nothing

maybe_rescale!(x, A::NormalizedMatrix) = _apply_scaling_(*, x, A)
maybe_unscale!(x, A::NormalizedMatrix) = _apply_scaling_(/, x, A)

