#
# Functions for computing spectral radius
#
function estimate_spectral_radius(G::GramPlusDiag, J::Union{Diagonal,UniformScaling}; kwargs...)
  # M = AᵀA - J
  T = eltype(G)
  M = GramPlusDiag(
    G.A, G.AtA, J, G.n_obs, G.n_var, G.tmp, one(T), -one(T)
  )
  v = randn(size(G, 1))
  lambda, _, ch = powm!(M, v; log=true, kwargs...)
  return abs(lambda)
end

function estimate_spectral_radius(G, J::Union{Diagonal,UniformScaling}; kwargs...)
  # M = AᵀA - J
  T = eltype(G)
  M = GramPlusDiag(
    Matrix{T}(undef, 0, 0), G, J, 0, size(G, 1), Vector{T}(undef, 0), one(T), -one(T)
  )
  v = randn(size(G, 1))
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

"""
    simple_lanczos!(A, v; maxiter::Int = 1)

In-place Lanczos on linear map `A` with initial guess `v`.

- `A`: Support `mul!(z, A, q)` without allocations.
- `v`: Initial vector; will be normalized.
- `maxiter=1`: Number of iterations; sets the size of the Krylov basis.

Return `(a, β, Q)` for the tridiagonal `T = SymTridiagonal(α, β)` and basis `Q`.
"""
function simple_lanczos!(A, v; maxiter::Int = 1)
  # adapted from Caleb's code
  S = eltype(v)
  n = length(v)

  Q = zeros(S, n, maxiter)      # Krylov basis (columns)
  α = zeros(S, maxiter)         # main diagonal of T
  β = zeros(S, maxiter-1)       # off-diagonal of T

  q_prev = zeros(S, n)
  q = v / max(norm(v), eps(S))  # robust normalization
  Q[:, 1] = q

  z = similar(v)                # single reusable buffer z = A*q

  for k in 1:maxiter
    mul!(z, A, q)
    α[k] = dot(q, z)

    if k > 1
      @. z = z - β[k-1] * q_prev
    end
    @. z = z - α[k] * q

    # Full re-orthogonalization for robustness at small basis size
    @views for i in 1:k-1
      c = dot(Q[:, i], z)
      @. z = z - c*Q[:, i]
    end

    if k < maxiter
      β[k] = norm(z)
      if isapprox(β[k], zero(S))
        return (α[1:k], β[1:(k-1)], Q[:, 1:k])
      end
      @. q_prev = q
      @. q = z / β[k]
      Q[:, k+1] = q
    end
  end

  return α, β, Q
end

"""
    lanczos_ritz_fixed!(A, v; maxiter::Int=10, k::Int=1)

In-place Lanczos wrapper that returns top-k Ritz eigenpairs
via `T = SymTridiagonal(α, β)`. Lifts Ritz vectors as `V = Q*Z`.

- `A`: in-place operator `Afun!(z, q)`.
- `v`: initial vector (normalized inside).

Returns `(λ_ritz, V_ritz)`.
"""
function lanczos_ritz_fixed!(A, v; maxiter::Int=10, k::Int=1)
  # adapted from Caleb's code
  α, β, Q = simple_lanczos!(A, v; maxiter = maxiter)
  F = eigen!(SymTridiagonal(α, β))
  idxs = sortperm(F.values, rev=true)[1:k]
  λ_ritz = F.values[idxs]
  Z = F.vectors[:, idxs]
  V_ritz = Q * Z
  return λ_ritz, V_ritz
end

lanczos_ritz_fixed(A; kwargs...) = lanczos_ritz_fixed!(A, ones(eltype(A), size(A, 2)); kwargs...)

function eigenvalue_max(A::Matrix{T}; iters::Int = 10) where T
  n = size(A, 1)
  b = randn(T, n)
  for _ in 1:iters
    b = A * b
    b ./= norm(b)
  end
  return dot(b, A * b)
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
function _apply_scaling_(op, x, A::NormalizedMatrix)
  @. x = op(x, A.scale)
  return nothing
end

maybe_rescale!(x, A) = nothing
maybe_unscale!(x, A) = nothing

maybe_rescale!(x, A::NormalizedMatrix) = _apply_scaling_(*, x, A)
maybe_unscale!(x, A::NormalizedMatrix) = _apply_scaling_(/, x, A)

