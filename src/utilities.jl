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

run_power_method(M; maxiter = maximum(size(M))) = run_power_method!(randn(size(M, 1)), M; maxiter=maxiter)

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
    __matvec__(z, A, q)         # wrapper for computing matvec product; see Misc section below
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

"""
    lmax_SASmI!(S_A1, s, w_p1, w_p2, out; iters=4, v0=nothing)

Estimate `λ_max(S*A1*S - I)` with a few power iterations.
Returns `(λmax, v)` where `v` is the final normalized vector.
"""
function lmax_SASmI_W!(W, s, w_p1, w_p2, out, w_k; iters::Int=4, v0=nothing)
  T = eltype(s)
  p = length(s)
  v = v0 === nothing ? randn(T, p) : copy(v0)
  v ./= max(norm(v), eps(T))
  
  λmax = zero(T)
  @inbounds for _ in 1:iters
    apply_SASmI_W!(out, v, W, s, w_p1, w_p2, w_k)
    nrm = norm(out)
    if nrm == 0
      λmax = zero(T)
      break
    end
    @. v = out / nrm
    λmax = dot(v, out)                           
  end
  return λmax, v
end

"""
    apply_SASmI_W!(out, v, S_A1, s, w_p1, w_p2)

Compute `out = (S*A1*S - I) * v` in-place.

- `S_A1`: `Symmetric(A1, :U)`; only upper of `A1` is valid.
- `s`: diagonal of `S = diag(1 ./ sqrt.(diagA1))`.
- `w_p1`, `w_p2`: p-vectors preallocated work buffers.

Returns `out`.
"""
function apply_SASmI_W!(out, v, W, s, w_p1, w_p2, w_k)
  @inbounds @simd for i in eachindex(v)           # w_p1 = S * v
    w_p1[i] = s[i] * v[i]
  end
  mul!(w_k, transpose(W), w_p1)                   # w_k = W' * (S*v)         (k)
  mul!(w_p2, W, w_k)                              # w_p2 = W * w_k           (p)
  @inbounds @simd for i in eachindex(v)           # out = S * (-w_p2) - v
    out[i] = s[i] * (-w_p2[i]) - v[i]
  end
  return out
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

#
# Application of Woodbury and Sherman-Morrison Formulas
#

function woodbury_inverse(V::Matrix{Float64}, λ::Vector{Float64}, μ::Vector{Float64})
  # A = V * Diagonal(λ) * V' + Diagonal(μ)
  Dinv = Diagonal(1.0 ./ μ)
  Cinv = Diagonal(1.0 ./ λ)
  
  middle = Cinv + V' * Dinv * V
  middle_inv = inv(middle)
  
  Ainv = Dinv - Dinv * V * middle_inv * V' * Dinv
  return Ainv
end

function woodbury_solve(V::Matrix{Float64}, λ::Vector{Float64}, μ::Vector{Float64}, u::Vector{Float64})
  # Efficiently compute A⁻¹ * u where A = V*Diagonal(λ)*V' + Diagonal(μ)
  
  @assert size(V, 1) == length(μ) == length(u)
  @assert size(V, 2) == length(λ)
  
  Dinv = Diagonal(1.0 ./ μ)
  Cinv = Diagonal(1.0 ./ λ)
  
  # Step 1: z = D⁻¹ * u
  z = Dinv * u
  
  # Step 2: B = C⁻¹ + V' * D⁻¹ * V
  B = Cinv + V' * Dinv * V
  
  # Step 3: rhs = V' * z
  rhs = V' * z
  
  # Step 4: solve B \ rhs
  temp = B \ rhs
  
  # Step 5: final result
  result = z - Dinv * V * temp
  
  return result
end

"""
    woodbury_solve!(d, V, λ, μ, u, w_p1, w_p2, Msmall)

Solve for `d` in:
    d = (Diag(μ) + V*Diag(λ)*V') \\ u
using the Woodbury identity. Only the small k×k system is factorized.

- `V` is m×k
- `λ` is length k
- `μ` is length m (diagonal block)
- `u` is length m
- `w_p1`, `w_p2` are m-length workspaces
- `Msmall` is k×k (workspace for the small system)

No heap allocations besides what the linear solver needs.
"""
function woodbury_solve!(d::AbstractVector{T},
  V::AbstractMatrix{T},
  λ::AbstractVector{T},
  μ::AbstractVector{T},
  u::AbstractVector{T},
  w_p1::AbstractVector{T},
  w_p2::AbstractVector{T},
  Msmall::AbstractMatrix{T},
  ) where {T<:Real}
  #
  m = size(V, 1)           # = |F|
  k = size(V, 2)           # rank
  
  # Safety clamps to avoid division by 0 / Inf / NaN.
  ϵ = eps(T)
  @views λsafe = max.(λ, ϵ)
  @views μsafe = max.(μ, ϵ)
  
  # w_p1 = D^{-1} u
  @views @. w_p1[1:m] = u / μsafe
  
  # Msmall = Diag(λ)^{-1} + V' * D^{-1} * V
  @views Ms = Msmall[1:k, 1:k]
  fill!(Ms, zero(T))  # clear stale values
  
  @views begin
    for j in 1:k
      @. w_p2[1:m] = V[:, j] / μsafe
      Ms[:, j] = transpose(V) * view(w_p2, 1:m)
    end
    for i in 1:k
      Ms[i, i] += inv(λsafe[i])
    end
  end
  
  # Solve Ms * w = V' * (D^{-1} u)
  rhs = transpose(V) * view(w_p1, 1:m)
  w = Ms \ rhs
  
  # d = D^{-1} u - D^{-1} V w
  z = V * w
  @views @. d = w_p1[1:m] - z / μsafe
  
  return d
end

#
# Misc
#
function generate_decay_correlated_matrix(n, p, base_rho)
  Σ = [base_rho^abs(i - j) for i in 1:p, j in 1:p]
  L = cholesky(Σ).L  
  Z = rand(n, p)
  X = Z * L  
  return X
end

function mask_set(x, grad)
  return findall(x .< 1e-10 .&& grad .> 0)
end

function active_set(x, grad)
  return findall(.!(x .< 1e-10 .&& grad .> 0)) 
end

"""
    col_sumsq!(d, X)

Compute `d[j] = sum(X[:,j].^2)`; i.e., the diagonal of `X'X`.
Threaded loop; no allocations besides input/output.
"""
function col_sumsq!(d::AbstractVector{T}, X::AbstractMatrix{T}) where {T<:AbstractFloat}
  @assert length(d) == size(X,2)
  @threads for j in 1:size(X,2)
    s = zero(T)
    @inbounds @simd for i in 1:size(X,1)
      x = X[i,j]
      s += x*x
    end
    d[j] = s
  end
  return d
end

"""
    loss_ls(X, y, β)

Least-squares loss: `0.5 * ||Xβ - y||^2`.
"""
loss_ls(X, y, β) = 0.5 * norm(X*β - y)^2

"""
    pg_residual(X, y, β, t)

Prox-gradient residual for the ℓ1-ball formulation (∞-norm).
"""
pg_residual(X, y, β, t) = norm(β .- proj_l1_ball(β .- X'*(X*β - y), t), Inf)

# if A is a matrix, just dispatch to mul!()
__matvec__(out, A::AbstractMatrix, x) = mul!(out, A, x)

# otherwise, assume A is a function-like object with two arguments
__matvec__(out, Afun!, x) = Afun!(out, x)

