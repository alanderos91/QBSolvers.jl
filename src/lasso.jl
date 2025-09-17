#
# QUB
#
"""
    lasso_prox_newton_woodbury!(X, y; t, maxiter=100, ∇tol=1e-6,
                                eigen_k=1, eigen_iters=10,
                                α0=one(T), backtrack=T(0.5), c1=T(1e-4),
                                verbose=false, v0=nothing)

Projected (ℓ₁-ball) Prox-Newton method for Lasso using a low-rank (Lanczos) 
approximation and a Woodbury sub-solve on the free set.

Inputs
- `X :: AbstractMatrix{T}` : data matrix (n × p)
- `y :: AbstractVector{T}` : response (n)
- `t :: T`                 : ℓ₁-ball radius (constraint form)

Keyword parameters
- `maxiter, ∇tol`          : outer-iteration cap & relative-change tolerance
- `eigen_k, eigen_iters`   : Lanczos rank & iterations for low-rank approx
- `α0, backtrack, c1`      : line-search params (initial step, shrink, Armijo c₁)
- `verbose`                : print diagnostics (unused except for custom prints)
- `v0`                     : optional Lanczos starting vector (unused here)

Returns
- `(β, niters)` where `β` is the final coefficient vector (sparse) and `niters`
  the number of outer iterations performed.

Notes
- The function assumes `BlockedLS(X, y)` provides fields/buffers used below.
- Logic preserved exactly from the provided code; only formatting and comments improved.
"""
function lasso_prox_newton_woodbury!(
  X::AbstractMatrix{T},
  y::AbstractVector{T};
  t::T, maxiter::Int=100,
  ∇tol::Real=1e-6,
  eigen_k::Int=1,
  eigen_iters::Int=10,
  α0::T=one(T),
  backtrack::T=T(0.5),
  c1::T=T(1e-4),
  verbose::Bool=false,
  v0::Union{Nothing,AbstractVector{T}}=nothing
  ) where {T<:AbstractFloat}
  # -- Problem/context & workspace from helper type
  lasso = BlockedLS(X, y)
  n, p = size(X)
  
  # -- Lanczos / low-rank buffers
  V      = nothing                         # Ritz vectors
  λ      = nothing                         # Ritz eigenvalues
  W      = Matrix{T}(undef, p, eigen_k)    # V * sqrt.(λ)
  
  # -- Vectors (views/buffers reused across steps)
  s_scale = similar(lasso.β)               # S diagonal (1 ./ sqrt(diagA))
  outS    = similar(lasso.β)               # work for power iteration output
  β_new   = similar(lasso.β)               # tentative iterate (then sparse)
  Diag    = similar(lasso.β)               # (1+λmax)*diagA
  
  # -- Indices & small system
  idxF    = Int[]                          # free set indices
  Msmall  = Matrix{T}(undef, eigen_k, eigen_k)
  
  # -- Initialize views
  copyto!(β_new,   lasso.β)
  copyto!(s_scale, lasso.β)
  copyto!(outS,    lasso.β)
  copyto!(Diag,    lasso.β)
  
  # Switch both β and β_new to sparse storage (as in original logic)
  β_new   = sparse(β_new)
  lasso.β = sparse(lasso.β)
  
  # ------------------------------------------------------------------
  # 1) Lanczos on P = X'X via operator Afun(q) = X'*(X*q)
  # ------------------------------------------------------------------
  Afun! = let X = X, tmp_n = lasso.storage_n_1
    function(out, v)
      mul!(tmp_n, X, v)               # n-vector
      mul!(out, transpose(X), tmp_n)  # p-vector
      return out
    end
  end

  v1 = randn(T, p)
  v1 ./= norm(v1)
  λ, V = lanczos_ritz_fixed!(Afun!, v1; maxiter=eigen_iters, k=eigen_k)

  # W = V * sqrt.(λ) (column-wise scaling)
  @inbounds for j in 1:eigen_k
    sλ = λ[j] <= 0 ? zero(T) : sqrt(λ[j])
    @. W[:, j] = V[:, j] * sλ
  end

  #     # A ← - W W'  (upper only); then wrap as symmetric
  #     BLAS.syrk!('U', 'N', -one(T), W, one(T), A)
  #     A = Symmetric(A, :U)

  # diagA = diag(X'X) - diag(W W')
  col_sumsq!(lasso.storage_p_1, X)           # diag(X'X)
  col_sumsq!(lasso.storage_p_2, transpose(W))# diag(W W')
  @. lasso.storage_p_1 -= lasso.storage_p_2
  copyto!(Diag, lasso.storage_p_1)

  # S diagonal: s_scale = 1 ./ sqrt(diagA)
  @. s_scale = 1 / sqrt(lasso.storage_p_1)

  w_k = zeros(T, size(W,2))
  λmax, _ = lmax_SASmI_W!(W, s_scale,
  lasso.storage_p_1,  # p work
  lasso.storage_p_2,  # p work
  outS,               # p work / 输出
  w_k; iters=4)

  # Diag ← (1 + λmax) * diagA (block diagonal used in subproblem)
  @. Diag *= (one(T) + λmax)

  # ------------------------------------------------------------------
  # 2) Initialize objective and gradient at current β
  # ------------------------------------------------------------------
  copyto!(lasso.grad, -lasso.xty)           # grad = Pβ - X'y ; start with -X'y (β=0)
  y2   = T(0.5) * dot(lasso.y, lasso.y)
  fβ   = y2
  niters = maxiter
  α     = 2.0
  sizehint!(idxF, p)

  # ------------------------------------------------------------------
  # 3) Main loop: free-set Newton + ℓ₁ projection line search
  # ------------------------------------------------------------------
  for it in 1:maxiter
    # -- Free set F via ℓ₁-ball projection threshold τ
    #         @. lasso.storage_p_1 = lasso.β - α0 * lasso.grad
    copy!(lasso.storage_p_1, lasso.β)                # s ← β
    BLAS.axpy!(-α0, lasso.grad, lasso.storage_p_1)  # s ← s + (-α) * direction
    
    τ = _l1_proj_threshold(lasso.storage_p_1, t)
    
    if sum(abs, lasso.β) < t - 1e-12
      resize!(idxF, p)
      @inbounds for i in 1:p
        idxF[i] = i
      end
    else
      empty!(idxF)
      τp  = τ + 1e-10
      idxF = findall(x -> abs(x) > τp, lasso.storage_p_1) 
      
    end
    
    # -- Woodbury sub-solve on F to get Newton direction (lasso.direction[F])
    fill!(lasso.direction, zero(T))
    if !isempty(idxF)
      F  = idxF
      VI = @view V[F, :]
      μI = @view Diag[F]
      gI = @view lasso.grad[F]
      kI = size(VI, 2)
      woodbury_solve!(@view(lasso.direction[F]),
      VI, @view(λ[1:kI]), μI, gI,
      @view(lasso.storage_p_1[F]), @view(lasso.storage_p_2[F]),
      @view(Msmall[1:kI, 1:kI]))
    end
    
    # -- Backtracking line search with ℓ₁-ball projection
    accepted = false
    while α ≥ T(1e-12)
      #             @. lasso.storage_p_1 = lasso.β - α * lasso.direction
      copy!(lasso.storage_p_1, lasso.β)                # s ← β
      BLAS.axpy!(-α, lasso.direction, lasso.storage_p_1)  # s ← s + (-α) * direction
      
      proj_l1_ball!(β_new, lasso.storage_p_1, t)
      
      # Numerical cleanup -> sparse (as in original logic)
      β_new = map(x -> abs(x) < 1e-8 ? 0.0 : x, β_new)
      β_new = sparse(β_new)
      
      # f(β_new) = 0.5 β' P β - (X'y)' β + 0.5‖y‖²
      mul!(lasso.storage_n_1, lasso.X, β_new)
      mul!(lasso.storage_p_1, transpose(X), lasso.storage_n_1)
      fnew = T(0.5) * dot(β_new, lasso.storage_p_1) - dot(lasso.xty, β_new) + y2
      
      decr = c1 * α * dot(lasso.grad, lasso.direction)
      if fnew ≤ fβ - decr
        fβ = fnew
        accepted = true
        break
      end
      α *= backtrack
    end
    
    #         println("α: ", α)
    
    # -- Fallback: proximal gradient (soft-threshold on unconstrained step)
    if !accepted
      #             @. lasso.storage_p_1 = lasso.β - α * lasso.direction
      copy!(lasso.storage_p_1, lasso.β)                # s ← β
      BLAS.axpy!(-α, lasso.direction, lasso.storage_p_1)  # s ← s + (-α) * direction
      
      soft_threshold!(β_new, lasso.storage_p_1, α * t)
      mul!(lasso.storage_n_1, X, β_new)
      mul!(lasso.storage_p_1, transpose(X), lasso.storage_n_1)
      fβ = T(0.5) * dot(β_new, lasso.storage_p_1) - dot(lasso.xty, β_new) + y2
    end
    
    # -- Convergence: relative change in β
    if norm(β_new - lasso.β) / max(norm(lasso.β), eps(T)) < ∇tol
      niters = it
      lasso.β .= β_new
      break
    end
    
    # -- Accept step + refresh gradient (grad = Pβ - X'y)
    copyto!(lasso.β, β_new)
    @. lasso.grad = lasso.storage_p_1 - lasso.xty  # storage_p_1 currently holds Pβ
  end

  return lasso.β, niters, V
end
#
# FISTA
#
"""
    lipschitz_XTX(X; iters=50)

Estimate L = λ_max(X'X) by power iteration with small allocations.
"""
function lipschitz_XTX(X; iters::Int=50)
  n, p = size(X)
  T = eltype(X)
  v = randn(T, p); v ./= max(norm(v), eps(T))
  
  # Work buffers
  w_n = Vector{T}(undef, n)          # w_n = X * v
  w_p = Vector{T}(undef, p)          # w_p = X' * w_n
  
  for _ in 1:iters
    mul!(w_n, X, v)                # w_n = X*v
    mul!(w_p, transpose(X), w_n)   # w_p = X'*(X*v)
    nv = norm(w_p)
    nv == 0 && break
    @. v = w_p / nv
  end
  
  # Rayleigh quotient: v'*(X'X)*v
  mul!(w_n, X, v)
  mul!(w_p, transpose(X), w_n)
  return dot(v, w_p)
end

"""
    β, iters = fista_l1_ball(X, y; t, maxiter=1000, tol=1e-6, L=nothing)

Solve  min 0.5*||Xβ - y||²   s.t. ||β||₁ ≤ t
with FISTA (Nesterov) and ℓ1-ball projection. β, yk kept as `SparseVector`.

- If `L === nothing`, estimates `L = λ_max(X'X)` by power iteration.
- Stopping: rel change `‖β_new-β‖ / max(‖β‖, eps)` < `tol`.
"""
function fista_l1_ball(X::AbstractMatrix{T}, y::AbstractVector{T};
  t::Real, maxiter::Int=1000, tol::Real=1e-6, L=nothing) where {T<:Real}
  n, p = size(X)
  L === nothing && (L = lipschitz_XTX(X))
  
  # State (sparse)
  β  = spzeros(T, p)
  yk = copy(β)
  
  # Dense work buffers (reused)
  r  = similar(y)                # residual r = X*yk - y
  g  = zeros(T, p)               # gradient g = X' * r
  z  = zeros(T, p)               # z = yk - (1/L) * g
  
  tk = one(T)
  for k in 1:maxiter
    # Gradient: g = X'*(X*yk - y)
    mul!(r, X, yk)            # uses SparseVector yk efficiently
    @. r = r - y
    mul!(g, transpose(X), r)
    
    # Prox on ℓ1-ball → sparse β_new
    @. z = yk - (one(T)/L) * g
    β_new = proj_l1_ball_sparse(z, t)
    
    # Numerical cleanup -> sparse (as in original logic)
    β_new = map(x -> abs(x) < 1e-8 ? 0.0 : x, β_new)
    β_new = sparse(β_new)
    
    # Nesterov momentum
    tnew = (1 + sqrt(1 + 4tk^2))/2
    @. yk = β_new .+ ((tk - 1)/tnew) .* (β_new .- β)
    
    # Stopping
    rel = norm(β_new - β) / max(norm(β), eps(T))
    @. β   .= β_new
    tk  = tnew
    if rel < tol
      return β, k
    end
  end
  return β, maxiter
end
#
# Projected Newton
#
"""
    projected_newton_l1!(
        X, y;
        t, maxiter=100, ∇tol=1e-6,
        eigen_k=1, eigen_iters=10,
        α0=1.0, backtrack=0.5, c1=1e-4, δ=1e-8,
        verbose=false, v0=nothing
    )

Projected-Newton (constraint form) for:

    minimize   0.5 * ‖Xβ - y‖²               (implemented via 0.5 β' (X'X) β - (X'y)' β + 0.5‖y‖²)
    subject to ‖β‖₁ ≤ t

This in-place variant uses a preallocated `BlockedLS` workspace (user-defined type) to
avoid repeated allocations.

High-level procedure per iteration (logic unchanged):
1. **Free-set detection** via one-shot ℓ₁-ball projection threshold:
   - Form z = β - α₀ * ∇f(β) and compute τ = _l1_proj_threshold(z, t).
   - If ‖β‖₁ < t - 1e-12: all indices are free; otherwise F = { i : |zᵢ| > τ + 1e-10 }.
2. **Newton sub-solve on F** (direction stored in `lasso.direction[F]`):
   - Solve the damped normal equations (X_F' X_F + δ I) d_F = (H \\ g_F) (as in your code),
     then use backtracking with ℓ₁-ball projection.
3. **Line search** with backtracking:
   - Evaluate projected trial `β_new` and accept if sufficient decrease (Armijo-style
     with your chosen sign convention: `fnew ≤ fβ - c1 * α * (g⋅d)`).
   - If line search fails, fallback to a **proximal gradient** step.
4. **Convergence**:
   - Stop when relative change in coefficients is below `∇tol`.

Arguments:
- `X::AbstractMatrix{T}`, `y::AbstractVector{T}`: data.
- `t::T`: ℓ₁-ball radius.
- `maxiter`, `∇tol`: outer-loop controls.
- `eigen_k`, `eigen_iters`: kept for signature compatibility (unused here).
- `α0`, `backtrack`, `c1`: line-search parameters.
- `δ`: damping added to the normal equations on the free set.
- `verbose`, `v0`: preserved for interface compatibility.

Returns:
- `(β, niters)` with `β` stored inside the `BlockedLS` workspace and the number of iterations.
"""
function projected_newton_l1!(
  X::AbstractMatrix{T},
  y::AbstractVector{T};
  t::T, maxiter::Int=100,
  ∇tol::T=T(1e-6),
  eigen_k::Int=1,              # kept for signature compatibility
  eigen_iters::Int=10,         # kept for signature compatibility
  α0::T=one(T),
  backtrack::T=T(0.5),
  c1::T=T(1e-4),
  δ::T=T(1e-8),
  verbose::Bool=false,
  v0::Union{Nothing,AbstractVector{T}}=nothing  # kept for signature compatibility
  ) where {T<:AbstractFloat}
  
  # ------------------------------------------------------------------
  # 0) Construct workspace holder (must be provided by user code)
  #    BlockedLS is assumed to preallocate:
  #      β, grad, direction, storage_p_1, storage_p_2, storage_n_1, etc.
  # ------------------------------------------------------------------
  lasso = BlockedLS(X, y)
  n, p = size(X)
  
  # ------------------------------------------------------------------
  # 1) Preallocate low-rank/Lanczos related buffers (kept to match your signature;
  #    not used in the simplified projected-Newton path)
  # ------------------------------------------------------------------
  V      = Matrix{T}(undef, p, eigen_k)
  λ      = Vector{T}(undef, eigen_k)
  W      = Matrix{T}(undef, p, eigen_k)
  A      = zeros(T, p, p)
  
  # Vectors re-used across steps (kept for compatibility with your workspace style)
  s_scale = similar(lasso.β)    # not used in this function
  outS    = similar(lasso.β)    # not used in this function
  β_new   = similar(lasso.β)    # trial/accepted iterate
  Diag    = similar(lasso.β)    # not used in this function
  
  # Free-set and small system buffers
  idxF   = Int[]                # free-set indices
  Msmall = Matrix{T}(undef, eigen_k, eigen_k)  # not used here
  
  # Initialize views (no-op copies to align with your original code style)
  copyto!(β_new,   lasso.β)
  copyto!(s_scale, lasso.β)
  copyto!(outS,    lasso.β)
  copyto!(Diag,    lasso.β)
  
  # Keep β in sparse format as in your original logic
  β_new   = sparse(β_new)
  lasso.β = sparse(lasso.β)
  
  # ------------------------------------------------------------------
  # 2) Initialize objective and gradient at current β
  #    Using f(β) = 0.5*β' (X'X) β - (X'y)' β + 0.5‖y‖²
  #    grad(β) = X'X β - X'y
  # ------------------------------------------------------------------
  copyto!(lasso.grad, -lasso.xty)  # start at β=0 ⇒ grad = -X'y
  y2   = T(0.5) * dot(lasso.y, lasso.y)
  fβ   = y2
  niters = maxiter
  α     = α0
  
  # ------------------------------------------------------------------
  # 3) Main loop: free-set Newton + ℓ₁ projection line search
  # ------------------------------------------------------------------
  for it in 1:maxiter
    # ---- Free set F via ℓ₁-ball projection threshold τ (unchanged logic) ----
    @. lasso.storage_p_1 = lasso.β - α0 * lasso.grad
    τ = _l1_proj_threshold(lasso.storage_p_1, t)
    
    if sum(abs, lasso.β) < t - 1e-12
      resize!(idxF, p)
      @inbounds for i in 1:p
        idxF[i] = i
      end
    else
      empty!(idxF)
      τp  = τ + 1e-10
      idxF = findall(x -> abs(x) > τp, lasso.storage_p_1) 
    end
    
    # ---- Newton sub-solve on F → lasso.direction[F] (unchanged logic) ----
    fill!(lasso.direction, zero(T))
    if !isempty(idxF)
      F  = idxF
      fill!(lasso.direction, zero(T))
      
      # Normal equations (with damping δ only, as in your code):
      # H_F = X_F' X_F + δ I; direction[F] = H_F \ grad[F]
      Hf = (lasso.X[:, F]' * lasso.X[:, F]) + δ * I
      lasso.direction[F] = (Hf \ lasso.grad[F])
    end
    
    # ---- Backtracking line search with ℓ₁-ball projection (unchanged logic) ----
    accepted = false
    while α ≥ T(1e-12)
      @. lasso.storage_p_1 = lasso.β - α * lasso.direction           # tentative step
      proj_l1_ball!(β_new, lasso.storage_p_1, t)                      # project to ℓ₁ ball
      
      # Numerical cleanup → sparse (keep original pattern)
      β_new = map(x -> abs(x) < 1e-8 ? 0.0 : x, β_new)
      β_new = sparse(β_new)
      
      # f(β_new) = 0.5 β' (X'X) β - (X'y)' β + 0.5‖y‖²
      mul!(lasso.storage_n_1, lasso.X, β_new)                         # Xβ_new
      mul!(lasso.storage_p_1, transpose(X), lasso.storage_n_1)        # (X'X)β_new
      fnew = T(0.5) * dot(β_new, lasso.storage_p_1) - dot(lasso.xty, β_new) + y2
      
      # Armijo-like decrease with your sign convention: fnew ≤ fβ - c1*α*(g⋅d)
      decr = c1 * α * dot(lasso.grad, lasso.direction)
      if fnew ≤ fβ - decr
        fβ = fnew
        accepted = true
        break
      end
      α *= backtrack
    end
    
    # ---- Fallback: proximal gradient step (unchanged logic) ----
    if !accepted
      @. lasso.storage_p_1 = lasso.β - α * lasso.direction
      soft_threshold!(β_new, lasso.storage_p_1, α * t)
      
      mul!(lasso.storage_n_1, X, β_new)
      mul!(lasso.storage_p_1, transpose(X), lasso.storage_n_1)
      fβ = T(0.5) * dot(β_new, lasso.storage_p_1) - dot(lasso.xty, β_new) + y2
    end
    
    # ---- Convergence: relative change in β (unchanged) ----
    if norm(β_new - lasso.β) / max(norm(lasso.β), eps(T)) < ∇tol
      niters = it
      lasso.β .= β_new
      break
    end
    
    # ---- Accept step and refresh gradient: grad = (X'X)β - X'y (unchanged) ----
    copyto!(lasso.β, β_new)
    @. lasso.grad = lasso.storage_p_1 - lasso.xty   # storage_p_1 currently holds (X'X)β
  end
  
  return lasso.β, niters
end

