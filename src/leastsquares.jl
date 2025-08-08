###
### Initialization
###

function init_recurrences!(workspace, AtApI::GramPlusDiag, b, H)
  # Unpack
  (x, g, d, w, x_prev) = workspace

  # Initialize the difference, d₁ = x₁ - x₀
  T = eltype(AtApI)
  A = AtApI.A
  # -∇₀ = Aᵀ⋅r - λx
  if length(AtApI.AtA) > 0
    mul!(g, AtApI, x)
    mul!(g, transpose(A), b, one(T), -one(T))
  else
    r = AtApI.tmp
    mul!(r, A, x)
    @. r = b - r
    mul!(g, transpose(A), r)
    AtApI.beta > 0 && (@. g = g - AtApI.beta*x)
  end

  ldiv!(d, H, g) # d₁ = H⁻¹(-∇₀)
  @. x = x + d   # x₁ = x₀ + d₁

  return nothing
end

###
### Implementation
###

function __OLS_loop__(workspace, linmaps, gtol, maxiter, iter = 1, k = 1, accel = false)
  # unpack
  x, g, d, w, x_prev = workspace
  AtApI, H = linmaps
  converged = norm(g) <= gtol
  @. x_prev = x
  while !converged && (iter < maxiter)
    iter += 1

    # Compute (AᵀA+λI) dₙ and -∇ₙ; assumes g is always -∇
    mul!(w, AtApI, d)
    @. g = g - w
    converged = norm(g) <= gtol

    # Update difference dₙ₊₁ = H⁻¹gₙ
    ldiv!(d, H, g)

    # Update coefficients
    @. x = x + d

    if accel
      k += 1
      c = (k-1)/(k+2)
      @. w = x - x_prev
      @. x_prev = x
      @. d = d + c*w  # gₖ₊₁ = gₖ - (AᵀA+λI)[dₖ₊₁ + (k-1)/(k+2)(xₖ - xₖ₋₁)]
      @. x = x + c*w  # zₖ = xₖ + (k-1)/(k+2) * (xₖ - xₖ₋₁)
    end
  end

  maybe_unscale!(x, AtApI.A)

  return iter, converged
end

function solve_OLS(A::AbstractMatrix{T}, b::AbstractVector{T};
  lambda::Float64   = zero(T),
  normalize::Symbol = :none,
  gram::Bool        = _cache_gram_heuristic_(A),
  maxiter::Int      = maximum(size(A)),
  accel::Bool       = false,
  tol::Float64      = 1e-3 * sqrt(size(A, 2)),
) where T
  #
  n_obs, n_var = size(A)
  @assert lambda >= 0

  # linear maps
  AtA = GramPlusDiag(A; gram=gram)  # may cache AtA
  
  # workspace
  x = similar(b, T, n_var); fill!(x, zero(T)); x_prev = similar(b, T, n_var)
  g = similar(b, T, n_var)
  d = similar(b, T, n_var)
  w = similar(b, T, n_var)
  workspace = (x, g, d, w, x_prev)

  #
  # We cannot retrieve the QUB matrix directly due to type-instability in the way it is created.
  # This creates a closure taking the QUB matrix H as an input that will be invoked below by
  # the with_qub_matrix() function.
  #
  run = let
    function(AtApI, H)
      init_recurrences!(workspace, AtApI, b, H)
      __OLS_loop__(workspace, (AtApI, H), tol, maxiter, 1, 1, accel)
    end
  end
  iter, converged = with_qub_matrix(run, AtA, lambda, normalize)

  # final residual
  r = copy(b)
  mul!(r, A, x, -1.0, 1.0)

  stats = (
    iterations = iter,
    converged = converged,
    xnorm = norm(x),
    rnorm = norm(r),
    gnorm = norm(g),
  )

  return x, r, stats
end

function __OLS_lbfgs__(workspace, linmaps, gtol, maxiter, iter = 0)
  x, g, d, w, cache = workspace
  A, b, AtApI, H = linmaps
  T = eltype(A)

  # Initialize gradient
  mul!(g, AtApI, x) # -∇₀ = Aᵀ⋅r - λx
  mul!(g, transpose(A), b, one(T), -one(T))

  # Iterate the algorithm map
  converged = norm(g) <= gtol
  alpha = one(T)

  while !converged && (iter < maxiter)
    iter += 1

    # Update the LBFGS workspace and compute the next direction
    iter > 1 && update!(cache, alpha, d, g)
    compute_lbfgs_direction!(d, g, cache, H)

    # Compute (AᵀA + λI) dₙ₊₁
    mul!(w, AtApI, d)

    # Backtrack to make sure we satisfy descent
    # lossₙ₊₁ = lossₙ + α²/2 (|Adₙ₊₁|² + λ|dₙ₊₁|²) + α (∇ₙᵀdₙ₊₁)
    alpha = one(T)
    loss_1 = 1//2 * dot(d, w) # 1/2 [|Adₙ₊₁|² + λ|dₙ₊₁|²]
    loss_2 = -dot(g, d)       # ∇ₙᵀdₙ₊₁
    if loss_2 > 0 error("L-BFGS direction was not computed correctly at iteration $(iter)") end
    while (alpha*alpha*loss_1 + alpha*loss_2 > 0)
      alpha = 1//2 * alpha
    end
    @. x = x + alpha*d

    # Save the old gradient
    @. cache.q = g

    # Update -∇ₙ₊₁ = -∇ₙ - α (AᵀA + λI) dₙ₊₁
    @. g = g - alpha*w
    converged = norm(g) <= gtol
  end

  maybe_unscale!(x, AtApI.A)

  return iter, converged
end

function solve_OLS_lbfgs(A::AbstractMatrix{T}, b::Vector{T};
  lambda::Float64   = zero(T),
  normalize::Symbol = :none,
  precond::Symbol   = :none,
  gram::Bool        = _cache_gram_heuristic_(A),
  maxiter::Int      = maximum(size(A)),
  tol::Float64      = 1e-3 * sqrt(size(A, 2)),
  memory::Int       = 10,
) where T
  #
  n_obs, n_var = size(A)
  @assert lambda >= 0

  AtA = GramPlusDiag(A; gram=gram)              # may cache AtA
  AtApI = GramPlusDiag(AtA, one(T), T(lambda))  # same data, add lazy shift by λI

  # workspace
  x = zeros(T, n_var)
  g = zeros(T, n_var)
  d = zeros(T, n_var)
  w = zeros(T, n_var)
  cache = LBFGSCache{T}(n_var, memory)
  workspace = (x, g, d, w, cache)

  # Type-stability trick
  if precond == :none
    iter, converged = __OLS_lbfgs__(workspace, (A, b, AtApI, I), tol, maxiter, 0)
  elseif precond == :qub
    run = let b = b, workspace = workspace, tol = tol, maxiter = maxiter
      function(AtApI, H)
        __OLS_lbfgs__(workspace, (AtApI.A, b, AtApI, H), tol, maxiter, 0)
      end
    end
    iter, converged = with_qub_matrix(run, AtA, lambda, normalize)
  end

  # final residual
  r = copy(b)
  mul!(r, A, x, -1.0, 1.0)

  stats = (
    iterations = iter,
    converged = converged,
    xnorm = norm(x),
    rnorm = norm(r),
    gnorm = norm(g),
  )

  return x, r, stats
end

###
### Wrappers
###

function solve_OLS_qr(A, b)
  #
  size(A, 1) >= size(A, 2) || error("Need n ≥ p; got n = $(size(A, 1)), p = $(size(A, 2))")
  T = eltype(A)

  x = A \ b

  r = copy(b)
  mul!(r, A, x, -one(T), one(T))
  g = transpose(A) * r  # -∇ = Aᵀr

  stats = (
    iterations = 1,
    converged = true,
    xnorm = norm(x),
    rnorm = norm(r),
    gnorm = norm(g),
  )

  return x, r, stats
end

function solve_OLS_chol(A, b; lambda::Real=0.0)
  #
  lambda >= 0 || error("Regularization λ must be non-negative. Got: $(lambda)")
  T = eltype(A)

  At = transpose(A)
  AtA = Symmetric(At * A + lambda*I)
  chol = cholesky!(AtA)
  x = chol \ (At*b)

  r = copy(b)
  mul!(r, A, x, -one(T), one(T))
  g = copy(x)
  mul!(g, transpose(A), r, one(T), -lambda) # -∇ = Aᵀr - λx

  stats = (
    iterations = 1,
    converged = true,
    xnorm = norm(x),
    rnorm = norm(r),
    gnorm = norm(g),
  )

  return x, r, stats
end

function solve_OLS_lsmr(A, b;
  x0 = IterativeSolvers.zerox(A, b),
  lambda::Real=0.0,
  kwargs...
)
  #
  lambda >= 0 || error("Regularization λ must be non-negative. Got: $(lambda)")
  T = eltype(A)
  
  x, ch = lsmr!(x0, A, b; λ=sqrt(lambda), log=true, kwargs...)

  r = copy(b)
  mul!(r, A, x, -one(T), one(T))
  g = copy(x)
  mul!(g, transpose(A), r, one(T), -lambda) # -∇ = Aᵀr - λx

  stats = (
    iterations = ch.iters,
    converged = ch.isconverged,
    xnorm = norm(x),
    rnorm = norm(r),
    gnorm = norm(g),
  )

  return x, r, stats
end

function solve_OLS_lsqr(A, b;
  x0 = IterativeSolvers.zerox(A, b),
  lambda::Real=0.0,
  kwargs...
)
  #
  lambda >= 0 || error("Regularization λ must be non-negative. Got: $(lambda)")
  T = eltype(A)

  x, ch = lsqr!(x0, A, b; damp=sqrt(lambda), log=true, kwargs...)

  r = copy(b)
  mul!(r, A, x, -one(T), one(T))
  g = copy(x)
  mul!(g, transpose(A), r, one(T), -lambda) # -∇ = Aᵀr - λx

  stats = (
    iterations = ch.iters,
    converged = ch.isconverged,
    xnorm = norm(x),
    rnorm = norm(r),
    gnorm = norm(g),
  )

  return x, r, stats
end

function solve_OLS_cg(A, b;
  x0 = IterativeSolvers.zerox(A, b),
  lambda::Real=0.0,
  gram::Bool = _cache_gram_heuristic_(A),
  kwargs...  
)
  #
  lambda >= 0 || error("Regularization λ must be non-negative. Got: $(lambda)")
  T = eltype(A)

  AtA = GramPlusDiag(A; gram=gram)
  AtApI = GramPlusDiag(AtA, one(T), T(lambda))

  x, ch = cg!(x0, AtApI, transpose(A)*b; log=true, kwargs...)

  r = copy(b)
  mul!(r, A, x, -one(T), one(T))
  g = copy(x)
  mul!(g, transpose(A), r, one(T), -lambda) # -∇ = Aᵀr - λx
  
  stats = (
    iterations = ch.iters,
    converged = ch.isconverged,
    xnorm = norm(x),
    rnorm = norm(r),
    gnorm = norm(g),
  )

  return x, r, stats
end

