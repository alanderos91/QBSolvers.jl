###
### Initialization
###

function init_recurrences!(workspace, AtApI::GramPlusDiag, b, H)
  # Unpack
  (x, g, d, w) = workspace

  # Initialize the difference, d₁ = x₁ - x₀
  T = eltype(AtApI)
  A = AtApI.A
  mul!(g, AtApI, x) # -∇₀ = Aᵀ⋅r - λx
  mul!(g, transpose(A), b, one(T), -one(T))

  ldiv!(d, H, g) # d₁ = H⁻¹(-∇₀)
  @. x = x + d   # x₁ = x₀ + d₁

  return nothing
end

###
### Implementation
###

function __OLS_loop__(workspace, linmaps, gtol, maxiter, iter = 1)
  # unpack
  x, g, d, w = workspace
  AtApI, H = linmaps
  converged = norm(g) <= gtol

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
  end

  return iter, converged
end

function solve_OLS(A::AbstractMatrix{T}, b::Vector{T}, x0::Vector{T}, n_blk::Int;
  lambda::Float64 = 0.0,
  use_qub::Bool = false,
  normalize::Bool = false,
  gram::Bool = _cache_gram_heuristic_(A),
  maxiter::Int = 100,
  gtol::Float64 = 1e-3,
) where T
  #
  n_obs, n_var = size(A)
  var_per_blk = cld(n_var, n_blk)

  @assert rem(n_var, n_blk) == 0
  @assert var_per_blk > 0
  @assert lambda >= 0

  # linear maps
  AtA = GramPlusDiag(A; gram=gram)              # may cache AtA
  AtApI = GramPlusDiag(AtA, one(T), T(lambda))  # same data, add lazy shift by λI
  
  # workspace
  x = deepcopy(x0)
  g = zeros(n_var)
  d = zeros(n_var)
  w = zeros(n_var)
  workspace = (x, g, d, w)

  #
  # We cannot retrieve the QUB matrix directly due to type-instability in the way it is created.
  # This creates a closure taking the QUB matrix H as an input that will be invoked below by
  # the with_qub_matrix() function.
  #
  run = let
    function(H)
      init_recurrences!(workspace, AtApI, b, H)
      __OLS_loop__(workspace, (AtApI, H), gtol, maxiter, 1)
    end
  end
  iter, converged = with_qub_matrix(run, AtA, n_obs, n_var, n_blk, var_per_blk, lambda, use_qub, normalize)

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

  return iter, converged
end

function solve_OLS_lbfgs(A::AbstractMatrix{T}, b::Vector{T}, x0::Vector{T}, n_blk;
  lambda::Float64 = 0.0,
  precond::Symbol = :none,
  gram::Bool = _cache_gram_heuristic_(A),
  maxiter::Int = 100,
  gtol::Float64 = 1e-3,
  memory::Int = 10,  
) where T
  #
  n_obs, n_var = size(A)
  var_per_blk = cld(n_var, n_blk)

  @assert rem(n_var, n_blk) == 0
  @assert var_per_blk > 0
  @assert lambda >= 0

  AtA = GramPlusDiag(A; gram=gram)              # may cache AtA
  AtApI = GramPlusDiag(AtA, one(T), T(lambda))  # same data, add lazy shift by λI

  # workspace
  x = deepcopy(x0)
  g = zeros(T, n_var)
  d = zeros(T, n_var)
  w = zeros(T, n_var)
  cache = LBFGSCache{T}(n_var, memory)
  workspace = (x, g, d, w, cache)

  # Type-stability trick
  if precond == :none
    iter, converged = __OLS_lbfgs__(workspace, (A, b, AtApI, I), gtol, maxiter, 0)
  elseif precond == :qub
    run = let A = A, AtApI = AtApI, b = b, workspace = workspace, gtol = gtol, maxiter = maxiter
      function(H)
        __OLS_lbfgs__(workspace, (A, b, AtApI, H), gtol, maxiter, 0)
      end
    end
    iter, converged = with_qub_matrix(run, AtA, n_obs, n_var, n_blk, var_per_blk, lambda, true, false)
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

function solve_OLS_cg(A, b;
  x0 = IterativeSolvers.zerox(A, b),
  lambda::Real=0.0,
  use_qub::Bool=false,
  gram::Bool = _cache_gram_heuristic_(A),
  kwargs...  
)
  #
  lambda >= 0 || error("Regularization λ must be non-negative. Got: $(lambda)")
  T = eltype(A)

  AtA = GramPlusDiag(A; gram=gram)
  AtApI = GramPlusDiag(AtA, one(T), T(lambda))

  if use_qub # precondition with QUB matrix
    D = compute_main_diagonal(AtA.A, AtA.AtA)
    rho = estimate_spectral_radius(AtA, D, maxiter=3)
    @. D.diag += lambda + rho

    x, ch = cg!(x0, AtApI, transpose(A)*b; Pl=D, log=true, kwargs...)
  else
    x, ch = cg!(x0, AtApI, transpose(A)*b; log=true, kwargs...)
  end

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

