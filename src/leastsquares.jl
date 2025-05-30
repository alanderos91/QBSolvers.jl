###
### Initialization
###

function initblocks!(::Type{T}, d, x, g, linmap, b, n_blk, lambda, use_qlb, tol_powm) where T
  A, AtA = linmap.A, linmap.AtA
  n_obs, n_var = size(A)
  var_per_blk = cld(n_var, n_blk)
  # Compute blocks along diagonal, Dₖ = Dₖₖ = AₖᵀAₖ + λI and extract their Cholesky decompositions
  if use_qlb
    if size(AtA, 1) > 0
      D = BlkDiagHessian(linmap, n_blk; alpha=1, beta=0, factor=false)
    else
      D = BlkDiagHessian(A, n_blk; alpha=1, beta=0, factor=false, gram=n_obs >= var_per_blk)
    end
    lambda_max = estimate_dominant_eigval(GramPlusDiag(linmap, 1, 0), D, maxiter=3)
    D = update_factors!(D, 1, lambda + lambda_max)
  else
    if size(AtA, 1) > 0
      D = BlkDiagHessian(linmap, n_blk; alpha=n_blk, beta=lambda, factor=true)
    else
      D = BlkDiagHessian(A, n_blk; alpha=n_blk, beta=lambda, factor=true, gram=n_obs >= var_per_blk)
    end
  end

  # Initialize the difference, d₁ = x₁ - x₀
  r = copy(b)
  mul!(r, A, x, -one(T), one(T))  # r = b - A⋅x
  mul!(g, transpose(A), r)        # -∇₀ = Aᵀ⋅r - λx
  !iszero(lambda) && axpy!(-T(lambda), x, g)

  ldiv!(d, D, g) # d₁ = D⁻¹(-∇₀)
  @. x = x + d   # x₁ = x₀ + d₁

  return r, D
end

function initdiag!(::Type{T}, d, x, g, linmap, b, lambda, use_qlb, tol_powm) where T
  A, AtA = linmap.A, linmap.AtA
  if size(AtA, 1) == 0
    diag = zeros(T, size(A, 2))
    for k in axes(A, 2)
      @views diag[k] = dot(A[:, k], A[:, k])
    end
  else
    diag = AtA[diagind(AtA)]
  end
  D = Diagonal(diag)

  if use_qlb
    lambda_max = estimate_dominant_eigval(GramPlusDiag(linmap, 1, 0), D, maxiter=3)
    @. D.diag += lambda + lambda_max
  end

  # Initialize the difference, d₁ = x₁ - x₀
  r = copy(b)
  mul!(r, A, x, -one(T), one(T))  # r = b - A⋅x
  mul!(g, transpose(A), r)        # -∇₀ = Aᵀ⋅r - λx
  !iszero(lambda) && axpy!(-T(lambda), x, g)

  ldiv!(d, D, g) # d₁ = D⁻¹(-∇₀)
  @. x = x + d   # x₁ = x₀ + d₁

  return r, D
end

###
### Implementation
###

function solve_OLS(A::AbstractMatrix{T}, b::Vector{T}, x0::Vector{T}, n_blk::Int;
  lambda::Float64 = 0.0,
  kwargs...
) where T
  #
  n_obs, n_var = size(A)
  var_per_blk = cld(n_var, n_blk)

  @assert rem(n_var, n_blk) == 0
  @assert var_per_blk > 0
  @assert lambda >= 0

  if var_per_blk > 1
    _solve_OLS_blkdiag(A, b, x0, n_blk; lambda=lambda, kwargs...)
  else
    _solve_OLS_diag(A, b, x0, n_blk; lambda=lambda, kwargs...)
  end
end

function _solve_OLS_blkdiag(A::AbstractMatrix{T}, b::Vector{T}, x0::Vector{T}, n_blk::Int;
  lambda::Float64 = 0.0,
  maxiter::Int = 100,
  gtol::Float64 = 1e-3,
  use_qlb::Bool = false,
  tol_powm::Float64 = T(minimum(size(A))),
  gram::Bool = _cache_gram_heuristic_(A),
) where T
  #
  n_obs, n_var = size(A)

  # Main matrices and vectors
  x = deepcopy(x0)
  d = zeros(n_var)
  g = zeros(n_var)
  tmp = zeros(n_var)
  AtApI = GramPlusDiag(A; alpha=one(T), beta=T(lambda), gram=gram)
  r, D = initblocks!(T, d, x, g, AtApI, b, n_blk, lambda, use_qlb, tol_powm)

  # Current negative gradient, -∇₁
  mul!(tmp, AtApI, d) # (AᵀA+λI) d₁
  @. g = g - tmp      # -∇₁ = -∇₀ - (AᵀA+λI) d₁

  # Iterate the algorithm map
  iter = 1
  converged = norm(g) <= gtol

  while !converged && (iter < maxiter)
    iter += 1

    # Update difference dₙ₊₁ = [I - D⁻¹(AᵀA+λI)] dₙ
    ldiv!(D, tmp)
    @. d = d - tmp

    # Update coefficients
    @. x = x + d

    # Compute (AᵀA+λI) dₙ₊₁ and -∇ₙ₊₁
    mul!(tmp, AtApI, d)
    @. g = g - tmp      # assumes g is always -∇
    converged = norm(g) <= gtol
  end

  # final residual
  copyto!(r, b)
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

function _solve_OLS_diag(A::AbstractMatrix{T}, b::Vector{T}, x0::Vector{T}, n_blk::Int;
  lambda::Float64 = 0.0,
  maxiter::Int = 100,
  gtol::Float64 = 1e-3,
  use_qlb::Bool = false,
  tol_powm::Float64 = T(minimum(size(A))),
  gram::Bool = _cache_gram_heuristic_(A),
) where T
  #
  n_obs, n_var = size(A)

  # Main matrices and vectors
  x = deepcopy(x0)
  d = zeros(n_var)
  g = zeros(n_var)
  tmp = zeros(n_var)
  AtApI = GramPlusDiag(A; alpha=one(T), beta=T(lambda), gram=gram)
  r, D = initdiag!(T, d, x, g, AtApI, b, lambda, use_qlb, tol_powm)

  # Current negative gradient, -∇₁
  mul!(tmp, AtApI, d) # (AᵀA+λI) d₁
  @. g = g - tmp      # -∇₁ = -∇₀ - (AᵀA+λI) d₁

  # Iterate the algorithm map
  iter = 1
  converged = norm(g) <= gtol

  while !converged && (iter < maxiter)
    iter += 1

    # Update difference dₙ₊₁ = [I - D⁻¹(AᵀA+λI)] dₙ
    ldiv!(D, tmp)
    @. d = d - tmp

    # Update coefficients
    @. x = x + d

    # Compute (AᵀA+λI) dₙ₊₁ and -∇ₙ₊₁
    mul!(tmp, AtApI, d)
    @. g = g - tmp      # assumes g is always -∇
    converged = norm(g) <= gtol
  end

  # final residual
  copyto!(r, b)
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
  use_qlb::Bool=false,
  gram::Bool = _cache_gram_heuristic_(A),
  kwargs...  
)
  #
  lambda >= 0 || error("Regularization λ must be non-negative. Got: $(lambda)")
  T = eltype(A)

  AtApI = GramPlusDiag(A; alpha=one(T), beta=T(lambda), gram=gram)

  if use_qlb # precondition with QLB matrix
    AtA = AtApI.AtA
    if size(AtA, 1) == 0
      diag = zeros(T, size(A, 2))
      for k in axes(A, 2)
        @views diag[k] = dot(A[:, k], A[:, k])
      end
    else
      diag = AtA[diagind(AtA)]
    end
    D = Diagonal(diag)
    
    lambda_max = estimate_dominant_eigval(GramPlusDiag(AtApI, 1, 0), D, maxiter=3)
    @. D.diag += lambda + lambda_max

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
