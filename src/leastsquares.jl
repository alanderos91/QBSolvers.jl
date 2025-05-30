###
### Initialization
###

function initblocks!(::Type{T}, AtApI::GramPlusDiag, lambda, use_qlb, n_blk) where T
  A, AtA = AtApI.A, AtApI.AtA
  n_obs, n_var = size(A)
  var_per_blk = cld(n_var, n_blk)
  # Compute blocks along diagonal, Dₖ = Dₖₖ = AₖᵀAₖ + λI and extract their Cholesky decompositions
  if use_qlb
    if size(AtA, 1) > 0
      D = BlkDiagHessian(AtApI, n_blk; alpha=1, beta=0, factor=false)
    else
      D = BlkDiagHessian(A, n_blk; alpha=1, beta=0, factor=false, gram=n_obs >= var_per_blk)
    end
    lambda_max = estimate_dominant_eigval(GramPlusDiag(AtApI, 1, 0), D, maxiter=3)
    D = update_factors!(D, 1, lambda + lambda_max)
  else
    if size(AtA, 1) > 0
      D = BlkDiagHessian(AtApI, n_blk; alpha=n_blk, beta=lambda, factor=true)
    else
      D = BlkDiagHessian(A, n_blk; alpha=n_blk, beta=lambda, factor=true, gram=n_obs >= var_per_blk)
    end
  end
  return D
end

function initdiag!(::Type{T}, AtApI::GramPlusDiag, lambda, use_qlb) where T
  A, AtA = AtApI.A, AtApI.AtA
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
    lambda_max = estimate_dominant_eigval(GramPlusDiag(AtApI, 1, 0), D, maxiter=3)
    @. D.diag += lambda + lambda_max
  else
    n_blk = length(D.diag)
    @. D.diag = n_blk * D.diag
  end
  return D
end

function init_recurrences!(d, x, g, w, AtApI::GramPlusDiag, b, D, lambda)
  # Initialize the difference, d₁ = x₁ - x₀
  T = eltype(AtApI)
  A = AtApI.A
  r = copy(b)
  mul!(r, A, x, -one(T), one(T))  # r = b - A⋅x
  mul!(g, transpose(A), r)        # -∇₀ = Aᵀ⋅r - λx
  !iszero(lambda) && axpy!(-T(lambda), x, g)

  ldiv!(d, D, g) # d₁ = D⁻¹(-∇₀)
  @. x = x + d   # x₁ = x₀ + d₁

  # Current negative gradient, -∇₁
  mul!(w, AtApI, d) # (AᵀA+λI) d₁
  @. g = g - w      # -∇₁ = -∇₀ - (AᵀA+λI) d₁

  return r
end

###
### Implementation
###

function solve_OLS(A::AbstractMatrix{T}, b::Vector{T}, x0::Vector{T}, n_blk::Int;
  lambda::Float64 = 0.0,
  use_qlb::Bool = false,
  normalize::Bool = false,
  gram::Bool = _cache_gram_heuristic_(A),
  kwargs...
) where T
  #
  n_obs, n_var = size(A)
  var_per_blk = cld(n_var, n_blk)

  @assert rem(n_var, n_blk) == 0
  @assert var_per_blk > 0
  @assert lambda >= 0

  AtA = GramPlusDiag(A; gram=gram)              # may cache AtA
  AtApI = GramPlusDiag(AtA, one(T), T(lambda))  # same data, add lazy shift by λI

  if var_per_blk > 1
    if use_qlb && normalize
      AtA0 = NormalizedGramPlusDiag(AtA)
      D_blkd0 = initblocks!(T, AtA0, lambda, true, n_blk)
      SDSpuuT = BlkDiagPlusRank1(A, D_blkd0)
      _solve_OLS_loop(AtApI, SDSpuuT, b, x0, T(lambda); kwargs...)
    else
      D_blkd = initblocks!(T, AtA, lambda, use_qlb, n_blk)
      _solve_OLS_loop(AtApI, D_blkd, b, x0, T(lambda); kwargs...)
    end
  else
    if use_qlb && normalize
      AtA0 = NormalizedGramPlusDiag(AtA)
      D_diag0 = initdiag!(T, AtA0, lambda, true)
      SDSpuuT = BlkDiagPlusRank1(A, D_diag0)
      _solve_OLS_loop(AtApI, SDSpuuT, b, x0, T(lambda); kwargs...)
    else
      D_diag = initdiag!(T, AtA, lambda, use_qlb)
      _solve_OLS_loop(AtApI, D_diag, b, x0, T(lambda); kwargs...)
    end
  end
end

function _solve_OLS_loop(AtApI::GramPlusDiag{T}, D, b::Vector{T}, x0::Vector{T}, lambda::T;
  maxiter::Int = 100,
  gtol::Float64 = 1e-3,
) where T
  #
  A = AtApI.A
  n_var = size(A, 2)

  # Main matrices and vectors
  x = deepcopy(x0)
  d = zeros(n_var)
  g = zeros(n_var)
  w = zeros(n_var)
  r = init_recurrences!(d, x, g, w, AtApI, b, D, lambda)

  # Iterate the algorithm map
  iter = 1
  converged = norm(g) <= gtol

  while !converged && (iter < maxiter)
    iter += 1

    # Update difference dₙ₊₁ = [I - D⁻¹(AᵀA+λI)] dₙ
    ldiv!(D, w)
    @. d = d - w

    # Update coefficients
    @. x = x + d

    # Compute (AᵀA+λI) dₙ₊₁ and -∇ₙ₊₁
    mul!(w, AtApI, d)
    @. g = g - w      # assumes g is always -∇
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
