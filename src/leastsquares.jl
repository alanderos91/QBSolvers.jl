const OLSTimer = TimerOutput()
disable_timer!(OLSTimer)

###
### Initialization
###

function init_recurrences!(d, x, g, w, AtApI::GramPlusDiag, b, H, lambda)
  # Initialize the difference, d₁ = x₁ - x₀
  global OLSTimer
  T = eltype(AtApI)
  A = AtApI.A
  @timeit OLSTimer "compute -∇₀" begin
  mul!(g, AtApI, x) # -∇₀ = Aᵀ⋅r - λx
  mul!(g, transpose(A), b, one(T), -one(T))
  !iszero(lambda) && axpy!(-T(lambda), x, g)
  end

  @timeit OLSTimer "compute d₁" ldiv!(d, H, g) # d₁ = H⁻¹(-∇₀)
  @timeit OLSTimer "compute x₁" @. x = x + d   # x₁ = x₀ + d₁

  # Current negative gradient, -∇₁
  @timeit "w = (AᵀA+λI)d₁" mul!(w, AtApI, d) # (AᵀA+λI) d₁
  @timeit "compute g₁"     @. g = g - w      # -∇₁ = -∇₀ - (AᵀA+λI) d₁

  return nothing
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
  global OLSTimer; enable_timer!(OLSTimer); reset_timer!(OLSTimer)
  n_obs, n_var = size(A)
  var_per_blk = cld(n_var, n_blk)

  @assert rem(n_var, n_blk) == 0
  @assert var_per_blk > 0
  @assert lambda >= 0

  @timeit OLSTimer "init AtA + λI; gram=$(gram)" begin
  AtA = GramPlusDiag(A; gram=gram)              # may cache AtA
  AtApI = GramPlusDiag(AtA, one(T), T(lambda))  # same data, add lazy shift by λI
  end

  results = if var_per_blk > 1
    #
    # Block Diagonal Hessian
    #
    if use_qlb && normalize
      @timeit OLSTimer "BlkDiag QLB; normalize=true" begin
      let
        @timeit OLSTimer "normalize" AtA0 = NormalizedGramPlusDiag(AtA)
        @timeit OLSTimer "BlkDiag J" begin
        J = compute_block_diagonal(AtA0, n_blk;
          alpha   = one(T),
          beta    = zero(T),
          factor  = false,
          gram    = n_obs > var_per_blk
        )
        end
        @timeit OLSTimer "spectral radius" rho = estimate_spectral_radius(AtA0, J, maxiter=3)
        @timeit OLSTimer "Hessian; BlkDiag + Rank-1" begin
        S = Diagonal(@. rho*AtA0.A.scale^2 + lambda)
        @. AtA0.A.scale = 1 # need J̃ = √S⋅J⋅√S + ρS + λ = ZᵀZ + ρS + λ
        J̃ = update_factors!(J, AtA0.A, S, one(T), one(T))
        H = BlkDiagPlusRank1(n_obs, n_var, J̃, AtA0.A.shift, one(T), T(n_obs))
        end
        @timeit OLSTimer "OLS loop" _solve_OLS_loop(AtApI, H, b, x0, T(lambda); kwargs...)
      end
      end
    elseif use_qlb
      @timeit OLSTimer "BlkDiag QLB; normalize=false" begin
      let
        @timeit OLSTimer "BlkDiag J" begin
        J = compute_block_diagonal(AtA, n_blk;
          alpha   = one(T),
          beta    = zero(T),
          factor  = false,
          gram    = n_obs > var_per_blk
        )
        end
        @timeit OLSTimer "spectral radius" rho = estimate_spectral_radius(AtA, J, maxiter=3)
        @timeit OLSTimer "Hessian; BlkDiag" H = update_factors!(J, one(T), lambda + rho)
        @timeit OLSTimer "OLS loop" _solve_OLS_loop(AtApI, H, b, x0, T(lambda); kwargs...)
      end
      end
    else
      @timeit OLSTimer "BlkDiag Jensen" begin
      let
        @timeit OLSTimer "Hessian; BlkDiag" begin
        H = compute_block_diagonal(AtA, n_blk;
          alpha   = T(n_blk),
          beta    = T(lambda),
          factor  = true,
          gram    = n_obs > var_per_blk
        )
        end
        @timeit OLSTimer "OLS loop" _solve_OLS_loop(AtApI, H, b, x0, T(lambda); kwargs...)
      end
      end
    end
  else
    #
    # Diagonal (Plus Rank-1) Hessian
    #
    if use_qlb && normalize
      @timeit OLSTimer "Diag QLB; normalize=true" begin
      let
        @timeit OLSTimer "normalize" AtA0 = NormalizedGramPlusDiag(AtA)
        @timeit OLSTimer "Diag J" J = compute_main_diagonal(AtA0.A, AtA0.AtA)
        @timeit OLSTimer "spectral radius" rho = estimate_spectral_radius(AtA0, J, maxiter=3)
        @timeit OLSTimer "Hessian; Diag + Rank-1" begin
        @. J.diag = (1+rho)*AtA0.A.scale^2 + T(lambda)
        H = BlkDiagPlusRank1(n_obs, n_var, J, AtA0.A.shift, one(T), T(n_obs))
        end
        @timeit OLSTimer "OLS loop" _solve_OLS_loop(AtApI, H, b, x0, T(lambda); kwargs...)
      end
      end
    elseif use_qlb
      @timeit OLSTimer "Diag QLB; normalize=false" begin
      let
        @timeit OLSTimer "Diag J" J = compute_main_diagonal(AtA.A, AtA.AtA)
        @timeit OLSTimer "spectral radius" rho = estimate_spectral_radius(AtA, J, maxiter=3)
        @timeit OLSTimer "Hessian; Diag" begin
        H = J
        @. H.diag = H.diag + rho + lambda
        end
        @timeit OLSTimer "OLS loop" _solve_OLS_loop(AtApI, H, b, x0, T(lambda); kwargs...)
      end
      end
    else
      @timeit OLSTimer "Diag Jensen" begin
      let
        @timeit OLSTimer "Hessian; Diag" begin
        H = compute_main_diagonal(AtA.A, AtA.AtA)
        @. H.diag = n_blk*H.diag + lambda
        end
        @timeit OLSTimer "OLS loop" _solve_OLS_loop(AtApI, H, b, x0, T(lambda); kwargs...)
      end
      end
    end
  end
  disable_timer!(OLSTimer)
  display(OLSTimer)
  return results
end

function _solve_OLS_loop(AtApI::GramPlusDiag{T}, H, b::Vector{T}, x0::Vector{T}, lambda::T;
  maxiter::Int = 100,
  gtol::Float64 = 1e-3,
) where T
  #
  global OLSTimer
  A = AtApI.A
  n_var = size(A, 2)

  # Main matrices and vectors
  @timeit OLSTimer "init workspace" begin
  x = deepcopy(x0)
  d = zeros(n_var)
  g = zeros(n_var)
  w = zeros(n_var)
  init_recurrences!(d, x, g, w, AtApI, b, H, lambda)

  # Iterate the algorithm map
  iter = 1
  @timeit OLSTimer "check convergence" converged = norm(g) <= gtol

  while !converged && (iter < maxiter)
    iter += 1

    # Update difference dₙ₊₁ = [I - H⁻¹(AᵀA+λI)] dₙ
    @timeit OLSTimer "compute dₙ₊₁" begin
    @timeit OLSTimer "solve H⋅w = y" ldiv!(H, w)
    @timeit OLSTimer "axpy!" @. d = d - w
    end

    # Update coefficients
    @timeit OLSTimer "compute xₙ₊₁" @. x = x + d

    # Compute (AᵀA+λI) dₙ₊₁ and -∇ₙ₊₁
    @timeit OLSTimer "compute gₙ₊₁" begin
    @timeit OLSTimer "w = (AᵀA+λI)dₙ₊₁" mul!(w, AtApI, d)
    @timeit OLSTimer "axpy!" @. g = g - w      # assumes g is always -∇
    end
    @timeit OLSTimer "check convergence" converged = norm(g) <= gtol
  end

  # final residual
  @timeit OLSTimer "compute residual" begin
  r = copy(b)
  mul!(r, A, x, -1.0, 1.0)
  end

  @timeit OLSTimer "summary" begin
  stats = (
    iterations = iter,
    converged = converged,
    xnorm = norm(x),
    rnorm = norm(r),
    gnorm = norm(g),
  )
  end
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

  AtA = GramPlusDiag(A; gram=gram)
  AtApI = GramPlusDiag(AtA, one(T), T(lambda))

  if use_qlb # precondition with QLB matrix
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
