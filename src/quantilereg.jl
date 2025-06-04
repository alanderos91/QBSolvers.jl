const QREGTimer = TimerOutput()
disable_timer!(QREGTimer)

###
### Helper functions
###

function default_bandwidth(A::AbstractMatrix{T}) where T
  n, p = size(A)
  return max(T(0.05), ((log(n) + p) / n)^0.4)
end

#
# Evaluate prox[h|⋅|](r).
#
prox_abs(r::T, h::T) where T = (1 - h / max(abs(r), h)) * r

#
# Evaluate prox[h|⋅|](r) element-wise and store in z.
#
function prox_abs!(z::AbstractVector{T}, r::AbstractVector{T}, h::T) where T
  map!(Base.Fix2(prox_abs, h), z, r)
  return z
end

# check function
function qreg_loss(r::T, q::T) where T <: Real
  return (q-1//2)*r + 1//2*abs(r)
end

function qreg_objective(r, q)
  let r = r, q = q
    f(r) = qreg_loss(r, q)
    mapreduce(f, +, r) / length(r)
  end
end

# Uniform kernel
function qreg_loss_uniform(r::T, q::T, h::T) where T <: Real
  absr = abs(r)
  C = ifelse(absr > h, absr, h/2 * (1 + (r/h)^2))
  return (q-1//2)*r + 1//2*C
end

function qreg_objective_uniform(r, q, h)
  let r =r, q = q, h = h
    f(r) = qreg_loss_uniform(r, q, h)
    mapreduce(f, +, r) / length(r)
  end
end

function init_recurrences_qreg!(d, x, g, u, v, w, z_old, z_new, AtApI::GramPlusDiag, b, H, q, h, lambda)
  # Initialize the difference, d₁ = x₁ - x₀ = D⁻¹ (-∇₀)
  global QREGTimer
  T = eltype(AtApI)
  A = AtApI.A
  @timeit QREGTimer "compute r₀" begin
  r = copy(b) # r₀ = b - Ax₀
  mul!(r, A, x, -one(T), one(T))
  end
  @timeit QREGTimer "compute z₀" prox_abs!(z_new, r, h) # z₀ = prox[h|⋅|](r₀)
  @timeit QREGTimer "compute -∇₀" begin
  @. u = r - z_new                # -∇₀ = Aᵀ[r₀ - z₀ .+ (2q-1)h]
  @. u = u + (2*q-1)*h
  mul!(g, transpose(A), u, inv(T(2*h)), zero(T))
  # !iszero(g) && axpy!(-T(lambda), x, g)
  end

  @timeit QREGTimer "compute d₁" ldiv!(d, H, g)  # d₁ = H⁻¹(-∇₀)
  @timeit QREGTimer "compute x₁" @. x = x + d    # x₁ = x₀ + d₁

  # Update recurrences to n = 1
  @timeit QREGTimer "compute r₁" begin
  mul!(u, A, d)  # r₁ = r₀ - A d₁
  @. r = r - u
  end
  @timeit QREGTimer "compute z₁" prox_abs!(z_old, r, h)      # z₁ = prox[h|⋅|](r₁)
  @timeit QREGTimer "compute -∇₁" begin
  @. v = u + z_old - z_new
  mul!(w, transpose(A), v)    # Aᵀ(A d₁ + z₁ - z₀)
  @. g = g - inv(T(2*h)) * w  # -∇₁ = -∇₀ - (2h)⁻¹Aᵀ(A d₁ + z₁ - z₀)
  end

  return r
end

###
### Implementation
###

function solve_QREG(A::AbstractMatrix{T}, b::Vector{T}, x0::Vector{T}, n_blk::Int;
  q::Real         = T(0.5),
  h::Real         = T(default_bandwidth(A)),
  lambda::Real    = T(0.0),
  gram::Bool      = _cache_gram_heuristic_(A),
  normalize::Bool = false,
  kwargs...
) where T
  #
  global QREGTimer; enable_timer!(QREGTimer); reset_timer!(QREGTimer)
  n_obs, n_var = size(A)
  var_per_blk = cld(n_var, n_blk)

  @assert rem(n_var, n_blk) == 0
  @assert var_per_blk > 0
  @assert 0 < q < 1

  @timeit QREGTimer "init AtA + λI; gram=$(gram)" begin
  AtA = GramPlusDiag(A; gram=gram)              # may cache AtA
  AtApI = GramPlusDiag(AtA, one(T), T(lambda))  # same data, add lazy shift by λI
  end

  results = if var_per_blk > 1
    #
    # Block Diagonal Hessian
    #
    if normalize
      @timeit QREGTimer "BlkDiag QLB; normalize=true" begin
      let
        @timeit QREGTimer "normalize" AtA0 = NormalizedGramPlusDiag(AtA)
        @timeit QREGTimer "BlkDiag J" begin
        J = compute_block_diagonal(AtA0, n_blk;
          alpha   = one(T),
          beta    = zero(T),
          factor  = false,
          gram    = n_obs > var_per_blk
        )
        end
        @timeit QREGTimer "spectral radius" rho = estimate_spectral_radius(AtA0, J, maxiter=1)
        @timeit QREGTimer "Hessian; BlkDiag + Rank-1" begin
        S = Diagonal(@. rho*AtA0.A.scale^2 + lambda)
        @. AtA0.A.scale = 1 # need J̃ = √S⋅J⋅√S + ρS + λ = ZᵀZ + ρS + λ
        J̃ = update_factors!(J, AtA0.A, S, one(T), one(T))
        H = BlkDiagPlusRank1(n_obs, n_var, J̃, AtA0.A.shift, one(T), T(n_obs))
        end
        @timeit QREGTimer "QREG loop" _solve_QREG_loop(AtApI, H, b, x0, q, h, T(lambda); kwargs...)
      end
      end
    else
      @timeit QREGTimer "BlkDiag QLB; normalize=false" begin
      let
        @timeit QREGTimer "BlkDiag J" begin
        J = compute_block_diagonal(AtA, n_blk;
          alpha   = one(T),
          beta    = zero(T),
          factor  = false,
          gram    = n_obs > var_per_blk
        )
        end
        @timeit QREGTimer "spectral radius" rho = estimate_spectral_radius(AtA, J, maxiter=1)
        @timeit QREGTimer "Hessian; BlkDiag" H = update_factors!(J, one(T), lambda + rho)
        @timeit QREGTimer "QREG loop" _solve_QREG_loop(AtApI, H, b, x0, q, h, T(lambda); kwargs...)
      end
      end
    end
  else
    #
    # Diagonal (Plus Rank-1) Hessian
    #
    if normalize
      @timeit QREGTimer "Diag QLB; normalize=true" begin
      let
        @timeit QREGTimer "normalize" AtA0 = NormalizedGramPlusDiag(AtA)
        @timeit QREGTimer "Diag J" J = compute_main_diagonal(AtA0.A, AtA0.AtA)
        @timeit QREGTimer "spectral radius" rho = estimate_spectral_radius(AtA0, J, maxiter=1)
        @timeit QREGTimer "Hessian; Diag + Rank-1" begin
        @. J.diag = (1+rho)*AtA0.A.scale^2 + T(lambda)
        H = BlkDiagPlusRank1(n_obs, n_var, J, AtA0.A.shift, one(T), T(n_obs))
        end
        @timeit QREGTimer "QREG loop" _solve_QREG_loop(AtApI, H, b, x0, q, h, T(lambda); kwargs...)
      end
      end
    else
      @timeit QREGTimer "Diag QLB; normalize=false" begin
      let
        @timeit QREGTimer "Diag J" J = compute_main_diagonal(AtA.A, AtA.AtA)
        @timeit QREGTimer "spectral radius" @time rho = estimate_spectral_radius(AtA, J, maxiter=1)
        @timeit QREGTimer "Hessian; Diag" begin
        H = J
        @. H.diag = H.diag + rho + lambda
        end
        @timeit QREGTimer "QREG loop" _solve_QREG_loop(AtApI, H, b, x0, q, h, T(lambda); kwargs...)
      end
      end
    end
  end
  disable_timer!(QREGTimer)
  display(QREGTimer)
  return results
end

function _solve_QREG_loop(AtApI::GramPlusDiag{T}, H, b::Vector{T}, x0::Vector{T}, q::T, h::T, lambda::T;
  maxiter::Int  = 100,
  gtol::Float64 = 1e-3,
) where T
  #
  global QREGTimer
  A = AtApI.A
  n_obs, n_var = size(A)

  # 
  @timeit QREGTimer "init workspace" begin
  x = deepcopy(x0)
  d = zeros(n_var)
  g = zeros(n_var)
  u = zeros(n_obs)
  v = zeros(n_obs)
  w = zeros(n_var)
  z_new = zeros(n_obs)
  z_old = zeros(n_obs)
  end
  @timeit QREGTimer "init recurrences" r = init_recurrences_qreg!(d, x, g, u, v, w, z_old, z_new, AtApI, b, H, q, h, lambda)

  # Iterate the algorithm map
  iter = 1
  @timeit QREGTimer "check convergence" converged = norm(g) <= gtol

  while !converged && (iter < maxiter)
    iter += 1

    # Compute dₙ₊₁ = dₙ - (2h)⁻¹D⁻¹ Aᵀ (A dₙ + zₙ - zₙ₋₁)
    @timeit QREGTimer "compute dₙ₊₁" begin
    @timeit QREGTimer "solve H⋅w = y" ldiv!(H, w)
    @timeit QREGTimer "axpy!" @. d = d - inv(T(2*h)) * w
    end

    # Update coefficients
    @timeit QREGTimer "compute xₙ₊₁" @. x = x + d

    # Compute rₙ₊₁ = rₙ - A dₙ₊₁
    @timeit QREGTimer "compute rₙ₊₁" begin
    @timeit QREGTimer "A⋅dₙ₊₁" mul!(u, A, d)
    @timeit QREGTimer "axpy!" @. r = r - u
    end

    # Compute zₙ₊₁ = prox[h|⋅|](rₙ₊₁)
    @timeit QREGTimer "compute zₙ₊₁" prox_abs!(z_new, r, h)

    # Compute A dₙ₊₁ + zₙ₊₁ - zₙ
    @timeit QREGTimer "compute gₙ₊₁" begin
    @timeit QREGTimer "axpy! × 2" @. v = u + z_new - z_old

    # Compute Aᵀ (A dₙ₊₁ + zₙ₊₁ - zₙ) and -∇ₙ₊₁
    @timeit QREGTimer "Aᵀv" mul!(w, transpose(A), v)
    @timeit QREGTimer "axpy!" @. g = g - inv(T(2*h)) * w  # assumes g is always -∇
    end
    @timeit QREGTimer "check convergence" converged = norm(g) <= gtol

    # don't copy, swap'em
    @timeit QREGTimer "buffer swap" z_new, z_old = z_old, z_new
  end

  @timeit QREGTimer "summary" begin
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
