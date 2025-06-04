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
        @timeit QREGTimer "spectral radius" rho = estimate_spectral_radius(AtA0, J, maxiter=3)
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
        @timeit QREGTimer "spectral radius" rho = estimate_spectral_radius(AtA, J, maxiter=3)
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
        @timeit QREGTimer "spectral radius" rho = estimate_spectral_radius(AtA0, J, maxiter=3)
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
        @timeit QREGTimer "spectral radius" rho = estimate_spectral_radius(AtA, J, maxiter=3)
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
  rtol::Float64 = 1e-6,
) where T
  #
  global QREGTimer
  A = AtApI.A
  n_obs, n_var = size(A)

  # 
  @timeit QREGTimer "init workspace" begin
  x = deepcopy(x0); x_next = deepcopy(x0); x_prev = deepcopy(x0)
  d = zeros(n_var)
  g = zeros(n_var)
  u = zeros(n_obs)
  z = zeros(n_obs)
  end

  # Initialize
  @timeit QREGTimer "init residual" begin
  r = copy(b) # r = b - A*x
  mul!(r, A, x, -one(T), one(T))
  end

  # Iterate the algorithm map
  iter = 0  # outer iterations; Moreau majorization
  inner = 0 # inner iterations; QUB majorization
  k = 0     # Nesterov iterations
  converged = false
  @timeit QREGTimer "eval objective" f_curr = qreg_objective_uniform(r, q, h)
  while !converged && (iter < maxiter)
    #
    # Setup the next LS problem, |Ax - u|²
    #   where u = b - zₙ + (2*q-1)*h 1
    #
    @timeit QREGTimer "eval prox" prox_abs!(z, r, h)
    @timeit QREGTimer "set RHS u" @. u = b - z + (2*q-1)*h
    #
    # Solve with QUB
    #
    @timeit QREGTimer "compute Aᵀu" mul!(x_next, transpose(A), u) # save this constant
    @timeit QREGTimer "compute g (neg gradient)" begin
    mul!(g, AtApI, x)             # -∇ = Aᵀ(u - A*x)
    @. g = x_next - g
    end
    @timeit QREGTimer "check convergence (inner)" not_stationary = norm(g) > gtol
    if not_stationary
      @timeit QREGTimer "inner iterations" begin
      while not_stationary
        inner += 1
        @timeit QREGTimer "d = H⁻¹ g" ldiv!(d, H, g)
        @timeit QREGTimer "x = x + d" @. x = x + d
        @timeit QREGTimer "update g" begin
        mul!(g, AtApI, x)
        @. g = x_next - g
        end
        @timeit QREGTimer "check convergence (inner)" not_stationary = norm(g) > gtol
      end
      end
      # Nesterov acceleration
      @timeit QREGTimer "Nesterov acceleration" begin
      k += 1
      @. x_next = x
      @. x = k/(k+3) * (x_next - x_prev); @. x = x + x_next
      @. x_prev = x_next
      end
      # Update residual
      @timeit QREGTimer "compute residual" begin
      @. r = b
      mul!(r, A, x, -one(T), one(T))
      end
    end
    iter += 1
    f_prev = f_curr
    @timeit QREGTimer "eval objective" f_curr = qreg_objective_uniform(r, q, h)
    if f_curr > f_prev k = 0 end
    @timeit QREGTimer "check convergence (outer)" converged = abs(f_curr - f_prev) < rtol * (f_prev + 1)
  end

  @timeit QREGTimer "summary" begin
  stats = (
    iterations = iter,
    inner = inner,
    converged = converged,
    xnorm = norm(x),
    rnorm = norm(r),
    loss1 = qreg_objective(r, q), # check function loss
    loss2 = f_curr,               # smoothed loss
  )
  end

  return x, r, stats
end

function solve_QREG_lbfgs(A::AbstractMatrix{T}, b::Vector{T}, x0::Vector{T}, n_blk::Int;
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
        @timeit QREGTimer "spectral radius" rho = estimate_spectral_radius(AtA0, J, maxiter=3)
        @timeit QREGTimer "Hessian; BlkDiag + Rank-1" begin
        S = Diagonal(@. rho*AtA0.A.scale^2 + lambda)
        @. AtA0.A.scale = 1 # need J̃ = √S⋅J⋅√S + ρS + λ = ZᵀZ + ρS + λ
        J̃ = update_factors!(J, AtA0.A, S, one(T), one(T))
        H = BlkDiagPlusRank1(n_obs, n_var, J̃, AtA0.A.shift, one(T), T(n_obs))
        end
        @timeit QREGTimer "QREG loop" _solve_QREG_loop_lbfgs(AtApI, H, b, x0, q, h, T(lambda); kwargs...)
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
        @timeit QREGTimer "spectral radius" rho = estimate_spectral_radius(AtA, J, maxiter=3)
        @timeit QREGTimer "Hessian; BlkDiag" H = update_factors!(J, one(T), lambda + rho)
        @timeit QREGTimer "QREG loop" _solve_QREG_loop_lbfgs(AtApI, H, b, x0, q, h, T(lambda); kwargs...)
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
        @timeit QREGTimer "spectral radius" rho = estimate_spectral_radius(AtA0, J, maxiter=3)
        @timeit QREGTimer "Hessian; Diag + Rank-1" begin
        @. J.diag = (1+rho)*AtA0.A.scale^2 + T(lambda)
        H = BlkDiagPlusRank1(n_obs, n_var, J, AtA0.A.shift, one(T), T(n_obs))
        end
        @timeit QREGTimer "QREG loop" _solve_QREG_loop_lbfgs(AtApI, H, b, x0, q, h, T(lambda); kwargs...)
      end
      end
    else
      @timeit QREGTimer "Diag QLB; normalize=false" begin
      let
        @timeit QREGTimer "Diag J" J = compute_main_diagonal(AtA.A, AtA.AtA)
        @timeit QREGTimer "spectral radius" rho = estimate_spectral_radius(AtA, J, maxiter=3)
        @timeit QREGTimer "Hessian; Diag" begin
        H = J
        @. H.diag = H.diag + rho + lambda
        end
        @timeit QREGTimer "QREG loop" _solve_QREG_loop_lbfgs(AtApI, H, b, x0, q, h, T(lambda); kwargs...)
      end
      end
    end
  end
  disable_timer!(QREGTimer)
  display(QREGTimer)
  return results
end

function _solve_QREG_loop_lbfgs(AtApI::GramPlusDiag{T}, H, b::Vector{T}, x0::Vector{T}, q::T, h::T, lambda::T;
  memory::Int   = 10,
  maxiter::Int  = 100,
  gtol::Float64 = 1e-3,
  rtol::Float64 = 1e-6,
) where T
  #
  global QREGTimer
  A = AtApI.A
  n_obs, n_var = size(A)

  #
  @timeit QREGTimer "init workspace" begin
  x = deepcopy(x0); x_next = deepcopy(x0); x_prev = deepcopy(x0)
  d = zeros(n_var)
  g = zeros(n_var)
  u = zeros(n_obs)
  z = zeros(n_obs)
  end

  # L-BFGS cache for inner solver
  @timeit QREGTimer "init L-BFGS cache" cache = LBFGSCache{T}(n_var, memory)

  # Initialize
  @timeit QREGTimer "init residual" begin
  r = copy(b) # r = b - A*x
  mul!(r, A, x, -one(T), one(T))
  end

  # Iterate the algorithm map
  iter = 0  # outer iterations; Moreau majorization
  inner = 0 # inner iterations; QUB majorization
  k = 0     # Nesterov iterations
  converged = false
  @timeit QREGTimer "eval objective" f_curr = qreg_objective_uniform(r, q, h)
  alpha = one(T) # L-BFGS steps size
  while !converged && (iter < maxiter)
    #
    # Setup the next LS problem, |Ax - u|²
    #   where u = b - zₙ + (2*q-1)*h 1
    #
    @timeit QREGTimer "eval prox" prox_abs!(z, r, h)
    @timeit QREGTimer "set RHS u" @. u = b - z + (2*q-1)*h
    #
    # Solve with QUB
    #
    @timeit QREGTimer "compute Aᵀu" mul!(x_next, transpose(A), u) # save this constant
    @timeit QREGTimer "compute g (neg gradient)" begin
    mul!(g, AtApI, x)             # -∇ = Aᵀ(u - A*x)
    @. g = x_next - g
    end
    @timeit QREGTimer "check convergence (inner)" not_stationary = norm(g) > gtol
    if not_stationary
      current_inner = 0
      cache.current_size = 0
      cache.current_index = 0
      w = x_next # reuse this workspace inside inner solve
      @timeit QREGTimer "inner iterations" begin
      while not_stationary
        current_inner += 1
        inner += 1

        @timeit QREGTimer "update L-BFGS cache" begin
        current_inner > 1 && update!(cache, alpha, d, g)
        end
        @timeit QREGTimer "compute search direction" compute_lbfgs_direction!(d, g, cache, H)

        # Compute (AᵀA + λI) dₙ₊₁
        @timeit QREGTimer "w = (AᵀA+λI)dₙ₊₁" mul!(w, AtApI, d)

        # Backtrack to make sure we satisfy descent
        # lossₙ₊₁ = lossₙ + α²/2 (|Adₙ₊₁|² + λ|dₙ₊₁|²) + α (∇ₙᵀdₙ₊₁)
        @timeit QREGTimer "backtracking linesearch" begin
        alpha = one(T)
        loss_1 = 1//2 * dot(d, w) # 1/2 [|Adₙ₊₁|² + λ|dₙ₊₁|²]
        loss_2 = -dot(g, d)       # ∇ₙᵀdₙ₊₁
        if loss_2 > 0 error("L-BFGS direction was not computed correctly at iteration $(iter)") end
        while (alpha*alpha*loss_1 + alpha*loss_2 > 0)
          alpha = 1//2 * alpha
        end
        end
        @timeit QREGTimer "compute xₙ₊₁" begin
        @. x = x + alpha*d
        end

        # Save the old gradient
        @timeit QREGTimer "save prev gradient" begin
        @. cache.q = g
        end

        # Update -∇
        @timeit QREGTimer "update gradient" begin
        @. g = g - alpha*w
        end
        @timeit QREGTimer "check convergence (inner)" not_stationary = norm(g) > gtol
      end
      end
      # Nesterov acceleration
      @timeit QREGTimer "Nesterov acceleration" begin
      k += 1
      @. x_next = x
      @. x = k/(k+3) * (x_next - x_prev); @. x = x + x_next
      @. x_prev = x_next
      end
      # Update residual
      @timeit QREGTimer "compute residual" begin
      @. r = b
      mul!(r, A, x, -one(T), one(T))
      end
    end
    iter += 1
    f_prev = f_curr
    @timeit QREGTimer "eval objective" f_curr = qreg_objective_uniform(r, q, h)
    if f_curr > f_prev k = 0 end
    @timeit QREGTimer "check convergence (outer)" converged = abs(f_curr - f_prev) < rtol * (f_prev + 1)
  end

  @timeit QREGTimer "summary" begin
  stats = (
    iterations = iter,
    inner = inner,
    converged = converged,
    xnorm = norm(x),
    rnorm = norm(r),
    loss1 = qreg_objective(r, q), # check function loss
    loss2 = f_curr,               # smoothed loss
  )
  end

  return x, r, stats
end

