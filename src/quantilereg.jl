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

function __QREG_loop__(workspace, linmaps, q, h, rtol, gtol, maxiter)
  global PkgTimer
  # unpack
  x, x_next, x_prev, g, d, r, u, z = workspace
  A, b, AtApI, H = linmaps
  w = x_next
  T = eltype(A)

  # Initialize r = b - A*x
  @timeit PkgTimer "init r₀ = b - Ax₀" begin
    @. r = b
    mul!(r, A, x, -one(T), one(T))
  end

  # Iterate the algorithm map
  iter = 0  # outer iterations; Moreau majorization
  inner = 0 # inner iterations; QUB majorization
  k = 0     # Nesterov iterations
  converged = false
  @timeit PkgTimer "eval objective" f_curr = qreg_objective_uniform(r, q, h)
  while !converged && (iter < maxiter)
    #
    # Setup the next LS problem, |Ax - u|²
    #   where u = b - zₙ + (2*q-1)*h 1
    #
    @timeit PkgTimer "eval prox" prox_abs!(z, r, h)
    @timeit PkgTimer "set RHS u" @. u = b - z + (2*q-1)*h
    #
    # Solve with QUB
    #
    @timeit PkgTimer "init recurrences" init_recurrences!((x, g, d, w), AtApI, u, H)
    @timeit PkgTimer "inner solve" inner_iter, is_stationary = __OLS_loop__((x, g, d, w), (AtApI, H), gtol, maxiter, 1)
    # @show iter, inner_iter, is_stationary
    inner += inner_iter
    @timeit PkgTimer "Nesterov accel" begin
      if inner_iter > 0
        #
        # Nesterov acceleration
        #
        k += 1
        @. x_next = x
        @. x = k/(k+3) * (x_next - x_prev); @. x = x + x_next
        @. x_prev = x_next
      end
    end
    #
    # Update residual
    #
    @timeit PkgTimer "update rₙ = b - Axₙ" begin
      @. r = b
      mul!(r, A, x, -one(T), one(T))
    end
    iter += 1
    f_prev = f_curr
    @timeit PkgTimer "eval objective" f_curr = qreg_objective_uniform(r, q, h)
    if f_curr > f_prev k = 0 end
    @timeit PkgTimer "check convergence (outer)" converged = abs(f_curr - f_prev) < rtol * (f_prev + 1)
  end

  return r, iter, inner, converged
end

function solve_QREG(A::AbstractMatrix{T}, b::Vector{T}, x0::Vector{T}, n_blk::Int;
  q::Real         = T(0.5),
  h::Real         = T(default_bandwidth(A)),
  lambda::Real    = T(0.0),
  gram::Bool      = _cache_gram_heuristic_(A),
  normalize::Bool = false,
  memory::Int     = 10,
  maxiter::Int    = 100,
  gtol::Float64   = 1e-3,
  rtol::Float64   = 1e-6,
) where T
  #
  global PkgTimer; enable_timer!(PkgTimer); reset_timer!(PkgTimer)
  n_obs, n_var = size(A)
  var_per_blk = cld(n_var, n_blk)

  @assert rem(n_var, n_blk) == 0
  @assert var_per_blk > 0
  @assert 0 < q < 1

  @timeit PkgTimer "init AtA + λI; gram=$(gram)" begin
    AtA = GramPlusDiag(A; gram=gram)              # may cache AtA
    AtApI = GramPlusDiag(AtA, one(T), T(lambda))  # same data, add lazy shift by λI
  end

  @timeit PkgTimer "init workspace" begin
    x = deepcopy(x0)
    x_next = deepcopy(x)
    x_prev = deepcopy(x)
    g = zeros(n_var)
    d = zeros(n_var)
    r = zeros(n_obs)
    u = zeros(n_obs)
    z = zeros(n_obs)
    workspace = (x, x_next, x_prev, g, d, r, u, z)
  end

  run = let
    function(H)
      # constants/linear maps
      linmaps = (A, b, AtApI, H)
      # Construct the OLS solution to use as our initial guess.
      @timeit PkgTimer "init L-BFGS cache" cache = LBFGSCache{T}(n_var, memory); w = x_next
      @timeit PkgTimer "init x₀ OLS" begin
        iter0, _ = __OLS_lbfgs__((x, g, d, w, cache), linmaps, gtol, maxiter, 0)
        @. x_next = x
        @. x_prev = x
      end
      # Solve the QREG problem
      @timeit PkgTimer "QREG Loop" begin
        _, _iter, _inner, _converged = __QREG_loop__(workspace, linmaps, q, h, rtol, gtol, maxiter)
        _inner += iter0
      end
      return _iter, _inner, _converged
    end
  end
  iter, inner, converged = with_qub_matrix(run, AtA, n_obs, n_var, n_blk, var_per_blk, lambda, true, false)

  @timeit PkgTimer "summary" begin
    stats = (
      iterations = iter,
      inner = inner,
      converged = converged,
      xnorm = norm(x),
      rnorm = norm(r),
      loss1 = qreg_objective(r, q),             # check function loss
      loss2 = qreg_objective_uniform(r, q, h),  # smoothed loss
    )
  end
  disable_timer!(PkgTimer)
  display(PkgTimer)
  return x, r, stats
end

function solve_QREG_lbfgs(A::AbstractMatrix{T}, b::Vector{T}, x0::Vector{T}, n_blk::Int;
  q::Real         = T(0.5),
  h::Real         = T(default_bandwidth(A)),
  lambda::Real    = T(0.0),
  gram::Bool      = _cache_gram_heuristic_(A),
  normalize::Bool = false,
  memory::Int     = 10,
  maxiter::Int    = 100,
  gtol::Float64   = 1e-3,
  rtol::Float64   = 1e-6,
) where T
  #
  global PkgTimer; enable_timer!(PkgTimer); reset_timer!(PkgTimer)
  n_obs, n_var = size(A)
  var_per_blk = cld(n_var, n_blk)

  @assert rem(n_var, n_blk) == 0
  @assert var_per_blk > 0
  @assert 0 < q < 1

  @timeit PkgTimer "init AtA + λI; gram=$(gram)" begin
    lambda = zero(T)
    AtA = GramPlusDiag(A; gram=gram)              # may cache AtA
    AtApI = GramPlusDiag(AtA, one(T), T(lambda))  # same data, add lazy shift by λI
  end

  @timeit PkgTimer "init workspace" begin
    x = deepcopy(x0)
    x_next = deepcopy(x)
    x_prev = deepcopy(x)
    g = zeros(n_var)
    d = zeros(n_var)
    r = zeros(n_obs)
    u = zeros(n_obs)
    z = zeros(n_obs)
    @timeit PkgTimer "init L-BFGS cache" cache = LBFGSCache{T}(n_var, memory)
    workspace = (x, x_next, x_prev, g, d, r, u, z, cache)
  end
  run = let
    function(H)
      # constants/linear maps
      linmaps = (A, b, AtApI, H)
      # Construct the OLS solution to use as our initial guess.
      w = x_next
      @timeit PkgTimer "init x₀ OLS" begin
        iter0, _ = __OLS_lbfgs__((x, g, d, w, cache), linmaps, gtol, maxiter, 0)
        @. x_next = x
        @. x_prev = x
      end
      # Solve the QREG problem
      @timeit PkgTimer "QREG Loop" begin
        _, _iter, _inner, _converged = __QREG_lbfgs__(workspace, linmaps, q, h, rtol, gtol, maxiter)
        _inner += iter0
      end
      return _iter, _inner, _converged
    end
  end
  iter, inner, converged = with_qub_matrix(run, AtA, n_obs, n_var, n_blk, var_per_blk, lambda, true, false)

  @timeit PkgTimer "summary" begin
    stats = (
      iterations = iter,
      inner = inner,
      converged = converged,
      xnorm = norm(x),
      rnorm = norm(r),
      loss1 = qreg_objective(r, q),             # check function loss
      loss2 = qreg_objective_uniform(r, q, h),  # smoothed loss
    )
  end
  disable_timer!(PkgTimer)
  display(PkgTimer)
  return x, r, stats
end

function __QREG_lbfgs__(workspace, linmaps, q, h, rtol, gtol, maxiter)
  # unpack
  x, x_next, x_prev, g, d, r, u, z, cache = workspace
  A, b, AtApI, H = linmaps
  w = x_next
  T = eltype(A)

  # Initialize r = b - A*x
  @timeit PkgTimer "init r₀ = b - Ax₀" begin
    @. r = b
    mul!(r, A, x, -one(T), one(T))
  end

  # Iterate the algorithm map
  iter = 0  # outer iterations; Moreau majorization
  inner = 0 # inner iterations; QUB majorization
  k = 0     # Nesterov iterations
  converged = false
  @timeit PkgTimer "eval objective" f_curr = qreg_objective_uniform(r, q, h)
  while !converged && (iter < maxiter)
    #
    # Setup the next LS problem, |Ax - u|²
    #   where u = b - zₙ + (2*q-1)*h 1
    #
    @timeit PkgTimer "eval prox" prox_abs!(z, r, h)
    @timeit PkgTimer "set RHS u" @. u = b - z + (2*q-1)*h
    #
    # Solve with QUB; need to reset the L-BFGS cache first
    #
    cache.current_index = 0
    cache.current_size = 0
    @timeit PkgTimer "inner solve" inner_iter, is_stationary = __OLS_lbfgs__((x, g, d, w, cache), (A, u, AtApI, H), gtol, maxiter, 0)
    # @show iter, inner_iter, is_stationary
    inner += inner_iter
    @timeit PkgTimer "Nesterov accel" begin
      if inner_iter > 0
        #
        # Nesterov acceleration
        #
        k += 1
        @. x_next = x
        @. x = k/(k+3) * (x_next - x_prev); @. x = x + x_next
        @. x_prev = x_next
      end
    end
    #
    # Update residual
    #
    @timeit PkgTimer "update rₙ = b - Axₙ" begin
      @. r = b
      mul!(r, A, x, -one(T), one(T))
    end
    iter += 1
    f_prev = f_curr
    @timeit PkgTimer "eval objective" f_curr = qreg_objective_uniform(r, q, h)
    if f_curr > f_prev k = 0 end
    @timeit PkgTimer "check convergence (outer)" converged = abs(f_curr - f_prev) < rtol * (f_prev + 1)
  end

  return r, iter, inner, converged
end

