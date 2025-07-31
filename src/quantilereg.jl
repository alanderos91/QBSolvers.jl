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
    f(ri) = qreg_loss(ri, q)
    mapreduce(f, +, r) / length(r)
  end
end

# Uniform kernel
function qreg_loss_uniform(r::T, q::T, h::T) where T <: Real
  absr = abs(r)
  C = ifelse(absr > h, absr, h/2 * (1 + (r/h)^2))
  return (q-1//2)*r + 1//2*C
end

weight_uniform(ri, h) = ifelse(abs(ri) > h, sign(ri), ri)
weight_uniform(ri, q, h) = (q-1//2) + weight_uniform(ri, h)

function qreg_objective_uniform(r, q, h)
  let r =r, q = q, h = h
    f(ri) = qreg_loss_uniform(ri, q, h)
    mapreduce(f, +, r) / length(r)
  end
end

# sharp approximation
abs_approx(x, ϵ) = sqrt(x*x + ϵ)
weight_sharp(ri, ϵ) = 1//2*ri/abs_approx(ri, ϵ)
weight_sharp(ri, q, ϵ) = (q-1//2) + weight_sharp(ri, ϵ)

function qreg_loss_sharp(r::T, q::T, ϵ::T) where T <: Real
  return (q-1//2)*r + 1//2*abs_approx(r, ϵ)
end

function qreg_objective_sharp(r, q, ϵ)
  let r = r, q = q, ϵ = ϵ
    f(ri) = qreg_loss_sharp(ri, q, ϵ)
    mapreduce(f, +, r) / length(r)
  end
end

function compute_weights!(weightf, z, r, q, c)
  let r = r, q = q, c = c
    f(ri) = weightf(ri, q, c)
    map!(f, z, r)
  end
end

###
### Implementation
###

function __QREG_loop__(workspace, linmaps, q, h, rtol, gtol, maxiter, accel)
  # unpack
  x, x_next, x_prev, g, d, r, u, z = workspace
  A, b, AtApI, H = linmaps
  w = x_next
  T = eltype(A)

  # Initialize r = b - A*x
  @. r = b
  mul!(r, A, x, -one(T), one(T))

  # Iterate the algorithm map
  iter = 0  # outer iterations; Moreau majorization
  inner = 0 # inner iterations; QUB majorization
  k = 1     # Nesterov iterations
  converged = false
  f_curr = qreg_objective_uniform(r, q, h)
  while !converged && (iter < maxiter)
    #
    # Setup the next LS problem, |Ax - u|²
    #   where u = b - zₙ + (2*q-1)*h 1
    #
    prox_abs!(z, r, h)
    @. u = b - z + (2*q-1)*h
    #
    # Solve with QUB
    #
    init_recurrences!((x, g, d, w), AtApI, u, H)
    inner_iter, _ = __OLS_loop__((x, g, d, w), (AtApI, H), gtol, maxiter, 1)
    inner += inner_iter
    if accel
      #
      # Nesterov acceleration
      #
      if k > 1
        @. x_next = x
        @. x = (x_next - x_prev)
        @. x_prev = x_next
        @. x = x_next + (k-1)/(k+2)*x
      else
        @. x_prev = x
      end
      @. r = b
      mul!(r, A, x, -one(T), one(T))
      f_prev = f_curr
      f_curr = qreg_objective_uniform(r, q, h)
      k = ifelse(f_curr > f_prev, 1, k+1)
    else
      @. r = b
      mul!(r, A, x, -one(T), one(T))
      f_prev = f_curr
      f_curr = qreg_objective_uniform(r, q, h)
    end
    iter += 1
    converged = abs(f_curr - f_prev) < rtol * (f_prev + 1)
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
  accel::Bool     = false,
) where T
  #
  n_obs, n_var = size(A)
  var_per_blk = cld(n_var, n_blk)

  @assert rem(n_var, n_blk) == 0
  @assert var_per_blk > 0
  @assert 0 < q < 1

  lambda = zero(T)
  AtA = GramPlusDiag(A; gram=gram)              # may cache AtA

  x = deepcopy(x0)
  x_next = deepcopy(x)
  x_prev = deepcopy(x)
  g = zeros(n_var)
  d = zeros(n_var)
  r = zeros(n_obs)
  u = zeros(n_obs)
  z = zeros(n_obs)
  workspace = (x, x_next, x_prev, g, d, r, u, z)

  run = let
    function(AtApI, H)
      # constants/linear maps
      linmaps = (AtApI.A, b, AtApI, H)
      # Construct the OLS solution to use as our initial guess.
      cache = LBFGSCache{T}(n_var, memory); w = x_next
      iter0, _ = __OLS_lbfgs__((x, g, d, w, cache), linmaps, gtol*sqrt(n_var), maxiter, 0)
      @. x_next = x
      @. x_prev = x
      # Solve the QREG problem
      _, _iter, _inner, _converged = __QREG_loop__(workspace, linmaps, q, h, rtol, gtol, maxiter, accel)
      _inner += iter0
      return _iter, _inner, _converged
    end
  end
  iter, inner, converged = with_qub_matrix(run, AtA, n_obs, n_var, n_blk, var_per_blk, lambda, true, normalize)

  stats = (
    iterations = iter,
    inner = inner,
    converged = converged,
    xnorm = norm(x),
    rnorm = norm(r),
    loss1 = qreg_objective(r, q),             # check function loss
    loss2 = qreg_objective_uniform(r, q, h),  # smoothed loss
  )

  return x, r, stats
end

function solve_QREG_lbfgs(_A::AbstractMatrix{T}, b::Vector{T}, x0::Vector{T}, n_blk::Int;
  q::Real         = T(0.5),
  h::Real         = T(default_bandwidth(A)),
  lambda::Real    = T(0.0),
  gram::Bool      = _cache_gram_heuristic_(A),
  normalize::Bool = false,
  memory::Int     = 10,
  maxiter::Int    = 100,
  gtol::Float64   = 1e-3,
  rtol::Float64   = 1e-6,
  version::Int    = 1,
  epsilon::Real   = h*h/4,
  accel::Bool     = false,
) where T
  #
  n_obs, n_var = size(_A)
  var_per_blk = cld(n_var, n_blk)

  @assert rem(n_var, n_blk) == 0
  @assert var_per_blk > 0
  @assert 0 < q < 1

  lambda = zero(T)
  AtA = GramPlusDiag(_A; gram=gram)              # may cache AtA

  x = deepcopy(x0)
  x_next = deepcopy(x)
  x_prev = deepcopy(x)
  g = zeros(n_var)
  d = zeros(n_var)
  r = zeros(n_obs)
  u = zeros(n_obs)
  z = zeros(n_obs)
  cache = LBFGSCache{T}(n_var, memory)
  workspace = (x, x_next, x_prev, g, d, r, u, z, cache)
  run = let
    wfun1(dst, src) = compute_weights!(weight_uniform, dst, src, q, h)
    ofun1(r) = qreg_objective_uniform(r, q, h)
    
    wfun2(dst, src) = compute_weights!(weight_sharp, dst, src, q, epsilon)
    ofun2(r) = qreg_objective_sharp(r, q, epsilon)

    function(AtApI, H)
      # constants/linear maps
      A = AtApI.A # ugly, confusing, this A is not global!
      linmaps = (A, b, AtApI, H)
      # Construct the OLS solution to use as our initial guess.
      w = x_next
      iter0, _ = __OLS_lbfgs__((x, g, d, w, cache), linmaps, gtol*sqrt(n_var), maxiter, 0)
      @. x_next = x
      @. x_prev = x
      # Solve the QREG problem
      if version == 1
        # double loop
        _, _iter, _inner, _converged = __QREG_lbfgs__(workspace, linmaps, q, h, rtol, gtol, maxiter, accel)
      elseif version == 2
        #
        # L-BFGS on kernel-smoothed objective
        #
        _, _iter, _inner, _converged = __QREG_lbfgs_single__(wfun1, ofun1, workspace, linmaps, rtol, maxiter, accel)
      elseif version == 3
        #
        # L-BFGS on kernel-smoothed objective; re-calculate QUB matrix
        #
        @. r = b
        mul!(r, A, x, -one(T), one(T))
        w = map(ri -> weight_uniform(ri, h), r)
        AtWA = A'*Diagonal(w)*A
        rho, _ = powm!(AtA - AtWA, ones(T, n_var), maxiter=3)
        J = Diagonal(AtWA) + abs(rho)*I
        _, _iter, _inner, _converged = __QREG_lbfgs_single__(wfun1, ofun1, workspace, (A, b, AtApI, J), rtol, maxiter, accel)
      elseif version == 4
        #
        # L-BFGS on sharp quadratic approximation
        #
        _, _iter, _inner, _converged = __QREG_lbfgs_single__(wfun2, ofun2, workspace, linmaps, rtol, maxiter, accel)
      elseif version == 5
        #
        # L-BFGS on sharp quadratic approximation; re-calculate QUB matrix
        #
        @. r = b
        mul!(r, A, x, -one(T), one(T))
        w = map(ri -> weight_sharp(ri, epsilon), r)
        AtWA = A'*Diagonal(w)*A
        rho, _ = powm!(AtA - AtWA, ones(T, n_var), maxiter=3)
        J = Diagonal(AtWA) + abs(rho)*I
        _, _iter, _inner, _converged = __QREG_lbfgs_single__(wfun2, ofun2, workspace, (A, b, AtApI, J), rtol, maxiter, accel)
      end
      _inner += iter0
      return _iter, _inner, _converged
    end
  end
  iter, inner, converged = with_qub_matrix(run, AtA, n_obs, n_var, n_blk, var_per_blk, lambda, true, normalize)

  stats = (
    iterations = iter,
    inner = inner,
    converged = converged,
    xnorm = norm(x),
    rnorm = norm(r),
    loss1 = qreg_objective(r, q),               # check function loss
    loss2 = qreg_objective_uniform(r, q, h),    # smoothed loss
    loss3 = qreg_objective_sharp(r, q, epsilon) # sharp approximation
  )

  return x, r, stats
end

function __QREG_lbfgs__(workspace, linmaps, q, h, rtol, gtol, maxiter, accel)
  # unpack
  x, x_next, x_prev, g, d, r, u, z, cache = workspace
  A, b, AtApI, H = linmaps
  w = x_next
  T = eltype(A)
  n_var, n_obs = size(A)

  # Initialize r = b - A*x
  @. r = b
  mul!(r, A, x, -one(T), one(T))

  # Iterate the algorithm map
  iter = 0  # outer iterations; Moreau majorization
  inner = 0 # inner iterations; QUB majorization
  k = 0     # Nesterov iterations
  converged = false
  f_curr = qreg_objective_uniform(r, q, h)
  while !converged && (iter < maxiter)
    #
    # Setup the next LS problem, |Ax - u|²
    #   where u = b - zₙ + (2*q-1)*h 1
    #
    prox_abs!(z, r, h)
    @. u = b - z + (2*q-1)*h
    #
    # Solve with QUB; need to reset the L-BFGS cache first
    #
    cache.current_index = 0
    cache.current_size = 0
    inner_iter, _ = __OLS_lbfgs__((x, g, d, w, cache), (A, u, AtApI, H), gtol*sqrt(n_var), cache.memory_size, 0)
    inner += inner_iter
    if accel
      #
      # Nesterov acceleration
      #
      if k > 1
        @. x_next = x
        @. x = (x_next - x_prev)
        @. x_prev = x_next
        @. x = x_next + (k-1)/(k+2)*x
      else
        @. x_prev = x
      end
      @. r = b
      mul!(r, A, x, -one(T), one(T))
      f_prev = f_curr
      f_curr = qreg_objective_uniform(r, q, h)
      k = ifelse(f_curr > f_prev, 1, k+1)
    else
      @. r = b
      mul!(r, A, x, -one(T), one(T))
      f_prev = f_curr
      f_curr = qreg_objective_uniform(r, q, h)
    end
    iter += 1
    converged = abs(f_curr - f_prev) < rtol * (f_prev + 1)
  end

  return r, iter, inner, converged
end

function __QREG_lbfgs_single__(compute_weights!, objective, workspace, linmaps, rtol, maxiter, accel)
  # unpack
  x, x_next, x_prev, g, d, r, u, z, cache = workspace
  A, b, _, H = linmaps
  T = eltype(A)

  # Initialize r = b - A*x
  @. r = b
  mul!(r, A, x, -one(T), one(T))

  # Make sure L-BFGS cache is empty
  cache.current_index = 0
  cache.current_size = 0

  # Iterate the algorithm map
  iter = 0
  k = 1     # Nesterov iterations
  converged = false
  f_curr = objective(r)
  alpha = one(T)
  while !converged && (iter < maxiter)
    # Compute negative gradient
    compute_weights!(z, r)
    mul!(g, transpose(A), z)

    # Update the LBFGS workspace and compute the next direction
    iter > 1 && update!(cache, alpha, d, g)
    compute_lbfgs_direction!(d, g, cache, H)

    # Save the old gradient
    @. cache.q = g

    # Compute perturbation in residual due to update
    mul!(z, A, d)

    # Backtrack to make sure we satisfy descent
    alpha = one(T)
    @. u = r - alpha*z
    f_prev = f_curr
    f_curr = objective(u)
    while f_curr > f_prev
      alpha = 1//2 * alpha
      @. u = r - alpha*z
      f_curr = objective(u)
    end
    @. x = x + alpha*d
    if accel
      #
      # Nesterov acceleration
      #
      if k > 1
        @. x_next = x
        @. x = (x_next - x_prev)
        @. x_prev = x_next
        @. x = x_next + (k-1)/(k+2)*x

        @. r = b
        mul!(r, A, x, -one(T), one(T))

        f_prev = f_curr
        f_curr = objective(r)
      else
        @. x_prev = x
        @. r = u
      end
      k = ifelse(f_curr > f_prev, 1, k+1)
    else
      @. r = u
    end
    iter += 1
    converged = abs(f_curr - f_prev) < rtol * (f_prev + 1)
  end

  return r, iter, 0, converged
end

