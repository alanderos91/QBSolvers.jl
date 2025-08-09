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

function __QREG_loop__(out_workspace, out_linmaps, inn_workspace, inn_linmaps, q, h, rtol, gtol, maxiter, accel)
  # unpack
  x, x_next, x_prev, r, u, z = out_workspace
  A, b = out_linmaps
  AtApI, H = inn_linmaps
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
    maybe_rescale!(x, AtApI.A)
    init_recurrences!(inn_workspace, AtApI, u, H)
    inner_iter, _ = __OLS_loop__(inn_workspace, inn_linmaps, gtol, maxiter, 1, 1, accel)
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

function solve_QREG(A::AbstractMatrix{T}, b::Vector{T};
  q::Real           = T(0.5),
  h::Real           = T(default_bandwidth(A)),
  lambda::Real      = zero(T),
  gram::Bool        = _cache_gram_heuristic_(A),
  normalize::Symbol = false,
  memory::Int       = 10,
  maxiter::Int      = maximum(size(A)),
  gtol::Float64     = 1e-3 * sqrt(size(A, 2)),
  rtol::Float64     = 1e-6,
  accel::Bool       = false,
) where T
  #
  n_obs, n_var = size(A)
  @assert 0 < q < 1

  lambda = zero(T)
  AtA = GramPlusDiag(A; gram=gram)              # may cache AtA

  x = zeros(T, n_var)
  x_next = deepcopy(x)
  x_prev = deepcopy(x)
  w = deepcopy(x)
  g = zeros(T, n_var)
  d = zeros(T, n_var)
  r = zeros(T, n_obs)
  u = zeros(T, n_obs)
  z = zeros(T, n_obs)
  cache = LBFGSCache{T}(n_var, memory)
  
  ols_workspace = (x, g, d, w, cache)
  out_workspace = (x, x_next, x_prev, r, u, z)
  inn_workspace = (x, g, d, w, x_next)

  run = let A = A, b = b
    function(AtApI, H)
      ols_linmaps = (AtApI.A, b, AtApI, H)
      out_linmaps = (A, b)
      inn_linmaps = (AtApI, H)

      # Construct the OLS solution to use as our initial guess.
      ols_iter, _ = __OLS_lbfgs__(ols_workspace, ols_linmaps, gtol, maxiter, 0)
      @. x_next = x
      @. x_prev = x

      # Solve the QREG problem
      _, out_iter, inn_iter, _converged = __QREG_loop__(
        out_workspace, out_linmaps,
        inn_workspace, inn_linmaps,
        q, h, rtol, gtol, maxiter, accel
      )

      return ols_iter, out_iter, inn_iter, _converged
    end
  end
  iter_init, iter_outer, iter_inner, converged = with_qub_matrix(run, AtA, lambda, normalize)

  stats = (
    iterations = iter_outer,
    init  = iter_init,
    inner = iter_inner,
    converged = converged,
    xnorm = norm(x),
    rnorm = norm(r),
    loss1 = qreg_objective(r, q),             # check function loss
    loss2 = qreg_objective_uniform(r, q, h),  # smoothed loss
  )

  return x, r, stats
end

function solve_QREG_lbfgs(_A::AbstractMatrix{T}, b::Vector{T};
  q::Real           = T(0.5),
  h::Real           = T(default_bandwidth(_A)),
  lambda::Real      = T(0.0),
  gram::Bool        = _cache_gram_heuristic_(_A),
  normalize::Symbol = :none,
  memory::Int       = 10,
  maxiter::Int      = maximum(size(_A)),
  gtol::Float64     = 1e-3 * sqrt(size(_A, 2)),
  rtol::Float64     = 1e-6,
  version::Int      = 1,
  epsilon::Real     = h*h/4,
  accel::Bool       = false,
) where T
  #
  n_obs, n_var = size(_A)
  @assert 0 < q < 1

  lambda = zero(T)
  AtA = GramPlusDiag(_A; gram=gram)              # may cache AtA

  x = zeros(T, n_var)
  x_next = deepcopy(x)
  x_prev = deepcopy(x)
  w = deepcopy(x)
  g = zeros(T, n_var)
  d = zeros(T, n_var)
  r = zeros(T, n_obs)
  u = zeros(T, n_obs)
  z = zeros(T, n_obs)
  cache = LBFGSCache{T}(n_var, memory)

  out_workspace = (x, x_next, x_prev, r, u, z)
  inn_workspace = (x, g, d, w, cache)
  workspace = (x, x_next, x_prev, g, d, r, u, z, cache)

  run = let A = AtA.A, b = b, u = u
    wfun1(dst, src) = compute_weights!(weight_uniform, dst, src, q, h)
    ofun1(r) = qreg_objective_uniform(r, q, h)
    
    wfun2(dst, src) = compute_weights!(weight_sharp, dst, src, q, epsilon)
    ofun2(r) = qreg_objective_sharp(r, q, epsilon)

    function(AtApI, H)
      ols_linmaps = (AtApI.A, b, AtApI, H)
      out_linmaps = (A, b)
      inn_linmaps = (AtApI.A, u, AtApI, H)
      linmaps = (AtApI.A, b, H)

      # Construct the OLS solution to use as our initial guess.
      iter0, _ = __OLS_lbfgs__(inn_workspace, ols_linmaps, gtol, maxiter, 0)
      @. x_prev = x

      # Solve the QREG problem
      if version == 1
        # double loop
        _, _iter, _inner, _converged = __QREG_lbfgs__(out_workspace, out_linmaps, inn_workspace, inn_linmaps, q, h, rtol, gtol, maxiter, accel)
      elseif version == 2
        #
        # L-BFGS on kernel-smoothed objective
        #
        _, _iter, _inner, _converged = __QREG_lbfgs_single__(wfun1, ofun1, workspace, linmaps, rtol, maxiter, accel)
      elseif version == 3
        #
        # L-BFGS on sharp quadratic approximation
        #
        _, _iter, _inner, _converged = __QREG_lbfgs_single__(wfun2, ofun2, workspace, linmaps, rtol, maxiter, accel)
      else
        _iter, _inner, _converged = 0, 0, false
      end
      _inner += iter0
      return _iter, _inner, _converged
    end
  end
  iter, inner, converged = with_qub_matrix(run, AtA, lambda, normalize)

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

function __QREG_lbfgs__(out_workspace, out_linmaps, inn_workspace, inn_linmaps, q, h, rtol, gtol, maxiter, accel)
  # unpack
  x, x_next, x_prev, r, u, z = out_workspace
  A, b = out_linmaps
  _, _, AtApI, _ = inn_linmaps
  cache = last(inn_workspace)
  T = eltype(A)

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
    maybe_rescale!(x, AtApI.A)
    inner_iter, _ = __OLS_lbfgs__(inn_workspace, inn_linmaps, gtol, cache.memory_size, 0)
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
  A, b, H = linmaps
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
  maybe_unscale!(x, A)

  return r, iter, 0, converged
end

