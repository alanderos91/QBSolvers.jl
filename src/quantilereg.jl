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
  n_obs, n_var = size(A)
  var_per_blk = cld(n_var, n_blk)

  @assert rem(n_var, n_blk) == 0
  @assert var_per_blk > 0
  @assert 0 < q < 1

  AtA = GramPlusDiag(A; gram=gram)              # may cache AtA
  AtApI = GramPlusDiag(AtA, one(T), T(lambda))  # same data, add lazy shift by λI

  if var_per_blk > 1
    #
    # Block Diagonal Hessian
    #
    if normalize
      let
        AtA0 = NormalizedGramPlusDiag(AtA)
        J = compute_block_diagonal(AtA0, n_blk;
          alpha   = one(T),
          beta    = zero(T),
          factor  = false,
          gram    = n_obs > var_per_blk
        )
        rho = estimate_spectral_radius(AtA0, J, maxiter=3)
        S = Diagonal(@. rho*AtA0.A.scale^2 + lambda)
        @. AtA0.A.scale = 1 # need J̃ = √S⋅J⋅√S + ρS + λ = ZᵀZ + ρS + λ
        J̃ = update_factors!(J, AtA0.A, S, one(T), one(T))
        H = BlkDiagPlusRank1(n_obs, n_var, J̃, AtA0.A.shift, one(T), T(n_obs))
        _solve_QREG_loop(AtApI, H, b, x0, q, h, T(lambda); kwargs...)
      end
    else
      let
        J = compute_block_diagonal(AtA, n_blk;
          alpha   = one(T),
          beta    = zero(T),
          factor  = false,
          gram    = n_obs > var_per_blk
        )
        rho = estimate_spectral_radius(AtA, J, maxiter=3)
        H = update_factors!(J, one(T), lambda + rho)
        _solve_QREG_loop(AtApI, H, b, x0, q, h, T(lambda); kwargs...)
      end
    end
  else
    #
    # Diagonal (Plus Rank-1) Hessian
    #
    if normalize
      let
        AtA0 = NormalizedGramPlusDiag(AtA)
        J = compute_main_diagonal(AtA0.A, AtA0.AtA)
        rho = estimate_spectral_radius(AtA0, J, maxiter=3)
        @. J.diag = (1+rho)*AtA0.A.scale^2 + T(lambda)
        H = BlkDiagPlusRank1(n_obs, n_var, J, AtA0.A.shift, one(T), T(n_obs))
        _solve_QREG_loop(AtApI, H, b, x0, q, h, T(lambda); kwargs...)
      end
    else
      let
        J = compute_main_diagonal(AtA.A, AtA.AtA)
        rho = estimate_spectral_radius(AtA, J, maxiter=3)
        H = J
        @. H.diag = H.diag + rho + lambda
        _solve_QREG_loop(AtApI, H, b, x0, q, h, T(lambda); kwargs...)
      end
    end
  end
end

function _solve_QREG_loop(AtApI::GramPlusDiag{T}, H, b::Vector{T}, x0::Vector{T}, q::T, h::T, lambda::T;
  maxiter::Int  = 100,
  gtol::Float64 = 1e-3,
  rtol::Float64 = 1e-6,
) where T
  #
  A = AtApI.A
  n_obs, n_var = size(A)

  # 
  x = deepcopy(x0); x_next = deepcopy(x0); x_prev = deepcopy(x0)
  d = zeros(n_var)
  g = zeros(n_var)
  u = zeros(n_obs)
  z = zeros(n_obs)

  # Initialize
  r = copy(b) # r = b - A*x
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
    # Solve with QUB
    #
    mul!(x_next, transpose(A), u) # save this constant
    mul!(g, AtApI, x)             # -∇ = Aᵀ(u - A*x)
    @. g = x_next - g
    not_stationary = norm(g) > gtol
    if not_stationary
      while not_stationary
        inner += 1
        ldiv!(d, H, g)
        @. x = x + d
        mul!(g, AtApI, x)
        @. g = x_next - g
        not_stationary = norm(g) > gtol
      end
      # Nesterov acceleration
      k += 1
      @. x_next = x
      @. x = k/(k+3) * (x_next - x_prev); @. x = x + x_next
      @. x_prev = x_next
      # Update residual
      @. r = b
      mul!(r, A, x, -one(T), one(T))
    end
    iter += 1
    f_prev = f_curr
    f_curr = qreg_objective_uniform(r, q, h)
    if f_curr > f_prev k = 0 end
    converged = abs(f_curr - f_prev) < rtol * (f_prev + 1)
  end

  stats = (
    iterations = iter,
    inner = inner,
    converged = converged,
    xnorm = norm(x),
    rnorm = norm(r),
    loss1 = qreg_objective(r, q), # check function loss
    loss2 = f_curr,               # smoothed loss
  )

  return x, r, stats
end

