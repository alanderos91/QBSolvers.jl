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

###
### Implementation
###

function solve_QREG(A::Matrix{T}, b::Vector{T}, x0::Vector{T}, n_blk::Int;
  q::Float64    = 0.5,
  h::Float64    = default_bandwidth(A),
  maxiter::Int  = 100,
  gtol::Float64 = 1e-3,
) where T
  #
  n_obs, n_var = size(A)
  var_per_blk = cld(n_var, n_blk)

  @assert rem(n_var, n_blk) == 0
  @assert var_per_blk > 0
  @assert 0 < q < 1

  # Setup AᵀA and block diagonal D
  AtApI = GramPlusDiag(A; alpha=one(T), beta=zero(T))
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
  @. D.diag += lambda_max

  # 
  x = deepcopy(x0)
  d = zeros(n_var)
  g = zeros(n_var)
  u = zeros(n_obs)
  v = zeros(n_obs)
  w = zeros(n_var)
  z_new = zeros(n_obs)
  z_old = zeros(n_obs)

  # Initialize the difference, d₁ = x₁ - x₀ = D⁻¹ (-∇₀)
  r = copy(b)                     # r₀ = b - Ax₀
  mul!(r, A, x, -one(T), one(T))
  prox_abs!(z_new, r, h)          # z₀ = prox[h|⋅|](r₀)
  @. u = r - z_new                # -∇₀ = Aᵀ[r₀ - z₀ .+ (2q-1)h]
  @. u = u + (2*q-1)*h
  mul!(g, transpose(A), u, inv(T(2*h)), zero(T))
  ldiv!(d, D, g)                  # d₁ = D⁻¹(-∇₀)
  @. x = x + d                    # x₁ = x₀ + d₁

  # Update recurrences to n = 1
  mul!(u, A, d)               # r₁ = r₀ - A d₁
  @. r = r - u
  prox_abs!(z_old, r, h)      # z₁ = prox[h|⋅|](r₁)
  @. v = u + z_old - z_new
  mul!(w, transpose(A), v)    # Aᵀ(A d₁ + z₁ - z₀)
  @. g = g - inv(T(2*h)) * w  # -∇₁ = -∇₀ - (2h)⁻¹Aᵀ(A d₁ + z₁ - z₀)

  # Iterate the algorithm map
  iter = 1
  converged = norm(g) <= gtol

  while !converged && (iter < maxiter)
    iter += 1

    # Compute dₙ₊₁ = dₙ - (2h)⁻¹D⁻¹ Aᵀ (A dₙ + zₙ - zₙ₋₁)
    ldiv!(D, w)
    @. d = d - inv(T(2*h)) * w

    # Update coefficients
    @. x = x + d

    # Compute rₙ₊₁ = rₙ - A dₙ₊₁
    mul!(u, A, d)
    @. r = r - u

    # Compute zₙ₊₁ = prox[h|⋅|](rₙ₊₁)
    prox_abs!(z_new, r, h)

    # Compute A dₙ₊₁ + zₙ₊₁ - zₙ
    @. v = u + z_new - z_old

    # Compute Aᵀ (A dₙ₊₁ + zₙ₊₁ - zₙ) and -∇ₙ₊₁
    mul!(w, transpose(A), v)
    @. g = g - inv(T(2*h)) * w  # assumes g is always -∇
    converged = norm(g) <= gtol

    # don't copy, swap'em
    z_new, z_old = z_old, z_new
  end

  stats = (
    iterations = iter,
    converged = converged,
    xnorm = norm(x),
    rnorm = norm(r),
    gnorm = norm(g),
  )

  return x, r, stats
end
