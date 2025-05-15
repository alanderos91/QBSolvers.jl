module ParallelLeastSquares

using BlockArrays
using LinearAlgebra
using IterativeSolvers

import Base: getindex, size, eltype
import LinearAlgebra: issymmetric, mul!, ldiv!

const BLAS_THREADS = Ref{Int}(BLAS.get_num_threads())

###
### Gram plus Diagonal
###
### αAᵀA+βI
###

struct GramPlusDiag{T,matT1,matT2,vecT} <: AbstractMatrix{T}
  A::matT1
  AtA::matT2
  n_obs::Int
  n_var::Int
  tmp::vecT
  alpha::T
  beta::T
end

function GramPlusDiag(A::AbstractMatrix{T}; alpha::Real=one(T), beta::Real=zero(T)) where T
  n_obs, n_var = size(A)
  if n_obs > n_var
    AtA = transpose(A) * A
    tmp = similar(A, 0)
  else
    AtA = similar(A, 0, 0)
    tmp = similar(A, n_obs)
  end
  return GramPlusDiag(A, AtA, n_obs, n_var, tmp, T(alpha), T(beta))
end

function GramPlusDiag(gpd::GramPlusDiag{T}, alpha, beta) where T
  return GramPlusDiag(gpd.A, gpd.AtA, gpd.n_obs, gpd.n_var, gpd.tmp, T(alpha), T(beta))
end

function Base.getindex(gpd::GramPlusDiag, i, j)
  alpha, beta = gpd.alpha, gpd.beta
  @views begin
    alpha * dot(gpd.A[:,i], gpd.A[:,j]) + (i == j)*beta
  end
end

LinearAlgebra.issymmetric(::GramPlusDiag) = true
Base.size(gpd::GramPlusDiag) = (gpd.n_var, gpd.n_var)
Base.eltype(::GramPlusDiag{T}) where T = T

function LinearAlgebra.mul!(y::AbstractVector, gpd::GramPlusDiag, x::AbstractVector)
  if gpd.n_obs > gpd.n_var
    mul!(y, Symmetric(gpd.AtA), x)
  else
    mul!(gpd.tmp, gpd.A, x)
    mul!(y, transpose(gpd.A), gpd.tmp)
  end
  if iszero(gpd.beta)
    @. y = gpd.alpha * y
  else
    axpby!(gpd.beta, x, gpd.alpha, y)
  end
  return y
end

###
### Block Diagonal Hessian 
###
### | αA₁ᵀA₁+βI                         |
### |           αA₂ᵀA₂+βI               |
### |                     ⋱            |
### |                        αAₘᵀAₘ+βI  |
###

struct BlkDiagHessian{T,matT,blkmatT,gpdT} <: AbstractMatrix{T}
  A::matT
  n_obs::Int
  n_var::Int
  n_blk::Int
  A_blk::blkmatT
  diag::Vector{gpdT}
  chol::Vector{Cholesky{T,matT}}
  alpha::T
  beta::T
end

function BlkDiagHessian(A::Matrix{T}, n_blk::Int; alpha::Real=one(T), beta::Real=zero(T), factor::Bool=true) where T
  

  n_obs, n_var = size(A)
  var_per_blk = cld(n_var, n_blk)
  A_blk = BlockedArray(A, [n_obs], repeat([var_per_blk], n_blk))

  A_11 = view(A_blk, Block(1))
  D_11 = GramPlusDiag(A_11; alpha=alpha, beta=beta)
  diag = Vector{typeof(D_11)}(undef, n_blk)
  chol = Vector{Cholesky{T,Matrix{T}}}(undef, n_blk)

  diag[1] = D_11
  if factor
    chol[1] = cholesky!(Symmetric(form_AtApI(A_11, alpha, beta), :L))
  end
  for k in 2:n_blk
    A_kk = view(A_blk, Block(k))
    diag[k] = GramPlusDiag(A_kk; alpha=alpha, beta=beta)
    if factor
      chol[k] = cholesky!(Symmetric(form_AtApI(A_kk, alpha, beta), :L))
    end
  end
  return BlkDiagHessian(A, n_obs, n_var, n_blk, A_blk, diag, chol, T(alpha), T(beta))
end

function form_AtApI(A::AbstractMatrix{T}, alpha, beta) where T
  p = size(A, 2)
  form_AtApI!(Matrix{T}(I, p, p), A, alpha, beta)
end

function form_AtApI!(AtApI::AbstractMatrix{T}, A::AbstractMatrix{T}, alpha, beta) where T
  BLAS.syrk!('L', 'T', T(alpha), A, T(beta), AtApI) # AtApI enters as I
end

function update_factors!(bdh::BlkDiagHessian{T}, alpha, beta) where T
  for k in 1:bdh.n_blk
    A_kk = view(bdh.A_blk, Block(k))
    bdh.diag[k] = GramPlusDiag(bdh.diag[k], alpha, beta)
    if isassigned(bdh.chol, k)
      AtApI = form_AtApI!(bdh.chol[k].factors, A_kk, alpha, beta)
    else
      AtApI = form_AtApI(A_kk, alpha, beta)
    end
    bdh.chol[k] = cholesky!(Symmetric(AtApI, :L))
  end
  return bdh
end

function Base.getindex(bdh::BlkDiagHessian{T}, i, j) where T
  blkrng = axes(bdh.A_blk, 2)
  blk_i = findblock(blkrng, i)
  blk_j = findblock(blkrng, j)
  if blk_i == blk_j
    k = first(blk_i.n)
    D_kk = bdh.diag[k]
    var_per_blk = size(D_kk, 1)
    return getindex(D_kk, mod1(i, var_per_blk), mod1(j, var_per_blk))
  else
    return zero(T)
  end
end

LinearAlgebra.issymmetric(::BlkDiagHessian) = true
Base.size(bdh::BlkDiagHessian) = (bdh.n_var, bdh.n_var)
Base.eltype(::BlkDiagHessian{T}) where T = T

function LinearAlgebra.mul!(y::AbstractVector, bdh::BlkDiagHessian, x::AbstractVector)
  baxes = axes(bdh.A_blk, 2)
  for k in 1:bdh.n_blk
    D_kk = bdh.diag[k]
    @views idx = baxes[Block(k)]
    @views mul!(y[idx], D_kk, x[idx])
  end
  return y
end

function LinearAlgebra.ldiv!(y::AbstractVector, bdh::BlkDiagHessian, x::AbstractVector)
  baxes = axes(bdh.A_blk, 2)
  for k in 1:bdh.n_blk
    chol_D_kk = bdh.chol[k]
    @views idx = baxes[Block(k)]
    @views ldiv!(y[idx], chol_D_kk, x[idx])
  end
  return y
end

function LinearAlgebra.ldiv!(bdh::BlkDiagHessian, x::AbstractVector)
  baxes = axes(bdh.A_blk, 2)
  for k in 1:bdh.n_blk
    chol_D_kk = bdh.chol[k]
    @views idx = baxes[Block(k)]
    @views ldiv!(chol_D_kk, x[idx])
  end
  return x
end

###
### GramMinusBlkDiag
###
struct GramMinusBlkDiag{T,gpdT,bdhT,vecT} <: AbstractMatrix{T}
  AtA::gpdT
  D::bdhT
  tmp::vecT
end

function GramMinusBlkDiag(AtA::AbstractMatrix{T}, D::AbstractMatrix{T}) where T
  tmp = zeros(size(AtA, 1))
  gpdT = typeof(AtA)
  bdhT = typeof(D)
  vecT = typeof(tmp)
  GramMinusBlkDiag{T,gpdT,bdhT,vecT}(AtA, D, tmp)
end

function Base.getindex(gmbd::GramMinusBlkDiag, i, j)
  getindex(gmbd.AtA, i, j) - getindex(gmbd.D, i, j)
end

LinearAlgebra.issymmetric(::GramMinusBlkDiag) = true
Base.size(gmbd::GramMinusBlkDiag) = size(gmbd.AtA)
Base.eltype(::GramMinusBlkDiag{T}) where T = T

function LinearAlgebra.mul!(y::AbstractVector, gmbd::GramMinusBlkDiag, x::AbstractVector)
  # y = A' * (A * x)
  mul!(y, gmbd.AtA, x)

  # block substraction
  mul!(gmbd.tmp, gmbd.D, x)
  @. y = y - gmbd.tmp

  return y
end

###
### Initialization
###

function estimate_dominant_eigval(AtA, D; kwargs...)
  lambda_max, _, ch = powm!(GramMinusBlkDiag(AtA, D), ones(size(D, 1)); log=true, kwargs...)
  # @show ch
  return lambda_max
end

function initblocks!(::Type{T}, d, x, g, linmap, b, n_blk, lambda, use_qlb, tol_powm) where T
  A, AtA = linmap.A, linmap.AtA
  # Compute blocks along diagonal, Dₖ = Dₖₖ = AₖᵀAₖ + λI and extract their Cholesky decompositions
  if use_qlb
    D = BlkDiagHessian(A, n_blk; alpha=1, beta=lambda, factor=false)
    lambda_max = estimate_dominant_eigval(GramPlusDiag(linmap, 1, 0), D, maxiter=3)#tol = tol_powm)
    D = update_factors!(D, 1, lambda + lambda_max)
  else
    D = BlkDiagHessian(A, n_blk; alpha=n_blk, beta=lambda, factor=true)
  end

  # Initialize the difference, d₁ = x₁ - x₀
  r = copy(b)
  mul!(r, A, x, -one(T), one(T))  # r = b - A⋅x
  mul!(g, transpose(A), r)        # -∇ = Aᵀ⋅r - λx
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
      @views diag[k] = dot(A[:, k], A[:, k]) + lambda
    end
  else
    diag = AtA[diagind(AtA)]
    @. diag += lambda 
  end
  D = Diagonal(diag)

  if use_qlb
    lambda_max = estimate_dominant_eigval(GramPlusDiag(linmap, 1, 0), D, maxiter=3)#tol = tol_powm)
    @. D.diag += lambda_max
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

function solve_OLS(A::Matrix{T}, b::Vector{T}, x0::Vector{T}, n_blk::Int;
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

function _solve_OLS_blkdiag(A::Matrix{T}, b::Vector{T}, x0::Vector{T}, n_blk::Int;
  lambda::Float64 = 0.0,
  maxiter::Int = 100,
  gtol::Float64 = 1e-3,
  use_qlb::Bool = false,
  tol_powm::Float64 = T(minimum(size(A)))
) where T
  #
  n_obs, n_var = size(A)

  # Main matrices and vectors
  x = deepcopy(x0)
  d = zeros(n_var)
  g = zeros(n_var)
  tmp = zeros(n_var)
  AtApI = GramPlusDiag(A; alpha=one(T), beta=T(lambda))
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

function _solve_OLS_diag(A::Matrix{T}, b::Vector{T}, x0::Vector{T}, n_blk::Int;
  lambda::Float64 = 0.0,
  maxiter::Int = 100,
  gtol::Float64 = 1e-3,
  use_qlb::Bool = false,
  tol_powm::Float64 = T(minimum(size(A)))
) where T
  #
  n_obs, n_var = size(A)

  # Main matrices and vectors
  x = deepcopy(x0)
  d = zeros(n_var)
  g = zeros(n_var)
  tmp = zeros(n_var)
  AtApI = GramPlusDiag(A; alpha=one(T), beta=T(lambda))
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

export GramPlusDiag, BlkDiagHessian, GramMinusBlkDiag
export solve_OLS, solve_OLS_lsmr

end # module
