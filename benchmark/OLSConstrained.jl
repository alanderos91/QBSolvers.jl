#
# Load Packages
#
using Pkg, InteractiveUtils

if "MKL" in keys(Pkg.project().dependencies)
  using MKL
end

Pkg.activate(pwd())
Pkg.instantiate()

using QBSolvers
using LinearAlgebra, StatsBase, Random, LowRankApprox
using IterativeSolvers, NonNegLeastSquares, GLMNet
using BenchmarkTools, DataFrames, PrettyTables
#
# Computing Environment
#
BLAS.set_num_threads(10)
QBSolvers.BLAS_THREADS[] = BLAS.get_num_threads()
Pkg.status(); println()
versioninfo(); println()
BLAS.get_config() |> display; println()
#
# Projections + Helper Functions
#
"""Standardizes the columns of X."""
function StandardizeColumns!(X::Matrix)
  (n, p) = size(X)
  for j = 1:p
    avg = sum(@view X[:, j]) / n
    @. @views X[:, j] = X[:, j] - avg
    stdev = norm(@view X[:, j])
    @. @views X[:, j] = X[:, j] / stdev
  end
end

"""Projects the point y onto itself."""
function IdentityProjection(y::Vector{T}, r) where T <: Real
  return y
end

"""Projects the point y onto the nonengative orthant."""
function OrthantProjection(y::Vector{T}, r) where T <: Real
  return max.(y, zero(T))
end

"""Projects the point y onto the simplex {x | x >= 0, sum(x) = r}."""
function SimplexProjection(y::Vector{T}, r = 1) where T <: Real
#
  n = length(y)
  z = sort(y, rev = true)
  (s, lambda) = (zero(T), zero(T))
  for i = 1:n
    s = s + z[i]
    lambda = (s - r) / i
    if i < n && lambda < z[i] && lambda >= z[i + 1]
      break
    end
  end
  return max.(y .- lambda, zero(T))
end

"""Projects the point y onto the set of vectors with at most k
nonzero entries. Note the arbitrary sparsity default of k = 2."""
function SparseProjection(y::Vector{T}, r) where T <: Real
#
  n = length(y)
  x = copy(y)
  p = partialsortperm(y, by = abs, 1:(n - r))
  for i = 1:(n - r)
    x[p[i]] = zero(T)
  end
  return x
end
#
# Algorithms
#
"""Minimizes the least squares loss with response y and design matrix
X subject to a set constraint. The columns of X should be standardized."""
function FastLeastSquares(Projection::Function, X::Matrix{T}, y::Vector{T}, 
  tol::T, r = 1) where T <: Real
#
  ((n, p), maxiters) = (size(X), 1000)
  XtX = Transpose(X) * X
  Xty = Transpose(X) * y
  (beta, old_beta) = (zeros(T, p), zeros(T, p)) # initialize regression coefficients
  (delta, grad) = (zeros(T, p), zeros(T, p))    # workspace
  rho = QBSolvers.estimate_spectral_radius(XtX, Diagonal(XtX); maxiter=4)
  for iter = 1:maxiters
    mul!(grad, XtX, beta)
    @. grad = grad - Xty                # gradient of loss
    # mul!(delta, XtX, grad)
    # t = norm(grad)^2 / dot(grad, delta) # optimal step length
    t = one(T)
    @. grad = beta - t*inv(1+rho)*grad  # iterate before projection
    beta .= Projection(grad, r)         # update regression coefficients
    @. delta = beta - old_beta
    if norm(delta) < tol * norm(old_beta) # convergence test
      return (iter, beta)
    else
      @. old_beta = beta 
    end
  end
  return (maxiters, beta)
end

"""Minimizes the least squares loss with response y and design matrix
X subject to a nonnegativity constraint using coordinate descent."""
function CD(X::Matrix{T}, y::Vector{T}, tol::T) where T <: Real
#
  ((n, p), maxiters) = (size(X), 1000)
  XtX = Transpose(X) * X
  Xty = Transpose(X) * y
  (beta, old_beta) = (zeros(T, p), zeros(T, p)) # initialize regression coefficients
  delta = zeros(T, p)
   for iter = 0:maxiters
#     println(iter," ",norm(y - X * beta)^2)
    @views for j = 1:p
      increment = (Xty[j] - dot(XtX[:, j], old_beta)) / XtX[j, j]
      beta[j] = max(old_beta[j] + increment, zero(T))
    end
    @. delta = beta - old_beta
    if norm(delta) < tol * norm(old_beta)
      return (iter, beta)
    else
      old_beta .= beta
    end
  end
  return (maxiters, beta)
end

"""Minimizes the least squares loss with response y and design matrix
X subject to a set constraint. The columns of X should be standardized."""
function PGD(Projection::Function, X::Matrix{T}, y::Vector{T}, 
  tol::T, r = 1) where T <: Real
#
  ((n, p), maxiters) = (size(X), 1000)
  XtX = Transpose(X) * X
  Xty = Transpose(X) * y
  (beta, old_beta) = (zeros(T, p), zeros(T, p)) # initialize regression coefficients
  (delta, grad) = (zeros(T, p), zeros(T, p))    # workspace
  for iter = 1:maxiters
    mul!(grad, XtX, beta)
    @. grad = grad - Xty                # gradient of loss
    mul!(delta, XtX, grad)
    t = norm(grad)^2 / dot(grad, delta) # optimal step length
    @. grad = beta - t*grad             # iterate before projection
    beta .= Projection(grad, r)         # update regression coefficients
    @. delta = beta - old_beta
    if norm(delta) < tol * norm(old_beta) # convergence test
      return (iter, beta)
    else
      @. old_beta = beta 
    end
  end
  return (maxiters, beta)
end
#
# Wrappers
#
function OLS_QR(X, y)
  beta = X \ y
  return (-1, beta)
end

function OLS_LSMR(X, y)
  (beta, ch) = lsmr(X, y, log = true)
  return (ch.iters, beta)
end

function OLS_LSQR(X, y)
  (beta, ch) = lsqr(X, y, log = true)
  return (ch.iters, beta)
end

function NNLSQ(X, y)
  beta = nonneg_lsq(X, y; alg=:pivot, variant=:comb)
  return (-1, beta)
end

function NNLSQ_GLMNET(X, y, M)
  sol = glmnet(X, y, lambda = [0.0], constraints = M);
  return (sol.npasses, sol.betas)
end

function SPARSE_GLMNET(X, y)
  sol = glmnet(X, y, lambda = [0.025])
  return (sol.npasses, sol.betas)
end
#
# Benchmark Script
#
function main()
  #
  # Simulate data for problem instances.
  #
  Random.seed!(1234)
  (n, p, r, tol) = (10000, 2000, 1, 1e-6); # cases, parameters, and tolerance
  X = randn(n, p); # design matrix
  StandardizeColumns!(X);
  X = [ones(n) ./ sqrt(n) X]; # include intercept
  p = p + 1;
  beta = randn(p); # regression coefficients
  y = X * beta + randn(n); # responses with noise
  #
  # Closure for DQUB + CD code to fix parameters
  #
  FASTLS = let tol=tol, r=r
    function(P, X, y)
      FastLeastSquares(P, X, y, tol, r)
    end
  end
  FASTCD = let tol=tol
    function (X, y)
      CD(X, y, tol)
    end
  end
  #
  # Initialize table for collecting and presenting benchmark results.
  #
  results = DataFrame(
    Regression=String[],
    Method=String[],
    Time=Float64[],
    Iters=Any[],
    Resid=Float64[],
  )
  record_result! = let df=results, X=X, y=y
    # This helper records median times in seconds from a BenchmarkTrial.
    function(regtype, algorithm, b, iter, beta)
      push!(df, (regtype, algorithm, median(b.times)*1e-9, (iter == -1) ? "-" : iter, norm(y - X*beta)))
    end
  end
  #
  # Test Case 1: Ordinary Regression
  #
  let X=X, y=y, FASTLS=FASTLS
    regtype = "Ordinary"
    DQUB(X, y) = FASTLS(IdentityProjection, X, y)
    algorithms = (
      ("QR", OLS_QR),
      ("LSQR", OLS_LSQR),
      ("LSMR", OLS_LSMR),
      ("DQUB", DQUB),
    )
    for (alg, f) in algorithms
      iter, β = f(X, y)         # obtain solution
      b = @benchmark $f($X, $y) # time it
      record_result!(regtype, alg, b, iter, β)
    end
  end
  #
  # Test Case 2: Nonnegative Regression
  #
  let X=X, y=y, FASTLS=FASTLS, FASTCD=FASTCD, p=p
    regtype = "Nonnegative"
    M = reshape(repeat([0.0 Inf], p), (2, p)) # constraint matrix required by glmnet
    DQUB(X, y) = FASTLS(OrthantProjection, X, y)
    GLMNET = let M=M
      function(X, y)
        NNLSQ_GLMNET(X, y, M)
      end
    end
    algorithms = (
      ("CD", FASTCD),
      ("NNLS", NNLSQ),
      ("GLMNET", GLMNET),
      ("DQUB", DQUB),
    )
    for (alg, f) in algorithms
      iter, β = f(X, y)         # obtain solution
      b = @benchmark $f($X, $y) # time it
      record_result!(regtype, alg, b, iter, β)
    end
  end
  #
  # Test Case 3: Simplex
  #
  let X=X, y=y, FASTLS=FASTLS, tol=tol, r=r
    regtype = "Simplex"
    DQUB(X, y) = FASTLS(SimplexProjection, X, y)
    SIMPLEX_PGD(X, y) = let tol=tol, r=r
      PGD(SimplexProjection, X, y, tol, r)
    end
    algorithms = (
      ("PGD", SIMPLEX_PGD),
      ("DQUB", DQUB),
    )
    for (alg, f) in algorithms
      iter, β = f(X, y)         # obtain solution
      b = @benchmark $f($X, $y) # time it
      record_result!(regtype, alg, b, iter, β)
    end
  end
  #
  # Test Case 4: 
  #
  let X=X, y=y, tol=tol
    #
    # First we need to calibrate sparsity to that of glmnet.
    #
    _, beta = SPARSE_GLMNET(X, y)
    r = count(x -> abs(x) > 1e-6, beta)
    println("glmnet solution found to have $(r) nonzero elements")
    #
    # Proceed with the benchmarks
    #
    regtype = "Sparse"
    DQUB = let tol=tol, r=r
      function(X, y)
        FastLeastSquares(SparseProjection, X, y, tol, r)
      end
    end
    algorithms = (
      ("GLMNET", SPARSE_GLMNET),
      ("DQUB", DQUB),
    )
    for (alg, f) in algorithms
      iter, β = f(X, y)         # obtain solution
      b = @benchmark $f($X, $y) # time it
      record_result!(regtype, alg, b, iter, β)
    end
  end

  println("PrettyTable")
  pretty_table(
    results;
    header = names(results),
    formatters = (
      ft_printf("%.4f", [3]),
      ft_printf("%.5f", [5]),
    )
  )

  latex_header = [names(results)[1:4]; latex_cell"$\|\by-\bX\bbeta\|$"]
  println("LaTeX")
  pretty_table(
    results;
    backend = Val(:latex),
    tf = tf_latex_booktabs,
    header = latex_header,
    formatters = (
      ft_printf("%.4f", [3]),
      ft_printf("%.5f", [5]),
    )
  )

end
#
# Runtime
#
main()

