#
# Load Packages
#
using Pkg, InteractiveUtils

if "MKL" in keys(Pkg.project().dependencies)
  using MKL
end

Pkg.activate(pwd())
Pkg.instantiate()

using QBSolvers, NonNegLeastSquares, GoldfarbIdnaniSolver
using NonNegLeastSquares: pivot_cache
using Distributions, LinearAlgebra, Statistics, Random
using BenchmarkTools, CSV, DataFrames, PrettyTables

#
# Computing Environment
#
BLAS.set_num_threads(10)
QBSolvers.BLAS_THREADS[] = BLAS.get_num_threads()
Pkg.status(); println()
versioninfo(); println()
BLAS.get_config() |> display; println()

#
# Helper functions for managing benchmark results
#

# Extract benchmark statistics in milliseconds
extract_ms_stats(b::BenchmarkTools.Trial) = (
  minimum = minimum(b).time / 1e9,
  median  = median(b).time  / 1e9,
  mean    = mean(b).time    / 1e9,
  std     = std(b).time     / 1e9
)

#
# Wrapper for GoldfarbIdnaniSolver.jl 
#
function GISolver(A, q)
  #
  # Construct constraint matrices for x ≥ 0
  #
  C = 1.0*I(size(A, 1))
  b0 = zeros(size(A, 1))
  #
  # Run the solver. Output description from: https://github.com/fabienlefloch/GoldfarbIdnaniSolver.jl/blob/main/src/QP.jl#L83-L92
  #
  # 1. sol   nx1 the final solution (x in the notation above)
  # 2. lagr  qx1 the final Lagrange multipliers
  # 3. crval scalar, the value of the criterion at the minimum      
  # 4. iact  qx1 vector, the constraints which are active in the final
  #       fit (int)
  # 5. nact  scalar, the number of constraints active in the final fit (int)
  # 6. iter  2x1 vector, first component gives the number of "main" 
  #       iterations, the second one says how many constraints were
  #       deleted after they became active
  #
  sol, lagr, crval, iact, nact, iter = solveQP(A, q, C, b0; meq=0, factorized=false)
  #
  # Zero out any coefficients subject to active constraints. The iact vector is padded with 0s.
  #
  iact = sort!(filter(>(0), iact))
  @. sol[iact] = 0
  return sol, iter[1]
end

#
# Benchmark Script
#
function main()
  #
  # Create problem instance
  #
  Random.seed!(123)
  p = 1000
  n = 10p
  
  covM = [0.4^abs(i-j) for i in 1:p, j in 1:p]
  
  d = MvNormal(zeros(p), covM)
  X = Transpose(rand(d, n))
  β = 0.1ones(p)
  d = TDist(1.5)
  truth =  X * β .+ 1
  y = truth + rand(d, n) .- Statistics.quantile(d,0.5)
  X = Matrix(X)
  
  A = X'X
  q = X'y
  q_mat = reshape(q, :, 1)

  maxiter = 1000
  tol = 1e-10
  

  # Loss function: squared residual norm
  loss(β) = norm(X*β .- y)^2
  
  # ---------- Warm-up (JIT compilation) ----------
  # Run each solver once to compile Julia code before benchmarking
  β̂_nqub, niters_nqub = NQUB_nqp_TwoMat(A, q, maxiter = maxiter, 
    ∇tol = tol, nonnegative = true, correlation_eigenvalue = true)
  β̂_fnnls = nonneg_lsq(A, q; gram=true, alg=:fnnls)
  β̂_pivot = pivot_cache(A, q_mat; gram=true, tol=1e-8)
  β̂_nnls  = nonneg_lsq(A, q; gram=true, alg=:nnls)
  β̂_gis, niters_gis = GISolver(A, q)

  # ---------- Benchmarking ----------
  b_nqub  = @benchmark NQUB_nqp_TwoMat($A, $q; maxiter=$maxiter, ∇tol=$tol,
    nonnegative=true, correlation_eigenvalue=true)
  b_fnnls = @benchmark nonneg_lsq($A, $q; gram=true, alg=:fnnls)
  b_pivot = @benchmark pivot_cache($A, $q_mat; gram=true, tol=1e-8)
  b_nnls  = @benchmark nonneg_lsq($A, $q; gram=true, alg=:nnls)
  b_gis   = @benchmark GISolver($A, $q)

  # Store benchmark statistics
  s_nqub  = extract_ms_stats(b_nqub)
  s_fnnls = extract_ms_stats(b_fnnls)
  s_pivot = extract_ms_stats(b_pivot)
  s_nnls  = extract_ms_stats(b_nnls)
  s_gis   = extract_ms_stats(b_gis)
  
  # ---------- Recompute solutions to obtain loss/iterations ----------
  β̂_nqub, niters_nqub = NQUB_nqp_TwoMat(
    A, q; maxiter=maxiter, ∇tol=tol, nonnegative=true, correlation_eigenvalue=true
  )
  L_nqub = loss(β̂_nqub)
  
  β̂_fnnls = nonneg_lsq(A, q; gram=true, alg=:fnnls)
  L_fnnls = loss(β̂_fnnls)
  
  β̂_pivot = pivot_cache(A, q_mat; gram=true, tol=1e-8)
  L_pivot = loss(β̂_pivot)
  
  β̂_nnls = nonneg_lsq(A, q; gram=true, alg=:nnls)
  L_nnls = loss(β̂_nnls)
  
  β̂_gis, niters_gis = GISolver(A, q)
  L_gis = loss(β̂_gis)

  # ---------- Assemble results table ----------
  header = ["Method", "Time(s)", "Iterations", "Nonzeros", "Loss"]
  
  rows = [
    ("QUB",                  s_nqub.mean,  niters_nqub, sum(β̂_nqub  .!= 0),  L_nqub),
    ("NNLS",                 s_nnls.mean,  -,           sum(β̂_nnls  .!= 0), L_nnls),
    ("FNNLS",                s_fnnls.mean, -,           sum(β̂_fnnls .!= 0), L_fnnls),
    ("PIVOT",                s_pivot.mean, -,           sum(β̂_pivot .!= 0), L_pivot),
    ("GoldfarbIdnaniSolver", s_gis.mean,   niters_gis,  sum(β̂_gis   .!= 0), L_gis),
  ]
  
  df = DataFrame([Symbol(h) => [r[i] for r in rows] for (i,h) in enumerate(header)])
  
  # ---------- Pretty table in terminal ----------
  println("\nPretty Table:")
  pretty_table(df;
    header = header,
    formatters = (
      ft_printf("%.3f", [2]),       # Time in ms
      ft_printf("%.0f", [3,4]),     # Iterations, #Params, Nonzeros
      ft_printf("%.6f", [5])        # Loss in scientific notation
    )
  )
  
  # ---------- Pretty table in LaTeX ----------
  io = IOBuffer()
  pretty_table(io, df;
    header = header,
    tf = tf_latex_booktabs,
    formatters = (
      ft_printf("%.3f", [2]),       # Time in ms
      ft_printf("%.0f", [3,4]),     # Iterations, #Params, Nonzeros
      ft_printf("%.6f", [5])        # Loss in scientific notation
    )
  )
  
  println("\nLaTeX Table Output:")
  println(String(take!(io)))

  return
end

#
# Runtime
#
main()

