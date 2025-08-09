###
### usage:
###
### julia -t 1 QREG.jl 8192 2047 0.5 1903 0.2
###

using Pkg, InteractiveUtils

if "MKL" in keys(Pkg.project().dependencies)
  using MKL
end

Pkg.activate(pwd())
Pkg.instantiate()

using QBSolvers
using LinearAlgebra, Statistics, Random, Distributions
using BenchmarkTools, DataFrames, PrettyTables
import MMDeweighting
using RCall

PLS = QBSolvers

BLAS.set_num_threads(10)
QBSolvers.BLAS_THREADS[] = BLAS.get_num_threads()

Pkg.status(); println()
versioninfo(); println()
BLAS.get_config() |> display; println()

R"""
library("conquer")
sessionInfo()
""" |> println

function solve_QREG_conquer(_X, y, q, h)
  result = rcall(:conquer, X=_X, Y=y, tau=q, h=h, kernel="uniform") # default tol = 1e-04
  β̂ = rcopy(result[1])
  iter = rcopy(result[2])
  r = rcopy(result[3])
  return β̂, r, iter
end

function main(n, p, q, seed, rho)
  @assert n >= p

  N = 1000      # number of @benchmark samples
  Random.seed!(seed)
  tscale = 1e-6 # report time in milliseconds
  rtol = 1e-8   # change in objective, relative to previous value

  results = DataFrame(
    n=Int[],
    p=Int[],
    q=Float64[],
    h=Float64[],
    method=String[],
    time=Float64[],
    iter=Int[],
    xnorm=Float64[],
    rnorm=Float64[],
    objv1=Float64[], # check function
    objv2=Float64[], # uniform kernel
  )

  # create problem instance
  if iszero(rho)
    Σ = Matrix{Float64}(I, p, p)
  else
    Σ = [rho^abs(i-j) for i in 1:p, j in 1:p];
  end
  cholΣ = cholesky!(Symmetric(Σ))
  X = randn(n, p) * cholΣ.L
  β = 0.1*ones(p)
  y0 =  X * β
  y = y0 + rand(TDist(1.5), n) .- Statistics.quantile(TDist(1.5), q)
  println("Condition number of X: ", cond(X))

  # Set bandwidth for both methods
  h = QBSolvers.default_bandwidth(X)

  record! = let results=results, n=n, p=p, q=q, h=h, tscale=tscale
    function(alg, bench, stats)
      push!(results,
        (n, p, q, h, alg,
          median(bench.times) * tscale, stats.iterations,
          stats.xnorm, stats.rnorm,
          stats.loss1, stats.loss2
        )
      )
    end
  end

  # MMDeweighting
  β̂, _, iter, _ = MMDeweighting.FastQR(X, y, q; tol=rtol, h=h, verbose=false)
  benchMMD = @benchmark MMDeweighting.FastQR($X, $y, q; tol=$rtol, h=$h, verbose=false) samples=N
  r = y - X*β̂
  statsMMD = (;
    iterations=iter,
    xnorm=norm(β̂),
    rnorm=norm(r),
    loss1=PLS.qreg_objective(r, q),
    loss2=PLS.qreg_objective_uniform(r, q, h)
  )
  record!("MMDW", benchMMD, statsMMD)

  # conquer
  β̂, r, iter = solve_QREG_conquer(X, y, q, h)
  benchCQR = @benchmark solve_QREG_conquer($X, $y, $q, $h) samples=N
  statsCQR = (;
    iterations=iter,
    xnorm=norm(β̂),
    rnorm=norm(r),
    loss1=PLS.qreg_objective(r, q),
    loss2=PLS.qreg_objective_uniform(r, q, h)
  )
  record!("conquer", benchCQR, statsCQR)

  # QUB closures
  QUB = let N=N, q=q, h=h, rtol=rtol
    function(X, y, version, normalize, alg)
      _, _, stats = solve_QREG_lbfgs(X, y; q=q, h=h, rtol=rtol, version=version, normalize=normalize, accel=true)
      bench = @benchmark solve_QREG_lbfgs($X, $y; q=$q, h=$h, rtol=$rtol, version=$version, normalize=$normalize, accel=true) samples=N
      record!(alg, bench, stats)
    end
  end

  QUB(X, y, 1, :none, "QUBd1")    # DOUBLE-LOOP + NO NORMALIZATION
  QUB(X, y, 1, :qub, "QUBd2")     # DOUBLE-LOOP + DIAG + RANK-1
  QUB(X, y, 1, :rescale, "QUBd3") # DOUBLE-LOOP + RESCALE
  QUB(X, y, 2, :none, "QUBs1")    # LBFGS + NO NORMALIZATION
  QUB(X, y, 2, :qub, "QUBs2")     # LBFGS + DIAG + RANK-1
  QUB(X, y, 2, :rescale, "QUBs3") # LBFGS + RESCALE

  fmt_time = ft_printf("%5.2f", findfirst(==("time"), names(results)))
  fmt_norm = ft_latex_sn(4, findfirst(==("xnorm"), names(results)) .+ (0:2))

  # for human readability
  pretty_table(
    results;
    formatters = fmt_time,
    header = [
      "samples", "variables", "q", "bandwidth",
      "method", "time (ms)", "iterations", "xnorm", "rnorm", "objv1", "objv2",
    ]
  )

  # for manuscript
  pretty_table(
    results;
    backend = Val(:latex),
    formatters = (fmt_time, fmt_norm),
    tf = tf_latex_booktabs,
    header = [
      "samples", "variables", latex_cell"$q$", latex_cell"$\mu$",
      "method", latex_cell"time (ms)", "iterations", latex_cell"$\|x\|$", latex_cell"$\|r\|$", "objv1", "objv2",
    ]
  )

  return nothing
end

n         = parse(Int, ARGS[1])
p         = parse(Int, ARGS[2])
q         = parse(Float64, ARGS[3])
seed      = parse(Int, ARGS[4])
rho       = parse(Float64, ARGS[5])
main(n, p, q, seed, rho)
