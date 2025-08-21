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

function solve_QREG_conquer(_X, y, q, h, tol)
  result = rcall(:conquer, X=_X, Y=y, tau=q, h=h, kernel="uniform", tol=tol) # default tol = 1e-04
  β̂ = rcopy(result[1])
  iter = rcopy(result[2])
  r = rcopy(result[3])
  return β̂, r, iter
end

function main(n, p, q, seed, ρ)
  @assert n >= p

  N = 1000      # number of @benchmark samples
  Random.seed!(seed)
  tscale = 1e-9 # report time in seconds
  rtol = 1e-8   # change in objective, relative to previous value
  gtol = 1e-5

  results = DataFrame(
    ρ=Float64[],
    q=Float64[],
    method=String[],
    time=Float64[],
    iter=Int[],
    xnorm=Float64[],
    rnorm=Float64[],
    objv1=Float64[], # check function
    objv2=Float64[], # uniform kernel
  )

  # create problem instance
  if iszero(ρ)
    Σ = Matrix{Float64}(I, p, p)
  else
    Σ = [ρ^abs(i-j) for i in 1:p, j in 1:p];
  end
  cholΣ = cholesky!(Symmetric(Σ))
  _X = randn(n, p) * cholΣ.L
  β = [0.1*ones(p); 1.0]
  X = [_X ones(n)]
  y0 =  X * β
  y = y0 + rand(TDist(1.5), n) .- Statistics.quantile(TDist(1.5), q)
  h = QBSolvers.default_bandwidth(_X)

  println("seed:    ", seed)
  println("size(X): ", size(_X, 1), " × ", size(_X, 2))
  println("cond(X): ", cond(_X))
  println("ρ:       ", ρ)
  println("q:       ", q)
  println("h:       ", h)
  println()

  record! = let results=results, ρ=ρ, q=q, tscale=tscale
    function(alg, bench, stats)
      median_time = median(bench.times) * tscale
      push!(results,
        (
          ρ, q, alg,
          median_time, stats.iterations,
          stats.xnorm, stats.rnorm,
          stats.loss1, stats.loss2
        )
      )
      println("$(lpad(alg, 8)) .......... $(round(median_time, digits=6)) seconds")
    end
  end

  # MMDeweighting
  β̂, _, iter, _ = MMDeweighting.FastQR(X, y, q; tol=rtol, h=h, verbose=false)
  benchMMD = @benchmark MMDeweighting.FastQR($X, $y, $q; tol=$rtol, h=$h, verbose=false) samples=N
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
  β̂, r, iter = solve_QREG_conquer(_X, y, q, h, gtol)
  benchCQR = @benchmark solve_QREG_conquer($_X, $y, $q, $h, $gtol) samples=N
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
  QUB(X, y, 1, :std, "QUBd2")     # DOUBLE-LOOP + DIAG + RANK-1
  QUB(X, y, 1, :corr, "QUBd3")    # DOUBLE-LOOP + DIAG + RANK-1
  QUB(X, y, 1, :deflate, "QUBd4") # DOUBLE-LOOP + DIAG + RANK-1
  # QUB(X, y, 1, :rescale, "QUBd5") # DOUBLE-LOOP + RESCALE
  QUB(X, y, 2, :none, "QUBs1")    # LBFGS + NO NORMALIZATION
  QUB(X, y, 2, :std, "QUBs2")     # LBFGS + DIAG + RANK-1
  QUB(X, y, 2, :corr, "QUBs3")    # LBFGS + DIAG + RANK-1
  QUB(X, y, 2, :deflate, "QUBs4") # LBFGS + DIAG + RANK-1
  # QUB(X, y, 2, :rescale, "QUBs5") # LBFGS + RESCALE

  fmt_time = ft_printf("%5.2f", findfirst(==("time"), names(results)))
  fmt_norm = ft_latex_sn(6, findfirst(==("xnorm"), names(results)) .+ (0:2))

  # for human readability
  println()
  pretty_table(
    results;
    formatters = fmt_time,
    header = [
      "ρ", "q",
      "method", "time (s)", "iterations",
      "xnorm", "rnorm", "objv1", "objv2",
    ]
  )

  # for manuscript
  println()
  pretty_table(
    results;
    backend = Val(:latex),
    formatters = (fmt_time, fmt_norm),
    tf = tf_latex_booktabs,
    header = [
      latex_cell"$\rho$", latex_cell"$q$",
      "method", latex_cell"time (s)", "iterations",
      latex_cell"$\|x\|$", latex_cell"$\|r\|$", latex_cell"$f_{q}(x)$", latex_cell"$h_{q}(x)$",
    ]
  )

  return nothing
end

n         = parse(Int, ARGS[1])
p         = parse(Int, ARGS[2])
q         = parse(Float64, ARGS[3])
seed      = parse(Int, ARGS[4])
ρ         = parse(Float64, ARGS[5])
main(n, p, q, seed, ρ)
