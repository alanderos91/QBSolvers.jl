###
### usage:
###
### julia -t 1 QREG.jl 8192 2047 0.5 1903 blkdiag false 16
### NOTE: Number of variables, p, should satisfy p = 2^k-1 for some positive integer k.
###

using Pkg, InteractiveUtils

if "MKL" in keys(Pkg.project().dependencies)
  using MKL
end

Pkg.activate(pwd())
Pkg.instantiate()
include("utilities.jl")

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

function main(n, p, q, seed, corrtype, use_noise, ngroups)
  @assert n >= p+1
  @assert rem(p+1, 2) == 0

  N = 1000      # number of @benchmark samples
  Random.seed!(seed)
  maxiter = 2*10^3
  tscale = 1e-6 # report time in milliseconds
  rtol = 1e-6   # change in objective, relative to previous value
  gtol = 1e-2   # gradient tolerance for inner solve in QUB

  results = DataFrame(
    n=Int[],
    p=Int[],
    q=Float64[],
    h=Float64[],
    blocks=Int[],
    blocksize=Int[],
    method=String[],
    time=Float64[],
    iter=Int[],
    xnorm=Float64[],
    rnorm=Float64[],
    objv1=Float64[], # check function
    objv2=Float64[], # uniform kernel
  )

  Σ = make_Σ(corrtype, p, use_noise, ngroups)
  _X = Transpose(rand(MvNormal(zeros(p), Σ), n)) |> Matrix
  β = 0.1*ones(p)
  y0 =  _X * β .+ 1
  y = y0 + rand(TDist(1.5), n) .- Statistics.quantile(TDist(1.5), q)
  X = [ones(n) _X]
  println("Condition number of X: ", cond(X))
  println("Number of blocks in Σ: ", corrtype == "blkdiag" || corrtype == "blkband" ? ngroups : 1)
  println()

  # Set bandwidth for both methods
  h = QBSolvers.default_bandwidth(X)

  # MMDeweighting
  β̂, _, iter, _ = MMDeweighting.FastQR(X, y, q; tol=rtol, h=h, verbose=false)
  benchMMD = @benchmark MMDeweighting.FastQR($X, $y, q; tol=$rtol, h=$h, verbose=false) samples=N

  r = y - X*β̂
  push!(results,
    (n, p, q, h, 1, p+1, "MMDeweighting",
      median(benchMMD.times) * tscale, iter,
      norm(β̂), norm(r),
      PLS.qreg_objective(r, q),
      PLS.qreg_objective_uniform(r, q, h),
    )
  )

  # conquer
  β̂, r, iter = solve_QREG_conquer(_X, y, q, h)
  benchCQR = @benchmark solve_QREG_conquer($_X, $y, $q, $h) samples=N

  push!(results,
    (n, p, q, h, 1, p+1, "conquer",
      median(benchCQR.times) * tscale, iter,
      norm(β̂), norm(r),
      PLS.qreg_objective(r, q),
      PLS.qreg_objective_uniform(r, q, h),
    )
  )

  β0 = zeros(size(X, 2))
  for var_per_blk in (2^k for k in 0:Int(log2(p+1)))
    n_blk = fld(p+1, var_per_blk)
    for normalize in (false,)# true)
      β̂, r, stats = solve_QREG(X, y, β0, n_blk;
        q=q, h=h, maxiter=maxiter, gtol=gtol, rtol=rtol)

      benchMM = @benchmark solve_QREG($X, $y, $β0, $n_blk;
        q=$q, h=$h, maxiter=$maxiter, gtol=$gtol, rtol=$rtol) samples=N

      push!(results,
        (n, p, q, h, n_blk, var_per_blk, normalize ? "QUBn" : "QUB",
          median(benchMM.times) * tscale, stats.iterations,
          stats.xnorm, stats.rnorm,
          PLS.qreg_objective(r, q),
          PLS.qreg_objective_uniform(r, q, h),
        )
      )
    end
  end

  fmt_time = ft_printf("%5.2f", findfirst(==("time"), names(results)))
  fmt_norm = ft_latex_sn(4, findfirst(==("xnorm"), names(results)) .+ (0:2))

  # for human readability
  pretty_table(
    results;
    formatters = fmt_time,
    header = [
      "samples", "variables", "q", "bandwidth", "blocks", "block size",
      "method", "time (us)", "iterations", "xnorm", "rnorm", "objv1", "objv2",
    ]
  )

  # for manuscript
  pretty_table(
    results;
    backend = Val(:latex),
    formatters = (fmt_time, fmt_norm),
    tf = tf_latex_booktabs,
    header = [
      "samples", "variables", latex_cell"$q$", latex_cell"$h$", "blocks", "block size",
      "method", latex_cell"time ($\mu$s)", "iterations", latex_cell"$\|x\|$", latex_cell"$\|r\|$", "objv1", "objv2",
    ]
  )

  return nothing
end

n         = parse(Int, ARGS[1])
p         = parse(Int, ARGS[2])
q         = parse(Float64, ARGS[3])
seed      = parse(Int, ARGS[4])
corrtype  = ARGS[5]
use_noise = parse(Bool, ARGS[6])
ngroups   = parse(Int, ARGS[7])
main(n, p, q, seed, corrtype, use_noise, ngroups)
