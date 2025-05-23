###
### usage:
###
### julia -t 1 QREG.jl 8192 2047 0.5 1903 > QREG_8192x2048_1903_q=0.5.out
###
### NOTE: Number of variables, p, should satisfy p = 2^k-1 for some positive integer k.
###

using Pkg, InteractiveUtils

if "MKL" in keys(Pkg.project().dependencies)
  using MKL
end

Pkg.activate(pwd())
Pkg.instantiate()

using ParallelLeastSquares
using LinearAlgebra, Statistics, Random, Distributions
using BenchmarkTools, DataFrames, PrettyTables
import MMDeweighting
using RCall

PLS = ParallelLeastSquares

BLAS.set_num_threads(10)
ParallelLeastSquares.BLAS_THREADS[] = BLAS.get_num_threads()

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

function main(n, p, q, seed)
  @assert n >= p+1
  @assert rem(p+1, 2) == 0

  N = 1000      # number of @benchmark samples
  Random.seed!(seed)
  maxiter = 10^4
  tscale = 1e-6 # report time in milliseconds
  tol = 1e-6    # change in objective, relative to previous value

  results = DataFrame(
    threads=Int[],
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
    gnorm=Float64[], # from surrogate
    objv1=Float64[], # check function
    objv2=Float64[], # uniform kernel
  )

  ρ = 0.7
  Σ = [ρ^abs(i-j) for i in 1:p, j in 1:p]
  _X = Transpose(rand(MvNormal(zeros(p), Σ), n)) |> Matrix
  β = 0.1*ones(p)
  y0 =  _X * β .+ 1
  y = y0 + rand(TDist(1.5), n) .- Statistics.quantile(TDist(1.5), q)
  X = [ones(n) _X]

  # Set bandwidth for both methods
  h = ParallelLeastSquares.default_bandwidth(X)

  # MMDeweighting
  β̂, _, iter, _ = MMDeweighting.FastQR(X, y, q; tol=tol, h=h, verbose=false)
  benchMMD = @benchmark MMDeweighting.FastQR($X, $y, q; tol=$tol, h=$h, verbose=false) samples=N

  # -∇₀ = Aᵀ[r - z .+ (2q-1)h]
  r = y - X*β̂
  z = PLS.prox_abs!(similar(r), r, h)
  @. z = r - z + (2*q-1)*h
  grad = inv(2*h) * transpose(X) * z

  push!(results,
    (Threads.nthreads(), n, p, q, h, 1, p+1, "MMDeweighting",
      median(benchMMD.times) * tscale, iter,
      norm(β̂), norm(r), norm(grad),
      PLS.qreg_objective(r, q),
      PLS.qreg_objective_uniform(r, q, h),
    )
  )

  gtol = norm(grad) # match on gradient of surrogate
  β0 = zeros(size(X, 2))

  # conquer
  β̂, r, iter = solve_QREG_conquer(_X, y, q, h)
  benchCQR = @benchmark solve_QREG_conquer($_X, $y, $q, $h) samples=N

  # -∇₀ = Aᵀ[r - z .+ (2q-1)h]
  z = PLS.prox_abs!(similar(r), r, h)
  @. z = r - z + (2*q-1)*h
  grad = inv(2*h) * transpose(X) * z

  push!(results,
    (Threads.nthreads(), n, p, q, h, 1, p+1, "conquer",
      median(benchCQR.times) * tscale, iter,
      norm(β̂), norm(r), norm(grad),
      PLS.qreg_objective(r, q),
      PLS.qreg_objective_uniform(r, q, h),
    )
  )

  for var_per_blk in (2^k for k in 0:Int(log2(p+1)))
    n_blk = fld(p+1, var_per_blk)
    β̂, _, stats = solve_QREG(X, y, β0, n_blk;
      q=q, h=h, maxiter=maxiter, gtol=gtol)
    r = y - X*β̂

    benchMM = @benchmark solve_QREG($X, $y, $β0, $n_blk;
      q=$q, h=$h, maxiter=$maxiter, gtol=$gtol) samples=N

    push!(results,
      (Threads.nthreads(), n, p, q, h, n_blk, var_per_blk, "MM-QLB",
        median(benchMM.times) * tscale, stats.iterations,
        stats.xnorm, stats.rnorm, stats.gnorm,
        PLS.qreg_objective(r, q),
        PLS.qreg_objective_uniform(r, q, h),
      )
    )
  end

  fmt_time = ft_printf("%5.2f", findfirst(==("time"), names(results)))
  fmt_norm = ft_latex_sn(4, findfirst(==("xnorm"), names(results)) .+ (0:2))

  # for human readability
  pretty_table(
    results;
    formatters = fmt_time,
    header = [
      "threads", "samples", "variables", "q", "bandwidth", "blocks", "block size",
      "method", "time (us)", "iterations", "xnorm", "rnorm", "gnorm", "objv1", "objv2",
    ]
  )

  # for manuscript
  pretty_table(
    results;
    backend = Val(:latex),
    formatters = (fmt_time, fmt_norm),
    tf = tf_latex_booktabs,
    header = [
      "threads", "samples", "variables", latex_cell"$q$", latex_cell"$h$", "blocks", "block size",
      "method", latex_cell"time ($\mu$s)", "iterations", latex_cell"$\|x\|$", latex_cell"$\|r\|$", latex_cell"$\|g\|$", "objv1", "objv2",
    ]
  )

  return nothing
end

n = parse(Int, ARGS[1])
p = parse(Int, ARGS[2])
q = parse(Float64, ARGS[3])
seed = parse(Int, ARGS[4])
main(n, p, q, seed)
