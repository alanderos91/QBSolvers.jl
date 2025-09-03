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

function main(n, p, q, seed, ρ; standardize=false, intercept=false, accel=false)
  @assert n >= p

  N = 100      # number of @benchmark samples
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
  getprob = let n=n, p=p, ρ=ρ, standardize=standardize
    function(rng)
      if iszero(ρ)
        Σ = Matrix{Float64}(I, p, p)
      else
        Σ = [ρ^abs(i-j) for i in 1:p, j in 1:p];
      end
      cholΣ = cholesky!(Symmetric(Σ))
      _X = randn(rng, n, p) * cholΣ.U
      _β = 0.1*ones(p)
      if standardize
        X_mean = vec(mean(_X, dims = 1))
        X_std  = vec(std(_X,  dims = 1))
        @inbounds for j in 1:p
            μ = X_mean[j]
            σj = X_std[j]
            σj = (σj == 0.0) ? 1.0 : σj
            @views _X[:, j] .-= μ
            @views _X[:, j] ./= σj
        end
      end
      if intercept
        β = [_β; 1.0]
        X = [_X ones(n)]
      else
        β = _β
        X = _X
      end
      y = X*β + rand(rng, TDist(1.5), n) .- Statistics.quantile(TDist(1.5), q)
      return (y, _X, X)
    end
  end
  y, _X, X = getprob(Xoshiro(seed))
  h = QBSolvers.default_bandwidth(_X)

  println("seed:    ", seed)
  println("size(X): ", size(X, 1), " × ", size(_X, 2), ifelse(intercept, "+1", ""))
  println("cond(X): ", cond(X))
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
  benchMMD = @benchmark MMDeweighting.FastQR(prob[3], prob[1], $q; tol=$rtol, h=$h, verbose=false) samples=N setup=(rng = Xoshiro($seed); prob = $getprob(rng);)
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
  benchCQR = @benchmark solve_QREG_conquer($_X, $y, $q, $h, $gtol) samples=N setup=(rng = Xoshiro($seed); prob = $getprob(rng);)
  statsCQR = (;
    iterations=iter,
    xnorm=norm(β̂),
    rnorm=norm(r),
    loss1=PLS.qreg_objective(r, q),
    loss2=PLS.qreg_objective_uniform(r, q, h)
  )
  record!("conquer", benchCQR, statsCQR)

  # QUB closures
  QUB = let N=N, q=q, h=h, rtol=rtol, accel=accel, seed=seed
    function(X, y, version, normalize, alg)
      _, _, stats = solve_QREG_lbfgs(X, y; q=q, h=h, rtol=rtol, version=version, normalize=normalize, accel=accel, memory=20)
      bench = @benchmark solve_QREG_lbfgs($X, $y; q=$q, h=$h, rtol=$rtol, version=$version, normalize=$normalize, accel=$accel, memory=20) samples=N setup=(rng = Xoshiro($seed); prob = $getprob(rng);)
      record!(alg, bench, stats)
    end
  end

  QUB(X, y, 1, :none,    "QUBd0") # DOUBLE-LOOP
  QUB(X, y, 1, :deflate, "QUBd1") # DOUBLE-LOOP + DEFLATE
  QUB(X, y, 2, :none,    "QUBs0") # LBFGS
  QUB(X, y, 2, :deflate, "QUBs1") # LBFGS + DEFLATE
  # QUB(X, y, 3, :none,    "QUBt0") # LBFGS
  # QUB(X, y, 3, :deflate, "QUBt1") # LBFGS + DEFLATE

  if isinteractive()
    pretty_table(results)
  else
    fmt_time = ft_printf("%5.4f", findfirst(==("time"), names(results)))
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
  end

  return nothing
end

if !isinteractive()
  n         = parse(Int, ARGS[1])
  p         = parse(Int, ARGS[2])
  q         = parse(Float64, ARGS[3])
  seed      = parse(Int, ARGS[4])
  ρ         = parse(Float64, ARGS[5])
  main(n, p, q, seed, ρ; standardize=true, intercept=true, accel=true)
end

