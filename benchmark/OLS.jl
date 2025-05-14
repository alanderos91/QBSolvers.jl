###
### usage:
###
### julia -t 1 OLS.jl 8192 2048 0 1903 > GQLB_8192x2048_1903.out
###

using Pkg, InteractiveUtils

if "MKL" in keys(Pkg.project().dependencies)
  using MKL
end

Pkg.activate(pwd())
Pkg.instantiate()

using ParallelLeastSquares
using LinearAlgebra, Statistics, Random
using BenchmarkTools, DataFrames, PrettyTables

BLAS.set_num_threads(10)

Pkg.status(); println()
versioninfo(); println()
BLAS.get_config() |> display; println()

function main(n, p, λ, seed)
  @assert n > p
  @assert p > 2^7

  N = 100 # number of @benchmark samples
  Random.seed!(seed)
  maxiter = 10^4
  use_qlb = true

  results = DataFrame(
    threads=Int[],
    n=Int[],
    p=Int[],
    λ=Float64[],
    blocks=Int[],
    blocksize=Int[],
    method=String[],
    time=Float64[],
    iter=Int[],
    xnorm=Float64[],
    rnorm=Float64[],
    gnorm=Float64[],
  )

  A = randn(n, p)
  x = ones(p)
  b = A*x + 1/p .* randn(n)
  x0 = zeros(p)

  xLSMR, rLSMR, statsLSMR = solve_OLS_lsmr(A, b; lambda=λ)
  benchLSMR = @benchmark solve_OLS_lsmr($A, $b; lambda=$λ) samples=N
  push!(results,
    (Threads.nthreads(), n, p, λ, 1, p, "LSMR",
      median(benchLSMR.times) * 1e-6, statsLSMR.iterations,
      statsLSMR.xnorm, statsLSMR.rnorm, statsLSMR.gnorm,
    )
  )
  gnormLSMR = statsLSMR.gnorm

  for var_per_blk in (2^k for k in 0:8)
    n_blk = fld(p, var_per_blk)
    _, _, stats = solve_OLS(A, b, x0, n_blk;
      lambda=λ, maxiter=maxiter, gtol=gnormLSMR, use_qlb=use_qlb)
    benchMM = @benchmark solve_OLS($A, $b, $x0, $n_blk;
      lambda=$λ, maxiter=$maxiter, gtol=$gnormLSMR, use_qlb=$use_qlb) samples=N
    push!(results,
      (Threads.nthreads(), n, p, λ, n_blk, var_per_blk, use_qlb ? "MM-QLB" : "MM",
        median(benchMM.times) * 1e-6, stats.iterations,
        stats.xnorm, stats.rnorm, stats.gnorm,
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
      "threads", "samples", "variables", "λ", "blocks", "block size",
      "method", "time (ms)", "iterations", "xnorm", "rnorm", "gnorm"
    ]
  )

  # for manuscript
  pretty_table(
    results;
    backend = Val(:latex),
    formatters = (fmt_time, fmt_norm),
    tf = tf_latex_booktabs,
    header = [
      "threads", "samples", "variables", latex_cell"$\lambda$", "blocks", "block size",
      "method", "time (ms)", "iterations", latex_cell"$\|x\|$", latex_cell"$\|r\|$", latex_cell"$\|g\|$"
    ]
  )

  return nothing
end

n = parse(Int, ARGS[1])
p = parse(Int, ARGS[2])
λ = parse(Float64, ARGS[3])
seed = parse(Int, ARGS[4])
main(n, p, λ, seed)
