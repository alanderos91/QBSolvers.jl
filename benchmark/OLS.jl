###
### usage:
###
### julia -t 1 OLS.jl 8192 2048 0 1903 blkdiag false 16
###

using Pkg, InteractiveUtils

if "MKL" in keys(Pkg.project().dependencies)
  using MKL
end

Pkg.activate(pwd())
Pkg.instantiate()
include("utilities.jl")

using QBSolvers
using Distributions, LinearAlgebra, Statistics, Random
using BenchmarkTools, DataFrames, PrettyTables

BLAS.set_num_threads(10)
QBSolvers.BLAS_THREADS[] = BLAS.get_num_threads()

Pkg.status(); println()
versioninfo(); println()
BLAS.get_config() |> display; println()

function main(n, p, λ, seed, corrtype, use_noise, ngroups)
  iszero(λ) && @assert n > p
  @assert p > 2^7

  # benchmark parameters
  N = 1000 # number of @benchmark samples
  Random.seed!(seed)
  maxiter = 2*10^3

  results = DataFrame(
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

  # create problem instance
  Σ = make_Σ(corrtype, p, use_noise, ngroups)
  A = Transpose(rand(MvNormal(zeros(p), Σ), n)) |> Matrix
  x = ones(p)
  b = A*x + 1/p .* randn(n)
  x0 = zeros(p)
  println("Condition number of A: ", cond(A))
  println("Number of blocks in Σ: ", corrtype == "blkdiag" || corrtype == "blkband" ? ngroups : 1)
  println()

  # LSMR
  _, _, statsLSMR = solve_OLS_lsmr(A, b; lambda=λ)
  benchLSMR = @benchmark solve_OLS_lsmr($A, $b; lambda=$λ) samples=N
  push!(results,
    (n, p, λ, 1, p, "LSMR",
      median(benchLSMR.times) * 1e-6, statsLSMR.iterations,
      statsLSMR.xnorm, statsLSMR.rnorm, statsLSMR.gnorm,
    )
  )
  gnormLSMR = statsLSMR.gnorm
  cgtol = gnormLSMR^2

  # CG
  _, _, statsCG = solve_OLS_cg(A, b; lambda=λ, reltol=cgtol, abstol=cgtol, use_qlb=false)
  benchCG = @benchmark solve_OLS_cg($A, $b; lambda=$λ, reltol=$cgtol, abstol=$cgtol, use_qlb=false)
  push!(results,
    (n, p, λ, 1, p, "CG",
      median(benchCG.times) * 1e-6, statsCG.iterations,
      statsCG.xnorm, statsCG.rnorm, statsCG.gnorm,
    )
  )

  # CG with QLB preconditioner
  _, _, statsPCG = solve_OLS_cg(A, b; lambda=λ, reltol=cgtol, abstol=cgtol, use_qlb=true)
  benchPCG = @benchmark solve_OLS_cg($A, $b; lambda=$λ, reltol=$cgtol, abstol=$cgtol, use_qlb=true)
  push!(results,
    (n, p, λ, 1, p, "PCG",
      median(benchPCG.times) * 1e-6, statsPCG.iterations,
      statsPCG.xnorm, statsPCG.rnorm, statsPCG.gnorm,
    )
  )

  for var_per_blk in (2^k for k in 0:8)
    n_blk = fld(p, var_per_blk)
    for normalize in (false, true)
      _, _, stats = solve_OLS(A, b, x0, n_blk;
        lambda=λ, maxiter=maxiter, gtol=gnormLSMR, use_qlb=true, normalize=normalize)
      benchMM = @benchmark solve_OLS($A, $b, $x0, $n_blk;
        lambda=$λ, maxiter=$maxiter, gtol=$gnormLSMR, use_qlb=true, normalize=$normalize) samples=N
      push!(results,
        (n, p, λ, n_blk, var_per_blk, normalize ? "QUBn" : "QUB",
          median(benchMM.times) * 1e-6, stats.iterations,
          stats.xnorm, stats.rnorm, stats.gnorm,
        )
      )
    end
  end

  fmt_time = ft_printf("%5.0f", findfirst(==("time"), names(results)))
  fmt_norm = ft_latex_sn(4, findfirst(==("xnorm"), names(results)) .+ (0:2))

  # for human readability
  pretty_table(
    results;
    formatters = fmt_time,
    header = [
      "samples", "variables", "λ", "blocks", "block size",
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
      "samples", "variables", latex_cell"$\lambda$", "blocks", "block size",
      "method", "time (ms)", "iterations", latex_cell"$\|x\|$", latex_cell"$\|r\|$", latex_cell"$\|g\|$"
    ]
  )

  return nothing
end

n         = parse(Int, ARGS[1])
p         = parse(Int, ARGS[2])
λ         = parse(Float64, ARGS[3])
seed      = parse(Int, ARGS[4])
corrtype  = ARGS[5]
use_noise = parse(Bool, ARGS[6])
ngroups   = parse(Int, ARGS[7])
main(n, p, λ, seed, corrtype, use_noise, ngroups)
