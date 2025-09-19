###
### usage:
###
### julia -t 1 OLS.jl 8192 2048 0 1903 0.2
###

using Pkg, InteractiveUtils

if "MKL" in keys(Pkg.project().dependencies)
  using MKL
end

Pkg.activate(pwd())
Pkg.instantiate()

using QBSolvers
using Distributions, LinearAlgebra, Statistics, Random, StableRNGs
using BenchmarkTools, DataFrames, PrettyTables

BLAS.set_num_threads(10)
QBSolvers.BLAS_THREADS[] = BLAS.get_num_threads()

Pkg.status(); println()
versioninfo(); println()
BLAS.get_config() |> display; println()

function main(n, p, λ, seed, ρ)
  # benchmark parameters
  N = 1000              # number of @benchmark samples
  RNG = StableRNG(seed) # this benchmark script
  Random.seed!(seed)    # for QBSolvers code
  tscale = 1e-9 # report time in seconds

  results = DataFrame(
    method=String[],
    time=Float64[],
    iter=Int[],
    xnorm=Float64[],
    rnorm=Float64[],
    gnorm=Float64[],
  )

  # closure
  record! = let results=results, tscale=tscale
    function record!(alg, bench, stats)
      median_time = median(bench.times) * tscale
      push!(results,
        (
          alg,
          median_time, stats.iterations,
          stats.xnorm, stats.rnorm, stats.gnorm,
        )
      )
      println("$(lpad(alg, 6)) .......... $(round(median_time, digits=6)) seconds")
    end
  end

  # create problem instance
  Σ = [ρ^abs(i-j) for i in 1:p, j in 1:p];
  cholΣ = cholesky!(Symmetric(Σ))
  A = randn(RNG, n, p) * cholΣ.L
  x = randn(RNG, p)
  b = A*x + 1/sqrt(p) .* randn(RNG, n)

  println("seed:    ", seed)
  println("size(A): ", size(A, 1), " × ", size(A, 2))
  println("cond(A): ", cond(A))
  println("λ:       ", λ)
  println("ρ:       ", ρ)
  println()

  # QR, Julia default
  if n >= p && iszero(λ)
    _, _, statsQR = solve_OLS_qr(A, b)
    benchQR = @benchmark solve_OLS_qr($A, $b) samples=N
    record!("QR", benchQR, statsQR)
  end

  # Cholesky
  _, _, statsCHOL = solve_OLS_chol(A, b; lambda=λ)
  benchCHOL = @benchmark solve_OLS_chol($A, $b; lambda=$λ) samples=N
  record!("CHOL", benchCHOL, statsCHOL)

  # LSMR
  _, _, statsLSMR = solve_OLS_lsmr(A, b; lambda=λ)
  benchLSMR = @benchmark solve_OLS_lsmr($A, $b; lambda=$λ) samples=N
  record!("LSMR", benchLSMR, statsLSMR)

  gnormLSMR = statsLSMR.gnorm
  cgtol = 0.1*statsLSMR.rnorm 

  # LSQR
    _, _, statsLSQR = solve_OLS_lsqr(A, b; lambda=λ)
  benchLSQR = @benchmark solve_OLS_lsqr($A, $b; lambda=$λ) samples=N
  record!("LSQR", benchLSQR, statsLSQR)

  # CG
  _, _, statsCG = solve_OLS_cg(A, b; lambda=λ, reltol=zero(λ), abstol=cgtol)
  benchCG = @benchmark solve_OLS_cg($A, $b; lambda=$λ, reltol=zero($λ), abstol=$cgtol)
  record!("CG", benchCG, statsCG)

  # QUB closure
  QUB = let N=N, λ=λ, gnormLSMR=gnormLSMR
    function(A, b, normalize, alg)
      _, _, stats = solve_OLS(A, b; lambda=λ, tol=gnormLSMR, normalize=normalize, accel=false)
      bench = @benchmark solve_OLS($A, $b; lambda=$λ, tol=$gnormLSMR, normalize=$normalize, accel=false) samples=N
      record!(alg, bench, stats)
    end
  end

  QUB(A, b, :none, "QUB1")    # QUB: NO NORMALIZATION
  QUB(A, b, :std, "QUB2")     # QUB: DIAG + RANK-1, standardize
  QUB(A, b, :corr, "QUB3")    # QUB: DIAG + RANK-1, correlation
  QUB(A, b, :deflate, "QUB4") # QUB: DIAG + RANK-1, deflate by dominant eigenvalue
  # QUB(A, b, :rescale, "QUB5") # QUB: RESCALED PROBLEM

  # L-BFGS closure
  LBFGS = let N=N, λ=λ, gnormLSMR=gnormLSMR
    function(A, b, precond, normalize, alg)
      _, _, stats = solve_OLS_lbfgs(A, b;
        lambda=λ, tol=gnormLSMR, precond=precond, normalize=normalize)
      bench = @benchmark solve_OLS_lbfgs($A, $b;
        lambda=$λ, tol=$gnormLSMR, precond=$precond, normalize=$normalize) samples=N
      record!(alg, bench, stats)
    end
  end

  LBFGS(A, b, :none, :none, "LBFGS0")   # L-BFGS
  LBFGS(A, b, :qub, :none, "LBFGS1")    # L-BFGS + QUB PRECONDITIONER: NO NORMALIZATION
  LBFGS(A, b, :qub, :std, "LBFGS2")     # L-BFGS + QUB PRECONDITIONER: DIAG + RANK-1, standardize
  LBFGS(A, b, :qub, :corr, "LBFGS3")    # L-BFGS + QUB PRECONDITIONER: DIAG + RANK-1, correlation
  LBFGS(A, b, :qub, :deflate, "LBFGS4") # L-BFGS + QUB PRECONDITIONER: DIAG + RANK-1, deflate by dominant eigenvalue
  # LBFGS(A, b, :qub, :rescale, "LBFGS5") # L-BFGS + QUB PRECONDITIONER: RESCALED PROBLEM

  fmt_time = ft_printf("%5.2f", findfirst(==("time"), names(results)))
  fmt_norm = ft_latex_sn(4, findfirst(==("xnorm"), names(results)) .+ (0:2))

  # for human readability
  println()
  pretty_table(
    results;
    formatters = fmt_time,
    header = [
      "method", "time (s)", "iterations", "xnorm", "rnorm", "gnorm"
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
      "method", "time (s)", "iterations", latex_cell"$\|x\|$", latex_cell"$\|r\|$", latex_cell"$\|g\|$"
    ]
  )

  return nothing
end

n         = parse(Int, ARGS[1])
p         = parse(Int, ARGS[2])
λ         = parse(Float64, ARGS[3])
seed      = parse(Int, ARGS[4])
ρ         = parse(Float64, ARGS[5])
main(n, p, λ, seed, ρ)

