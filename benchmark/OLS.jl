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
using Distributions, LinearAlgebra, Statistics, Random
using BenchmarkTools, DataFrames, PrettyTables

BLAS.set_num_threads(10)
QBSolvers.BLAS_THREADS[] = BLAS.get_num_threads()

Pkg.status(); println()
versioninfo(); println()
BLAS.get_config() |> display; println()

function main(n, p, λ, seed, rho)
  # benchmark parameters
  N = 1000 # number of @benchmark samples
  Random.seed!(seed)
  tscale = 1e-6 # report time in milliseconds

  results = DataFrame(
    n=Int[],
    p=Int[],
    λ=Float64[],
    method=String[],
    time=Float64[],
    iter=Int[],
    xnorm=Float64[],
    rnorm=Float64[],
    gnorm=Float64[],
  )

  # closure
  record! = let n=n, p=p, λ=λ, results=results, tscale=tscale
    function record!(alg, bench, stats)
      push!(results,
        (n, p, λ, alg,
          median(bench.times) * tscale, stats.iterations,
          stats.xnorm, stats.rnorm, stats.gnorm,
        )
      )
    end
  end

  # create problem instance
  Σ = [rho^abs(i-j) for i in 1:p, j in 1:p];
  cholΣ = cholesky!(Symmetric(Σ))
  A = randn(n, p) * cholΣ.L
  x = randn(p)
  b = A*x + 1/sqrt(p) .* randn(n)
  println("Condition number of A: ", cond(A))
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
  cgtol = gnormLSMR^2

  # LSQR
    _, _, statsLSQR = solve_OLS_lsqr(A, b; lambda=λ)
  benchLSQR = @benchmark solve_OLS_lsqr($A, $b; lambda=$λ) samples=N
  record!("LSQR", benchLSQR, statsLSQR)

  # CG
  _, _, statsCG = solve_OLS_cg(A, b; lambda=λ, reltol=cgtol, abstol=cgtol)
  benchCG = @benchmark solve_OLS_cg($A, $b; lambda=$λ, reltol=$cgtol, abstol=$cgtol)
  record!("CG", benchCG, statsCG)

  # QUB closure
  QUB = let N=N, λ=λ, gnormLSMR=gnormLSMR
    function(A, b, normalize, alg)
      _, _, stats = solve_OLS(A, b; lambda=λ, tol=gnormLSMR, normalize=normalize, maxiter=10^4, accel=true)
      bench = @benchmark solve_OLS($A, $b; lambda=$λ, tol=$gnormLSMR, normalize=$normalize, maxiter=10^4, accel=true) samples=N
      record!(alg, bench, stats)
    end
  end

  QUB(A, b, :none, "QUB1")    # QUB: NO NORMALIZATION
  QUB(A, b, :qub, "QUB2")     # QUB: DIAG + RANK-1
  QUB(A, b, :rescale, "QUB3") # QUB: RESCALE

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
  LBFGS(A, b, :qub, :qub, "LBFGS2")     # L-BFGS + QUB PRECONDITIONER: DIAG + RANK-1
  LBFGS(A, b, :qub, :rescale, "LBFGS3") # L-BFGS + QUB PRECONDITIONER: RESCALE

  fmt_time = ft_printf("%5.0f", findfirst(==("time"), names(results)))
  fmt_norm = ft_latex_sn(4, findfirst(==("xnorm"), names(results)) .+ (0:2))

  # for human readability
  pretty_table(
    results;
    formatters = fmt_time,
    header = [
      "samples", "variables", "λ",
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
      "samples", "variables", latex_cell"$\lambda$",
      "method", "time (ms)", "iterations", latex_cell"$\|x\|$", latex_cell"$\|r\|$", latex_cell"$\|g\|$"
    ]
  )

  return nothing
end

n         = parse(Int, ARGS[1])
p         = parse(Int, ARGS[2])
λ         = parse(Float64, ARGS[3])
seed      = parse(Int, ARGS[4])
rho       = parse(Float64, ARGS[5])
main(n, p, λ, seed, rho)

