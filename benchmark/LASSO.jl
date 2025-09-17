#
# Load Packages
#
using Pkg, InteractiveUtils

if "MKL" in keys(Pkg.project().dependencies)
  using MKL
end

Pkg.activate(pwd())
Pkg.instantiate()

using QBSolvers
using Distributions, LinearAlgebra, Statistics, SparseArrays, Random
using BenchmarkTools, CSV, DataFrames, PrettyTables
QBLS = QBSolvers
#
# Computing Environment
#
BLAS.set_num_threads(10)
QBSolvers.BLAS_THREADS[] = BLAS.get_num_threads()
Pkg.status(); println()
versioninfo(); println()
BLAS.get_config() |> display; println()
#
# Helper Functions
#
# Extract benchmark statistics in *seconds* (BenchmarkTools uses ns)
extract_sec_stats(b::BenchmarkTools.Trial) = (
    minimum = minimum(b).time / 1e9,
    median  = median(b).time  / 1e9,
    mean    = mean(b).time    / 1e9,
    std     = std(b).time     / 1e9
)
#
# Benchmark Script
#
function main(seed=45)
  #
  # Create problem instance
  #
  Random.seed!(seed)
  n = 1000
  p = 10n
  covM = [0.4^abs(i-j) for i in 1:p, j in 1:p]
  d = MvNormal(zeros(p), covM)
  X = Transpose(rand(d, n))
  s = 0.05          # density of βtrue (≈5% nonzeros)
  σ = 0.1
  t_scale = 0.1

  # 1) design + ground-truth
  βtrue = sprand(p, s)              # SparseVector length p, density s
  # d = TDist(1.5)
  # truth =  X * βtrue .+ 1
  # y = truth + rand(d, n) .- Statistics.quantile(d,0.5)
  X = Matrix(X)

  # 2) normalize X column-wise: (X[:,j] - mean) / std, with std==0 -> 1
  X_mean = vec(mean(X, dims = 1))   # p-vector
  X_std  = vec(std(X,  dims = 1))   # p-vector (population or sample; pick one)
  @inbounds for j in 1:p
    μ = X_mean[j]
    σj = X_std[j]
    σj = (σj == 0.0) ? 1.0 : σj    # avoid divide-by-zero for constant columns
    @views X[:, j] .-= μ
    @views X[:, j] ./= σj
  end

  # 3) response (build AFTER normalization)
  y = X * βtrue + σ * randn(n)

  # 4) ℓ1-ball radius (based on the true coefficients)
  t = t_scale * sum(abs, βtrue)
  #
  # Benchmarks
  #
  # ---------- Configuration ----------
  maxiter = 1000
  tol = 1e-8

  # ---------- Run FISTA ----------
  βF, itF = fista_l1_ball(X, y; t=t, maxiter=maxiter, tol=tol)
  fF  = QBLS.loss_ls(X,y,βF)
  l1F = sum(abs, βF)
  pgF = QBLS.pg_residual(X,y,βF,t)
  tF  = @benchmark βF, itF = fista_l1_ball($X, $y; t=$t, maxiter=$maxiter, tol=$tol)
  s_F = extract_sec_stats(tF)

  # ---------- Projected Newton ----------
  βN, itN = projected_newton_l1!(X, y; t=t, maxiter=maxiter, ∇tol=tol)
  fN  = QBLS.loss_ls(X,y,βN)
  l1N = sum(abs, βN)
  pgN = QBLS.pg_residual(X,y,βN,t)
  tN  = @benchmark βN, itN = projected_newton_l1!($X, $y; t=$t, maxiter=$maxiter, ∇tol=$tol)
  s_N = extract_sec_stats(tN)

  # ---------- QUB: run multiple configurations ----------
  qub_configs = [
    (ek=1,  ei=5),
    (ek=3,  ei=10),
    (ek=5,  ei=10),
    (ek=10, ei=20),
  ]

  # For collecting QUB rows
  qub_rows = Vector{Tuple}()
  qub_stats = Vector{NamedTuple}()

  for cfg in qub_configs
    ek = cfg.ek
    ei = cfg.ei

    βQ, itQ = lasso_prox_newton_woodbury!(X, y;
      t=t, ∇tol=tol, maxiter=maxiter, eigen_k=ek, eigen_iters=ei, verbose=false)

    fQ  = QBLS.loss_ls(X,y,βQ)
    l1Q = sum(abs, βQ)
    pgQ = QBLS.pg_residual(X,y,βQ,t)

    tQ  = @benchmark βQ, itQ = lasso_prox_newton_woodbury!($X, $y;
      t=$t, ∇tol=$tol, maxiter=$maxiter, eigen_k=$ek, eigen_iters=$ei, verbose=false)

    s_Q = extract_sec_stats(tQ)

    # Push row with a clear method label
    push!(qub_rows, (
      "QUB(k=$(ek),eiter=$(ei))",
      s_Q.mean,
      itQ,
      sum(βQ .!= 0),
      fQ,
      fQ + t * sum(abs.(βQ)),
      sum(abs.(βQ))
    ))

    push!(qub_stats, s_Q)
  end

  # ---------- Assemble results table ----------
  header = ["Method", "Time(s)", "Iters", "Nonzeros", "Loss", "f + t*|β|₁", "|β|₁"]

  # Base rows: FISTA & Newton
  base_rows = [
    ("FISTA",  s_F.mean, itF, sum(βF .!= 0), fF, fF + t * sum(abs.(βF)), sum(abs.(βF))),
    ("ProNewton", s_N.mean, itN, sum(βN .!= 0), fN, fN + t * sum(abs.(βN)), sum(abs.(βN))),
  ]

  # Final rows = QUB (4 rows) + base_rows
  rows = vcat(qub_rows, base_rows)

  df = DataFrame([Symbol(h) => [r[i] for r in rows] for (i,h) in enumerate(header)])

  # ---------- Pretty table in terminal ----------
  println("\nPretty Table:")
  pretty_table(df;
    header = header,
    formatters = (
      ft_printf("%.6f", [2]),       # Time(s)
      ft_printf("%.0f",  [3,4]),    # Iterations, Zeros
      ft_printf("%.6f", [5,6,7])    # Loss, f + t*|β|1, |β|1
    )
  )

  # ---------- Pretty table in LaTeX ----------
  io = IOBuffer()
  pretty_table(io, df;
    header = header,
    tf = tf_latex_booktabs,
    formatters = (
      ft_printf("%.6f", [2]),       # Time(s)
      ft_printf("%.0f",  [3,4]),    # Iterations, Zeros
      ft_printf("%.6f", [5,6,7])    # Loss, f + t*|β|1, |β|1
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

