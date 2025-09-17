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
function diffmat(n::Int)
  D = spdiagm(0 => -ones(n-1), 1 => ones(n-1))
  return D[1:end-1, :]
end

function extract_stats(trial)
  times = trial.times ./ 1e6  # ns to ms
  return (
    minimum = minimum(times),
    median = median(times),
    mean = mean(times),
    std = std(times)
  )
end

function run_experiment_time(n, p, X, y, t_scale_list::Vector{Float64})
  results = DataFrame()

  for t_scale in t_scale_list
    println("Running for t_scale = $t_scale")

    args = (
      lower = -t_scale,
      upper = t_scale,
      gtol = 1e-6,
      eigen_k = 4,
      eigen_iters = 10,
      maxiter = 10000,
      verbose = false
    )

    w0 = ones(n-1) .* t_scale

    lsq_box_QUB_woodbury!(X, y; args..., method=:proj_newton, use_thomas=false, use_nesterov=false)
    lsq_box_QUB_woodbury!(X, y; args..., method=:proj_newton, use_thomas=true, use_nesterov=false)
    lsq_box_QUB_woodbury!(X, y; args..., method=:woodbury,    use_nesterov=false)
    lsq_box_QUB_woodbury!(X, y; args..., method=:woodbury,    use_nesterov=true)
    lsq_box_QUB_woodbury!(X, y; args..., method=:QUB,    use_nesterov=false)
    lsq_box_QUB_woodbury!(X, y; args..., method=:QUB,    use_nesterov=true)
    DD = X'*X
    prox_wTV_MM(y, maxit = 10000, w = w0, eps = 1e-6, verbose=false)
    prox_wTV_MM(y, maxit = 10000, w = w0, eps = 1e-6, verbose=false)

    # Run 3 methods
    b0 = @benchmark lsq_box_QUB_woodbury!($X, $y; $args..., method=:proj_newton, use_thomas=false, use_nesterov=false)
    b1 = @benchmark lsq_box_QUB_woodbury!($X, $y; $args..., method=:proj_newton, use_thomas=true,  use_nesterov=false)
    b2 = @benchmark lsq_box_QUB_woodbury!($X, $y; $args..., method=:woodbury,    use_nesterov=false)
    b3 = @benchmark lsq_box_QUB_woodbury!($X, $y; $args..., method=:woodbury,    use_nesterov=true)
    b4 = @benchmark lsq_box_QUB_woodbury!($X, $y; $args..., method=:QUB,    use_nesterov=false)
    b5 = @benchmark lsq_box_QUB_woodbury!($X, $y; $args..., method=:QUB,    use_nesterov=true)
    b6 = @benchmark prox_wTV_MM($y, maxit = 10000, w = $w0, eps = 1e-6, verbose=false, use_nesterov=false)
    b7 = @benchmark prox_wTV_MM($y, maxit = 10000, w = $w0, eps = 1e-6, verbose=false, use_nesterov=true)

    s0 = extract_stats(b0)
    s1 = extract_stats(b1)
    s2 = extract_stats(b2)
    s3 = extract_stats(b3)
    s4 = extract_stats(b4)
    s5 = extract_stats(b5)
    s6 = extract_stats(b6)
    s7 = extract_stats(b7)

    rows = [
      ("ProjNewton",        t_scale, s0.mean),
      ("ProjNewton Thomas", t_scale, s1.mean),
      ("DQUB",              t_scale, s2.mean),
      ("DQUB Nesterov",     t_scale, s3.mean),
      ("SQUB",              t_scale, s4.mean),
      ("SQUB Nesterov",     t_scale, s5.mean),
      ("MM",                t_scale, s6.mean),
      ("MM Nesterov",       t_scale, s7.mean),
    ]

    append!(results, DataFrame([Symbol(h) => [r[i] for r in rows] for (i,h) in enumerate([
    "Method", "t_scale", "Time(ms)"
    ])]))
  end

  return results
end

function run_experiment_stats(n, p, X, y, t_scale_list::Vector{Float64})
  results = DataFrame()
  for t_scale in t_scale_list
    println("Running for t_scale = $t_scale")

    args = (
      lower = -t_scale,
      upper = t_scale,
      gtol = 1e-6,
      eigen_k = 4,
      eigen_iters = 10,
      maxiter = 10000,
    verbose = false
    )

    w0 = ones(n-1)*t_scale

    β0, it0, fβ0, kkt0, len0  = lsq_box_QUB_woodbury!(X, y; args..., method=:proj_newton, use_nesterov=false, verbose=true, use_thomas=false)
    Fβ0 = QBLS.obj_tv(y - X * β0, y, ones(length(y)-1); eps=0.0)
    β1, it1, fβ1, kkt1, len1  = lsq_box_QUB_woodbury!(X, y; args..., method=:proj_newton, use_nesterov=false, verbose=true, use_thomas=true)
    Fβ1 = QBLS.obj_tv(y - X * β1, y, ones(length(y)-1); eps=0.0)
    β2, it2, fβ2, kkt2, len2  = lsq_box_QUB_woodbury!(X, y; args..., method=:woodbury,    use_nesterov=false, verbose=true)
    Fβ2 = QBLS.obj_tv(y - X * β2, y, ones(length(y)-1); eps=0.0)
    β3, it3, fβ3, kkt3, len3  = lsq_box_QUB_woodbury!(X, y; args..., method=:woodbury,    use_nesterov=true,  verbose=true)
    Fβ3 = QBLS.obj_tv(y - X * β3, y, ones(length(y)-1); eps=0.0)
    β4, it4, fβ4, kkt4, len4  = lsq_box_QUB_woodbury!(X, y; args..., method=:QUB,         use_nesterov=false, verbose=true)
    Fβ4 = QBLS.obj_tv(y - X * β4, y, ones(length(y)-1); eps=0.0)
    β5, it5, fβ5, kkt5, len5  = lsq_box_QUB_woodbury!(X, y; args..., method=:QUB,         use_nesterov=true,  verbose=true)
    Fβ5 = QBLS.obj_tv(y - X * β5, y, ones(length(y)-1); eps=0.0)

    DD = X'*X
    β6, it6, Fβ6, kkt6, len6  =  prox_wTV_MM(y, maxit = 10000, w = w0, eps = 1e-6, verbose=true, use_nesterov=false)
    fβ6 = QBLS.loss_ls(X, y, Matrix(DD)\(X'*(y.-β6)))
    β7, it7, Fβ7, kkt7, len7  =  prox_wTV_MM(y, maxit = 10000, w = w0, eps = 1e-6, verbose=true, use_nesterov=true)
    fβ7 = QBLS.loss_ls(X, y, Matrix(DD)\(X'*(y.-β7)))

    rows = [
      ("ProjNewton",        t_scale, it0, fβ0, Fβ0, kkt0, len0),
      ("ProjNewton Thomas", t_scale, it1, fβ1, Fβ1, kkt1, len1),
      ("DQUB",              t_scale, it2, fβ2, Fβ2, kkt2, len2),
      ("DQUB Nesterov",     t_scale, it3, fβ3, Fβ3, kkt3, len3),
      ("SQUB",              t_scale, it4, fβ4, Fβ4, kkt4, len4),
      ("SQUB Nesterov",     t_scale, it5, fβ5, Fβ5, kkt5, len5),
      ("MM",                t_scale, it6, fβ6, Fβ6, kkt6, len6),
      ("MM Nesterov",       t_scale, it7, fβ7, Fβ7, kkt7, len7),
    ]

    append!(results, DataFrame([Symbol(h) => [r[i] for r in rows] for (i,h) in enumerate([
    "Method", "t_scale", "Iterations", "g", "f", "# KKT", "# Inactive"
    ])]))
  end

  return results
end
#
# Benchmark Script
#
function main(; σ=10, seed=45, s=0.05)
  n = 1000
  p = n - 1
  Random.seed!(seed)

  βtrue = sprand(p, s)              # sparse signal
  D = diffmat(n)
  X = transpose(D)
  y = X * βtrue + σ * randn(n)

  # Run the benchmarks
  t_scales = [0.5, 1.0, 5.0, 10.0, 20.0]
  df_time = run_experiment_time(n, p, X, y, t_scales)
  df_stats = run_experiment_stats(n, p, X, y, t_scales)
  df = innerjoin(df_time, df_stats, on=[:t_scale, :Method])
  df0 = df[:, [1,2,3,4,6,5,8]]

  # Pretty table
  pretty_table(df0;
    header=names(df0),
    formatters = (
      ft_printf("%.2f", [2, 3]),
      ft_printf("%.0d", [4, 7]),
      ft_printf("%.4f", [5, 6])
    )
  )

  # LaTeX table
  io = IOBuffer()
  pretty_table(io, df0;
    header=names(df0),
    tf=tf_latex_booktabs,
    formatters = (
      ft_printf("%.2f", [2, 3]),
      ft_printf("%.0d", [4, 7]),
      ft_printf("%.4f", [5, 6])
    )
  )
  println("\nLaTeX Table Output:")
  println(String(take!(io)))

  return nothing
end
#
# Runtime
#
main(σ=10, seed=45, s=0.05)

