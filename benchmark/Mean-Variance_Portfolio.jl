#
# Load Packages
#
using Pkg, InteractiveUtils

if "MKL" in keys(Pkg.project().dependencies)
  using MKL
end

Pkg.activate(pwd())
Pkg.instantiate()

using QBSolvers, JuMP, Gurobi, MathOptInterface
using LinearAlgebra, Statistics
using BenchmarkTools, CSV, DataFrames, PrettyTables

#
# Computing Environment
#
BLAS.set_num_threads(10)
QBSolvers.BLAS_THREADS[] = BLAS.get_num_threads()
Pkg.status(); println()
versioninfo(); println()
BLAS.get_config() |> display; println()

#
# Helper functions for managing benchmark results
#
function extract_stats(trial)
    times = trial.times ./ 1e6 
    return (
        minimum = minimum(times),
        median = median(times),
        mean = mean(times),
        std = std(times)
    )
end

#
# Mean-Variance Optimization with Gurobi
#
function mean_variance_optimization(μ::Vector{Float64}, Σ::Matrix{Float64}, target_return::Float64;
  tol::Float64        = 1e-8,
  short_selling::Bool = false
  )
  #
  n = length(μ)
  model = Model(Gurobi.Optimizer)
  set_optimizer_attribute(model, "OutputFlag", 0)    # silent mode
  set_optimizer_attribute(model, "BarConvTol", tol)  # tolerance for convergence

  @variable(model, w[1:n])
  @objective(model, Min, dot(w, Σ * w))               # minimize portfolio variance
  @constraint(model, sum(w) == 1)                     # fully invested
  @constraint(model, dot(μ, w) >= target_return)      # target return

  if !short_selling
      @constraint(model, w .>= 0)                     # long-only constraint
  end

  optimize!(model)

  status = termination_status(model)
  if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
      error("Optimization failed with status: $status")
  end
  
  weights = value.(w)
  @. weights = ifelse(weights < 1e-6, 0.0, weights)
  weights .= weights/sum(weights)
  
  port_return = dot(μ, weights)
  port_variance = dot(weights, Σ * weights)
  port_risk = sqrt(port_variance)
  iters = JuMP.MOI.get(model, MOI.BarrierIterations())

  return weights, port_variance, port_return, port_risk, iters
end

#
# Benchmark Script
#
function main()
  #
  # Load the S&P 500 data
  #
  df = CSV.read(joinpath("..", "data", "sp500.csv"), DataFrame)

  prices = Matrix(df[:, 2:end])
  logrets = diff(log.(prices); dims=1)
  μ = dropdims(mean(logrets; dims=1), dims=1)
  Σ = cov(logrets; dims=1)
  bad_idx = findall(ismissing, μ)

  df_clean = select(df, Not(bad_idx .+ 1))
  prices = Matrix(df_clean[:, 2:end])
  logrets = diff(log.(prices); dims=1)
  logrets = logrets[:, Not(bad_idx)]

  μ = dropdims(mean(logrets; dims=1), dims=1)
  Σ = cov(logrets; dims=1);
  r = 0.001
  tol = 1e-9

  #
  # Run the benchmarks
  #
  @btime mean_variance_mle($μ, $Σ, $r, tol = $tol, 
    standard=false, duality=false, use_nesterov=true, verbose=false)

  b1 = @benchmark mean_variance_mle($μ, $Σ, $r, tol = $tol, 
    standard=false, duality=false, use_nesterov=true, verbose=false)

  b2 = @benchmark mean_variance_mle($μ, $Σ, $r, tol = $tol, 
    standard=true, duality=false, use_nesterov=true, verbose=false)

  b3 = @benchmark mean_variance_mle($μ, $Σ, $r, tol = $tol, 
    standard=true, duality=true, use_nesterov=true, verbose=false)

  b4 = @benchmark mean_variance_optimization($μ, $Σ, $r, tol = $tol)

  s1 = extract_stats(b1)
  s2 = extract_stats(b2)
  s3 = extract_stats(b3)
  s4 = extract_stats(b4)

  w1, var1, ret1, risk1, iter1 = mean_variance_mle(μ, Σ, r, tol = tol, 
    standard=false, duality=false, use_nesterov=true, verbose=false)
  
  w2, var2, ret2, risk2, iter2 = mean_variance_mle(μ, Σ, r, tol = tol, 
    standard=true, duality=false, use_nesterov=true, verbose=false)
  
  w3, var3, ret3, risk3, iter3 = mean_variance_mle(μ, Σ, r, tol = tol, 
    standard=true, duality=true, use_nesterov=true, verbose=false)
  
  w4, var4, ret4, risk4, iter4 = mean_variance_optimization(μ, Σ, r, tol = tol)

  #
  # Assemble results into 'pretty tables'
  #
  rows = [
    ("QUB",     s1.mean, iter1, length(w1), sum(w1 .== 0), ret1, risk1),
    ("SQUB",    s2.mean, iter2, length(w2), sum(w2 .== 0), ret2, risk2),
    ("Duality", s3.mean, iter3, length(w3), sum(w3 .== 0), ret3, risk3),
    ("Gurobi",  s4.mean, iter4, length(w4), sum(w4 .== 0), ret4, risk4),
  ]
  header = ["Method", "Time(ms)", "Iterations", "# of Params", "Zeros", "Return", "Risk"]
  results_df = DataFrame([Symbol(h) => [r[i] for r in rows] for (i, h) in enumerate(header)])

  println("\nPretty Table:")
  pretty_table(results_df; 
    header=header, 
    formatters=(
      ft_printf("%.2f", [2]),     # Time
      ft_printf("%.0f", [3,4,5]), # Iterations
      ft_printf("%.3f", [6]), 
      ft_printf("%.7f", [7]) 
    )
  )

  io = IOBuffer()
  pretty_table(io, results_df;
    header=header,
    tf=tf_latex_booktabs,
    formatters=(
      ft_printf("%.2f", [2]),     # Time
      ft_printf("%.0f", [3,4,5]), # Iterations
      ft_printf("%.3f", [6]), 
      ft_printf("%.7f", [7]) 
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

