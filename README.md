# QBSolvers.jl

1. **OLS**: Linear Regression via least squares
2. **QREG**: Kernel-Smoothed Quantile Regression
3. **Markowitz**: Markowitz Mean-Variance Portfolio Optimization
4. **NNQP**: Non-Negative Quadratic Programming
5. **LASSO**: LASSO Regression
6. **TV**: Fused Lasso Proximity

## Quickstart

### OLS

Solves the OLS problem and ridge penalized LS. Wrappers for LSMR and LSQR implementations from [IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl) are also provided for comparison.

<details><summary>View Example Code</summary>

```julia
using QBSolvers, LinearAlgebra, Distributions

#
# simulate a problem instance
#
n, p, ρ = 16*512, 2*512, 0.5
Σ = [ρ^abs(i-j) for i in 1:p, j in 1:p]
cholΣ = cholesky!(Symmetric(Σ))
A = randn(n, p) * cholΣ.L
x0 = [k/p for k in 1:p]
b = A*x0 + 1/p .* randn(n)

#
# solve with QUB; no normalization
#
x, r, stats = @time solve_OLS(A, b; normalize=:none, tol=1e-3, lambda=0.0);
stats |> pairs |> display

#
# solve with QUB; Z-score standardization
#
x, r, stats = @time solve_OLS(A, b; normalize=:std, tol=1e-3, lambda=0.0);
stats |> pairs |> display

#
# solve with QUB; normalize to correlation matrix
#
x, r, stats = @time solve_OLS(A, b; normalize=:corr, tol=1e-3, lambda=0.0);
stats |> pairs |> display

#
# solve with QUB; deflate top eigenvalue + normalize to correlation matrix
#
x, r, stats = @time solve_OLS(A, b; normalize=:deflate, tol=1e-3, lambda=0.0);
stats |> pairs |> display

#
# solve with L-BFGS
#
x, r, stats = @time solve_OLS_lbfgs(A, b; precond=:none, tol=1e-3, lambda=0.0);
stats |> pairs |> display

#
# solve with L-BFGS; QUB preconditioner
#
x, r, stats = @time solve_OLS_lbfgs(A, b; precond=:qub, normalize=:deflate, tol=1e-3, lambda=0.0);
stats |> pairs |> display

#
# solve with LSMR; wrapped from IterativeSolvers.jl
#
x, r, stats = @time solve_OLS_lsmr(A, b; atol=1e-6, btol=1e-6, conlim=0.0, lambda=0.0);
stats |> pairs |> display
```

</details>

### QREG

Minimizes a **smoothed version** of the quantile loss; based on uniform kernel. See also [MMDeweighting.jl](https://github.com/qhengncsu/MMDeweighting.jl) and conquer ([GitHub](https://github.com/XiaoouPan/conquer), [CRAN](https://cran.r-project.org/web/packages/conquer/index.html)) for comparison.

<details><summary>View Example Code</summary>

```julia
using QBSolvers, LinearAlgebra, Distributions, Statistics
using MMDeweighting # requires separate installation: https://github.com/qhengncsu/MMDeweighting.jl

#
# simulate a problem instance
#
n, p, ρ = 16*512, 2*512, 0.4
q = 0.5
Σ = [ρ^abs(i-j) for i in 1:p, j in 1:p]
cholΣ = cholesky!(Symmetric(Σ))
A = [randn(n, p) * cholΣ.L ones(n)]
x0 = 0.1*ones(p+1)
b = A*x0 + rand(TDist(1.5), n) .- Statistics.quantile(TDist(1.5), q)
h = QBSolvers.default_bandwidth(A)

#
# solve with two-loop algorithm; preconditioned L-BFGS
#
x, r, stats = @time solve_QREG_lbfgs(A, b; q=q, h=h, version=1, normalize=:none, rtol=1e-6, accel=false);
stats |> pairs |> display

#
# solve with two-loop algorithm; preconditioned L-BFGS + Nesterov
#
x, r, stats = @time solve_QREG_lbfgs(A, b; q=q, h=h, version=1, normalize=:none, rtol=1e-6, accel=true);
stats |> pairs |> display

#
# solve with single-loop algorithm; preconditioned L-BFGS + Nesterov
#
x, r, stats = @time solve_QREG_lbfgs(A, b; q=q, h=h, version=2, normalize=:deflate, rtol=1e-6, accel=false);
stats |> pairs |> display

#
# solve it with Fast Quantile Regression from MMDeweighting.jl
#
x, _, iter, _ = @time MMDeweighting.FastQR(A, b, q; tol=1e-6, h=h, verbose=false);
r = b - A*x;

stats = (;
    iterations = iter,
    converged = true,
    xnorm = norm(x),
    rnorm = norm(r),
    loss1 = QBSolvers.qreg_objective(r, q),             # check function loss
    loss2 = QBSolvers.qreg_objective_uniform(r, q, h),  # smoothed loss (uniform)
);
stats |> pairs |> display
```

</details>

### Markowitz

Mean-Variance Portfolio Optimization via QUB + Dykstra's Algorithm or QUB + Duality.

<details><summary>View Example Code</summary>

```julia
using QBSolvers, LinearAlgebra, Statistics, CSV, DataFrames
#
# Load + Clean S&P 500 data
#
PKG_LOC = pathof(QBSolvers) |> dirname |> dirname # where package is installed
df = CSV.read(joinpath(PKG_LOC, "data", "sp500.csv"), DataFrame)
prices = Matrix(df[:, 2:end])
logrets = diff(log.(prices); dims=1)
μ = dropdims(mean(logrets; dims=1), dims=1)
bad_idx = findall(ismissing, μ)

df_clean = select(df, Not(bad_idx .+ 1))
prices = Matrix(df_clean[:, 2:end])
logrets = diff(log.(prices); dims=1)
logrets = logrets[:, Not(bad_idx)]
μ = dropdims(mean(logrets; dims=1), dims=1)
Σ = cov(logrets; dims=1)
#
# Configuration
#
r = 0.001
tol = 1e-9
#
# solve with 'QUB'
#
w1, var1, ret1, risk1, iter1 = mean_variance_mle(μ, Σ, r, tol = tol, 
  standard=false, duality=false, use_nesterov=true, verbose=false)
#
# solve with 'SQUB'
#
w2, var2, ret2, risk2, iter2 = mean_variance_mle(μ, Σ, r, tol = tol, 
  standard=true, duality=false, use_nesterov=true, verbose=false)
#
# solve with 'Duality'
#
w3, var3, ret3, risk3, iter3 = mean_variance_mle(μ, Σ, r, tol = tol, 
  standard=true, duality=true, use_nesterov=true, verbose=false)
```

</details>

### NNQP

Non-Negative Quadratic Programming via Spectral QUB.

<details><summary>View Example Code</summary>

```julia
using QBSolvers, LinearAlgebra, Statistics
#
# Simulate data
#
p = 1000
n = 10p
X = QBSolvers.generate_decay_correlated_matrix(n, p, 0.4)
β = 0.1ones(p)
truth =  X * β
y = truth + randn(n)
A = X'X
q = X'y
#
# Solve with Spectral QUB (SQUB) 
#
β̂_nqub, niters_nqub = NQUB_nqp_TwoMat(A, q;
  maxiter     = 10^3,
  ∇tol        = 1e-10,
  nonnegative = true,
  correlation_eigenvalue = true)
```

</details>

### LASSO

LASSO via Spectral QUB.

<details><summary>View Example Code</summary>

```julia
using QBSolvers, LinearAlgebra, SparseArrays, Statistics
#
# Configuration
#
n = 1000
p = 10n
s = 0.05
σ = 0.1
t_scale = 0.1
maxiter = 1000
tol = 1e-8
#
# Simulate data
#
X = QBSolvers.generate_decay_correlated_matrix(n, p, 0.4)
X_mean = vec(mean(X, dims = 1))
X_std  = vec(std(X,  dims = 1))
@inbounds for j in 1:p # careful standardization
  μ = X_mean[j]
  σj = X_std[j]
  σj = (σj == 0.0) ? 1.0 : σj # avoid divide-by-zero for constant columns
  @views X[:, j] .-= μ
  @views X[:, j] ./= σj
end
βtrue = sprand(p, s)
y = X * βtrue + σ * randn(n)
t = t_scale * sum(abs, βtrue)
#
# Solve with SQUB
#
βQ, itQ = lasso_prox_newton_woodbury!(X, y;
  t =t,             # |β| <= t
  ∇tol=tol,         # |βₙ₊₁ - βₙ|² / |βₙ|² ≤ ∇tol
  maxiter=maxiter,
  eigen_k=1,        # k = 1, eigenpair
  eigen_iters=5,    # s = 5, Krylov subspace dimension
  verbose=false
)
```

</details>

### TV

Several algorithms based QUB and related principles to the proximity problem under fusion constraints.

<details><summary>View Example Code</summary>

```julia
using QBSolvers
using Distributions, LinearAlgebra, Statistics, SparseArrays, Random
#
# Difference matrix
#
function diffmat(n::Int)
  D = spdiagm(0 => -ones(n-1), 1 => ones(n-1))
  return D[1:end-1, :]
end
#
# Simulate data
#
n = 1000
p = n - 1
s = 0.05
σ = 10
t_scale = 1.0 # [-1, 1]

βtrue = sprand(p, s)
D = diffmat(n)
X = transpose(D)
y = X * βtrue + σ * randn(n)
#
# Set common arguments used in this demo
#
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
#
# solve it with Projected Newton
#
β, iter = lsq_box_QUB_woodbury!(X, y; args..., method=:proj_newton, use_thomas=false, use_nesterov=false)
#
# solve it with Projected Newton + Thomas' Algorithm
#
β, iter = lsq_box_QUB_woodbury!(X, y; args..., method=:proj_newton, use_thomas=true, use_nesterov=false)
#
# solve it with DQUB
#
β, iter = lsq_box_QUB_woodbury!(X, y; args..., method=:woodbury, use_nesterov=false)
#
# solve it with DQUB + Nesterov
#
β, iter = lsq_box_QUB_woodbury!(X, y; args..., method=:woodbury, use_nesterov=true)
#
# solve it with SQUB
#
β, iter = lsq_box_QUB_woodbury!(X, y; args..., method=:QUB, use_nesterov=false)
#
# solve it with SQUB + Nesterov
#
β, iter = lsq_box_QUB_woodbury!(X, y; args..., method=:QUB, use_nesterov=true)
#
# solve it with MM
#
β, iter = prox_wTV_MM(y, maxit = 10000, w = w0, eps = 1e-6, verbose=false, use_nesterov=false)
#
# solve it with MM + Nesterov
#
β, iter = prox_wTV_MM(y, maxit = 10000, w = w0, eps = 1e-6, verbose=false, use_nesterov=true)
```

</details>

