# QBSolvers.jl

- **OLS**: linear regression via least squares
- **QREG**: kernel-smoothed quantile regression

## Quickstart

### Least Squares

Solves the OLS problem and ridge penalized LS. The LSMR implementation from IterativeSolvers.jl is also provided for comparison.

```julia
using QBSolvers, LinearAlgebra, Distributions

# block structure of correlation matrix: block diagonal with 16 Toeplitz blocks
n, p = 16*512, 2*512
n_blk = 16
var_per_blk = div(p, n_blk)
blksizes = repeat([var_per_blk], n_blk)
S = simulate_group_corr_matrix(Float64, [AutoRegressive(0.8) for _ in 1:n_blk], p, blksizes;
    m = 8,
    epsilon = 0.0,
    verynoisy = false
)

# simulate a problem instance
A = Transpose(rand(MvNormal(zeros(p), S), n)) |> Matrix
x0 = [k/p for k in 1:p]
b = A*x0 + 1/p .* randn(n)
x_init = zeros(p)

#
# solve it with diagonal QUB approximation (n_blk = 1)
#
x, r, stats = @time solve_OLS(A, b, x_init, 1; maxiter=10^3, use_qlb=true, gtol=1e-6, lambda=0.0)
stats |> pairs |> display

#
# solve it with block diagonal QUB approximation (n_blk = 16)
#
x, r, stats = @time solve_OLS(A, b, x_init, 16; maxiter=10^3, use_qlb=true, gtol=1e-6, lambda=0.0)
stats |> pairs |> display

#
# compare to LSMR
#
x, r, stats = @time solve_OLS_lsmr(A, b; lambda=0.0)
stats |> pairs |> display
```

### Quantile Regression

Minimizes a **smoothed version** of the quantile loss; based on uniform kernel.

```julia
using QBSolvers, LinearAlgebra, Distributions, Statistics
using MMDeweighting

# block structure of correlation matrix: block diagonal with 16 Toeplitz blocks
n, p = 16*512, 2*512
n_blk = 16
var_per_blk = div(p, n_blk)
blksizes = repeat([var_per_blk], n_blk)
S = simulate_group_corr_matrix(Float64, [AutoRegressive(0.8) for _ in 1:n_blk], p, blksizes;
    m = 8,
    epsilon = 0.0,
    verynoisy = false
)

# simulate a problem instance
q = 0.5
A = Transpose(rand(MvNormal(zeros(p), S), n)) |> Matrix
x0 = 0.1*ones(p)
b = A*x0 + rand(TDist(1.5), n) .- Statistics.quantile(TDist(1.5), q)
x_init = zeros(p)
h = QBSolvers.default_bandwidth(A)

#
# solve it with diagonal QUB approximation (n_blk = 1)
#
x, r, stats = @time solve_QREG(A, b, x_init, 1; q=q, h=h, maxiter=10^3, gtol=1e-2, gram=false)
stats = (; stats...,
    loss = QBSolvers.qreg_objective_uniform(r, q, h),
)
stats |> pairs |> display

#
# solve it with Fast Quantile Regression from MMDeweighting.jl
#
x, _, iter, _ = @time MMDeweighting.FastQR(A, b, q; tol=1e-8, h=h, verbose=false)

# compute 'gradient' from a surrogate function for comparison
r = b - A*x
z = QBSolvers.prox_abs!(similar(r), r, h)
@. z = r - z + (2*q-1)*h
g = inv(2*h) * transpose(A) * z

stats = (;
    iterations = iter,
    converged = true,
    xnorm = norm(x),
    rnorm = norm(r),
    gnorm = norm(g),
    loss = QBSolvers.qreg_objective_uniform(r, q, h),
)
stats |> pairs |> display
```

### Boosting in Classification [WIP]

Wraps SAMME implementation in DecisionTree.jl. We implement a QUB-based algorithm to estimate classifier weights for a pre-trained ensemble of weak classifiers.

```julia
using QBSolvers, LinearAlgebra, Distributions

# load some data; reexported from DecisionTree.jl
features, labels = QBSolvers.load_data("digits")
features = Float64.(features)
labels = string.(labels)
M = 100 # number of classifiers

#
# solve with multiclass AdaBoost (aka SAMME)
#
model, theta, stats = @time fit_adaboost(labels, features, M)
stats |> pairs |> display

#
# solve with QUB
#
model, theta, stats = @time fit_classifier(labels, features, M;
    update=:proj, # projected Newton
    maxiter=1000,
    rtol=1e-8
)
stats |> pairs |> display
```
