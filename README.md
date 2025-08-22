# QBSolvers.jl

- **OLS**: linear regression via least squares
- **QREG**: kernel-smoothed quantile regression

## Quickstart

### Least Squares

Solves the OLS problem and ridge penalized LS. The LSMR implementation from IterativeSolvers.jl is also provided for comparison.

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

### Quantile Regression

Minimizes a **smoothed version** of the quantile loss; based on uniform kernel.

```julia
using QBSolvers, LinearAlgebra, Distributions, Statistics
using MMDeweighting

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
