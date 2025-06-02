import Pkg
Pkg.activate(pwd())

using ParallelLeastSquares
using Profile

Profile.init(delay=1e-6)

function main()
  n, p, maxiter, gtol = 1024, 4*1024, 10^3, 1e-4
  n_blk = div(p, 32)
  A = randn(n, p)
  x = ones(p)
  b = A*x + 1/p .* randn(n)
  x0 = zeros(p)

  solve_OLS(A, b, x0, n_blk; maxiter=maxiter, gtol=gtol, lambda=1e3, use_qlb=true, gram=false, normalize=false)
  Profile.clear_malloc_data()
  solve_OLS(A, b, x0, n_blk; maxiter=maxiter, gtol=gtol, lambda=1e3, use_qlb=true, gram=false, normalize=false)

  return nothing
end

@info "Starting profiling..."
main()
