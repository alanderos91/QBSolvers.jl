import Pkg
Pkg.activate(pwd())

using ParallelLeastSquares, IterativeSolvers
using LinearAlgebra, Statistics, Random
using BenchmarkTools

n, p = 8*1024, 2*1024
Random.seed!(1903)

A = randn(n, p)
x = ones(p)
b = A*x + 1/p .* randn(n)
x0 = zeros(p)

# Gram + Diag
AtA = A'A
w, v= similar(AtA, p), fill!(similar(AtA, p), 1)


@info "[GramPlusDiag] Construction"
@benchmark GramPlusDiag($A, alpha=1, beta=0)

@info "[GramPlusDiag] w = AᵀA*v, Matrix"
@benchmark mul!($w, Symmetric($AtA), $v)

@info "[GramPlusDiag] w = AᵀA*v, LinearMap"
AtAm = GramPlusDiag(A, alpha=1, beta=0);
@benchmark mul!($w, $AtAm, $v)

# BlkDiagHessian
n_blk = 256

@info "[BlkDiagHessian] Construction, factor=false"
@benchmark BlkDiagHessian($A, $n_blk, alpha=1, beta=0, factor=false)

@info "[BlkDiagHessian] Construction, factor=true"
@benchmark BlkDiagHessian($A, $n_blk, alpha=1, beta=0, factor=true)

Dm = BlkDiagHessian(A, n_blk, alpha=1, beta=0, factor=true);
D = convert(Matrix, Dm);

@info "[BlkDiagHessian] w = D*v, Matrix"
@benchmark mul!($w, $D, $v)

@info "[BlkDiagHessian] w = D*v, LinearMap"
@benchmark mul!($w, $Dm, $v)

@benchmark ldiv!($w, $Dm, $v)

# Gram - BlkDiag

@info "[GramMinusBlkDiag] Construction, full"

@benchmark begin
  local D = BlkDiagHessian($A, $n_blk, alpha=1, beta=0, factor=false)
  GramMinusBlkDiag($A, D)
end

@info "GramMinusBlkDiag] Construction, object only"
@benchmark GramMinusBlkDiag($A, $Dm)

Mm = GramMinusBlkDiag(A, Dm);
M = convert(Matrix, Mm);
@benchmark mul!($w, $M, $v)
@benchmark mul!($w, $Mm, $v)