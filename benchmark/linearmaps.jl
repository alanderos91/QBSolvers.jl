import Pkg
Pkg.activate(pwd())

using QBSolvers, IterativeSolvers
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
@benchmark( GramPlusDiag($A, alpha=1, beta=0) ) |> display

@info "[GramPlusDiag] w = AᵀA*v, Matrix"
@benchmark( mul!($w, Symmetric($AtA), $v) ) |> display

@info "[GramPlusDiag] w = AᵀA*v, LinearMap"
AtAm = GramPlusDiag(A, alpha=1, beta=0);
@benchmark( mul!($w, $AtAm, $v) ) |> display

