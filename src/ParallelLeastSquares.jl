module ParallelLeastSquares

using BlockArrays
using LinearAlgebra
using IterativeSolvers

import Base: getindex, size, eltype
import LinearAlgebra: issymmetric, mul!, ldiv!

const BLAS_THREADS = Ref{Int}(BLAS.get_num_threads())

# Building blocks: efficient linear maps
include(joinpath("linearmaps", "GramPlusDiag.jl"))
include(joinpath("linearmaps", "BlkDiagHessian.jl"))
include(joinpath("linearmaps", "GramMinusBlkDiag.jl"))
export GramPlusDiag, BlkDiagHessian, GramMinusBlkDiag

# Other helpful abstractions
include("utilities.jl")

# Problems
include("leastsquares.jl")
export solve_OLS, solve_OLS_lsmr, solve_OLS_cg

end # module
