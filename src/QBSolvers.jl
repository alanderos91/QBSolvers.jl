module QBSolvers

using BlockArrays
using LinearAlgebra, Statistics, Random
using IterativeSolvers

import Base: getindex, size, eltype, view
import Base: iterate, length, last, isdone, IteratorEltype
import LinearAlgebra: issymmetric, mul!, ldiv!, *

const BLAS_THREADS = Ref{Int}(BLAS.get_num_threads())

# Building blocks: efficient linear maps
include(joinpath("linearmaps", "GramPlusDiag.jl"))
include(joinpath("linearmaps", "BlkDiagHessian.jl"))
include(joinpath("linearmaps", "GramMinusBlkDiag.jl"))
include(joinpath("linearmaps", "NormalizedMatrix.jl"))
include(joinpath("linearmaps", "BlkDiagPlusRank1.jl"))
export GramPlusDiag, BlkDiagHessian, GramMinusBlkDiag, NormalizedMatrix, BlkDiagPlusRank1

# heuristics
_cache_gram_heuristic_(A::AbstractMatrix) = size(A, 1) >= size(A, 2)

# Other helpful abstractions
include("utilities.jl")
include("qubmatrix.jl")
include("lbfgs.jl")
export simulate_corr_matrix, simulate_group_corr_matrix, Exchangeable, AutoRegressive

# Problems
include("leastsquares.jl")
export solve_OLS, solve_OLS_lbfgs, solve_OLS_lsmr, solve_OLS_cg

include("quantilereg.jl")
export solve_QREG, solve_QREG_lbfgs

end # module
