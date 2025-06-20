###
### usage:
###
### julia -t 1 GQLB.jl 8192 2048 0 1903 > GQLB_8192x2048_1903.out
###

using Pkg, InteractiveUtils

if "MKL" in keys(Pkg.project().dependencies)
  using MKL
end

Pkg.activate(pwd())
Pkg.instantiate()

using QBSolvers # for LSMR wrapper
using LinearAlgebra, Random, BlockArrays, BlockDiagonals, IterativeSolvers
using DataFrames, BenchmarkTools, PrettyTables

BLAS.set_num_threads(10)
QBSolvers.BLAS_THREADS[] = BLAS.get_num_threads()

Pkg.status(); println()
versioninfo(); println()
BLAS.get_config() |> display; println()

###
### Implementation
###

struct BlockedLS{T <: AbstractFloat}
  # data
  X    :: Matrix{T}
  y    :: Vector{T}
  Dblk :: Vector{Matrix{T}}
  # parameter
  β    :: Vector{T}
  # working arrays
  xtx :: Matrix{T} # only relaized when p <= n
  xty :: Vector{T}
  storage_n :: Vector{T}
  storage_p_1 :: Vector{T}
  storage_p_1_blk :: BlockedVector{T, Vector{T}, Tuple{BlockedOneTo{Int64, Vector{Int64}}}}
  storage_p_2 :: Vector{T}
  storage_p_2_blk :: BlockedVector{T, Vector{T}, Tuple{BlockedOneTo{Int64, Vector{Int64}}}}
end

# constructor
function BlockedLS(
      X         :: Matrix{T}, 
      y         :: Vector{T}, 
      blocksize :: Int
      ) where T <: AbstractFloat
  # dimensions
  n, p = size(X)
  @assert length(y) == n
  @assert blocksize <= p
  nblocks, lastblocksize = divrem(p, blocksize)
  blocksizes = repeat([blocksize], nblocks)
  if lastblocksize > 0
      push!(blocksizes, lastblocksize)
      nblocks += 1
  end 
  # sufficient statistics: xtx, xty
  # Dblk[b] = Xb'Xb
  if p > n
      xtx = Matrix{T}(undef, 0, 0) # empty matrix
      Xblk = BlockedArray(X, [n], blocksizes)
      Dblk = [view(Xblk, Block(1, b))' * view(Xblk, Block(1, b)) for b in 1:nblocks]
  else
      xtx = X'X
      xtxblk = BlockedArray(xtx, blocksizes, blocksizes)
      Dblk = [xtxblk[Block(b, b)] for b in 1:nblocks]
  end
  xty = X'y
  # parameter
  β = Vector{T}(undef, p)
  storage_n = Vector{T}(undef, n)
  storage_p_1 = Vector{T}(undef, p)
  storage_p_1_blk = BlockedVector(storage_p_1, blocksizes) # thin wrapper
  storage_p_2 = Vector{T}(undef, p)
  storage_p_2_blk = BlockedVector(storage_p_2, blocksizes) # thin wrapper
  BlockedLS(
      X, y, Dblk, 
      β, 
      xtx, xty,
      storage_n, 
      storage_p_1, storage_p_1_blk, 
      storage_p_2, storage_p_2_blk
  )
end

nblocks(bls :: BlockedLS) = length(bls.Dblk)
eltype(::BlockedLS{T}) where T = T
Base.size(bls::BlockedLS) = Base.size(bls.X)

struct BLSGmD{T <: AbstractFloat} <: AbstractMatrix{T} 
  bls :: BlockedLS{T}
end

# constructor
function BLSGmD(bls :: BlockedLS{T}) where T
  BLSGmD(bls)
end

Base.getindex(m::BLSGmD, i, j) = @views dot(m.bls.X[:, i], m.bls.X[:, j]) - BlockDiagonal(m.bls.Dblk)[i, j]
LinearAlgebra.issymmetric(::BLSGmD) = true
Base.size(m::BLSGmD) = size(m.bls.X, 2), size(m.bls.X, 2)
eltype(::BLSGmD{T}) where T = T

# overwrite `out` by `(X'X - D) * v`
function LinearAlgebra.mul!(
      out :: Vector{T}, 
      m   :: BLSGmD{T}, 
      v   :: Vector{T}
      ) where T <: AbstractFloat
  # out = X' * (X * v)
  mul!(out, transpose(m.bls.X), mul!(m.bls.storage_n, m.bls.X, v))
  # block substraction
  copyto!(m.bls.storage_p_1, v)
  vblk = m.bls.storage_p_1_blk
  wblk = m.bls.storage_p_2_blk
  for b in 1:nblocks(m.bls)
      mul!(view(wblk, Block(b)), m.bls.Dblk[b], view(vblk, Block(b)))
  end
  out .-= parent(wblk)
  out
end

function fit(X, y, block_size; kwargs...)
  bls = BlockedLS(X, y, block_size)
  fit!(bls; kwargs...)
end

function fit!(
  bls :: BlockedLS{T};
  λridge :: T = T(0),
  maxiter :: Integer = 1000,
  ∇tol :: AbstractFloat = 1e-3
  ) where T
  n, p = size(bls)
  # largest eigenvalue of X'X - D
  # λmax :: T = eigsolve(BLSGmD(bls), ones(T, p), 1, :LR, issymmetric = true, maxiter = 3)[1][1]
  λmax :: T = powm!(BLSGmD(bls), ones(T, p), maxiter = 3)[1]
  λ = λmax + λridge
  # pre-calculate Cholesky factors of diagonal blocks
  cblk = [cholesky(Symmetric(bls.Dblk[b] + λ * I)) for b in 1:nblocks(bls)]
  # cblk = [inv(cholesky(Symmetric(bls.Dblk[b] + λ * I))) for b in 1:nblocks(bls)]
  # starting point
  copyto!(bls.storage_p_2, bls.xty)
  @inbounds for b in 1:nblocks(bls)
      ldiv!(view(bls.storage_p_1_blk, Block(b)), cblk[b], view(bls.storage_p_2_blk, Block(b)))
      # mul!(view(bls.storage_p_1_blk, Block(b)), cblk[b], view(bls.storage_p_2_blk, Block(b)))
  end
  copyto!(bls.β, bls.storage_p_1)
  # GQLB loop
  niters = maxiter
  for iter in 1:maxiter
      # update residul storage_p_2 = X'(y - Xβ) = X'y - X'Xβ
      copyto!(bls.storage_p_2, bls.xty)
      if p > n
          mul!(bls.storage_p_2, transpose(bls.X), mul!(bls.storage_n, bls.X, bls.β), T(-1), T(1))
      else
          mul!(bls.storage_p_2, bls.xtx, bls.β, T(-1), T(1))
      end
      @. bls.storage_p_2 -= λridge*bls.β
      # convergence check
      if norm(bls.storage_p_2) < ∇tol
          niters = iter
          break
      end
      # block update β = βₙ + D⁻¹ X'(y - Xβ)
      @inbounds for b in 1:nblocks(bls)
          ldiv!(view(bls.storage_p_1_blk, Block(b)), cblk[b], view(bls.storage_p_2_blk, Block(b)))
          # mul!(view(bls.storage_p_1_blk, Block(b)), cblk[b], view(bls.storage_p_2_blk, Block(b)))
      end
      bls.β .+= bls.storage_p_1
  end
  bls.β, niters
end

###
### Script
###

function main(n, p, λ, seed)
  iszero(λ) && @assert n > p
  @assert p > 2^7

  N = 1000 # number of @benchmark samples
  Random.seed!(seed)
  maxiter = 10^4

  results = DataFrame(
    threads=Int[],
    n=Int[],
    p=Int[],
    λ=Float64[],
    blocks=Int[],
    blocksize=Int[],
    method=String[],
    time=Float64[],
    iter=Int[],
    xnorm=Float64[],
    rnorm=Float64[],
    gnorm=Float64[],
  )

  A = randn(n, p)
  x = ones(p)
  b = A*x + 1/p .* randn(n)

  # LSMR
  xLSMR, rLSMR, statsLSMR = solve_OLS_lsmr(A, b; lambda=λ)
  benchLSMR = @benchmark solve_OLS_lsmr($A, $b; lambda=$λ) samples=N
  push!(results,
    (Threads.nthreads(), n, p, λ, 1, p, "LSMR",
      median(benchLSMR.times) * 1e-6, statsLSMR.iterations,
      statsLSMR.xnorm, statsLSMR.rnorm, statsLSMR.gnorm,
    )
  )
  gnormLSMR = statsLSMR.gnorm
  cgtol = gnormLSMR^2

  # CG
  _, _, statsCG = solve_OLS_cg(A, b; lambda=λ, reltol=cgtol, abstol=cgtol, use_qlb=false)
  benchCG = @benchmark solve_OLS_cg($A, $b; lambda=$λ, reltol=$cgtol, abstol=$cgtol, use_qlb=false)
  push!(results,
    (Threads.nthreads(), n, p, λ, 1, p, "CG",
      median(benchCG.times) * 1e-6, statsCG.iterations,
      statsCG.xnorm, statsCG.rnorm, statsCG.gnorm,
    )
  )

  # CG with QLB preconditioner
  _, _, statsPCG = solve_OLS_cg(A, b; lambda=λ, reltol=cgtol, abstol=cgtol, use_qlb=true)
  benchPCG = @benchmark solve_OLS_cg($A, $b; lambda=$λ, reltol=$cgtol, abstol=$cgtol, use_qlb=true)
  push!(results,
    (Threads.nthreads(), n, p, λ, 1, p, "PCG",
      median(benchPCG.times) * 1e-6, statsPCG.iterations,
      statsPCG.xnorm, statsPCG.rnorm, statsPCG.gnorm,
    )
  )

  for var_per_blk in (2^k for k in 0:8)
    n_blk = fld(p, var_per_blk)
    xMM, iters = fit(A, b, var_per_blk;
      λridge=λ, maxiter=maxiter, ∇tol=gnormLSMR)
    rMM = A*xMM-b
    xnormMM = norm(xMM)
    rnormMM = norm(rMM)
    gnormMM = norm(A'rMM + λ*xMM)
    benchMM = @benchmark fit($A, $b, $var_per_blk;
      λridge=$λ, maxiter=$maxiter, ∇tol=$gnormLSMR) samples=N
    push!(results,
      (Threads.nthreads(), n, p, λ, n_blk, var_per_blk, "MM-QLB",
        median(benchMM.times) * 1e-6, iters,
        xnormMM, rnormMM, gnormMM,
      )
    )
  end

  fmt_time = ft_printf("%5.0f", findfirst(==("time"), names(results)))
  fmt_norm = ft_latex_sn(4, findfirst(==("xnorm"), names(results)) .+ (0:2))

  # for human readability
  pretty_table(
    results;
    formatters = fmt_time,
    header = [
      "threads", "samples", "variables", "λ", "blocks", "block size",
      "method", "time (ms)", "iterations", "xnorm", "rnorm", "gnorm"
    ]
  )
  println()

  # for manuscript
  pretty_table(
    results;
    backend = Val(:latex),
    formatters = (fmt_time, fmt_norm),
    tf = tf_latex_booktabs,
    header = [
      "threads", "samples", "variables", latex_cell"$\lambda$", "blocks", "block size",
      "method", "time (ms)", "iterations", latex_cell"$\|x\|$", latex_cell"$\|r\|$", latex_cell"$\|g\|$"
    ]
  )

  return nothing
end

n = parse(Int, ARGS[1])
p = parse(Int, ARGS[2])
λ = parse(Float64, ARGS[3])
seed = parse(Int, ARGS[4])
main(n, p, λ, seed)
