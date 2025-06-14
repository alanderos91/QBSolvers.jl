#
# Spectral radius of AtA - J, where J is (block) diagonal Matrix based on AtA.
#
function estimate_spectral_radius(AtA, J; kwargs...)
  lambda, _, ch = powm!(GramMinusBlkDiag(AtA, J), ones(size(J, 1)); log=true, kwargs...)
  return abs(lambda)
end

#
# Efficient computation of AtA (block) diagonal
#
function compute_main_diagonal(A::AbstractMatrix, AtA)
  data = similar(A, size(A, 2))
  if size(AtA, 2) > 0
    idx = diagind(AtA)
    @views @. data = AtA[idx]
  else
    fdot(x) = dot(x, x)
    map!(fdot, data, eachcol(A))
  end
  return Diagonal(data)
end

function compute_main_diagonal(A::NormalizedMatrix, AtA)
  data = similar(A.A, size(A, 2))
  fill!(data, 1)
  return Diagonal(data)
end

function compute_block_diagonal(AtApD, n_blk; gram::Bool=false, kwargs...)
  if size(AtApD.AtA, 2) > 0
    J = BlkDiagHessian(AtApD, n_blk; kwargs...)
  else
    J = BlkDiagHessian(AtApD.A, AtApD.D, n_blk; gram=gram, kwargs...)
  end
  return J
end

#
# Allocation-free block extraction of Dⱼ in BlkDiagHessian αAⱼᵀAⱼ + βDⱼ
#
function get_diag_block(D::UniformScaling, baxes, blkI, blkJ)
  return D
end

function get_diag_block(D::Diagonal, baxes, blkI, blkJ)
  idxI, idxJ = baxes[blkI], baxes[blkJ]
  return view(D, idxI, idxJ)
end

function get_A_and_D_blk(A_blk, D, k)
  baxes = axes(A_blk, 2)
  A_k = view(A_blk, Block(k))
  D_kk = get_diag_block(D, baxes, Block(k), Block(k))
  return A_k, D_kk
end

function get_A_and_D_blk(A_blk, AtA_blk, D, k)
  baxes = axes(A_blk, 2)
  A_k = view(A_blk, Block(k))
  A_kk = view(AtA_blk, Block(k), Block(k))
  D_kk = get_diag_block(D, baxes, Block(k), Block(k))
  return A_k, A_kk, D_kk
end

function make_AtApD_blk(A_blk::BlockedArray, D, k::Integer, alpha::Real, beta::Real, gram::Bool)
  A_k, D_kk = get_A_and_D_blk(A_blk, D, k)
  return GramPlusDiag(A_k, D_kk; alpha=alpha, beta=beta, gram=gram)
end

function make_AtApD_blk(A_blk::BlockedArray, AtA_blk::BlockedArray, D, k::Integer, alpha::Real, beta::Real)
  T = eltype(A_blk)
  A_k, A_kk, D_kk = get_A_and_D_blk(A_blk, AtA_blk, D, k)
  return GramPlusDiag(A_k, copy(A_kk), D_kk, size(A_k, 1), size(A_k, 2), similar(parent(A_blk), 0), T(alpha), T(beta))
end

function make_AtApD_blk(bdh::BlkDiagHessian, A_blk::BlockedArray, D, k::Integer, alpha::Real, beta::Real)
  T = eltype(bdh)
  A_k, D_kk = get_A_and_D_blk(A_blk, D, k)
  return GramPlusDiag(A_k, bdh.diag[k].AtA, D_kk,
      bdh.diag[k].n_obs, bdh.diag[k].n_var, bdh.diag[k].tmp, alpha, beta)
end

#
# Gram + Diag computations
#
function factor_AtApD_blk(AtApD::GramPlusDiag, alpha, beta)
  T = eltype(AtApD)
  data = Matrix{T}(undef, size(AtApD))
  factor_AtApD_blk!(data, AtApD, alpha, beta)
end

function factor_AtApD_blk!(data, AtApD::GramPlusDiag, alpha, beta)
  A, AtA, D = AtApD.A, AtApD.AtA, AtApD.D
  if size(AtA, 2) > 0
    @. data = AtA
  else
    form_AtA!(data, A)
  end
  maybe_normalize!(data, A)
  @. data = alpha*data
  !iszero(beta) && make_AtApD_add_diag!(data, AtApD.D, beta)
  return cholesky!(Symmetric(data, :L))
end

function make_AtApD_add_diag!(data, D::UniformScaling, beta)
  T, idx = eltype(data), diagind(data)
  if !iszero(beta)
    for ii in idx
      data[ii] += T(beta*D.λ)
    end
  end
  data
end

function make_AtApD_add_diag!(data, D::Diagonal, beta)
  T, idx = eltype(data), diagind(data)
  if !iszero(beta)
    for (j, ii) in enumerate(idx)
      data[ii] += T(beta)*D.diag[j]
    end
  end
  data
end

function make_AtApD_add_diag!(data, D, beta) # D is a view of a Diagonal
  T, idx = eltype(data), diagind(data)
  if !iszero(beta)
    for (j, ii) in enumerate(idx)
      data[ii] += T(beta)*D[ii]
    end
  end
  data
end

function form_AtA!(data, A::AbstractMatrix)
  T = eltype(A)
  BLAS.syrk!('L', 'T', one(T), C.A, zero(T), data)
  return data
end

function form_AtA!(data, C::NormalizedMatrix)
  form_AtA!(data, C.A)
end

maybe_normalize!(data, A::AbstractMatrix) = data

function maybe_normalize!(data, C::NormalizedMatrix)
  T = eltype(C)
  BLAS.syr!('L', -T(C.n_obs), C.shift, data)
  ldiv!(Diagonal(C.scale), data)
  rdiv!(data, Diagonal(C.scale))
  return data # AtApD = S⁻¹(AᵀA - nāāᵀ)S⁻¹
end

#
# Simulation
#
abstract type AbstractCorrelation end

struct Exchangeable{T} <: AbstractCorrelation
  ρ::T
end

function get_corr(o::Exchangeable{T}, i, j) where T
  if i == j
    one(T)
  else
    T(o.ρ)
  end
end

struct AutoRegressive{T} <: AbstractCorrelation
  ρ::T
end

function get_corr(o::AutoRegressive{T}, i, j) where T
  if i == j
    one(T)
  else
    o.ρ^abs(i-j)
  end
end

function fill_noise!(rng, Σ::DenseMatrix{T}, m, epsilon, verynoisy) where T
  n = size(Σ, 1)
  ϵ = T(epsilon)
  fill!(Σ, zero(T))
  if ϵ > 0
    if verynoisy
      α = sin.(π/2*rand(rng, n)) .^ 2
      @. α = 2*(α - 0.5)
      U = Matrix{T}(undef, m+1, n)
      @. U[1,:] = α
      @views begin
        randn!(rng, U[2:end,:])
        for (j, u) in enumerate(eachcol(U[2:end,:]))
          normalize!(u)
          @. u = sqrt(1 - abs2(α[j])) * u
        end
      end
    else
      U = randn(rng, T, m, n)
      foreach(normalize!, eachcol(U))
    end
    BLAS.syrk!('U', 'T', ϵ, U, zero(T), Σ)
    @views Σ[diagind(Σ)] .= zero(T)
  end
  return Σ
end

function add_structure!(Σ, corr_spec)
  for j in axes(Σ, 2), i in axes(Σ, 1)
    if i <= j
      Σ[i,j] += get_corr(corr_spec, i, j)
    end
  end
  return Σ
end

function add_structure!(Σ, corr_spec, baxes, k, l)
  for j in axes(Σ, 2), i in axes(Σ, 1)
    # need indices in parent array
    ii = view(baxes, Block(k))[i]
    jj = view(baxes, Block(l))[j]
    Σ[i,j] += get_corr(corr_spec, ii, jj)
  end
  return Σ
end

function simulate_corr_matrix(::Type{T}, corr_spec, n; kwargs...) where T <: Real
  simulate_corr_matrix!(Matrix{T}(undef, n, n), corr_spec; kwargs...)
end

function simulate_corr_matrix!(Σ, corr_spec::AbstractCorrelation;
  m::Integer        = 3,
  epsilon::Real     = 0.0,
  rng::AbstractRNG  = Random.default_rng(),
  verynoisy::Bool   = false,
  kwargs...
)
  @assert size(Σ, 1) == size(Σ, 2)
  validate_parameters(corr_spec, epsilon)

  # Initialize with noise, ϵUᵀU
  fill_noise!(rng, Σ, m, epsilon, verynoisy)

  # Add correlation structure, Σ⁰
  add_structure!(Σ, corr_spec)

  # Now have Σ = Σ⁰ + ϵ(UᵀU - I)
  return Symmetric(Σ, :U)
end

function simulate_group_corr_matrix(::Type{T}, corr_spec::Vector{S}, n, blksizes; kwargs...) where {T <: Real, S <: AbstractCorrelation}
  simulate_group_corr_matrix!(Matrix{T}(undef, n, n), corr_spec, blksizes; kwargs...)
end

function simulate_group_corr_matrix!(Σ, corr_spec::Vector{<:Exchangeable}, blksizes::Vector{<:Integer};
  m::Integer        = 3,
  epsilon::Real     = 0.0,
  delta::Real       = 0.0,
  rng::AbstractRNG  = Random.default_rng(),
  verynoisy::Bool   = false,
  kwargs...
)
  @assert size(Σ, 1) == size(Σ, 2)
  @assert sum(blksizes) == size(Σ, 1)
  validate_parameters(corr_spec, epsilon, delta)

  # Initialize with noise, ϵUᵀU
  fill_noise!(rng, Σ, m, epsilon, verynoisy)

  # Add correlation structure, Σ⁰
  Σblk = BlockedArray(Σ, blksizes, blksizes)
  nblk = length(blksizes)
  baxes = axes(Σblk, 1)
  for i in 1:nblk, j in 1:nblk
    Σ_ij = view(Σblk, Block(i), Block(j))
    if i == j
      add_structure!(Σ_ij, corr_spec[i], baxes, i, i)
    elseif i < j
      delta > 0 && add_structure!(Σ_ij, Exchangeable(delta), baxes, i, j)
    end
  end

  # Now have Σ = Σ⁰ + ϵ(UᵀU - I)
  return Symmetric(Σ, :U)
end

function simulate_group_corr_matrix!(Σ, corr_spec::Vector{<:AutoRegressive}, blksizes::Vector{<:Integer};
  m::Integer        = 3,
  epsilon::Real     = 0.0,
  rng::AbstractRNG  = Random.default_rng(),
  verynoisy::Bool   = false,
  kwargs...
)
  @assert size(Σ, 1) == size(Σ, 2)
  @assert sum(blksizes) == size(Σ, 1)
  validate_parameters(corr_spec, epsilon)

  # Initialize with noise, ϵUᵀU
  fill_noise!(rng, Σ, m, epsilon, verynoisy)

  # Add correlation structure, Σ⁰
  Σblk = BlockedArray(Σ, blksizes, blksizes)
  for i in eachindex(blksizes)
    Σ_ii = view(Σblk, Block(i), Block(i))
    add_structure!(Σ_ii, corr_spec[i])
  end

  # Now have Σ = Σ⁰ + ϵ(UᵀU - I)
  return Symmetric(Σ, :U)
end

function validate_parameters(corr_spec::Exchangeable, ϵ)
  ρ = corr_spec.ρ
  @assert 0 ≤ ρ < 1
  @assert 0 ≤ ϵ < 1-ρ
  return nothing
end

function validate_parameters(corr_spec::AutoRegressive, ϵ)
  ρ = corr_spec.ρ
  @assert 0 ≤ ρ < 1
  @assert 0 ≤ ϵ < (1-ρ)/(1+ρ)
  return nothing
end

function validate_parameters(corr_spec::Vector{<:Exchangeable}, ϵ, δ)
  f = Base.Fix2(getproperty, :ρ)
  ρmin, ρmax = extrema(f(o) for o in corr_spec)
  @assert all(o -> 0 ≤ f(o) < 1, corr_spec)
  @assert 0 ≤ δ < ρmin
  @assert 0 ≤ ϵ < 1-ρmax
  return nothing
end

function validate_parameters(corr_spec::Vector{<:AutoRegressive}, ϵ)
  f = Base.Fix2(getproperty, :ρ)
  ρmax = maximum(f(o) for o in corr_spec)
  @assert all(o -> 0 ≤ f(o) < 1, corr_spec)
  @assert 0 ≤ ϵ < (1-ρmax)/(1+ρmax)
  return nothing
end
