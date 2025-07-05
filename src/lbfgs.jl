#
# L-BFGS Cache
#
mutable struct LBFGSCache{T}
  memory_size::Int
  s::Matrix{T}  # stores xₙ₊₁ - xₙ
  y::Matrix{T}  # stores gₙ - gₙ₊₁; we define gₙ ≡ -∇ₙ
  ρ::Vector{T}  # stores (yₖᵀsₖ)⁻¹
  α::Vector{T}  # stores ρₖ (sₖᵀgₙ); we define gₙ ≡ -∇ₙ
  q::Vector{T}  # same dimensions as gₙ
  current_size::Int   # not to exceed memory_size
  current_index::Int  # points to column with latest update
end

function LBFGSCache{T}(n_var, memory_size) where T
  s = zeros(T, n_var, memory_size)
  y = zeros(T, n_var, memory_size)
  ρ = zeros(T, memory_size)
  α = zeros(T, memory_size)
  q = zeros(T, n_var)
  return LBFGSCache(memory_size, s, y, ρ, α, q, 0, 0)
end

Base.IteratorSize(::Type{<:LBFGSCache}) = Base.HasLength()
Base.length(cache::LBFGSCache) = cache.current_size
Base.IteratorEltype(::LBFGSCache) = Base.HasEltype()
Base.eltype(::Type{<:LBFGSCache{T}}) where {T} = T
Base.isdone(cache::LBFGSCache{T}, state::Int) where T = state > cache.current_size

@inline function Base.iterate(cache::LBFGSCache{T}, state::Int = 1) where T
  if isdone(cache, state)
    return nothing
  end
  k = mod1(cache.current_index + state, cache.current_size)
  @views begin
    s = cache.s[:, k]
    y = cache.y[:, k]
    ρ = cache.ρ[k]
  end
  return ((k, s, y, ρ), state + 1)
end

Base.isdone(rev::Iterators.Reverse{LBFGSCache{T}}, state::Int) where T = state > rev.itr.current_size

@inline function Base.iterate(rev::Iterators.Reverse{LBFGSCache{T}}, state::Int = 1) where T
  if isdone(rev, state)
    return nothing
  end
  cache = rev.itr
  k = mod1(cache.current_index - state + 1, cache.current_size)
  @views begin
    s = cache.s[:, k]
    y = cache.y[:, k]
    ρ = cache.ρ[k]
  end
  return ((k, s, y, ρ), state + 1)
end

function getlast(cache::LBFGSCache{T}) where T
  k = cache.current_index
  @views begin
    s = cache.s[:, k]
    y = cache.y[:, k]
    ρ = cache.ρ[k]
  end
  return (k, s, y, ρ)
end

function update!(cache::LBFGSCache{T}, alpha, d, g) where T
  k, m, memory_size = cache.current_index, cache.current_size, cache.memory_size
  k = mod1(k+1, memory_size)
  m = min(m + 1, memory_size)
  q = cache.q # assumed to have previous negative gradient
  @views begin
    s = cache.s[:, k]
    y = cache.y[:, k]
  end
  @. s = alpha * d
  @. y = q - g
  cache.ρ[k] = inv(dot(s, y))
  cache.current_index = k
  cache.current_size = m
  return nothing
end

function compute_lbfgs_direction!(d, g, cache::LBFGSCache, D)
  #
  # Check for empty cache
  #
  if cache.current_size == 0
    _descent_direction!(d, g, D)
    return d
  end

  #
  # Apply two-loop recursion
  #
  α, q = cache.α, cache.q
  @. q = -g # ∇ₙ
  
  # αₖ = ρₖ(sₖᵀ∇ₙ) and q = q - αₖyₖ
  for (k, s, y, ρ) in Iterators.reverse(cache)
    α[k] = ρ * dot(s, q)
    @. q = q - α[k] * y
  end

  # d = Hₖq so sign is flipped here
  _descent_direction!(d, q, D)

  # βₖ = ρₖ(yₖᵀd) and d = d + (αₖ-βₖ)sₖ so signs in αₖ and βₖ are flipped here
  for (k, s, y, ρ) in cache
    β = ρ * dot(y, d)
    @. d = d + (α[k] - β) * s
  end
  @. d = -d

  return d
end

_descent_direction!(d, q, ::UniformScaling) = (@. d = q; return nothing)
_descent_direction!(d, q, D) = (ldiv!(d, D, q); return nothing)

function _descent_direction!(d, q, ::UniformScaling, cache::LBFGSCache)
  _, s, y, _ = getlast(cache)
  γ = dot(s, y) / dot(y, y)
  @. d = γ * q
  return nothing
end

_descent_direction!(d, q, D, cache::LBFGSCache) = _descent_direction!(d, q, D)

