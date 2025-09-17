#
# (Weighted) Simplex Projection (Condat algorithm(s))
#
"""
    wfilter(data, w, b)

Filter technique for weighted simplex projection
"""
function wfilter(data::Array{Float64, 1}, w::Array{Float64, 1}, b::Real)
  let
    #initialize
    active_list = Int[1]
    s1 = data[1] * w[1]
    s2 = w[1]^2
    pivot = (s1 - b)/(s2)
    wait_list = Int[]
    #check all terms
    for i in 2:length(data)
      #remove inactive terms
      if data[i]/w[i] > pivot
        #update pivot
        pivot = (s1 + data[i] * w[i] - b)/(s2 + w[i]^2)
        if pivot > (data[i] * w[i] - b)/(w[i]^2)
          push!(active_list, i)
          s1 += data[i] * w[i]
          s2 += w[i]^2
        else
          #for large pivot
          append!!(wait_list, active_list)
          active_list = Int[i]
          s1 = data[i] * w[i]
          s2 = w[i]^2
          pivot = (s1 - b)/s2
        end
      end
    end
    #reuse terms from waiting list
    for j in wait_list
      if data[j]/w[j] >pivot
        push!(active_list, j)
        s1 += data[j]*w[j]
        s2 += w[j]^2
        pivot = (s1 - b)/s2
      end
    end
    return active_list, s1, s2
  end
end

"""
    wcheckL(active_list, s1, s2, data, w, b)

Remaining algorithm (after Filter) of weighted simplex projection based on Condat's method
"""
function wcheckL(active_list::Array{Int, 1}, s1::Float64, s2::Float64, data::Array{Float64, 1}, w::Array{Float64, 1}, b::Real)::AbstractVector
  let
    pivot = (s1 - b)/s2
    while true
      length_cache = length(active_list)
      for _ in 1:length_cache
        i = popfirst!(active_list)
        if data[i]/w[i] > pivot
          push!(active_list, i)
        else
          s1 = s1 - data[i]*w[i]
          s2 = s2 - w[i]^2
          pivot = (s1 - b)/s2
        end
      end
      if length_cache == length(active_list)
        break
      end
    end
    
    value_list = Float64[]
    for j in active_list
      push!(value_list, data[j] - w[j]*pivot)
    end
    return sparsevec(active_list, value_list, length(data))
  end
end

"""
    parallel_wfilter(data, w, b, numthread)

Parallel filter technique for weighted simplex projection
"""
function parallel_wfilter(data::Array{Float64, 1}, w::Array{Float64, 1}, b::Real, numthread::Int)
  #the length for subvectors
  width = floor(Int, length(data)/numthread)
  #lock global value
  spl = SpinLock()
  #initialize a global list
  glist = Int[]
  gs1 = 0.0
  gs2 = 0.0
  @threads for id in 1:numthread
    let
      #determine start and end position for subvectors
      local st = (id-1) * width + 1
      if id == numthread
        local en = length(data)
      else
        local en = id * width
      end
      local active_list = Int[st]
      local s1 = data[st] * w[st]
      local s2 = w[st]^2
      local pivot = (s1 - b)/(s2)
      local wait_list = Int[]
      #check all terms
      for i in (st+1):en
        #remove inactive terms
        if data[i]/w[i] > pivot
          #update pivot
          pivot = (s1 + data[i] * w[i] - b)/(s2 + w[i]^2)
          if pivot > (data[i] * w[i] - b)/(w[i]^2)
            push!(active_list, i)
            s1 += data[i] * w[i]
            s2 += w[i]^2
          else
            #for large pivot
            append!!(wait_list, active_list)
            active_list = Int[i]
            s1 = data[i] * w[i]
            s2 = w[i]^2
            pivot = (s1 - b)/s2
          end
        end
      end
      #reuse terms from waiting list
      for j in wait_list
        if data[j]/w[j] >pivot
          push!(active_list, j)
          s1 += data[j]*w[j]
          s2 += w[j]^2
          pivot = (s1 - b)/s2
        end
      end
      while true
        length_cache = length(active_list)
        for _ in 1:length_cache
          i = popfirst!(active_list)
          if data[i]/w[i] > pivot
            push!(active_list, i)
          else
            s1 = s1 - data[i]*w[i]
            s2 = s2 - w[i]^2
            pivot = (s1 - b)/s2
          end
        end
        if length_cache == length(active_list)
          break
        end
      end
      #reduce with locking
      lock(spl)
      append!!(glist, active_list)
      gs1 += s1
      gs2 += s2
      unlock(spl)
    end
  end
  return glist, gs1, gs2
end

"""
    wcondat_s(data, w, b)

Weighted simplex projection based on serial Condat's method
"""
function wcondat_s(data::Array{Float64, 1}, w::Array{Float64, 1}, b::Real)::AbstractVector
  active_list, s1, s2 = wfilter(data, w, b)
  return wcheckL(active_list, s1, s2, data, w, b)
end

"""
    wcondat_s(data, w, b, numthread)

Weighted simplex projection based on parallel Condat's method
"""
function wcondat_p(data::Array{Float64, 1}, w::Array{Float64, 1}, b::Real, numthread)::AbstractVector
  active_list, s1, s2 = parallel_wfilter(data, w, b, numthread)
  return wcheckL(active_list, s1, s2, data, w, b)
end

# using Condat's method (2016)
function proj_simplex!(out::Vector{T}, y::Vector{T}, scratch::Vector{T}) where T
  @assert length(out) == length(y)
  @assert length(scratch) >= length(y)
  
  N = length(y)
  v = view(scratch, 1:N)
  vtilde = view(out, 1:N)
  v_len = 1
  vtilde_len = 0
  
  v[1] = y[1]
  ρ = y[1] - T(1)
  
  # Step 2: iterate
  for n in 2:N
    yₙ = y[n]
    if yₙ > ρ
      ρ = ρ + (yₙ - ρ) / (v_len + 1)
      if ρ > yₙ - T(1)
        v[v_len += 1] = yₙ
      else
        @inbounds for i in 1:v_len
          vtilde[vtilde_len += 1] = v[i]
        end
        v[1] = yₙ
        v_len = 1
        ρ = yₙ - T(1)
      end
    end
  end
  
  # Step 3: re-add vtilde
  @inbounds for i in 1:vtilde_len
    yₖ = vtilde[i]
    if yₖ > ρ
      v[v_len += 1] = yₖ
      ρ += (yₖ - ρ) / v_len
    end
  end
  
  # Step 4: prune
  changed = true
  while changed
    changed = false
    i = 1
    while i <= v_len
      yₖ = v[i]
      if yₖ <= ρ
        v[i] = v[v_len]
        v_len -= 1
        ρ += (ρ - yₖ) / max(1, v_len)
        changed = true
      else
        i += 1
      end
    end
  end
  
  τ = ρ
  
  @inbounds @simd for i in 1:N
    out[i] = max(y[i] - τ, 0)
  end
  return out
end

# # weighted simplex projection
# function proj_weighted_simplex!(out::Vector{T}, v::Vector{T}, w::Vector{T}, scratch::Vector{T}, scratch2::Vector{Int}) where T
#   @assert length(out) == length(v) == length(w) == length(scratch)
#   n = length(v)
  
#   @. scratch = v / w
#   sortperm!(scratch2, scratch, rev=true) #sortperm!
#   perm = scratch2
  
#   num = zero(T)
#   denom = zero(T)
#   θ = zero(T)
#   ρ = 0
  
#   for k in 1:n
#     i = perm[k]
#     num += w[i] * v[i]
#     denom += w[i]^2
#     θ = (num - one(T)) / denom
#     if v[i] - θ * w[i] > 0
#       ρ = k
#     else
#       break
#     end
#   end
  
#   # final θ from active set
#   num = zero(T)
#   denom = zero(T)
#   @inbounds for k in 1:ρ
#     i = perm[k]
#     num += w[i] * v[i]
#     denom += w[i]^2
#   end
#   θ = (num - one(T)) / denom
  
#   @inbounds @simd for i in 1:n
#     out[i] = max(v[i] - θ * w[i], zero(T))
#   end
  
#   return out
# end

#
# Halfspace xᵀμ ≥ r
#
function proj_mean!(out::Vector{T}, x::Vector{T}, μ::Vector{T}, r::T) where T
  dotval = dot(μ, x)
  if dotval ≥ r
    copyto!(out, x)
  else
    α = (r - dotval) / dot(μ, μ)
    @. out = x + α * μ
  end
  return out
end

#
# Dykstra: Halfspace ∩ (Weighted) Simplex
#
function dykstra_proj!(
  out::Vector{T}, x0::Vector{T}, μ::Vector{T}, r::T,
  p1::Vector{T}, p2::Vector{T}, buffer::Vector{T}, w::Vector{T},
  scratch::Vector{T}, scratch2::Vector{Int}, standard::Bool = false;
  max_iter::Int = 100, tol::T = 1e-10
  ) where T
  copyto!(out, x0)
  fill!(p1, 0)
  fill!(p2, 0)
  x_old = similar(out)
  
  for iter in 1:max_iter
    copyto!(x_old, out)
    
    @. buffer = out + p1
    proj_mean!(buffer, buffer, μ, r)
    @. p1 += out - buffer
    
    @. out = buffer + p2
    if standard
      out .= wcondat_s(out, w, 1.0)
      #             proj_weighted_simplex!(out, out, w, scratch, scratch2)
    else
      proj_simplex!(out, out, scratch)
    end
    @. p2 += buffer - out
    
    if norm(out - x_old) < tol
      break
    end
  end
  
  return out
end

#
# Soft-Thresholding & ℓ1-ball Projection
#

"""
    soft_threshold!(dest, v, τ)

Scalar soft-threshold: `dest[i] = sign(v[i])*max(|v[i]| - τ, 0)`.
`τ` must be a scalar. No allocations.
"""
function soft_threshold!(dest::AbstractVector{T}, v::AbstractVector{T}, τ::T) where {T<:Real}
    @inbounds for i in eachindex(v)
        vi = v[i]
        avi = abs(vi)
        dest[i] = avi > τ ? copysign(avi - τ, vi) : zero(T)
    end
    return dest
end

"""
    _l1_proj_threshold(z, t) -> τ

Return the soft-threshold τ for projecting `z` onto the ℓ1-ball {x : ||x||₁ ≤ t}.
Guaranteed to return a finite scalar; never returns `nothing`.

No-sorting Condat-style implementation:
find τ ∈ [0, maximum(abs.(z))] such that sum(max(abs(z) .- τ, 0)) = t
via adaptive bisection (numerically exact up to floating-point precision).
"""
function _l1_proj_threshold(z::AbstractVector{T}, t::T) where {T<:Real}
    # promote to a floating type for robust arithmetic
    TF = float(T)
    if !(isfinite(t)) || t < 0
        throw(ArgumentError("t must be finite and ≥ 0, got t = $t"))
    end
    n = length(z)
    n == 0 && return zero(TF)

    # quick pass: ||z||₁ and max|z|
    s1  = zero(TF)
    zmx = zero(TF)
    @inbounds @simd for i in eachindex(z)
        ai = abs(TF(z[i]))
        s1 += ai
        if ai > zmx
            zmx = ai
        end
    end

    # already inside the ball → τ = 0
    if s1 ≤ TF(t)
        return zero(TF)
    end
    # trivial cases: radius 0 or all zeros → τ = max|z|
    if t == zero(T) || zmx == zero(TF)
        return zmx
    end

    # bisection for τ on [0, zmx]
    lo  = zero(TF)
    hi  = zmx
    tol = eps(TF)

    @inbounds while hi - lo > max(tol, eps(TF) * (one(TF) + hi))
        τ = (lo + hi) / 2
        s = zero(TF)
        @simd for i in eachindex(z)
            d = abs(TF(z[i])) - τ
            s += ifelse(d > 0, d, zero(TF))
        end
        if s > TF(t)
            lo = τ        # τ too small → increase
        else
            hi = τ        # τ too large → decrease
        end
        # early exit if equality is already matched tightly
        if abs(s - TF(t)) ≤ max(tol, eps(TF) * max(s, TF(t), one(TF)))
            hi = τ
            break
        end
    end

    return hi  # τ
end

"""
    proj_l1_ball!(dest, z, t; tol=eps(eltype(dest)))

Project vector `z` onto the ℓ₁-ball { x : ||x||₁ ≤ t }.
This version delegates the threshold search to `_l1_proj_threshold(z, t)`,
then performs an in-place soft-thresholding into `dest`.

- Returns `dest`.
- If `||z||₁ ≤ t`, `_l1_proj_threshold` returns 0, and we simply copy `z`.
"""
function proj_l1_ball!(
    dest::AbstractVector{T}, z::AbstractVector{<:Real}, t::Real; tol::T = eps(T)
) where {T<:AbstractFloat}
    @assert length(dest) == length(z)

    # Get the global soft-threshold τ for the l1 projection
    τF = _l1_proj_threshold(z, t)    # may be Float32/Float64 depending on z,t
    τ  = T(τF)                       # convert to dest's element type

    # Fast path: τ == 0 ⇒ already inside the ball → copy
    if τ == zero(T)
        @inbounds dest .= T.(z)
        return dest
    end

    # Soft-threshold with τ (with a small tolerance for ties/rounding)
    thr = τ + tol
    @inbounds @simd for i in eachindex(z, dest)
        xi  = T(z[i])
        axi = abs(xi)
        dest[i] = axi > thr ? copysign(axi - τ, xi) : zero(T)
    end
    return dest
end

"""
    proj_l1_ball(x, t) -> y

Functional ℓ1-ball projection. Allocates a new dense vector `y`.
"""
function proj_l1_ball(x::AbstractVector{<:Real}, t::Real)
    s1 = sum(abs, x)
    if s1 ≤ t
        return copy(x)
    end
    τ1 = _l1_proj_threshold(x, t)
    return sign.(x) .* max.(abs.(x) .- τ1, 0)
end

"""
    proj_l1_ball_sparse(x, t) -> SparseVector

Functional ℓ1-ball projection. Returns a `SparseVector`
without creating a dense result first.
"""
function proj_l1_ball_sparse(x::AbstractVector{T}, t::Real) where {T<:Real}
  s1 = sum(abs, x)
  if s1 ≤ t
    return sparsevec(x)  # drop zeros automatically
  end
  τ = _l1_proj_threshold(x, T(t))
  
  n = length(x)
  # Count nonzeros after shrink
  nnz = count(i -> abs(x[i]) > τ, 1:n)
  I = Vector{Int}(undef, nnz)
  V = Vector{T}(undef, nnz)
  
  k = 0
  @inbounds for i in 1:n
    xi = x[i]; axi = abs(xi)
    if axi > τ
      k += 1
      I[k] = i
      V[k] = copysign(axi - τ, xi)
    end
  end
  return SparseVector(n, I[1:k], V[1:k])
end


