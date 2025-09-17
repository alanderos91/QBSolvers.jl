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

