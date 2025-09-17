#
# Struct for representing and solving a mean-variance Markowitz problem.
#
struct MeanVarianceProblem{T}
  μ::Vector{T}
  Σ::Matrix{T}
  r::T
  maxiter::Int
  tol::T
  use_nesterov::Bool
  verbose::Bool
  
  x::Vector{T}
  x_prev::Vector{T}
  x_new::Vector{T}
  grad::Vector{T}
  y::Vector{T}
  buffer::Vector{T}
  scratch::Vector{T}
  scratch2::Vector{Int}
  p1::Vector{T}
  p2::Vector{T}
  
  λ::T
  Λ::Vector{T}
  loss_all::Vector{T}
end

"""
  MeanVarianceProblem(μ, Σ, r; [kwargs])

# Arguments

- `μ`: `Vector` of expected asset returns.
- `Σ`: `Matrix` representing variance-covariance matrix for asset returns.
- `r`: Target return for portfolio.

# Options

- `maxiter`: Maximum number of iterations (default: `5000`).
- `tol`: Convergence tolerance (default: `1e-6`).
- `use_nesterov`: Toggles Nesterov acceleration (default: `false`).
- `verbose`: Display convergence information (default: `false`). 
"""
function MeanVarianceProblem(μ::Vector{T}, Σ::Matrix{T}, r::T;
  maxiter::Int = 5000,
  tol::T = 1e-6,
  use_nesterov::Bool = false,
  verbose::Bool = false) where T
  #
  n = length(μ)
  x = fill(one(T)/n, n)
  x_prev = copy(x)
  x_new = similar(x)
  grad = similar(x)
  y = copy(x)
  buffer = similar(x)
  scratch = similar(x)
  scratch2 = Vector{Int}(undef, size(x)...)
  p1 = zeros(T, n)
  p2 = zeros(T, n)
  
  λ = run_power_method(Σ .- Diagonal(diag(Σ)), maxiter=4)
  Λ = diag(Σ) .+ λ
  
  return MeanVarianceProblem{T}(μ, Σ, r, maxiter, tol, use_nesterov, verbose,
  x, x_prev, x_new,
  grad, y, buffer, scratch, scratch2, p1, p2,
  λ, Λ, T[])
end

function markowitz_dual(
  out1::Vector{Float64},
  y::Vector{Float64},
  μ::Vector{Float64},
  w::Vector{Float64},
  λ::Float64,
  r::Float64,
  scratch::Vector{Float64}, # for weighted simplex projection only
  scratch2::Vector{Int};    # for weighted simplex projection only
  tol=1e-6,
  ρ=1.5
  )
  #
  p = length(μ)
  
  function ψ(σ)
    out1 .= wcondat_s(y .+ σ/(1+λ) .* μ, w, 1.0)
    #         proj_weighted_simplex!(out1, y .+ σ/(1+λ) .* μ, w, scratch, scratch2)
    return r - dot(μ, out1)
  end
  
  σl = 0.0
  σu = ρ
  rl = ψ(σl)
  ru = ψ(σu)
  
  while ru > 0
    σl = σu
    rl = ru
    σu *= ρ
    ru = ψ(σu)
    if σu > 1e6
      error("σ too large — problem may be infeasible")
    end
    
  end
  
  rl = ψ(σl)
  ru = ψ(σu)
  s = 1 - rl / ru
  # Modified secant algorithm (Algorithm 2)
  σ = σu - (σu - σl) / (1 - rl / ru)
  rσ = ψ(σ)
  
  iter = 1
  #     while abs(σu - σl) > tol
  while abs(rσ) > tol && σ ≥ 0
    if rσ < 0
      if s ≤ 2
        σu, ru = σ, rσ
        s = 1 - rl / ru
        σ = σu - (σu - σl) / s
      else
        s = max(ru / rσ - 1, 0.1)
        Δσ = (σu - σ) / s
        σu, ru = σ, rσ
        σ = max(σu - Δσ, 0.6 * σl + 0.4 * σu)
        s = (σu - σl) / (σu - σ)
      end
    else
      if s ≥ 2
        σl, rl = σ, rσ
        s = 1 - rl / ru
        σ = σu - (σu - σl) / s
      else
        s = max(rl / rσ - 1, 0.1)
        Δσ = (σ - σl) / s
        σl = σ
        rl = rσ
        σ = min(σl + Δσ, 0.6 * σu + 0.4 * σl)
        s = (σu - σl) / (σu - σ)
      end
    end
    rσ = ψ(σ)
    iter += 1
    
  end
  σ = max(σ, 0.0)
  
  out1 .= wcondat_s(y .+ σ/(1+λ) .* μ, w, 1.0)
  #     out1 .= proj_weighted_simplex!(out1, y .+ σ/(1+λ) .* μ, w, scratch, scratch2)
  return 
end

"""
  mean_variance_mle(μ, Σ, r; [kwargs])

# Arguments

- `μ`: `Vector` of expected asset returns.
- `Σ`: `Matrix` representing variance-covariance matrix for asset returns.
- `r`: Target return for portfolio.

# Options

- `maxiter`: Maximum number of iterations (default: `5000`).
- `tol`: Convergence tolerance (default: `1e-6`).
- `use_nesterov`: Toggles Nesterov acceleration (default: `false`).
- `verbose`: Display convergence information (default: `false`).
- `standard`: Toggles standardization of Σ as a correlation matrix (default: false).
- `duality`: Iterate via Dykstra's algorithm (`false`) or dual sub-problem (`true`) (default: `false`).
"""
function mean_variance_mle(μ::Vector{T}, Σ::Matrix{T}, r::T;
  maxiter::Int        = 5000,
  tol::T              = 1e-6,
  use_nesterov::Bool  = false,
  verbose::Bool       = false, 
  standard::Bool      = false,
  duality::Bool       = false
  ) where T
  #
  MV = MeanVarianceProblem(μ, Σ, r; maxiter=maxiter, tol=tol, use_nesterov=use_nesterov, verbose=verbose)
  
  t = one(T)
  n = length(μ)
  
  if standard || duality
    d = diag(Σ)
  else
    d = ones(n)
  end
  
  D_sqrt = sqrt.(d)
  D_inv_sqrt = 1.0 ./ D_sqrt
  
  Σ_0 = Diagonal(D_inv_sqrt) * Σ * Diagonal(D_inv_sqrt)
  λ = run_power_method(Σ_0 .- Diagonal(diag(Σ_0)), maxiter=4)
  Λ = diag(Σ_0) .+ λ
  μ_tilde = D_inv_sqrt .* μ
  
  mul!(MV.grad, Σ_0, MV.y)
  loss = dot(MV.grad, MV.y)
  push!(MV.loss_all, loss)
  
  iters = MV.maxiter
  
  ρ = T(100)
  
  for iter in 1:MV.maxiter
    
    @. MV.buffer = MV.y - MV.grad / Λ
    
    if duality
      markowitz_dual(MV.x_new, MV.buffer, μ_tilde, D_inv_sqrt, 
      λ, r, MV.scratch, MV.scratch2; tol=1e-8, ρ=1.5)
    else
      dykstra_proj!(MV.x_new, MV.buffer, μ_tilde, MV.r, MV.p1, 
      MV.p2, MV.grad, D_inv_sqrt, MV.scratch, MV.scratch2, standard;
      tol=MV.tol)
    end
    
    if norm(MV.x_new - MV.x_prev) < MV.tol * length(MV.x_prev)
      MV.verbose && println("Converged at iteration $iter")
      iters = iter
      break
    end
    
    if MV.use_nesterov
      t_next = (one(T) + sqrt(one(T) + 4 * t^2)) / 2
      β = (t - one(T)) / t_next
      @. MV.scratch = MV.x_new + β * (MV.x_new - MV.x_prev)
      
      mul!(MV.grad, Σ_0, MV.scratch)
      loss_nest = dot(MV.grad, MV.scratch)
      
      if loss_nest < loss
        copyto!(MV.y, MV.scratch)
        t = t_next
        loss = loss_nest
      else
        copyto!(MV.y, MV.x_new)
        mul!(MV.grad, Σ_0, MV.y)
        t = one(T)
      end
    else
      copyto!(MV.y, MV.x_new)
      mul!(MV.grad, Σ_0, MV.y)
      loss = dot(MV.grad, MV.x_new)
    end
    push!(MV.loss_all, loss)
    copyto!(MV.x_prev, MV.x_new)
  end
  
  weights = MV.x_prev .* D_inv_sqrt
  port_return = dot(MV.μ, weights)
  port_variance = dot(weights, MV.Σ * weights)
  port_risk = sqrt(port_variance)
  
  return weights, port_variance, port_return, port_risk, iters, MV.loss_all
end

