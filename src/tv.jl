#
# LS + Box Constraints
#
"""
Solve the bound-constrained least-squares problem

    minimize  f(β) = 0.5 * ‖X*β - y‖²   subject to   ℓ ≤ β ≤ u

using a projected Newton method whose Hessian is approximated by a
diagonal + low-rank (Lanczos) decomposition, solved via Woodbury.

Arguments
---------
X, y                 : data matrix and response
lower, upper         : scalar or vector bounds (default: unbounded)
maxiter              : maximum outer iterations
gtol                 : ∞-norm tolerance on projected gradient
act_tol              : tolerance for active set detection
eigen_k              : number of Lanczos Ritz pairs to keep
eigen_iters          : Lanczos iterations to compute the pairs
α_init               : initial step size
backtrack            : Armijo backtracking factor (0 < backtrack < 1)
c1                   : Armijo constant
verbose              : print iteration log

Returns
-------
β, niters            : solution and number of iterations taken
"""
function lsq_box_QUB_woodbury!(X::AbstractMatrix{T}, y::AbstractVector{T};
  lower::Union{T,AbstractVector{T}} = -Inf, 
  upper::Union{T,AbstractVector{T}} = Inf,
  method::Symbol                    = :woodbury,  # :woodbury, :nesterov, :proj_newton
  maxiter::Int                      = 200, 
  gtol::Real                        = 1e-6, 
  act_tol::T                        = T(1e-10),
  eigen_k::Int                      = 1, 
  eigen_iters::Int                  = 10,
  α_init::T                         = T(1), 
  backtrack::T                      = T(0.5), 
  c1::T                             = T(1e-4),
  use_nesterov::Bool                = true,
  use_thomas::Bool                  = true,
  verbose::Bool                     = false
) where {T<:AbstractFloat}
  #
  lss = BlockedLS(X, y)
  n, p = size(X)

  # Normalize bounds and initialize β
  ℓ = isa(lower, AbstractVector) ? T.(lower) : fill(T(lower), p)
  u = isa(upper, AbstractVector) ? T.(upper) : fill(T(upper), p)
  @assert all(ℓ .<= u) "lower must be ≤ upper"
  lss.β .= clamp.(lss.β, ℓ, u)
  
  # ---- Lanczos eigendecomposition of X'X ----
  if method == :woodbury
    Afun! = let X = X, tmp_n = lss.storage_n_1
      function(out, v)
        mul!(tmp_n, X, v)               # n-vector
        mul!(out, transpose(X), tmp_n)  # p-vector
        return out
      end
    end
    v1 = randn(T, p); v1 ./= norm(v1)
    λ, V = lanczos_ritz_fixed!(Afun!, v1; maxiter=eigen_iters, k=eigen_k)

    W = Matrix{T}(undef, p, eigen_k)
    @inbounds for j in 1:eigen_k
      sλ = λ[j] <= 0 ? zero(T) : sqrt(λ[j])
      @. W[:, j] = V[:, j] * sλ
    end
    col_sumsq!(lss.storage_p_1, X)
    col_sumsq!(lss.storage_p_2, transpose(W))
    @. lss.storage_p_1 -= lss.storage_p_2
    Diag = copy(lss.storage_p_1)
  elseif method == :QUB
    avgx = vec(mean(X, dims=1))
    devx = vec(std(X, dims=1))
    v_copy = zeros(T, p)

    Afun! = let X = X, tmp_n = lss.storage_n_1, avgx = avgx, devx = devx, v_copy = v_copy, n = n
      function (out, v)
        # w = D^{-1} v
        @. v_copy = v / devx
        mul!(tmp_n, X, v_copy)
        mul!(out, transpose(X), tmp_n)
        @. out -= n * avgx * dot(avgx, v_copy)
        @. out /= devx
        return out
      end
    end
    
    v1 = randn(T, p); v1 ./= norm(v1)
    λmax, V11 = lanczos_ritz_fixed!(Afun!, v1; maxiter=eigen_iters, k=1)
    
    Diag = similar(avgx)
    @. Diag = λmax * devx^2
    
    λ = [T(n)]
    V = avgx
  else
    D = X'X
    D = use_thomas ? Tridiagonal(D) : sparse(D)
  end
      
  # Gradient, buffers, initial values
  y2 = T(0.5) * dot(y, y)
  mul!(lss.storage_n_1, X, lss.β)
  mul!(lss.storage_p_1, transpose(X), lss.storage_n_1)
  lss.grad .= lss.storage_p_1 .- lss.xty
  fβ = T(0.5) * dot(lss.β, lss.storage_p_1) - dot(lss.xty, lss.β) + y2

  β_new  = similar(lss.β)
  β_prev = copy(lss.β)
  t_k    = one(T)
  idxF   = Int[]
  Msmall = Matrix{T}(undef, eigen_k, eigen_k)
  α      = α_init
  gproj  = similar(lss.β)

  proj_grad! = let lss=lss, ℓ=ℓ
    function (work)
      @. work = lss.β - lss.grad
      @. work = clamp(work, ℓ, u)
      @. work = lss.β - work
      work
    end
  end

  niters = maxiter
  for it in 1:maxiter
    lower_active = (lss.β .<= ℓ .+ act_tol) .& (lss.grad .>= 0)
    upper_active = (lss.β .>= u .- act_tol) .& (lss.grad .<= 0)
    free = .!(lower_active .| upper_active)
    empty!(idxF); @inbounds for i in 1:p; free[i] && push!(idxF, i); end

    fill!(lss.direction, zero(T))
    if method != :proj_newton && !isempty(idxF)
      F  = idxF
      VI = @view V[F, :]
      μI = @view Diag[F]
      gI = @view lss.grad[F]
      kI = size(VI, 2)
      woodbury_solve!(@view(lss.direction[F]), VI, @view(λ[1:kI]), μI, gI,
                      @view(lss.storage_p_1[F]), @view(lss.storage_p_2[F]),
                      @view(Msmall[1:kI, 1:kI]))
    elseif method == :proj_newton && !isempty(idxF)
      F  = idxF
      gI = @view lss.grad[F]
      if use_thomas
        DI = @view D[F, F]
        triD = Tridiagonal(DI)
        dI = triD \ gI
      else
        DI = D[F, F]
        DI = @view D[F, F]
        DI = DI
        dI = DI \ gI
      end
      # Store into full direction
      @views lss.direction[F] .= dI
    end

    accepted = false
    for _ in 1:50
      if use_nesterov
        t_k_new = (1 + sqrt(1 + 4 * t_k^2)) / 2
        γ = (t_k - 1) / t_k_new
        t_k = t_k_new
        copyto!(β_new, lss.β)
        BLAS.axpy!(-α, lss.direction, β_new)
        β_new = @. β_new + γ * (β_new - β_prev)  # extrapolation
        copyto!(β_prev, lss.β)
      else
        copyto!(β_new, lss.β)
        BLAS.axpy!(-α, lss.direction, β_new)
      end
      
      @. β_new = clamp(β_new, ℓ, u)
      
      ## main computation cost
      mul!(lss.storage_n_1, X, β_new)
      mul!(lss.storage_p_1, transpose(X), lss.storage_n_1)
      fnew = T(0.5) * norm(lss.storage_n_1 .- lss.y)^2
      
      gd_tot = isempty(idxF) ? zero(T) : dot(@view(lss.grad[idxF]), @view(lss.direction[idxF]))
      decr = c1 * α * max(gd_tot, zero(T))
      if fnew ≤ fβ - decr
        fβ = fnew
        accepted = true
        break
      end
      α *= backtrack
      if α < T(1e-12); break; end
    end
    
    denom = norm(lss.β)
    
    if norm(lss.β - β_new) / denom < gtol && it>1
        niters = it
        copyto!(lss.β, β_new)
        break
    end

    copyto!(lss.β, β_new)
    lss.grad .= lss.storage_p_1 .- lss.xty
    proj_grad!(gproj)
    kkt = norm(gproj, Inf)

    if verbose
        @printf "niters=%3d  f=%.4f  |g_proj|∞=%.2e  α=%.2e  |F|=%d\n" it fβ kkt α length(idxF)
    end
    α = min(α / backtrack, T(1))
  end
  
  if verbose
    kkt = norm(gproj, Inf)
    return lss.β, niters, fβ, kkt, length(idxF)
  else
    return lss.β, niters
  end
end
#
# MM + Thomas Algorithm
#
"""
Solve tridiagonal system A x = b using Thomas algorithm (O(n)).

dl: lower diagonal (length n-1)
d : main  diagonal (length n)
du: upper diagonal (length n-1)
b : right-hand side (length n)

NOTE: d and b will be overwritten (in-place). 
Return x (same array as b).
"""
# Thomas solver (in place on d,b)
function thomas_solve!(dl::AbstractVector{T}, d::AbstractVector{T},
                       du::AbstractVector{T}, b::AbstractVector{T}) where {T<:Real}
  #
  n = length(d)
  @inbounds for i in 1:n-1
      w = dl[i] / d[i]
      d[i+1] -= w * du[i]
      b[i+1] -= w * b[i]
  end
  @inbounds b[n] /= d[n]
  @inbounds for i in (n-1):-1:1
      b[i] = (b[i] - du[i]*b[i+1]) / d[i]
  end
  return b
end

# 0.5*||y-x||^2 + sum w_i * sqrt((Δx)^2 + eps)
function obj_tv(x::AbstractVector{T}, y::AbstractVector{T}, w::AbstractVector{T}; eps::T=T(1e-6)) where {T<:Real}
  Δ = @view(x[2:end]) .- @view(x[1:end-1])
  return T(0.5)*norm(y .- x)^2 + sum(w .* sqrt.(Δ.^2 .+ eps))
end

# surrogate gradient: ∇g(x|x) = (x - y) + D' diag(γ(x)) D x
# γ_i(x) = w_i / sqrt((Δx_i)^2 + eps)
function grad_surrogate!(g::AbstractVector{T}, γ::AbstractVector{T},
                         x::AbstractVector{T}, y::AbstractVector{T}, w::AbstractVector{T};
                         eps::T=T(1e-6)) where {T<:Real}
  #
  n = length(x); @assert length(y)==n && length(w)==n-1 && length(γ)==n-1 && length(g)==n
  # γ from current x
  @inbounds for i in 1:n-1
    Δ = x[i+1] - x[i]
    γ[i] = w[i] / sqrt(Δ*Δ + eps)
  end
  # g = x - y
  @inbounds for i in 1:n
    g[i] = x[i] - y[i]
  end
  # add D'ΓDx
  @inbounds begin
    g[1]   +=  γ[1]    * (x[1] - x[2])
    for j in 2:n-1
      g[j] +=  γ[j]   * (x[j] - x[j+1]) +
                γ[j-1] * (x[j] - x[j-1])
    end
    g[n]   +=  γ[n-1]  * (x[n] - x[n-1])
  end
  return g
end

function prox_wTV_MM(y::AbstractVector{T};
                     w::AbstractVector{T}=ones(eltype(y), length(y)-1),
                     eps::T=T(1e-6), maxit::Int=200, tol::T=T(1e-6),
                     verbose::Bool=true, use_nesterov::Bool=false) where {T<:Real}
  #
  n = length(y); @assert length(w) == n-1
  x    = copy(y)     # current iterate
  xold = similar(x)  # previous iterate
  yk   = copy(x)     # look-ahead point (for Nesterov)

  γ  = similar(w)    # length n-1
  d  = similar(x)    # main diag
  dl = similar(w)    # subdiag
  du = similar(w)    # superdiag
  g  = similar(x)    # grad buffer

  hist = Vector{NamedTuple}(undef, 0)

  # Nesterov params
  tk = one(T)

  if verbose
    println(@sprintf("%4s  %15s  %12s  %10s", "it", "F(x)", "||∇g||₂", "relchg"))
  end

  for it in 1:maxit
    copyto!(xold, x)

    # 用 yk 来算权重 (Nesterov lookahead)
    xref = use_nesterov ? yk : x

    # weights γ from current reference point
    @inbounds for i in 1:n-1
      Δ = xref[i+1] - xref[i]
      γ[i] = w[i] / sqrt(Δ*Δ + eps)
    end

    # assemble A = I + D' diag(γ) D
    d[1]      = one(T) + γ[1]
    @inbounds for i in 2:n-1
      d[i] = one(T) + γ[i-1] + γ[i]
    end
    d[n]      = one(T) + γ[n-1]
    @. dl = -γ
    @. du = -γ

    # solve A x = y
    xnew = thomas_solve!(copy(dl), copy(d), copy(du), copy(y))

    # Nesterov extrapolation
    if use_nesterov
      tnew  = (one(T) + sqrt(one(T) + 4tk*tk)) / 2
      gamma = (tk - one(T)) / tnew
      yk    = xnew .+ gamma .* (xnew .- x)
      tk    = tnew
    end

    relc  = norm(xnew - xold) / max(one(T), norm(xold))
    x .= xnew

    if verbose
      Fx    = obj_tv(x, y, w; eps=eps)
      grad_surrogate!(g, γ, x, y, w; eps=eps)
      gnorm = norm(g)
      push!(hist, (it=it, Fx=Fx, gnorm=gnorm, relchg=relc))
      println(@sprintf("%4d  %15.8e  %12.4e  %10.3e", it, Fx, gnorm, relc))
    end
    
    if relc < tol
      if verbose
        return x, it, Fx, gnorm, n
      else
        return x, it
      end
    end
  end
  if verbose
      return x, maxit, Fx, gnorm, n
  else
      return x, maxit
  end
end

