
#
# Two Projection Method
#
function NQUB_nqp_TwoMat(P::AbstractMatrix{T}, q::AbstractVector{T};
  λridge::T                     = 0.0,
  maxiter::Integer              = 100,
  ∇tol::AbstractFloat           = 1e-3,
  correlation_eigenvalue::Bool  = true,
  sparse_parameter::Bool        = true,
  nonnegative::Bool             = false,
  ) where T
  #
  n, n = size(P)
  β = sparse_parameter ? spzeros(T, n) : zeros(T, n)
  β_new = similar(β); 
  β_prev = similar(β)
  
  eigen_num = 1
  eigen_num_max = 5
  
  v₁ = randn(n)
  λ1, V1 = lanczos_ritz_fixed!(P, v₁; maxiter=eigen_num_max, k=eigen_num)
  
  A1 = similar(P)
  W = similar(V1)
  
  @inbounds for j in axes(V1, 2), i in axes(V1, 1)
    λ = λ1[j]
    W[i, j] = λ < 0 ? 0.0 : V1[i, j] * sqrt(λ)
  end
  copyto!(A1, P)
  BLAS.syrk!('U', 'N', -1.0, W, 1.0, A1)
  LinearAlgebra.copytri!(A1, 'U')
  
  if correlation_eigenvalue
    diagA1 = diag(A1)
    SA1 = Diagonal(sqrt.(1 ./diagA1)) * A1 * Diagonal(sqrt.(1 ./diagA1)) - 1.0*I
    λmax = run_power_method(SA1, maxiter = 4)
    cblk = @. (1 + λmax) * diagA1 + λridge
  else
    diagA1 = diag(A1)
    A1 -= Diagonal(diagA1)
    λmax = run_power_method(A1, maxiter = 4)
    cblk = @. diagA1 + λmax + λridge
  end
  
  
  α = one(T)
  storage_p_1 = similar(q)
  storage_p_2 = similar(q)
  storage_p_3 = similar(q)
  fill!(β, zero(T))
  copyto!(β_new, β)
  mul!(storage_p_2, P, β_new)
  
  niters = maxiter
  
  @inbounds for iter in 1:maxiter
    
    copyto!(storage_p_3, storage_p_2)
    
    copyto!(storage_p_1, -q)
    @. storage_p_2 += storage_p_1
    active_indices = nonnegative ? active_set(β, storage_p_2) : Base.OneTo(p)
    V_I = V1[active_indices, :]
    μ1 = cblk[active_indices]
    u1 = storage_p_2[active_indices]
    copyto!(storage_p_1, storage_p_2)
    view(storage_p_1, active_indices) .= woodbury_solve(V_I, λ1, μ1, u1)
    
    f1 = dot(β, storage_p_3)/2 - dot(q, β)
    
    f2 = f1 + 1
    
    while f1 < f2
      @. β_new = max(β - α * storage_p_1, 0)
      
      β_new = map(x -> abs(x) < 1e-8 ? 0.0 : x, β_new)
      sparse_parameter && (β_new = sparse(β_new))
      mul!(storage_p_2, P, β_new)
      f2 = dot(β_new, storage_p_2)/2 - dot(q, β_new)
      if f1 < f2
        α *= 0.5
      end
      
    end
    
    if norm(β - β_new)/norm(β) < ∇tol
      niters = iter
      break
    end
    
    copyto!(β, β_new)
  end
  
  return β, niters
end

