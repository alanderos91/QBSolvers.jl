function with_qub_matrix(f, AtA, n_obs, n_var, n_blk, var_per_blk, lambda, use_qub, normalize)
  T = eltype(AtA)
  if var_per_blk > 1
    #
    # Block Diagonal Hessian
    #
    if use_qub && normalize
      let
        AtA0 = NormalizedGramPlusDiag(AtA)
        J = compute_block_diagonal(AtA0, n_blk;
          alpha   = one(T),
          beta    = zero(T),
          factor  = false,
          gram    = n_obs > var_per_blk
        )
        rho = estimate_spectral_radius(AtA0, J, maxiter=3)
        S = Diagonal(@. rho*AtA0.A.scale^2 + lambda)
        @. AtA0.A.scale = 1 # need J̃ = √S⋅J⋅√S + ρS + λ = ZᵀZ + ρS + λ
        J̃ = update_factors!(J, AtA0.A, S, one(T), one(T))
        H = BlkDiagPlusRank1(n_obs, n_var, J̃, AtA0.A.shift, one(T), T(n_obs))
        f(H)
      end
    elseif use_qub
      let
        J = compute_block_diagonal(AtA, n_blk;
          alpha   = one(T),
          beta    = zero(T),
          factor  = false,
          gram    = n_obs > var_per_blk
        )
        rho = estimate_spectral_radius(AtA, J, maxiter=3)
        H = update_factors!(J, one(T), lambda + rho)
        f(H)
      end
    else
      let
        H = compute_block_diagonal(AtA, n_blk;
          alpha   = T(n_blk),
          beta    = T(lambda),
          factor  = true,
          gram    = n_obs > var_per_blk
        )
        f(H)
      end
    end
  else
    #
    # Diagonal (Plus Rank-1) Hessian
    #
    if use_qub && normalize
      let
        AtA0 = NormalizedGramPlusDiag(AtA)
        J = compute_main_diagonal(AtA0.A, AtA0.AtA)
        rho = estimate_spectral_radius(AtA0, J, maxiter=3)
        @. J.diag = (1+rho)*AtA0.A.scale^2 + T(lambda)
        H = BlkDiagPlusRank1(n_obs, n_var, J, AtA0.A.shift, one(T), T(n_obs))
        f(H)
      end
    elseif use_qub
      let
        J = compute_main_diagonal(AtA.A, AtA.AtA)
        rho = estimate_spectral_radius(AtA, J, maxiter=3)
        H = J
        @. H.diag = H.diag + rho + lambda
        f(H)
      end
    else
      let
        H = compute_main_diagonal(AtA.A, AtA.AtA)
        @. H.diag = n_blk*H.diag + lambda
        f(H)
      end
    end
  end
end