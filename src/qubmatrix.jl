function with_qub_matrix(f, AtA, n_obs, n_var, n_blk, var_per_blk, lambda, use_qub, normalize)
  global PkgTimer
  T = eltype(AtA)
  if var_per_blk > 1
    #
    # Block Diagonal Hessian
    #
    if use_qub && normalize
      let
        @timeit PkgTimer "BlkDiag, normalize = true" begin
          @timeit PkgTimer "normalize" AtA0 = NormalizedGramPlusDiag(AtA)
          @timeit PkgTimer "init" begin
            J = compute_block_diagonal(AtA0, n_blk;
              alpha   = one(T),
              beta    = zero(T),
              factor  = false,
              gram    = n_obs > var_per_blk
            )
          end
          @timeit PkgTimer "spectral radius" rho = estimate_spectral_radius(AtA0, J, maxiter=3)
          @timeit PkgTimer "update factors" begin
            S = Diagonal(@. rho*AtA0.A.scale^2 + lambda)
            @. AtA0.A.scale = 1 # need J̃ = √S⋅J⋅√S + ρS + λ = ZᵀZ + ρS + λ
            J̃ = update_factors!(J, AtA0.A, S, one(T), one(T))
            H = BlkDiagPlusRank1(n_obs, n_var, J̃, AtA0.A.shift, one(T), T(n_obs))
          end
        end
        f(H)
      end
    elseif use_qub
      let
        @timeit PkgTimer "BlkDiag, normalize = false" begin
          @timeit PkgTimer "init" begin
            J = compute_block_diagonal(AtA, n_blk;
              alpha   = one(T),
              beta    = zero(T),
              factor  = false,
              gram    = n_obs > var_per_blk
            )
          end
          @timeit PkgTimer "spectral radius" rho = estimate_spectral_radius(AtA, J, maxiter=3)
          @timeit PkgTimer "update factors" H = update_factors!(J, one(T), lambda + rho)
        end
        f(H)
      end
    else
      let
        @timeit PkgTimer "BlkDiag, Jensen" begin
          H = compute_block_diagonal(AtA, n_blk;
            alpha   = T(n_blk),
            beta    = T(lambda),
            factor  = true,
            gram    = n_obs > var_per_blk
          )
        end
        f(H)
      end
    end
  else
    #
    # Diagonal (Plus Rank-1) Hessian
    #
    if use_qub && normalize
      let
        @timeit PkgTimer "Diag, normalize = true" begin
          @timeit PkgTimer "normalize" AtA0 = NormalizedGramPlusDiag(AtA)
          @timeit PkgTimer "init" J = compute_main_diagonal(AtA0.A, AtA0.AtA)
          @timeit PkgTimer "spectral radius" rho = estimate_spectral_radius(AtA0, J, maxiter=3)
          @timeit PkgTimer "update factors" begin
            @. J.diag = (1+rho)*AtA0.A.scale^2 + T(lambda)
            H = BlkDiagPlusRank1(n_obs, n_var, J, AtA0.A.shift, one(T), T(n_obs))
          end
        end
        f(H)
      end
    elseif use_qub
      let
        @timeit PkgTimer "Diag, normalize = false" begin
          @timeit PkgTimer "init" J = compute_main_diagonal(AtA.A, AtA.AtA)
          @timeit PkgTimer "spectral radius" rho = estimate_spectral_radius(AtA, J, maxiter=3)
          H = J
          @. H.diag = H.diag + rho + lambda
        end
        f(H)
      end
    else
      let
        @timeit PkgTimer "Diag, Jensen" begin
          H = compute_main_diagonal(AtA.A, AtA.AtA)
          @. H.diag = n_blk*H.diag + lambda
        end
        f(H)
      end
    end
  end
end

