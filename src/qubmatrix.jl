function with_qub_matrix(f, AtA, lambda, normalize)
  T = eltype(AtA)
  #
  # Diagonal (Plus Rank-1) Hessian
  #
  if normalize == :rescale
    let
      AtA0 = NormalizedGramPlusDiag(AtA)
      J = compute_main_diagonal(AtA0.A, AtA0.AtA)
      rho = estimate_spectral_radius(AtA0, J, maxiter=3)
      if lambda > 0
        S = Diagonal(@. inv(AtA0.A.scale^2))
        H = Diagonal(similar(S.diag))
        @. H.diag = (1+rho) + T(lambda) * S.diag
        AtApI = GramPlusDiag(AtA0.A, AtA0.AtA, S, AtA0.n_obs, AtA0.n_var, AtA0.tmp, one(T), T(lambda))
        f(AtApI, H)
      else
        f(AtA0, (1+rho)*I)
      end
    end
  elseif normalize == :qub
    let
      AtA0 = NormalizedGramPlusDiag(AtA)
      u, s, n = AtA0.A.shift, AtA0.A.scale, AtA.n_obs
      J = compute_main_diagonal(AtA0.A, AtA0.AtA)
      rho = estimate_spectral_radius(AtA0, J, maxiter=3)
      D = Diagonal(similar(s))
      @. D.diag = T(1+rho)*s^2 + T(lambda)
      H = EasyPlusRank1(D, u, one(T), T(n))
      AtApI = GramPlusDiag(AtA, one(T), T(lambda))
      f(AtApI, H)
    end
  else
    let
      J = compute_main_diagonal(AtA.A, AtA.AtA)
      rho = estimate_spectral_radius(AtA, J, maxiter=3)
      H = J
      @. H.diag = H.diag + rho + lambda
      AtApI = GramPlusDiag(AtA, one(T), T(lambda))
      f(AtApI, H)
    end
  end
end