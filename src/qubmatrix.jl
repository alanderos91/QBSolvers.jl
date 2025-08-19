function with_qub_matrix(f, AtA, lambda, normalize)
  T = eltype(AtA)
  POWER_ITER = 3
  #
  # Diagonal (Plus Rank-1) Hessian
  #
  if normalize == :rescale
    let
      AtA0 = NormalizedGramPlusDiag(AtA)
      u, s, n = AtA0.A.shift, AtA0.A.scale, AtA.n_obs
      @. s = sqrt(n-1) * s
      J = compute_main_diagonal(AtA0.A, AtA0.AtA)
      rho = estimate_spectral_radius(AtA0, J, maxiter=POWER_ITER)
      if lambda > 0
        S = Diagonal(1 ./ (s .^ 2))
        H = Diagonal(T(1 + rho)*I + T(lambda)*S)
        AtApI = GramPlusDiag(AtA0.A, AtA0.AtA, S, AtA0.n_obs, AtA0.n_var, AtA0.tmp, one(T), T(lambda))
        f(AtApI, H)
      else
        f(AtA0, (1+rho)*I)
      end
    end
  elseif normalize == :std
    let
      AtA0 = NormalizedGramPlusDiag(AtA)
      u, s, n = AtA0.A.shift, AtA0.A.scale, AtA.n_obs

      J = (n-1) * compute_main_diagonal(AtA0.A, AtA0.AtA)
      rho = estimate_spectral_radius(AtA0, J, maxiter=POWER_ITER)
      D = T(n-1 + rho)*Diagonal(s .^ 2) + T(lambda)*I
      H = EasyPlusRank1(D, u, one(T), T(n))
      AtApI = GramPlusDiag(AtA, one(T), T(lambda))
      f(AtApI, H)
    end
  elseif normalize == :corr
    let
      AtA0 = NormalizedGramPlusDiag(AtA)
      u, s, n = AtA0.A.shift, AtA0.A.scale, AtA.n_obs
      @. s = sqrt(n-1) * s
      J = compute_main_diagonal(AtA0.A, AtA0.AtA)
      rho = estimate_spectral_radius(AtA0, J, maxiter=POWER_ITER)
      D = T(1 + rho)*Diagonal(s .^ 2) + T(lambda)*I
      H = EasyPlusRank1(D, u, one(T), T(n))
      AtApI = GramPlusDiag(AtA, one(T), T(lambda))
      f(AtApI, H)
    end
  else
    let
      J = compute_main_diagonal(AtA.A, AtA.AtA)
      rho = estimate_spectral_radius(AtA, J, maxiter=POWER_ITER)
      H = J
      @. H.diag = H.diag + rho + lambda
      AtApI = GramPlusDiag(AtA, one(T), T(lambda))
      f(AtApI, H)
    end
  end
end