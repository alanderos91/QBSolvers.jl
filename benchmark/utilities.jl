function make_Σ(corrtype, p, use_noise, ngroups)
  dim_noise = 8     # dimension for noise distribution
  var_per_blk, r = divrem(p, ngroups)
  blksizes = repeat([var_per_blk], ngroups)
  verynoisy = true

  if r > 0 blksizes[1:r] .+= 1 end

  if corrtype == "indep"
    # Σ = I + ϵ(UᵀU - I)
    ρmax = 0.0
    ϵ = use_noise ? 5e-2 : 0.0
    simulate_corr_matrix(Float64, Exchangeable(ρmax), p;
      m = dim_noise,
      epsilon = ϵ,
      verynoisy = verynoisy,
    )
  elseif corrtype == "diag"
    # Σ = C + ϵ(UᵀU - I); C[i,i] = 1, C[i,j] = 1/(p+1)
    ρmax = 1/(p+1)
    ϵ = use_noise ? ρmax*p/(p+1) : 0.0
    simulate_corr_matrix(Float64, Exchangeable(ρmax), p;
      m = dim_noise,
      epsilon = ϵ,
      verynoisy = verynoisy
    )
  elseif corrtype == "blkdiag"
    # Σ = [C]ₖ + ϵ(UᵀU - I); block diagonal [C]ₖ, [C]ₖ[i,i] = 1, [C]ₖ[i,j] = 0.3
    ρmax = 0.3
    ϵ = use_noise ? ρmax*p/(p+1) : 0.0
    simulate_group_corr_matrix(Float64, [Exchangeable(ρmax) for _ in 1:ngroups], p, blksizes;
      m = dim_noise,
      epsilon = ϵ,
      verynoisy = verynoisy
    )
  elseif corrtype == "band"
    # Σ = C + ϵ(UᵀU - I); C[i,i] = 1, C Toeplitz
    κ = 1e2
    ρmax = (√κ - 1) / (√κ + 1)
    ρmax = max(ρmax, -ρmax)
    ϵ = use_noise ? ρmax*p/(p+1) : 0.0
    simulate_corr_matrix(Float64, AutoRegressive(ρmax), p;
      m = dim_noise,
      epsilon = ϵ,
      verynoisy = verynoisy
    )
  elseif corrtype == "blkband"
    # Σ = [C]ₖ + ϵ(UᵀU - I); block diagonal [C]ₖ Toeplitz
    κ = 1e2
    ρmax = (√κ - 1) / (√κ + 1)
    ρmax = max(ρmax, -ρmax)
    ϵ = use_noise ? ρmax*p/(p+1) : 0.0
    simulate_group_corr_matrix(Float64, [AutoRegressive(ρmax) for _ in 1:ngroups], p, blksizes;
      m = dim_noise,
      epsilon = ϵ,
      verynoisy = verynoisy
    )
  else
    error("Unknown option $(corrtype)")
  end
end
