#
# In-place prediction for "stump" classifiers
# See: https://github.com/JuliaAI/DecisionTree.jl/blob/dev/src/classification/main.jl#L290-L297
#
function _predict!(out, clf, features::AbstractMatrix{T}) where T
  @assert length(out) == size(features, 1)
  @inbounds for i in eachindex(out)
    @views out[i] = _predict(clf, features[i, :])
  end
  return out
end

_predict(clf, features::AbstractMatrix{T}) where T = _predict!(Vector{label_type(clf)}(undef, size(features, 1)), clf, features)

_predict(tree::Root, features::AbstractVector) = apply_tree(tree.node, features)
_predict(tree::LeafOrNode, features::AbstractVector) = apply_tree(tree, features)
_predict(ens::Tuple{<:Ensemble,<:AbstractVector}, features::AbstractVector) = apply_adaboost_stumps(ens, features)

label_type(::Root{T,S}) where {T,S} = S
label_type(::LeafOrNode{T,S}) where {T,S} = S
label_type(ens::Tuple{<:Ensemble,<:AbstractVector}) = label_type(ens[1])
label_type(::Ensemble{T,S}) where {T,S} = S

#
# The following is a re-implementation of SAMME from DecisionTree.jl
# See: https://github.com/JuliaAI/DecisionTree.jl/blob/dev/src/classification/main.jl#L677-L712
#
function fit_adaboost(labels::AbstractVector{S}, features::AbstractMatrix{T}, M::Int;
  rng::AbstractRNG=Random.default_rng()
) where {S,T}
  #
  N = length(labels)
  K = length(unique(labels))
  p = size(features, 2)
  δ = 1 - 1 / K
  predictions = Vector{S}(undef, N)
  misclassified = Vector{Bool}(undef, N)

  # Step 1: Initialize weights and allocate storage arrays for model
  w = 1/N * ones(N)
  f = Node{T,S}[]
  θ = T[]
  # Step 2: For each classifier...
  iter = 0
  for j in 1:M
    iter += 1
    # Step 2(a): Generate a classifier
    f_j = build_stump(
      labels, features, w; rng=mk_rng(rng), impurity_importance=false
    )
    _predict!(predictions, f_j, features)
    @. misclassified = labels != predictions

    # Step 2(b): Compute its error and check that it may be a weak classifier
    @views ϵ = sum(w[misclassified]) # / sum(w); w always sums to 1
    if ϵ >= δ continue end    # go back and try again

    # Step 2(c): Compute classifier weight and record the basis element
    θ_j = log((1-ϵ) / ϵ) + log(K-1)
    push!(f, f_j.node)
    push!(θ, θ_j)

    if ϵ < 1e-6 break end # check if we are done

    # Step 2(d): Update data weights
    @. w[misclassified] = w[misclassified] * exp(θ_j)

    # Step 2(e): Normalize the weights
    w .= w / sum(w)
  end

  # Step 3: Aggregate the classifiers
  clf = (Ensemble{T,S}(f, p, T[]), θ)
  _predict!(predictions, clf, features)
  @. misclassified = labels != predictions
  @views train_err = sum(w[misclassified]) / N

  stats = (
    iterations = iter,
    converged = true,
    gnorm = Inf,
    accuracy = 1 - train_err,
    error = train_err,
  )

  return clf, θ, stats
end

function fit_classifier(labels::AbstractVector{S}, features::AbstractMatrix{T}, M::Int;
  rng::AbstractRNG=Random.default_rng(),
  maxiter::Int=100,
  rtol::Real=1e-4,
  update::Symbol=:proj,
  n_train::Int = round(Int, 0.2 * length(labels))
) where {S,T}
  #
  N = length(labels)
  K = length(unique(labels))
  p = size(features, 2)
  w = Vector{T}(undef, N)
  g = Vector{T}(undef, M)
  d = Vector{T}(undef, M)
  H = Matrix{T}(undef, M, M); fill!(H, zero(T))
  predictions = Vector{S}(undef, N)
  misclassified = ones(Bool, N)
  sqrtWZ = Matrix{T}(undef, N, M) # buffer for constructing H
  buffer = Vector{T}(undef, M)
  ix = collect(1:M)
  just_ones = ones(T, M)

  # Step 1: Generate an ensemble of M classifiers and construct Z
  f = Vector{Node{T,S}}(undef, M)
  Z = Matrix{T}(undef, N, M)
  train_idx = collect(1:N)
  for j in axes(Z, 2)
    rand!(rng, w); @. w = exp(w); w .= w / sum(w)
    shuffle!(train_idx)
    @views train_subset = train_idx[1:n_train]
    @views f_j = build_stump(
      labels[train_subset], features[train_subset, :], w[train_subset]; rng=mk_rng(rng), impurity_importance=false
    )
    f[j] = f_j.node
    _predict!(predictions, f_j, features)
    @. misclassified = labels != predictions
    Z[misclassified, j] .= -1 / (K-1)^2
    Z[.!misclassified, j] .= 1 / (K-1)
  end
  @show rank(Z)

  # Step 2: Provide a feasible point for classifier weights
  θ = 1/M * ones(T, M)

  # Step 4: For each iteration...
  iter = 0
  converged = false
  gnorm_prev = Inf
  while !converged && (iter < maxiter)
    iter += 1

    # Step 4(a): Update the weights
    mul!(w, Z, θ)
    @. w = exp.(w)
    w .= w / sum(w)

    # Step 4(b): Compute the negative gradient
    mul!(g, transpose(Z), w, -one(T), zero(T))

    # Step 4(c): Form the approximate Hessian under QUB: Zᵀ*W*Z + ρI
    ρ = dot(g, g) # spectral radius of ∇² - J
    @. w = sqrt(w)
    mul!(sqrtWZ, Diagonal(w), Z)
    BLAS.syrk!('U', 'T', one(T), sqrtWZ, zero(T), H)
    @views H[diagind(H)] .+= ρ

    # Step 4(d): Solve for the search direction and update θ
    H⁻¹ = cholesky!(Symmetric(H, :U))
    ldiv!(d, H⁻¹, g)

    if update == :proj
      @. d = θ + d
      project_simplex!(θ, d, just_ones, buffer, ix)
    elseif update == :fw
      error("not implemented")
    end

    converged = abs(sqrt(ρ) - gnorm_prev) <= rtol # lags behind 1 iteration
    gnorm_prev = sqrt(ρ)
  end

  # Step X: Aggregate the classifiers using the final weights and compute the classification error
  clf = (Ensemble{T,S}(f, p, T[]), θ)
  _predict!(predictions, clf, features)
  @. misclassified = labels != predictions
  @views train_err = sum(w[misclassified]) / N

  stats = (
    iterations = iter,
    converged = converged,
    gnorm = norm(g),
    accuracy = 1 - train_err,
    error = train_err,
  )

  return clf, θ, stats
end

function project_simplex!(x, y, w, z = w .* y, ix = collect(1:length(y)))
  #
  T = eltype(y)
  n = length(y)
  @. x = y
  p = sortperm!(ix, z, rev = true)
  (s, t, lambda) = (zero(T), zero(T), zero(T))
  for i in eachindex(x)
    j = p[i]
    s = s + 1 / w[j]
    t = t + y[j]
    lambda = (t - 1) / s
    if i < n && lambda < z[j] && lambda >= z[p[i]]
      break
    end
  end
  for i in eachindex(x)
    x[i] = max(y[i] - lambda / w[i], zero(T))
  end
  return x
end
