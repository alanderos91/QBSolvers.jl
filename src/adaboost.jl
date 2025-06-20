#
# In-place prediction for "stump" classifiers
# See: https://github.com/JuliaAI/DecisionTree.jl/blob/dev/src/classification/main.jl#L290-L297
#
_predict!(out::AbstractVector{S}, tree::Root{T,S}, features::AbstractMatrix{T}) where {T,S} = _predict!(out, tree.node, features)

function _predict!(out::AbstractVector{S}, tree::LeafOrNode{T,S}, features::AbstractMatrix{T}) where {T,S}
  @assert length(out) == size(features, 1)
  @inbounds for i in eachindex(out)
    @views out[i] = apply_tree(tree, features[i, :])
  end
  return out
end

_predict(tree::Root{T,S}, features::AbstractMatrix{T}) where {T,S} = _predict(tree.node, features)
_predict(tree::LeafOrNode{T,S}, features::AbstractMatrix{T}) where {T,S} = _predict!(Vector{S}(undef, size(features, 1)), tree, features)

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
  for j in 1:M
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
  return (Ensemble{T,S}(f, p, T[]), θ)
end
