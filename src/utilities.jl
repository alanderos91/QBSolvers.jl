#
# Dominant eigenvalue of AtA - D, where D is block diagonal Hessian.
#
function estimate_dominant_eigval(AtA, D; kwargs...)
  lambda_max, _, ch = powm!(GramMinusBlkDiag(AtA, D), ones(size(D, 1)); log=true, kwargs...)
  # @show ch
  return lambda_max
end
