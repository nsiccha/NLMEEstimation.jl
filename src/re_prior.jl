"""
    NormalREPrior(layout::NLMELayout, ω)

The random-effect prior `η_i ~ N(0, Ω)` with `Ω = omega_matrix(layout, ω)`, with
its Cholesky factor, precision and log-determinant computed once for repeated
per-subject evaluation. This package owns the η prior; models never implement it.
"""
struct NormalREPrior{T <: Real, LT <: LowerTriangular{T}}
    cholesky_factor::LT
    precision::Matrix{T}
    logdet_omega::T
end

function NormalREPrior(layout::NLMELayout, ω::AbstractVector)
    L = omega_cholesky(layout, ω)
    Linv = inv(L)
    P = Matrix(Linv' * Linv)
    precision = (P + P') / 2
    logdet_omega = 2 * sum(log, diag(L))
    return NormalREPrior(L, precision, logdet_omega)
end

neta(prior::NormalREPrior) = size(prior.cholesky_factor, 1)

"""
    re_prior(model, ω) -> NormalREPrior or RaggedNormalREPrior

The η prior of `model` at Ω parameters ω.
"""
re_prior(model, ω::AbstractVector) = _re_prior(nlme_layout(model), ω)
_re_prior(layout::NLMELayout, ω) = NormalREPrior(layout, ω)

"""
    RaggedNormalREPrior(layout, ω)

Shared Gaussian covariance and precision for each random-effect family of a
[`RaggedNLMELayout`](@ref). Repeated blocks use the same family parameters.
"""
struct RaggedNormalREPrior{T}
    layout::RaggedNLMELayout
    cholesky_factors::Vector{LowerTriangular{T,Matrix{T}}}
    precisions::Vector{Matrix{T}}
    logdet_omega::Vector{T}
end

function RaggedNormalREPrior(layout::RaggedNLMELayout, ω::AbstractVector)
    factors = omega_cholesky(layout, ω)
    inverses = inv.(factors)
    precisions = [Matrix((L' * L + (L' * L)') / 2) for L in inverses]
    logdets = [2sum(log, diag(L)) for L in factors]
    return RaggedNormalREPrior(layout, factors, precisions, logdets)
end

_re_prior(layout::RaggedNLMELayout, ω) = RaggedNormalREPrior(layout, ω)

"""
    re_logprior(prior::RaggedNormalREPrior, i, η)

Log density for subject `i`'s compact η vector, summing one Normal density per
listed family block.
"""
function re_logprior(prior::RaggedNormalREPrior, i::Integer, η::AbstractVector)
    _check_length("subject $i η", η, neta(prior.layout, i))
    return _ragged_logprior(prior, _ragged_subject_blocks(prior.layout, i), η)
end

function _ragged_logprior(prior::RaggedNormalREPrior, blocks, η)
    value = zero(promote_type(eltype(η), eltype(prior.logdet_omega)))
    for (family, r) in blocks
        z = prior.cholesky_factors[family] \ view(η, r)
        value -= (length(r) * log(2π) + prior.logdet_omega[family] + dot(z, z)) / 2
    end
    return value
end

function re_logprior_gradient_and_hessian(prior::RaggedNormalREPrior, i::Integer, η::AbstractVector)
    lp = re_logprior(prior, i, η)
    gradient = zeros(promote_type(eltype(η), eltype(prior.logdet_omega)), length(η))
    hessian = zeros(eltype(gradient), length(η), length(η))
    for (family, r) in _ragged_subject_blocks(prior.layout, i)
        P = prior.precisions[family]
        gradient[r] = -(P * view(η, r))
        hessian[r, r] = -P
    end
    return lp, gradient, hessian
end

re_logprior(model, ω::AbstractVector, i::Integer, η::AbstractVector) =
    re_logprior(RaggedNormalREPrior(nlme_layout(model), ω), i, η)

# The η prior of one subject, for the per-subject kernels and gradients: an
# equal-size layout's `NormalREPrior` itself, or one subject's slice of a
# `RaggedNormalREPrior`, whose covariance is block diagonal. The slice works
# block by block, so a subject with thousands of nested effects never forms a
# dense covariance, precision or factor.
struct _RaggedSubjectPrior{T}
    prior::RaggedNormalREPrior{T}
    subject::Int
    blocks::Vector{Pair{Int,UnitRange{Int}}}
end

_subject_prior(prior::NormalREPrior, i) = prior
_subject_prior(prior::RaggedNormalREPrior, i) =
    _RaggedSubjectPrior(prior, Int(i), _ragged_subject_blocks(prior.layout, i))

neta(p::_RaggedSubjectPrior) = isempty(p.blocks) ? 0 : last(last(p.blocks)[2])
function re_logprior(p::_RaggedSubjectPrior, η::AbstractVector)
    _check_length("subject $(p.subject) η", η, neta(p))
    return _ragged_logprior(p.prior, p.blocks, η)
end
_subject_logprior(prior, i, η) = re_logprior(_subject_prior(prior, i), η)

# `f(M_family, η_block)` for every block of the subject, with `M` one matrix per
# family.
function _blockwise(f, p::_RaggedSubjectPrior, matrices, η)
    y = zeros(promote_type(eltype(η), eltype(p.prior.logdet_omega)), length(η))
    for (family, r) in p.blocks
        y[r] = f(matrices[family], view(η, r))
    end
    return y
end

# Ω⁻¹η: minus the prior's η gradient.
_precision_times(p::NormalREPrior, η) = p.precision * η
_precision_times(p::_RaggedSubjectPrior, η) = _blockwise(*, p, p.prior.precisions, η)

# The prior variances of subject i's η, the diagonal of Ω_i.
_factor_row_variances(L) = [sum(abs2(L[j, k]) for k in 1:j) for j in axes(L, 1)]
_prior_variances(prior::NormalREPrior, i) = _factor_row_variances(prior.cholesky_factor)
_prior_variances(prior::RaggedNormalREPrior, i) = reduce(vcat,
    [_factor_row_variances(prior.cholesky_factors[family])
        for (family, _) in _ragged_subject_blocks(prior.layout, i)])

complete_data_loglikelihood(model, i, θ, σ, prior::RaggedNormalREPrior, η) =
    conditional_loglikelihood(model, i, θ, σ, η) + re_logprior(prior, i, η)

function complete_data_loglikelihood_gradient_and_hessian(model, i, θ, σ,
        prior::RaggedNormalREPrior, η)
    ℓ, g, H = conditional_loglikelihood_gradient_and_hessian(model, i, θ, σ, η)
    lp, gp, Hp = re_logprior_gradient_and_hessian(prior, i, η)
    return ℓ, ℓ + lp, g + gp, H + Hp
end

"""
    re_logprior(prior::NormalREPrior, η) -> Real
    re_logprior(model, ω, η) -> Real

The random-effect prior log-density `log N(η | 0, Ω)`.
"""
function re_logprior(prior::NormalREPrior, η::AbstractVector)
    _check_length("η", η, neta(prior))
    z = prior.cholesky_factor \ η
    return -(length(η) * log(2π) + prior.logdet_omega + dot(z, z)) / 2
end
re_logprior(model, ω::AbstractVector, η::AbstractVector) = re_logprior(re_prior(model, ω), η)

"""
    re_logprior_gradient_and_hessian(prior::NormalREPrior, η) -> (lp, ∇η lp, ∇²η lp)

The prior log-density with its gradient `-Ω⁻¹η` and Hessian `-Ω⁻¹` in η, in closed
form (exact).
"""
function re_logprior_gradient_and_hessian(prior::NormalREPrior, η::AbstractVector)
    lp = re_logprior(prior, η)
    return lp, -(prior.precision * η), -prior.precision
end

"""
    complete_data_loglikelihood(model, i, θ, σ, ω_or_prior, η) -> Real

The complete-data log-density `log p(y_i, η_i | θ, σ, Ω) =
conditional_loglikelihood(model, i, θ, σ, η) + re_logprior(prior, η)`. The fifth
argument is either the Ω parameter vector ω or a precomputed [`NormalREPrior`](@ref).
"""
complete_data_loglikelihood(model, i, θ, σ, prior::NormalREPrior, η) =
    conditional_loglikelihood(model, i, θ, σ, η) + re_logprior(prior, η)
complete_data_loglikelihood(model, i, θ, σ, ω::AbstractVector, η) =
    complete_data_loglikelihood(model, i, θ, σ, re_prior(model, ω), η)

"""
    complete_data_loglikelihood_gradient_and_hessian(model, i, θ, σ, ω_or_prior, η)
        -> (ℓ, f, ∇η f, ∇²η f)

The conditional log-density `ℓ`, the complete-data log-density `f`, and the
gradient and Hessian of `f` in η_i. The conditional part comes from
`conditional_loglikelihood_gradient_and_hessian` (AD), the prior part in
closed form.
"""
function complete_data_loglikelihood_gradient_and_hessian(model, i, θ, σ, prior::NormalREPrior, η)
    ℓ, g, H = conditional_loglikelihood_gradient_and_hessian(model, i, θ, σ, η)
    lp, gp, Hp = re_logprior_gradient_and_hessian(prior, η)
    return ℓ, ℓ + lp, g + gp, H + Hp
end
complete_data_loglikelihood_gradient_and_hessian(model, i, θ, σ, ω::AbstractVector, η) =
    complete_data_loglikelihood_gradient_and_hessian(model, i, θ, σ, re_prior(model, ω), η)

# Batched complete-data evaluations for the estimators: per-subject conditional and
# complete-data log-densities, gradients and Hessian blocks of a batch whose η are
# the columns of `H`. `_complete_values` evaluates trial points, so a subject whose
# evaluation throws `_InfeasibleEvaluation` scores -Inf there.
function _complete_values(model, subjects, θ, σ, prior::NormalREPrior, H)
    ℓ = _proposal_loglikelihoods(model, subjects, θ, σ, H)
    return ℓ, ℓ .+ [re_logprior(prior, view(H, :, k)) for k in axes(H, 2)]
end
function _complete_gradients(model, subjects, θ, σ, prior::NormalREPrior, H)
    ℓ, Gc = conditional_loglikelihoods_and_gradients(model, subjects, θ, σ, H)
    f = ℓ .+ [re_logprior(prior, view(H, :, k)) for k in axes(H, 2)]
    return ℓ, f, Gc .- prior.precision * H
end
_complete_hessians(model, subjects, θ, σ, prior::NormalREPrior, H) =
    [B - prior.precision for B in conditional_loglikelihood_hessian_blocks(model, subjects, θ, σ, H)]
