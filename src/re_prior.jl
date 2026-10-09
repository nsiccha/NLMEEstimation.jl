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
    re_prior(model, ω) -> NormalREPrior

The η prior of `model` at Ω parameters ω.
"""
re_prior(model, ω::AbstractVector) = NormalREPrior(nlme_layout(model), ω)

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
[`conditional_loglikelihood_gradient_and_hessian`](@ref) (AD), the prior part in
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
# the columns of `H`.
function _complete_values(model, subjects, θ, σ, prior::NormalREPrior, H)
    ℓ = conditional_loglikelihoods(model, subjects, θ, σ, H)
    return ℓ, ℓ .+ [re_logprior(prior, view(H, :, k)) for k in axes(H, 2)]
end
function _complete_gradients(model, subjects, θ, σ, prior::NormalREPrior, H)
    ℓ, Gc = conditional_loglikelihoods_and_gradients(model, subjects, θ, σ, H)
    f = ℓ .+ [re_logprior(prior, view(H, :, k)) for k in axes(H, 2)]
    return ℓ, f, Gc .- prior.precision * H
end
_complete_hessians(model, subjects, θ, σ, prior::NormalREPrior, H) =
    [B - prior.precision for B in conditional_loglikelihood_hessian_blocks(model, subjects, θ, σ, H)]
