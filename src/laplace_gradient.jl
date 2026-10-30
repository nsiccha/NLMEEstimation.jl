# A covariance step on the model's own Laplace gradient. TMB's `sdreport` takes R
# the same way, from central differences of its gradient. The model keeps its mode
# search, so the package makes none: `2p + 1` calls of the hook, where differences
# of values cost `2p² + 1` marginals, each with its own search.
struct _PreparedLaplaceGradient{E, C}
    subjects::Vector{Int}
    start::E
    constraints::C
end

_marginal_subjects(prepared::_PreparedLaplaceGradient) = prepared.subjects

function _prepare_covariance(approximation::AdaptiveGaussHermite, model, θ, σ, ω, subjects,
        η0, rng, progress)
    approximation.nodes == 1 && provides_laplace_gradient(model) ||
        return _prepare_marginal(approximation, model, θ, σ, ω, subjects, η0, rng, progress)
    _check_approximation(approximation)
    ids = _covariance_subjects(model, subjects)
    return _PreparedLaplaceGradient(ids, _starting_batch(nlme_layout(model), η0, ids),
        _omega_constraints(model))
end

_covariance_progress(::_PreparedLaplaceGradient, method, p) = (2p + 1, "Laplace gradients")

function _covariance_point(prepared::_PreparedLaplaceGradient, model, θ, σ, ω, lp0, progress)
    terms, gθ, gσ, gω = laplace_marginals_and_gradient(model, prepared.subjects, θ, σ, ω,
        prepared.start)
    _check_length("the Laplace gradient in θ", gθ, length(θ))
    _check_length("the Laplace gradient in σ", gσ, length(σ))
    _check_length("the Laplace gradient in ω", gω, length(ω))
    lp, pθ, pσ, pω = population_logprior_and_gradient(model, θ, σ, ω)
    return terms, lp, vcat(gθ .+ pθ, gσ .+ pσ,
        _free_omega_gradient(prepared.constraints, ω, gω .+ pω))
end
