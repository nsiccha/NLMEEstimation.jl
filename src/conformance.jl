"""
    ProtocolViolation(model_type, message)

Thrown by [`check_protocol`](@ref) when a model breaks the protocol contract.
"""
struct ProtocolViolation <: Exception
    model_type::Type
    message::String
end
Base.showerror(io::IO, e::ProtocolViolation) =
    print(io, "ProtocolViolation for ", e.model_type, ": ", e.message)

"""
    check_protocol(model, θ, σ, ω; subjects = 1:nsubjects(model), η = nothing,
                   reference_backend = nothing, population_gradient = false,
                   rtol = 1e-8) -> true

Check that `model` implements the protocol consistently at the given parameters,
and throw a [`ProtocolViolation`](@ref) naming the first violation otherwise.

Checked for each subject in `subjects`, at `η` (zeros by default; a vector, or one
vector per checked subject):

- the layout matches the lengths of θ, σ and ω;
- the conditional log-density is a finite real;
- the values returned by the derivative methods agree with it, the gradients
  agree with each other, and the Hessian is square and symmetric;
- a declared [`mu_referencing`](@ref) trait has one design per subject of the right
  size, and the density is invariant under the mu-referencing shift;
- a declared [`observation_model`](@ref) reproduces the conditional log-density.

With `reference_backend` (an AD backend, never finite differences), the model's η
gradient and Hessian are also compared against that backend. With
`population_gradient = true`, the optional θ/σ gradient is checked as well (against
`reference_backend` when given).
"""
function check_protocol(model, θ::AbstractVector, σ::AbstractVector, ω::AbstractVector;
        subjects = nothing, η = nothing, reference_backend = nothing,
        population_gradient::Bool = false, rtol::Real = 1.0e-8)
    violation(msg) = throw(ProtocolViolation(typeof(model), msg))
    n = nsubjects(model)
    n isa Integer && n >= 1 || violation("nsubjects returned $(repr(n)); expected a positive integer")
    layout = nlme_layout(model)
    layout isa NLMELayout || violation("nlme_layout returned a $(typeof(layout)); expected an NLMELayout")
    length(θ) == layout.ntheta || violation("θ has length $(length(θ)), the layout declares ntheta = $(layout.ntheta)")
    length(σ) == layout.nsigma || violation("σ has length $(length(σ)), the layout declares nsigma = $(layout.nsigma)")
    length(ω) == nomega(layout) || violation("ω has length $(length(ω)), the layout requires nomega = $(nomega(layout))")
    subjects = subjects === nothing ? (1:n) : subjects
    all(i -> 1 <= i <= n, subjects) || violation("subjects $subjects are outside 1:$n")
    m = neta(layout)
    mr = mu_referencing(model)
    _check_mu_layout(violation, mr, n, m)
    reference = reference_backend === nothing ? nothing :
        _checked_backend(model, reference_backend, :check_protocol)
    agrees(a, b) = isapprox(a, b; rtol = rtol, atol = rtol * (1 + maximum(abs, a; init = 0.0)))
    for (k, i) in enumerate(subjects)
        ηi = _subject_eta(η, k, m)
        length(ηi) == m || violation("η for subject $i has length $(length(ηi)); neta = $m")
        ℓ = conditional_loglikelihood(model, i, θ, σ, ηi)
        ℓ isa Real && isfinite(ℓ) ||
            violation("conditional_loglikelihood for subject $i is $(repr(ℓ)) at the checked point; expected a finite real")
        ℓg, g = conditional_loglikelihood_and_gradient(model, i, θ, σ, ηi)
        agrees(ℓg, ℓ) || violation("subject $i: conditional_loglikelihood_and_gradient value $ℓg ≠ conditional_loglikelihood $ℓ")
        length(g) == m || violation("subject $i: η gradient has length $(length(g)); neta = $m")
        ℓh, gh, H = conditional_loglikelihood_gradient_and_hessian(model, i, θ, σ, ηi)
        agrees(ℓh, ℓ) || violation("subject $i: conditional_loglikelihood_gradient_and_hessian value $ℓh ≠ conditional_loglikelihood $ℓ")
        agrees(gh, g) || violation("subject $i: the gradients of the two derivative methods disagree: $gh vs $g")
        size(H) == (m, m) || violation("subject $i: η Hessian has size $(size(H)); expected ($m, $m)")
        agrees(H, H') || violation("subject $i: η Hessian is not symmetric")
        _check_reference(violation, agrees, reference, model, i, θ, σ, ηi, g, H)
        _check_mu_invariance(violation, agrees, mr, model, i, θ, σ, ηi, ℓ)
        _check_observation_model(violation, agrees, observation_model(model), model, i, θ, σ, ηi, ℓ)
        population_gradient &&
            _check_population_gradient(violation, agrees, reference, model, i, θ, σ, ηi, ℓ, layout)
    end
    return true
end

_subject_eta(::Nothing, k, m) = zeros(m)
_subject_eta(η::AbstractVector{<:Real}, k, m) = η
_subject_eta(η::AbstractVector{<:AbstractVector}, k, m) = η[k]

_check_reference(violation, agrees, ::Nothing, model, i, θ, σ, η, g, H) = nothing
function _check_reference(violation, agrees, backend, model, i, θ, σ, η, g, H)
    _, gref, Href = DI.value_gradient_and_hessian(_eta_objective, backend, η, _eta_contexts(model, i, θ, σ)...)
    agrees(g, gref) || violation("subject $i: η gradient $g disagrees with the reference backend's $gref")
    agrees(H, Href) || violation("subject $i: η Hessian disagrees with the reference backend's")
    return nothing
end

_check_mu_layout(violation, ::Nothing, n, m) = nothing
function _check_mu_layout(violation, mr::MuReferencing, n, m)
    length(mr.design) == n ||
        violation("mu_referencing declares $(length(mr.design)) designs for $n subjects")
    for (i, X) in enumerate(mr.design)
        size(X, 1) == m || violation("mu_referencing design[$i] has $(size(X, 1)) rows; neta = $m")
    end
    return nothing
end

_check_mu_invariance(violation, agrees, ::Nothing, model, i, θ, σ, η, ℓ) = nothing
function _check_mu_invariance(violation, agrees, mr::MuReferencing, model, i, θ, σ, η, ℓ)
    k = length(mr.theta_indices)
    k == 0 && return nothing
    δ = [0.1 * (-1)^j * j / k for j in 1:k]
    θs = collect(float.(θ))
    θs[mr.theta_indices] .+= δ
    ηs = η - mr.design[i] * δ
    ℓs = conditional_loglikelihood(model, i, θs, σ, ηs)
    agrees(ℓs, ℓ) || violation(
        "subject $i: the conditional log-density changes from $ℓ to $ℓs under the declared mu-referencing " *
        "shift (θ[$(mr.theta_indices)] += δ, η -= X_i δ), so it does not depend on them only through X_i θ + η")
    return nothing
end

_check_observation_model(violation, agrees, ::Nothing, model, i, θ, σ, η, ℓ) = nothing
function _check_observation_model(violation, agrees, ::GaussianObservations, model, i, θ, σ, η, ℓ)
    y = observations(model, i)
    μ, v = observation_moments(model, i, θ, σ, η)
    length(μ) == length(y) == length(v) || violation(
        "subject $i: observation_moments returned $(length(μ)) means and $(length(v)) variances for $(length(y)) observations")
    all(>(0), v) || violation("subject $i: observation_moments returned non-positive variances")
    ℓobs = sum(j -> -(log(2π * v[j]) + (y[j] - μ[j])^2 / v[j]) / 2, eachindex(y); init = zero(ℓ))
    agrees(ℓobs, ℓ) || violation(
        "subject $i: the declared Gaussian observation model gives log-density $ℓobs, conditional_loglikelihood gives $ℓ")
    return nothing
end

function _check_population_gradient(violation, agrees, reference, model, i, θ, σ, η, ℓ, layout)
    ℓp, gθ, gσ = conditional_loglikelihood_and_population_gradient(model, i, θ, σ, η)
    agrees(ℓp, ℓ) || violation("subject $i: conditional_loglikelihood_and_population_gradient value $ℓp ≠ $ℓ")
    length(gθ) == layout.ntheta || violation("subject $i: θ gradient has length $(length(gθ))")
    length(gσ) == layout.nsigma || violation("subject $i: σ gradient has length $(length(gσ))")
    reference === nothing && return nothing
    x = vcat(θ, σ)
    gref = DI.gradient(_population_objective, reference, x,
        DI.Constant(model), DI.Constant(i), DI.Constant(η), DI.Constant(length(θ)))
    agrees(vcat(gθ, gσ), gref) || violation("subject $i: θ/σ gradient disagrees with the reference backend's")
    return nothing
end
