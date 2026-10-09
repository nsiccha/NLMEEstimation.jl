# The model protocol. A model is any Julia value; it takes part by adding methods
# to the functions below. Subjects are indexed `1:nsubjects(model)`. θ, σ and η may
# be any `AbstractVector` (views, dual numbers), so implementations must not
# restrict their element type.

# ── Required ──────────────────────────────────────────────────────────────────

"""
    nsubjects(model) -> Int

**Required.** Number of subjects (independent units carrying their own η_i).
"""
function nsubjects end

"""
    nlme_layout(model) -> NLMELayout

**Required.** The parameter layout: lengths of θ and σ and the η block structure.
See [`NLMELayout`](@ref) for the coordinate conventions.
"""
function nlme_layout end

"""
    conditional_loglikelihood(model, i, θ, σ, η) -> Real

**Required.** The conditional log-density `log p(y_i | θ, σ, η_i)` of subject `i`'s
observations, *without* the random-effect prior (which this package owns, see
[`re_logprior`](@ref)). θ and σ are in the model's unconstrained coordinates.

Return `-Inf` where the density is zero (a failed solve, for example); estimators
treat non-finite values as infeasible and report them, never as success.
"""
function conditional_loglikelihood end

# ── Derivatives: implement them, or declare an AD backend ─────────────────────

"""
    ad_backend(model) -> Union{Nothing, ADTypes.AbstractADType}

**Optional.** The automatic-differentiation backend this package uses, through
DifferentiationInterface, for every derivative method the model does not implement
itself. The default `nothing` declares no backend; calling a missing derivative
method is then an error.

Finite-difference backends (`AutoFiniteDiff`, `AutoFiniteDifferences`, also nested
inside `AutoSparse`, `SecondOrder` or `MixedMode`) are refused: every gradient,
Hessian and Jacobian in this package comes from AD.
"""
ad_backend(model) = nothing

"""
    hessian_ad_backend(model) -> Union{Nothing, ADTypes.AbstractADType}

**Optional.** The backend used for the default η Hessian. Defaults to
[`ad_backend`](@ref); override it with, for example,
`DifferentiationInterface.SecondOrder(AutoForwardDiff(), AutoEnzyme())`.
"""
hessian_ad_backend(model) = ad_backend(model)

"""
    conditional_loglikelihood_and_gradient(model, i, θ, σ, η) -> (ℓ, ∇η ℓ)

**Required, or provided by [`ad_backend`](@ref).** The conditional log-density and
its gradient in η_i.
"""
function conditional_loglikelihood_and_gradient(model, i, θ, σ, η)
    backend = _checked_backend(model, ad_backend(model), :conditional_loglikelihood_and_gradient)
    return DI.value_and_gradient(_eta_objective, backend, η, _eta_contexts(model, i, θ, σ)...)
end

"""
    conditional_loglikelihood_gradient_and_hessian(model, i, θ, σ, η) -> (ℓ, ∇η ℓ, ∇²η ℓ)

**Required, or provided by [`hessian_ad_backend`](@ref).** The conditional
log-density with its gradient and Hessian in η_i.
"""
function conditional_loglikelihood_gradient_and_hessian(model, i, θ, σ, η)
    backend = _checked_backend(model, hessian_ad_backend(model), :conditional_loglikelihood_gradient_and_hessian)
    return DI.value_gradient_and_hessian(_eta_objective, backend, η, _eta_contexts(model, i, θ, σ)...)
end

"""
    conditional_loglikelihood_and_population_gradient(model, i, θ, σ, η) -> (ℓ, ∇θ ℓ, ∇σ ℓ)

**Optional, or provided by [`ad_backend`](@ref).** The conditional log-density with
its gradients in θ and σ at fixed η_i. Used by population-level estimators
(FOCEI's outer step, for example), not by [`empirical_bayes`](@ref).
"""
function conditional_loglikelihood_and_population_gradient(model, i, θ, σ, η)
    backend = _checked_backend(model, ad_backend(model), :conditional_loglikelihood_and_population_gradient)
    ntheta = length(θ)
    x = vcat(θ, σ)
    ℓ, g = DI.value_and_gradient(_population_objective, backend, x,
        DI.Constant(model), DI.Constant(i), DI.Constant(η), DI.Constant(ntheta))
    return ℓ, g[1:ntheta], g[(ntheta + 1):end]
end

_eta_objective(η, model, i, θ, σ) = conditional_loglikelihood(model, i, θ, σ, η)
_eta_contexts(model, i, θ, σ) = (DI.Constant(model), DI.Constant(i), DI.Constant(θ), DI.Constant(σ))
_population_objective(x, model, i, η, ntheta) =
    conditional_loglikelihood(model, i, view(x, 1:ntheta), view(x, (ntheta + 1):length(x)), η)

# Backend validation, by dispatch on the backend type.
_checked_backend(model, ::Nothing, op) = throw(ArgumentError(
    "$(typeof(model)) has no method for `$op` and declares no AD backend. Either implement " *
    "`NLMEEstimation.$op` for it or define `NLMEEstimation.ad_backend(::$(typeof(model)))`."))
_checked_backend(model, backend::Union{ADTypes.AutoFiniteDiff, ADTypes.AutoFiniteDifferences}, op) =
    throw(ArgumentError(
        "`$op` for $(typeof(model)): refused finite-difference backend $(nameof(typeof(backend))). " *
        "NLMEEstimation takes every derivative from automatic differentiation, never from finite " *
        "differences; declare an AD backend (ForwardDiff, Enzyme, Mooncake, …) instead."))
_checked_backend(model, backend::ADTypes.NoAutoDiff, op) = _checked_backend(model, nothing, op)
function _checked_backend(model, backend::ADTypes.AutoSparse, op)
    _checked_backend(model, ADTypes.dense_ad(backend), op)
    return backend
end
function _checked_backend(model, backend::DI.SecondOrder, op)
    _checked_backend(model, DI.outer(backend), op)
    _checked_backend(model, DI.inner(backend), op)
    return backend
end
function _checked_backend(model, backend::DI.MixedMode, op)
    _checked_backend(model, DI.forward_backend(backend), op)
    _checked_backend(model, DI.reverse_backend(backend), op)
    return backend
end
_checked_backend(model, backend::ADTypes.AbstractADType, op) = backend
_checked_backend(model, backend, op) = throw(ArgumentError(
    "`ad_backend` for $(typeof(model)) returned $(repr(backend)), which is not an ADTypes backend"))

# ── Reporting ─────────────────────────────────────────────────────────────────

"""
    parameter_names(model) -> (theta = Vector{String}, sigma = Vector{String}, eta = Vector{String})

**Optional.** Human-readable coordinate names for reporting. Defaults to
`θ1, θ2, …`, `σ1, …`, `η1, …`.
"""
function parameter_names(model)
    layout = nlme_layout(model)
    return (theta = ["θ$k" for k in 1:layout.ntheta],
        sigma = ["σ$k" for k in 1:layout.nsigma],
        eta = ["η$k" for k in 1:neta(layout)])
end

# ── Optional structural traits (used by later estimators) ─────────────────────

"""
    MuReferencing(theta_indices, design)

Mu-referencing declaration (see [`mu_referencing`](@ref)).

- `theta_indices`: the θ coordinates that are mu-referenced.
- `design`: one `neta × length(theta_indices)` matrix `X_i` per subject.

Contract: for every subject `i`, [`conditional_loglikelihood`](@ref) depends on
`θ[theta_indices]` and `η` only through the individual linear predictor
`ϕ_i = X_i * θ[theta_indices] + η`. Equivalently, it is unchanged under
`θ[theta_indices] .+= δ`, `η .-= X_i * δ`. The model applies its own inverse link to
ϕ_i. A zero row of `X_i` marks an η component without a mu-referenced typical value.
"""
struct MuReferencing{M <: AbstractMatrix{<:Real}}
    theta_indices::Vector{Int}
    design::Vector{M}
    function MuReferencing(theta_indices::AbstractVector{<:Integer}, design::AbstractVector{M}) where {M <: AbstractMatrix{<:Real}}
        allunique(theta_indices) ||
            throw(ArgumentError("MuReferencing: theta_indices $theta_indices are not unique"))
        for (i, X) in enumerate(design)
            size(X, 2) == length(theta_indices) || throw(DimensionMismatch(
                "MuReferencing: design[$i] has $(size(X, 2)) columns for $(length(theta_indices)) mu-referenced θ coordinates"))
        end
        return new{M}(collect(Int, theta_indices), collect(design))
    end
end

"""
    mu_referencing(model) -> Union{Nothing, MuReferencing}

**Optional trait.** Declares which θ coordinates enter the model as typical values
of the random effects, `ϕ_i = X_i θ_μ + η_i`. SAEM's closed-form M-step needs it.
Default: `nothing` (not declared).
"""
mu_referencing(model) = nothing

"""
    GaussianObservations()

Observation-model declaration (see [`observation_model`](@ref)): the observations
of subject `i` are independent Gaussians. The model then implements
[`observations`](@ref) and [`observation_moments`](@ref).

Contract: `conditional_loglikelihood(model, i, θ, σ, η)` equals
`Σ_j log N(y_ij | μ_ij, v_ij)` with `y_i = observations(model, i)` and
`(μ_i, v_i) = observation_moments(model, i, θ, σ, η)`.
"""
struct GaussianObservations end

"""
    observation_model(model) -> Union{Nothing, GaussianObservations}

**Optional trait.** Declares the observation-model structure FOCE(I) linearizes.
Default: `nothing`, under which estimators use the exact Laplace Hessian.
"""
observation_model(model) = nothing

"""
    observations(model, i) -> AbstractVector

Subject `i`'s observations; required when [`observation_model`](@ref) is declared.
"""
function observations end

"""
    observation_moments(model, i, θ, σ, η) -> (mean::AbstractVector, variance::AbstractVector)

Predicted means and variances of subject `i`'s observations; required when
[`observation_model`](@ref) is declared.
"""
function observation_moments end
