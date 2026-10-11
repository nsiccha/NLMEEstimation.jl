"""
    NLMEEstimation.ReferenceModels

Small NLME models implementing the protocol, for tests, examples and cross-checks
of other implementations of the same model. Not part of the protocol.
"""
module ReferenceModels

using LinearAlgebra
using Random
import ..NLMEEstimation as NE
import DifferentiationInterface as DI
using ..NLMEEstimation: NLMELayout, MuReferencing, GaussianObservations

export OneCompartmentOral, PKSubject, AdditiveError, CombinedError, simulate_one_compartment_oral

"""
    PKSubject(dose, weight, times, concentrations)

One subject of a single-oral-dose study: dose amount, body weight, sampling times
and observed concentrations.
"""
struct PKSubject
    dose::Float64
    weight::Float64
    times::Vector{Float64}
    concentrations::Vector{Float64}
    function PKSubject(dose::Real, weight::Real, times::AbstractVector{<:Real}, concentrations::AbstractVector{<:Real})
        length(times) == length(concentrations) || throw(DimensionMismatch(
            "PKSubject: $(length(times)) times for $(length(concentrations)) concentrations"))
        return new(dose, weight, collect(Float64, times), collect(Float64, concentrations))
    end
end

"""
Residual-error model of [`OneCompartmentOral`](@ref).
"""
abstract type ResidualError end

"""
    AdditiveError()

Additive residual error: `σ = [log a]`, variance `a²`.
"""
struct AdditiveError <: ResidualError end

"""
    CombinedError()

Combined additive and proportional residual error: `σ = [log a, log b]`, variance
`a² + b² C²` at predicted concentration `C`.
"""
struct CombinedError <: ResidualError end

nsigma(::AdditiveError) = 1
nsigma(::CombinedError) = 2
sigma_names(::AdditiveError) = ["log_additive_sd"]
sigma_names(::CombinedError) = ["log_additive_sd", "log_proportional_sd"]
residual_variance(::AdditiveError, σ, C) = exp(2 * σ[1])
residual_variance(::CombinedError, σ, C) = exp(2 * σ[1]) + exp(2 * σ[2]) * C^2

"""
    OneCompartmentOral(subjects; backend, error_model = CombinedError(),
                       eta_blocks = [1, 1, 1], reference_weight = 70.0)

One-compartment model with first-order absorption and elimination after a single
oral dose, solved analytically. With `θ = [log ka, log CL, log V, β_WT]`:

    log ka_i = θ[1] + η_i[1]
    log CL_i = θ[2] + β_WT · log(WT_i / reference_weight) + η_i[2]
    log V_i  = θ[3] + η_i[3]
    C_i(t)   = dose_i · ka_i / (V_i (ka_i − ke_i)) · (exp(−ke_i t) − exp(−ka_i t)),  ke_i = CL_i / V_i
    y_ij     ~ N(C_i(t_ij), residual variance)

Implements the full protocol, including the [`MuReferencing`](@ref) and
[`GaussianObservations`](@ref) traits. Derivatives come from `backend` (an ADTypes
backend whose package is loaded, for example `ADTypes.AutoEnzyme()` with Enzyme).
`eta_blocks` sets the Ω block structure (`[3]` for a dense Ω).
"""
struct OneCompartmentOral{E <: ResidualError, B}
    subjects::Vector{PKSubject}
    error_model::E
    eta_blocks::Vector{Int}
    reference_weight::Float64
    backend::B
end
function OneCompartmentOral(subjects::AbstractVector{PKSubject}; backend,
        error_model::ResidualError = CombinedError(), eta_blocks = [1, 1, 1], reference_weight = 70.0)
    sum(eta_blocks) == 3 || throw(ArgumentError("OneCompartmentOral has three random effects; eta_blocks = $eta_blocks"))
    isempty(subjects) && throw(ArgumentError("OneCompartmentOral: no subjects"))
    return OneCompartmentOral(collect(subjects), error_model, collect(Int, eta_blocks), Float64(reference_weight), backend)
end

NE.nsubjects(m::OneCompartmentOral) = length(m.subjects)
NE.nlme_layout(m::OneCompartmentOral) = NLMELayout(4, nsigma(m.error_model), m.eta_blocks)
NE.ad_backend(m::OneCompartmentOral) = m.backend
NE.parameter_names(m::OneCompartmentOral) = (
    theta = ["log_ka", "log_CL", "log_V", "CL_weight_exponent"],
    sigma = sigma_names(m.error_model),
    eta = ["eta_ka", "eta_CL", "eta_V"])

function NE.mu_referencing(m::OneCompartmentOral)
    design = [[1.0 0 0 0; 0 1 0 log(s.weight / m.reference_weight); 0 0 1 0] for s in m.subjects]
    return MuReferencing(1:4, design)
end

NE.observation_model(::OneCompartmentOral) = GaussianObservations()
NE.observations(m::OneCompartmentOral, i::Integer) = m.subjects[i].concentrations
NE.observation_moments(m::OneCompartmentOral, i::Integer, θ, σ, η) =
    subject_moments(m.error_model, m.reference_weight, m.subjects[i], θ, σ, η)

# Differentiate the scalar observation kernel with Enzyme. This avoids placing
# a returned array of mixed constant/active values inside one AD call.
function NE.observation_moments_and_jacobians(m::OneCompartmentOral, i::Integer, θ, σ, η)
    s = m.subjects[i]
    μ, v = subject_moments(m.error_model, m.reference_weight, s, θ, σ, η)
    T = promote_type(eltype(η), eltype(μ), eltype(v))
    Jμ = zeros(T, length(s.times), length(η))
    Jv = zeros(T, length(s.times), length(η))
    ctx = (DI.Constant(m.error_model), DI.Constant(m.reference_weight),
        DI.Constant(s), DI.Constant(θ), DI.Constant(σ))
    for j in eachindex(s.times)
        Jμ[j, :] = DI.gradient(_mean_at, m.backend, η, ctx..., DI.Constant(j))
        Jv[j, :] = DI.gradient(_variance_at, m.backend, η, ctx..., DI.Constant(j))
    end
    return μ, v, Jμ, Jv
end

_mean_at(η, error_model, reference_weight, s, θ, σ, j) =
    first(observation_moment(error_model, individual_parameters(reference_weight, s, θ, η),
        s.dose, s.times[j], σ))
_variance_at(η, error_model, reference_weight, s, θ, σ, j) =
    last(observation_moment(error_model, individual_parameters(reference_weight, s, θ, η),
        s.dose, s.times[j], σ))

# One scalar per-observation kernel, `observation_moment`, defines the model: the
# log-density accumulates it without allocating, `observation_moments` collects it
# into arrays, and its per-observation (prediction, variance) pair is what FOCE(I)
# linearizes.
function NE.conditional_loglikelihood(m::OneCompartmentOral, i::Integer, θ, σ, η)
    s = m.subjects[i]
    p = individual_parameters(m.reference_weight, s, θ, η)
    ℓ = zero(float(promote_type(eltype(θ), eltype(σ), eltype(η))))
    for j in eachindex(s.times)
        C, var = observation_moment(m.error_model, p, s.dose, s.times[j], σ)
        ℓ += -(log(2π * var) + (s.concentrations[j] - C)^2 / var) / 2
    end
    return ℓ
end

# Predicted concentrations and residual variances of one subject.
function subject_moments(error_model, reference_weight, s::PKSubject, θ, σ, η)
    p = individual_parameters(reference_weight, s, θ, η)
    moments = [observation_moment(error_model, p, s.dose, t, σ) for t in s.times]
    return first.(moments), last.(moments)
end

function individual_parameters(reference_weight, s::PKSubject, θ, η)
    ka = exp(θ[1] + η[1])
    cl = exp(θ[2] + θ[4] * log(s.weight / reference_weight) + η[2])
    v = exp(θ[3] + η[3])
    return (ka = ka, ke = cl / v, v = v)
end

# Predicted concentration and residual variance at time t after the dose.
function observation_moment(error_model, p, dose, t, σ)
    C = dose * p.ka / p.v * bateman(p.ka, p.ke, t)
    return C, residual_variance(error_model, σ, C)
end

# (exp(-ke t) - exp(-ka t)) / (ka - ke), accurate when ka ≈ ke.
function bateman(ka, ke, t)
    d = ka - ke
    iszero(d) && return t * exp(-ke * t)
    return -exp(-ke * t) * expm1(-d * t) / d
end

"""
    simulate_one_compartment_oral(rng, θ, σ, Ω; error_model = CombinedError(),
        nsubjects = 20, dose = 320.0, weight_range = (50.0, 90.0),
        times = [0.25, 0.5, 1, 2, 3.5, 5, 7, 9, 12, 24], reference_weight = 70.0)
        -> (subjects::Vector{PKSubject}, η::Vector{Vector{Float64}})

Simulate a study from [`OneCompartmentOral`](@ref) with `η_i ~ N(0, Ω)` and body
weights uniform on `weight_range`. Returns the subjects and the true random effects.
"""
function simulate_one_compartment_oral(rng::AbstractRNG, θ, σ, Ω::AbstractMatrix;
        error_model::ResidualError = CombinedError(), nsubjects::Integer = 20, dose::Real = 320.0,
        weight_range = (50.0, 90.0), times = [0.25, 0.5, 1, 2, 3.5, 5, 7, 9, 12, 24],
        reference_weight::Real = 70.0)
    L = cholesky(Symmetric(Matrix(Ω))).L
    subjects = PKSubject[]
    etas = Vector{Float64}[]
    for _ in 1:nsubjects
        η = L * randn(rng, 3)
        weight = weight_range[1] + (weight_range[2] - weight_range[1]) * rand(rng)
        design = PKSubject(dose, weight, times, zeros(length(times)))
        μ, v = subject_moments(error_model, reference_weight, design, θ, σ, η)
        y = μ .+ sqrt.(v) .* randn(rng, length(μ))
        push!(subjects, PKSubject(dose, weight, times, y))
        push!(etas, η)
    end
    return subjects, etas
end

end
