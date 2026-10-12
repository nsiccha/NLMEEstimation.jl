using NLMEEstimation
using Test
using Random
using LinearAlgebra
using ADTypes: AutoEnzyme
using Enzyme
import NLMEEstimation as NE

struct ADPopulationPrior end
NE.ad_backend(::ADPopulationPrior) = AutoEnzyme()
NE.population_logprior(::ADPopulationPrior, θ, σ, ω) =
    -sum(abs2, θ) - 2sum(abs2, σ) - 3sum(abs2, ω)

@testset "population prior: default AD gradient" begin
    value, gθ, gσ, gω = population_logprior_and_gradient(
        ADPopulationPrior(), [0.2], [-0.3], [0.4])
    @test value ≈ -0.04 - 0.18 - 0.48
    @test gθ ≈ [-0.4]
    @test gσ ≈ [1.2]
    @test gω ≈ [-2.4]
end

struct PriorGaussianModel
    y::Vector{Vector{Float64}}
    active::Bool
end

NE.nsubjects(m::PriorGaussianModel) = length(m.y)
NE.nlme_layout(::PriorGaussianModel) = NLMELayout(1, 0, [1])
NE.mu_referencing(m::PriorGaussianModel) =
    MuReferencing([1], [ones(1, 1) for _ in m.y])
NE.observation_model(::PriorGaussianModel) = GaussianObservations()
NE.observations(m::PriorGaussianModel, subjects::AbstractVector{<:Integer}) = m.y[subjects]

function NE.conditional_loglikelihood(m::PriorGaussianModel, i, θ, σ, η)
    return sum(y -> -(log(2π) + (y - θ[1] - η[1])^2) / 2, m.y[i])
end

function NE.conditional_loglikelihood_and_gradient(m::PriorGaussianModel, i, θ, σ, η)
    return NE.conditional_loglikelihood(m, i, θ, σ, η),
        [sum(y -> y - θ[1] - η[1], m.y[i])]
end

function NE.conditional_loglikelihood_and_population_gradient(m::PriorGaussianModel,
        i, θ, σ, η)
    return NE.conditional_loglikelihood(m, i, θ, σ, η),
        [sum(y -> y - θ[1] - η[1], m.y[i])], Float64[]
end

function NE.observation_moments_and_jacobians(m::PriorGaussianModel,
        subjects::AbstractVector{<:Integer}, θ, σ, H)
    return [(fill(θ[1] + H[1, k], length(m.y[i])), ones(length(m.y[i])),
        ones(length(m.y[i]), 1), zeros(length(m.y[i]), 1))
        for (k, i) in enumerate(subjects)]
end

# Exponential prior on the natural-scale exp(θ) includes the +θ Jacobian;
# the second factor regularizes the package-owned log standard deviation ω.
function NE.population_logprior(m::PriorGaussianModel, θ, σ, ω)
    m.active || return nothing
    return θ[1] - exp(θ[1]) - (ω[1] - log(0.5))^2 / 0.18
end
function NE.population_logprior_and_gradient(m::PriorGaussianModel, θ, σ, ω)
    m.active || return (0.0, [0.0], Float64[], [0.0])
    return NE.population_logprior(m, θ, σ, ω), [1 - exp(θ[1])], Float64[],
        [-2(ω[1] - log(0.5)) / 0.18]
end
NE.population_prior_provenance(::PriorGaussianModel) = "synthetic transformed prior"

@testset "population prior: FOCE once, likelihood retained" begin
    y = [[-0.2, 0.1], [0.4, 0.7], [1.1, 1.3]]
    ml = PriorGaussianModel(y, false)
    map = PriorGaussianModel(y, true)
    θ, σ, ω = [0.3], Float64[], [log(0.6)]
    plain = focei_evaluate(ml, θ, σ, ω)
    regularized = focei_evaluate(map, θ, σ, ω)
    lp = population_logprior(map, θ, σ, ω)
    @test plain.objective == :ML
    @test plain.logposterior == plain.loglikelihood
    @test regularized.objective == :MAP
    @test regularized.loglikelihood ≈ plain.loglikelihood
    @test regularized.logposterior ≈ plain.loglikelihood + lp
    @test regularized.ofv ≈ plain.ofv - 2lp
    @test regularized.prior_provenance == "synthetic transformed prior"
    @test regularized.ebes.modes ≈ plain.ebes.modes

    ml_fit = focei(ml, θ, σ, ω; free = [1],
        options = FOCEOptions(maxeval = 35, rhobeg = 0.1))
    fit = focei(map, θ, σ, ω; free = [1],
        options = FOCEOptions(maxeval = 35, rhobeg = 0.1))
    @test fit.objective == :MAP
    @test fit.logposterior ≈ fit.loglikelihood + fit.population_logprior
    @test fit.ofv ≈ -2fit.logposterior
    @test fit.prior_provenance == "synthetic transformed prior"
    @test abs(fit.theta[1] - ml_fit.theta[1]) > 1e-3
end

@testset "population prior: SAEM mixture and online gradients" begin
    model = PriorGaussianModel([[0.0], [1.0]], true)
    layout = nlme_layout(model)
    θ, σ, ω = [0.2], Float64[], [log(0.6)]
    Φ = reshape([0.1, 0.9], 1, 2)
    mean_phi = copy(Φ)
    second_phi = reshape([Φ[1, 1]^2, Φ[1, 2]^2], 1, 1, 2)
    design = [ones(1, 1), ones(1, 1)]
    x = vcat(θ, σ, ω)
    no_prior = PriorGaussianModel(model.y, false)
    for mode in (Val(:online), Val(:mixture))
        value, gradient = NE._saem_map_value_gradient(x, model, 1:2, layout,
            [1], design, [Φ], [1.0], mean_phi, second_phi, mode)
        plain_value, plain_gradient = NE._saem_map_value_gradient(x, no_prior,
            1:2, layout, [1], design, [Φ], [1.0], mean_phi, second_phi, mode)
        lp, gθ, _, gω = population_logprior_and_gradient(model, θ, σ, ω)
        @test value - plain_value ≈ lp
        @test gradient - plain_gradient ≈ vcat(gθ, gω)
        repeated, repeated_gradient = NE._saem_map_value_gradient(x, model,
            1:2, layout, [1], design, [Φ, Φ], [0.25, 0.75], mean_phi,
            second_phi, mode)
        @test repeated ≈ value
        @test repeated_gradient ≈ gradient
        for j in eachindex(x)
            dx = zeros(length(x))
            dx[j] = 1e-6
            vp = first(NE._saem_map_value_gradient(x + dx, model, 1:2,
                layout, [1], design, [Φ], [1.0], mean_phi, second_phi, mode))
            vm = first(NE._saem_map_value_gradient(x - dx, model, 1:2,
                layout, [1], design, [Φ], [1.0], mean_phi, second_phi, mode))
            @test gradient[j] ≈ (vp - vm) / 2e-6 atol = 1e-6
        end
    end
    fit = saem(model, θ, σ, ω; rng = MersenneTwister(7),
        options = SAEMOptions(iterations = 4, burnin = 2, mcmc_burnin = 1,
            mh_rounds = 1, numeric_maxiter = 3, numeric_mstep = :online))
    @test fit.objective == :MAP
    @test fit.population_logprior ≈ population_logprior(model, fit.theta,
        fit.sigma, fit.omega)
    @test fit.prior_provenance == "synthetic transformed prior"
    @test size(fit.eta) == (1, 2)
    plain_fit = saem(no_prior, θ, σ, ω; rng = MersenneTwister(7),
        options = SAEMOptions(iterations = 4, burnin = 2, mcmc_burnin = 1,
            mh_rounds = 1, numeric_maxiter = 3, numeric_mstep = :online))
    @test plain_fit.objective == :ML
    @test plain_fit.population_logprior == 0
    @test plain_fit.prior_provenance === nothing
    @test population_logprior_and_gradient(nothing, θ, σ, ω) ==
        (0.0, [0.0], Float64[], [0.0])
    @test NE._saem_floor_omega(layout, [log(0.01)], 0.04)[1] ≈ log(0.2)
end

@testset "population prior: correlated random-effect M-step gradient" begin
    layout = NLMELayout(2, 0, [2])
    θ = [0.2, -0.1]
    ω = [log(0.7), 0.15, log(0.5)]
    mean_phi = [0.6 0.3; -0.2 0.4]
    second_phi = zeros(2, 2, 2)
    for k in 1:2
        φ = mean_phi[:, k]
        second_phi[:, :, k] = φ * φ' + [0.2 0.03; 0.03 0.1]
    end
    design = [Matrix{Float64}(I, 2, 2) for _ in 1:2]
    value, gθ, gω = NE._saem_gaussian_map_value_gradient(layout, θ, ω,
        [1, 2], design, mean_phi, second_phi)
    @test isfinite(value)
    x = vcat(θ, ω)
    for j in eachindex(x)
        dx = zeros(length(x))
        dx[j] = 1e-6
        f(z) = first(NE._saem_gaussian_map_value_gradient(layout, z[1:2],
            z[3:end], [1, 2], design, mean_phi, second_phi))
        @test vcat(gθ, gω)[j] ≈ (f(x + dx) - f(x - dx)) / 2e-6 atol = 1e-6
    end
end
