using NLMEEstimation
using NLMEEstimation.ReferenceModels
using LinearAlgebra
using Test
import NLMEEstimation as NE
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon

# A joint-program Gaussian model with unequal observations per subject. Its
# marginal likelihood is known exactly, including the determinant term.
struct BatchedLinearGaussian
    y::Vector{Vector{Float64}}
    moment_calls::Base.RefValue{Int}
end

BatchedLinearGaussian(y) = BatchedLinearGaussian(y, Ref(0))
NE.nsubjects(m::BatchedLinearGaussian) = length(m.y)
NE.nlme_layout(::BatchedLinearGaussian) = NLMELayout(1, 1, [1])
NE.evaluation_style(::BatchedLinearGaussian) = BatchedEvaluation()
NE.observation_model(::BatchedLinearGaussian) = GaussianObservations()
NE.observations(m::BatchedLinearGaussian, subjects::AbstractVector{<:Integer}) = m.y[subjects]

function NE.conditional_loglikelihoods(m::BatchedLinearGaussian, subjects, θ, σ, H)
    v = exp(2σ[1])
    return [sum(y -> -(log(2π * v) + (y - θ[1] - H[1, k])^2 / v) / 2,
        m.y[i]) for (k, i) in enumerate(subjects)]
end

function NE.conditional_loglikelihoods_and_gradients(m::BatchedLinearGaussian, subjects, θ, σ, H)
    ℓ = NE.conditional_loglikelihoods(m, subjects, θ, σ, H)
    v = exp(2σ[1])
    G = reshape([sum(y -> (y - θ[1] - H[1, k]) / v, m.y[i])
        for (k, i) in enumerate(subjects)], 1, :)
    return ℓ, G
end
function NE.observation_moments(m::BatchedLinearGaussian, subjects::AbstractVector{<:Integer}, θ, σ, H)
    v = exp(2σ[1])
    return [(fill(θ[1] + H[1, k], length(m.y[i])), fill(v, length(m.y[i])))
        for (k, i) in enumerate(subjects)]
end
function NE.observation_moments_and_jacobians(m::BatchedLinearGaussian,
        subjects::AbstractVector{<:Integer}, θ, σ, H::AbstractMatrix)
    m.moment_calls[] += 1
    return [(fill(θ[1] + H[1, k], length(m.y[i])),
        fill(exp(2σ[1]), length(m.y[i])), ones(length(m.y[i]), 1),
        zeros(length(m.y[i]), 1)) for (k, i) in enumerate(subjects)]
end

function exact_linear_ll(m, θ, σ, ω)
    variance = exp(2σ[1])
    omega = exp(2ω[1])
    return sum(m.y) do y
        n = length(y)
        covariance = variance * Matrix{Float64}(I, n, n) + omega * ones(n, n)
        ch = cholesky(Symmetric(covariance))
        residual = y .- θ[1]
        -(n * log(2π) + 2sum(log, diag(ch.L)) + dot(residual, ch \ residual)) / 2
    end
end

@testset "FOCE(I): exact linear-Gaussian marginal" begin
    model = BatchedLinearGaussian([[-0.7, -0.3, 0.2], [1.1, 1.4], [2.1, 2.4, 1.9, 2.2]])
    θ, σ, ω = [0.8], [log(0.25)], [log(0.7)]
    @test check_protocol(model, θ, σ, ω)
    model.moment_calls[] = 0
    fixed = focei_evaluate(model, θ, σ, ω; ebe_options = EBEOptions(gtol = 1e-11))
    @test fixed.loglikelihood ≈ exact_linear_ll(model, θ, σ, ω) atol = 1e-8
    @test fixed.ofv ≈ -2 * fixed.loglikelihood
    @test length(fixed.curvatures) == nsubjects(model)
    @test all(C -> isposdef(Symmetric(C)), fixed.curvatures)
    @test model.moment_calls[] == 1
    @test focei_evaluate(model, θ, σ, ω; interaction = false).loglikelihood ≈ fixed.loglikelihood atol = 1e-8

    fit = focei(model, [0.4], [log(0.5)], [log(0.5)];
        options = FOCEOptions(maxeval = 120, rhobeg = 0.15, rhoend = 1e-5))
    @test fit.ofv < focei_evaluate(model, [0.4], [log(0.5)], [log(0.5)]).ofv
    @test fit.loglikelihood ≈ exact_linear_ll(model, fit.theta, fit.sigma, fit.omega) atol = 1e-7
    @test fit.omega_matrix ≈ omega_matrix(nlme_layout(model), fit.omega)
    @test size(fit.ebes.modes) == (1, nsubjects(model))
    @test fit.evaluations <= 120
end

@testset "FOCE(I): simulated one-compartment recovery" begin
    model, _ = simulated_model(seed = 21, nsubjects = 16)
    ω = omega_parameters(nlme_layout(model), Ω_SIM)
    with_interaction = focei_evaluate(model, θ_SIM, σ_SIM, ω)
    without_interaction = focei_evaluate(model, θ_SIM, σ_SIM, ω;
        η0 = with_interaction.ebes, interaction = false)
    @test tr(with_interaction.curvatures[1]) > tr(without_interaction.curvatures[1])
    θ0 = copy(θ_SIM)
    θ0[1] += 0.5
    fit = focei(model, θ0, σ_SIM, ω; free = [1],
        options = FOCEOptions(maxeval = 25, rhobeg = 0.2, rhoend = 1e-3,
            ebe_options = EBEOptions(gtol = 1e-7)))
    @test fit.ofv < focei_evaluate(model, θ0, σ_SIM, ω).ofv
    @test abs(fit.theta[1] - θ_SIM[1]) < abs(θ0[1] - θ_SIM[1])
    @test fit.theta[2:end] == θ_SIM[2:end]
    @test fit.sigma == σ_SIM
    @test fit.omega == ω
end
