using NLMEEstimation
using NLMEEstimation.ReferenceModels
using DifferentiationInterface: AutoForwardDiff, AutoFiniteDiff, AutoFiniteDifferences, AutoReverseDiff,
    AutoSparse, SecondOrder, MixedMode
import ADTypes
using LinearAlgebra
using Random
using Test
import ForwardDiff
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon

# Minimal models: one subject, θ = [μ], η of size one, ℓ = -(η - μ)².
struct BareModel end
NLMEEstimation.nsubjects(::BareModel) = 1
NLMEEstimation.nlme_layout(::BareModel) = NLMELayout(1, 0, [1])
NLMEEstimation.conditional_loglikelihood(::BareModel, i, θ, σ, η) = -(η[1] - θ[1])^2

struct BackendModel{B}
    backend::B
end
NLMEEstimation.nsubjects(::BackendModel) = 1
NLMEEstimation.nlme_layout(::BackendModel) = NLMELayout(1, 0, [1])
NLMEEstimation.conditional_loglikelihood(::BackendModel, i, θ, σ, η) = -(η[1] - θ[1])^2
NLMEEstimation.ad_backend(m::BackendModel) = m.backend

# The reference model with one declaration tampered, as negative controls for
# check_protocol.
struct Tampered{M, K}
    inner::M
    kind::K
end
struct WrongMuDesign end
struct WrongVariance end
NLMEEstimation.nsubjects(m::Tampered) = nsubjects(m.inner)
NLMEEstimation.nlme_layout(m::Tampered) = nlme_layout(m.inner)
NLMEEstimation.ad_backend(m::Tampered) = ad_backend(m.inner)
NLMEEstimation.conditional_loglikelihood(m::Tampered, i, θ, σ, η) = conditional_loglikelihood(m.inner, i, θ, σ, η)
NLMEEstimation.observation_model(m::Tampered) = observation_model(m.inner)
NLMEEstimation.observations(m::Tampered, i) = observations(m.inner, i)
NLMEEstimation.mu_referencing(m::Tampered{<:Any, WrongVariance}) = mu_referencing(m.inner)
function NLMEEstimation.mu_referencing(m::Tampered{<:Any, WrongMuDesign})
    mr = mu_referencing(m.inner)
    return MuReferencing(mr.theta_indices, [X .* [1 1 1 0] for X in mr.design])  # drops the weight covariate
end
NLMEEstimation.observation_moments(m::Tampered{<:Any, WrongMuDesign}, i, θ, σ, η) =
    observation_moments(m.inner, i, θ, σ, η)
function NLMEEstimation.observation_moments(m::Tampered{<:Any, WrongVariance}, i, θ, σ, η)
    μ, v = observation_moments(m.inner, i, θ, σ, η)
    return μ, 2 .* v
end

@testset "protocol" begin
    model, _ = simulated_model()
    layout = nlme_layout(model)
    ω = omega_parameters(layout, Ω_SIM)
    rng = Xoshiro(3)
    η = [0.2 .* randn(rng, 3) for _ in 1:nsubjects(model)]

    @testset "reference model conforms" begin
        @test check_protocol(model, θ_SIM, σ_SIM, ω; η, reference_backend = AutoForwardDiff(),
            population_gradient = true)
        @test check_protocol(model, θ_SIM, σ_SIM, ω; subjects = [2, 7])

        i = 5
        f(x) = conditional_loglikelihood(model, i, θ_SIM, σ_SIM, x)
        ℓ, g, H = conditional_loglikelihood_gradient_and_hessian(model, i, θ_SIM, σ_SIM, η[i])
        @test ℓ ≈ f(η[i])
        @test g ≈ ForwardDiff.gradient(f, η[i])
        @test H ≈ ForwardDiff.hessian(f, η[i])
        ℓp, gθ, gσ = conditional_loglikelihood_and_population_gradient(model, i, θ_SIM, σ_SIM, η[i])
        @test ℓp ≈ ℓ
        @test gθ ≈ ForwardDiff.gradient(t -> conditional_loglikelihood(model, i, t, σ_SIM, η[i]), θ_SIM)
        @test gσ ≈ ForwardDiff.gradient(s -> conditional_loglikelihood(model, i, θ_SIM, s, η[i]), σ_SIM)
        @test complete_data_loglikelihood(model, i, θ_SIM, σ_SIM, ω, η[i]) ≈ ℓ + re_logprior(model, ω, η[i])
        _, f_cd, g_cd, H_cd = complete_data_loglikelihood_gradient_and_hessian(model, i, θ_SIM, σ_SIM, ω, η[i])
        cd(x) = complete_data_loglikelihood(model, i, θ_SIM, σ_SIM, ω, x)
        @test f_cd ≈ cd(η[i])
        @test g_cd ≈ ForwardDiff.gradient(cd, η[i])
        @test H_cd ≈ ForwardDiff.hessian(cd, η[i])

        names = parameter_names(model)
        @test length(names.theta) == 4 && length(names.sigma) == 2 && length(names.eta) == 3
        @test parameter_names(BareModel()) == (theta = ["θ1"], sigma = String[], eta = ["η1"])
    end

    @testset "check_protocol catches broken declarations" begin
        @test_throws ProtocolViolation check_protocol(Tampered(model, WrongMuDesign()), θ_SIM, σ_SIM, ω; η)
        @test_throws ProtocolViolation check_protocol(Tampered(model, WrongVariance()), θ_SIM, σ_SIM, ω; η)
        @test_throws ProtocolViolation check_protocol(model, θ_SIM[1:3], σ_SIM, ω)
        @test_throws ProtocolViolation check_protocol(model, θ_SIM, σ_SIM, ω; η = zeros(2))
    end

    @testset "derivatives come from AD" begin
        @test conditional_loglikelihood_and_gradient(BackendModel(AutoForwardDiff()), 1, [0.5], Float64[], [2.0]) ==
              (-2.25, [-3.0])
        @test conditional_loglikelihood_gradient_and_hessian(BackendModel(AutoForwardDiff()), 1, [0.5], Float64[], [2.0]) ==
              (-2.25, [-3.0], fill(-2.0, 1, 1))
        # A model that implements no derivative method and declares no backend gets an
        # error that names both ways to provide one.
        err = try
            conditional_loglikelihood_and_gradient(BareModel(), 1, [0.0], Float64[], [0.0])
        catch e
            e
        end
        @test err isa ArgumentError && occursin("ad_backend", err.msg)
        @test_throws ArgumentError conditional_loglikelihood_and_gradient(BackendModel(ADTypes.NoAutoDiff()), 1, [0.0], Float64[], [0.0])
        @test_throws ArgumentError conditional_loglikelihood_and_gradient(BackendModel(:forwarddiff), 1, [0.0], Float64[], [0.0])

        # refused: every derivative comes from AD, never from finite differences
        # (user principle on decision BayesianRegressionModels:nlmxr 2026-10-09T11-09-31-472-1xp1b14:
        # "Why would anyone in their right minds use FD").
        for backend in (AutoFiniteDiff(), AutoFiniteDifferences(; fdm = nothing), AutoSparse(AutoFiniteDiff()),
                SecondOrder(AutoForwardDiff(), AutoFiniteDiff()), MixedMode(AutoFiniteDiff(), AutoReverseDiff()))
            m = BackendModel(backend)
            @test_throws ArgumentError conditional_loglikelihood_and_gradient(m, 1, [0.0], Float64[], [0.0])
            @test_throws ArgumentError conditional_loglikelihood_gradient_and_hessian(m, 1, [0.0], Float64[], [0.0])
            @test_throws ArgumentError conditional_loglikelihood_and_population_gradient(m, 1, [0.0], Float64[], [0.0])
            @test_throws ArgumentError check_protocol(BackendModel(AutoForwardDiff()), [0.0], Float64[], [0.0];
                reference_backend = backend)
        end
    end
end
