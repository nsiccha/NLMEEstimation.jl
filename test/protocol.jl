using NLMEEstimation
using NLMEEstimation.ReferenceModels
import NLMEEstimation as NE
import DifferentiationInterface as DI
import ADTypes
using LinearAlgebra
using Random
using Test
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon

# Minimal models: one subject, θ = [μ], η of size one, ℓ = -(η - μ)².
struct BareModel end
NE.nsubjects(::BareModel) = 1
NE.nlme_layout(::BareModel) = NLMELayout(1, 0, [1])
NE.conditional_loglikelihood(::BareModel, i, θ, σ, η) = -(η[1] - θ[1])^2

struct BackendModel{B}
    backend::B
end
NE.nsubjects(::BackendModel) = 1
NE.nlme_layout(::BackendModel) = NLMELayout(1, 0, [1])
NE.conditional_loglikelihood(::BackendModel, i, θ, σ, η) = -(η[1] - θ[1])^2
NE.ad_backend(m::BackendModel) = m.backend

# A batched model that forgot its batched method.
struct HollowBatched end
NE.evaluation_style(::HollowBatched) = BatchedEvaluation()
NE.nsubjects(::HollowBatched) = 2
NE.nlme_layout(::HollowBatched) = NLMELayout(1, 0, [1])

# The reference model with one declaration tampered, as negative controls for
# check_protocol.
struct Tampered{M, K}
    inner::M
    kind::K
end
struct WrongMuDesign end
struct WrongVariance end
NE.nsubjects(m::Tampered) = nsubjects(m.inner)
NE.nlme_layout(m::Tampered) = nlme_layout(m.inner)
NE.ad_backend(m::Tampered) = ad_backend(m.inner)
NE.conditional_loglikelihood(m::Tampered, i, θ, σ, η) = conditional_loglikelihood(m.inner, i, θ, σ, η)
NE.observation_model(m::Tampered) = observation_model(m.inner)
NE.observations(m::Tampered, i) = observations(m.inner, i)
NE.mu_referencing(m::Tampered{<:Any, WrongVariance}) = mu_referencing(m.inner)
function NE.mu_referencing(m::Tampered{<:Any, WrongMuDesign})
    mr = mu_referencing(m.inner)
    return MuReferencing(mr.theta_indices, [X .* [1 1 1 0] for X in mr.design])  # drops the weight covariate
end
NE.observation_moments(m::Tampered{<:Any, WrongMuDesign}, i, θ, σ, η) = observation_moments(m.inner, i, θ, σ, η)
function NE.observation_moments(m::Tampered{<:Any, WrongVariance}, i, θ, σ, η)
    μ, v = observation_moments(m.inner, i, θ, σ, η)
    return μ, 2 .* v
end

# A batched model whose subjects are coupled: each subject's density also depends
# on the batch mean of η, which breaks conditional independence.
struct Coupled{M}
    inner::M
end
NE.evaluation_style(::Coupled) = BatchedEvaluation()
NE.nsubjects(c::Coupled) = nsubjects(c.inner)
NE.nlme_layout(c::Coupled) = nlme_layout(c.inner)
NE.ad_backend(c::Coupled) = RUNTIME_REVERSE
NE.conditional_loglikelihoods(c::Coupled, subjects, θ, σ, H) =
    [conditional_loglikelihood(c.inner, i, θ, σ, H[:, k]) - sum(H)^2 for (k, i) in enumerate(subjects)]

_cl(η, model, i, θ, σ) = conditional_loglikelihood(model, i, θ, σ, η)
_cl_θ(θ, model, i, σ, η) = conditional_loglikelihood(model, i, θ, σ, η)
_cl_σ(σ, model, i, θ, η) = conditional_loglikelihood(model, i, θ, σ, η)

@testset "protocol" begin
    model, _ = simulated_model()
    layout = nlme_layout(model)
    ω = omega_parameters(layout, Ω_SIM)
    n = nsubjects(model)
    H = 0.2 .* randn(Xoshiro(3), 3, n)

    @testset "reference model conforms (per-subject evaluation)" begin
        @test evaluation_style(model) === PerSubjectEvaluation()
        @test provides_eta_hessian(model)
        @test check_protocol(model, θ_SIM, σ_SIM, ω; η = H, reference_backend = RUNTIME_REVERSE,
            reference_hessian_backend = RUNTIME_HESSIAN, population_gradient = true)
        @test check_protocol(model, θ_SIM, σ_SIM, ω; subjects = [2, 7])

        # The model's static reverse-mode Enzyme against the runtime-activity reference.
        i = 5
        ctx = (DI.Constant(model), DI.Constant(i), DI.Constant(θ_SIM), DI.Constant(σ_SIM))
        ℓ, g, Hi = conditional_loglikelihood_gradient_and_hessian(model, i, θ_SIM, σ_SIM, H[:, i])
        @test ℓ ≈ conditional_loglikelihood(model, i, θ_SIM, σ_SIM, H[:, i])
        @test g ≈ DI.gradient(_cl, RUNTIME_REVERSE, H[:, i], ctx...)
        @test Hi ≈ DI.hessian(_cl, RUNTIME_HESSIAN, H[:, i], ctx...)
        ℓp, gθ, gσ = conditional_loglikelihood_and_population_gradient(model, i, θ_SIM, σ_SIM, H[:, i])
        @test ℓp ≈ ℓ
        @test gθ ≈ DI.gradient(_cl_θ, RUNTIME_REVERSE, θ_SIM, DI.Constant(model), DI.Constant(i), DI.Constant(σ_SIM), DI.Constant(H[:, i]))
        @test gσ ≈ DI.gradient(_cl_σ, RUNTIME_REVERSE, σ_SIM, DI.Constant(model), DI.Constant(i), DI.Constant(θ_SIM), DI.Constant(H[:, i]))
        @test complete_data_loglikelihood(model, i, θ_SIM, σ_SIM, ω, H[:, i]) ≈ ℓ + re_logprior(model, ω, H[:, i])

        # The batched entry points are the per-subject ones, column by column.
        ℓb, G = conditional_loglikelihoods_and_gradients(model, 1:n, θ_SIM, σ_SIM, H)
        @test ℓb ≈ conditional_loglikelihoods(model, 1:n, θ_SIM, σ_SIM, H)
        @test G[:, i] ≈ g
        @test conditional_loglikelihood_hessian_blocks(model, [i], θ_SIM, σ_SIM, H[:, [i]])[1] ≈ Hi

        names = parameter_names(model)
        @test length(names.theta) == 4 && length(names.sigma) == 2 && length(names.eta) == 3
        @test parameter_names(BareModel()) == (theta = ["θ1"], sigma = String[], eta = ["η1"])
    end

    @testset "batched evaluation (joint-program models)" begin
        joint = JointView(model)
        gradient_only = JointView(model; hessian_backend = nothing)
        @test evaluation_style(joint) === BatchedEvaluation()
        @test provides_eta_hessian(joint)
        @test !provides_eta_hessian(gradient_only)
        @test check_protocol(joint, θ_SIM, σ_SIM, ω; η = H, reference_backend = RUNTIME_REVERSE,
            reference_hessian_backend = RUNTIME_HESSIAN, population_gradient = true)
        @test check_protocol(gradient_only, θ_SIM, σ_SIM, ω; η = H)

        # Derived per-subject methods and the default batched derivatives (AD of the
        # summed batch log-density; Hessian blocks from one HVP per η component)
        # equal the per-subject model's.
        subjects = [3, 8, 11]
        Hs = H[:, subjects]
        ℓj, Gj = conditional_loglikelihoods_and_gradients(joint, subjects, θ_SIM, σ_SIM, Hs)
        ℓm, Gm = conditional_loglikelihoods_and_gradients(model, subjects, θ_SIM, σ_SIM, Hs)
        @test ℓj ≈ ℓm
        @test Gj ≈ Gm
        @test conditional_loglikelihood_hessian_blocks(joint, subjects, θ_SIM, σ_SIM, Hs) ≈
              conditional_loglikelihood_hessian_blocks(model, subjects, θ_SIM, σ_SIM, Hs)
        @test conditional_loglikelihood(joint, 8, θ_SIM, σ_SIM, Hs[:, 2]) ≈ ℓm[2]
        # Gradient-only: second order is unavailable and says so.
        @test_throws ArgumentError conditional_loglikelihood_hessian_blocks(gradient_only, subjects, θ_SIM, σ_SIM, Hs)
    end

    @testset "check_protocol catches broken declarations" begin
        @test_throws ProtocolViolation check_protocol(Tampered(model, WrongMuDesign()), θ_SIM, σ_SIM, ω; η = H)
        @test_throws ProtocolViolation check_protocol(Tampered(model, WrongVariance()), θ_SIM, σ_SIM, ω; η = H)
        @test_throws ProtocolViolation check_protocol(Coupled(model), θ_SIM, σ_SIM, ω; η = H)
        @test_throws ProtocolViolation check_protocol(model, θ_SIM[1:3], σ_SIM, ω)
        @test_throws ProtocolViolation check_protocol(model, θ_SIM, σ_SIM, ω; η = zeros(2))
    end

    @testset "missing methods fail with instructions" begin
        # No derivative method and no backend: the error names both remedies.
        err = try
            conditional_loglikelihood_and_gradient(BareModel(), 1, [0.0], Float64[], [0.0])
        catch e
            e
        end
        @test err isa ArgumentError && occursin("ad_backend", err.msg)
        @test !provides_eta_hessian(BareModel())
        # A style without its required evaluation method.
        @test_throws ArgumentError conditional_loglikelihoods(HollowBatched(), [1, 2], [0.0], Float64[], zeros(1, 2))
        @test_throws ArgumentError conditional_loglikelihood(HollowBatched(), 1, [0.0], Float64[], [0.0])
        @test_throws ArgumentError conditional_loglikelihood_and_gradient(BackendModel(ADTypes.NoAutoDiff()), 1, [0.0], Float64[], [0.0])
        @test_throws ArgumentError conditional_loglikelihood_and_gradient(BackendModel(:enzyme), 1, [0.0], Float64[], [0.0])
    end

    @testset "derivatives from the declared backend (Enzyme)" begin
        m = BackendModel(REVERSE)
        @test conditional_loglikelihood_and_gradient(m, 1, [0.5], Float64[], [2.0]) == (-2.25, [-3.0])
        ℓ, g, Hm = conditional_loglikelihood_gradient_and_hessian(m, 1, [0.5], Float64[], [2.0])
        @test (ℓ, g) == (-2.25, [-3.0])
        @test Hm ≈ fill(-2.0, 1, 1)
        @test check_protocol(m, [0.5], Float64[], [0.0]; η = [2.0], reference_backend = RUNTIME_REVERSE,
            reference_hessian_backend = RUNTIME_HESSIAN)
    end
end
