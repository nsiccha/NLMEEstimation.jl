using NLMEEstimation
using Test
using Random
using LinearAlgebra
using DelimitedFiles
using ADTypes
using Enzyme
import NLMEEstimation as NE
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon

struct LinearSAEM
    observations::Vector{Vector{Float64}}
end

@testset "SAEM Theoph against FOCE(I) and saemix" begin
    model = theoph_model()
    ω0 = omega_parameters(nlme_layout(model), Ω_THEOPH)
    result = saem(model, θ_THEOPH, σ_THEOPH, ω0; rng = MersenneTwister(2026),
        options = SAEMOptions(iterations = 120, burnin = 40, mcmc_burnin = 20,
            mh_rounds = 2, adapt_iterations = 80, numeric_maxiter = 4,
            mstep_interval = 5))
    foce = focei(model, θ_THEOPH, σ_THEOPH, ω0;
        options = FOCEOptions(maxeval = 100, rhobeg = 0.15, rhoend = 1e-4,
            ebe_options = EBEOptions(gtol = 1e-7)))
    @test all(abs.(result.theta - foce.theta) .< [0.1, 0.1, 0.1, 0.3])
    @test abs(result.sigma[1] - foce.sigma[1]) < 0.1
    @test all(abs.(diag(result.omega_matrix) - diag(foce.omega_matrix)) .< [0.15, 0.025, 0.01])

    data, _ = readdlm(joinpath(@__DIR__, "data", "saemix_saem_theoph.csv"), ','; header = true)
    oracle = Dict(String(row[1]) => Float64(row[2]) for row in eachrow(data))
    @test abs(log(exp(result.theta[1]) / oracle["ka"])) < 0.1
    @test abs(log(exp(result.theta[2]) / oracle["CL"])) < 0.1
    @test abs(log(exp(result.theta[3]) / oracle["V"])) < 0.1
    @test abs(result.theta[4] - oracle["beta_logW(CL)"]) < 0.3
    @test abs(exp(result.sigma[1]) - oracle["residual_sd"]) < 0.1
    @test all(abs.(diag(result.omega_matrix) -
        [oracle["omega_ka"], oracle["omega_CL"], oracle["omega_V"]]) .< [0.15, 0.025, 0.01])
end

@testset "SAEM one-compartment simulation" begin
    model, _ = simulated_model(seed = 21, nsubjects = 16)
    θ0 = copy(θ_SIM)
    θ0[1] += 0.5
    ω0 = omega_parameters(nlme_layout(model), Ω_SIM)
    result = saem(model, θ0, σ_SIM, ω0; rng = MersenneTwister(15),
        options = SAEMOptions(iterations = 40, burnin = 15, mcmc_burnin = 10,
            mh_rounds = 2, adapt_iterations = 25, numeric_maxiter = 3,
            mstep_interval = 4))
    @test size(result.eta) == (3, 16)
    @test all(isfinite, (result.theta..., result.sigma..., result.omega...))
    @test isposdef(Symmetric(result.omega_matrix))
    @test abs(result.theta[1] - θ_SIM[1]) < abs(θ0[1] - θ_SIM[1])
end

NE.nsubjects(m::LinearSAEM) = length(m.observations)
NE.nlme_layout(::LinearSAEM) = NLMELayout(1, 1, [1])
NE.ad_backend(::LinearSAEM) = AutoEnzyme()
NE.mu_referencing(m::LinearSAEM) = MuReferencing([1], [ones(1, 1) for _ in m.observations])

struct FlatBatchedSAEM
    n::Int
    calls::Base.RefValue{Int}
end
NE.nsubjects(m::FlatBatchedSAEM) = m.n
NE.nlme_layout(::FlatBatchedSAEM) = NLMELayout(0, 0, [1])
NE.evaluation_style(::FlatBatchedSAEM) = BatchedEvaluation()
function NE.conditional_loglikelihoods(m::FlatBatchedSAEM, subjects, θ, σ, H)
    m.calls[] += 1
    return zeros(length(subjects))
end

function NE.conditional_loglikelihood(m::LinearSAEM, i, θ, σ, η)
    v = exp(2σ[1])
    μ = θ[1] + η[1]
    result = zero(μ + v)
    for y in m.observations[i]
        result -= (log(2π * v) + (y - μ)^2 / v) / 2
    end
    return result
end

@testset "SAEM lockstep MH" begin
    model = FlatBatchedSAEM(7, Ref(0))
    H = zeros(1, 7)
    ℓ = zeros(7)
    prior = re_prior(model, [log(0.5)])
    ap, ar = NE._saem_mh_round!(MersenneTwister(7), model, 1:7,
        Float64[], Float64[], prior, H, ℓ, 0.5)
    @test ap == 7
    @test 0 <= ar <= 7
    @test model.calls[] == 2
    @test all(isfinite, H)
end

@testset "SAEM batched numeric M-step" begin
    y = [[-0.4, -0.2], [0.1, 0.2], [0.8, 0.9], [1.1, 1.3]]
    model = JointView(LinearSAEM(y); backend = RUNTIME_REVERSE, hessian_backend = nothing)
    result = saem(model, [0.0], [log(0.4)], [log(0.5)]; rng = MersenneTwister(23),
        options = SAEMOptions(iterations = 8, burnin = 3, mcmc_burnin = 2,
            numeric_maxiter = 2, mstep_interval = 2))
    @test all(isfinite, (result.theta..., result.sigma..., result.omega...))
    @test model.calls[] > 0
end

@testset "SAEM Gaussian M-step" begin
    layout = NLMELayout(1, 0, [1])
    mean_phi = reshape([1.0, 3.0], 1, 2)
    second_phi = reshape([1.0, 9.0], 1, 1, 2)
    θ, Ω = NE._saem_gaussian_mstep(layout, [0.0], ones(1, 1), [1],
        [ones(1, 1), ones(1, 1)], mean_phi, second_phi, 1e-8)
    @test θ ≈ [2.0]
    @test Ω ≈ ones(1, 1)
end

@testset "SAEM Hessian warmup metric" begin
    model = LinearSAEM([[1.0, 2.0]])
    prior = re_prior(model, [log(0.5)])
    scale = NE._saem_hessian_metric(model, 1, [0.0], [log(0.5)], prior, [0.0])
    @test size(scale) == (1, 1)
    @test scale[1, 1] ≈ 1 / (2 / 0.5^2 + 1 / 0.5^2)
    no_hessian = FlatBatchedSAEM(1, Ref(0))
    @test NE._saem_hessian_metric(no_hessian, 1, Float64[], Float64[],
        re_prior(no_hessian, [log(0.5)]), [0.0]) ≈ fill(0.25, 1, 1)
end

@testset "SAEM linear Gaussian recovery" begin
    rng = MersenneTwister(44)
    y = [[0.7 + 0.6randn(rng) + 0.2randn(rng) for _ in 1:4] for _ in 1:18]
    model = LinearSAEM(y)
    θ0, σ0, ω0 = [-0.3], [log(0.65)], [log(0.9)]
    result = saem(model, θ0, σ0, ω0; rng = MersenneTwister(15),
        options = SAEMOptions(iterations = 36, burnin = 12, mcmc_burnin = 8,
            mh_rounds = 2, adapt_iterations = 20, numeric_maxiter = 4))
    @test size(result.eta) == (1, 18)
    @test all(isfinite, (result.theta..., result.sigma..., result.omega...))
    @test abs(result.theta[1] - 0.7) < abs(θ0[1] - 0.7)
    @test abs(result.sigma[1] - log(0.2)) < abs(σ0[1] - log(0.2))
    @test result.omega_matrix[1, 1] > 0
    @test 0 < result.acceptance.prior < 1
    @test 0 < result.acceptance.random_walk < 1
    @test result.rw_scale > 0
end
