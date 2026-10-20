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

struct HybridBatchedSAEM
    y::Vector{Float64}
    batch_calls::Base.RefValue{Int}
    population_gradient_calls::Base.RefValue{Int}
end
NE.evaluation_style(::HybridBatchedSAEM) = BatchedEvaluation()
NE.saem_population_gradient_style(::HybridBatchedSAEM) = PerSubjectEvaluation()
NE.saem_population_gradient_style(::HybridBatchedSAEM, ::Val{:mixture}) = BatchedEvaluation()
NE.ad_backend(::HybridBatchedSAEM) = RUNTIME_REVERSE
function NE.conditional_loglikelihoods(m::HybridBatchedSAEM, subjects, θ, σ, H)
    m.batch_calls[] += 1
    [-0.5 * (m.y[i] - θ[1] - σ[1] - H[1, k])^2 for (k, i) in enumerate(subjects)]
end
function NE.conditional_loglikelihood_and_population_gradient(m::HybridBatchedSAEM, i, θ, σ, η)
    m.population_gradient_calls[] += 1
    residual = m.y[i] - θ[1] - σ[1] - η[1]
    return -0.5 * residual^2, [residual], [residual]
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
            numeric_maxiter = 2, mstep_interval = 2, numeric_mstep = :online))
    @test all(isfinite, (result.theta..., result.sigma..., result.omega...))
    @test model.calls[] > 0
    @test result.numeric_mstep == :online
    historical = saem(model, [0.0], [log(0.4)], [log(0.5)];
        rng = MersenneTwister(23),
        options = SAEMOptions(iterations = 8, burnin = 3, mcmc_burnin = 2,
            numeric_maxiter = 2, mstep_interval = 2, numeric_mstep = :mixture))
    @test all(isfinite, (historical.theta..., historical.sigma..., historical.omega...))
    @test historical.numeric_mstep == :mixture
end

@testset "SAEM population gradient style is operation-specific" begin
    model = HybridBatchedSAEM([1.0, 1.5], Ref(0), Ref(0))
    @test evaluation_style(model) isa BatchedEvaluation
    @test saem_population_gradient_style(model) isa PerSubjectEvaluation
    @test saem_population_gradient_style(model, Val(:online)) isa PerSubjectEvaluation
    @test saem_population_gradient_style(model, Val(:mixture)) isa BatchedEvaluation
    θ, σ = [0.2], [-0.1]
    Φ = reshape([0.3, 0.4], 1, 2)
    design = [zeros(1, 0) for _ in 1:2]
    value, gradient = NE._saem_numeric_value_gradient(vcat(θ, σ), model, 1:2,
        θ, σ, [1, 2], Int[], design, [Φ], [1.0], Val(:online))
    @test value ≈ -0.68
    @test gradient ≈ [1.6, 1.6]
    @test model.population_gradient_calls[] == 2
    @test model.batch_calls[] == 0
    mixture_value, mixture_gradient = NE._saem_numeric_value_gradient(vcat(θ, σ),
        model, 1:2, θ, σ, [1, 2], Int[], design, [Φ], [1.0], Val(:mixture))
    @test mixture_value ≈ value
    @test mixture_gradient ≈ gradient
    @test model.population_gradient_calls[] == 2
    @test model.batch_calls[] > 0
    @test conditional_loglikelihoods(model, 1:2, θ, σ, Φ) ≈ [-0.18, -0.5]
    @test model.batch_calls[] > 1
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

@testset "SAEM online numerical M-step" begin
    model = LinearSAEM([[0.2, 0.5], [1.0, 1.3]])
    θ = [0.6]
    σ = [log(0.5)]
    Φ = reshape([0.1, -0.2], 1, 2)
    bank = Matrix{Float64}[]
    weights = Float64[]
    NE._saem_record!(Val(:online), bank, weights, Φ, 0.25)
    @test length(bank) == 1
    @test weights == [1.0]
    θnext, σnext = NE._saem_numeric_update(Val(:online), model, 1:2,
        θ, σ, [2], [1], [ones(1, 1), ones(1, 1)], bank, weights, 25, 0.25)
    residuals = [0.2 - 0.1, 0.5 - 0.1, 1.0 + 0.2, 1.3 + 0.2]
    optimum = log(sqrt(sum(abs2, residuals) / length(residuals)))
    @test θnext == θ
    @test σnext[1] ≈ 0.75σ[1] + 0.25optimum atol=1e-5
    NE._saem_record!(Val(:online), bank, weights, Φ .+ 1, 0.1)
    @test length(bank) == 1
    @test bank[1] ≈ Φ .+ 1

    historical = Matrix{Float64}[]
    historical_weights = Float64[]
    NE._saem_record!(Val(:mixture), historical, historical_weights, Φ, 1.0)
    NE._saem_record!(Val(:mixture), historical, historical_weights, Φ .+ 1, 0.25)
    @test length(historical) == 2
    @test historical_weights ≈ [0.75, 0.25]
    @test NE._saem_mode_name(NE._saem_numeric_mode(:auto, [2], 1)) == :online
    @test NE._saem_mode_name(NE._saem_numeric_mode(:auto, [1, 2], 1)) == :mixture
    @test_throws ArgumentError NE._saem_check_options(SAEMOptions(numeric_mstep=:other))
end

@testset "SAEM Hessian warmup metric" begin
    model = LinearSAEM([[1.0, 2.0]])
    prior = re_prior(model, [log(0.5)])
    scale = NE._saem_hessian_metric(model, 1, [0.0], [log(0.5)], prior, [0.0])
    @test size(scale) == (1, 1)
    @test scale[1, 1] ≈ 1 / (2 / 0.5^2 + 1 / 0.5^2)
    @test NE._saem_hessian_metric(model, 1, [0.0], [log(0.5)], prior,
        [0.0]; diagonal=true) ≈ [scale[1, 1]]
    no_hessian = FlatBatchedSAEM(1, Ref(0))
    @test NE._saem_hessian_metric(no_hessian, 1, Float64[], Float64[],
        re_prior(no_hessian, [log(0.5)]), [0.0]) ≈ fill(0.25, 1, 1)
    @test NE._saem_hessian_metric(no_hessian, 1, Float64[], Float64[],
        re_prior(no_hessian, [log(0.5)]), [0.0]; diagonal=true) ≈ [0.25]
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

@testset "SAEM keeps each subject's φ across the mu-referenced update" begin
    # Re-centring shifts η so that φ = Dθ + η is unchanged, in both layouts, and
    # leaves η alone when no mu-referenced coordinate moved.
    θ_old, θ_new, indices = [0.2, -0.1, 0.4], [0.5, 0.3, 0.4], [1, 2]
    design = [[1.0 0.0; 0.0 2.0], [0.5 1.0; 0.0 1.0]]
    H = [0.1 -0.2; 0.3 0.4]
    Φ = NE._saem_phi(H, θ_old, indices, design)
    @test NE._saem_recenter!(H, θ_old, θ_new, indices, design)
    @test NE._saem_phi(H, θ_new, indices, design) ≈ Φ
    @test !NE._saem_recenter!(H, θ_new, θ_new, indices, design)
    @test !NE._saem_recenter!(H, θ_old, θ_new, Int[], [zeros(2, 0), zeros(2, 0)])
    ragged = [[0.1, 0.3], [-0.2, 0.4, 0.7]]
    ragged_design = [[1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0; 0.0 0.0]]
    Φr = NE._saem_phi(ragged, θ_old, indices, ragged_design)
    @test NE._saem_recenter!(ragged, θ_old, θ_new, indices, ragged_design)
    @test all(NE._saem_phi(ragged, θ_new, indices, ragged_design) .≈ Φr)

    # From a far start with an M-step every iteration, holding η fixed let θ
    # drift by about mean(η) per update while MALA's small steps barely moved the
    # chains: none of 18 such simulated fits converged (brief `dqtdys`; this
    # seed went to log ka ≈ -2.6). Keeping φ, MALA converges, and NUTS restarts
    # its fixed kernels at the shifted positions.
    model, _ = simulated_model(seed = 21, nsubjects = 24)
    θ0 = θ_SIM .+ [1.2, 0.7, 0.7, 0.0]
    ω0 = omega_parameters(nlme_layout(model), Ω_SIM)
    options = SAEMOptions(iterations = 120, burnin = 40, mcmc_burnin = 20,
        mh_rounds = 2, adapt_iterations = 80, numeric_maxiter = 4)
    for kernel in (DualAveragingMALA(), WarmupHMCNUTS(warmup_draws = 8,
            warmup_evaluations = 50, stepsize_adaptation_limit = 10, max_tree_depth = 6))
        fit = saem(model, θ0, σ_SIM, ω0; rng = MersenneTwister(21001), options, kernel)
        @test maximum(abs.(fit.theta[1:3] - θ_SIM[1:3])) < 0.3
    end
end
