using NLMEEstimation
using Test
using Random
using LinearAlgebra
using Statistics: var
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
    ap, ar, ac, apc = NE._saem_mh_round!(MersenneTwister(7), model, 1:7,
        Float64[], Float64[], prior, H, ℓ, 0.5, [0.5])
    @test ap == 7
    @test 0 <= ar <= 7
    @test apc == [7] && 0 <= only(ac) <= 7
    # One batch each for the prior, joint and per-coordinate proposals.
    @test model.calls[] == 3
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

@testset "SAEM anneals Ω while γ = 1" begin
    # The floor keeps every direction at least τ times the previous Ω and never
    # lowers the M-step's Ω; an Ω that already clears it is kept.
    Ω_previous = [0.5 0.3; 0.3 0.4]
    shrunk = [0.05 0.0; 0.0 0.38]
    annealed = NE._saem_anneal_floor(shrunk, Ω_previous, 0.95)
    @test eigmin(Symmetric(annealed - 0.95 * Ω_previous)) >= -1e-12
    @test eigmin(Symmetric(annealed - shrunk)) >= -1e-12
    @test NE._saem_anneal_floor(2Ω_previous, Ω_previous, 0.95) ≈ 2Ω_previous
    layout = NLMELayout(0, 0, [1, 2])
    ω_previous = omega_parameters(layout, [0.3 0 0; 0 0.5 0.3; 0 0.3 0.4])
    ω = NE._saem_anneal(layout, ω_previous, omega_parameters(layout,
        [0.01 0 0; 0 0.05 0.0; 0 0.0 0.38]), 0.95, nothing)
    Ω = omega_matrix(layout, ω)
    @test Ω[1, 1] ≈ 0.95 * 0.3
    @test eigmin(Symmetric(Ω[2:3, 2:3] - 0.95 * [0.5 0.3; 0.3 0.4])) >= -1e-12
    ragged = RaggedNLMELayout(ntheta = 0, nsigma = 0, family_sizes = [1, 2],
        subject_families = [[1, 2], [1]])
    ωr = NE._saem_anneal(ragged, omega_parameters(ragged, [fill(0.3, 1, 1), [0.5 0.3; 0.3 0.4]]),
        omega_parameters(ragged, [fill(0.01, 1, 1), [0.05 0.0; 0.0 0.38]]), 0.95, nothing)
    @test omega_matrix(ragged, ωr)[1][1, 1] ≈ 0.95 * 0.3
    # refused: a factor of 1 or more would forbid Ω from ever shrinking while
    # γ = 1, so the exploration phase could not move towards the estimate.
    @test_throws ArgumentError NE._saem_check_options(SAEMOptions(omega_annealing = 1.0))
    @test_throws ArgumentError NE._saem_check_options(SAEMOptions(omega_annealing = -0.1))
    # Annealing only acts while γ = 1, and 0 skips it.
    model = growth_model()
    θ, σ = [1.0, 0.5], [log(0.3)]
    ω0 = omega_parameters(NLMELayout(2, 1, [2]), [0.5 0.1; 0.1 0.2])
    fit(τ) = saem(model, θ, σ, ω0; rng = Xoshiro(3),
        options = SAEMOptions(iterations = 20, burnin = 10, mcmc_burnin = 2, omega_annealing = τ))
    @test fit(0.0).omega != fit(0.95).omega
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

# Linear growth with subject covariates on the intercept and a dense 2 × 2 Ω:
# θ = [intercept, rate, covariate slopes…], `w` holds one row per subject. With
# three subjects and one covariate, the intercept, covariate slope and growth
# rate can make one draw's residuals collinear, so the covariance maximizer of a
# γ = 1 M-step is singular. With three covariates the designs identify only
# three of the four intercept coefficients: the two standardized covariates are
# exactly collinear (their raw values have the same sum for every subject).
# `prior` adds a population prior (a MAP fit).
struct CovariateGrowth
    t::Vector{Vector{Float64}}
    w::Matrix{Float64}
    y::Vector{Vector{Float64}}
    prior::Bool
end

function covariate_growth_model(prior; seed = 5)
    rng = Xoshiro(seed)
    w = [0.1, -0.2, 0.3]
    t = [[0.5, 1.0, 2.0, 4.0] for _ in w]
    y = [(1.0 + 0.8w[i] + 0.4randn(rng)) .+ (0.5 + 0.2randn(rng)) .* t[i] .+
        0.1 .* randn(rng, 4) for i in eachindex(w)]
    return CovariateGrowth(t, reshape(w, :, 1), y, prior)
end

# Three subjects with an indicator and two standardized covariates.
function wide_covariate_growth_model(prior; seed = 5)
    rng = Xoshiro(seed)
    standardize(x) = (x .- sum(x) / 3) ./ sqrt(sum(abs2, x .- sum(x) / 3) / 2)
    w = hcat([0.0, 1.0, 1.0], standardize([2.0, 5.0, 3.0]), standardize([8.0, 5.0, 7.0]))
    t = [[0.5, 1.0, 2.0, 4.0] for _ in 1:3]
    y = [(1.0 + dot([0.3, -0.2, 0.1], w[i, :]) + 0.4randn(rng)) .+
        (0.5 + 0.2randn(rng)) .* t[i] .+ 0.1 .* randn(rng, 4) for i in 1:3]
    return CovariateGrowth(t, w, y, prior)
end

NE.nsubjects(m::CovariateGrowth) = length(m.y)
NE.nlme_layout(m::CovariateGrowth) = NLMELayout(2 + size(m.w, 2), 1, [2])
NE.mu_referencing(m::CovariateGrowth) = MuReferencing(collect(1:2+size(m.w, 2)),
    [[1.0 0.0 m.w[i, :]'; 0.0 1.0 zeros(1, size(m.w, 2))] for i in eachindex(m.y)])
covariate_growth_residuals(m, i, θ, η) =
    m.y[i] .- (θ[1] + dot(view(θ, 3:length(θ)), view(m.w, i, :)) + η[1]) .-
    (θ[2] + η[2]) .* m.t[i]

function NE.conditional_loglikelihood(m::CovariateGrowth, i, θ, σ, η)
    r = covariate_growth_residuals(m, i, θ, η)
    return -sum(abs2, r) / (2exp(2σ[1])) - length(r) * (log(2π) / 2 + σ[1])
end

function NE.conditional_loglikelihood_and_population_gradient(m::CovariateGrowth,
        i, θ, σ, η)
    r = covariate_growth_residuals(m, i, θ, η)
    v = exp(2σ[1])
    return NE.conditional_loglikelihood(m, i, θ, σ, η),
        vcat(sum(r), dot(r, m.t[i]), m.w[i, :] .* sum(r)) ./ v, [sum(abs2, r) / v - length(r)]
end

NE.population_logprior(m::CovariateGrowth, θ, σ, ω) =
    m.prior ? -sum(abs2, vcat(θ, σ, ω)) / 2 : nothing
NE.population_logprior_and_gradient(m::CovariateGrowth, θ, σ, ω) =
    (population_logprior(m, θ, σ, ω), -θ, -σ, -ω)

@testset "SAEM Gaussian M-step returns its iterate where Ω's maximizer is singular" begin
    # One draw's statistics, as at γ = 1: the residuals of three subjects can be
    # made collinear, so the floored covariance maximizer is singular and the
    # coordinate updates creep towards the floor.
    layout = NLMELayout(3, 0, [2])
    w = [0.1, -0.2, 0.3]
    design = [[1.0 0.0 x; 0.0 1.0 0.0] for x in w]
    mean_phi = [1.3 0.7 1.6; 0.4 0.9 0.2]
    second_phi = zeros(2, 2, 3)
    for k in 1:3
        second_phi[:, :, k] = mean_phi[:, k] * mean_phi[:, k]'
    end
    θ0, Ω0 = zeros(3), Matrix(0.25I, 2, 2)
    gaussian(θ, Ω) = first(NE._saem_gaussian_map_value_gradient(layout, θ,
        omega_parameters(layout, Ω), 1:3, design, mean_phi, second_phi))
    θ1, Ω1 = NE._saem_gaussian_mstep(layout, θ0, Ω0, 1:3, design, mean_phi, second_phi, 1e-8)
    θ2, Ω2 = NE._saem_gaussian_mstep(layout, θ1, Ω1, 1:3, design, mean_phi, second_phi, 1e-8)
    @test gaussian(θ1, Ω1) > gaussian(θ0, Ω0)
    # The step stopped on its budget, not at the maximizer: the next one moves on.
    @test gaussian(θ2, Ω2) > gaussian(θ1, Ω1) + 0.1
    @test eigmin(Symmetric(Ω2)) < eigmin(Symmetric(Ω1))
    @test eigmin(Symmetric(Ω1)) >= 1e-8
    ragged = RaggedNLMELayout(ntheta = 3, nsigma = 0, family_sizes = [2],
        subject_families = [[1], [1], [1]])
    θr, Ωr = NE._ragged_gaussian_mstep(ragged, θ0, [Ω0], 1:3, design,
        collect(eachcol(mean_phi)), [[second_phi[:, :, k]] for k in 1:3], 1e-8, 1:3)
    @test θr ≈ θ1
    @test only(Ωr) ≈ Ω1
end

@testset "SAEM fits three subjects whose γ = 1 covariance maximizer is singular" begin
    θ0, σ0, ω0 = [1.0, 0.5, 0.0], [log(0.2)], [log(0.5), 0.0, log(0.5)]
    for prior in (false, true)
        fit = saem(covariate_growth_model(prior), θ0, σ0, ω0; rng = Xoshiro(1),
            options = SAEMOptions(iterations = 10, burnin = 5, mcmc_burnin = 5))
        @test fit.objective == (prior ? :MAP : :ML)
        @test all(isfinite, vcat(fit.theta, fit.sigma, fit.omega))
        @test eigmin(Symmetric(fit.omega_matrix)) >= 1e-8
    end
end

@testset "SAEM Gaussian M-step keeps θ along combinations the designs do not identify" begin
    # Three subjects carry four intercept coefficients, so one combination δ of
    # them (the two collinear covariates' slopes) moves no subject's φ
    # (X_k δ = 0). A Cholesky solve of the whole generalized-least-squares
    # system threw a PosDefException.
    design = NE.mu_referencing(wide_covariate_growth_model(false)).design
    N = nullspace(reduce(vcat, design))
    @test size(N) == (5, 1)
    @test size(NE._saem_identified(design, 1:5)) == (5, 4)
    @test NE._saem_identified(NE.mu_referencing(covariate_growth_model(false)).design,
        1:3) === nothing
    layout = NLMELayout(5, 0, [2])
    mean_phi = [1.3 0.7 1.6; 0.4 0.9 0.2]
    second_phi = zeros(2, 2, 3)
    for k in 1:3
        second_phi[:, :, k] = mean_phi[:, k] * mean_phi[:, k]' + [0.2 0.03; 0.03 0.1]
    end
    θ0, Ω0 = [0.5, 0.2, -0.3, 0.4, 0.1], Matrix(0.25I, 2, 2)
    θ1, Ω1 = NE._saem_gaussian_mstep(layout, θ0, Ω0, 1:5, design, mean_phi, second_phi, 1e-8)
    @test N' * θ1 ≈ N' * θ0
    _, gμ, _ = NE._saem_gaussian_map_value_gradient(layout, θ1,
        omega_parameters(layout, Ω1), 1:5, design, mean_phi, second_phi)
    # μ maximizes at the previous Ω of the alternation, which then moved < 1e-8.
    @test maximum(abs, gμ) < 1e-6
    @test isposdef(Symmetric(Ω1))
    ragged = RaggedNLMELayout(ntheta = 5, nsigma = 0, family_sizes = [2],
        subject_families = [[1], [1], [1]])
    θr, Ωr = NE._ragged_gaussian_mstep(ragged, θ0, [Ω0], 1:5, design,
        collect(eachcol(mean_phi)), [[second_phi[:, :, k]] for k in 1:3], 1e-8, 1:3)
    @test θr ≈ θ1
    @test only(Ωr) ≈ Ω1
end

@testset "SAEM MAP fit where the designs do not identify every mu-referenced θ" begin
    # The likelihood does not depend on the combination N of the intercept
    # coefficients that no subject's design carries, so the prior -|θ|²/2 alone
    # sets it, at N'θ = 0.
    θ0, σ0, ω0 = [1.0, 0.5, 0.0, 0.0, 0.0], [log(0.2)], [log(0.5), 0.0, log(0.5)]
    N = nullspace(reduce(vcat, NE.mu_referencing(wide_covariate_growth_model(true)).design))
    for options in (SAEMOptions(iterations = 2, burnin = 1, mcmc_burnin = 0,
                adapt_iterations = 0, numeric_maxiter = 2, omega_floor = 0.05),
            SAEMOptions(iterations = 30, burnin = 10, mcmc_burnin = 5)), seed in 1:3
        fit = saem(wide_covariate_growth_model(true), θ0, σ0, ω0; rng = Xoshiro(seed),
            options)
        @test fit.objective == :MAP
        @test all(isfinite, vcat(fit.theta, fit.sigma, fit.omega))
        @test abs(only(N' * fit.theta)) < 1e-5
    end
    # refused: without a prior the likelihood is flat along N, so the
    # maximum-likelihood estimate is not unique.
    err = try
        saem(wide_covariate_growth_model(false), θ0, σ0, ω0; rng = Xoshiro(1))
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("combination(s) of θ4, θ5 move", err.msg)
    @test occursin("population_logprior", err.msg)
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
        θ, σ, [2], [1], [ones(1, 1), ones(1, 1)], bank, weights, 25, 0.25, nothing,
        Ref{Union{Nothing, Matrix{Float64}}}(nothing))
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

@testset "SAEM M-step search reaches its optimum through rounding" begin
    # f is a non-quadratic optimum under a large constant, plus a few ulps of
    # rounding noise in either direction, as a long sum has. Near the optimum a
    # full step's predicted gain is below that noise; it is judged by its
    # prediction. Halving it until the noise happened to pass took 276 screens
    # and stopped at a gradient of 4.8e-8.
    M = [1.0 0.4; -0.3 0.8]
    a = [0.4, -1.2]
    gradient(x) = -M' * sinh.(M * (x - a))
    noise(x) = eps(1e11) * (Int(reinterpret(UInt64, x[1] + x[2]) % 9) - 4)
    objective(x) = (1e11 - sum(d -> cosh(d) - 1, M * (x - a)) + noise(x), gradient(x))
    screens = Ref(0)
    screen(x) = (screens[] += 1; first(objective(x)))
    x = NE._bfgs_ascent(objective, [3.0, -2.0], 100, _ -> true; screen, gtol = 1e-9)
    @test maximum(abs, gradient(x)) < 1e-9
    @test screens[] < 40
end

@testset "SAEM MALA Hessian metric" begin
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
    # seed went to log ka ≈ -2.6). Keeping φ, MALA converges, and NUTS moves its
    # chains to the shifted positions.
    model, _ = simulated_model(seed = 21, nsubjects = 24)
    θ0 = θ_SIM .+ [1.2, 0.7, 0.7, 0.0]
    ω0 = omega_parameters(nlme_layout(model), Ω_SIM)
    options = SAEMOptions(iterations = 120, burnin = 40, mcmc_burnin = 20,
        mh_rounds = 2, adapt_iterations = 80, numeric_maxiter = 4)
    for kernel in (DualAveragingMALA(), WarmupHMCNUTS(max_tree_depth = 6))
        fit = saem(model, θ0, σ_SIM, ω0; rng = MersenneTwister(21001), options, kernel)
        @test maximum(abs.(fit.theta[1:3] - θ_SIM[1:3])) < 0.3
    end
end

# A population rate in σ that trades off against a mu-referenced random effect:
# μ = b_i + A_i (1 - exp(-κ x)) ≈ b_i + A_i κ x, with log A_i = θ[2] + η[2] and
# κ = exp(σ[1]). Each subject pins A_i κ, so a numerical M-step that holds every
# φ fixed cannot move κ, and the chains cannot move A_i until κ does.
struct SaturatingSAEM
    x::Vector{Vector{Float64}}
    y::Vector{Vector{Float64}}
end
NE.nsubjects(m::SaturatingSAEM) = length(m.y)
NE.nlme_layout(::SaturatingSAEM) = NLMELayout(2, 2, [1, 1])
NE.mu_referencing(m::SaturatingSAEM) = MuReferencing([1, 2], [Matrix(1.0I, 2, 2) for _ in m.y])
NE.observation_model(::SaturatingSAEM) = GaussianObservations()
NE.observations(m::SaturatingSAEM, i::Integer) = m.y[i]
function saturating_parts(m, i, θ, σ, η)
    A, κ = exp(θ[2] + η[2]), exp(σ[1])
    e = exp.(-κ .* m.x[i])
    return A, κ, exp(σ[2]), e, (θ[1] + η[1]) .+ A .* (1 .- e)
end
function NE.conditional_loglikelihood(m::SaturatingSAEM, i::Integer, θ, σ, η)
    _, _, s, _, μ = saturating_parts(m, i, θ, σ, η)
    return -sum(abs2, m.y[i] .- μ) / (2s^2) - length(μ) * (log(2π) / 2 + log(s))
end
function NE.conditional_loglikelihood_and_gradient(m::SaturatingSAEM, i::Integer, θ, σ, η)
    A, _, s, e, μ = saturating_parts(m, i, θ, σ, η)
    w = (m.y[i] .- μ) ./ s^2
    return NE.conditional_loglikelihood(m, i, θ, σ, η), [sum(w), dot(w, A .* (1 .- e))]
end
function NE.conditional_loglikelihood_and_population_gradient(m::SaturatingSAEM, i::Integer,
        θ, σ, η)
    A, κ, s, e, μ = saturating_parts(m, i, θ, σ, η)
    r = m.y[i] .- μ
    w = r ./ s^2
    return NE.conditional_loglikelihood(m, i, θ, σ, η), [sum(w), dot(w, A .* (1 .- e))],
        [dot(w, A .* κ .* m.x[i] .* e), sum(abs2, r) / s^2 - length(r)]
end
NE.observation_moments(m::SaturatingSAEM, i::Integer, θ, σ, η) =
    let (_, _, s, _, μ) = saturating_parts(m, i, θ, σ, η)
        (μ, fill(s^2, length(μ)))
    end
function NE.observation_moments_and_jacobians(m::SaturatingSAEM, i::Integer, θ, σ, η)
    A, _, s, e, μ = saturating_parts(m, i, θ, σ, η)
    n = length(μ)
    return μ, fill(s^2, n), hcat(ones(n), A .* (1 .- e)), zeros(n, 2)
end

@testset "SAEM moves a σ rate along its random-effect ridge" begin
    rng = Xoshiro(11)
    x = [sort(6 .* rand(rng, 30)) for _ in 1:40]
    y = map(x) do xi
        A = exp(log(2.0) + 0.3randn(rng))
        (1.0 + 0.3randn(rng)) .+ A .* (1 .- exp.(-0.15 .* xi)) .+ 0.1 .* randn(rng, 30)
    end
    model = SaturatingSAEM(x, y)
    θ0, σ0, ω0 = [0.0, 0.0], [0.0, log(0.5)], [0.0, 0.0]
    reference = focei(model, θ0, σ0, ω0; options = FOCEOptions(maxeval = 3000))
    # Holding each φ fixed, the default fit ended at κ 0.24 and A 1.51 against
    # FOCE(I)'s 0.146 and 1.98, 38 log-likelihood units short; 3000 iterations
    # reached it (snag `saem-population-8975f340`).
    for seed in (1, 2)
        fit = saem(model, θ0, σ0, ω0; rng = Xoshiro(seed))
        @test abs(fit.sigma[1] - reference.sigma[1]) < 0.1
        @test abs(fit.theta[2] - reference.theta[2]) < 0.1
        @test 0 < fit.acceptance.coordinates < 1
    end
end

@testset "SAEM statistics follow the numerical step's move of the mu-referenced θ" begin
    # Holding η, a move δ of the mu-referenced θ moves every φ by D δ. The
    # transported statistics then give the closed-form step θ + δ and the same Ω.
    rng = Xoshiro(3)
    layout = NLMELayout(2, 0, [2])
    design = [[1.0 0.0; 0.0 1.0], [1.0 0.0; 0.5 1.0], [2.0 0.0; 0.0 1.0]]
    draws = [randn(rng, 2, 3) for _ in 1:4]
    mean_phi, second_phi = zeros(2, 3), zeros(2, 2, 3)
    for (j, Φ) in enumerate(draws)
        NE._saem_moments!(mean_phi, second_phi, Φ, 1 / j)
    end
    θ1, Ω1 = NE._saem_closed_form_mstep(layout, [0.0, 0.0], Matrix(1.0I, 2, 2), [1, 2],
        design, mean_phi, second_phi, 1e-8, 1:3, nothing)
    δ = [0.3, -0.2]
    bank = deepcopy(draws)
    NE._saem_transport!(mean_phi, second_phi, bank, design, δ)
    θ2, Ω2 = NE._saem_closed_form_mstep(layout, θ1, Ω1, [1, 2], design, mean_phi,
        second_phi, 1e-8, 1:3, nothing)
    @test θ2 ≈ θ1 + δ
    @test Ω2 ≈ Ω1
    @test all(bank[j] ≈ draws[j] + reduce(hcat, [X * δ for X in design]) for j in 1:4)

    ragged = RaggedNLMELayout(ntheta=1, nsigma=0, family_sizes=[1],
        subject_families=[[1], [1, 1], [1, 1, 1]])
    rdesign = [ones(1, 1), ones(2, 1), ones(3, 1)]
    rdraws = [[randn(rng, n) for n in 1:3] for _ in 1:4]
    rmean, rsecond = NE._saem_statistics(ragged, 1:3, [zeros(n) for n in 1:3])
    for (j, Φ) in enumerate(rdraws)
        NE._saem_moments!(rmean, rsecond, Φ, 1 / j)
    end
    rΩ = [ones(1, 1)]
    θr1, Ωr1 = NE._saem_closed_form_mstep(ragged, [0.0], rΩ, [1], rdesign, rmean,
        rsecond, 1e-8, 1:3, nothing)
    rbank = deepcopy(rdraws)
    NE._saem_transport!(rmean, rsecond, rbank, rdesign, [0.4])
    θr2, Ωr2 = NE._saem_closed_form_mstep(ragged, θr1, Ωr1, [1], rdesign, rmean,
        rsecond, 1e-8, 1:3, nothing)
    @test θr2 ≈ θr1 .+ 0.4
    @test Ωr2[1] ≈ Ωr1[1]
    @test all(rbank[j][k] ≈ rdraws[j][k] .+ 0.4 for j in 1:4, k in 1:3)
end

# η[1] is pinned to within 0.001 of its observation and η[2] has no likelihood,
# so its posterior is its N(0, 1) prior. The joint random walk's one scale
# follows η[1]; the coordinate sweep gives η[2] its own.
struct TwoWidthsSAEM
    y::Vector{Float64}
end
NE.nsubjects(m::TwoWidthsSAEM) = length(m.y)
NE.nlme_layout(::TwoWidthsSAEM) = NLMELayout(0, 0, [1, 1])
NE.conditional_loglikelihood(m::TwoWidthsSAEM, i::Integer, θ, σ, η) =
    -0.5e6 * (m.y[i] - η[1])^2

@testset "SAEM MH coordinate sweep follows each coordinate's width" begin
    rng = Xoshiro(5)
    model = TwoWidthsSAEM(randn(rng, 20))
    prior = NE.re_prior(model, [0.0, 0.0])
    ids = collect(1:20)
    H = vcat(model.y', zeros(1, 20))
    ℓ = NE.conditional_loglikelihoods(model, ids, Float64[], Float64[], H)
    logscale, logscales = log(0.5), fill(log(0.5), 2)
    second = Float64[]
    for sweep in 1:1200
        _, ar, ac, apc = NE._saem_mh_round!(rng, model, ids, Float64[], Float64[],
            prior, H, ℓ, exp(logscale), exp.(logscales))
        @test apc == [20, 20]
        if sweep <= 400
            logscale = NE._saem_adapt_mh_scales!(logscale, logscales, ar, 20, ac, apc, sweep)
        else
            append!(second, H[2, :])
        end
    end
    @test exp(logscales[2]) > 100 * exp(logscales[1])
    @test exp(logscale) < 0.01
    @test 0.8 < var(second) < 1.25
end
