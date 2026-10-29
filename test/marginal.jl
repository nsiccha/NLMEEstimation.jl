using NLMEEstimation
using LinearAlgebra
using Random
using Test
using DelimitedFiles
import ForwardDiff
import NLMEEstimation as NE
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon

@testset "marginal likelihood: exact linear-Gaussian" begin
    model = growth_model()
    θ, σ, ω = [0.9, 0.45], [log(0.32)], [log(0.68), 0.08, log(0.46)]
    x = vcat(θ, σ, ω)
    exact = growth_exact(model, x)
    nobs = sum(length, model.y)

    # The posterior of each subject is Gaussian, so adaptive quadrature centred
    # at the mode with the exact curvature is exact at any number of nodes.
    laplace = marginal_loglikelihood(model, θ, σ, ω; approximation = AdaptiveGaussHermite(nodes = 1))
    @test laplace.loglikelihood ≈ exact atol = 1e-8
    @test laplace.subject_loglikelihoods ≈ [growth_exact_subject(model, i, x)
        for i in 1:NE.nsubjects(model)] atol = 1e-8
    @test laplace.standard_error == 0 && laplace.effective_sample_size === nothing
    @test laplace.evaluations == 1
    agq = marginal_loglikelihood(model, θ, σ, ω; approximation = AdaptiveGaussHermite(nodes = 4))
    @test agq.loglikelihood ≈ exact atol = 1e-8
    @test agq.evaluations == 4^2

    # A Gaussian proposal with the exact mode and curvature is the posterior
    # itself: every importance weight is the subject's marginal.
    gaussian = marginal_loglikelihood(model, θ, σ, ω;
        approximation = ImportanceSampling(draws = 50, df = Inf), rng = Xoshiro(1))
    @test gaussian.loglikelihood ≈ exact atol = 1e-8
    @test gaussian.standard_error < 1e-8

    is = marginal_loglikelihood(model, θ, σ, ω;
        approximation = ImportanceSampling(draws = 4000), rng = Xoshiro(2))
    @test abs(is.loglikelihood - exact) < 4 * is.standard_error
    @test is.standard_error < 0.02
    @test is.standard_error ≈ sqrt(sum(abs2, is.subject_standard_errors))
    @test all(>(1000), is.effective_sample_size)
    @test is.evaluations == 4000

    @test is.objective == :ML && is.ofv == -2 * is.loglikelihood
    @test is.nobs == nobs && is.nparameters == 6
    @test is.aic ≈ -2 * is.loglikelihood + 12
    @test is.bic ≈ -2 * is.loglikelihood + 6 * log(nobs)
    @test is.subjects == 1:NE.nsubjects(model)
    @test is.point == x

    focei_ll = marginal_loglikelihood(model, θ, σ, ω; approximation = FOCEApproximation())
    @test focei_ll.loglikelihood ≈ exact atol = 1e-7
    @test focei_ll.evaluations === missing

    fit = focei(model, θ, σ, ω; free = [1, 2, 3],
        options = FOCEOptions(maxeval = 300, rhoend = 1e-7))
    from_fit = marginal_loglikelihood(model, fit; approximation = AdaptiveGaussHermite(nodes = 2))
    @test from_fit.nparameters == 3
    @test from_fit.loglikelihood ≈ fit.loglikelihood atol = 1e-6

    prior_model = growth_model(prior_sd = 2.0)
    posterior = marginal_loglikelihood(prior_model, θ, σ, ω; approximation = AdaptiveGaussHermite(nodes = 1))
    @test posterior.objective == :MAP
    @test posterior.population_logprior ≈ -sum(abs2, x) / 8
    @test posterior.logposterior ≈ posterior.loglikelihood + posterior.population_logprior
    @test posterior.ofv ≈ -2 * posterior.logposterior
    @test posterior.aic ≈ -2 * posterior.loglikelihood + 12
end

@testset "marginal likelihood: ragged nested effects" begin
    model = occasions_model()
    layout = nlme_layout(model)
    θ, σ, ω = [1.9], [log(0.27)], [log(0.55), log(0.35)]
    x = vcat(θ, σ, ω)
    exact = occasions_exact(model, x)

    # No η Hessian: the curvature comes from central differences of gradients.
    laplace = marginal_loglikelihood(model, θ, σ, ω; approximation = AdaptiveGaussHermite(nodes = 1))
    @test laplace.loglikelihood ≈ exact atol = 1e-6
    agq = marginal_loglikelihood(model, θ, σ, ω; approximation = AdaptiveGaussHermite(nodes = 3))
    @test agq.loglikelihood ≈ exact atol = 1e-6
    @test agq.evaluations == 3^maximum(i -> neta(layout, i), 1:NE.nsubjects(model))

    is = marginal_loglikelihood(model, θ, σ, ω;
        approximation = ImportanceSampling(draws = 3000), rng = Xoshiro(3))
    @test abs(is.loglikelihood - exact) < 4 * is.standard_error
    @test is.standard_error < 0.03

    # No observation model is declared: BIC needs an explicit count.
    @test is.nobs === missing && is.bic === missing
    nobs = sum(o -> sum(length, o), model.y)
    counted = marginal_loglikelihood(model, θ, σ, ω;
        approximation = AdaptiveGaussHermite(nodes = 1), nobs)
    @test counted.bic ≈ -2 * counted.loglikelihood + 4 * log(nobs)
end

@testset "covariance step from quadrature and importance sampling" begin
    model = growth_model()
    fit = focei(model, [0.8, 0.4], [log(0.4)], [log(0.6), 0.0, log(0.5)];
        options = FOCEOptions(maxeval = 2000, rhoend = 1e-9,
            ebe_options = EBEOptions(gtol = 1e-11)))
    x̂ = vcat(fit.theta, fit.sigma, fit.omega)
    exact_R = -ForwardDiff.hessian(z -> growth_exact(model, z), x̂)
    agq = covariance_step(model, fit.theta, fit.sigma, fit.omega; method = :r,
        approximation = AdaptiveGaussHermite(nodes = 5))
    @test agq.R ≈ exact_R rtol = 1e-5
    is = covariance_step(model, fit.theta, fit.sigma, fit.omega; method = :r,
        approximation = ImportanceSampling(draws = 2000), rng = Xoshiro(4))
    @test is.R ≈ exact_R rtol = 0.05

    # Importance sampling on a ragged SAEM fit. Its draws are fixed in the
    # mu-referenced coordinates, so perturbing θ (mu-referenced) or ω reuses the
    # conditional values: only σ and its pairs evaluate the model. (Ragged fits
    # default to Laplace or Louis's identity: test/ragged_covariance.jl.)
    occasions = occasions_model()
    saem_fit = saem(occasions, [1.9], [log(0.3)], [log(0.5), log(0.4)];
        rng = Xoshiro(5), options = SAEMOptions(iterations = 120, burnin = 60))
    @test saem_fit isa RaggedSAEMResult
    draws = 400
    occasions.calls[] = 0
    result = covariance_step(occasions, saem_fit; method = :r,
        approximation = ImportanceSampling(draws = draws), rng = Xoshiro(6))
    @test result.approximation == ImportanceSampling(draws = draws)
    @test result.names == ["θ1", "σ1", "ω1[1,1]", "ω2[1,1]"]
    x̃ = vcat(saem_fit.theta, saem_fit.sigma, saem_fit.omega)
    occasions_R = -ForwardDiff.hessian(z -> occasions_exact(occasions, z), x̃)
    @test result.R ≈ occasions_R rtol = 0.05
    # Three distinct (σ) keys: the centre and σ ± h. Mode search and curvature
    # take well under one draw's worth of calls per subject.
    nsub = NE.nsubjects(occasions)
    @test occasions.calls[] < 3 * draws * nsub + 200 * nsub
end

@testset "reference model: Bateman kernel for either rate ordering" begin
    bateman = NLMEEstimation.ReferenceModels.bateman
    direct(ka, ke, t) = (exp(-ke * t) - exp(-ka * t)) / (ka - ke)
    for (ka, ke, t) in ((1.5, 0.1, 3.0), (0.1, 1.5, 3.0), (2.0, 2.0 + 1e-9, 4.0), (0.8, 0.8, 2.0))
        reference = ka == ke ? t * exp(-ke * t) : direct(ka, ke, t)
        @test bateman(ka, ke, t) ≈ reference rtol = 1e-6
        @test bateman(ka, ke, t) == bateman(ke, ka, t)
    end
    # Elimination much faster than absorption: the factored form used to
    # evaluate 0 · Inf here.
    @test bateman(1.0, 1000.0, 5.0) ≈ exp(-5.0) / 999 rtol = 1e-12
    @test bateman(1000.0, 1.0, 5.0) ≈ exp(-5.0) / 999 rtol = 1e-12
end

@testset "marginal likelihood: saemix oracle on theophylline" begin
    data = readdlm(joinpath(@__DIR__, "data", "saemix_loglik_theoph.csv"), ','; header = true)[1]
    value(name) = Float64(data[findfirst(==(name), data[:, 1]), 2])
    model = theoph_model()
    θ = [log(value("ka")), log(value("CL")), log(value("V")), value("beta_logW(CL)")]
    σ = [log(value("residual_sd"))]
    ω = log.(sqrt.([value("omega_ka"), value("omega_CL"), value("omega_V")]))
    agq9 = marginal_loglikelihood(model, θ, σ, ω; approximation = AdaptiveGaussHermite(nodes = 9))
    agq12 = marginal_loglikelihood(model, θ, σ, ω; approximation = AdaptiveGaussHermite(nodes = 12))
    @test abs(agq9.loglikelihood - agq12.loglikelihood) < 2e-3  # measured 5e-4
    is = marginal_loglikelihood(model, θ, σ, ω;
        approximation = ImportanceSampling(draws = 20000), rng = Xoshiro(2))
    @test abs(is.loglikelihood - agq12.loglikelihood) < 4 * is.standard_error
    # saemix 3.5 at its own estimate: Gauss–Hermite (12 nodes over ±4 SD) and
    # importance sampling (50,000 t₄ draws). Measured gaps: 0.007 and 0.011.
    @test abs(agq12.loglikelihood - value("ll_gq")) < 0.02
    @test abs(agq12.loglikelihood - value("ll_is")) < 0.03
    # Same parameter count (4 θ, the residual SD, 3 variances) and AIC rule.
    @test agq12.nparameters == 8
    @test (value("aic_is") + 2value("ll_is")) / 2 ≈ 8
    @test agq12.nobs == value("nobs")
    @test agq12.aic ≈ -2 * agq12.loglikelihood + 16
    # saemix's BIC penalizes log(number of subjects); `bic` uses log(N_obs).
    @test (value("bic_is") + 2value("ll_is")) / 8 ≈ log(value("nsubjects"))
    @test agq12.bic ≈ -2 * agq12.loglikelihood + 8 * log(value("nobs"))
end
