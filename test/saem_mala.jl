using Test
using Random
using Statistics
using NLMEEstimation
import NLMEEstimation as NE
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon

struct BatchedGaussianMALA
    y::Vector{Float64}
    precision::Vector{Float64}
    gradient_calls::Base.RefValue{Int}
end
NE.nsubjects(m::BatchedGaussianMALA) = length(m.y)
NE.nlme_layout(::BatchedGaussianMALA) = NLMELayout(0, 0, [1])
NE.evaluation_style(::BatchedGaussianMALA) = BatchedEvaluation()
NE.provides_eta_hessian(::BatchedGaussianMALA) = true
NE.conditional_loglikelihoods(m::BatchedGaussianMALA, ids, θ, σ, H) =
    [-0.5 * m.precision[i] * (H[1, j] - m.y[i])^2 for (j, i) in enumerate(ids)]
function NE.conditional_loglikelihoods_and_gradients(m::BatchedGaussianMALA,
        ids, θ, σ, H)
    m.gradient_calls[] += 1
    return NE.conditional_loglikelihoods(m, ids, θ, σ, H),
        reshape([-m.precision[i] * (H[1, j] - m.y[i])
            for (j, i) in enumerate(ids)], 1, :)
end
function NE.conditional_loglikelihood_gradient_and_hessian(m::BatchedGaussianMALA,
        i, θ, σ, η)
    d = η[1] - m.y[i]
    return -0.5 * m.precision[i] * d^2, [-m.precision[i] * d],
        fill(-m.precision[i], 1, 1)
end

struct DomainBatchedMALA end
NE.nsubjects(::DomainBatchedMALA) = 2
NE.nlme_layout(::DomainBatchedMALA) = NLMELayout(0, 0, [1])
NE.evaluation_style(::DomainBatchedMALA) = BatchedEvaluation()
NE.conditional_loglikelihoods(::DomainBatchedMALA, ids, θ, σ, H) =
    [abs(H[1, j]) > 1 ? NaN : -H[1, j]^2 / 2 for j in eachindex(ids)]
function NE.conditional_loglikelihoods_and_gradients(m::DomainBatchedMALA,
        ids, θ, σ, H)
    return NE.conditional_loglikelihoods(m, ids, θ, σ, H),
        reshape([abs(H[1, j]) > 1 ? NaN : -H[1, j]
            for j in eachindex(ids)], 1, :)
end

struct ADBatchedMALA end
NE.nsubjects(::ADBatchedMALA) = 3
NE.nlme_layout(::ADBatchedMALA) = NLMELayout(0, 0, [1])
NE.evaluation_style(::ADBatchedMALA) = BatchedEvaluation()
NE.ad_backend(::ADBatchedMALA) = RUNTIME_REVERSE
NE.conditional_loglikelihoods(::ADBatchedMALA, ids, θ, σ, H) =
    [-0.5 * (H[1, j] - 0.2i)^2 for (j, i) in enumerate(ids)]

@testset "Batched MALA detailed balance and per-subject adaptation" begin
    model = BatchedGaussianMALA([0.4, -0.3, 1.0], [0.5, 5.0, 50.0], Ref(0))
    prior = re_prior(model, [0.0])
    kernel = DualAveragingMALA(initial_stepsize=0.2)
    states = NE._saem_mala_states(kernel, 3)
    H = zeros(1, 3)
    rng = MersenneTwister(128)
    NE._saem_mala_round!(rng, kernel, states, model, 1:3,
        Float64[], Float64[], prior, H)
    @test model.gradient_calls[] == 2
    @test all(s -> s.count == 1, states)
    for _ in 2:100
        NE._saem_mala_round!(rng, kernel, states, model, 1:3,
            Float64[], Float64[], prior, H)
    end
    @test model.gradient_calls[] == 200
    @test all(s -> s.count == 100, states)
    @test length(unique(round.([s.log_stepsize for s in states]; digits=4))) > 1
    cached = BatchedGaussianMALA([0.4, -0.3, 1.0], [0.5, 5.0, 50.0], Ref(0))
    HC = zeros(1, 3)
    cs = NE._saem_mala_states(kernel, 3)
    cache = NE._saem_mala_cache(HC)
    for _ in 1:3
        NE._saem_mala_round!(rng, kernel, cs, cached, 1:3,
            Float64[], Float64[], prior, HC; cache)
    end
    @test cached.gradient_calls[] == 4 # 2 on first, then 1 per sweep
    cache.valid .= false # a population M-step changes the target
    NE._saem_mala_round!(rng, kernel, cs, cached, 1:3,
        Float64[], Float64[], prior, HC; cache)
    @test cached.gradient_calls[] == 6
    refreshed_prior = re_prior(cached, [log(0.7)])
    NE._saem_mala_refresh_prior!(cache, HC, prior, refreshed_prior)
    _, exact_f, exact_g = NE._complete_gradients(cached, 1:3,
        Float64[], Float64[], refreshed_prior, HC)
    @test cache.f ≈ exact_f
    @test cache.G ≈ exact_g
    hessian_kernel = DualAveragingMALA(metric=:hessian)
    metrics = NE._saem_mala_metrics(hessian_kernel, model, 1:3,
        Float64[], Float64[], prior, zeros(1, 3))
    @test [m[1, 1]^2 for m in metrics] ≈ 1 ./ (model.precision .+ 1)
    pk = theoph_model()
    pk_prior = re_prior(pk, omega_parameters(nlme_layout(pk), Ω_THEOPH))
    pk_metrics = NE._saem_mala_metrics(hessian_kernel, pk, 1:2,
        θ_THEOPH, σ_THEOPH, pk_prior, zeros(3, 2))
    @test length(pk_metrics) == 2
    @test all(m -> all(isfinite, m), pk_metrics)
    domain = DomainBatchedMALA()
    repaired = reshape([2.0, 0.2], 1, 2)
    f_repaired, g_repaired, rejected = NE._saem_mala_proposal_gradients(
        domain, 1:2, Float64[], Float64[], re_prior(domain, [0.0]),
        repaired, zeros(1, 2))
    @test rejected == [true, false]
    @test repaired == reshape([0.0, 0.2], 1, 2)
    @test all(isfinite, f_repaired) && all(isfinite, g_repaired)

    # For a fixed Gaussian target and fixed proposal scale, the MH correction
    # recovers its known mean and variance despite the asymmetric drift.
    one = BatchedGaussianMALA([0.8], [4.0], Ref(0))
    state = NE._saem_mala_states(DualAveragingMALA(initial_stepsize=0.6), 1)
    h = zeros(1, 1)
    draws = Float64[]
    for t in 1:6000
        NE._saem_mala_round!(rng, kernel, state, one, 1:1,
            Float64[], Float64[], prior, h; adapt=false)
        t > 1000 && push!(draws, h[1, 1])
    end
    @test abs(mean(draws) - 4 * 0.8 / 5) < 0.07
    @test abs(var(draws) - 1 / 5) < 0.06
    @test state[1].count == 0

    fit = saem(model, Float64[], Float64[], [0.0]; rng=MersenneTwister(129),
        options=SAEMOptions(iterations=20, burnin=6, mcmc_burnin=0,
            mh_rounds=1, mstep_interval=5), kernel)
    @test fit.kernel == :dual_averaging_mala
    @test fit.mala_adaptation_steps == 20
    @test all(isfinite, fit.mala_stepsizes)
    @test all(x -> 0 <= x <= 1, fit.mala_acceptance)
    @test all(isfinite, fit.eta)
    @test ismissing(fit.rw_scale)
    joint = ADBatchedMALA()
    joint_fit = saem(joint, Float64[], Float64[], [log(0.5)];
        rng=MersenneTwister(22), options=SAEMOptions(iterations=5, burnin=2,
            mcmc_burnin=0, mstep_interval=2, numeric_maxiter=2),
        kernel=DualAveragingMALA(initial_stepsize=0.05))
    @test all(isfinite, joint_fit.eta)
    @test joint_fit.mala_adaptation_steps == 5
    @test_throws ArgumentError NE._saem_check_mala(
        DualAveragingMALA(target_acceptance_rate=1.0))
    @test_throws ArgumentError NE._saem_check_mala(DualAveragingMALA(metric=:bad))
end
