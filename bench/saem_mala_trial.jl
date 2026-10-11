# Reproducible trial of continuously adapted, per-subject batched MALA.
# From repository root: julia --project=test bench/saem_mala_trial.jl
# Fixed-target means include every draw (no MCMC burn-in). Selection and
# held-out seeds are disjoint. Changing-target SAEM fits also set
# mcmc_burnin=0; its `burnin` is only the stochastic-approximation schedule.
using NLMEEstimation, Random, LinearAlgebra, Statistics
import NLMEEstimation as NE
include(joinpath(@__DIR__, "..", "test", "common.jl"))
using .TestCommon

struct CorrelatedGaussianBatch
    y::Matrix{Float64}
    Q::Vector{Matrix{Float64}}
end
NE.nsubjects(m::CorrelatedGaussianBatch) = size(m.y, 2)
NE.nlme_layout(m::CorrelatedGaussianBatch) = NLMELayout(0, 0, [size(m.y, 1)])
NE.evaluation_style(::CorrelatedGaussianBatch) = BatchedEvaluation()
NE.provides_eta_hessian(::CorrelatedGaussianBatch) = true
function NE.conditional_loglikelihoods_and_gradients(m::CorrelatedGaussianBatch,
        ids, θ, σ, H)
    values = Vector{Float64}(undef, length(ids))
    G = similar(H)
    for (j, i) in enumerate(ids)
        d = view(H, :, j) - view(m.y, :, i)
        G[:, j] = -(m.Q[i] * d)
        values[j] = -0.5 * dot(d, m.Q[i] * d)
    end
    return values, G
end
NE.conditional_loglikelihoods(m::CorrelatedGaussianBatch, ids, θ, σ, H) =
    first(NE.conditional_loglikelihoods_and_gradients(m, ids, θ, σ, H))
function NE.conditional_loglikelihood_gradient_and_hessian(m::CorrelatedGaussianBatch,
        i, θ, σ, η)
    d = η - view(m.y, :, i)
    return -0.5 * dot(d, m.Q[i] * d), -(m.Q[i] * d), -m.Q[i]
end

function gaussian_fixture()
    y = [0.4 -0.8 0.7 1.1 -0.5 0.2;
         -0.2 0.6 -0.9 0.4 1.0 -0.7]
    Q = [begin
        a = 0.17 * i
        R = [cos(a) -sin(a); sin(a) cos(a)]
        R * Diagonal([0.4 + 0.6i, 1.0 + 5i]) * R'
    end for i in 1:6]
    Ω = [1.0 0.65; 0.65 1.0]
    model = CorrelatedGaussianBatch(y, Q)
    prior = re_prior(model, omega_parameters(nlme_layout(model), Ω))
    truth = hcat([((Q[i] + inv(Ω)) \ (Q[i] * y[:, i])) for i in 1:6]...)
    return model, prior, truth
end

function fixed_trial(method, cfg, seed; steps=400, fixture=gaussian_fixture)
    model, prior, truth = fixture()
    ids = collect(1:6)
    rng = MersenneTwister(seed)
    H = zeros(size(truth))
    sumH = zeros(size(truth))
    if method == :MALA
        kernel = DualAveragingMALA(metric=cfg.metric, initial_stepsize=cfg.eps,
            target_acceptance_rate=cfg.target)
        states = NE._saem_mala_states(kernel, 6)
        cache = NE._saem_mala_cache(H)
        elapsed = @elapsed begin
            metrics = NE._saem_mala_metrics(kernel, model, ids,
                Float64[], Float64[], prior, H)
            for _ in 1:steps
                NE._saem_mala_round!(rng, kernel, states, model, ids,
                    Float64[], Float64[], prior, H; metrics, cache)
                sumH .+= H
            end
        end
        acceptance = mean(s.accepted / s.count for s in states)
        independence_acceptance = NaN
        final_steps = [exp(s.log_stepsize) for s in states]
    else
        accepted = 0
        independence_accepted = 0
        elapsed = @elapsed begin
            ℓ = NE.conditional_loglikelihoods(model, ids, Float64[], Float64[], H)
            for _ in 1:steps
                ai, ar = NE._saem_mh_round!(rng, model, ids, Float64[],
                    Float64[], prior, H, ℓ, cfg.scale)
                independence_accepted += ai
                accepted += ar
                sumH .+= H
            end
        end
        acceptance = accepted / (steps * 6)
        independence_acceptance = independence_accepted / (steps * 6)
        final_steps = fill(cfg.scale, 6)
    end
    return (mse=mean(abs2, sumH / steps - truth), seconds=elapsed,
        acceptance=acceptance, independence_acceptance=independence_acceptance,
        steps=final_steps)
end

function summarize_fixed(method, cfg, seeds; steps=400, fixture=gaussian_fixture)
    trials = [fixed_trial(method, cfg, seed; steps, fixture) for seed in seeds]
    return (rmse=sqrt(mean(t.mse for t in trials)),
        seconds=median(t.seconds for t in trials),
        acceptance=median(t.acceptance for t in trials),
        independence_acceptance=median(t.independence_acceptance for t in trials),
        min_step=median(minimum(t.steps) for t in trials),
        max_step=median(maximum(t.steps) for t in trials))
end

function theoph_fit(method, cfg, seed, model, ω;
        iterations=60, burnin=20, ids=1:6, numeric_maxiter=2)
    options = SAEMOptions(; iterations, burnin, mcmc_burnin=0,
        mh_rounds=1, rw_scale=method == :MH ? cfg.scale : 0.5,
        adapt_iterations=0, numeric_maxiter, mstep_interval=5,
        numeric_mstep=:mixture)
    kernel = method == :MALA ? DualAveragingMALA(metric=cfg.metric,
        initial_stepsize=cfg.eps,
        target_acceptance_rate=cfg.target) : method == :MH ?
        StandardSAEMMH() : WarmupHMCNUTS(metric=:dense, warmup_draws=20,
        warmup_evaluations=50, stepsize_adaptation_limit=20,
        max_tree_depth=8, target_acceptance_rate=0.8)
    timed = @timed saem(model, θ_THEOPH, σ_THEOPH, ω;
        rng=MersenneTwister(seed), subjects=collect(ids), options, kernel)
    fit = timed.value
    return (seconds=timed.time, theta=fit.theta, sigma=fit.sigma,
        omega=diag(fit.omega_matrix), acceptance=method == :MALA ?
        median(fit.mala_acceptance) : method == :MH ?
        fit.acceptance.random_walk : NaN,
        min_step=method == :MALA ? minimum(fit.mala_stepsizes) : NaN,
        max_step=method == :MALA ? maximum(fit.mala_stepsizes) : NaN,
        divergences=fit.nuts_divergences)
end

if abspath(PROGRAM_FILE) == @__FILE__
    BLAS.set_num_threads(1)
    println("Julia=$(VERSION) threads=$(Threads.nthreads()) BLAS_threads=$(BLAS.get_num_threads())")
    println("repo_sha=$(readchomp(`git rev-parse HEAD`)) host=$(readchomp(`hostname`))")
    println("fixed_target=6 correlated 2D Gaussian subjects; 400 draws, all retained")
    println("selection_seeds=301:308 heldout_seeds=401:416")
    mh_cfgs = [(scale=s,) for s in (0.25, 0.5, 0.8, 1.2)]
    mala_cfgs = [(metric=m, eps=e, target=a) for m in (:prior, :hessian),
        e in (0.05, 0.1, 0.2), a in (0.45, 0.574, 0.7)]
    fixed_trial(:MH, first(mh_cfgs), 1; steps=2)
    fixed_trial(:MALA, first(mala_cfgs), 1; steps=2)
    println("phase\tmethod\tconfig\trmse\tmedian_seconds\tacceptance\tmin_step\tmax_step")
    mh_tune = [(cfg=c, summary=summarize_fixed(:MH, c, 301:308)) for c in mh_cfgs]
    mala_tune = [(cfg=c, summary=summarize_fixed(:MALA, c, 301:308)) for c in mala_cfgs]
    for (method, trials) in ((:MH, mh_tune), (:MALA, mala_tune)), x in trials
        s = x.summary
        println(join(("selection", method, x.cfg, s.rmse, s.seconds,
            s.acceptance, s.min_step, s.max_step), '\t'))
    end
    # For equal Monte Carlo error, cost scales approximately with RMSE² × time.
    mh_best = argmin(x -> x.summary.rmse^2 * x.summary.seconds, mh_tune)
    mala_best = argmin(x -> x.summary.rmse^2 * x.summary.seconds, mala_tune)
    println("selected_MH=$(mh_best.cfg) selected_MALA=$(mala_best.cfg)")
    for (method, cfg) in ((:MH, mh_best.cfg), (:MALA, mala_best.cfg))
        s = summarize_fixed(method, cfg, 401:416)
        println(join(("heldout", method, cfg, s.rmse, s.seconds,
            s.acceptance, s.min_step, s.max_step), '\t'))
    end
    println("long_fixed_target=4000 draws per subject, all retained, seeds=701:716")
    for (method, cfg) in ((:MH, mh_best.cfg), (:MALA, mala_best.cfg))
        s = summarize_fixed(method, cfg, 701:716; steps=4000)
        println(join(("longheldout", method, cfg, s.rmse, s.seconds,
            s.acceptance, s.min_step, s.max_step), '\t'))
    end
    model = theoph_model()
    ω = omega_parameters(nlme_layout(model), Ω_THEOPH)
    println("changing_target=Theoph 6 subjects, 60 SAEM iterations, no MCMC burn-in")
    println("fit_seeds=501:506")
    println("phase\tmethod\tconfig\tmedian_seconds\tacceptance\tmin_step\tmax_step\tdivergences\tmedian_theta\tmedian_sigma\tmedian_omega_diagonal")
    fit_configs = [(:MH, mh_best.cfg),
        (:MALA, mala_best.cfg),
        (:MALA, (metric=:prior, eps=0.05, target=0.574)),
        (:MALA, (metric=:prior, eps=0.1, target=0.574)),
        (:MALA, (metric=:prior, eps=0.2, target=0.574)),
        (:MALA, (metric=:prior, eps=0.1, target=0.7)),
        (:NUTS, nothing)]
    for (method, cfg) in fit_configs
        theoph_fit(method, cfg, 1, model, ω) # compile outside timings
        fits = [theoph_fit(method, cfg, seed, model, ω) for seed in 501:506]
        medvec(field) = [median(getproperty(f, field)[j] for f in fits)
            for j in eachindex(getproperty(first(fits), field))]
        println(join(("fit", method, cfg, median(f.seconds for f in fits),
            median(f.acceptance for f in fits),
            median(f.min_step for f in fits), median(f.max_step for f in fits),
            sum(f.divergences for f in fits), medvec(:theta), medvec(:sigma),
            medvec(:omega)), '\t'))
    end
    println("longfit=Theoph all 12 subjects, 120 SAEM iterations, no MCMC burn-in")
    println("longfit_seeds=601:603")
    for (method, cfg) in ((:MH, mh_best.cfg),
            (:MALA, (metric=:prior, eps=0.05, target=0.574)),
            (:MALA, (metric=:prior, eps=0.1, target=0.7)),
            (:MALA, mala_best.cfg))
        fits = [theoph_fit(method, cfg, seed, model, ω;
            iterations=120, burnin=40, ids=1:12, numeric_maxiter=3)
            for seed in 601:603]
        medvec(field) = [median(getproperty(f, field)[j] for f in fits)
            for j in eachindex(getproperty(first(fits), field))]
        println(join(("longfit", method, cfg, median(f.seconds for f in fits),
            median(f.acceptance for f in fits),
            median(f.min_step for f in fits), median(f.max_step for f in fits),
            sum(f.divergences for f in fits), medvec(:theta), medvec(:sigma),
            medvec(:omega)), '\t'))
    end
end
