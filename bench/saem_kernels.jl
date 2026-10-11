# Fixed-parameter comparison of per-subject SAEM kernels on the public Theoph
# model. Run from the repository root with `julia --project=test bench/saem_kernels.jl`.
# Compilation and model construction are excluded from the timed region.
using NLMEEstimation
using Random
using LinearAlgebra
using Statistics
import NLMEEstimation as NE
include(joinpath(@__DIR__, "..", "test", "common.jl"))
using .TestCommon

BLAS.set_num_threads(1)

function ess_initial_positive(x)
    n = length(x)
    centered = x .- mean(x)
    variance = sum(abs2, centered) / n
    variance == 0 && return 0.0
    tau = 1.0
    for lag in 1:2:(n - 2)
        pair = (dot(view(centered, 1:(n-lag)), view(centered, (lag+1):n)) +
            dot(view(centered, 1:(n-lag-1)), view(centered, (lag+2):n))) / (n * variance)
        pair <= 0 && break
        tau += 2pair
    end
    return n / tau
end

function summarize_draws(draws)
    m, n, _ = size(draws)
    return median([ess_initial_positive(vec(view(draws, j, i, :)))
        for j in 1:m, i in 1:n])
end

function standard_mh_draws(rng, model, θ, σ, ω;
        ids=collect(1:nsubjects(model)), burnin=200, n_draws=800)
    prior = re_prior(model, ω)
    m, n = neta(nlme_layout(model)), length(ids)
    H = zeros(m, n)
    ℓ = conditional_loglikelihoods(model, ids, θ, σ, H)
    draws = Array{Float64}(undef, m, n, n_draws)
    logscale = log(0.5)
    accepted_prior = accepted_rw = 0
    for iteration in 1:(burnin + n_draws)
        ap, ar = NE._saem_mh_round!(rng, model, ids, θ, σ, prior, H, ℓ, exp(logscale))
        accepted_prior += ap
        accepted_rw += ar
        if iteration <= burnin
            logscale = clamp(logscale + (ar / n - 0.3) / sqrt(iteration),
                log(1e-3), log(1e2))
        else
            draws[:, :, iteration - burnin] = H
        end
    end
    return (draws=draws, acceptance=(prior=accepted_prior / (n * (burnin + n_draws)),
        random_walk=accepted_rw / (n * (burnin + n_draws))), scale=exp(logscale))
end

function nuts_draws(rng, model, θ, σ, ω;
        ids=collect(1:nsubjects(model)), n_draws=800, init_metric=:hessian)
    init_metric in (:hessian, :prior) || throw(ArgumentError("invalid init_metric"))
    prior = re_prior(model, ω)
    m, n = neta(nlme_layout(model)), length(ids)
    draws = Array{Float64}(undef, m, n, n_draws)
    divergences = 0
    stream_seconds = 0.0
    mktempdir() do root
        for (k, i) in enumerate(ids)
            target = NE._saem_target(model, i, θ, σ, prior)
            warm_dir = joinpath(root, "warm_$(k)")
            Σ = init_metric == :hessian ?
                NE._saem_hessian_metric(model, i, θ, σ, prior, zeros(m);
                    diagonal=true) :
                [sum(abs2(prior.cholesky_factor[j, k]) for k in 1:j)
                    for j in 1:m]
            NE.WarmupHMC.adaptive_warmup_mcmc(Xoshiro(rand(rng, UInt64)), target;
                init=(position=zeros(m), squared_scale=Σ), checkpoint_dir=warm_dir,
                n_draws=20, n_evaluations=100, recording_target=20,
                stepsize_adaptation_limit=20, max_tree_depth=8,
                nonlinear_adapt=false, monitor_ess=false)
            checkpoint = joinpath(warm_dir, "cp_latest.jls")
            stream = joinpath(root, "subject_$(k).stream")
            result = NE.WarmupHMC.stream_mcmc(checkpoint, target;
                path=stream, n_draws=n_draws)
            draws[:, k, :] = result.draws
            divergences += result.n_divergent
            # A separate measure of the repeated one-transition file reopen
            # cost used by changing-target SAEM; same fixed target here.
            stream_seconds += @elapsed for step in 1:20
                NE.WarmupHMC.stream_mcmc(checkpoint, target;
                    path=stream, n_draws=n_draws + step)
            end
        end
    end
    return (draws=draws, divergences=divergences,
        stream_seconds=stream_seconds,
        stream_ms_per_transition=1000 * stream_seconds / (20n))
end

if abspath(PROGRAM_FILE) == @__FILE__
    model = theoph_model()
    ω = omega_parameters(nlme_layout(model), Ω_THEOPH)
    ids = collect(1:6)
    standard_mh_draws(MersenneTwister(1), model, θ_THEOPH, σ_THEOPH, ω;
        ids, burnin=5, n_draws=5) # compile
    mh_trials = [@timed standard_mh_draws(MersenneTwister(seed), model,
        θ_THEOPH, σ_THEOPH, ω; ids) for seed in 2026:2028]
    mh_time = median([t.time for t in mh_trials])
    mh_ess = median([summarize_draws(t.value.draws) for t in mh_trials])
    mh_prior = median([t.value.acceptance.prior for t in mh_trials])
    mh_rw = median([t.value.acceptance.random_walk for t in mh_trials])
    println("subjects=$(length(ids)); retained_draws=800; replicates=3; Julia=$(VERSION)")
    println("kernel\tseconds\tmedian_ess\tmedian_ess_per_second\tprior_acceptance\trw_acceptance\tdivergences\tstream_ms_per_transition")
    println(join(("standard_MH", mh_time, mh_ess, mh_ess / mh_time,
        mh_prior, mh_rw, 0, NaN), '\t'))
    nuts_draws(MersenneTwister(1), model, θ_THEOPH, σ_THEOPH, ω;
        ids=ids[1:1], n_draws=5) # compile
    for (label, metric) in (("NUTS_Hessian", :hessian), ("NUTS_prior", :prior))
        nuts_trials = [@timed nuts_draws(MersenneTwister(seed), model,
            θ_THEOPH, σ_THEOPH, ω; ids, init_metric=metric) for seed in 2026:2028]
        nuts_time = median([t.time - t.value.stream_seconds for t in nuts_trials])
        nuts_ess = median([summarize_draws(t.value.draws) for t in nuts_trials])
        divergences = sum(t.value.divergences for t in nuts_trials)
        stream_ms = median([t.value.stream_ms_per_transition for t in nuts_trials])
        println(join((label, nuts_time, nuts_ess, nuts_ess / nuts_time,
            NaN, NaN, divergences, stream_ms), '\t'))
    end
end
