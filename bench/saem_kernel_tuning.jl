# Tune each SAEM subject kernel for fixed-target median ESS per elapsed second.
# From the repository root: julia --project=test bench/saem_kernel_tuning.jl
# The held-out seeds are not used to select settings.
include(joinpath(@__DIR__, "saem_kernels.jl"))

const TUNE_SEEDS = 2026:2030
const CHECK_SEEDS = 2050:2054
const RETAINED_DRAWS = 1600

function measure_mh(model, ω, cfg, seeds)
    trials = [@timed standard_mh_draws(MersenneTwister(seed), model,
        θ_THEOPH, σ_THEOPH, ω; ids=collect(1:6), n_draws=RETAINED_DRAWS,
        burnin=200, rw_scale=cfg.scale,
        adapt_iterations=cfg.adapt ? 200 : 0,
        target_acceptance=cfg.target) for seed in seeds]
    ess = [summarize_draws(t.value.draws) for t in trials]
    return (score=median([ess[i] / trials[i].time for i in eachindex(trials)]),
        seconds=median([t.time for t in trials]), ess=median(ess),
        rw_acceptance=median([t.value.acceptance.random_walk for t in trials]),
        final_scale=median([t.value.scale for t in trials]), divergences=0)
end

function measure_nuts(model, ω, cfg, seeds)
    trials = [@timed nuts_draws(MersenneTwister(seed), model,
        θ_THEOPH, σ_THEOPH, ω; ids=collect(1:6), n_draws=RETAINED_DRAWS,
        init_metric=cfg.metric, warmup_draws=cfg.draws,
        warmup_evaluations=cfg.evaluations,
        stepsize_adaptation_limit=cfg.draws,
        max_tree_depth=cfg.depth,
        target_acceptance_rate=cfg.acceptance, probe_steps=0) for seed in seeds]
    ess = [summarize_draws(t.value.draws) for t in trials]
    return (score=median([ess[i] / trials[i].time for i in eachindex(trials)]),
        seconds=median([t.time for t in trials]), ess=median(ess),
        rw_acceptance=NaN, final_scale=NaN,
        divergences=sum(t.value.divergences for t in trials))
end

function print_row(phase, kernel, cfg, result)
    println(join((phase, kernel, cfg, round(result.score; digits=1),
        round(result.seconds; digits=6), round(result.ess; digits=1),
        round(result.rw_acceptance; digits=3),
        round(result.final_scale; digits=3), result.divergences), '\t'))
end

if abspath(PROGRAM_FILE) == @__FILE__
    model = theoph_model()
    ω = omega_parameters(nlme_layout(model), Ω_THEOPH)
    # Compile both covariance representations and the MH sampler outside timing.
    measure_mh(model, ω, (scale=0.5, adapt=true, target=0.3), 1:1)
    for metric in (:hessian, :dense, :prior)
        measure_nuts(model, ω,
            (metric, draws=8, evaluations=50, depth=4, acceptance=0.8), 1:1)
    end

    mh_configs = vcat(
        [(scale=s, adapt=false, target=0.3) for s in
            (0.25, 0.35, 0.45, 0.6, 0.8, 1.2)],
        [(scale=0.5, adapt=true, target=a) for a in
            (0.2, 0.3, 0.4)])
    nuts_configs = [(metric=metric, draws=draws, evaluations=evaluations,
        depth=depth, acceptance=0.8) for metric in (:hessian, :dense, :prior),
        draws in (8, 20), evaluations in (50, 100),
        depth in (4, 8)]
    println("Julia=$(VERSION); subjects=6; retained_draws=$(RETAINED_DRAWS); tuning_seeds=$(collect(TUNE_SEEDS)); heldout_seeds=$(collect(CHECK_SEEDS))")
    println("phase\tkernel\tconfig\tmedian_ESS_per_second\tmedian_seconds\tmedian_ESS\trw_acceptance\tfinal_scale\tdivergences")
    mh_results = [(cfg=cfg, result=measure_mh(model, ω, cfg, TUNE_SEEDS))
        for cfg in mh_configs]
    nuts_results = [(cfg=cfg, result=measure_nuts(model, ω, cfg, TUNE_SEEDS))
        for cfg in nuts_configs]
    for item in mh_results
        print_row("tune", "MH", item.cfg, item.result)
    end
    for item in nuts_results
        print_row("tune", "NUTS", item.cfg, item.result)
    end
    mh_best = argmax(item -> item.result.score, mh_results)
    clean_nuts = filter(item -> item.result.divergences == 0, nuts_results)
    isempty(clean_nuts) && error("no divergence-free NUTS configuration")
    nuts_top = sort(clean_nuts; by=item -> item.result.score, rev=true)[1:3]
    nuts_finalists = unique([merge(item.cfg, (acceptance=acceptance,))
        for item in nuts_top for acceptance in (0.6, 0.7, 0.8, 0.9)])
    nuts_rate_results = [(cfg=cfg, result=measure_nuts(model, ω, cfg, TUNE_SEEDS))
        for cfg in nuts_finalists]
    for item in nuts_rate_results
        print_row("rate_tune", "NUTS", item.cfg, item.result)
    end
    clean_finalists = filter(item -> item.result.divergences == 0,
        nuts_rate_results)
    isempty(clean_finalists) && error("no divergence-free NUTS finalist")
    nuts_best = argmax(item -> item.result.score, clean_finalists)
    println("provisional_fixed_target_MH=$(mh_best.cfg); provisional_fixed_target_NUTS=$(nuts_best.cfg)")
    println("For changing-target divergence qualification and the final comparison, run bench/saem_kernel_final.jl.")
    print_row("crosscheck", "MH", mh_best.cfg,
        measure_mh(model, ω, mh_best.cfg, CHECK_SEEDS))
    print_row("crosscheck", "NUTS", nuts_best.cfg,
        measure_nuts(model, ω, nuts_best.cfg, CHECK_SEEDS))
    println("baseline configs on held-out seeds:")
    mh_base = (scale=0.5, adapt=true, target=0.3)
    nuts_base = (metric=:hessian, draws=20, evaluations=100,
        depth=8, acceptance=0.8)
    print_row("baseline", "MH", mh_base,
        measure_mh(model, ω, mh_base, CHECK_SEEDS))
    print_row("baseline", "NUTS", nuts_base,
        measure_nuts(model, ω, nuts_base, CHECK_SEEDS))
end
