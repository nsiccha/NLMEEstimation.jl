# Stability selection after saem_kernel_tuning.jl, then a fresh-seed comparison.
# Run from repository root: julia --project=test bench/saem_kernel_final.jl
include(joinpath(@__DIR__, "saem_kernel_tuning.jl"))

const SELECTION_SEEDS = 2060:2069
const QUALIFY_SEEDS = 2080:2084
const FINAL_SEEDS = 2090:2099

function nuts_kernel(cfg)
    return WarmupHMCNUTS(metric=cfg.metric == :hessian ? :diagonal : cfg.metric,
        warmup_draws=cfg.draws,
        warmup_evaluations=cfg.evaluations,
        stepsize_adaptation_limit=cfg.draws,
        max_tree_depth=cfg.depth,
        target_acceptance_rate=cfg.acceptance)
end

function nuts_fit_check(model, ω, cfg, seeds)
    options = SAEMOptions(iterations=40, burnin=15, mcmc_burnin=5,
        mh_rounds=1, adapt_iterations=0,
        numeric_maxiter=2, mstep_interval=5)
    kernel = nuts_kernel(cfg)
    saem(model, θ_THEOPH, σ_THEOPH, ω;
        rng=MersenneTwister(1), subjects=collect(1:6),
        options=SAEMOptions(iterations=2, burnin=1, mcmc_burnin=1,
            mh_rounds=1, adapt_iterations=0,
            numeric_maxiter=1, mstep_interval=2), kernel)
    trials = [@timed saem(model, θ_THEOPH, σ_THEOPH, ω;
        rng=MersenneTwister(seed), subjects=collect(1:6),
        options, kernel) for seed in seeds]
    return (seconds=median([t.time for t in trials]),
        divergences=sum(t.value.nuts_divergences for t in trials),
        max_divergences=maximum(t.value.nuts_divergences for t in trials),
        finite=all(t -> all(isfinite,
            (t.value.theta..., t.value.sigma..., t.value.omega...)), trials))
end

function fit_cost(model, ω, mh_cfg, nuts_cfg, seeds)
    mh_options = SAEMOptions(iterations=40, burnin=15, mcmc_burnin=5,
        mh_rounds=1, rw_scale=mh_cfg.scale,
        adapt_iterations=mh_cfg.adapt ? 20 : 0,
        numeric_maxiter=2, mstep_interval=5)
    kernel = StandardSAEMMH()
    # Compile the end-to-end path outside the timed region.
    saem(model, θ_THEOPH, σ_THEOPH, ω;
        rng=MersenneTwister(1), subjects=collect(1:6),
        options=SAEMOptions(iterations=2, burnin=1, mcmc_burnin=1,
            mh_rounds=1, rw_scale=mh_options.rw_scale,
            adapt_iterations=mh_options.adapt_iterations > 0 ? 1 : 0,
            numeric_maxiter=1, mstep_interval=2), kernel)
    trials = [@timed saem(model, θ_THEOPH, σ_THEOPH, ω;
        rng=MersenneTwister(seed), subjects=collect(1:6),
        options=mh_options, kernel) for seed in seeds]
    finite = all(t -> all(isfinite,
        (t.value.theta..., t.value.sigma..., t.value.omega...)), trials)
    println(join(("fit", "MH", "40 SAEM iterations + 5 MCMC burn-in",
        round(median([t.time for t in trials]); digits=6),
        maximum(t.value.nuts_divergences for t in trials), finite), '\t'))
    nuts = nuts_fit_check(model, ω, nuts_cfg, seeds)
    println(join(("fit", "NUTS", "40 SAEM iterations + 5 MCMC burn-in",
        round(nuts.seconds; digits=6), nuts.max_divergences,
        nuts.finite), '\t'))
end

if abspath(PROGRAM_FILE) == @__FILE__
    model = theoph_model()
    ω = omega_parameters(nlme_layout(model), Ω_THEOPH)
    mh_configs = [
        (scale=0.35, adapt=false, target=0.3),
        (scale=0.45, adapt=false, target=0.3),
        (scale=0.5, adapt=true, target=0.2),
        (scale=0.5, adapt=true, target=0.3)]
    nuts_configs = [
        (metric=:dense, draws=8, evaluations=50, depth=8, acceptance=0.8),
        (metric=:dense, draws=8, evaluations=100, depth=8, acceptance=0.8),
        (metric=:dense, draws=20, evaluations=50, depth=8, acceptance=0.8),
        (metric=:dense, draws=20, evaluations=100, depth=8, acceptance=0.8),
        (metric=:dense, draws=8, evaluations=50, depth=8, acceptance=0.9),
        (metric=:hessian, draws=20, evaluations=100, depth=8, acceptance=0.8)]
    measure_mh(model, ω, first(mh_configs), 1:1)
    measure_nuts(model, ω, first(nuts_configs), 1:1)
    println("Julia=$(VERSION); retained_draws=$(RETAINED_DRAWS); selection_seeds=$(collect(SELECTION_SEEDS)); qualification_seeds=$(collect(QUALIFY_SEEDS)); final_seeds=$(collect(FINAL_SEEDS))")
    println("phase\tkernel\tconfig\tmedian_ESS_per_second\tmedian_seconds\tmedian_ESS\trw_acceptance\tfinal_scale\tdivergences")
    mh_results = [(cfg=cfg, result=measure_mh(model, ω, cfg, SELECTION_SEEDS))
        for cfg in mh_configs]
    nuts_results = [(cfg=cfg, result=measure_nuts(model, ω, cfg, SELECTION_SEEDS),
        fit=nuts_fit_check(model, ω, cfg, QUALIFY_SEEDS))
        for cfg in nuts_configs]
    for item in mh_results
        print_row("selection", "MH", item.cfg, item.result)
    end
    for item in nuts_results
        print_row("selection", "NUTS", item.cfg, item.result)
        println(join(("qualification", "NUTS", item.cfg,
            round(item.fit.seconds; digits=6), item.fit.divergences,
            item.fit.finite), '\t'))
    end
    mh_best = argmax(item -> item.result.score, mh_results)
    clean_nuts = filter(item -> item.result.divergences == 0 &&
        item.fit.divergences == 0 && item.fit.finite, nuts_results)
    isempty(clean_nuts) && error("no divergence-free NUTS finalist")
    nuts_best = argmax(item -> item.result.score, clean_nuts)
    println("selected_MH=$(mh_best.cfg); selected_NUTS=$(nuts_best.cfg)")
    print_row("final", "MH", mh_best.cfg,
        measure_mh(model, ω, mh_best.cfg, FINAL_SEEDS))
    print_row("final", "NUTS", nuts_best.cfg,
        measure_nuts(model, ω, nuts_best.cfg, FINAL_SEEDS))
    fit_cost(model, ω, mh_best.cfg, nuts_best.cfg, FINAL_SEEDS)
end
