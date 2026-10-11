# End-to-end SAEM comparison on the public Theoph data. Run from repository root
# with `julia --project=test bench/saem_fit_kernels.jl`. Compilation is excluded.
using NLMEEstimation
using Random
using LinearAlgebra
using Statistics
include(joinpath(@__DIR__, "..", "test", "common.jl"))
using .TestCommon

BLAS.set_num_threads(1)
const IDS = collect(1:6)
const FIT_OPTIONS = SAEMOptions(iterations=40, burnin=15, mcmc_burnin=5,
    mh_rounds=1, adapt_iterations=20, numeric_maxiter=2, mstep_interval=5,
    numeric_mstep=:mixture)
const COMPILE_OPTIONS = SAEMOptions(iterations=3, burnin=1, mcmc_burnin=1,
    mh_rounds=1, adapt_iterations=2, numeric_maxiter=1, mstep_interval=3,
    numeric_mstep=:mixture)
const NUTS = WarmupHMCNUTS(warmup_draws=8, warmup_evaluations=50,
    stepsize_adaptation_limit=10, max_tree_depth=6)

function fit_once(seed, model, ω, options, kernel)
    return saem(model, θ_THEOPH, σ_THEOPH, ω;
        rng=MersenneTwister(seed), subjects=IDS, options, kernel)
end

if abspath(PROGRAM_FILE) == @__FILE__
    model = theoph_model()
    ω = omega_parameters(nlme_layout(model), Ω_THEOPH)
    for kernel in (StandardSAEMMH(), NUTS)
        fit_once(1, model, ω, COMPILE_OPTIONS, kernel)
    end
    println("subjects=$(length(IDS)); SAEM_iterations=$(FIT_OPTIONS.iterations); replicates=3; Julia=$(VERSION)")
    println("kernel\tmedian_seconds\tmax_divergences\tall_finite")
    for (label, kernel) in (("standard_MH", StandardSAEMMH()),
            ("WarmupHMC_NUTS", NUTS))
        trials = [@timed fit_once(seed, model, ω, FIT_OPTIONS, kernel)
            for seed in 2026:2028]
        seconds = median([t.time for t in trials])
        divergent = maximum(t.value.nuts_divergences for t in trials)
        finite = all(t -> all(isfinite, (t.value.theta..., t.value.sigma...,
            t.value.omega...)), trials)
        println(join((label, seconds, divergent, finite), '\t'))
    end
end
