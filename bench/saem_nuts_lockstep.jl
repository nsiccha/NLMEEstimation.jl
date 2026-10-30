# SAEM's NUTS kernel on a batched model whose every call evaluates the whole model,
# as BRM evaluates a nested model without subject cells (each subject's gradient is
# one whole-model value and gradient, which returns every subject's at once): what
# the kernel's chains cost per SA iteration when each leapfrog step reaches the model
# per subject (NLMEEstimation 0.2.0) and in lockstep (0.2.1), and whether the fits
# agree.
#
# Run from the repository root with the test environment:
#   julia --project=test bench/saem_nuts_lockstep.jl [output]
#
# The output (default `bench/saem_nuts_lockstep_results.txt`) receives everything
# printed here.
#
# The model is the test suite's `RaggedOccasions` (one subject effect and one effect
# per nested unit) with the twelve subjects of `bench/saem_nuts_ragged_scale.jl`,
# 3,294 to 35,783 nested units, 157,808 in all. `Sweep{BatchedEvaluation}` declares
# the batched protocol and evaluates every subject in each call (the listed ones at
# the given η, the others at zero), so a call costs the whole model whatever its
# batch. `Sweep{PerSubjectEvaluation}` is the same model as the 0.2.0 kernel reached
# it: its chains' evaluations (Pathfinder and transitions) arrive one subject at a
# time, each a whole sweep, while SAEM's own batched density calls (each
# iteration's log-likelihoods, the M-step's domain screens) stay one sweep each.
# Both answer every subject with `RaggedOccasions`' own numbers, so their fits must
# be identical. The numerical M-step takes per-subject population gradients from
# `RaggedOccasions` directly in both (no sweep), so the two runs differ only in how
# the kernel's chains reach the model.
using NLMEEstimation
import NLMEEstimation as NE
using Printf
using Random
include(joinpath(@__DIR__, "..", "test", "common.jl"))
using .TestCommon: RaggedOccasions, occasions_model

const OUTPUT = open(length(ARGS) >= 1 ? ARGS[1] :
    joinpath(@__DIR__, "saem_nuts_lockstep_results.txt"), "w")
report(args...) = (println(args...); println(OUTPUT, args...); flush(OUTPUT))

const UNITS = [3294, 5000, 7000, 8500, 10000, 11000, 12000, 13500, 15000, 17000, 19731, 35783]
const θ0, σ0, ω0 = [2.0], [log(0.25)], [log(0.6), log(0.4)]

struct Sweep{S}
    inner::RaggedOccasions
    gradient_sweeps::Base.RefValue{Int}
    value_sweeps::Base.RefValue{Int}
end
Sweep{S}(inner) where {S} = Sweep{S}(inner, Ref(0), Ref(0))
NE.evaluation_style(::Sweep{S}) where {S} = S()
NE.saem_population_gradient_style(::Sweep) = PerSubjectEvaluation()
NE.nsubjects(m::Sweep) = nsubjects(m.inner)
NE.nlme_layout(m::Sweep) = nlme_layout(m.inner)
NE.mu_referencing(m::Sweep) = NE.mu_referencing(m.inner)
NE.conditional_loglikelihood_and_population_gradient(m::Sweep, i::Integer, θ, σ, η) =
    NE.conditional_loglikelihood_and_population_gradient(m.inner, i, θ, σ, η)

# One whole-model evaluation: every subject, the listed ones at `H`'s columns.
function sweep(f, counter, m, subjects, θ, σ, H)
    counter[] += 1
    layout = nlme_layout(m.inner)
    full = [zeros(NE.neta(layout, i)) for i in 1:nsubjects(m.inner)]
    for (k, i) in enumerate(subjects)
        full[i] = H[k]
    end
    all = [f(m.inner, i, θ, σ, full[i]) for i in eachindex(full)]
    return all[subjects]
end
NE.conditional_loglikelihoods(m::Sweep{BatchedEvaluation}, subjects, θ, σ, H) =
    sweep(conditional_loglikelihood, m.value_sweeps, m, subjects, θ, σ, H)
function NE.conditional_loglikelihoods_and_gradients(m::Sweep{BatchedEvaluation}, subjects,
        θ, σ, H)
    pairs = sweep(conditional_loglikelihood_and_gradient, m.gradient_sweeps, m, subjects,
        θ, σ, H)
    return first.(pairs), last.(pairs)
end
NE.conditional_loglikelihoods(m::Sweep{PerSubjectEvaluation}, subjects, θ, σ, H) =
    sweep(conditional_loglikelihood, m.value_sweeps, m, subjects, θ, σ, H)
NE.conditional_loglikelihood(m::Sweep{PerSubjectEvaluation}, i::Integer, θ, σ, η) =
    only(sweep(conditional_loglikelihood, m.value_sweeps, m, [i], θ, σ, [η]))
NE.conditional_loglikelihood_and_gradient(m::Sweep{PerSubjectEvaluation}, i::Integer, θ,
        σ, η) =
    only(sweep(conditional_loglikelihood_and_gradient, m.gradient_sweeps, m, [i], θ, σ, [η]))

# The fit, timed in phases: the Pathfinder start alone (`iterations = 0` is invalid,
# so a separate start is timed through the kernel's internals), then the whole fit.
function measured(model, options, seed)
    ids = collect(1:nsubjects(model))
    layout = nlme_layout(model)
    prior = re_prior(model, ω0)
    H = [zeros(NE.neta(layout, i)) for i in ids]
    g0, v0 = model.gradient_sweeps[], model.value_sweeps[]
    start = @elapsed states = NE._saem_nuts_start!(Xoshiro(seed), WarmupHMCNUTS(), model,
        ids, θ0, σ0, prior, H)
    start_sweeps = (model.gradient_sweeps[] - g0, model.value_sweeps[] - v0)
    g0, v0 = model.gradient_sweeps[], model.value_sweeps[]
    round = @elapsed NE._saem_nuts_round!(states, model, ids, θ0, σ0, prior, H)
    round_sweeps = (model.gradient_sweeps[] - g0, model.value_sweeps[] - v0)
    evaluations = [s.total_evaluation_counter for s in states]
    g0, v0 = model.gradient_sweeps[], model.value_sweeps[]
    stats = @timed saem(model, θ0, σ0, ω0; rng = Xoshiro(seed), options,
        kernel = WarmupHMCNUTS())
    fit_sweeps = (model.gradient_sweeps[] - g0, model.value_sweeps[] - v0)
    return (; start, start_sweeps, round, round_sweeps, evaluations, fit = stats.value,
        fit_time = stats.time, fit_sweeps)
end

function main()
    report("SAEM's NUTS kernel on a whole-model-sweep batched model, per subject against ",
        "lockstep: Julia $VERSION, NLMEEstimation ",
        readchomp(`git -C $(joinpath(@__DIR__, "..")) rev-parse --short HEAD`),
        ", WarmupHMC ", pkgversion(NE.WarmupHMC), ", threads ", Threads.nthreads())
    # Compile every path on a small model first, so the timings are warm.
    small = occasions_model(seed = 0, units = [5, 8])
    warm = SAEMOptions(iterations = 3, burnin = 1, mcmc_burnin = 1)
    for S in (PerSubjectEvaluation, BatchedEvaluation)
        measured(Sweep{S}(small), warm, 1)
    end
    inner = occasions_model(seed = 3, units = UNITS)
    options = SAEMOptions(iterations = 10, burnin = 5, mcmc_burnin = 2)
    report("Twelve subjects of $(minimum(UNITS))–$(maximum(UNITS)) nested units ",
        "($(sum(UNITS)) effects); one sweep evaluates all twelve subjects. Fit: ",
        "$(options.mcmc_burnin) burn-in sweeps and $(options.iterations) SA iterations, ",
        "seed 1.")
    one = @elapsed sweep(conditional_loglikelihood_and_gradient, Ref(0), Sweep{BatchedEvaluation}(inner),
        [1], θ0, σ0, [zeros(1 + UNITS[1])])
    report(@sprintf("One whole-model value-and-gradient sweep: %.3f s", one))
    results = map((PerSubjectEvaluation, BatchedEvaluation)) do S
        r = measured(Sweep{S}(inner), options, 1)
        name = S === PerSubjectEvaluation ? "per subject (0.2.0)" : "lockstep (0.2.1)"
        report("\n", name)
        report(@sprintf("  Pathfinder start: %.1f s, %d gradient and %d value sweeps",
            r.start, r.start_sweeps...))
        report(@sprintf("  one NUTS round (one transition per subject): %.2f s, %d gradient sweeps",
            r.round, r.round_sweeps[1]),
            "; gradient evaluations per subject over start and round: ", r.evaluations)
        per = options.mcmc_burnin + options.iterations
        report(@sprintf("  fit (start included): %.1f s, %d gradient and %d value sweeps; ",
            r.fit_time, r.fit_sweeps...),
            @sprintf("%.1f gradient sweeps per SA iteration after the start",
                (r.fit_sweeps[1] - r.start_sweeps[1]) / per))
        r
    end
    serial, lockstep = results
    report("\nFits identical (θ, σ, ω, every η, divergences): ",
        serial.fit.theta == lockstep.fit.theta && serial.fit.sigma == lockstep.fit.sigma &&
        serial.fit.omega == lockstep.fit.omega && serial.fit.eta == lockstep.fit.eta &&
        serial.fit.nuts_divergences == lockstep.fit.nuts_divergences)
    report(@sprintf("Gradient sweeps, per subject / lockstep: start %.1f×, NUTS round %.1f×, fit %.1f×",
        serial.start_sweeps[1] / lockstep.start_sweeps[1],
        serial.round_sweeps[1] / lockstep.round_sweeps[1],
        serial.fit_sweeps[1] / lockstep.fit_sweeps[1]))
    report(@sprintf("Wall time, per subject / lockstep: start %.1f×, NUTS round %.1f×, fit %.1f×",
        serial.start / lockstep.start, serial.round / lockstep.round,
        serial.fit_time / lockstep.fit_time))
end

main()
close(OUTPUT)
