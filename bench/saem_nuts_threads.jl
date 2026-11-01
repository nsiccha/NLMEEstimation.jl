# What `WarmupHMCNUTS(threads = true)` gains on the twelve subjects of
# `bench/saem_nuts_lockstep.jl` (3,294 to 35,783 nested units): wall time against the
# serial route and against lockstep over a batched model whose every call evaluates the
# whole model; the part a stop-the-world collector takes back; and the ceiling set by
# the largest subject.
#
# Run from the repository root with the test environment, with as many threads as CPUs
# the run may use (more threads than CPUs ran about 2× slower than serial):
#   julia -t 2 --project=test bench/saem_nuts_threads.jl [output]
#
# The output (default `bench/saem_nuts_threads_results.txt`) receives everything printed
# here. Timings are the best of three passes, interleaved in one process after a warm-up.
using NLMEEstimation
import NLMEEstimation as NE
using Printf
using Random
using Statistics
include(joinpath(@__DIR__, "..", "test", "common.jl"))
using .TestCommon: RaggedOccasions, occasions_model

const OUTPUT = open(length(ARGS) >= 1 ? ARGS[1] :
    joinpath(@__DIR__, "saem_nuts_threads_results.txt"), "w")
report(args...) = (println(args...); println(OUTPUT, args...); flush(OUTPUT))

Threads.nthreads() > 1 || error("run this bench with several threads (`julia -t <n>`)")

const UNITS = [3294, 5000, 7000, 8500, 10000, 11000, 12000, 13500, 15000, 17000, 19731, 35783]
const θ0, σ0, ω0 = [2.0], [log(0.25)], [log(0.6), log(0.4)]
const SERIAL, THREADED = WarmupHMCNUTS(), WarmupHMCNUTS(threads = true)

# The model behind the batched protocol, every call one whole-model evaluation.
struct Sweep
    inner::RaggedOccasions
end
NE.evaluation_style(::Sweep) = BatchedEvaluation()
NE.saem_population_gradient_style(::Sweep) = PerSubjectEvaluation()
NE.nsubjects(m::Sweep) = nsubjects(m.inner)
NE.nlme_layout(m::Sweep) = nlme_layout(m.inner)
NE.mu_referencing(m::Sweep) = NE.mu_referencing(m.inner)
NE.conditional_loglikelihood_and_population_gradient(m::Sweep, i::Integer, θ, σ, η) =
    NE.conditional_loglikelihood_and_population_gradient(m.inner, i, θ, σ, η)
function sweep(f, m, subjects, θ, σ, H)
    full = [zeros(NE.neta(nlme_layout(m.inner), i)) for i in 1:nsubjects(m.inner)]
    for (k, i) in enumerate(subjects)
        full[i] = H[k]
    end
    return [f(m.inner, i, θ, σ, full[i]) for i in eachindex(full)][subjects]
end
NE.conditional_loglikelihoods(m::Sweep, subjects, θ, σ, H) =
    sweep(conditional_loglikelihood, m, subjects, θ, σ, H)
function NE.conditional_loglikelihoods_and_gradients(m::Sweep, subjects, θ, σ, H)
    pairs = sweep(conditional_loglikelihood_and_gradient, m, subjects, θ, σ, H)
    return first.(pairs), last.(pairs)
end

function start_states(model, kernel)
    ids = collect(1:nsubjects(model))
    prior = re_prior(model, ω0)
    H = [zeros(NE.neta(nlme_layout(model), i)) for i in ids]
    time = @elapsed states = NE._saem_nuts_start!(Xoshiro(1), kernel, model, ids, θ0, σ0, prior, H)
    return (; ids, prior, H, states, time)
end
rounds!(s, model, kernel, n) = for _ in 1:n
    NE._saem_nuts_round!(s.states, model, s.ids, θ0, σ0, s.prior, s.H; kernel.threads)
end

function timed(model, kernel, options)
    s = start_states(model, kernel)
    five = @elapsed rounds!(s, model, kernel, 5)
    stats = @timed saem(model, θ0, σ0, ω0; rng = Xoshiro(1), options, kernel)
    return (; start = s.time, rounds = five, whole = stats.time, gc = stats.gctime, fit = stats.value)
end

same(a, b) = a.theta == b.theta && a.sigma == b.sigma && a.omega == b.omega && a.eta == b.eta

function wall_time()
    report("Wall time, ", Threads.nthreads(), " threads: the start, 5 rounds, and a fit of ",
        "2 burn-in sweeps and 10 SA iterations, seed 1")
    small = occasions_model(seed = 0, units = [5, 8])
    warm = SAEMOptions(iterations = 3, burnin = 1, mcmc_burnin = 1)
    timed(small, SERIAL, warm); timed(small, THREADED, warm); timed(Sweep(small), SERIAL, warm)
    inner = occasions_model(seed = 3, units = UNITS)
    options = SAEMOptions(iterations = 10, burnin = 5, mcmc_burnin = 2)
    routes = (("serial", inner, SERIAL), ("threads = true", inner, THREADED),
        ("lockstep over whole sweeps", Sweep(inner), SERIAL))
    rows = Dict(name => [] for (name, _, _) in routes)
    for pass in 1:3, (name, model, kernel) in routes
        r = timed(model, kernel, options)
        push!(rows[name], r)
        report(@sprintf("  pass %d, %-27s start %6.2f s  5 rounds %6.2f s  fit %6.2f s  (GC %.2f s)",
            pass, name, r.start, r.rounds, r.whole, r.gc))
    end
    fits = [last(rows[name]).fit for (name, _, _) in routes]
    report("  the three fits are identical: ", same(fits[1], fits[2]) && same(fits[1], fits[3]))
    best(name, f) = minimum(f, rows[name])
    for (label, f) in (("start", r -> r.start), ("5 rounds", r -> r.rounds), ("fit", r -> r.whole))
        s, t, l = best("serial", f), best("threads = true", f), best("lockstep over whole sweeps", f)
        report(@sprintf("  %-9s best: serial %.2f s, threads = true %.2f s (%.2f×), lockstep over sweeps %.2f s",
            label, s, t, s / t, l))
    end
    return inner
end

# Five rounds from one start, with the collector on and paused.
function collector(inner)
    report("\nThe collector: 5 rounds from one start, best of three")
    function run(kernel, paused)
        s = start_states(inner, SERIAL)
        GC.gc()
        paused && GC.enable(false)
        t = try
            @elapsed rounds!(s, inner, kernel, 5)
        finally
            GC.enable(true)
        end
        GC.gc()
        return t
    end
    best = Dict((kernel, paused) => Inf for kernel in (SERIAL, THREADED), paused in (false, true))
    for _ in 1:3, paused in (false, true), kernel in (SERIAL, THREADED)
        best[(kernel, paused)] = min(best[(kernel, paused)], run(kernel, paused))
    end
    for paused in (false, true)
        s, t = best[(SERIAL, paused)], best[(THREADED, paused)]
        report(@sprintf("  GC %-6s serial %.2f s, threads = true %.2f s (%.2f×)",
            paused ? "paused" : "on", s, t, s / t))
    end
end

# Each subject's evaluations over 200 rounds: the ceiling a thread per chain could reach,
# and what lockstep over whole sweeps needs, whatever the wall time on this host.
function ceiling(inner)
    report("\nThe ceiling: evaluations per subject and round, 200 rounds at the starting point")
    s = start_states(inner, SERIAL)
    sizes = [NE.neta(nlme_layout(inner), i) for i in s.ids]
    share = sizes ./ sum(sizes)
    used = zeros(Int, length(s.ids), 200)
    for r in 1:200
        before = [state.total_evaluation_counter for state in s.states]
        rounds!(s, inner, SERIAL, 1)
        used[:, r] = [state.total_evaluation_counter for state in s.states] .- before
    end
    work = mean(sum(used .* share; dims = 1))        # whole-model sweeps of CPU
    slowest = mean(maximum(used .* share; dims = 1)) # the wall time with a thread per chain
    report(@sprintf("  evaluations per subject and round: mean %.1f, range %d–%d",
        mean(used), minimum(used), maximum(used)))
    report(@sprintf("  largest subject: %.1f%% of the units", 100maximum(share)))
    report(@sprintf("  CPU work %.1f sweep-equivalents per round, slowest chain %.1f: at most %.1f× on any number of threads",
        work, slowest, work / slowest))
    report(@sprintf("  lockstep over whole sweeps: %.1f sweeps per round", mean(maximum(used; dims = 1))))
end

inner = wall_time()
collector(inner)
ceiling(inner)
report("\nJulia ", VERSION, ", WarmupHMC ", pkgversion(NE.WarmupHMC), ", NLMEEstimation ", pkgversion(NE))
close(OUTPUT)
