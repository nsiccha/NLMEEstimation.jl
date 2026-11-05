# Wall time of SAEM's NUTS kernel on a BRM model with unit partitions: lockstep
# (`WarmupHMCNUTS()`) against `WarmupHMCNUTS(threads = true)`, the measurement that
# decides whether lockstep can go (todo `0kbdz33`).
#
# The model is BRM's flat PK benchmark (BRM `research/nlme_threads/benchmark.jl`): `units`
# subjects with `rows` observations each. 80 × 8 is the fixture brm-use reports on.
#
# Both routes run in one process, interleaved and alternating which goes first, after a
# warm-up of each. The pair counts are fixed here before measuring, and every sample is
# printed. A ratio is lockstep's wall time over threads', so above 1 means threads are
# faster. Run one process per thread count on at least as many CPUs as threads:
#
#   bash bench/saem_threads_vs_lockstep_brm.sh
#
# or by hand, in an environment holding this checkout, BRM and BRM's Enzyme:
#
#   julia --project=<env> --threads=<T> bench/saem_threads_vs_lockstep_brm.jl \
#       <units> <rows> <output> [<round pairs> <start pairs> <fit pairs>]
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface: AutoEnzyme
using NLMEEstimation, LinearAlgebra, Printf, Random, Statistics
import NLMEEstimation as NE

const UNITS, ROWS = parse(Int, ARGS[1]), parse(Int, ARGS[2])
const OUTPUT = open(ARGS[3], "a")
const ROUND_PAIRS, START_PAIRS, FIT_PAIRS =
    length(ARGS) >= 6 ? parse.(Int, ARGS[4:6]) : (30, 12, 12)
const ROUNDS_PER_SAMPLE = 5
const WARM_ROUNDS = 20
const WORK_ROUNDS = 200
BLAS.set_num_threads(1)

report(args...) = (println(args...); println(OUTPUT, args...); flush(stdout); flush(OUTPUT))

function pk_data(rng, n, r)
    times = collect(range(0.5, 12.0; length = r))
    weight = 0.3 .* randn(rng, n)
    dose = rand(rng, [80.0, 100.0, 120.0], n)
    dv = map(1:n) do i
        CL = exp(log(2.0) + 0.5 * weight[i] + 0.25 * randn(rng))
        V = exp(log(10.0) + 0.2 * randn(rng))
        dose[i] / V .* exp.(-(CL / V) .* times) .+ 0.4 .* randn(rng, r)
    end
    return (; subject = ["s$i" for i in 1:n], dose, weight, t = fill(times, n), dv)
end

pk_model(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | pk | subject)
    log_V ~ 1 + (1 | pk | subject)
    @plate for i in eachindex(log_CL)
        CL = exp(log_CL[i])
        Vc = exp(log_V[i])
        pred[i] = dose[i] / Vc .* exp.(-(CL / Vc) .* t[i])
        dv[i] ~ normal(pred[i], sigma)
    end
end

const MODEL = brm_nlme_model(RKBRMI(pk_model(pk_data(Xoshiro(1), UNITS, ROWS)));
    ad_backend = AutoEnzyme(; mode = Enzyme.Reverse))
const START = brm_nlme_start(MODEL)
const θ0, σ0, ω0, H0 = START.theta, START.sigma, START.omega, START.eta
const IDS = collect(1:UNITS)
const PRIOR = re_prior(MODEL, ω0)
const KERNELS = (lockstep = WarmupHMCNUTS(), threads = WarmupHMCNUTS(threads = true))

function started(kernel, seed)
    H = copy(H0)
    time = @elapsed states = NE._saem_nuts_start!(Xoshiro(seed), kernel, MODEL, IDS, θ0, σ0,
        PRIOR, H)
    return (; states, H, time)
end
rounds!(chains, kernel, n) = for _ in 1:n
    NE._saem_nuts_round!(chains.states, MODEL, IDS, θ0, σ0, PRIOR, chains.H; kernel.threads)
end
evaluations(chains) = sum(state.total_evaluation_counter for state in chains.states)

# Run `sample(route)` for both routes `pairs` times, alternating which goes first, after
# one warm-up of each; returns one value per pair and route.
function paired(sample, pairs)
    sample(:lockstep, 0)
    sample(:threads, 0)
    return map(1:pairs) do k
        order = isodd(k) ? (:lockstep, :threads) : (:threads, :lockstep)
        results = Dict(route => sample(route, k) for route in order)
        (lockstep = results[:lockstep], threads = results[:threads])
    end
end

quartiles(x) = (quantile(x, 0.25), median(x), quantile(x, 0.75))
function summarize(label, lockstep, threads; unit = "ms", scale = 1e3)
    ratios = lockstep ./ threads
    a, b, c = quartiles(ratios)
    report(@sprintf("  %-22s lockstep %8.2f %s, threads %8.2f %s (medians); lockstep/threads %.2f (quartiles %.2f–%.2f), threads no slower in %d of %d pairs",
        label, scale * median(lockstep), unit, scale * median(threads), unit, b, a, c,
        count(>=(1), ratios), length(ratios)))
end

# A round: every subject's NUTS transition at fixed θ and σ, the kernel alone.
function measure_rounds()
    report("\nRounds: $ROUND_PAIRS pairs of $ROUNDS_PER_SAMPLE rounds from one start per route, ",
        "after $WARM_ROUNDS warm rounds")
    chains = map(k -> started(k, 1), KERNELS)
    foreach(keys(KERNELS)) do route
        rounds!(chains[route], KERNELS[route], WARM_ROUNDS)
    end
    samples = paired(ROUND_PAIRS) do route, _
        before = evaluations(chains[route])
        time = @elapsed rounds!(chains[route], KERNELS[route], ROUNDS_PER_SAMPLE)
        (; time, evaluations = evaluations(chains[route]) - before)
    end
    for (k, s) in enumerate(samples)
        report(@sprintf("  pair %2d  lockstep %8.3f ms (%5d evaluations)  threads %8.3f ms (%5d evaluations)",
            k, 1e3s.lockstep.time, s.lockstep.evaluations, 1e3s.threads.time, s.threads.evaluations))
    end
    lockstep = [s.lockstep.time for s in samples] ./ ROUNDS_PER_SAMPLE
    threads = [s.threads.time for s in samples] ./ ROUNDS_PER_SAMPLE
    summarize("per round", lockstep, threads)
    summarize("per evaluation",
        [s.lockstep.time / s.lockstep.evaluations for s in samples],
        [s.threads.time / s.threads.evaluations for s in samples]; unit = "µs", scale = 1e6)
    return median(lockstep), median(threads)
end

function measure_starts()
    report("\nThe Pathfinder start: $START_PAIRS pairs, a new seed per pair")
    samples = paired(START_PAIRS) do route, k
        started(KERNELS[route], k + 1).time
    end
    for (k, s) in enumerate(samples)
        report(@sprintf("  pair %2d  lockstep %8.2f ms  threads %8.2f ms", k, 1e3s.lockstep, 1e3s.threads))
    end
    summarize("start", [s.lockstep for s in samples], [s.threads for s in samples])
    return median(s.lockstep for s in samples), median(s.threads for s in samples)
end

# A whole fit, with BRM's options: the wall time the acceptance names. The two routes'
# seeded fits are not expected to be bit-identical, since BRM's batched and one-unit routes
# can differ by roundoff, so the difference is printed rather than asserted.
const OPTIONS = SAEMOptions(; iterations = 40, burnin = 20, mcmc_burnin = 5)
function measure_fits()
    report("\nA fit of 20 burn-in and 20 SA iterations (5 MCMC burn-in sweeps): $FIT_PAIRS pairs, ",
        "the same seed within a pair")
    samples = paired(FIT_PAIRS) do route, k
        GC.gc()
        @timed saem(MODEL, θ0, σ0, ω0; rng = Xoshiro(k + 100), options = OPTIONS,
            kernel = KERNELS[route])
    end
    for (k, s) in enumerate(samples)
        report(@sprintf("  pair %2d  lockstep %8.1f ms (GC %6.1f)  threads %8.1f ms (GC %6.1f)  largest |Δθ| %.2e  |Δσ| %.2e",
            k, 1e3s.lockstep.time, 1e3s.lockstep.gctime, 1e3s.threads.time, 1e3s.threads.gctime,
            maximum(abs, s.lockstep.value.theta .- s.threads.value.theta),
            maximum(abs, s.lockstep.value.sigma .- s.threads.value.sigma)))
    end
    lockstep, threads = [s.lockstep.time for s in samples], [s.threads.time for s in samples]
    summarize("fit", lockstep, threads)
    return median(lockstep), median(threads)
end

# The work a round asks for, which no host's load changes.
function measure_work()
    report("\nThe work in a round: $WORK_ROUNDS lockstep rounds at the starting point")
    chains = started(KERNELS.lockstep, 1)
    rounds!(chains, KERNELS.lockstep, WARM_ROUNDS)
    used = zeros(Int, UNITS, WORK_ROUNDS)
    for r in 1:WORK_ROUNDS
        before = [state.total_evaluation_counter for state in chains.states]
        rounds!(chains, KERNELS.lockstep, 1)
        used[:, r] = [state.total_evaluation_counter for state in chains.states] .- before
    end
    report(@sprintf("  evaluations per subject and round: mean %.1f, range %d–%d",
        mean(used), minimum(used), maximum(used)))
    report(@sprintf("  subject evaluations per round: %.0f; lockstep's batched calls per round: %.1f, of which every subject waiting: %.1f",
        mean(sum(used; dims = 1)), mean(maximum(used; dims = 1)), mean(minimum(used; dims = 1))))
    report(@sprintf("  the longest chain bounds threads at %.1f× on any number of threads",
        mean(sum(used; dims = 1)) / mean(maximum(used; dims = 1))))
end

# What a batched call costs by its size, warm, serial: the median of 21 calls.
function measure_calls()
    report("\nA conditional_loglikelihoods_and_gradients call by the subjects it lists (median of 21)")
    for n in unique(clamp.((1, 2, 4, 20, UNITS - 1, UNITS), 1, UNITS))
        subjects, H = IDS[1:n], H0[:, 1:n]
        NE.conditional_loglikelihoods_and_gradients(MODEL, subjects, θ0, σ0, H)
        time = median(@elapsed(NE.conditional_loglikelihoods_and_gradients(MODEL, subjects, θ0, σ0, H))
                      for _ in 1:21)
        report(@sprintf("  %3d subjects: %9.1f µs, %7.2f µs per subject", n, 1e6time, 1e6time / n))
    end
end

report("\n══ ", UNITS, " subjects × ", ROWS, " observations, ", Threads.nthreads(), " thread(s), ",
    Threads.ngcthreads(), " GC thread(s), a grant of ", get(ENV, "KB_COMPUTE_CPUS", "?"), " CPU(s) ══")
report("Julia ", VERSION, ", NLMEEstimation ", pkgversion(NE), ", WarmupHMC ", pkgversion(NE.WarmupHMC),
    ", BayesianRegressionModels ", pkgversion(BayesianRegressionModels), ", Enzyme ", pkgversion(Enzyme))
measure_calls()
measure_work()
round_times = measure_rounds()
start_times = measure_starts()
fit_times = measure_fits()
kernel = start_times .+ (OPTIONS.iterations + OPTIONS.mcmc_burnin) .* round_times
report(@sprintf("\nThe kernel's share of the fit, start plus %d rounds: lockstep %.0f%%, threads %.0f%%",
    OPTIONS.iterations + OPTIONS.mcmc_burnin, 100kernel[1] / fit_times[1], 100kernel[2] / fit_times[2]))
report("Evaluators the model's pool prepared: ", MODEL.cell_gradient.evaluators.count[])
close(OUTPUT)
