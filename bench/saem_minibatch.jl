# Minibatch SAEM on the oral one-compartment reference model: accuracy against
# cost for several batch sizes and iteration budgets.
#
# Run from the repository root with the test environment:
#   julia --project=test bench/saem_minibatch.jl [nsubjects] [seeds] [output]
#
# The output (default `bench/saem_minibatch_results.txt`) receives everything
# printed here.
#
# The reference is a long full-batch fit. Each configuration runs `seeds` fits
# and reports, per fit, the largest population-parameter error in units of the
# reference's standard errors (covariance step at the reference), the warm wall
# time, the subject evaluations the model performed (value and gradient calls
# per subject) and the time spent outside the model.
using NLMEEstimation
using NLMEEstimation.ReferenceModels
import NLMEEstimation as NE
using ADTypes: AutoEnzyme
import Enzyme
using LinearAlgebra
using Printf
using Random
using Statistics

const NSUBJECTS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 400
const SEEDS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
const OUTPUT = open(length(ARGS) >= 3 ? ARGS[3] : joinpath(@__DIR__, "saem_minibatch_results.txt"), "w")
report(args...) = (println(args...); println(OUTPUT, args...); flush(OUTPUT))
const θ_TRUE = [log(1.5), log(2.8), log(32.0), 0.75]
const σ_TRUE = [log(0.3), log(0.1)]
const Ω_TRUE = Matrix(Diagonal([0.4, 0.3, 0.2] .^ 2))

# Counts subject evaluations and the time they take, around the reference model.
struct Counted{M}
    inner::M
    evaluations::Base.RefValue{Int}
    seconds::Base.RefValue{Float64}
end
Counted(inner) = Counted(inner, Ref(0), Ref(0.0))
for f in (:nsubjects, :nlme_layout, :mu_referencing, :observation_model, :ad_backend,
        :provides_eta_hessian)
    @eval NE.$f(m::Counted) = NE.$f(m.inner)
end
NE.observations(m::Counted, i::Integer) = NE.observations(m.inner, i)
for f in (:conditional_loglikelihood, :conditional_loglikelihood_and_gradient,
        :conditional_loglikelihood_and_population_gradient, :observation_moments,
        :observation_moments_and_jacobians, :conditional_loglikelihood_gradient_and_hessian)
    @eval function NE.$f(m::Counted, i::Integer, θ, σ, η)
        m.evaluations[] += 1
        t = time()
        result = NE.$f(m.inner, i, θ, σ, η)
        m.seconds[] += time() - t
        return result
    end
end

function fit(model, options, seed)
    model.evaluations[] = 0
    model.seconds[] = 0.0
    ω0 = omega_parameters(nlme_layout(model), Ω_TRUE)
    θ0 = θ_TRUE .+ [0.3, -0.2, 0.2, 0.0]
    stats = @timed saem(model, θ0, σ_TRUE .+ 0.2, ω0; rng = Xoshiro(seed), options)
    return stats.value, stats.time, model.evaluations[], stats.time - model.seconds[]
end

point(f) = vcat(f.theta, f.sigma, f.omega)

subjects, _ = simulate_one_compartment_oral(Xoshiro(1), θ_TRUE, σ_TRUE, Ω_TRUE;
    nsubjects = NSUBJECTS)
model = Counted(OneCompartmentOral(subjects; backend = AutoEnzyme()))
n = NSUBJECTS

report("Minibatch SAEM benchmark: $n subjects, $SEEDS seeds, Julia $VERSION, ",
    "$(Threads.nthreads()) threads, NLMEEstimation ",
    readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`))
reference, reference_time, _, _ = fit(model, SAEMOptions(iterations = 2000, burnin = 500), 0)
se = covariance_step(model.inner, reference).standard_errors
report(@sprintf("reference: 2000 iterations in %.1f s; standard errors %s", reference_time,
    join([@sprintf("%.3g", s) for s in se], ", ")))

# `sweeps` SA iterations' worth of transitions per chain: the iteration counts
# scale by n / b. The MCMC burn-in sweeps every chain whatever the batch; five
# sweeps keep it small against the smallest budgets.
function options(b, sweeps)
    scale = b === nothing ? 1 : n ÷ b
    return SAEMOptions(; iterations = sweeps * scale, burnin = (sweeps ÷ 2) * scale,
        mcmc_burnin = 5, adapt_iterations = (sweeps ÷ 2) * scale, minibatch_size = b)
end

fit(model, options(n ÷ 10, 20), 99)   # compile the minibatch path
report("batch\tsweeps\titerations\tseed\tmax |error|/SE\tseconds\tsubject evaluations\tseconds outside the model")
for b in (nothing, n ÷ 4, n ÷ 10, n ÷ 40), sweeps in (10, 20, 50, 100)
    o = options(b, sweeps)
    for seed in 1:SEEDS
        f, seconds, evaluations, outside = fit(model, o, seed)
        error = maximum(abs.(point(f) - point(reference)) ./ se)
        report(@sprintf("%s\t%d\t%d\t%d\t%.2f\t%.2f\t%d\t%.2f", b === nothing ? "all" : string(b),
            sweeps, o.iterations, seed, error, seconds, evaluations, outside))
    end
end
