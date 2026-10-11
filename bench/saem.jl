# Run from the repository root with `julia --project=test bench/saem.jl` after
# developing the local package into the test environment. Timings exclude Julia
# compilation and model construction; they include the full SAEM fit.
using NLMEEstimation
using Random
using LinearAlgebra
using Statistics
include(joinpath(@__DIR__, "..", "test", "common.jl"))
using .TestCommon

BLAS.set_num_threads(1)

const OPTIONS = SAEMOptions(iterations = 120, burnin = 40, mcmc_burnin = 20,
    mh_rounds = 2, adapt_iterations = 80, numeric_maxiter = 4,
    mstep_interval = 5)
const WARMUP = SAEMOptions(iterations = 8, burnin = 3, mcmc_burnin = 2,
    mh_rounds = 2, adapt_iterations = 5, numeric_maxiter = 2,
    mstep_interval = 4)

function run_case(label, model, θ0, σ0, ω0, data_seed, sampler_seed)
    trial = @timed saem(model, θ0, σ0, ω0;
        rng = MersenneTwister(sampler_seed), options = OPTIONS)
    fit = trial.value
    println(join((label, data_seed, sampler_seed, trial.time, trial.bytes,
        repr(fit.theta), repr(fit.sigma), repr(diag(fit.omega_matrix)),
        fit.acceptance.prior, fit.acceptance.random_walk), '\t'))
    return trial.time, fit
end

theoph = theoph_model()
theoph_ω0 = omega_parameters(nlme_layout(theoph), Ω_THEOPH)
saem(theoph, θ_THEOPH, σ_THEOPH, theoph_ω0;
    rng = MersenneTwister(1), options = WARMUP)

println("Julia $(VERSION); threads=$(Threads.nthreads()); BLAS threads=$(BLAS.get_num_threads())")
println("case\tdata_seed\tsampler_seed\tseconds\tallocated_bytes\ttheta\tsigma\tomega_diagonal\tprior_acceptance\trandom_walk_acceptance")
theoph_times = Float64[]
for seed in 2026:2028
    elapsed, _ = run_case("Theoph", theoph, θ_THEOPH, σ_THEOPH, theoph_ω0,
        "Theoph.csv", seed)
    push!(theoph_times, elapsed)
end

sim_times = Float64[]
sim_errors = Float64[]
sim_θ0 = copy(θ_SIM)
sim_θ0[1] += 0.5
for seed in 21:30
    model, _ = simulated_model(seed = seed, nsubjects = 24)
    ω0 = omega_parameters(nlme_layout(model), Ω_SIM)
    elapsed, fit = run_case("Simulated24", model, sim_θ0, σ_SIM, ω0,
        seed, 1000 + seed)
    push!(sim_times, elapsed)
    push!(sim_errors, norm(fit.theta - θ_SIM))
end
println("median seconds: Theoph=$(median(theoph_times)), Simulated24=$(median(sim_times))")
println("mean simulated θ error norm=$(mean(sim_errors)); max=$(maximum(sim_errors))")
