# SAEM kernels on a ragged layout whose subjects carry thousands of nested
# random effects: how far each kernel moves the unit effects, and how close each
# fit ends to the maximum likelihood.
#
# Run from the repository root with the test environment:
#   julia --project=test bench/saem_ragged_kernels.jl [seeds] [output]
#
# The output (default `bench/saem_ragged_kernels_results.txt`) receives everything
# printed here.
#
# The model is the test suite's `RaggedOccasions` (`occasions_model`) with many
# nested units per subject: each subject (a region) has one region effect a_i
# (family 1, SD 0.6) and one effect b_ij per nested unit (family 2, SD 0.4), and
# unit j has 2–4 observations y_ijt = θ + a_i + b_ij + ε, θ = 2, ε ~ N(0, 0.25²).
# Its marginal likelihood and the posterior of every unit effect are available
# in closed form, so each fit is compared with the exact maximum and each chain
# with the exact posterior at the maximum-likelihood estimate.
using NLMEEstimation
import NLMEEstimation as NE
using LinearAlgebra
using Printf
using Random
using Statistics
import ForwardDiff
import Optim
include(joinpath(@__DIR__, "..", "test", "common.jl"))
using .TestCommon: RaggedOccasions, occasions_exact, occasions_exact_nested,
    occasions_model, occasions_unit_posterior

const SEEDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
const OUTPUT = open(length(ARGS) >= 2 ? ARGS[2] :
    joinpath(@__DIR__, "saem_ragged_kernels_results.txt"), "w")
report(args...) = (println(args...); println(OUTPUT, args...); flush(OUTPUT))

const UNITS = [250, 500, 1000, 2000, 4000, 8000]
const θ_TRUE, σ_TRUE, SD_REGION, SD_UNIT = 2.0, 0.25, 0.6, 0.4

function maximum_likelihood(m)
    f(x) = -occasions_exact_nested(m, x)
    x0 = [θ_TRUE, log(σ_TRUE), log(SD_REGION), log(SD_UNIT)]
    r = Optim.optimize(f, x -> ForwardDiff.gradient(f, x), x0, Optim.LBFGS(); inplace = false)
    return Optim.minimizer(r), -Optim.minimum(r)
end

const KERNELS = [("MH", StandardSAEMMH()), ("MALA", DualAveragingMALA()),
    ("NUTS", WarmupHMCNUTS())]
const OPTIONS = SAEMOptions(iterations = 300, burnin = 150)
const θ0, σ0, ω0 = [θ_TRUE], [log(σ_TRUE)], [log(SD_REGION), log(SD_UNIT)]

function measure(model, kernel, seed, x̂, ℓ̂)
    stats = @timed saem(model, θ0, σ0, ω0; rng = Xoshiro(seed), options = OPTIONS, kernel)
    fit = stats.value
    x = vcat(fit.theta, fit.sigma, fit.omega)
    # Unit-effect chains against their exact posterior at the maximum-likelihood
    # estimate: z² averages 1 for posterior draws and grows with the distance a
    # chain keeps from that posterior; `moved` is the chain's correlation with
    # the posterior mean, about 0.93 for a posterior draw here (the chains start
    # at zero).
    z2 = Float64[]
    moved = Float64[]
    for (k, i) in enumerate(fit.subjects)
        mb, sb = occasions_unit_posterior(model, i, x̂)
        b = fit.eta[k][2:end]
        append!(z2, ((b .- mb) ./ sb) .^ 2)
        push!(moved, cor(b, mb))
    end
    return (; seconds = stats.time, gb = stats.bytes / 1e9, gap = ℓ̂ - occasions_exact_nested(model, x),
        sd_unit = exp(x[4]), sd_unit_mle = exp(x̂[4]), z2 = mean(z2),
        moved_min = minimum(moved), moved_max = maximum(moved))
end

function main()
    report("SAEM kernels on a ragged layout with large nested subjects: Julia $VERSION, ",
        "$(Threads.nthreads()) threads, NLMEEstimation ",
        readchomp(`git -C $(joinpath(@__DIR__, "..")) rev-parse --short HEAD`))
    report("units per subject: $(join(UNITS, ", ")) ($(sum(UNITS)) unit effects plus ",
        "$(length(UNITS)) region effects); SAEMOptions(iterations = 300, burnin = 150)")
    # Positive control: the closed-form likelihood equals the dense one.
    small = occasions_model()
    xs = [2.1, log(0.3), log(0.5), log(0.35)]
    report(@sprintf("closed-form vs dense exact likelihood on the test fixture: %.3e",
        abs(occasions_exact_nested(small, xs) - occasions_exact(small, xs))))
    # Compile every kernel on a small model first, so the timings are warm.
    for (_, kernel) in KERNELS
        saem(occasions_model(seed = 0, units = [5, 8]), θ0, σ0, ω0; rng = Xoshiro(0),
            options = SAEMOptions(iterations = 4, burnin = 2), kernel)
    end
    report("seed\tkernel\tseconds\tGB allocated\tmarginal gap\tsd_unit (MLE)\t",
        "mean z² of unit effects\tcor(chain, posterior mean) min–max")
    for seed in 1:SEEDS
        model = occasions_model(seed = 100 + seed, units = UNITS)
        x̂, ℓ̂ = maximum_likelihood(model)
        for (name, kernel) in KERNELS
            r = measure(model, kernel, seed, x̂, ℓ̂)
            report(@sprintf("%d\t%s\t%.1f\t%.2f\t%.2f\t%.4f (%.4f)\t%.2f\t%.3f–%.3f",
                seed, name, r.seconds, r.gb, r.gap, r.sd_unit, r.sd_unit_mle, r.z2,
                r.moved_min, r.moved_max))
        end
    end
end

main()
close(OUTPUT)
