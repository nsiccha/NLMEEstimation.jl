# Empirical Bayes on a gradient-only model with no declared η Hessian pattern, as a
# subject grows: what `QuasiNewtonEBE`'s curvature check costs, and the whole pass.
#
# Run from the repository root with the test environment:
#   julia --project=test bench/ebe_gradient_only_scale.jl [sizes...]
#
# The model is the test suite's `NestedWells`: one subject of n units under a subject
# effect, starting at η = 0. With a well, that point is a saddle whose upward
# direction combines two units and does not show on the Hessian's diagonal, so the
# mode has u = ±√(a - 1). Without one it is the mode.
#
# 1. The check alone at η = 0, with the units' information spread over five orders
#    of magnitude (0.1 to 10⁴ prior precisions): batched calls, time and memory,
#    beside what a dense difference block costs (n + 1 calls, (n + 1)² numbers).
# 2. The whole pass from η = 0 with a well, information 0.1 to 1. (Over five orders
#    of magnitude BFGS itself needs more than `maxiter = 200` iterations from any
#    start off the mode, with a dense check as with this one.)
using NLMEEstimation
import NLMEEstimation as NE
using Printf
include(joinpath(@__DIR__, "..", "test", "common.jl"))
using .TestCommon: nested_wells

const SIZES = isempty(ARGS) ? [2_000, 8_000, 36_000, 50_000] : parse.(Int, ARGS)

well(n, saddle) = saddle ? [(n ÷ 3, 2n ÷ 3)] : Tuple{Int, Int}[]
unit_prior(model) = omega_parameters(nlme_layout(model), [ones(1, 1), ones(1, 1)])

println("1. The check at η = 0, information 0.1 to 10⁴")
println("units\tsaddle\tfinding\tcalls\tseconds\tGB allocated\tdense calls\tdense GB")
for n in SIZES, saddle in (true, false)
    model = nested_wells(units = [n], wells = [well(n, saddle)])
    prior = re_prior(model, unit_prior(model))
    η = zeros(n + 1)
    g = last(NE.conditional_loglikelihood_and_gradient(model, 1, Float64[], Float64[], η))
    NE._lanczos_curvature(model, [1], Float64[], Float64[], prior, [η], [g], 20)
    model.calls[] = 0
    stats = @timed only(NE._lanczos_curvature(model, [1], Float64[], Float64[], prior, [η], [g], 20))
    @printf("%d\t%s\t%s\t%d\t%.2f\t%.2f\t%d\t%.1f\n", n, saddle,
        stats.value isa NE._UpwardCurvature ? "upward" : "none found", model.calls[],
        stats.time, stats.bytes / 1e9, n + 1, 8 * (n + 1)^2 / 1e9)
end

println("\n2. The whole pass from η = 0 with a well, information 0.1 to 1")
println("units\tcalls\titerations\tseconds\tGB allocated\tconverged\t|u| error")
for n in SIZES
    model = nested_wells(units = [n], wells = [well(n, true)], information = (0.1, 1.0))
    stats = @timed empirical_bayes(model, Float64[], Float64[], unit_prior(model);
        require_convergence = false)
    r = stats.value
    u = (r.modes[1][1 + n ÷ 3] + r.modes[1][1 + 2n ÷ 3]) / sqrt(2)
    @printf("%d\t%d\t%d\t%.2f\t%.2f\t%s\t%.1e\n", n, model.calls[], only(r.iterations),
        stats.time, stats.bytes / 1e9, only(r.converged), abs(abs(u) - sqrt(model.a - 1)))
end
