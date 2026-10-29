# SAEM with `WarmupHMCNUTS` on ragged subjects with tens of thousands of nested
# random effects, for a model that provides dense η Hessian blocks and declares its
# η Hessian pattern (BRM's adapter declares both): what each subject's warmup metric
# costs, how many dense Hessians the fit forms, and its time and peak memory.
#
# Run from the repository root with the test environment:
#   julia --project=test bench/saem_nuts_ragged_scale.jl [output]
#
# The output (default `bench/saem_nuts_ragged_scale_results.txt`) receives everything
# printed here.
#
# The model is the test suite's `RaggedOccasions` behind `OccasionsBlocks`: one
# subject effect (family 1) and one effect per nested unit (family 2), unit j with
# 2–4 Gaussian observations, dense Hessians on request (counted) and the declared
# pattern. Twelve subjects of 3,294 to 35,783 nested units, 157,808 in all. The
# marginal likelihood is available in closed form (`occasions_exact_nested`), so the
# fit is compared with the exact maximum.
using NLMEEstimation
import NLMEEstimation as NE
using LinearAlgebra
using Printf
using Random
import ForwardDiff
import Optim
include(joinpath(@__DIR__, "..", "test", "common.jl"))
using .TestCommon: OccasionsBlocks, occasions_exact_nested, occasions_model

const OUTPUT = open(length(ARGS) >= 1 ? ARGS[1] :
    joinpath(@__DIR__, "saem_nuts_ragged_scale_results.txt"), "w")
report(args...) = (println(args...); println(OUTPUT, args...); flush(OUTPUT))

const UNITS = [3294, 5000, 7000, 8500, 10000, 11000, 12000, 13500, 15000, 17000, 19731, 35783]
@assert sum(UNITS) == 157_808
const θ0, σ0, ω0 = [2.0], [log(0.25)], [log(0.6), log(0.4)]

peak_gb() = Sys.maxrss() / 2^30

function maximum_likelihood(m)
    f(x) = -occasions_exact_nested(m, x)
    r = Optim.optimize(f, x -> ForwardDiff.gradient(f, x), vcat(θ0, σ0, ω0), Optim.LBFGS();
        inplace = false)
    return Optim.minimizer(r), -Optim.minimum(r)
end

# The exact diagonal of subject i's negative complete-data η Hessian at x.
function exact_curvature(m, i, x)
    counts = length.(m.y[i])
    s2, a2, b2 = exp(2x[2]), exp(2x[3]), exp(2x[4])
    return vcat(sum(counts) / s2 + 1 / a2, counts ./ s2 .+ 1 / b2)
end

function metric_row(model, i, x, prior)
    model.dense_hessians[] = 0
    η = zeros(NE.neta(NE.nlme_layout(model), i))
    stats = @timed NE._saem_hessian_metric(model, i, x[1:1], x[2:2], prior, η; diagonal = true)
    error = maximum(abs, stats.value .* exact_curvature(model.inner, i, x) .- 1)
    row = @sprintf("%.3f\t%.1f\t%d\t%.1e", stats.time, stats.bytes / 1e6,
        model.dense_hessians[], error)
    model.declared || return row
    # The star coloring of the pattern on its own. The metric above colored it
    # (once per session since 0.1.7, which caches colorings by pattern), so this
    # times SparseMatrixColorings directly rather than the cache.
    pattern = NE._eta_hessian_pattern(model, i, length(η))
    coloring = @elapsed NE.SMC.coloring(pattern, NE._STAR_COLORING...)
    return row * @sprintf("\t%.3f", coloring)
end

function main()
    report("SAEM's NUTS warmup metric and fit on ragged subjects with many nested effects, for a ",
        "model with dense Hessian blocks and a declared pattern: Julia $VERSION, NLMEEstimation ",
        readchomp(`git -C $(joinpath(@__DIR__, "..")) rev-parse --short HEAD`))
    # Compile every path on a small model first, so the timings are warm.
    small = occasions_model(seed = 0, units = [5, 8])
    for declared in (true, false)
        m = OccasionsBlocks(small; declared)
        metric_row(m, 1, vcat(θ0, σ0, ω0), re_prior(m, ω0))
    end
    saem(OccasionsBlocks(small), θ0, σ0, ω0; rng = Xoshiro(0), kernel = WarmupHMCNUTS(),
        options = SAEMOptions(iterations = 4, burnin = 2, mcmc_burnin = 1))
    report(@sprintf("peak resident memory after compilation: %.2f GB", peak_gb()))

    inner = occasions_model(seed = 103, units = UNITS)
    x, best = maximum_likelihood(inner)
    report("subjects: $(join(UNITS, ", ")) nested units ($(sum(UNITS)) in all); exact ",
        "maximum-likelihood estimate ", join((@sprintf("%.4f", v) for v in x), ", "))
    declared, undeclared = OccasionsBlocks(inner), OccasionsBlocks(inner; declared = false)
    prior = re_prior(declared, x[3:4])

    report("\nwarmup metric (diagonal) per subject at η = 0, declared pattern: seconds, MB ",
        "allocated, dense Hessians formed, max |metric × exact curvature − 1|, and the seconds ",
        "the pattern's star coloring takes on its own")
    for (i, n) in enumerate(UNITS)
        report(n, "\t", metric_row(declared, i, x, prior))
    end
    report(@sprintf("peak resident memory after the metrics: %.2f GB", peak_gb()))

    declared.dense_hessians[] = 0
    stats = @timed saem(declared, θ0, σ0, ω0; rng = Xoshiro(1), kernel = WarmupHMCNUTS())
    fit = stats.value
    report("\nsaem(…; kernel = WarmupHMCNUTS()) with default SAEMOptions on all twelve: ",
        @sprintf("%.0f s, %.1f GB allocated, peak resident memory %.2f GB, ", stats.time,
            stats.bytes / 1e9, peak_gb()),
        @sprintf("%d dense Hessians formed, %d divergent transitions", declared.dense_hessians[],
            fit.nuts_divergences))
    xf = vcat(fit.theta, fit.sigma, fit.omega)
    report(@sprintf("exact maximum minus the fit's marginal log-likelihood: %.3f; ",
            best - occasions_exact_nested(inner, xf)),
        @sprintf("nested SD %.4f (maximum likelihood %.4f)", exp(xf[4]), exp(x[4])))

    # Last, so that its memory does not count in the peaks above.
    report("\nthe same metric without a declared pattern (a dense Hessian per subject, as for ",
        "every model with Hessian blocks before NLMEEstimation 0.1.4)")
    for (i, n) in enumerate(UNITS)
        report(n, "\t", n <= 10_000 ? metric_row(undeclared, i, x, prior) :
            @sprintf("(skipped: one dense Hessian is %.1f GB)", (n + 1)^2 * 8 / 1e9))
    end
    report(@sprintf("peak resident memory after them: %.2f GB", peak_gb()))
end

main()
close(OUTPUT)
