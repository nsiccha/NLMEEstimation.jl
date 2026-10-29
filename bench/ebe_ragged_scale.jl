# Empirical Bayes, the marginal likelihood and the covariance step on ragged subjects
# with thousands of nested random effects: time, allocation and model calls as a
# subject grows, against the closed-form answers.
#
# Run from the repository root with the test environment:
#   julia --project=test bench/ebe_ragged_scale.jl [output]
#
# The output (default `bench/ebe_ragged_scale_results.txt`) receives everything
# printed here.
#
# The model is the test suite's `RaggedOccasions` (`occasions_model`): each subject
# has one subject effect (family 1) and one effect per nested unit (family 2), and
# unit j has 2–4 Gaussian observations. Its posterior is Gaussian, so the exact
# modes are the posterior means (`occasions_unit_posterior`), and Laplace is exact.
# `occasions_sparse` declares its η Hessian pattern (`eta_hessian_sparsity`): the
# subject effect shares observations with every unit, units share none.
using NLMEEstimation
import NLMEEstimation as NE
using LinearAlgebra
using Printf
using Random
import ForwardDiff
import Optim
include(joinpath(@__DIR__, "..", "test", "common.jl"))
using .TestCommon: RaggedOccasions, occasions_exact_nested, occasions_model,
    occasions_unit_posterior, occasions_sparse

const OUTPUT = open(length(ARGS) >= 1 ? ARGS[1] :
    joinpath(@__DIR__, "ebe_ragged_scale_results.txt"), "w")
report(args...) = (println(args...); println(OUTPUT, args...); flush(OUTPUT))

# The units of `bench/saem_ragged_kernels.jl`'s first seed, and the atlas's largest
# region.
const UNITS = [250, 500, 1000, 2000, 4000, 8000]
const LARGEST = 25_871

function maximum_likelihood(m)
    f(x) = -occasions_exact_nested(m, x)
    x0 = [2.0, log(0.25), log(0.6), log(0.4)]
    r = Optim.optimize(f, x -> ForwardDiff.gradient(f, x), x0, Optim.LBFGS(); inplace = false)
    return Optim.minimizer(r)
end

# The exact mode of subject i: the subject effect's posterior mean, then the units'.
function exact_mode(m, i, x)
    θ, s2, a2, b2 = x[1], exp(2x[2]), exp(2x[3]), exp(2x[4])
    counts = length.(m.y[i])
    means = [sum(o) / length(o) for o in m.y[i]]
    w = counts ./ (s2 .+ counts .* b2)
    va = inv(inv(a2) + sum(w))
    return vcat(va * sum(w .* (means .- θ)), first(occasions_unit_posterior(m, i, x)))
end

function ebes(inner, model, x, method)
    inner.calls[] = 0
    stats = @timed empirical_bayes(model, x[1:1], x[2:2], x[3:4]; method)
    r = stats.value
    error = maximum(k -> maximum(abs, r.modes[k] - exact_mode(inner, r.subjects[k], x)),
        eachindex(r.subjects); init = 0.0)
    return @sprintf("%.2f\t%.1f\t%d\t%d\t%.1e", stats.time, stats.bytes / 1e6, inner.calls[],
        maximum(r.iterations), error)
end

function main()
    report("Empirical Bayes, marginal likelihood and covariance step on ragged subjects with many ",
        "nested effects: Julia $VERSION, NLMEEstimation ",
        readchomp(`git -C $(joinpath(@__DIR__, "..")) rev-parse --short HEAD`))
    # Compile every path on a small model first, so the timings are warm.
    small = occasions_model(seed = 0, units = [5, 8])
    xs = [2.0, log(0.25), log(0.6), log(0.4)]
    for model in (small, occasions_sparse(small)), method in (NewtonEBE(), QuasiNewtonEBE())
        model isa RaggedOccasions && method isa NewtonEBE && continue
        empirical_bayes(model, xs[1:1], xs[2:2], xs[3:4]; method)
    end
    marginal_loglikelihood(occasions_sparse(small), xs[1:1], xs[2:2], xs[3:4];
        approximation = AdaptiveGaussHermite(nodes = 1))
    covariance_step(occasions_sparse(small), xs[1:1], xs[2:2], xs[3:4];
        approximation = AdaptiveGaussHermite(nodes = 1), method = :r)
    # S alone (eight subjects, so it has full rank): twenty draws need not give a
    # positive definite R.
    for approximation in (ImportanceSampling(draws = 20), LouisInformation(draws = 20, burnin = 5))
        covariance_step(occasions_sparse(occasions_model()), xs[1:1], xs[2:2], xs[3:4];
            approximation, method = :s)
    end

    model = occasions_model(seed = 101, units = UNITS)
    x = maximum_likelihood(model)
    report("subjects: $(join(UNITS, ", ")) nested units; exact maximum-likelihood estimate ",
        join((@sprintf("%.4f", v) for v in x), ", "))
    report("\nempirical_bayes per subject (seconds, MB allocated, model calls, iterations, ",
        "max |mode − exact|)")
    report("units\tdeclared, Newton\tdeclared, BFGS\tundeclared BFGS (dense curvature check)")
    for n in UNITS
        one = RaggedOccasions([model.y[findfirst(==(n), UNITS)]])
        dense = n <= 2000 ? ebes(one, one, x, QuasiNewtonEBE()) : "(skipped: n² memory)"
        report(n, "\t", ebes(one, occasions_sparse(one), x, NewtonEBE()), "\t",
            ebes(one, occasions_sparse(one), x, QuasiNewtonEBE()), "\t", dense)
    end
    report("all six\t", ebes(model, occasions_sparse(model), x, NewtonEBE()), "\t",
        ebes(model, occasions_sparse(model), x, QuasiNewtonEBE()))

    declared = occasions_sparse(model)
    stats = @timed marginal_loglikelihood(declared, x[1:1], x[2:2], x[3:4];
        approximation = AdaptiveGaussHermite(nodes = 1))
    report(@sprintf("\nLaplace marginal of all six: %.2f s, %.1f MB; minus the exact marginal: %.1e",
        stats.time, stats.bytes / 1e6, stats.value.loglikelihood - occasions_exact_nested(model, x)))
    stats = @timed covariance_step(declared, x[1:1], x[2:2], x[3:4];
        approximation = AdaptiveGaussHermite(nodes = 1), method = :r)
    exact_R = -ForwardDiff.hessian(y -> occasions_exact_nested(model, y), x)
    report(@sprintf("covariance step (:r, Laplace) of all six: %.2f s, %.1f MB, %d evaluations; ",
        stats.time, stats.bytes / 1e6, stats.value.evaluations),
        @sprintf("max relative error of R against the exact Hessian: %.1e",
            maximum(abs, stats.value.R - exact_R) / maximum(abs, exact_R)))
    report("standard errors: ", join((@sprintf("%.5f", v) for v in stats.value.standard_errors), ", "),
        "; exact: ", join((@sprintf("%.5f", v) for v in sqrt.(diag(inv(exact_R)))), ", "))

    # The covariance step's approximations on the same six subjects: today's
    # default for a ragged SAEM fit (importance sampling, t₄ proposal), Laplace,
    # and Louis's identity from NUTS draws; standard errors against the exact ones.
    report("\ncovariance step (:r) of all six by approximation: seconds, MB, max |SE/exact SE − 1|")
    exact_se = sqrt.(diag(inv(exact_R)))
    for (name, approximation) in (("ImportanceSampling() (default)", ImportanceSampling()),
            ("AdaptiveGaussHermite(nodes = 1)", AdaptiveGaussHermite(nodes = 1)),
            ("LouisInformation()", LouisInformation()))
        stats = @timed covariance_step(declared, x[1:1], x[2:2], x[3:4]; approximation,
            rng = Xoshiro(1), method = :r)
        ratio = stats.value.standard_errors ./ exact_se
        report(@sprintf("%s\t%.1f\t%.0f\t%.3f", name, stats.time, stats.bytes / 1e6,
            maximum(abs, ratio .- 1)), "\t(", join((@sprintf("%.3f", v) for v in ratio), ", "), ")")
    end

    largest = occasions_model(seed = 102, units = [LARGEST])
    report("\none subject of $LARGEST nested units ($(LARGEST + 1) random effects), at the estimate above")
    report("declared, Newton\t", ebes(largest, occasions_sparse(largest), x, NewtonEBE()))
    report("declared, BFGS\t", ebes(largest, occasions_sparse(largest), x, QuasiNewtonEBE()))
    # The first call colors the subject's pattern (time quadratic in its effects);
    # later calls reuse the coloring.
    for label in ("Laplace marginal", "Laplace marginal again")
        stats = @timed marginal_loglikelihood(occasions_sparse(largest), x[1:1], x[2:2], x[3:4];
            approximation = AdaptiveGaussHermite(nodes = 1))
        report(@sprintf("%s: %.2f s, %.1f MB; minus the exact marginal: %.1e", label,
            stats.time, stats.bytes / 1e6,
            stats.value.loglikelihood - occasions_exact_nested(largest, x)))
    end
end

main()
close(OUTPUT)
