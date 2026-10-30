# SAEM with `WarmupHMCNUTS` (NLMEEstimation 0.2.0: Pathfinder start, then a kernel
# that adapts its diagonal metric and step size on every transition) from starts
# whose scales are right, 4× too wide and 4× too narrow, against the exact maximum
# likelihood of the test suite's nested Gaussian model: how far the fit ends from
# the maximum, how many transitions diverged, and how many gradient evaluations
# each subject's chain costs per transition as the population parameters move.
#
# Run from the repository root with the test environment:
#   julia --project=test bench/saem_nuts_continuous.jl [output]
#
# The output (default `bench/saem_nuts_continuous_results.txt`) receives everything
# printed here. The 0.1.x kernel, warmed up once at the start and then fixed, was
# measured the same way (snag `warmuphmcnuts-in-6e015ad3`): a 4× too-wide start
# ended 0.79–3.65 log-likelihood units below the maximum with up to 136 divergent
# transitions, and a 4× too-narrow start cost 21,000–25,000 evaluations per subject
# over the SA phase against 4,800–6,600 from a start at the right scales.
using NLMEEstimation
import NLMEEstimation as NE
using LinearAlgebra
using Printf
using Random
using Statistics
import ForwardDiff
import Optim
include(joinpath(@__DIR__, "..", "test", "common.jl"))
using .TestCommon: OccasionsBlocks, occasions_exact_nested, occasions_model

const OUTPUT = open(length(ARGS) >= 1 ? ARGS[1] :
    joinpath(@__DIR__, "saem_nuts_continuous_results.txt"), "w")
report(args...) = (println(args...); println(OUTPUT, args...); flush(OUTPUT))

# Counts the model's η-gradient evaluations in segments at each change of (θ, σ):
# segment 1 is the start (Pathfinder) plus the MCMC burn-in, each later one an SA
# iteration's kernel round.
struct Counted
    inner::OccasionsBlocks
    segments::Vector{Int}
    last::Base.RefValue{Vector{Float64}}
end
Counted(m) = Counted(m, Int[], Ref(Float64[]))
for f in (:nsubjects, :nlme_layout, :mu_referencing, :provides_eta_hessian)
    @eval NE.$f(m::Counted) = NE.$f(m.inner)
end
NE.eta_hessian_sparsity(m::Counted, i) = NE.eta_hessian_sparsity(m.inner, i)
NE.conditional_loglikelihood(m::Counted, i::Integer, θ, σ, η) =
    NE.conditional_loglikelihood(m.inner, i, θ, σ, η)
NE.conditional_loglikelihood_and_population_gradient(m::Counted, i::Integer, θ, σ, η) =
    NE.conditional_loglikelihood_and_population_gradient(m.inner, i, θ, σ, η)
NE.conditional_loglikelihood_gradient_and_hessian(m::Counted, i::Integer, θ, σ, η) =
    NE.conditional_loglikelihood_gradient_and_hessian(m.inner, i, θ, σ, η)
NE.conditional_loglikelihood_hessian_blocks(m::Counted, subjects, θ, σ, H) =
    NE.conditional_loglikelihood_hessian_blocks(m.inner, subjects, θ, σ, H)
function NE.conditional_loglikelihood_and_gradient(m::Counted, i::Integer, θ, σ, η)
    key = vcat(θ, σ)
    if key != m.last[]
        push!(m.segments, 0)
        m.last[] = copy(key)
    end
    m.segments[end] += 1
    return NE.conditional_loglikelihood_and_gradient(m.inner, i, θ, σ, η)
end

const UNITS = [250, 500, 1000, 2000, 4000]
const OPTIONS = SAEMOptions()  # iterations 300, burnin 150, mcmc_burnin 20

function maximum_likelihood(m, x0)
    f(x) = -occasions_exact_nested(m, x)
    r = Optim.optimize(f, x -> ForwardDiff.gradient(f, x), x0, Optim.LBFGS(); inplace = false)
    return Optim.minimizer(r), -Optim.minimum(r)
end

function scenario(name, inner, x̂, best, θ0, σ0, ω0, seed)
    m = Counted(OccasionsBlocks(inner))
    n = length(UNITS)
    t = @elapsed fit = saem(m, θ0, σ0, ω0; rng = Xoshiro(seed), kernel = WarmupHMCNUTS(),
        options = OPTIONS)
    seg = m.segments
    sa = seg[2:end] ./ n
    xf = vcat(fit.theta, fit.sigma, fit.omega)
    report(@sprintf("%s, seed %d: %.1f s; %d divergent transitions; exact maximum minus the fit %.3f; nested SD %.4f (maximum likelihood %.4f), residual SD %.4f (%.4f)",
        name, seed, t, fit.nuts_divergences, best - occasions_exact_nested(inner, xf),
        exp(xf[4]), exp(x̂[4]), exp(xf[2]), exp(x̂[2])))
    report(@sprintf("  gradient evaluations per subject: start (Pathfinder + %d burn-in transitions) %.0f; per SA transition: iterations 1-10 %.1f, 11-50 %.1f, 51-150 %.1f (γ = 1), 151-300 %.1f; SA phase %.0f, %d dense Hessians formed",
        OPTIONS.mcmc_burnin, seg[1] / n, mean(sa[1:10]), mean(sa[11:50]), mean(sa[51:150]),
        mean(sa[151:end]), sum(sa), m.inner.dense_hessians[]))
end

function main()
    report("SAEM with WarmupHMCNUTS() on ragged subjects of ", join(UNITS, ", "),
        " nested units from three starts: Julia $VERSION, NLMEEstimation ",
        readchomp(`git -C $(joinpath(@__DIR__, "..")) rev-parse --short HEAD`),
        ", WarmupHMC ", pkgversion(NE.WarmupHMC))
    small = occasions_model(seed = 0, units = [5, 8])
    saem(Counted(OccasionsBlocks(small)), [2.0], [log(0.25)], [log(0.6), log(0.4)];
        rng = Xoshiro(0), kernel = WarmupHMCNUTS(),
        options = SAEMOptions(iterations = 4, burnin = 2, mcmc_burnin = 1))
    inner = occasions_model(seed = 103, units = UNITS)
    x̂, best = maximum_likelihood(inner, [2.0, log(0.25), log(0.6), log(0.4)])
    report("exact maximum-likelihood estimate θ, log σ, log ω: ",
        join((@sprintf("%.4f", v) for v in x̂), ", "), "; default SAEMOptions")
    starts = [("start at the maximum-likelihood scales", [2.0], [x̂[2]], x̂[3:4]),
        ("residual, subject and nested SDs started 4x too wide", [2.0], [x̂[2] + log(4)],
            x̂[3:4] .+ log(4)),
        ("residual, subject and nested SDs started 4x too narrow", [2.0], [x̂[2] - log(4)],
            x̂[3:4] .- log(4))]
    for (name, θ0, σ0, ω0) in starts
        report()
        for seed in 1:3
            scenario(name, inner, x̂, best, θ0, σ0, ω0, seed)
        end
    end
end

main()
