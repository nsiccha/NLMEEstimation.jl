# SAEM with random-effect families integrated by the model's own Laplace
# approximation (NLMEEstimation 0.2.4), on subjects with thousands of nested
# units: how close Laplace-EM (every family integrated) and SAEM sampling only the
# region effects (the units integrated) end to the maximum likelihood, against
# plain SAEM sampling every effect with NUTS.
#
# Run from the repository root with the test environment:
#   julia --project=test bench/saem_integrated.jl [seeds] [output]
#
# The output (default `bench/saem_integrated_results.txt`) receives everything
# printed here.
#
# The model is `bench/saem_ragged_kernels.jl`'s: the test suite's `RaggedOccasions`
# with six subjects (regions) of 250–8,000 nested units, one region effect a_i
# (family 1, SD 0.6) and one effect b_ij per unit (family 2, SD 0.4), and 2–4
# observations y_ijt = θ + a_i + b_ij + ε per unit, ε ~ N(0, 0.25²). It is Gaussian
# in its effects, so the Laplace approximation is exact and the marginal
# likelihood has a closed form. `OccasionsLaplace` integrates the units, or every
# effect, in closed form from each unit's count, mean and within-unit sum of
# squares: O(observations) per subject and call, as a sparse Laplace engine
# would be. Its population gradient is ForwardDiff's of the same closed form.
using NLMEEstimation
import NLMEEstimation as NE
using LinearAlgebra
using Printf
using Random
import ForwardDiff
import Optim
include(joinpath(@__DIR__, "..", "test", "common.jl"))
using .TestCommon: RaggedOccasions, occasions_exact_nested, occasions_model,
    occasions_unit_posterior

const SEEDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
const OUTPUT = open(length(ARGS) >= 2 ? ARGS[2] :
    joinpath(@__DIR__, "saem_integrated_results.txt"), "w")
report(args...) = (println(args...); println(OUTPUT, args...); flush(OUTPUT))

const UNITS = [250, 500, 1000, 2000, 4000, 8000]
const θ0, σ0, ω0 = [2.0], [log(0.25)], [log(0.6), log(0.4)]

struct OccasionsLaplace
    inner::RaggedOccasions
    counts::Vector{Vector{Int}}
    means::Vector{Vector{Float64}}
    within::Vector{Vector{Float64}}
    calls::Dict{Symbol, Int}
end
function OccasionsLaplace(m::RaggedOccasions)
    counts = [length.(o) for o in m.y]
    means = [[sum(u) / length(u) for u in o] for o in m.y]
    within = [[sum(abs2, u .- sum(u) / length(u)) for u in o] for o in m.y]
    return OccasionsLaplace(m, counts, means, within, Dict{Symbol, Int}())
end
for f in (:nsubjects, :nlme_layout, :mu_referencing)
    @eval NE.$f(m::OccasionsLaplace) = NE.$f(m.inner)
end
for f in (:conditional_loglikelihood, :conditional_loglikelihood_and_gradient,
        :conditional_loglikelihood_and_population_gradient)
    @eval NE.$f(m::OccasionsLaplace, i::Integer, θ, σ, η) = NE.$f(m.inner, i, θ, σ, η)
end
counted!(m, what) = (m.calls[what] = get(m.calls, what, 0) + 1)

# Subject i's Laplace conditional (exact here) with the units integrated (`F = [2]`,
# given the region effect a) or every effect (`F = [1, 2]`), and its moments: the
# derivative in a, the region effect's posterior mean and variance (when
# integrated), and the units' posterior means and variances.
function occasions_laplace(m::OccasionsLaplace, F, i, θ, σ, ω, a)
    n, ȳ, ss = m.counts[i], m.means[i], m.within[i]
    v, b2 = exp(2σ[1]), exp(2ω[2])
    d = v .+ n .* b2
    base = -sum(@. n * log(2π) + (n - 1) * log(v) + log(d) + ss / v) / 2
    if F == [2]
        r = ȳ .- (θ[1] + a)
        value = base - sum(@. n * r^2 / d) / 2
        shrink = @. b2 * n / d
        return value, sum(@. n * r / d), nothing, shrink .* r, @. b2 * v / d
    end
    a2 = exp(2ω[1])
    w = n ./ d
    r = ȳ .- θ[1]
    W, S, Q = sum(w), sum(w .* r), sum(w .* r .^ 2)
    value = base - (log(1 + a2 * W) + Q - S^2 / (inv(a2) + W)) / 2
    va = inv(inv(a2) + W)
    ma = va * S
    shrink = b2 .* w
    return value, 0.0, (ma, va), shrink .* (r .- ma), @. b2 * v / d + shrink^2 * va
end

function NE.laplace_conditionals_and_gradients(m::OccasionsLaplace, F, subjects, θ, σ, ω, H)
    counted!(m, :gradients)
    parts = [occasions_laplace(m, F, i, θ, σ, ω, H[k][1]) for (k, i) in enumerate(subjects)]
    return [p[1] for p in parts], [vcat(p[2], zeros(length(H[k]) - 1)) for (k, p) in enumerate(parts)]
end
function NE.laplace_conditionals(m::OccasionsLaplace, F, subjects, θ, σ, ω, H)
    counted!(m, :values)
    return [occasions_laplace(m, F, i, θ, σ, ω, H[k][1])[1] for (k, i) in enumerate(subjects)]
end
function NE.laplace_conditional_moments(m::OccasionsLaplace, F, subjects, θ, σ, ω, H)
    counted!(m, :moments)
    values, modes, covariances = Float64[], Vector{Float64}[], Vector{Matrix{Float64}}[]
    for (k, i) in enumerate(subjects)
        value, _, region, means, variances = occasions_laplace(m, F, i, θ, σ, ω, H[k][1])
        push!(values, value)
        push!(modes, vcat(region === nothing ? H[k][1] : region[1], means))
        push!(covariances, vcat(region === nothing ? Matrix{Float64}[] : [fill(region[2], 1, 1)],
            [fill(s2, 1, 1) for s2 in variances]))
    end
    return values, modes, covariances
end
function NE.laplace_conditionals_and_population_gradient(m::OccasionsLaplace, F, subjects,
        θ, σ, ω, H)
    counted!(m, :population)
    total(x) = sum(occasions_laplace(m, F, i, x[1:1], x[2:2], x[3:4], H[k][1])[1]
        for (k, i) in enumerate(subjects))
    g = ForwardDiff.gradient(total, vcat(θ, σ, ω))
    return NE.laplace_conditionals(m, F, subjects, θ, σ, ω, H), g[1:1], g[2:2], g[3:4]
end

function maximum_likelihood(m)
    f(x) = -occasions_exact_nested(m, x)
    r = Optim.optimize(f, x -> ForwardDiff.gradient(f, x), vcat(θ0, σ0, ω0), Optim.LBFGS();
        inplace = false)
    return Optim.minimizer(r), -Optim.minimum(r)
end

function measure(model, data, ℓ̂; kwargs...)
    stats = @timed saem(model, θ0, σ0, ω0; kwargs...)
    fit = stats.value
    x = vcat(fit.theta, fit.sigma, fit.omega)
    return (; fit, seconds = stats.time, gap = ℓ̂ - occasions_exact_nested(data, x),
        sd_unit = exp(x[4]))
end

function main()
    report("SAEM with integrated random-effect families on large nested subjects: Julia ",
        "$VERSION, $(Threads.nthreads()) threads, NLMEEstimation ",
        readchomp(`git -C $(joinpath(@__DIR__, "..")) rev-parse --short HEAD`))
    report("units per subject: $(join(UNITS, ", ")) ($(sum(UNITS)) unit effects plus ",
        "$(length(UNITS)) region effects); SAEMOptions(iterations = 300, burnin = 150) ",
        "unless stated")
    # Positive control: the integrated conditional equals the exact marginal.
    small = occasions_model()
    lap = OccasionsLaplace(small)
    xs = [2.1, log(0.3), log(0.5), log(0.35)]
    η = [zeros(1 + length(o)) for o in small.y]
    every = sum(NE.laplace_conditionals(lap, [1, 2], 1:length(small.y), xs[1:1], xs[2:2],
        xs[3:4], η))
    report(@sprintf("every effect integrated vs the exact marginal on the test fixture: %.3e",
        abs(every - occasions_exact_nested(small, xs))))
    check_protocol(lap, xs[1:1], xs[2:2], xs[3:4]; integrated = [2], subjects = 1:4)
    check_protocol(lap, xs[1:1], xs[2:2], xs[3:4]; integrated = [1, 2], subjects = 1:4)
    report("check_protocol passes for both sets on the test fixture")
    # Compile every configuration on a small model first, so the timings are warm.
    warm = occasions_model(seed = 0, units = [5, 8])
    short = SAEMOptions(iterations = 4, burnin = 2)
    for kernel in (StandardSAEMMH(), WarmupHMCNUTS())
        saem(OccasionsLaplace(warm), θ0, σ0, ω0; rng = Xoshiro(0), options = short, kernel,
            integrated = [2])
    end
    saem(OccasionsLaplace(warm), θ0, σ0, ω0; options = short, integrated = [1, 2])
    saem(warm, θ0, σ0, ω0; rng = Xoshiro(0), options = short, kernel = WarmupHMCNUTS())

    report("\nLaplace-EM (every family integrated): marginal gap to the maximum after k iterations")
    report("seed\tk = 10\t25\t50\t100\t200\t400\tseconds for 400\tmoments and population-gradient calls per iteration (400)")
    for seed in 1:SEEDS
        data = occasions_model(seed = 100 + seed, units = UNITS)
        _, ℓ̂ = maximum_likelihood(data)
        gaps = map((10, 25, 50, 100, 200, 400)) do k
            measure(OccasionsLaplace(data), data, ℓ̂; integrated = [1, 2],
                options = SAEMOptions(iterations = k)).gap
        end
        model = OccasionsLaplace(data)
        r = measure(model, data, ℓ̂; integrated = [1, 2], options = SAEMOptions(iterations = 400))
        report(@sprintf("%d\t%s\t%.2f\t%.1f / %.1f", seed,
            join([@sprintf("%.2e", g) for g in gaps], "\t"), r.seconds,
            model.calls[:moments] / 400, model.calls[:population] / 400))
    end

    report("\nSAEM sampling the region effects (one per subject, 6 in all), units integrated, ",
        "against plain SAEM sampling all $(sum(UNITS) + length(UNITS)) effects with NUTS")
    report("seed\tfit\tseconds\tmarginal gap\tsd_unit (MLE)\tdivergences\t",
        "Laplace gradient / value calls")
    for seed in 1:SEEDS
        data = occasions_model(seed = 100 + seed, units = UNITS)
        x̂, ℓ̂ = maximum_likelihood(data)
        for (name, kernel) in (("integrated, MH", StandardSAEMMH()),
                ("integrated, NUTS", WarmupHMCNUTS()))
            model = OccasionsLaplace(data)
            r = measure(model, data, ℓ̂; rng = Xoshiro(seed), kernel, integrated = [2],
                options = SAEMOptions(iterations = 300, burnin = 150))
            report(@sprintf("%d\t%s\t%.1f\t%.4f\t%.4f (%.4f)\t%d\t%d / %d", seed, name,
                r.seconds, r.gap, r.sd_unit, exp(x̂[4]), r.fit.nuts_divergences,
                get(model.calls, :gradients, 0), get(model.calls, :values, 0)))
        end
        r = measure(data, data, ℓ̂; rng = Xoshiro(seed), kernel = WarmupHMCNUTS(),
            options = SAEMOptions(iterations = 300, burnin = 150))
        report(@sprintf("%d\tplain, NUTS\t%.1f\t%.4f\t%.4f (%.4f)\t%d\t–", seed, r.seconds,
            r.gap, r.sd_unit, exp(x̂[4]), r.fit.nuts_divergences))
    end
end

main()
close(OUTPUT)
