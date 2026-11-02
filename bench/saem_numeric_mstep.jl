# SAEM's numerical M-step for θ without a closed-form update: model evaluations
# and accuracy of `:mixture` (every retained draw), `:online` (the current draw's
# optimizer, damped) and `:score` (one Robbins–Monro step along the current
# draw's score), on models whose θ are not all mu-referenced.
#
# Run from the repository root with the test environment:
#   julia --project=test bench/saem_numeric_mstep.jl [seeds] [output]
#
# Models (each wraps a test fixture and declares a different `mu_referencing`, so
# that `numeric_mstep = :auto` does not resolve to `:online`):
# - growth: the linear growth model of `test/common.jl` with 40 subjects and no
#   mu-referencing; its marginal likelihood is exact, and so is its maximum.
# - pk-exponent: the oral one-compartment model (24 subjects) with the weight
#   exponent left out of `MuReferencing`, a structural θ in the likelihood.
# - pk-none: the same model without mu-referencing (four structural θ).
# - wide: a linear growth model (30 subjects, mu-referenced intercept and slope)
#   with 20 structural θ, the coefficients of observation-level covariates whose
#   scales span 10⁻² to 10², so that their standard errors span four orders of
#   magnitude; its marginal likelihood is exact, and its maximum is profiled
#   (θ by generalized least squares at fixed σ and Ω).
# The pk reference is the maximum of the adaptive Gauss–Hermite (3 nodes)
# marginal, and each fit is scored by its marginal there.
using NLMEEstimation
using NLMEEstimation.ReferenceModels
import NLMEEstimation as NE
using LinearAlgebra
using Printf
using Random
using Statistics
import Optim
import ForwardDiff
include(joinpath(@__DIR__, "..", "test", "common.jl"))
using .TestCommon

const SEEDS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
const OUTPUT = open(length(ARGS) >= 2 ? ARGS[2] :
    joinpath(@__DIR__, "saem_numeric_mstep_results.txt"), "w")
report(args...) = (println(args...); println(OUTPUT, args...); flush(OUTPUT))

# Counts subject evaluations by kind around a fixture, with its own mu-referencing.
struct Probe{M}
    inner::M
    mu::Any
    values::Base.RefValue{Int}
    gradients::Base.RefValue{Int}
    population::Base.RefValue{Int}
end
Probe(inner, mu) = Probe(inner, mu, Ref(0), Ref(0), Ref(0))
reset!(m::Probe) = (m.values[] = m.gradients[] = m.population[] = 0; m)
counts(m::Probe) = (values = m.values[], gradients = m.gradients[], population = m.population[])
for f in (:nsubjects, :nlme_layout, :observation_model, :ad_backend, :provides_eta_hessian,
        :parameter_names)
    @eval NE.$f(m::Probe) = NE.$f(m.inner)
end
NE.mu_referencing(m::Probe) = m.mu
NE.observations(m::Probe, i::Integer) = NE.observations(m.inner, i)
NE.population_logprior(m::Probe, θ, σ, ω) = NE.population_logprior(m.inner, θ, σ, ω)
for (f, counter) in ((:conditional_loglikelihood, :values),
        (:conditional_loglikelihood_and_gradient, :gradients),
        (:conditional_loglikelihood_gradient_and_hessian, :gradients),
        (:conditional_loglikelihood_and_population_gradient, :population),
        (:observation_moments, :values), (:observation_moments_and_jacobians, :gradients))
    @eval function NE.$f(m::Probe, i::Integer, θ, σ, η)
        m.$counter[] += 1
        return NE.$f(m.inner, i, θ, σ, η)
    end
end

point(f) = vcat(f.theta, f.sigma, f.omega)

function fit(model, θ0, σ0, ω0, mode, seed; iterations = 300, burnin = 150, kernel = StandardSAEMMH())
    reset!(model)
    options = SAEMOptions(; iterations, burnin, numeric_mstep = mode)
    stats = @timed saem(model, θ0, σ0, ω0; rng = Xoshiro(seed), options, kernel)
    return stats.value, stats.time, counts(model)
end

# ---------------------------------------------------------------------------
growth_inner = growth_model(n = 40)
growth = Probe(growth_inner, nothing)
growth_score(x) = growth_exact(growth_inner, x)
growth_x0 = vcat([0.5, 0.0], [log(0.5)], omega_parameters(GROWTH_LAYOUT, [0.3 0.0; 0.0 0.3]))
mle = Optim.optimize(x -> -growth_score(x),
    (G, x) -> (G .= .-ForwardDiff.gradient(growth_score, x)),
    vcat([1.0, 0.5], [log(0.3)], omega_parameters(GROWTH_LAYOUT, [0.5 0.1; 0.1 0.2])),
    Optim.LBFGS(), Optim.Options(g_tol = 1.0e-10))
Optim.converged(mle) || error("growth MLE did not converge: $mle")
growth_mle = Optim.minimizer(mle)
growth_se = sqrt.(diag(inv(-ForwardDiff.hessian(growth_score, growth_mle))))

pk_inner, _ = simulated_model(nsubjects = 24)
pk_layout = nlme_layout(pk_inner)
exponent_design = [Matrix(1.0I, 3, 3) for _ in 1:nsubjects(pk_inner)]
pk_exponent = Probe(pk_inner, MuReferencing(1:3, exponent_design))
pk_none = Probe(pk_inner, nothing)
const AGHQ = AdaptiveGaussHermite(nodes = 3)
pk_split(x) = (x[1:4], x[5:6], x[7:end])
pk_score(x) = try
    θ, σ, ω = pk_split(x)
    marginal_loglikelihood(pk_inner, θ, σ, ω; approximation = AGHQ).loglikelihood
catch err
    err isa Union{DomainError, PosDefException, NE.EBEConvergenceError} || rethrow()
    -Inf
end
pk_θ0 = θ_SIM .+ [0.3, -0.2, 0.2, -0.3]
pk_σ0 = σ_SIM .+ 0.2
pk_ω0 = omega_parameters(pk_layout, Ω_SIM)

# The wide model: y_ij = θ₁ + η_i1 + (θ₂ + η_i2) t_ij + z_ijᵀ θ[3:end] + ε_ij.
struct Wide
    t::Vector{Vector{Float64}}
    Z::Vector{Matrix{Float64}}
    y::Vector{Vector{Float64}}
end
const WIDE_K = 20
const WIDE_BLOCK = NLMELayout(0, 0, [2])
NE.nsubjects(m::Wide) = length(m.y)
NE.nlme_layout(::Wide) = NLMELayout(2 + WIDE_K, 1, [2])
wide_residuals(m, i, θ, η) =
    m.y[i] .- (θ[1] + η[1]) .- (θ[2] + η[2]) .* m.t[i] .- m.Z[i] * θ[3:end]
function NE.conditional_loglikelihood(m::Wide, i::Integer, θ, σ, η)
    r = wide_residuals(m, i, θ, η)
    return -sum(abs2, r) / (2exp(2σ[1])) - length(r) * (log(2π) / 2 + σ[1])
end
function NE.conditional_loglikelihood_and_population_gradient(m::Wide, i::Integer, θ, σ, η)
    r = wide_residuals(m, i, θ, η)
    v = exp(2σ[1])
    return NE.conditional_loglikelihood(m, i, θ, σ, η),
        vcat(sum(r), dot(r, m.t[i]), m.Z[i]' * r) ./ v, [sum(abs2, r) / v - length(r)]
end
function wide_marginal(m, x)
    θ, σ, Ω = x[1:2+WIDE_K], x[3+WIDE_K], omega_matrix(WIDE_BLOCK, x[4+WIDE_K:end])
    return sum(eachindex(m.y)) do i
        X = hcat(ones(length(m.t[i])), m.t[i])
        L = cholesky(Symmetric(X * Ω * X' + exp(2σ) * I)).L
        z = L \ (m.y[i] .- θ[1] .- θ[2] .* m.t[i] .- m.Z[i] * θ[3:end])
        -(length(z) * log(2π) + 2sum(log, diag(L)) + dot(z, z)) / 2
    end
end
# θ maximizing the marginal at fixed v = [σ; ω] (generalized least squares).
function wide_gls(m, v)
    Ω = omega_matrix(WIDE_BLOCK, v[2:end])
    A, b = zeros(2 + WIDE_K, 2 + WIDE_K), zeros(2 + WIDE_K)
    for i in eachindex(m.y)
        X = hcat(ones(length(m.t[i])), m.t[i])
        D = hcat(X, m.Z[i])
        C = inv(cholesky(Symmetric(X * Ω * X' + exp(2v[1]) * I)))
        A += D' * C * D
        b += D' * C * m.y[i]
    end
    return A \ b
end
wide_rng = Xoshiro(17)
wide_scales = exp10.(range(-2, 2; length = WIDE_K))
wide_β = randn(wide_rng, WIDE_K) ./ wide_scales
wide_t = [sort(3rand(wide_rng, 8)) for _ in 1:30]
wide_Z = [randn(wide_rng, 8, WIDE_K) .* wide_scales' for _ in 1:30]
wide_Ω = [0.5 0.1; 0.1 0.2]
wide_y = [(1.0 .+ 0.5 .* wide_t[i]) .+ hcat(ones(8), wide_t[i]) * (cholesky(wide_Ω).L * randn(wide_rng, 2)) .+
    wide_Z[i] * wide_β .+ 0.3 .* randn(wide_rng, 8) for i in 1:30]
wide_inner = Wide(wide_t, wide_Z, wide_y)
wide = Probe(wide_inner, MuReferencing([1, 2], [Matrix(1.0I, 2, 2) for _ in 1:30]))
wide_score(x) = wide_marginal(wide_inner, x)
wide_profile(v) = try
    wide_score(vcat(wide_gls(wide_inner, v), v))
catch err
    err isa PosDefException || rethrow()
    -Inf
end
wide_v = Optim.minimizer(Optim.optimize(v -> -wide_profile(v),
    vcat(log(0.3), omega_parameters(WIDE_BLOCK, wide_Ω)), Optim.NelderMead(),
    Optim.Options(f_reltol = 1.0e-14, iterations = 50_000)))
wide_mle = vcat(wide_gls(wide_inner, wide_v), wide_v)
wide_se = sqrt.(diag(inv(-ForwardDiff.hessian(wide_score, wide_mle))))

report("SAEM numerical M-step benchmark: $SEEDS seeds, Julia $VERSION, ",
    "$(Threads.nthreads()) threads, NLMEEstimation ",
    readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`),
    isempty(readchomp(`git -C $(@__DIR__) status --porcelain -- ../src`)) ? "" : " (+ uncommitted src)")
report(@sprintf("growth: exact MLE log-likelihood %.6f, standard errors %s", growth_score(growth_mle),
    join([@sprintf("%.3g", s) for s in growth_se], ", ")))
report(@sprintf("wide: exact MLE log-likelihood %.6f (largest gradient entry there %.1e), standard errors %.2g to %.2g",
    wide_score(wide_mle), maximum(abs, ForwardDiff.gradient(wide_score, wide_mle)), extrema(wide_se)...))

# The pk reference: the AGHQ maximum (Nelder–Mead, which needs no gradient and
# takes a point outside the domain as -Inf), started from the best of two long
# fits.
t_ref = @elapsed begin
    starts = [point(first(fit(pk_exponent, pk_θ0, pk_σ0, pk_ω0, :mixture, s; iterations = 400,
        burnin = 150))) for s in 1:2]
    x_start = starts[argmax(pk_score.(starts))]
    ref = Optim.optimize(x -> -pk_score(x), x_start, Optim.NelderMead(),
        Optim.Options(f_reltol = 1.0e-12, iterations = 20_000))
end
pk_ref = Optim.minimizer(ref)
pk_best = pk_score(pk_ref)
θr, σr, ωr = pk_split(pk_ref)
pk_se = covariance_step(pk_inner, θr, σr, ωr; approximation = AGHQ, method = :r).standard_errors
report(@sprintf("pk: AGHQ(3) maximum %.6f (Optim converged=%s, %.0f s), standard errors %s", pk_best,
    Optim.converged(ref), t_ref, join([@sprintf("%.3g", s) for s in pk_se], ", ")))

const MODELS = (
    (name = "growth", model = growth, start = (growth_x0[1:2], growth_x0[3:3], growth_x0[4:end]),
        score = growth_score, best = growth_score(growth_mle), mle = growth_mle, se = growth_se),
    (name = "pk-exponent", model = pk_exponent, start = (pk_θ0, pk_σ0, pk_ω0),
        score = pk_score, best = pk_best, mle = pk_ref, se = pk_se),
    (name = "pk-none", model = pk_none, start = (pk_θ0, pk_σ0, pk_ω0),
        score = pk_score, best = pk_best, mle = pk_ref, se = pk_se),
    (name = "wide", model = wide, start = (zeros(2 + WIDE_K), [0.0],
        omega_parameters(WIDE_BLOCK, Matrix(0.3I, 2, 2))),
        score = wide_score, best = wide_score(wide_mle), mle = wide_mle, se = wide_se))

# Compile every mode once per model.
for m in MODELS, mode in (:mixture, :online, :score)
    fit(m.model, m.start..., mode, 0; iterations = 4, burnin = 2)
end

report("model\tmode\tresolved\tseed\tgap (log-lik units)\tmax |error|/SE\tseconds\t",
    "density evals\tη-gradient evals\tpopulation-gradient evals")
summary = Dict{Tuple{String, Symbol}, Vector{Any}}()
for m in MODELS, mode in (:mixture, :online, :score), seed in 1:SEEDS
    f, seconds, c = fit(m.model, m.start..., mode, seed)
    gap = m.best - m.score(point(f))
    err = maximum(abs.(point(f) - m.mle) ./ m.se)
    push!(get!(summary, (m.name, mode), Any[]), (; gap, err, seconds, c))
    report(@sprintf("%s\t%s\t%s\t%d\t%.4f\t%.2f\t%.2f\t%d\t%d\t%d", m.name, mode,
        f.numeric_mstep, seed, gap, err, seconds, c.values, c.gradients, c.population))
end

# Growth of the per-iteration M-step cost: population-gradient evaluations per
# SA iteration in the windows 2–51, 52–101 and 102–151 of the SA phase (the
# difference of fits stopped at 151 + 50k iterations; a fit stopped after k
# iterations equals the first k iterations of a longer one).
report("\nper-SA-iteration population-gradient (and density) subject evaluations by window")
report("model\tmode\tSA 2–51\tSA 52–101\tSA 102–151")
for m in MODELS, mode in (:mixture, :online, :score)
    totals = [last(fit(m.model, m.start..., mode, 1; iterations = 151 + 50k)) for k in 0:3]
    windows = [(totals[k+1].population - totals[k].population,
        totals[k+1].values - totals[k].values) ./ 50 for k in 1:3]
    report(m.name, "\t", mode, "\t", join([@sprintf("%.1f (%.1f)", w...) for w in windows], "\t"))
end

report("\nmeans over seeds")
report("model\tmode\tmean gap\tmax gap\tmean max|error|/SE\tmean seconds\tmean population evals\tmean density evals")
for m in MODELS, mode in (:mixture, :online, :score)
    rows = summary[(m.name, mode)]
    report(@sprintf("%s\t%s\t%.4f\t%.4f\t%.2f\t%.2f\t%.0f\t%.0f", m.name, mode,
        mean(r.gap for r in rows), maximum(r.gap for r in rows), mean(r.err for r in rows),
        mean(r.seconds for r in rows), mean(r.c.population for r in rows),
        mean(r.c.values for r in rows)))
end

close(OUTPUT)
