using NLMEEstimation
using Test
using Random
using LinearAlgebra
import NLMEEstimation as NE
using DifferentiationInterface: AutoForwardDiff
using Statistics: mean
import ForwardDiff
import Optim
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon
using .TestCommon: GROWTH_LAYOUT, LinearGrowth

# A batched view of a per-subject model that records the subjects of every
# conditional evaluation.
struct BatchRecorder{M}
    inner::M
    batches::Vector{Vector{Int}}
end
BatchRecorder(inner) = BatchRecorder(inner, Vector{Int}[])
NE.evaluation_style(::BatchRecorder) = BatchedEvaluation()
NE.ad_backend(::BatchRecorder) = AutoForwardDiff()
for f in (:nsubjects, :nlme_layout, :mu_referencing)
    @eval NE.$f(m::BatchRecorder) = NE.$f(m.inner)
end
function NE.conditional_loglikelihoods(m::BatchRecorder, subjects, θ, σ, H)
    push!(m.batches, collect(subjects))
    return [conditional_loglikelihood(m.inner, i, θ, σ, NE._eta_column(H, k))
        for (k, i) in enumerate(subjects)]
end

# η beyond ±1 lies outside the model's domain.
struct DomainBox
    y::Vector{Float64}
end
NE.nsubjects(m::DomainBox) = length(m.y)
NE.nlme_layout(::DomainBox) = NLMELayout(0, 0, [1])
NE.ad_backend(::DomainBox) = AutoForwardDiff()
NE.conditional_loglikelihood(m::DomainBox, i::Integer, θ, σ, η) =
    abs(η[1]) > 1 ? oftype(η[1], -Inf) : -(m.y[i] - η[1])^2 / 2

# `LinearGrowth` with an optional known covariance family and an optional
# independent normal population prior of standard deviation `prior_sd`.
struct GrowthVariant{F}
    inner::LinearGrowth
    fixed::F
    prior_sd::Union{Nothing, Float64}
end
GrowthVariant(inner; fixed = nothing, prior_sd = nothing) = GrowthVariant(inner, fixed, prior_sd)
for f in (:nsubjects, :nlme_layout, :mu_referencing)
    @eval NE.$f(m::GrowthVariant) = NE.$f(m.inner)
end
for f in (:conditional_loglikelihood, :conditional_loglikelihood_and_gradient,
        :conditional_loglikelihood_and_population_gradient)
    @eval NE.$f(m::GrowthVariant, i::Integer, θ, σ, η) = NE.$f(m.inner, i, θ, σ, η)
end
NE.fixed_omega(m::GrowthVariant) = m.fixed
NE.population_logprior(m::GrowthVariant, θ, σ, ω) = m.prior_sd === nothing ? nothing :
    -sum(abs2, vcat(θ, σ, ω)) / (2m.prior_sd^2)
NE.population_logprior_and_gradient(m::GrowthVariant, θ, σ, ω) =
    (population_logprior(m, θ, σ, ω), -θ / m.prior_sd^2, -σ / m.prior_sd^2, -ω / m.prior_sd^2)

with_minibatch(o::SAEMOptions, b) = SAEMOptions(; merge(NamedTuple{fieldnames(SAEMOptions)}(
    getfield.(Ref(o), fieldnames(SAEMOptions))), (; minibatch_size = b))...)

const GROWTH_θ0, GROWTH_σ0 = [1.0, 0.5], [log(0.3)]
const GROWTH_ω0 = omega_parameters(GROWTH_LAYOUT, [0.5 0.1; 0.1 0.2])
const OCCASIONS_θ0, OCCASIONS_σ0, OCCASIONS_ω0 = [2.0], [log(0.25)], [log(0.6), log(0.4)]
const SMALL_NUTS = WarmupHMCNUTS(warmup_draws = 6, warmup_evaluations = 30,
    stepsize_adaptation_limit = 6, max_tree_depth = 5)

@testset "minibatch SAEM options" begin
    model = growth_model()
    n = nsubjects(model)
    options = SAEMOptions(iterations = 30, burnin = 10, mcmc_burnin = 5)
    fit(options; kwargs...) = saem(model, GROWTH_θ0, GROWTH_σ0, GROWTH_ω0; rng = Xoshiro(3),
        options, kwargs...)
    # A batch of every fitted subject is the full-batch algorithm.
    for kernel in (StandardSAEMMH(), DualAveragingMALA(), SMALL_NUTS)
        full, same = fit(options; kernel), fit(with_minibatch(options, n); kernel)
        @test same.theta == full.theta && same.sigma == full.sigma &&
            same.omega == full.omega && same.eta == full.eta
    end
    @test_throws ArgumentError fit(with_minibatch(options, 0))
    @test_throws ArgumentError fit(with_minibatch(options, n + 1))
    # refused: `:mixture` re-evaluates every subject's retained draws at every
    # M-step, the cost a minibatch exists to avoid (todo `15cbo3w`: each iteration
    # evaluates only the batch).
    @test_throws ArgumentError fit(SAEMOptions(iterations = 30, burnin = 10,
        numeric_mstep = :mixture, minibatch_size = 4))
    # refused: `:online` searches each batch's objective to its optimum, whose
    # sampling error is n/b times a full batch's; on 10 of 400 PK subjects it put
    # the additive error SD where its score vanishes (`bench/saem_minibatch.jl`).
    @test_throws ArgumentError fit(SAEMOptions(iterations = 30, burnin = 10,
        numeric_mstep = :online, minibatch_size = 4))
    batched = fit(with_minibatch(options, 4))
    @test batched.numeric_mstep == :minibatch
    @test 0 < batched.acceptance.prior < 1
end

@testset "minibatch SAEM evaluates only its batch" begin
    n, b, iterations = 24, 6, 30
    cases = (
        (growth_model(n = n), GROWTH_θ0, GROWTH_σ0, GROWTH_ω0, StandardSAEMMH()),
        (growth_model(n = n), GROWTH_θ0, GROWTH_σ0, GROWTH_ω0, DualAveragingMALA()),
        (growth_model(n = n), GROWTH_θ0, GROWTH_σ0, GROWTH_ω0, SMALL_NUTS),
        (occasions_model(n = n), OCCASIONS_θ0, OCCASIONS_σ0, OCCASIONS_ω0, StandardSAEMMH()))
    for (inner, θ0, σ0, ω0, kernel) in cases
        model = BatchRecorder(inner)
        saem(model, θ0, σ0, ω0; rng = Xoshiro(1), kernel, options = SAEMOptions(;
            iterations, burnin = 15, mcmc_burnin = 2, minibatch_size = b))
        batches = model.batches
        # The start is checked on every subject once, and the MCMC burn-in
        # sweeps every chain (NUTS warmup and transitions evaluate one subject
        # at a time). Each SA iteration then opens with the domain check of
        # exactly its batch, and every later evaluation of the iteration (kernel
        # proposals or transitions, the numerical step's values and gradients)
        # lies within it. The only exceptions are the numerical step's metric
        # estimates at M-steps 1, 2, 4, 8 and 16, each a run of gradients over
        # every subject.
        @test first(batches) == 1:n
        start = findfirst(s -> length(s) == b, batches)
        @test all(s -> length(s) in (1, n), batches[2:start-1])
        current, changes, contained, full_runs = batches[start], 0, true, 0
        for (j, s) in pairs(batches)
            j < start && continue
            if length(s) == n
                full_runs += length(batches[j-1]) != n
                continue
            end
            if length(s) == b && s != current
                current, changes = s, changes + 1
            end
            contained &= issubset(s, current)
        end
        @test contained
        @test full_runs == count(ispow2, 1:iterations)
        @test changes <= iterations - 1
        @test sort!(unique(reduce(vcat, filter(s -> length(s) < n, batches[start:end])))) == 1:n
    end
    # Without an MCMC burn-in, the first iteration still moves every chain
    # before an M-step estimates Ω from them.
    model = BatchRecorder(growth_model(n = n))
    saem(model, GROWTH_θ0, GROWTH_σ0, GROWTH_ω0; rng = Xoshiro(1), options = SAEMOptions(;
        iterations = 10, burnin = 5, mcmc_burnin = 0, minibatch_size = b))
    @test length(model.batches[2]) == n
    @test findfirst(s -> length(s) == b, model.batches) >
        findlast(s -> length(s) == n, model.batches[1:findfirst(s -> length(s) == b, model.batches)])
end

@testset "minibatch SAEM replaces a drawn chain outside the domain" begin
    model = DomainBox([0.1, -0.2, 0.3, 0.0])
    prior = re_prior(model, [log(0.3)])
    H = [0.2 1.5 -3.0 0.4]
    ℓ = fill(NaN, 4)
    replaced = NE._saem_batch_loglikelihoods!(Xoshiro(1), model, 1:4, Float64[],
        Float64[], prior, H, ℓ, [2, 3, 4])
    @test sort(replaced) == [2, 3]
    @test H[1] == 0.2 && H[4] == 0.4 && all(abs.(H[2:3]) .<= 1)
    @test isnan(ℓ[1])
    @test ℓ[2:4] == conditional_loglikelihoods(model, 2:4, Float64[], Float64[], H[:, 2:4])
    ragged = RaggedNLMELayout(ntheta = 0, nsigma = 0, family_sizes = [1],
        subject_families = [[1], [1, 1]])
    rprior = NE.RaggedNormalREPrior(ragged, [log(0.3)])
    @test NE._saem_prior_draw(Xoshiro(2), rprior, 2) isa Vector{Float64}
    @test length(NE._saem_prior_draw(Xoshiro(2), rprior, 2)) == 2
    # A prior whose draws leave the domain cannot replace the chain.
    @test_throws DomainError NE._saem_batch_loglikelihoods!(Xoshiro(1), model, 1:4,
        Float64[], Float64[], re_prior(model, [log(1e3)]), [5.0 0.0 0.0 0.0],
        zeros(4), [1])
end

@testset "minibatch SAEM score step" begin
    model = growth_model()
    n = nsubjects(model)
    θ, σ = [1.0, 0.5], [log(0.5)]
    H = zeros(2, n)
    design = [zeros(2, 0) for _ in 1:n]
    active = [2, 5, 7]
    batch(z) = NE._saem_numeric_value_gradient(z, model, active, θ, σ, [3], Int[],
        design[active], [H[:, active]], [n / 3], Val(:online))
    full(z) = NE._saem_numeric_value_gradient(z, model, 1:n, θ, σ, [3], Int[], design,
        [H], [1.0], Val(:online))
    step(share, metric) = NE._SAEMScoreStep(share,
        Ref{Union{Nothing, Matrix{Float64}}}(metric), Ref(4))
    score_step(γ, s) = NE._saem_score_step(model, 1:n, θ, σ, [3], Int[], design, [H],
        [1.0], γ, nothing, active, s)
    _, g = batch(σ)
    # A step is min(γ, 4 share) M ĝ, ĝ the batch's score scaled by n/b, M held.
    small = step(0.05, fill(0.02, 1, 1))
    θ1, σ1 = score_step(0.05, small)
    @test θ1 == θ
    @test σ1 ≈ σ .+ 0.05 * 0.02 .* g
    _, σ2 = score_step(1.0, small)
    @test σ2 ≈ σ .+ 0.2 * 0.02 .* g
    @test small.metric[] == fill(0.02, 1, 1)
    # A step that overshoots the batch optimum is halved until it ascends.
    _, σ3 = score_step(1.0, step(0.05, fill(1e4, 1, 1)))
    @test first(batch(σ3)) > first(batch(σ))
    @test abs(σ3[1] - σ[1]) < 0.2 * 1e4 * abs(g[1])
    # M is the full objective's inverse curvature, estimated at M-steps 1, 2,
    # 4, 8, … and held between.
    fresh = NE._SAEMScoreStep(0.25)
    metrics = map(1:9) do _
        fresh.metric[] = fill(7.0, 1, 1)
        score_step(0.5, fresh)
        fresh.metric[]
    end
    @test all(k -> metrics[k] == fill(7.0, 1, 1), filter(!ispow2, 1:9))
    @test all(k -> metrics[k] ≈ NE._saem_search_metric(full, σ), filter(ispow2, 1:9))
end

@testset "minibatch MALA round moves only its batch" begin
    model = DomainBox([0.1, -0.2, 0.3, 0.0])
    prior = re_prior(model, [log(0.3)])
    kernel = DualAveragingMALA()
    states = NE._saem_mala_states(kernel, 4)
    H = fill(0.05, 1, 4)
    cache = NE._saem_mala_cache(H)
    for _ in 1:5
        NE._saem_mala_batch_round!(Xoshiro(2), kernel, states, model, 1:4, Float64[],
            Float64[], prior, H, [2, 4]; metrics = nothing, cache)
    end
    @test H[1] == H[3] == 0.05
    @test cache.valid == [false, true, false, true]
    @test [s.count for s in states] == [0, 5, 0, 5]
    _, f, G = NE._complete_gradients(model, [2, 4], Float64[], Float64[], prior, H[:, [2, 4]])
    @test cache.f[[2, 4]] ≈ f && cache.G[:, [2, 4]] ≈ G
end

# The maximum of `f` and its value.
function exact_maximum(f, x0)
    result = Optim.optimize(x -> -f(x), (G, x) -> (G .= .-ForwardDiff.gradient(f, x)), x0,
        Optim.LBFGS(), Optim.Options(g_tol = 1e-10))
    @assert Optim.converged(result)
    return Optim.minimizer(result), -Optim.minimum(result)
end

@testset "minibatch SAEM reaches the full-batch estimates" begin
    # Linear-Gaussian models have an exact marginal likelihood, so a fit's gap
    # to its maximum measures the fit's Monte Carlo error. With the default
    # options, eight seeded full-batch fits of these models ended 0.01-0.30
    # (MH, NUTS, Hessian-metric MALA) and 0.008-1.01 (ragged MH) log-likelihood
    # units below the maximum. A minibatch of a quarter of the subjects, with the
    # iteration counts scaled by four so that each chain moves as often, ends as
    # close. Before the MCMC burn-in swept every chain, minibatch fits with
    # prior-metric MALA stopped about 12 units below on every seed: the first
    # M-steps shrank Ω from chains still near their start. Prior-metric MALA
    # still ends more than 2 units below in 4 of 20 such minibatch fits (none
    # full-batch): its proposals follow Ω, so it is not checked here.
    n, b = 40, 10
    options = SAEMOptions(iterations = 300n ÷ b, burnin = 150n ÷ b,
        adapt_iterations = 150n ÷ b, minibatch_size = b)
    gaps(f, best, fits) = [best - f(vcat(fit.theta, fit.sigma, fit.omega)) for fit in fits]
    fits(model, θ0, σ0, ω0; kwargs...) =
        [saem(model, θ0, σ0, ω0; rng = Xoshiro(seed), options, kwargs...) for seed in 1:3]

    growth = growth_model(n = n)
    exact(x) = growth_exact(growth, x)
    _, best = exact_maximum(exact, vcat(GROWTH_θ0, GROWTH_σ0, GROWTH_ω0))
    nuts = WarmupHMCNUTS(warmup_draws = 8, warmup_evaluations = 50,
        stepsize_adaptation_limit = 10, max_tree_depth = 6)
    for (kernel, bound) in ((StandardSAEMMH(), 0.5), (DualAveragingMALA(metric = :hessian), 1.0),
            (nuts, 0.5))
        g = gaps(exact, best, fits(growth, GROWTH_θ0, GROWTH_σ0, GROWTH_ω0; kernel))
        @test all(<(bound), g)
    end

    # The maximum a posteriori estimate under a population prior.
    map_model = GrowthVariant(growth; prior_sd = 2.0)
    posterior(x) = exact(x) - sum(abs2, x) / (2 * 2.0^2)
    _, best = exact_maximum(posterior, vcat(GROWTH_θ0, GROWTH_σ0, GROWTH_ω0))
    map_fits = fits(map_model, GROWTH_θ0, GROWTH_σ0, GROWTH_ω0)
    @test all(fit -> fit.objective == :MAP, map_fits)
    @test all(<(0.5), gaps(posterior, best, map_fits))

    # A covariance family whose correlation is known (zero).
    uncorrelated = GrowthVariant(growth;
        fixed = 1 => FixedOmega(correlation = Matrix(1.0I, 2, 2)))
    zero_correlation(x) = exact(vcat(x[1:4], 0.0, x[5]))
    x, best = exact_maximum(zero_correlation, [1.0, 0.5, log(0.3), log(0.7), log(0.45)])
    ω0 = [log(0.7), 0.0, log(0.45)]
    fixed_fits = fits(uncorrelated, GROWTH_θ0, GROWTH_σ0, ω0)
    @test all(fit -> fit.omega[2] == 0, fixed_fits)
    @test all(<(0.5), gaps(exact, best, fixed_fits))

    # A ragged layout: occasion effects share one covariance family.
    occasions = occasions_model(n = n)
    exact_occasions(x) = occasions_exact(occasions, x)
    _, best = exact_maximum(exact_occasions,
        vcat(OCCASIONS_θ0, OCCASIONS_σ0, OCCASIONS_ω0))
    ragged_fits = fits(occasions, OCCASIONS_θ0, OCCASIONS_σ0, OCCASIONS_ω0)
    @test all(fit -> fit isa RaggedSAEMResult, ragged_fits)
    @test all(<(1.0), gaps(exact_occasions, best, ragged_fits))
end
