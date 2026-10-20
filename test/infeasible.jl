using NLMEEstimation
using Test
using Random
using LinearAlgebra
import NLMEEstimation as NE
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon
using .TestCommon: DoubleWell

# A model that leaves its domain the way a Stan-semantics model does: where the
# residual SD is below `min_sd`, or a subject's first random effect exceeds
# `max_eta` in magnitude, its density throws a `DomainError`, as a matrix
# exponential of an overflowed rate would. `throws = false` is the twin that
# scores the same points -Inf. `hits` counts out-of-domain evaluations by cause
# (residual SD, random effect). Derivatives come from Enzyme through the throw.
# It declares no mu-referencing, so a population update never moves a chain's
# η, and `prior_sd` adds an independent normal population prior (MAP). An inner
# model without σ has only the random-effect bound.
struct Bounded{M}
    inner::M
    min_sigma::Float64
    max_eta::Float64
    throws::Bool
    prior_sd::Union{Nothing, Float64}
    hits::Vector{Int}
end
Bounded(inner; min_sd = 0.1, max_eta = 1.5, throws = true, prior_sd = nothing) =
    Bounded(inner, log(min_sd), max_eta, throws, prior_sd, [0, 0])
NE.nsubjects(m::Bounded) = nsubjects(m.inner)
NE.nlme_layout(m::Bounded) = nlme_layout(m.inner)
NE.ad_backend(::Bounded) = RUNTIME_REVERSE
NE.hessian_ad_backend(::Bounded) = RUNTIME_HESSIAN
NE.population_logprior(m::Bounded, θ, σ, ω) = m.prior_sd === nothing ? nothing :
    -sum(abs2, vcat(θ, σ, ω)) / (2m.prior_sd^2)
function NE.population_logprior_and_gradient(m::Bounded, θ, σ, ω)
    m.prior_sd === nothing && return (0.0, zero(θ), zero(σ), zero(ω))
    v = m.prior_sd^2
    return population_logprior(m, θ, σ, ω), -θ ./ v, -σ ./ v, -ω ./ v
end

function NE.conditional_loglikelihood(m::Bounded, i::Integer, θ, σ, η)
    cause = !isempty(σ) && σ[1] < m.min_sigma ? 1 : abs(η[1]) > m.max_eta ? 2 : 0
    cause == 0 && return conditional_loglikelihood(m.inner, i, θ, σ, η)
    m.hits[cause] += 1
    m.throws && throw(DomainError(cause == 1 ? σ[1] : η[1], "the model leaves its domain"))
    return -Inf
end

const BOUNDED_OPTIONS = (iterations = 30, burnin = 10, mcmc_burnin = 5,
    mh_rounds = 2, numeric_maxiter = 5)

# Fitting the throwing model must retrace the -Inf twin exactly: both reject the
# same M-step trial points and MH proposals and draw the same random numbers.
function bounded_fits(build, θ0, σ0, ω0; seed = 3, kernel = StandardSAEMMH(), options...)
    throwing, twin = build(true), build(false)
    opts = SAEMOptions(; BOUNDED_OPTIONS..., options...)
    fit = saem(throwing, θ0, σ0, ω0; rng = Xoshiro(seed), options = opts, kernel)
    reference = saem(twin, θ0, σ0, ω0; rng = Xoshiro(seed), options = opts, kernel)
    return fit, reference, (twin isa JointView ? twin.inner : twin).hits
end

function same_fit(fit, reference)
    @test fit.theta == reference.theta
    @test fit.sigma == reference.sigma
    @test fit.omega == reference.omega
    @test fit.eta == reference.eta
end

@testset "infeasible trial points: line search" begin
    # The maximum is at 0.5. The first full step lands at 50, where the
    # objective throws, so the search must halve back into its domain.
    calls = Ref(0)
    function objective(x)
        x[1] > 1 && (calls[] += 1; throw(DomainError(x[1], "outside the domain")))
        return -50(x[1] - 0.5)^2, [-100(x[1] - 0.5)]
    end
    x = NE._bfgs_ascent(objective, [0.0], 50, _ -> true)
    @test calls[] > 0
    @test x[1] ≈ 0.5 atol = 1e-8
    calls[] = 0
    screened = NE._bfgs_ascent(objective, [0.0], 50, _ -> true;
        screen = z -> first(objective(z)))
    @test calls[] > 0
    @test screened ≈ x
    # The starting point must evaluate.
    @test_throws DomainError NE._bfgs_ascent(objective, [2.0], 50, _ -> true)
    # Only a domain-type error marks a trial point infeasible; others propagate.
    bug(x) = x[1] > 1 ? throw(ArgumentError("not a domain error")) : objective(x)
    @test_throws ArgumentError NE._bfgs_ascent(bug, [0.0], 50, _ -> true)
end

@testset "infeasible trial points: proposal densities" begin
    model = Bounded(growth_model(n = 4))
    θ, σ = [1.0, 0.5], [log(0.3)]
    H = [0.1 2.0 -0.2 -1.8; 0.0 0.1 0.2 0.0]
    @test_throws DomainError conditional_loglikelihoods(model, 1:4, θ, σ, H)
    ℓ = NE._proposal_loglikelihoods(model, 1:4, θ, σ, H)
    @test ℓ[[2, 4]] == [-Inf, -Inf]
    @test ℓ[[1, 3]] == conditional_loglikelihoods(model, [1, 3], θ, σ, H[:, [1, 3]])
    # A batched model is attributed subject by subject the same way.
    joint = JointView(model)
    @test NE._proposal_loglikelihoods(joint, 1:4, θ, σ, H) == ℓ
end

@testset "infeasible trial points: SAEM retraces the -Inf twin" begin
    θ0, σ0 = [1.0, 0.5], [log(0.5)]
    ω0 = omega_parameters(GROWTH_LAYOUT, [0.8 0.0; 0.0 0.3])
    for mode in (:online, :mixture)
        fit, reference, hits = bounded_fits(
            throws -> Bounded(growth_model(); throws), θ0, σ0, ω0; numeric_mstep = mode)
        @test all(>(0), hits)
        @test fit.objective == :ML
        same_fit(fit, reference)
        @test fit.sigma[1] > log(0.1)
    end

    fit, reference, hits = bounded_fits(
        throws -> Bounded(growth_model(); throws, prior_sd = 2.0), θ0, σ0, ω0)
    @test all(>(0), hits)
    @test fit.objective == :MAP
    same_fit(fit, reference)

    # Batched evaluation: the whole batch throws, inside Enzyme for the M-step.
    fit, reference, hits = bounded_fits(
        throws -> JointView(Bounded(growth_model(); throws)), θ0, σ0, ω0)
    @test all(>(0), hits)
    same_fit(fit, reference)

    fit, reference, hits = bounded_fits(
        throws -> Bounded(growth_model(); throws), θ0, σ0, ω0;
        kernel = DualAveragingMALA(initial_stepsize = 0.5))
    @test hits[2] > 0
    same_fit(fit, reference)
end

@testset "infeasible trial points: ragged SAEM retraces the -Inf twin" begin
    θ0, σ0, ω0 = [2.0], [log(0.5)], [log(0.8), log(0.4)]
    for mode in (:online, :mixture)
        fit, reference, hits = bounded_fits(
            throws -> Bounded(occasions_model(); throws), θ0, σ0, ω0; numeric_mstep = mode)
        @test all(>(0), hits)
        same_fit(fit, reference)
    end
end

# Empirical Bayes on the throwing model must retrace its -Inf twin: both reject
# the same trial points, so every iteration takes the same step.
function bounded_ebes(build, θ, σ, ω; kwargs...)
    throwing, twin = build(true), build(false)
    result = empirical_bayes(throwing, θ, σ, ω; kwargs...)
    reference = empirical_bayes(twin, θ, σ, ω; kwargs...)
    return result, reference, (twin isa JointView ? twin.inner : twin).hits
end

function same_ebes(result, reference)
    @test all(result.converged)
    @test result.modes == reference.modes
    @test result.iterations == reference.iterations
    @test result.complete_data_loglikelihood == reference.complete_data_loglikelihood
end

@testset "infeasible trial points: empirical Bayes" begin
    # Unit-prior double wells with modes near η = 0.7. At the start η = 0.1 the
    # density is not concave, so Newton's first trial goes one prior SD out, to
    # 1.1, past the domain bound 1.0.
    b = [0.05, 0.1, 0.2]
    layouts = (NLMELayout(0, 0, [1]), RaggedNLMELayout(ntheta = 0, nsigma = 0,
        family_sizes = [1], subject_families = [[1], [1], [1]]))
    for layout in layouts
        ω = layout isa NLMELayout ? omega_parameters(layout, [1.0;;]) :
            omega_parameters(layout, [[1.0;;]])
        η0 = layout isa NLMELayout ? fill(0.1, 1, 3) : [[0.1] for _ in 1:3]
        for method in (NewtonEBE(), QuasiNewtonEBE())
            result, reference, hits = bounded_ebes(
                throws -> Bounded(DoubleWell(layout, 1.5, b); max_eta = 1.0, throws),
                Float64[], Float64[], ω; η0, method)
            method isa NewtonEBE && @test hits[2] > 0
            same_ebes(result, reference)
        end
    end

    # The modes lie within |η₁| < 2, but BFGS starts from the prior covariance, so
    # its first steps on informative data overshoot them far past the bound 2.5.
    # Batched: the whole trial batch throws and is re-evaluated subject by subject.
    # The batched view is gradient-only, because Julia 1.13 Enzyme cannot take its
    # Hessian-vector products of this fixture (Enzyme snag julia-1-13-enzym-f89a6f51);
    # trial values need no second order.
    θ, σ = [1.0, 0.5], [log(0.3)]
    ω = omega_parameters(GROWTH_LAYOUT, [0.8 0.0; 0.0 0.3])
    batched(m) = JointView(m; hessian_backend = nothing)
    for (method, wrap) in ((NewtonEBE(), identity), (QuasiNewtonEBE(), identity),
            (QuasiNewtonEBE(), batched))
        result, reference, hits = bounded_ebes(
            throws -> wrap(Bounded(growth_model(); max_eta = 2.5, throws)), θ, σ, ω; method)
        method isa QuasiNewtonEBE && @test hits[2] > 0
        same_ebes(result, reference)
    end

    # A starting point outside the domain remains an error.
    @test_throws DomainError empirical_bayes(Bounded(growth_model(); max_eta = 2.5), θ, σ, ω;
        η0 = fill(3.0, 2, 12))

    # Newton's polishing step is a trial point too. With the bound 0.8, the step
    # from 0.6 lands at 0.83 and is dropped; the step from 0.65 lands at 0.78 and
    # is kept.
    well = Bounded(DoubleWell(NLMELayout(0, 0, [1]), 1.5, [0.05, 0.05]); max_eta = 0.8)
    prior = re_prior(well, omega_parameters(nlme_layout(well), [1.0;;]))
    H = [0.6 0.65]
    ℓ, f, G = NE._complete_gradients(well, [1, 2], Float64[], Float64[], prior, H)
    state = NE._initial_curvature(NewtonEBE(), well, [1, 2], Float64[], Float64[], prior, H, G)
    iterations = [0, 0]
    NE._polish!(NewtonEBE(), well, [1, 2], Float64[], Float64[], prior, H, ℓ, f, G, state,
        iterations, [1, 2])
    @test well.hits[2] > 0
    @test H[1] == 0.6
    @test 0.65 < H[2] < 0.8
    @test iterations == [0, 1]
end

@testset "infeasible trial points: SAEM NUTS retraces the -Inf twin" begin
    # A trajectory that reaches an infeasible position diverges there.
    θ0, σ0 = [1.0, 0.5], [log(0.5)]
    ω0 = omega_parameters(GROWTH_LAYOUT, [0.8 0.0; 0.0 0.3])
    kernel = WarmupHMCNUTS(warmup_draws = 4, warmup_evaluations = 20,
        stepsize_adaptation_limit = 5, max_tree_depth = 4)
    fit, reference, hits = bounded_fits(
        throws -> Bounded(growth_model(); max_eta = 0.6, throws), θ0, σ0, ω0; kernel)
    @test hits[2] > 0
    same_fit(fit, reference)
    @test fit.nuts_divergences == reference.nuts_divergences
end

@testset "infeasible trial points: marginal likelihood" begin
    # Importance-sampling draws outside the domain have zero density. The modes
    # lie within |η₁| < 2; the proposal's t tails reach past 2.2.
    θ, σ = [1.0, 0.5], [log(0.3)]
    ω = omega_parameters(GROWTH_LAYOUT, [0.8 0.0; 0.0 0.3])
    throwing, twin = (Bounded(growth_model(); max_eta = 2.2, throws) for throws in (true, false))
    approximation = ImportanceSampling(draws = 500)
    result = marginal_loglikelihood(throwing, θ, σ, ω; approximation, rng = Xoshiro(1))
    reference = marginal_loglikelihood(twin, θ, σ, ω; approximation, rng = Xoshiro(1))
    @test twin.hits[2] > 0
    @test result.loglikelihood == reference.loglikelihood
end
