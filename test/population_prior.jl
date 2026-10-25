using NLMEEstimation
using Test
using Random
using LinearAlgebra
using ADTypes: AutoEnzyme
using Enzyme
import Optim
import NLMEEstimation as NE

struct ADPopulationPrior end
NE.ad_backend(::ADPopulationPrior) = AutoEnzyme()
NE.population_logprior(::ADPopulationPrior, θ, σ, ω) =
    -sum(abs2, θ) - 2sum(abs2, σ) - 3sum(abs2, ω)

@testset "population prior: default AD gradient" begin
    value, gθ, gσ, gω = population_logprior_and_gradient(
        ADPopulationPrior(), [0.2], [-0.3], [0.4])
    @test value ≈ -0.04 - 0.18 - 0.48
    @test gθ ≈ [-0.4]
    @test gσ ≈ [1.2]
    @test gω ≈ [-2.4]
end

struct PriorGaussianModel
    y::Vector{Vector{Float64}}
    active::Bool
end

NE.nsubjects(m::PriorGaussianModel) = length(m.y)
NE.nlme_layout(::PriorGaussianModel) = NLMELayout(1, 0, [1])
NE.mu_referencing(m::PriorGaussianModel) =
    MuReferencing([1], [ones(1, 1) for _ in m.y])
NE.observation_model(::PriorGaussianModel) = GaussianObservations()
NE.observations(m::PriorGaussianModel, subjects::AbstractVector{<:Integer}) = m.y[subjects]

function NE.conditional_loglikelihood(m::PriorGaussianModel, i, θ, σ, η)
    return sum(y -> -(log(2π) + (y - θ[1] - η[1])^2) / 2, m.y[i])
end

function NE.conditional_loglikelihood_and_gradient(m::PriorGaussianModel, i, θ, σ, η)
    return NE.conditional_loglikelihood(m, i, θ, σ, η),
        [sum(y -> y - θ[1] - η[1], m.y[i])]
end

function NE.conditional_loglikelihood_and_population_gradient(m::PriorGaussianModel,
        i, θ, σ, η)
    return NE.conditional_loglikelihood(m, i, θ, σ, η),
        [sum(y -> y - θ[1] - η[1], m.y[i])], Float64[]
end

function NE.observation_moments_and_jacobians(m::PriorGaussianModel,
        subjects::AbstractVector{<:Integer}, θ, σ, H)
    return [(fill(θ[1] + H[1, k], length(m.y[i])), ones(length(m.y[i])),
        ones(length(m.y[i]), 1), zeros(length(m.y[i]), 1))
        for (k, i) in enumerate(subjects)]
end

# Exponential prior on the natural-scale exp(θ) includes the +θ Jacobian;
# the second factor regularizes the package-owned log standard deviation ω.
function NE.population_logprior(m::PriorGaussianModel, θ, σ, ω)
    m.active || return nothing
    return θ[1] - exp(θ[1]) - (ω[1] - log(0.5))^2 / 0.18
end
function NE.population_logprior_and_gradient(m::PriorGaussianModel, θ, σ, ω)
    m.active || return (0.0, [0.0], Float64[], [0.0])
    return NE.population_logprior(m, θ, σ, ω), [1 - exp(θ[1])], Float64[],
        [-2(ω[1] - log(0.5)) / 0.18]
end
NE.population_prior_provenance(::PriorGaussianModel) = "synthetic transformed prior"

@testset "population prior: FOCE once, likelihood retained" begin
    y = [[-0.2, 0.1], [0.4, 0.7], [1.1, 1.3]]
    ml = PriorGaussianModel(y, false)
    map = PriorGaussianModel(y, true)
    θ, σ, ω = [0.3], Float64[], [log(0.6)]
    plain = focei_evaluate(ml, θ, σ, ω)
    regularized = focei_evaluate(map, θ, σ, ω)
    lp = population_logprior(map, θ, σ, ω)
    @test plain.objective == :ML
    @test plain.logposterior == plain.loglikelihood
    @test regularized.objective == :MAP
    @test regularized.loglikelihood ≈ plain.loglikelihood
    @test regularized.logposterior ≈ plain.loglikelihood + lp
    @test regularized.ofv ≈ plain.ofv - 2lp
    @test regularized.prior_provenance == "synthetic transformed prior"
    @test regularized.ebes.modes ≈ plain.ebes.modes

    ml_fit = focei(ml, θ, σ, ω; free = [1],
        options = FOCEOptions(maxeval = 35, rhobeg = 0.1))
    fit = focei(map, θ, σ, ω; free = [1],
        options = FOCEOptions(maxeval = 35, rhobeg = 0.1))
    @test fit.objective == :MAP
    @test fit.logposterior ≈ fit.loglikelihood + fit.population_logprior
    @test fit.ofv ≈ -2fit.logposterior
    @test fit.prior_provenance == "synthetic transformed prior"
    @test abs(fit.theta[1] - ml_fit.theta[1]) > 1e-3
end

@testset "population prior: SAEM fit records the prior" begin
    model = PriorGaussianModel([[0.0], [1.0]], true)
    θ, σ, ω = [0.2], Float64[], [log(0.6)]
    no_prior = PriorGaussianModel(model.y, false)
    fit = saem(model, θ, σ, ω; rng = MersenneTwister(7),
        options = SAEMOptions(iterations = 4, burnin = 2, mcmc_burnin = 1,
            mh_rounds = 1, numeric_maxiter = 3, numeric_mstep = :online))
    @test fit.objective == :MAP
    @test fit.population_logprior ≈ population_logprior(model, fit.theta,
        fit.sigma, fit.omega)
    @test fit.prior_provenance == "synthetic transformed prior"
    @test size(fit.eta) == (1, 2)
    plain_fit = saem(no_prior, θ, σ, ω; rng = MersenneTwister(7),
        options = SAEMOptions(iterations = 4, burnin = 2, mcmc_burnin = 1,
            mh_rounds = 1, numeric_maxiter = 3, numeric_mstep = :online))
    @test plain_fit.objective == :ML
    @test plain_fit.population_logprior == 0
    @test plain_fit.prior_provenance === nothing
    @test population_logprior_and_gradient(nothing, θ, σ, ω) ==
        (0.0, [0.0], Float64[], [0.0])
end

# Normal observations around a mu-referenced intercept θ[1] plus a slope θ[2]
# with no declared statistic, with log residual SD σ[1]: both blocks of the
# SAEM M-step are non-empty. `prior` is `:none` (ML), `:flat` (a MAP objective
# whose prior is identically zero), `:normal` (independent unit normals) or
# `:strong` (standard deviation 0.2). `gradients` counts population gradient
# calls. The model is linear and Gaussian in η, so `slope_marginal` is exact.
struct PriorSlopeModel
    t::Vector{Vector{Float64}}
    y::Vector{Vector{Float64}}
    prior::Symbol
    gradients::Base.RefValue{Int}
end

function prior_slope_model(prior; n = 30, seed = 3)
    rng = Xoshiro(seed)
    t = [sort(rand(rng, 4)) for _ in 1:n]
    y = [(1.0 + 0.5randn(rng)) .+ 0.5 .* ti .+ 0.3 .* randn(rng, 4) for ti in t]
    return PriorSlopeModel(t, y, prior, Ref(0))
end

NE.nsubjects(m::PriorSlopeModel) = length(m.y)
NE.nlme_layout(::PriorSlopeModel) = NLMELayout(2, 1, [1])
NE.mu_referencing(m::PriorSlopeModel) = MuReferencing([1], [ones(1, 1) for _ in m.y])
slope_residuals(m, i, θ, η) = m.y[i] .- (θ[1] + η[1]) .- θ[2] .* m.t[i]

function NE.conditional_loglikelihood(m::PriorSlopeModel, i, θ, σ, η)
    r = slope_residuals(m, i, θ, η)
    return -sum(abs2, r) / (2exp(2σ[1])) - length(r) * (log(2π) / 2 + σ[1])
end

function NE.conditional_loglikelihood_and_population_gradient(m::PriorSlopeModel,
        i, θ, σ, η)
    m.gradients[] += 1
    r = slope_residuals(m, i, θ, η)
    v = exp(2σ[1])
    return NE.conditional_loglikelihood(m, i, θ, σ, η),
        [sum(r), dot(r, m.t[i])] ./ v, [sum(abs2, r) / v - length(r)]
end

slope_prior_offset(θ, σ, ω) = vcat(θ .- 1, σ, ω)
slope_prior_sd(m) = m.prior === :strong ? 0.2 : 1.0
NE.population_logprior(m::PriorSlopeModel, θ, σ, ω) =
    m.prior === :none ? nothing : m.prior === :flat ? 0.0 :
    -sum(abs2, slope_prior_offset(θ, σ, ω)) / (2slope_prior_sd(m)^2)

function NE.population_logprior_and_gradient(m::PriorSlopeModel, θ, σ, ω)
    m.prior in (:normal, :strong) || return (0.0, zero(θ), zero(σ), zero(ω))
    g = -slope_prior_offset(θ, σ, ω) / slope_prior_sd(m)^2
    return population_logprior(m, θ, σ, ω), g[1:2], g[3:3], g[4:end]
end

function slope_marginal(m::PriorSlopeModel, x)
    θ, σ, ω = x[1:2], x[3], x[4]
    return sum(eachindex(m.y)) do i
        n = length(m.y[i])
        C = cholesky(Symmetric(exp(2ω) * ones(n, n) + exp(2σ) * I))
        z = C.L \ slope_residuals(m, i, θ, [0.0])
        -(n * log(2π) + logdet(C) + dot(z, z)) / 2
    end
end

@testset "population prior: SAEM numerical block adds the prior once" begin
    model = prior_slope_model(:normal)
    plain = prior_slope_model(:none)
    θ, σ, ω = [0.9, 0.4], [log(0.4)], [log(0.6)]
    Φ = reshape(collect(range(0.5, 1.5; length = 30)), 1, 30)
    design = [ones(1, 1) for _ in 1:30]
    free = [2, 3]
    z = [0.45, log(0.35)]
    θz, σz = [θ[1], z[1]], z[2:2]
    lp, gθ, gσ, _ = population_logprior_and_gradient(model, θz, σz, ω)
    for mode in (Val(:online), Val(:mixture))
        value, gradient = NE._saem_numeric_value_gradient(z, model, 1:30, θ, σ,
            free, [1], design, [Φ], [1.0], mode, ω)
        plain_value, plain_gradient = NE._saem_numeric_value_gradient(z, plain,
            1:30, θ, σ, free, [1], design, [Φ], [1.0], mode)
        @test value - plain_value ≈ lp
        @test gradient - plain_gradient ≈ vcat(gθ, gσ)[free]
        # Retained draws add the conditional term per draw, the prior once.
        repeated, repeated_gradient = NE._saem_numeric_value_gradient(z, model,
            1:30, θ, σ, free, [1], design, [Φ, Φ], [0.25, 0.75], mode, ω)
        @test repeated ≈ value
        @test repeated_gradient ≈ gradient
        for j in eachindex(z)
            dz = zeros(length(z))
            dz[j] = 1e-6
            vp = NE._saem_numeric_objective(z + dz, model, 1:30, θ, σ, free, [1],
                design, [Φ], [1.0], ω)
            vm = NE._saem_numeric_objective(z - dz, model, 1:30, θ, σ, free, [1],
                design, [Φ], [1.0], ω)
            @test gradient[j] ≈ (vp - vm) / 2e-6 rtol = 1e-6
        end
    end
end

@testset "population prior: SAEM Gaussian block maximizes without the model" begin
    # `ADPopulationPrior` has no likelihood, so any model evaluation would throw.
    layout = NLMELayout(2, 1, [2])
    model = ADPopulationPrior()
    mean_phi = [0.6 0.3 0.1; -0.2 0.4 0.0]
    second_phi = zeros(2, 2, 3)
    for k in 1:3
        φ = mean_phi[:, k]
        second_phi[:, :, k] = φ * φ' + [0.2 0.03; 0.03 0.1]
    end
    design = [Matrix{Float64}(I, 2, 2) for _ in 1:3]
    θ, Ω = NE._saem_gaussian_mstep(layout, [0.0, 0.0], Matrix(1.0I, 2, 2), [1, 2],
        design, mean_phi, second_phi, 1e-8)
    σ = [-0.3]
    ω = omega_parameters(layout, Ω)
    θmap, ωmap = NE._saem_gaussian_map_block(model, layout, θ, σ, ω, [1, 2], design,
        mean_phi, second_phi, 1:3, 1e-8, nothing)
    posterior(θ, ω) = first(NE._saem_gaussian_map_value_gradient(layout, θ, ω,
        [1, 2], design, mean_phi, second_phi)) + population_logprior(model, θ, σ, ω)
    _, gμ, gω = NE._saem_gaussian_map_value_gradient(layout, θmap, ωmap, [1, 2],
        design, mean_phi, second_phi)
    _, pθ, _, pω = population_logprior_and_gradient(model, θmap, σ, ωmap)
    @test maximum(abs, vcat(gμ + pθ, gω + pω)) < 1e-6
    @test posterior(θmap, ωmap) > posterior(θ, ω)
    @test θmap != θ && ωmap != ω
end

@testset "population prior: the SAEM Gaussian block rejects a singular Ω trial" begin
    # A trial whose second Cholesky diagonal is below the floor (at -800 it
    # underflows to zero) next to a large scale. An eigenvalue floor read from
    # the formed Ω passed some of these, and the Gaussian term then threw
    # LAPACKException out of `saem`.
    layout = NLMELayout(0, 0, [2])
    ragged = RaggedNLMELayout(0, 0, [2], [[1], [1], [1]])
    constrained = NLMELayout(0, 0, [1, 2])
    constraints = NE._omega_constraints(constrained, [1 => FixedOmega(sd = [0.3])])
    admissible(ω) = (NE._omega_admissible(layout, nothing, ω, 1e-8),
        NE._saem_omega_above_floor(ragged, ω, 1e-8),
        NE._omega_admissible(constrained, constraints, vcat(log(0.3), ω), 1e-8))
    for a in range(-2, 12, length = 15), c in (0.0, 1e4, -1e6), b in (-800.0, -40.0)
        @test admissible([a, c, b]) == (false, false, false)
    end
    # Large scales, and a variance near the floor beside one, are above it.
    for ω in ([0.0, 0.5, 0.0], [log(1e3), 0.0, log(1e3)], [log(1e6), 0.0, log(1e-3)])
        @test admissible(ω) == (true, true, true)
    end
    # The line search through the Gaussian term, with a first step onto such a
    # trial: the expected statistics shrink Ω and turn its correlation, so the
    # gradient points the way the step goes in every coordinate.
    Ω0 = [0.2 0.06; 0.06 0.1]
    mean_phi = zeros(2, 3)
    second_phi = repeat([0.002 -0.0005; -0.0005 0.001], 1, 1, 3)
    design = [zeros(2, 0) for _ in 1:3]
    function objective(ω)
        value, _, gradient = NE._saem_gaussian_map_value_gradient(layout, Float64[], ω,
            Int[], design, mean_phi, second_phi)
        return value, gradient
    end
    ω0 = omega_parameters(layout, Ω0)
    _, g0 = objective(ω0)
    for a in range(-3, -1, length = 9), c in (1e5, 1e6)
        step = [a, copysign(c, g0[2]), -800.0] - ω0
        @test all(>(0), step ./ g0)
        ω = NE._bfgs_ascent(objective, ω0, 20,
            ω -> NE._omega_admissible(layout, nothing, ω, 1e-8);
            screen = first ∘ objective, inverse_hessian = Diagonal(step ./ g0))
        @test NE._omega_admissible(layout, nothing, ω, 1e-8)
        @test first(objective(ω)) > first(objective(ω0))
    end
end

@testset "population prior: SAEM with a zero prior retraces the ML fit" begin
    for mode in (:online, :mixture)
        options = SAEMOptions(iterations = 30, burnin = 10, mcmc_burnin = 5,
            numeric_mstep = mode)
        fit(prior) = saem(prior_slope_model(prior), [0.8, 0.3], [log(0.5)], [log(0.4)];
            rng = Xoshiro(4), options)
        ml, map = fit(:none), fit(:flat)
        @test ml.objective == :ML && map.objective == :MAP
        @test map.theta == ml.theta
        @test map.sigma == ml.sigma
        @test map.omega == ml.omega
        @test map.eta == ml.eta
    end
end

@testset "population prior: the SAEM MAP M-step costs what the ML step costs" begin
    for mode in (:online, :mixture)
        options = SAEMOptions(iterations = 40, burnin = 20, mcmc_burnin = 5,
            numeric_mstep = mode)
        ml, map = prior_slope_model(:none), prior_slope_model(:normal)
        saem(ml, [0.8, 0.3], [log(0.5)], [log(0.4)]; rng = Xoshiro(4), options)
        map_fit = saem(map, [0.8, 0.3], [log(0.5)], [log(0.4)]; rng = Xoshiro(4), options)
        @test map_fit.objective == :MAP
        # Population gradients are taken only by the numerical block, which a
        # prior does not enlarge (a joint search took about nine times as many).
        @test map.gradients[] <= 1.25ml.gradients[]
    end
end

@testset "population prior: SAEM finds the exact MAP point" begin
    model = prior_slope_model(:strong)
    x0 = [0.8, 0.3, log(0.5), log(0.4)]
    posterior(x) = slope_marginal(model, x) +
        population_logprior(model, x[1:2], x[3:3], x[4:4])
    exact(f) = Optim.minimizer(Optim.optimize(x -> -f(x), x0, Optim.NelderMead(),
        Optim.Options(g_tol = 1e-12, iterations = 20_000)))
    map_point = exact(posterior)
    ml_point = exact(x -> slope_marginal(model, x))
    # The prior moves the slope and the random-effect SD well beyond SAEM's noise.
    @test abs(map_point[2] - ml_point[2]) > 0.1
    @test abs(map_point[4] - ml_point[4]) > 0.1
    for mode in (:online, :mixture)
        fit = saem(model, x0[1:2], x0[3:3], x0[4:4]; rng = Xoshiro(1),
            options = SAEMOptions(iterations = 300, burnin = 150, mcmc_burnin = 5,
                numeric_mstep = mode))
        @test vcat(fit.theta, fit.sigma, fit.omega) ≈ map_point atol = 0.04
    end
end

@testset "population prior: correlated random-effect M-step gradient" begin
    layout = NLMELayout(2, 0, [2])
    θ = [0.2, -0.1]
    ω = [log(0.7), 0.15, log(0.5)]
    mean_phi = [0.6 0.3; -0.2 0.4]
    second_phi = zeros(2, 2, 2)
    for k in 1:2
        φ = mean_phi[:, k]
        second_phi[:, :, k] = φ * φ' + [0.2 0.03; 0.03 0.1]
    end
    design = [Matrix{Float64}(I, 2, 2) for _ in 1:2]
    value, gθ, gω = NE._saem_gaussian_map_value_gradient(layout, θ, ω,
        [1, 2], design, mean_phi, second_phi)
    @test isfinite(value)
    x = vcat(θ, ω)
    for j in eachindex(x)
        dx = zeros(length(x))
        dx[j] = 1e-6
        f(z) = first(NE._saem_gaussian_map_value_gradient(layout, z[1:2],
            z[3:end], [1, 2], design, mean_phi, second_phi))
        @test vcat(gθ, gω)[j] ≈ (f(x + dx) - f(x - dx)) / 2e-6 atol = 1e-6
    end
end
