using NLMEEstimation
using LinearAlgebra
using Random
using Test
import ForwardDiff
import Optim
using ADTypes: AutoForwardDiff
import NLMEEstimation as NE

# Three Gaussian channels per subject with mu-referenced means θ and unit
# residual variance. η blocks: channel 1 alone, channels 2 and 3 together.
# FOCE(I) is exact for this model, so its fits are maximum-likelihood fits.
struct ChannelGaussian
    y::Vector{Matrix{Float64}}
    fixed::Any
end
ChannelGaussian(y) = ChannelGaussian(y, nothing)
NE.nsubjects(m::ChannelGaussian) = length(m.y)
NE.nlme_layout(::ChannelGaussian) = NLMELayout(3, 0, [1, 2])
NE.fixed_omega(m::ChannelGaussian) = m.fixed
NE.mu_referencing(m::ChannelGaussian) =
    MuReferencing([1, 2, 3], [Matrix{Float64}(I, 3, 3) for _ in m.y])
NE.conditional_loglikelihood(m::ChannelGaussian, i, θ, σ, η) =
    -sum(abs2, m.y[i] .- (θ .+ η)) / 2 - length(m.y[i]) * log(2π) / 2
NE.conditional_loglikelihood_and_gradient(m::ChannelGaussian, i, θ, σ, η) =
    (NE.conditional_loglikelihood(m, i, θ, σ, η), vec(sum(m.y[i] .- (θ .+ η); dims = 2)))
NE.observation_model(::ChannelGaussian) = GaussianObservations()
NE.observations(m::ChannelGaussian, i::Integer) = vec(m.y[i])
NE.observation_moments(m::ChannelGaussian, i::Integer, θ, σ, η) =
    (vec((θ .+ η) .* ones(1, size(m.y[i], 2))), ones(length(m.y[i])))
function NE.observation_moments_and_jacobians(m::ChannelGaussian, i::Integer, θ, σ, η)
    μ, v = NE.observation_moments(m, i, θ, σ, η)
    T = size(m.y[i], 2)
    return μ, v, repeat(Matrix{Float64}(I, 3, 3), T), zeros(3T, 3)
end

function channel_data(; seed = 7, nsubjects = 200, times = 6)
    rng = Xoshiro(seed)
    θ = [0.5, -0.3, 1.0]
    Ω = [0.3^2 0 0; 0 0.5^2 0.6 * 0.5 * 0.4; 0 0.6 * 0.5 * 0.4 0.4^2]
    L = cholesky(Symmetric(Ω)).L
    return [θ .+ L * randn(rng, 3) .+ randn(rng, 3, times) for _ in 1:nsubjects]
end

# Exact marginal log-likelihood of ChannelGaussian: y_i ~ N(θ ⊗ 1, 1 1ᵀ ⊗ Ω + I).
function channel_marginal(y, θ, Ω)
    return sum(y) do Y
        T = size(Y, 2)
        C = kron(ones(T, T), Ω) + I
        r = vec(Y .- θ)
        ch = cholesky(Symmetric(C))
        -(length(r) * log(2π) + logdet(ch) + dot(r, ch \ r)) / 2
    end
end

# The expected Gaussian log density maximized by the covariance M-step.
expected_density(Ω, S̄) = -(logdet(Symmetric(Ω)) + tr(Symmetric(Ω) \ S̄)) / 2

@testset "fixed Ω: standard-deviation/correlation coordinates" begin
    rng = Xoshiro(1)
    for k in 1:4
        segment = randn(rng, k * (k + 1) ÷ 2)
        s, m = NE._scale_coordinates(segment, k)
        @test NE._logcholesky_coordinates(s, m, k) ≈ segment
        Ω = omega_matrix(NLMELayout(0, 0, [k]), segment)
        sd = sqrt.(diag(Ω))
        @test exp.(s) ≈ sd
        @test NE._correlation_from_coordinates(m, k) ≈ Ω ./ (sd * sd')
        @test NE._correlation_coordinates(Ω ./ (sd * sd')) ≈ m
        w = randn(rng, length(segment))
        f(x) = dot(w, NE._logcholesky_coordinates(x[1:k], x[(k + 1):end], k))
        gs, gm = NE._scale_gradient(w, s, m, k)
        @test vcat(gs, gm) ≈ ForwardDiff.gradient(f, vcat(s, m))
    end
end

@testset "fixed Ω: FixedOmega validation" begin
    Ω = [0.1 0.01; 0.01 0.1]
    c = FixedOmega(Ω)
    @test c.sd ≈ sqrt.([0.1, 0.1])
    @test c.correlation ≈ [1 0.1; 0.1 1]
    @test FixedOmega(sd = [0.3, nothing]).correlation === nothing
    @test FixedOmega(sd = [missing, 0.3]).sd == [nothing, 0.3]
    @test FixedOmega(correlation = [1 0.5; 0.5 1]).sd === nothing
    @test_throws ArgumentError FixedOmega()
    @test_throws ArgumentError FixedOmega(sd = [nothing, nothing])
    @test_throws ArgumentError FixedOmega(sd = [-0.1])
    @test_throws ArgumentError FixedOmega([1.0 2.0; 2.0 1.0])
    @test_throws ArgumentError FixedOmega(correlation = [1 0.5; 0.4 1])
    @test_throws ArgumentError FixedOmega(correlation = [2 0.5; 0.5 1])
    @test_throws DimensionMismatch FixedOmega(sd = [0.1, 0.2, 0.3], correlation = [1 0; 0 1])

    layout = NLMELayout(0, 0, [1, 2])
    @test NE._omega_constraints(layout, nothing) === nothing
    @test NE._omega_constraints(layout, Pair{Int, FixedOmega}[]) === nothing
    @test_throws ArgumentError NE._omega_constraints(layout, [3 => FixedOmega(sd = [0.1])])
    @test_throws ArgumentError NE._omega_constraints(layout, [1 => FixedOmega(sd = [0.1]), 1 => FixedOmega(sd = [0.2])])
    @test_throws DimensionMismatch NE._omega_constraints(layout, [1 => FixedOmega(sd = [0.1, 0.2])])
    @test_throws ArgumentError NE._omega_constraints(layout, [1 => FixedOmega(correlation = ones(1, 1))])
    @test_throws ArgumentError NE._omega_constraints(layout, [1 => 0.1])
    single = NE._omega_constraints(layout, 2 => FixedOmega(Ω))
    @test NE._omega_constraints(layout, Dict(2 => FixedOmega(Ω))).families[2].segment == single.families[2].segment
end

@testset "fixed Ω: apply_fixed_omega and check_protocol" begin
    model = ChannelGaussian(channel_data(nsubjects = 4, times = 2),
        [2 => FixedOmega(sd = [0.5, nothing], correlation = [1 0.2; 0.2 1])])
    layout = nlme_layout(model)
    ω = omega_parameters(layout, [0.09 0 0; 0 0.3 0.1; 0 0.1 0.2])
    fixed = apply_fixed_omega(model, ω)
    @test fixed[1] == ω[1]
    Ω = omega_matrix(layout, fixed)
    @test Ω[2, 2] ≈ 0.25
    @test Ω[3, 3] ≈ 0.2
    @test Ω[2, 3] / sqrt(Ω[2, 2] * Ω[3, 3]) ≈ 0.2
    @test apply_fixed_omega(model, fixed) ≈ fixed
    @test apply_fixed_omega(ChannelGaussian(model.y), ω) == ω
    @test check_protocol(model, zeros(3), Float64[], fixed)
    @test_throws ProtocolViolation check_protocol(model, zeros(3), Float64[], ω)
    bad = ChannelGaussian(model.y, [4 => FixedOmega(sd = [0.1])])
    @test_throws ProtocolViolation check_protocol(bad, zeros(3), Float64[], ω)
    @test_throws ArgumentError saem(model, zeros(3), Float64[], ω)
    @test_throws ArgumentError focei(model, zeros(3), Float64[], ω)
end

@testset "fixed Ω: partially fixed family M-step" begin
    rng = Xoshiro(3)
    A = randn(rng, 3, 3)
    S̄ = A * A' / 3 + 0.2I
    floor = 1.0e-8
    family(c, k) = NE._omega_constraints(NLMELayout(0, 0, [k]), [1 => c]).families[1]

    # One fixed standard deviation, the rest of the 2×2 block estimated: the
    # conditional-regression factorization gives the maximizer in closed form.
    S2 = S̄[1:2, 1:2]
    a = 0.4^2
    update = NE._fixed_family_update(family(FixedOmega(sd = [0.4, nothing]), 2), S2, Matrix(1.0I, 2, 2), floor)
    β = S2[2, 1] / S2[1, 1]
    @test update ≈ [a β * a; β * a S2[2, 2] - S2[2, 1]^2 / S2[1, 1] + β^2 * a] rtol = 1e-8

    # A known correlation with estimated standard deviations: u = 1 ./ sd solves
    # u .* ((R⁻¹ .* S̄) u) = 1.
    R = [1 0.3 -0.2; 0.3 1 0.1; -0.2 0.1 1]
    update = NE._fixed_family_update(family(FixedOmega(correlation = R), 3), S̄, Matrix(1.0I, 3, 3), floor)
    sd = sqrt.(diag(update))
    @test update ./ (sd * sd') ≈ R
    u = 1 ./ sd
    @test u .* ((inv(R) .* S̄) * u) ≈ ones(3) rtol = 1e-8
    identity_update = NE._fixed_family_update(family(FixedOmega(correlation = Matrix(1.0I, 3, 3)), 3),
        S̄, Matrix(1.0I, 3, 3), floor)
    @test identity_update ≈ Diagonal(diag(S̄)) rtol = 1e-8

    # Known standard deviations of margins 1 and 3 with an estimated correlation,
    # and a known middle standard deviation: compare with an independent
    # canonical-partial-correlation maximization.
    function cpc_correlation(x)
        z = tanh.(x)
        L = zeros(eltype(x), 3, 3)
        L[1, 1] = 1
        L[2, 1] = z[1]
        L[2, 2] = sqrt(1 - z[1]^2)
        L[3, 1] = z[2]
        L[3, 2] = z[3] * sqrt(1 - z[2]^2)
        L[3, 3] = sqrt((1 - z[2]^2) * (1 - z[3]^2))
        return L * L'
    end
    for (constraint, build, nx) in (
            (FixedOmega(sd = [0.7, nothing, 0.5]),
                x -> (d = [0.7, exp(x[1]), 0.5]; (d * d') .* cpc_correlation(x[2:4])), 4),
            (FixedOmega(sd = [nothing, 0.6, nothing]),
                x -> (d = [exp(x[1]), 0.6, exp(x[2])]; (d * d') .* cpc_correlation(x[3:5])), 5))
        update = NE._fixed_family_update(family(constraint, 3), S̄, Matrix(1.0I, 3, 3), floor)
        oracle = Optim.optimize(x -> -expected_density(build(x), S̄), zeros(nx), Optim.LBFGS(),
            Optim.Options(g_tol = 1e-12); autodiff = AutoForwardDiff())
        @test Optim.converged(oracle)
        @test expected_density(update, S̄) ≥ -Optim.minimum(oracle) - 1e-10
        @test update ≈ build(Optim.minimizer(oracle)) rtol = 1e-6
    end
    # A fully fixed family keeps its value.
    full = family(FixedOmega([0.2 0.05; 0.05 0.3]), 2)
    @test NE._fixed_family_update(full, S2, Matrix(1.0I, 2, 2), floor) == full.covariance
end

@testset "fixed Ω: FOCE(I)" begin
    y = channel_data()
    Ωfix = [0.25 0.06; 0.06 0.16]
    whole = ChannelGaussian(y, [2 => FixedOmega(Ωfix)])
    layout = nlme_layout(whole)
    θ0 = [0.3, 0.0, 0.8]
    ω0 = apply_fixed_omega(whole, omega_parameters(layout, Matrix(0.1I, 3, 3)))
    options = FOCEOptions(maxeval = 400, rhobeg = 0.2, rhoend = 1e-7,
        ebe_options = EBEOptions(gtol = 1e-10))
    fit = focei(whole, θ0, Float64[], ω0; options)
    @test fit.omega[2:4] == ω0[2:4]
    @test fit.omega_matrix[2:3, 2:3] ≈ Ωfix
    # The same search as holding the ω segment with `free`.
    masked = focei(ChannelGaussian(y), θ0, Float64[], ω0; free = 1:4, options)
    @test fit.theta == masked.theta
    @test fit.omega == masked.omega
    @test fit.ofv == masked.ofv
    @test_throws ArgumentError focei(whole, θ0, Float64[], ω0; free = [1, 5])

    # One fixed standard deviation inside the correlated block, against an
    # independent maximization of the exact marginal likelihood.
    partial = ChannelGaussian(y, [2 => FixedOmega(sd = [0.5, nothing])])
    ω0 = apply_fixed_omega(partial, omega_parameters(layout, Matrix(0.1I, 3, 3)))
    fit = focei(partial, θ0, Float64[], ω0; options)
    @test fit.omega_matrix[2, 2] ≈ 0.25 rtol = 1e-12
    @test fit.loglikelihood ≈ channel_marginal(y, fit.theta, fit.omega_matrix) atol = 1e-6
    function build(x)
        sd = [exp(x[4]), 0.5, exp(x[5])]
        ρ = tanh(x[6])
        R = [1 0 0; 0 1 ρ; 0 ρ 1]
        return x[1:3], (sd * sd') .* R
    end
    oracle = Optim.optimize(x -> -channel_marginal(y, build(x)...), vcat(θ0, log(0.3), log(0.3), 0.0),
        Optim.LBFGS(), Optim.Options(g_tol = 1e-10); autodiff = AutoForwardDiff())
    @test Optim.converged(oracle)
    θ̂, Ω̂ = build(Optim.minimizer(oracle))
    @test fit.theta ≈ θ̂ atol = 1e-4
    @test fit.omega_matrix ≈ Ω̂ atol = 1e-4
    @test fit.loglikelihood ≥ -Optim.minimum(oracle) - 1e-6
end

@testset "fixed Ω: SAEM" begin
    y = channel_data()
    layout = NLMELayout(3, 0, [1, 2])
    θ0 = [0.3, 0.0, 0.8]
    options = SAEMOptions(iterations = 300, burnin = 100, mcmc_burnin = 20, mh_rounds = 3)
    foce_options = FOCEOptions(maxeval = 400, rhobeg = 0.2, rhoend = 1e-6)

    Ωfix = [0.25 0.06; 0.06 0.16]
    whole = ChannelGaussian(y, [2 => FixedOmega(Ωfix)])
    ω0 = apply_fixed_omega(whole, omega_parameters(layout, Matrix(0.1I, 3, 3)))
    fit = saem(whole, θ0, Float64[], ω0; rng = Xoshiro(5), options)
    @test fit.omega[2:4] == ω0[2:4]
    @test fit.omega_matrix[2:3, 2:3] ≈ Ωfix
    reference = focei(whole, θ0, Float64[], ω0; options = foce_options)
    @test fit.theta ≈ reference.theta atol = 0.03
    @test fit.omega_matrix[1, 1] ≈ reference.omega_matrix[1, 1] atol = 0.02

    for constraint in (FixedOmega(sd = [0.5, nothing]), FixedOmega(correlation = [1 0.6; 0.6 1]))
        model = ChannelGaussian(y, [2 => constraint])
        ω0 = apply_fixed_omega(model, omega_parameters(layout, Matrix(0.1I, 3, 3)))
        fit = saem(model, θ0, Float64[], ω0; rng = Xoshiro(5), options)
        @test NE._unsatisfied_family(NE._omega_constraints(model), fit.omega) === nothing
        reference = focei(model, θ0, Float64[], ω0; options = foce_options)
        @test fit.theta ≈ reference.theta atol = 0.03
        @test fit.omega_matrix ≈ reference.omega_matrix atol = 0.03
        @test channel_marginal(y, fit.theta, fit.omega_matrix) > reference.loglikelihood - 1
    end

    # The 1×1 family fixed; the correlated block estimated.
    single = ChannelGaussian(y, [1 => FixedOmega(sd = [0.3])])
    ω0 = apply_fixed_omega(single, omega_parameters(layout, Matrix(0.1I, 3, 3)))
    fit = saem(single, θ0, Float64[], ω0; rng = Xoshiro(5), options)
    @test fit.omega[1] == log(0.3)
    @test isposdef(Symmetric(fit.omega_matrix[2:3, 2:3]))
end

# The same model with a population prior on θ and ω, so SAEM takes the joint
# MAP M-step.
struct ChannelGaussianMAP
    inner::ChannelGaussian
end
for f in (:nsubjects, :nlme_layout, :fixed_omega, :mu_referencing)
    @eval NE.$f(m::ChannelGaussianMAP) = NE.$f(m.inner)
end
NE.conditional_loglikelihood(m::ChannelGaussianMAP, i, θ, σ, η) = NE.conditional_loglikelihood(m.inner, i, θ, σ, η)
NE.conditional_loglikelihood_and_gradient(m::ChannelGaussianMAP, i, θ, σ, η) =
    NE.conditional_loglikelihood_and_gradient(m.inner, i, θ, σ, η)
NE.conditional_loglikelihood_and_population_gradient(m::ChannelGaussianMAP, i, θ, σ, η) =
    (NE.conditional_loglikelihood(m.inner, i, θ, σ, η),
        vec(sum(m.inner.y[i] .- (θ .+ η); dims = 2)), Float64[])
NE.population_logprior(::ChannelGaussianMAP, θ, σ, ω) = -sum(abs2, θ) / 2 - sum(abs2, ω) / 2
NE.population_logprior_and_gradient(m::ChannelGaussianMAP, θ, σ, ω) =
    (NE.population_logprior(m, θ, σ, ω), -collect(θ), Float64[], -collect(ω))

@testset "fixed Ω: SAEM MAP M-step" begin
    y = channel_data()
    layout = NLMELayout(3, 0, [1, 2])
    options = SAEMOptions(iterations = 150, burnin = 50, mcmc_burnin = 20, mh_rounds = 2)
    for constraint in (FixedOmega([0.25 0.06; 0.06 0.16]), FixedOmega(sd = [nothing, 0.4]))
        model = ChannelGaussianMAP(ChannelGaussian(y, [2 => constraint]))
        ω0 = apply_fixed_omega(model, omega_parameters(layout, Matrix(0.1I, 3, 3)))
        fit = saem(model, [0.3, 0.0, 0.8], Float64[], ω0; rng = Xoshiro(9), options)
        @test fit.objective == :MAP
        @test NE._unsatisfied_family(NE._omega_constraints(model), fit.omega) === nothing
        constraint.correlation === nothing || @test fit.omega[2:4] == ω0[2:4]
        @test all(isfinite, fit.theta) && isposdef(Symmetric(fit.omega_matrix))
        @test fit.theta ≈ [0.5, -0.3, 1.0] atol = 0.25
    end
end

# Ragged layout: a 1-dimensional subject family and a 2-dimensional family with
# one or two instances per subject, observed directly with unit noise.
struct RaggedFixedFixture
    layout::RaggedNLMELayout
    observed::Vector{Vector{Float64}}
    fixed::Any
    prior::Bool
end
NE.nsubjects(m::RaggedFixedFixture) = length(m.observed)
NE.nlme_layout(m::RaggedFixedFixture) = m.layout
NE.fixed_omega(m::RaggedFixedFixture) = m.fixed
NE.conditional_loglikelihood(m::RaggedFixedFixture, i, θ, σ, η) = -sum(abs2, η - m.observed[i]) / 2
NE.conditional_loglikelihood_and_gradient(m::RaggedFixedFixture, i, θ, σ, η) =
    (NE.conditional_loglikelihood(m, i, θ, σ, η), m.observed[i] - η)
NE.conditional_loglikelihood_and_population_gradient(m::RaggedFixedFixture, i, θ, σ, η) =
    (NE.conditional_loglikelihood(m, i, θ, σ, η), Float64[], Float64[])
NE.population_logprior(m::RaggedFixedFixture, θ, σ, ω) = m.prior ? -sum(abs2, ω) / 2 : nothing
NE.population_logprior_and_gradient(m::RaggedFixedFixture, θ, σ, ω) =
    (-sum(abs2, ω) / 2, Float64[], Float64[], -collect(ω))

@testset "fixed Ω: ragged SAEM" begin
    rng = Xoshiro(13)
    families = [isodd(i) ? [1, 2] : [1, 2, 2] for i in 1:40]
    layout = RaggedNLMELayout(0, 0, [1, 2], families)
    L2 = cholesky([0.5 0.1; 0.1 0.3]).L
    observed = map(families) do fs
        vcat([f == 1 ? 0.6 * randn(rng, 1) : L2 * randn(rng, 2) for f in fs]...) .+ randn(rng, sum(f -> f == 1 ? 1 : 2, fs))
    end
    options = SAEMOptions(iterations = 120, burnin = 40, mcmc_burnin = 10, mh_rounds = 2)
    ωstart = omega_parameters(layout, [fill(0.2, 1, 1), [0.2 0.0; 0.0 0.2]])
    for prior in (false, true)
        model = RaggedFixedFixture(layout, observed,
            [1 => FixedOmega(fill(0.36, 1, 1)), 2 => FixedOmega(sd = [nothing, sqrt(0.3)])], prior)
        ω0 = apply_fixed_omega(model, ωstart)
        @test check_protocol(model, Float64[], Float64[], ω0)
        fit = saem(model, Float64[], Float64[], ω0; rng = Xoshiro(2), options)
        @test fit.objective == (prior ? :MAP : :ML)
        @test fit.omega[1] == ω0[1]
        @test fit.omega_matrix[1] ≈ fill(0.36, 1, 1)
        @test fit.omega_matrix[2][2, 2] ≈ 0.3 rtol = 1e-10
        @test isposdef(Symmetric(fit.omega_matrix[2]))
    end
    # A fully fixed family needs no instances among the selected subjects.
    lone = RaggedNLMELayout(0, 0, [1, 1], [[1], [1], [1, 2]])
    model = RaggedFixedFixture(lone, [[0.1], [-0.2], [0.3, 0.4]], [2 => FixedOmega(fill(0.5, 1, 1))], false)
    fit = saem(model, Float64[], Float64[], apply_fixed_omega(model, zeros(2)); subjects = [1, 2],
        rng = Xoshiro(1), options = SAEMOptions(iterations = 10, burnin = 5, mcmc_burnin = 2))
    @test fit.omega_matrix[2] ≈ fill(0.5, 1, 1)
end

@testset "fixed Ω: covariance step" begin
    y = channel_data()
    layout = NLMELayout(3, 0, [1, 2])
    θ0 = [0.3, 0.0, 0.8]
    options = FOCEOptions(maxeval = 400, rhobeg = 0.2, rhoend = 1e-7,
        ebe_options = EBEOptions(gtol = 1e-10))

    # A fully fixed family: the same step as holding its ω positions with `free`.
    whole = ChannelGaussian(y, [2 => FixedOmega([0.25 0.06; 0.06 0.16])])
    ω0 = apply_fixed_omega(whole, omega_parameters(layout, Matrix(0.1I, 3, 3)))
    fit = focei(whole, θ0, Float64[], ω0; options)
    result = covariance_step(whole, fit)
    masked = covariance_step(ChannelGaussian(y), fit.theta, fit.sigma, fit.omega; free = 1:4,
        η0 = fit.ebes, approximation = FOCEApproximation(interaction = true))
    @test result.coordinates == vcat(fit.theta, fit.omega[1:1])
    @test result.point == vcat(fit.theta, fit.omega)
    @test result.covariance == masked.covariance
    @test result.names == masked.names
    Ωse = delta_method(result, x -> omega_elements(layout, x[4:end])).standard_errors
    @test Ωse[1] > 0 && all(iszero, Ωse[2:4])

    # A partially fixed family: R is the observed information of the exact
    # marginal likelihood in the estimated coordinates.
    for constraint in (FixedOmega(sd = [0.5, nothing]), FixedOmega(correlation = [1 0.6; 0.6 1]))
        model = ChannelGaussian(y, [2 => constraint])
        ω0 = apply_fixed_omega(model, omega_parameters(layout, Matrix(0.1I, 3, 3)))
        fit = focei(model, θ0, Float64[], ω0; options)
        result = covariance_step(model, fit; method = :r)
        c = NE._omega_constraints(model)
        ŷ = vcat(fit.theta, NE._free_omega(c, fit.omega))
        @test result.coordinates ≈ ŷ
        @test result.free == fit.free == collect(eachindex(ŷ))
        @test length(result.names) == length(ŷ) == 6
        loglik(v) = channel_marginal(y, v[1:3], omega_matrix(layout, NE._constrained_omega(c, v[4:end])))
        information = -ForwardDiff.hessian(loglik, ŷ)
        @test result.R ≈ information rtol = 1e-4
        f(x) = omega_elements(layout, x[4:end])
        central = delta_method(result, f)
        exact = delta_method(result, f; jacobian = x -> ForwardDiff.jacobian(f, x))
        J = ForwardDiff.jacobian(v -> f(vcat(v[1:3], NE._constrained_omega(c, v[4:end]))), ŷ)
        @test exact.covariance ≈ J * result.covariance * J' rtol = 1e-10
        @test central.covariance ≈ exact.covariance rtol = 1e-6
        constraint.sd === nothing || @test exact.standard_errors[2] ≈ 0 atol = 1e-14
    end

    # SAEM fits use every estimated coordinate; ω must hold the fixed values.
    model = ChannelGaussian(y, [2 => FixedOmega(sd = [0.5, nothing])])
    ω0 = apply_fixed_omega(model, omega_parameters(layout, Matrix(0.1I, 3, 3)))
    fit = saem(model, θ0, Float64[], ω0; rng = Xoshiro(5),
        options = SAEMOptions(iterations = 60, burnin = 20, mcmc_burnin = 10))
    result = covariance_step(model, fit; method = :s)
    @test result.names == ["θ1", "θ2", "θ3", "ω[η1,η1]", "log sd(η3)", "m(η3,η2)"]
    @test all(isfinite, result.standard_errors)
    @test_throws ArgumentError covariance_step(model, fit.theta, fit.sigma,
        omega_parameters(layout, Matrix(0.1I, 3, 3)))
    @test_throws ArgumentError covariance_step(model, fit.theta, fit.sigma, fit.omega; free = [7])
end

@testset "fixed Ω: marginal likelihood" begin
    y = channel_data(nsubjects = 40)
    layout = NLMELayout(3, 0, [1, 2])
    θ = [0.45, -0.25, 0.95]
    Ωfix = [0.25 0.06; 0.06 0.16]
    model = ChannelGaussian(y, [2 => FixedOmega(Ωfix)])
    ω = apply_fixed_omega(model, omega_parameters(layout, Matrix(0.09I, 3, 3)))
    Ω = omega_matrix(layout, ω)
    result = marginal_loglikelihood(model, θ, Float64[], ω;
        approximation = AdaptiveGaussHermite(nodes = 1))
    # Gaussian channels: Laplace is exact.
    @test result.loglikelihood ≈ channel_marginal(y, θ, Ω) atol = 1e-8
    # Three means and the free family's log SD; the fixed block is not counted.
    @test result.nparameters == 4
    @test result.aic ≈ -2 * result.loglikelihood + 8
    free = marginal_loglikelihood(ChannelGaussian(y), θ, Float64[], ω;
        approximation = AdaptiveGaussHermite(nodes = 1))
    @test free.nparameters == 7  # 3 means + 1 + 3 log-Cholesky coordinates
    @test free.loglikelihood ≈ result.loglikelihood
end
