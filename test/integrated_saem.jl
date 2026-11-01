using NLMEEstimation
using Test
using Random
using LinearAlgebra
import NLMEEstimation as NE
import ForwardDiff
import Optim
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon
using .TestCommon: GROWTH_LAYOUT

# The maximum of `f` and its value.
function integrated_maximum(f, x0)
    result = Optim.optimize(x -> -f(x), (G, x) -> (G .= .-ForwardDiff.gradient(f, x)), x0,
        Optim.LBFGS(), Optim.Options(g_tol = 1e-10))
    @assert Optim.converged(result)
    return Optim.minimizer(result), -Optim.minimum(result)
end

const INTEGRATED_OCCASIONS_START = ([2.0], [log(0.25)], [log(0.6), log(0.4)])
const INTEGRATED_GROWTH_START = ([1.0, 0.5], [log(0.3)],
    omega_parameters(GROWTH_LAYOUT, [0.5 0.1; 0.1 0.2]))

@testset "integrated families: check_protocol" begin
    occasions = occasions_model(n = 6)
    θ, σ, ω = [2.1], [log(0.3)], [log(0.5), log(0.35)]
    η = [0.1 .* randn(Xoshiro(1), NE.neta(NE.nlme_layout(occasions), i)) for i in 1:6]
    for F in ([2], [1], [1, 2])
        @test check_protocol(LinearIntegrated(occasions, F), θ, σ, ω; η)
    end
    growth = growth_model(n = 5)
    θg, σg, ωg = INTEGRATED_GROWTH_START
    @test check_protocol(LinearIntegrated(growth, [1]), θg, σg, ωg)
    # Negative controls: a shifted value, a scaled gradient, an inflated covariance.
    for (kw, what) in (((shift = 1e-3,), "differs from the package's Laplace"),
            ((scale = 1.01,), "central difference"), ((inflate = 1.01,), "covariance block"))
        bad = LinearIntegrated(occasions, [2]; kw...)
        err = try
            check_protocol(bad, θ, σ, ω; η)
            nothing
        catch e
            e
        end
        @test err isa ProtocolViolation && occursin(what, err.message)
    end
    # A model that integrates nothing is not asked for the hooks.
    @test check_protocol(LinearIntegrated(occasions, Int[]), θ, σ, ω; η)
    # refused: an integrated family the layout does not have is not a family to integrate
    @test_throws ArgumentError saem(LinearIntegrated(occasions, [3]), INTEGRATED_OCCASIONS_START...)
    # A model without the hooks names the missing method.
    err = try
        saem(occasions, INTEGRATED_OCCASIONS_START...; integrated = [2],
            options = SAEMOptions(iterations = 2, burnin = 1, mcmc_burnin = 0))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("laplace_conditional_moments", err.msg)
end

@testset "integrated families: Laplace-EM reaches the exact maximum" begin
    # The models are Gaussian in their effects, so the Laplace moments are exact
    # and Laplace-EM is EM: it converges to the maximum-likelihood estimate.
    occasions = occasions_model(n = 30)
    x̂, best = integrated_maximum(x -> occasions_exact_nested(occasions, x),
        vcat(INTEGRATED_OCCASIONS_START...))
    model = LinearIntegrated(occasions, [1, 2])
    fit = saem(model, INTEGRATED_OCCASIONS_START...; rng = Xoshiro(1),
        options = SAEMOptions(iterations = 400))
    @test fit isa RaggedSAEMResult
    @test fit.kernel == :laplace_em && fit.integrated == [1, 2]
    @test fit.iterations == 400 && !fit.interrupted
    x = vcat(fit.theta, fit.sigma, fit.omega)
    @test best - occasions_exact_nested(occasions, x) < 1e-6
    @test maximum(abs, x - x̂) < 1e-3
    # Nothing is sampled: no conditional log-likelihood, no η gradient.
    @test !haskey(model.calls, :gradients)
    # Laplace-EM has no step-size schedule, so the default burnin of 150 does not
    # bound its iterations.
    @test saem(model, INTEGRATED_OCCASIONS_START...;
        options = SAEMOptions(iterations = 20)).iterations == 20
    # eta holds the exact posterior means at the estimate.
    @test all(eachindex(fit.subjects)) do k
        means, _ = occasions_unit_posterior(occasions, fit.subjects[k], x)
        isapprox(fit.eta[k][2:end], means; atol = 1e-8)
    end

    # An equal-size layout with a dense 2 × 2 covariance block: the closed-form
    # step uses the instances' 2 × 2 covariance blocks.
    growth = growth_model(n = 40)
    x̂, best = integrated_maximum(x -> growth_exact(growth, x), vcat(INTEGRATED_GROWTH_START...))
    fit = saem(LinearIntegrated(growth, [1]), INTEGRATED_GROWTH_START...;
        options = SAEMOptions(iterations = 400))
    @test fit isa SAEMResult && fit.kernel == :laplace_em
    @test best - growth_exact(growth, vcat(fit.theta, fit.sigma, fit.omega)) < 1e-6

    # Under a population prior, the maximum a posteriori estimate.
    prior_sd = 2.0
    posterior(x) = growth_exact(growth, x) - sum(abs2, x) / (2prior_sd^2)
    _, best = integrated_maximum(posterior, vcat(INTEGRATED_GROWTH_START...))
    fit = saem(LinearIntegrated(growth_model(n = 40; prior_sd), [1]), INTEGRATED_GROWTH_START...;
        options = SAEMOptions(iterations = 400))
    @test fit.objective == :MAP
    @test best - posterior(vcat(fit.theta, fit.sigma, fit.omega)) < 1e-6
end

@testset "integrated families: SAEM sampling the rest" begin
    # Occasion effects integrated, subject effects sampled: SAEM samples one effect
    # per subject and ends within 0.5 log-likelihood units of the maximum. Plain
    # SAEM sampling every effect is no tighter bound here: seeded full-batch fits
    # of such ragged models end 0.008-1.01 units below (test/saem_minibatch.jl),
    # and one plain fit of this model ended 1.09 below on CI's macOS
    # runner. bench/saem_integrated.jl compares the two at scale.
    occasions = occasions_model(n = 30)
    exact(x) = occasions_exact_nested(occasions, x)
    _, best = integrated_maximum(exact, vcat(INTEGRATED_OCCASIONS_START...))
    gap(fit) = best - exact(vcat(fit.theta, fit.sigma, fit.omega))
    for (kernel, name) in ((StandardSAEMMH(), :standard_mh),
            (WarmupHMCNUTS(max_tree_depth = 6), :warmuphmc_nuts),
            (DualAveragingMALA(), :dual_averaging_mala))
        for seed in 1:3
            model = LinearIntegrated(occasions, [2])
            fit = saem(model, INTEGRATED_OCCASIONS_START...; rng = Xoshiro(seed), kernel)
            @test fit.integrated == [2] && fit.kernel == name
            @test gap(fit) < 0.5
        end
    end
    # Under a minibatch, with the same calibration.
    model = LinearIntegrated(occasions, [2])
    fit = saem(model, INTEGRATED_OCCASIONS_START...; rng = Xoshiro(1),
        options = SAEMOptions(iterations = 1200, burnin = 600, adapt_iterations = 600,
            minibatch_size = 8))
    @test gap(fit) < 0.5
    @test fit.numeric_mstep == :minibatch
    # With the running mixture of every retained draw as the numerical objective.
    fit = saem(LinearIntegrated(occasions, [2]), INTEGRATED_OCCASIONS_START...;
        rng = Xoshiro(1), options = SAEMOptions(numeric_mstep = :mixture))
    @test gap(fit) < 0.5
    @test fit.numeric_mstep == :mixture
end

@testset "integrated families: nothing integrated is the plain fit" begin
    occasions = occasions_model(n = 12)
    options = SAEMOptions(iterations = 60, burnin = 30, mcmc_burnin = 5)
    for kernel in (StandardSAEMMH(), WarmupHMCNUTS(max_tree_depth = 5), DualAveragingMALA())
        a = saem(occasions, INTEGRATED_OCCASIONS_START...; rng = Xoshiro(3), kernel, options)
        b = saem(LinearIntegrated(occasions, Int[]), INTEGRATED_OCCASIONS_START...;
            rng = Xoshiro(3), kernel, options)
        @test a.integrated == b.integrated == Int[]
        @test (a.theta, a.sigma, a.omega, a.eta) == (b.theta, b.sigma, b.omega, b.eta)
    end
end
