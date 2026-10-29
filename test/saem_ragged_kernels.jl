using NLMEEstimation
using LinearAlgebra
using Random
using Statistics
using Test
import ForwardDiff
import NLMEEstimation as NE
import Optim
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon: GROWTH_LAYOUT, LinearGrowth, growth_model, RaggedOccasions,
    occasions_model, occasions_exact, occasions_exact_nested, occasions_unit_posterior,
    occasions_sparse

# The growth model on a ragged layout with one block of its single family per
# subject: the same model, with each subject's η as its own vector.
struct RaggedGrowth
    inner::LinearGrowth
end
NE.nsubjects(m::RaggedGrowth) = NE.nsubjects(m.inner)
NE.nlme_layout(m::RaggedGrowth) =
    RaggedNLMELayout(2, 1, [2], fill([1], NE.nsubjects(m.inner)))
NE.mu_referencing(m::RaggedGrowth) = NE.mu_referencing(m.inner)
NE.provides_eta_hessian(::RaggedGrowth) = true
for f in (:conditional_loglikelihood, :conditional_loglikelihood_and_gradient,
        :conditional_loglikelihood_gradient_and_hessian,
        :conditional_loglikelihood_and_population_gradient)
    @eval NE.$f(m::RaggedGrowth, i::Integer, θ, σ, η) = NE.$f(m.inner, i, θ, σ, η)
end

const SMALL_NUTS = WarmupHMCNUTS(warmup_draws = 6, warmup_evaluations = 30,
    stepsize_adaptation_limit = 6, max_tree_depth = 5)
const OCCASIONS_θ0, OCCASIONS_σ0, OCCASIONS_ω0 = [2.0], [log(0.25)], [log(0.6), log(0.4)]

function exact_maximum(f, x0)
    result = Optim.optimize(x -> -f(x), (G, x) -> (G .= .-ForwardDiff.gradient(f, x)), x0,
        Optim.LBFGS(), Optim.Options(g_tol = 1e-10))
    @assert Optim.converged(result)
    return Optim.minimizer(result), -Optim.minimum(result)
end

@testset "ragged MALA and NUTS: one block per subject reproduces the matrix layout" begin
    # Every kernel works per subject, so a ragged layout whose subjects each
    # carry one block of one family is the matrix layout with its η stored as
    # vectors. The kernels draw the same random numbers on both; the fits agree
    # to rounding.
    growth = growth_model()
    θ0, σ0 = [1.0, 0.5], [log(0.3)]
    ω0 = omega_parameters(GROWTH_LAYOUT, [0.5 0.1; 0.1 0.2])
    options = SAEMOptions(iterations = 20, burnin = 10, mcmc_burnin = 3,
        adapt_iterations = 10)
    for kernel in (DualAveragingMALA(), DualAveragingMALA(metric = :hessian), SMALL_NUTS)
        dense = saem(growth, θ0, σ0, ω0; rng = Xoshiro(3), options, kernel)
        ragged = saem(RaggedGrowth(growth), θ0, σ0, ω0; rng = Xoshiro(3), options, kernel)
        @test ragged isa RaggedSAEMResult
        @test ragged.kernel == dense.kernel
        @test ragged.theta ≈ dense.theta rtol = 1e-8
        @test ragged.sigma ≈ dense.sigma rtol = 1e-8
        @test ragged.omega ≈ dense.omega rtol = 1e-8
        @test ragged.omega_matrix[1] ≈ dense.omega_matrix rtol = 1e-8
        @test reduce(hcat, ragged.eta) ≈ dense.eta rtol = 1e-8
        @test ragged.nuts_divergences == dense.nuts_divergences
        @test ismissing(ragged.rw_scale) && all(ismissing, ragged.acceptance)
        if kernel isa DualAveragingMALA
            @test ragged.mala_stepsizes ≈ dense.mala_stepsizes rtol = 1e-8
            @test ragged.mala_acceptance == dense.mala_acceptance
            @test ragged.mala_adaptation_steps == dense.mala_adaptation_steps
        else
            @test ismissing(ragged.mala_stepsizes)
        end
    end
end

@testset "the nested-occasions oracles" begin
    # The closed-form likelihood and unit posterior that the tests below compare
    # against: the likelihood equals the dense one, and for this linear-Gaussian
    # model the posterior means are the subject modes.
    model = occasions_model()
    x = [2.1, log(0.3), log(0.5), log(0.35)]
    @test occasions_exact_nested(model, x) ≈ occasions_exact(model, x) rtol = 1e-12
    ebes = empirical_bayes(model, x[1:1], x[2:2], x[3:4])
    for (k, i) in enumerate(ebes.subjects)
        @test ebes.modes[k][2:end] ≈ first(occasions_unit_posterior(model, i, x)) atol = 1e-8
    end
end

@testset "ragged MALA and NUTS reach the maximum likelihood" begin
    # 40 subjects of one to three occasions. With the default options, eight
    # seeded fits ended 0.15-0.78 (prior-metric MALA) and 0.005-0.07 (this NUTS)
    # log-likelihood units below the exact maximum; MH ended 0.008-1.01 below.
    occasions = occasions_model(n = 40)
    exact(x) = occasions_exact(occasions, x)
    _, best = exact_maximum(exact, vcat(OCCASIONS_θ0, OCCASIONS_σ0, OCCASIONS_ω0))
    nuts = WarmupHMCNUTS(warmup_draws = 8, warmup_evaluations = 50,
        stepsize_adaptation_limit = 10, max_tree_depth = 6)
    for (kernel, bound) in ((DualAveragingMALA(), 1.0), (nuts, 0.5))
        for seed in 1:3
            fit = saem(occasions, OCCASIONS_θ0, OCCASIONS_σ0, OCCASIONS_ω0;
                rng = Xoshiro(seed), kernel)
            @test fit isa RaggedSAEMResult
            @test best - exact(vcat(fit.theta, fit.sigma, fit.omega)) < bound
        end
    end
end

@testset "ragged NUTS follows subjects with hundreds of nested effects" begin
    # Three subjects of 150, 300 and 600 occasions, so 151 to 601 η each. With
    # 60 iterations NUTS ends at the maximum likelihood (gaps 0.007-0.044 over
    # three seeds) and its unit chains correlate 0.92-0.95 with their exact
    # posterior means, as posterior draws do. MH and prior-metric MALA end 730-850
    # units below with the occasion SD shrunk from 0.41 to 0.05-0.13: random walks
    # move hundreds of coordinates too little, the M-steps shrink Ω from chains
    # still near their start, and proposals that follow Ω shrink with it
    # (`bench/saem_ragged_kernels.jl` measures this at 250-8000 occasions).
    model = occasions_model(seed = 3, units = [150, 300, 600])
    exact(x) = occasions_exact_nested(model, x)
    x̂, best = exact_maximum(exact, vcat(OCCASIONS_θ0, OCCASIONS_σ0, OCCASIONS_ω0))
    nuts = WarmupHMCNUTS(warmup_draws = 8, warmup_evaluations = 50,
        stepsize_adaptation_limit = 10, max_tree_depth = 6)
    for seed in 1:2
        fit = saem(model, OCCASIONS_θ0, OCCASIONS_σ0, OCCASIONS_ω0; rng = Xoshiro(seed),
            kernel = nuts, options = SAEMOptions(iterations = 60, burnin = 30))
        @test length.(fit.eta) == [151, 301, 601]
        @test best - exact(vcat(fit.theta, fit.sigma, fit.omega)) < 0.5
        @test exp(fit.omega[2]) ≈ exp(x̂[4]) rtol = 0.02
        for (k, i) in enumerate(fit.subjects)
            mean_b, _ = occasions_unit_posterior(model, i, x̂)
            @test cor(fit.eta[k][2:end], mean_b) > 0.85
        end
    end
end

@testset "NUTS warmup metric from a declared η Hessian pattern" begin
    # A model that declares `eta_hessian_sparsity` seeds each subject's NUTS
    # metric from its sparse Hessian (the diagonal of the inverse curvature) rather
    # than from dense Hessian blocks, which for a subject with thousands of nested
    # effects would be an n × n matrix. A model that declares nothing and has no
    # Hessian blocks keeps the prior variances.
    inner = occasions_model(seed = 3, units = [150, 300, 600])
    model = occasions_sparse(inner)
    x = [2.0, log(0.25), log(0.6), log(0.4)]
    prior = re_prior(model, x[3:4])
    s2, a2, b2 = exp(2x[2]), exp(2x[3]), exp(2x[4])
    η = zeros(1 + length(inner.y[2]))
    counts = length.(inner.y[2])
    exact_diagonal = vcat(sum(counts) / s2 + 1 / a2, counts ./ s2 .+ 1 / b2)
    metric = NE._saem_hessian_metric(model, 2, x[1:1], x[2:2], prior, η; diagonal = true)
    @test metric ≈ inv.(exact_diagonal) rtol = 1e-8
    @test NE._saem_hessian_metric(inner, 2, x[1:1], x[2:2], prior, η; diagonal = true) ==
        NE._prior_variances(prior, 2)
    # NUTS on the declared model reaches the exact maximum, as on the undeclared one.
    exact(y) = occasions_exact_nested(inner, y)
    x̂, best = exact_maximum(exact, vcat(OCCASIONS_θ0, OCCASIONS_σ0, OCCASIONS_ω0))
    nuts = WarmupHMCNUTS(warmup_draws = 8, warmup_evaluations = 50,
        stepsize_adaptation_limit = 10, max_tree_depth = 6)
    fit = saem(model, OCCASIONS_θ0, OCCASIONS_σ0, OCCASIONS_ω0; rng = Xoshiro(1),
        kernel = nuts, options = SAEMOptions(iterations = 60, burnin = 30))
    @test best - exact(vcat(fit.theta, fit.sigma, fit.omega)) < 0.5
    @test exp(fit.omega[2]) ≈ exp(x̂[4]) rtol = 0.02
end
