using NLMEEstimation
using Test
using Random
using LinearAlgebra
import NLMEEstimation as NE
import Treebars
using Treebars: ProgressInterrupted, render_text, request_interrupt!
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon
using .TestCommon: LinearGrowth, GROWTH_LAYOUT

# A `LinearGrowth` that requests an interrupt on `node` at its `after`-th
# conditional log-density (or log-density and η gradient) evaluation: a
# deterministic Stop button. `calls` counts every such evaluation.
struct StopAfter
    inner::LinearGrowth
    node::Treebars.ProgressNode
    after::Int
    calls::Base.RefValue{Int}
end
StopAfter(inner, node, after) = StopAfter(inner, node, after, Ref(0))

for f in (:nsubjects, :nlme_layout, :observation_model, :provides_eta_hessian, :mu_referencing)
    @eval NE.$f(m::StopAfter) = NE.$f(m.inner)
end
NE.observations(m::StopAfter, i::Integer) = NE.observations(m.inner, i)
for f in (:conditional_loglikelihood_gradient_and_hessian,
        :conditional_loglikelihood_and_population_gradient, :observation_moments,
        :observation_moments_and_jacobians)
    @eval NE.$f(m::StopAfter, i::Integer, θ, σ, η) = NE.$f(m.inner, i, θ, σ, η)
end
for f in (:conditional_loglikelihood, :conditional_loglikelihood_and_gradient)
    @eval function NE.$f(m::StopAfter, i::Integer, θ, σ, η)
        m.calls[] += 1
        m.calls[] == m.after && request_interrupt!(m.node)
        return NE.$f(m.inner, i, θ, σ, η)
    end
end

# The number of evaluations `f(model)` makes, for placing a stop inside it.
function evaluations(f, inner)
    model = StopAfter(inner, Treebars.initialize_progress!(:state; description = "count"), 0)
    f(model)
    return model.calls[]
end

# Run `f(root)` under a fresh `:state` root and return its value and the tree.
function traced(f)
    root = Treebars.initialize_progress!(:state; description = "test")
    value = f(root)
    Treebars.finalize_progress!(root)
    return value, render_text(root)
end

stopped_root() = (root = Treebars.initialize_progress!(:state; description = "stopped");
    request_interrupt!(root); root)

const GROWTH_θ, GROWTH_σ = [1.0, 0.5], [log(0.3)]
const GROWTH_ω = omega_parameters(GROWTH_LAYOUT, [0.5 0.1; 0.1 0.2])
const SAEM_OPTIONS = SAEMOptions(iterations = 30, burnin = 5, mcmc_burnin = 2, mh_rounds = 2,
    adapt_iterations = 10, numeric_maxiter = 3)

same_fit(a, b) = a.theta == b.theta && a.sigma == b.sigma && a.omega == b.omega && a.eta == b.eta

@testset "progress trees" begin
    model = growth_model()
    fit(kernel; kwargs...) = saem(model, GROWTH_θ, GROWTH_σ, GROWTH_ω; rng = Xoshiro(3),
        options = SAEM_OPTIONS, kernel, kwargs...)
    nuts = WarmupHMCNUTS(warmup_draws = 6, warmup_evaluations = 30,
        stepsize_adaptation_limit = 6, max_tree_depth = 5)
    for (kernel, statistic) in ((StandardSAEMMH(), "acceptance"),
            (DualAveragingMALA(), "mean MALA step size"), (nuts, "divergent transitions"))
        plain = fit(kernel)
        reported, tree = traced(root -> fit(kernel; progress = root))
        @test same_fit(reported, plain)
        @test same_fit(fit(kernel; progress = nothing), plain)
        @test reported.iterations == plain.iterations == 30
        @test !reported.interrupted && !plain.interrupted
        @test occursin("✓ SAEM", tree)
        @test occursin("✓ MCMC burn-in (2/2)", tree)
        @test occursin("✓ Stochastic approximation (30/30)", tree)
        @test occursin("γ = ", tree) && occursin(statistic, tree)
        @test occursin("Σ log p(y|η)", tree) == !(kernel isa DualAveragingMALA)
        if kernel isa WarmupHMCNUTS
            @test occursin("✓ NUTS warmup (12/12)", tree)
            @test !occursin("subject", tree)   # each subject's WarmupHMC tree is transient
        end
    end
    @test fit(StandardSAEMMH(); progress = :state).iterations == 30

    occasions = occasions_model()
    ragged(; kwargs...) = saem(occasions, [1.9], [log(0.27)], [log(0.55), log(0.35)];
        rng = Xoshiro(17), options = SAEMOptions(iterations = 10, burnin = 4,
            mcmc_burnin = 0, mh_rounds = 2, numeric_mstep = :online), kwargs...)
    reported, tree = traced(root -> ragged(; progress = root))
    @test same_fit(reported, ragged())
    @test occursin("✓ Stochastic approximation (10/10)", tree)
    @test !occursin("burn-in", tree)

    for (method, label) in ((NewtonEBE(), "Newton"), (QuasiNewtonEBE(), "BFGS"))
        modes(; kwargs...) = empirical_bayes(model, GROWTH_θ, GROWTH_σ, GROWTH_ω; method,
            kwargs...).modes
        reported, tree = traced(root -> modes(; progress = root))
        @test reported == modes()
        @test occursin("✓ Empirical Bayes ($label)", tree)
        @test occursin("12/12 subjects converged", tree)
    end

    options = FOCEOptions(maxeval = 40)
    foce(; kwargs...) = focei(model, GROWTH_θ, GROWTH_σ, GROWTH_ω; options, kwargs...)
    plain = foce()
    reported, tree = traced(root -> foce(; progress = root))
    @test reported.theta == plain.theta && reported.sigma == plain.sigma &&
        reported.omega == plain.omega && reported.ofv == plain.ofv
    @test occursin("FOCE(I) ($(plain.evaluations)/40)", tree) && occursin("best OFV", tree)
    @test !occursin("Empirical Bayes", tree)   # each call's mode search is transient
    evaluation, tree = traced(root -> focei_evaluate(model, GROWTH_θ, GROWTH_σ, GROWTH_ω;
        progress = root))
    @test evaluation.ofv == focei_evaluate(model, GROWTH_θ, GROWTH_σ, GROWTH_ω).ofv
    @test occursin("✓ FOCE(I) objective", tree) && occursin("✓ Empirical Bayes (BFGS)", tree)

    covariance, tree = traced(root -> covariance_step(model, plain; progress = root))
    p = length(plain.free)
    @test covariance.covariance == covariance_step(model, plain).covariance
    @test occursin("✓ Covariance step", tree)
    @test occursin("✓ Marginal likelihoods ($(2p^2 + 1)/$(2p^2 + 1))", tree)
    sampled, tree = traced(root -> covariance_step(model, plain; method = :s,
        approximation = ImportanceSampling(draws = 50), rng = Xoshiro(2), progress = root))
    @test sampled.covariance == covariance_step(model, plain; method = :s,
        approximation = ImportanceSampling(draws = 50), rng = Xoshiro(2)).covariance
    @test occursin("✓ Marginal likelihoods ($(2p + 1)/$(2p + 1))", tree)

    marginal, tree = traced(root -> marginal_loglikelihood(model, plain;
        approximation = ImportanceSampling(draws = 100), rng = Xoshiro(5), progress = root))
    @test marginal.loglikelihood == marginal_loglikelihood(model, plain;
        approximation = ImportanceSampling(draws = 100), rng = Xoshiro(5)).loglikelihood
    @test occursin("✓ Marginal likelihood", tree) && occursin("Empirical Bayes", tree)

    diagnostics, tree = traced(root -> residual_diagnostics(model, plain; replicates = 30,
        rng = Xoshiro(6), progress = root))
    @test diagnostics.npde == residual_diagnostics(model, plain; replicates = 30,
        rng = Xoshiro(6)).npde
    @test occursin("✓ Residual diagnostics", tree)
    @test occursin("✓ Simulated replicates (30/30)", tree)
end

@testset "progress interrupts" begin
    model = growth_model()
    options = SAEMOptions(iterations = 40, burnin = 5, mcmc_burnin = 2, mh_rounds = 2,
        adapt_iterations = 10, numeric_maxiter = 3)

    # SAEM stops at the next iteration boundary with the estimate of the last
    # completed iteration, which is a full run of that many iterations.
    run_saem(m; kwargs...) = saem(m, GROWTH_θ, GROWTH_σ, GROWTH_ω; rng = Xoshiro(3), options,
        kwargs...)
    halfway = evaluations(run_saem, model) ÷ 2
    stopped, tree = traced(root -> run_saem(StopAfter(model, root, halfway); progress = root))
    k = stopped.iterations
    @test stopped.interrupted && options.burnin < k < options.iterations
    shorter = saem(model, GROWTH_θ, GROWTH_σ, GROWTH_ω; rng = Xoshiro(3),
        options = SAEMOptions(iterations = k, burnin = 5, mcmc_burnin = 2, mh_rounds = 2,
            adapt_iterations = 10, numeric_maxiter = 3))
    @test same_fit(stopped, shorter) && stopped.acceptance == shorter.acceptance
    @test !shorter.interrupted
    @test occursin("stopped by interrupt after $k of 40 SA iterations", tree)
    @test occursin("✓ Stochastic approximation ($k/40)", tree)

    # Before the first iteration: the starting point, and no sweep ran.
    start = saem(model, GROWTH_θ, GROWTH_σ, GROWTH_ω; options, progress = stopped_root())
    @test start.interrupted && start.iterations == 0
    @test start.theta == GROWTH_θ && start.omega == GROWTH_ω
    @test isnan(start.acceptance.prior)

    # During the NUTS warmup.
    nuts = WarmupHMCNUTS(warmup_draws = 6, warmup_evaluations = 30,
        stepsize_adaptation_limit = 6, max_tree_depth = 5)
    warm, tree = traced() do root
        saem(StopAfter(model, root, 40), GROWTH_θ, GROWTH_σ, GROWTH_ω; rng = Xoshiro(3),
            options, kernel = nuts, progress = root)
    end
    @test warm.interrupted && warm.iterations == 0 && warm.theta == GROWTH_θ
    @test occursin("⊘ Stochastic approximation", tree)

    # The ragged SAEM.
    occasions = occasions_model()
    ragged = saem(occasions, [1.9], [log(0.27)], [log(0.55), log(0.35)];
        options = SAEMOptions(iterations = 10, burnin = 4), progress = stopped_root())
    @test ragged isa RaggedSAEMResult && ragged.interrupted && ragged.iterations == 0

    # FOCE(I) returns its best point so far, unconverged.
    foce_options = FOCEOptions(maxeval = 200)
    run_focei(m; kwargs...) = focei(m, GROWTH_θ, GROWTH_σ, GROWTH_ω; options = foce_options,
        kwargs...)
    halfway = evaluations(run_focei, model) ÷ 2
    stopped, tree = traced(root -> run_focei(StopAfter(model, root, halfway); progress = root))
    initial = focei_evaluate(model, GROWTH_θ, GROWTH_σ, GROWTH_ω)
    @test !stopped.converged && stopped.optimizer_info.status == :FORCED_STOP
    @test 0 < stopped.evaluations < foce_options.maxeval
    @test stopped.ofv <= initial.ofv
    @test stopped.ofv == focei_evaluate(model, stopped.theta, stopped.sigma, stopped.omega;
        η0 = stopped.ebes).ofv
    @test occursin("stopped by interrupt after $(stopped.evaluations) objective calls", tree)
    @test occursin("FOCE(I) ($(stopped.evaluations)/200)", tree)
    @test_throws ProgressInterrupted focei(model, GROWTH_θ, GROWTH_σ, GROWTH_ω;
        options = foce_options, progress = stopped_root())

    # No partial result: an interrupt raises.
    fit = focei(model, GROWTH_θ, GROWTH_σ, GROWTH_ω; options = FOCEOptions(maxeval = 40))
    @test_throws ProgressInterrupted empirical_bayes(model, GROWTH_θ, GROWTH_σ, GROWTH_ω;
        progress = stopped_root())
    @test_throws ProgressInterrupted focei_evaluate(model, GROWTH_θ, GROWTH_σ, GROWTH_ω;
        progress = stopped_root())
    @test_throws ProgressInterrupted covariance_step(model, fit; progress = stopped_root())
    @test_throws ProgressInterrupted marginal_loglikelihood(model, fit;
        approximation = ImportanceSampling(draws = 50), progress = stopped_root())
    @test_throws ProgressInterrupted residual_diagnostics(model, fit; replicates = 30,
        progress = stopped_root())
end
