using Test
using NLMEEstimation
using Random
using LinearAlgebra
import NLMEEstimation as NE
import LogDensityProblems as LDP

struct NUTSQuadratic
    y::Vector{Float64}
end
NE.nsubjects(m::NUTSQuadratic) = length(m.y)
NE.nlme_layout(::NUTSQuadratic) = NLMELayout(0, 0, [1])
NE.provides_eta_hessian(::NUTSQuadratic) = true
NE.conditional_loglikelihood(m::NUTSQuadratic, i, θ, σ, η) =
    -(m.y[i] - η[1])^2 / 2
function NE.conditional_loglikelihood_and_gradient(m::NUTSQuadratic, i, θ, σ, η)
    return NE.conditional_loglikelihood(m, i, θ, σ, η), [m.y[i] - η[1]]
end
function NE.conditional_loglikelihood_gradient_and_hessian(m::NUTSQuadratic,
        i, θ, σ, η)
    ℓ, g = NE.conditional_loglikelihood_and_gradient(m, i, θ, σ, η)
    return ℓ, g, fill(-1.0, 1, 1)
end

@testset "SAEM WarmupHMC NUTS kernel" begin
    model = NUTSQuadratic([-0.6, 0.6])
    prior = re_prior(model, [log(0.7)])
    target = NE._saem_target(model, 1, Float64[], Float64[], prior)
    @test LDP.dimension(target) == 1
    @test LDP.capabilities(typeof(target)) == LDP.LogDensityOrder{1}()
    q = [0.2]
    lp, g = LDP.logdensity_and_gradient(target, q)
    @test lp ≈ LDP.logdensity(target, q)
    @test g[1] ≈ model.y[1] - q[1] - q[1] / 0.7^2
    @test NE._saem_hessian_metric(model, 1, Float64[], Float64[], prior, q)[1, 1] ≈
        1 / (1 + 1 / 0.7^2)

    fit = saem(model, Float64[], Float64[], [log(0.7)]; rng=MersenneTwister(42),
        options=SAEMOptions(iterations=3, burnin=1, mcmc_burnin=0,
            mh_rounds=1, adapt_iterations=0),
        kernel=WarmupHMCNUTS(max_tree_depth=4))
    @test fit.kernel == :warmuphmc_nuts
    @test size(fit.eta) == (1, 2)
    @test all(isfinite, fit.eta)
    @test fit.nuts_divergences >= 0
    @test isposdef(Symmetric(fit.omega_matrix))
    @test ismissing(fit.acceptance.prior)
    @test ismissing(fit.rw_scale)
    @test_throws ArgumentError saem(model, Float64[], Float64[], [log(0.7)];
        options=SAEMOptions(iterations=2, burnin=0),
        kernel=WarmupHMCNUTS(max_tree_depth=0))
    @test_throws ArgumentError saem(model, Float64[], Float64[], [log(0.7)];
        options=SAEMOptions(iterations=2, burnin=0),
        kernel=WarmupHMCNUTS(target_acceptance_rate=1.0))
end
