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
    occasions_model, occasions_exact, occasions_unit_posterior

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
