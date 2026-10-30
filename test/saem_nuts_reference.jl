using Test
using NLMEEstimation
using Random
using LinearAlgebra
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon

@testset "SAEM NUTS with changing Theoph target" begin
    model = theoph_model()
    ω0 = omega_parameters(nlme_layout(model), Ω_THEOPH)
    fit = saem(model, θ_THEOPH, σ_THEOPH, ω0;
        rng=MersenneTwister(73), subjects=collect(1:6),
        options=SAEMOptions(iterations=6, burnin=2, mcmc_burnin=1,
            mh_rounds=1, numeric_maxiter=2, mstep_interval=3),
        kernel=WarmupHMCNUTS(max_tree_depth=6))
    @test fit.kernel == :warmuphmc_nuts
    @test size(fit.eta) == (3, 6)
    @test all(isfinite, (fit.theta..., fit.sigma..., fit.omega...))
    @test isposdef(Symmetric(fit.omega_matrix))
    @test fit.nuts_divergences >= 0
end
