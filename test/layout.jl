using NLMEEstimation
using LinearAlgebra
using Random
using Test
import ForwardDiff

@testset "layout and Ω parameterization" begin
    @test_throws ArgumentError NLMELayout(-1, 0, [1])
    @test_throws ArgumentError NLMELayout(1, 0, Int[])
    @test_throws ArgumentError NLMELayout(1, 0, [1, 0])

    for (blocks, n, m) in (([1, 1, 1], 3, 3), ([3], 3, 6), ([2, 1], 3, 4))
        layout = NLMELayout(; ntheta = 2, nsigma = 1, eta_blocks = blocks)
        @test neta(layout) == n
        @test nomega(layout) == m
    end

    # A size-one block's coordinate is log(sd).
    diagonal = NLMELayout(0, 0, [1, 1])
    @test omega_matrix(diagonal, log.([0.3, 2.0])) ≈ Diagonal([0.09, 4.0])

    rng = Xoshiro(1)
    layout = NLMELayout(0, 0, [2, 1, 3])
    for _ in 1:5
        ω = randn(rng, nomega(layout))
        Ω = omega_matrix(layout, ω)
        @test issymmetric(Ω)
        @test isposdef(Ω)
        @test omega_parameters(layout, Ω) ≈ ω
        @test Ω[1:2, 3:6] == zeros(2, 4)
        @test Ω[3, 4:6] == zeros(3)
    end
    Ω = omega_matrix(layout, randn(rng, nomega(layout)))
    Ω[1, 4] = Ω[4, 1] = 0.1
    @test_throws ArgumentError omega_parameters(layout, Ω)
    @test_throws ArgumentError omega_parameters(NLMELayout(0, 0, [2]), [1.0 2.0; 2.0 1.0])
    @test_throws DimensionMismatch omega_matrix(layout, zeros(3))
end

@testset "η prior N(0, Ω)" begin
    rng = Xoshiro(2)
    layout = NLMELayout(0, 0, [2, 1])
    ω = randn(rng, nomega(layout))
    Ω = omega_matrix(layout, ω)
    prior = NormalREPrior(layout, ω)
    η = randn(rng, 3)
    expected = -(3 * log(2π) + logdet(Ω) + η' * (Ω \ η)) / 2
    @test re_logprior(prior, η) ≈ expected
    lp, g, H = re_logprior_gradient_and_hessian(prior, η)
    @test lp ≈ expected
    # The closed-form derivatives agree with AD of the log-density.
    @test g ≈ ForwardDiff.gradient(x -> re_logprior(prior, x), η)
    @test H ≈ ForwardDiff.hessian(x -> re_logprior(prior, x), η)
    # The prior is differentiable in ω (FOCEI's outer step needs this).
    gω = ForwardDiff.gradient(w -> re_logprior(NormalREPrior(layout, w), η), ω)
    @test all(isfinite, gω)
end
