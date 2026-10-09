using NLMEEstimation
using LinearAlgebra
using Random
using Test
import DifferentiationInterface as DI
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon

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

_prior_at(η, prior) = re_logprior(prior, η)
_prior_of_ω(ω, layout, η) = re_logprior(NormalREPrior(layout, ω), η)

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
    # The closed-form derivatives are -Ω⁻¹η and -Ω⁻¹ (Ω inverted independently here),
    # and agree with Enzyme reverse mode.
    @test g ≈ -(Ω \ η)
    @test H ≈ -inv(Ω)
    @test g ≈ DI.gradient(_prior_at, REVERSE, η, DI.Constant(prior))
    # The prior is differentiable in ω by reverse-mode AD (population estimators need
    # this); the ω gradient of the size-one-block terms is known in closed form.
    gω = DI.gradient(_prior_of_ω, REVERSE, ω, DI.Constant(layout), DI.Constant(η))
    @test all(isfinite, gω)
    # Block 2 is the single component η[3] with ω[4] = log sd: d/dω of
    # -(ω + (η/e^ω)²/2) is -1 + η² e^(-2ω).
    @test gω[4] ≈ -1 + η[3]^2 * exp(-2ω[4])
end
