using NLMEEstimation
using LinearAlgebra
using Random
using Test
import ForwardDiff
import NLMEEstimation as NE
import Optim
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon: growth_model, growth_exact, occasions_model, occasions_exact_nested,
    occasions_sparse

@testset "Louis's identity: nested occasions against the exact information" begin
    # Eight subjects of 10–60 occasions at the exact maximum. The posterior is
    # Gaussian and the variances are well informed, so 2000 draws give R within
    # 0.4% of the exact Hessian (0.2% declared; seeds measured on gordito) and the
    # standard errors within 1%.
    model = occasions_model(seed = 4, units = [20, 40, 60, 30, 10, 25, 50, 15])
    f(y) = -occasions_exact_nested(model, y)
    x̂ = Optim.minimizer(Optim.optimize(f, y -> ForwardDiff.gradient(f, y),
        [2.0, log(0.25), log(0.6), log(0.4)], Optim.LBFGS(); inplace = false))
    exact_R = -ForwardDiff.hessian(y -> occasions_exact_nested(model, y), x̂)
    exact_se = sqrt.(diag(inv(exact_R)))
    louis = LouisInformation(draws = 2000, burnin = 100)
    for m in (model, occasions_sparse(model))
        r = covariance_step(m, x̂[1:1], x̂[2:2], x̂[3:4]; method = :r, approximation = louis,
            rng = Xoshiro(2))
        @test r.approximation === louis
        @test r.evaluations == 2100
        @test maximum(abs, r.R - exact_R) / maximum(abs, exact_R) < 0.02
        @test r.standard_errors ./ exact_se ≈ ones(4) atol = 0.03
        # Fisher's identity: the subject scores sum to the gradient, close to zero
        # at the maximum.
        @test vec(sum(r.subject_gradients; dims = 2)) ≈ r.gradient
        @test maximum(abs, r.gradient) < 0.1 * maximum(abs, sqrt.(diag(r.S)))
    end
    # The sandwich and S-only forms take the same draws.
    rs = covariance_step(model, x̂[1:1], x̂[2:2], x̂[3:4]; approximation = louis,
        rng = Xoshiro(2))
    s = covariance_step(model, x̂[1:1], x̂[2:2], x̂[3:4]; method = :s, approximation = louis,
        rng = Xoshiro(2))
    @test s.R === nothing
    @test s.S ≈ rs.S
    # refused: Louis's identity gives information matrices, not a likelihood
    @test_throws ArgumentError marginal_loglikelihood(model, x̂[1:1], x̂[2:2], x̂[3:4];
        approximation = louis)
    # refused: the score variance needs at least two draws
    @test_throws ArgumentError LouisInformation(draws = 1) |> a -> NE._check_approximation(a)
end

@testset "Louis's identity: linear growth and a population prior" begin
    # Linear growth at its FOCE(I) estimate (FOCE is exact for it). Twelve subjects
    # leave the slope variance weakly informed, where the expected curvature and
    # the score variance nearly cancel, so its standard error converges slowest:
    # at 10,000 draws the θ and σ errors were within 1.2% and the Ω ones within
    # 13% over two seeds (40,000 draws: 0.5% and 7%).
    model = growth_model()
    fit = focei(model, [0.8, 0.4], [log(0.4)], [log(0.6), 0.0, log(0.5)];
        options = FOCEOptions(maxeval = 2000, rhoend = 1e-9,
            ebe_options = EBEOptions(gtol = 1e-11)))
    x̂ = vcat(fit.theta, fit.sigma, fit.omega)
    exact_se = sqrt.(diag(inv(-ForwardDiff.hessian(z -> growth_exact(model, z), x̂))))
    louis = LouisInformation(draws = 10_000, burnin = 200)
    r = covariance_step(model, fit; method = :r, approximation = louis, rng = Xoshiro(1))
    ratio = r.standard_errors ./ exact_se
    @test all(abs.(ratio[1:3] .- 1) .< 0.05)
    @test all(abs.(ratio .- 1) .< 0.25)
    # A population prior adds exactly its own curvature: the draws do not depend
    # on it, and this one is −‖x‖²/(2·2²).
    prior_model = growth_model(prior_sd = 2.0)
    short = LouisInformation(draws = 200, burnin = 50)
    plain = covariance_step(model, fit.theta, fit.sigma, fit.omega; method = :r,
        approximation = short, rng = Xoshiro(3))
    with_prior = covariance_step(prior_model, fit.theta, fit.sigma, fit.omega; method = :r,
        approximation = short, rng = Xoshiro(3))
    @test with_prior.objective == :MAP
    @test with_prior.R - plain.R ≈ I(6) / 4 atol = 1e-6
    @test with_prior.gradient - plain.gradient ≈ -x̂ / 4 atol = 1e-6
end
