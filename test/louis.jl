using NLMEEstimation
using LinearAlgebra
using Random
using Test
import ForwardDiff
import NLMEEstimation as NE
import Optim
using DelimitedFiles
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon: growth_model, growth_exact, occasions_model, occasions_exact_nested,
    occasions_sparse, theoph_model

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
    # A ragged subject's prior score, block by block, equals the one read from its
    # moments by the Gaussian M-step's gradient.
    layout = nlme_layout(model)
    prior = re_prior(model, x̂[3:4])
    indices, design = NE._saem_mu_trait(model, [3], layout)
    η = randn(Xoshiro(5), 1 + length(model.y[3]))
    mean_phi, second_phi = NE._saem_statistics(layout, [3], [η])
    NE._louis_add_moments!(mean_phi, second_phi, NE._saem_phi([η], x̂[1:1], indices, design))
    @test NE._louis_prior_score(layout, x̂[1:1], x̂[2:2], x̂[3:4], prior, nothing, indices,
            design[1], η, 3) ≈
        NE._louis_gaussian_gradient(layout, x̂[1:1], x̂[2:2], x̂[3:4], nothing, indices,
            design, mean_phi, second_phi, [3]) rtol = 1e-12
    # refused: Louis's identity gives information matrices, not a likelihood
    @test_throws ArgumentError marginal_loglikelihood(model, x̂[1:1], x̂[2:2], x̂[3:4];
        approximation = louis)
    # refused: the score variance needs at least two draws
    @test_throws ArgumentError LouisInformation(draws = 1) |> a -> NE._check_approximation(a)
end

@testset "Louis's identity: linear growth and a population prior" begin
    # Linear growth at its FOCE(I) estimate (FOCE is exact for it). Twelve subjects
    # leave the slope variance weakly informed, where the expected curvature and
    # the score variance nearly cancel, so its standard error converges slowest.
    # Over 40 seeds at 40,000 draws the θ and σ errors stayed within 2.8% and the
    # Ω ones within 17% (RMS 1.0% and 5.1% on the worst element of each). At
    # 10,000 draws they reached 11% and 53%, and 2 of the 40 seeds failed these
    # tolerances (3 of 40 on 0.1.x's warmed-up kernel).
    model = growth_model()
    fit = focei(model, [0.8, 0.4], [log(0.4)], [log(0.6), 0.0, log(0.5)];
        options = FOCEOptions(maxeval = 2000, rhoend = 1e-9,
            ebe_options = EBEOptions(gtol = 1e-11)))
    x̂ = vcat(fit.theta, fit.sigma, fit.omega)
    exact_se = sqrt.(diag(inv(-ForwardDiff.hessian(z -> growth_exact(model, z), x̂))))
    louis = LouisInformation(draws = 40_000, burnin = 200)
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

@testset "Louis's identity: nlmixr2 SAEM \"sa\" oracle on theophylline" begin
    # nlmixr2's SAEM estimate and its default covariance "sa" (Louis's identity from
    # a covariance phase at the estimate) for the oracle model of test/covariance.jl.
    # Here Louis's standard errors agree with the observed information of the
    # 3-node quadrature (5,000 draws: within 1%; 1,000: 3%), and nlmixr2's are
    # 2–20% smaller, the same with 500 and 5,000 covariance-phase iterations.
    function data(name)
        values, _ = readdlm(joinpath(@__DIR__, "data", name), ','; header = true)
        return String.(values[:, 1]), Float64.(values[:, 2:end])
    end
    _, estimates = data("nlmixr2_saem_theoph_estimates.csv")
    estimates = vec(estimates)
    model = theoph_model()
    θ, σ = vcat(estimates[1:3], 0.0), [log(estimates[4])]
    ω = log.(sqrt.(estimates[5:7]))
    free = [1, 2, 3, 5, 6, 7, 8]  # the oracle model has no weight effect
    natural(x) = vcat(x[1:3], exp(x[5]), exp.(2 .* x[6:8]))
    order = ["tka", "tcl", "tv", "add.sd", "om.eta.ka", "om.eta.cl", "om.eta.v"]
    function nlmixr2_covariance(n)
        names, covariance = data("nlmixr2_saem_theoph_cov_sa$n.csv")
        k = indexin(order, names)
        return covariance[k, k]
    end
    louis = covariance_step(model, θ, σ, ω; free, method = :r,
        approximation = LouisInformation(draws = 5000, burnin = 200), rng = Xoshiro(1))
    quadrature = covariance_step(model, θ, σ, ω; free, method = :r,
        approximation = AdaptiveGaussHermite(nodes = 3))
    se = delta_method(louis, natural).standard_errors
    @test se ≈ delta_method(quadrature, natural).standard_errors rtol = 0.02
    for n in (500, 5000)
        @test all(r -> 1.0 < r < 1.25, se ./ sqrt.(diag(nlmixr2_covariance(n))))
    end
    # The typical values are mu-referenced, so their complete-data information is
    # exactly N Ω⁻¹ and the information they lose to the unobserved η is the score
    # variance: Louis's gives 2.2, 24.8 and 187 for ka, CL and V, nlmixr2's "sa"
    # about two-thirds of that (1.5, 15.9, 125).
    complete = 12 ./ estimates[5:7]
    J = ForwardDiff.jacobian(y -> natural(vcat(y[1:3], 0.0, y[4:7])), louis.point[free])
    nlmixr2_information = J' * inv(nlmixr2_covariance(5000)) * J
    ratio = (complete - diag(nlmixr2_information)[1:3]) ./ (complete - diag(louis.R)[1:3])
    @test all(r -> 0.6 < r < 0.72, ratio)
end
