using NLMEEstimation
using NLMEEstimation.ReferenceModels
using LinearAlgebra
using Random
using Test
using DelimitedFiles
import ForwardDiff
import NLMEEstimation as NE
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon

@testset "covariance step: exact linear-Gaussian R and S" begin
    model = growth_model()
    @test check_protocol(model, [1.0, 0.5], [log(0.3)], [log(0.7), 0.1, log(0.45)])
    fit = focei(model, [0.8, 0.4], [log(0.4)], [log(0.6), 0.0, log(0.5)];
        options = FOCEOptions(maxeval = 2000, rhoend = 1e-9,
            ebe_options = EBEOptions(gtol = 1e-11)))
    @test fit.free == 1:6
    x̂ = vcat(fit.theta, fit.sigma, fit.omega)
    exact_R = -ForwardDiff.hessian(x -> growth_exact(model, x), x̂)
    exact_G = reduce(hcat, [ForwardDiff.gradient(x -> growth_exact_subject(model, i, x), x̂)
        for i in 1:NE.nsubjects(model)])
    exact_S = exact_G * exact_G'

    result = covariance_step(model, fit)
    @test result isa CovarianceResult
    @test result.method == :rs && result.objective == :ML
    @test result.approximation == FOCEApproximation(interaction = true)
    @test result.subjects == 1:NE.nsubjects(model)
    @test result.point == x̂
    @test result.free == 1:6
    @test result.names == ["θ1", "θ2", "σ1", "ω[η1,η1]", "ω[η2,η1]", "ω[η2,η2]"]
    # Central differences: second differences are accurate to O(h²) relative;
    # the subject gradients carry an O(h²) third-derivative error of order 1e-5.
    @test result.R ≈ exact_R rtol = 1e-6
    @test result.subject_gradients ≈ exact_G rtol = 1e-4 atol = 1e-6
    @test result.S ≈ exact_S rtol = 1e-4
    @test result.gradient ≈ vec(sum(exact_G; dims = 2)) atol = 1e-4
    @test maximum(abs, result.gradient) < 1e-3
    @test result.covariance ≈ inv(exact_R) * exact_S * inv(exact_R) rtol = 1e-4
    @test result.standard_errors ≈ sqrt.(diag(result.covariance))
    @test result.evaluations == 2 * 6^2 + 1
    @test result.step ≈ 1e-3 .* max.(1, abs.(x̂))

    r = covariance_step(model, fit; method = :r)
    @test r.covariance ≈ inv(exact_R) rtol = 1e-5
    @test r.R == result.R
    s = covariance_step(model, fit; method = :s)
    @test s.R === nothing
    @test s.evaluations == 2 * 6 + 1
    @test s.covariance ≈ inv(exact_S) rtol = 1e-4

    # BFGS modes need no η Hessians; their finite tolerance makes R noisier.
    quasi = covariance_step(model, fit; method = :r,
        approximation = FOCEApproximation(ebe_method = QuasiNewtonEBE()))
    @test quasi.R ≈ exact_R rtol = 1e-4

    # Fixed coordinates: the step covers only the estimated block.
    subset = covariance_step(model, fit.theta, fit.sigma, fit.omega;
        free = [1, 3, 6], method = :r, η0 = fit.ebes)
    @test subset.names == ["θ1", "σ1", "ω[η2,η2]"]
    @test subset.R ≈ exact_R[[1, 3, 6], [1, 3, 6]] rtol = 1e-6

    # Natural units: the residual SD and the Ω entries.
    f(x) = vcat(exp(x[3]), omega_elements(GROWTH_LAYOUT, x[4:6]))
    J = ForwardDiff.jacobian(f, x̂)
    natural = delta_method(result, f)
    @test natural.estimate ≈ vcat(exp(fit.sigma[1]), fit.omega_matrix[1, 1],
        fit.omega_matrix[2, 1], fit.omega_matrix[2, 2])
    @test natural.covariance ≈ J * result.covariance * J' rtol = 1e-7
    @test natural.standard_errors ≈ sqrt.(diag(J * result.covariance * J')) rtol = 1e-7
    @test delta_method(result, f; jacobian = x -> ForwardDiff.jacobian(f, x)).covariance ≈
        J * result.covariance * J' rtol = 1e-12

    # S has rank at most three for three subjects, so S⁻¹ does not exist for six
    # coordinates.
    @test_throws CovarianceStepError covariance_step(model, fit.theta, fit.sigma,
        fit.omega; subjects = 1:3, method = :s)
end

@testset "covariance step: SAEM estimates and population prior" begin
    # Forty subjects: with the default twelve the likelihood is flat in the
    # log-Cholesky ω[η2,η2] (it falls 0.25 from -1.6 to -9), so whether a short
    # SAEM run ends at a strict local optimum, R positive definite, was seed luck.
    ml = growth_model(n = 40)
    fit = saem(ml, [0.9, 0.45], [log(0.35)], [log(0.65), 0.05, log(0.5)];
        rng = Xoshiro(3), options = SAEMOptions(iterations = 80, burnin = 40, mcmc_burnin = 5))
    from_fit = covariance_step(ml, fit)
    at_point = covariance_step(ml, fit.theta, fit.sigma, fit.omega; η0 = fit.eta)
    @test from_fit.covariance == at_point.covariance
    @test from_fit.free == 1:6
    x̂ = vcat(fit.theta, fit.sigma, fit.omega)
    @test from_fit.R ≈ -ForwardDiff.hessian(z -> growth_exact(ml, z), x̂) rtol = 1e-6

    # With a population prior the objective is the log posterior: R gains the
    # prior's curvature, and the gradient its gradient.
    model = growth_model(n = 40, prior_sd = 2.0)
    result = covariance_step(model, fit.theta, fit.sigma, fit.omega; method = :r)
    @test result.objective == :MAP
    exact_R = -ForwardDiff.hessian(z -> growth_exact(model, z), x̂) + I / 2.0^2
    @test result.R ≈ exact_R rtol = 1e-6
    @test result.covariance ≈ inv(exact_R) rtol = 1e-5
    exact_gradient = ForwardDiff.gradient(z -> growth_exact(model, z), x̂) .- x̂ ./ 2.0^2
    @test result.gradient ≈ exact_gradient atol = 1e-4
    @test result.S ≈ from_fit.S
end

@testset "covariance step: the model's own Laplace gradient" begin
    # Laplace is exact for this linear-Gaussian model, so the hook's closed-form
    # marginals are the package's Laplace approximation, and R and S are exact up
    # to the error of central differences.
    inner = growth_model(n = 40)
    n = NE.nsubjects(inner)
    model = LaplaceHooked(inner, growth_exact_subject)
    θ, σ = [1.0, 0.5], [log(0.3)]
    ω = omega_parameters(GROWTH_LAYOUT, [0.5 0.1; 0.1 0.2])
    x̂ = vcat(θ, σ, ω)
    @test check_protocol(model, θ, σ, ω)
    @test_throws ProtocolViolation check_protocol(
        LaplaceHooked(inner, growth_exact_subject; shift = 1e-3), θ, σ, ω)
    @test_throws ProtocolViolation check_protocol(
        LaplaceHooked(inner, growth_exact_subject; scale = 1.001), θ, σ, ω)

    exact_R = -ForwardDiff.hessian(x -> growth_exact(inner, x), x̂)
    exact_G = reduce(hcat, [ForwardDiff.gradient(x -> growth_exact_subject(inner, i, x), x̂)
        for i in 1:n])
    laplace = AdaptiveGaussHermite(nodes = 1)
    empty!(model.starts)
    result = covariance_step(model, θ, σ, ω; approximation = laplace)
    @test result.evaluations == 2 * 6 + 1 == length(model.starts)
    # Every call starts the mode search from the same batch: zeros without η0.
    @test all(==(zeros(2, n)), model.starts)
    @test result.R ≈ exact_R rtol = 1e-6
    @test result.subject_gradients ≈ exact_G rtol = 1e-4 atol = 1e-6
    @test result.gradient ≈ vec(sum(exact_G; dims = 2)) rtol = 1e-10
    @test result.covariance ≈ inv(exact_R) * exact_G * exact_G' * inv(exact_R) rtol = 1e-4
    @test covariance_step(model, θ, σ, ω; approximation = laplace, method = :r).evaluations ==
        2 * 6 + 1

    # Without the hook, R takes second differences of 2p² + 1 Laplace marginals.
    values = covariance_step(LaplaceHooked(inner, growth_exact_subject; hook = false), θ, σ, ω;
        approximation = laplace)
    @test values.evaluations == 2 * 6^2 + 1
    @test values.R ≈ result.R rtol = 1e-5
    @test values.S ≈ result.S rtol = 1e-6
    # Other approximations never call it.
    calls = length(model.starts)
    covariance_step(model, θ, σ, ω; approximation = AdaptiveGaussHermite(nodes = 2), method = :s)
    @test length(model.starts) == calls

    # A subset of the coordinates, from a given starting batch.
    H = randn(Xoshiro(1), 2, n)
    subset = covariance_step(model, θ, σ, ω; approximation = laplace, free = [1, 3, 6],
        method = :r, η0 = H)
    @test subset.evaluations == 2 * 3 + 1
    @test model.starts[end] == H
    @test subset.R ≈ exact_R[[1, 3, 6], [1, 3, 6]] rtol = 1e-6

    # A population prior, which the hook leaves out, enters R and the gradient.
    prior = LaplaceHooked(growth_model(n = 40, prior_sd = 2.0), growth_exact_subject)
    posterior = covariance_step(prior, θ, σ, ω; approximation = laplace, method = :r)
    @test posterior.objective == :MAP
    @test posterior.R ≈ exact_R + I / 2.0^2 rtol = 1e-6
    @test posterior.gradient ≈ vec(sum(exact_G; dims = 2)) .- x̂ ./ 2.0^2 rtol = 1e-10

    # A partially fixed Ω family: the hook's ω gradient is chained into the
    # estimated coordinates.
    for constraint in (FixedOmega(sd = [sqrt(0.5), nothing]),
            FixedOmega(correlation = [1 0.3; 0.3 1]))
        fixed = LaplaceHooked(inner, growth_exact_subject; fixed = [1 => constraint])
        ωf = apply_fixed_omega(fixed, ω)
        c = NE._omega_constraints(fixed)
        ŷ = vcat(θ, σ, NE._free_omega(c, ωf))
        constrained = covariance_step(fixed, θ, σ, ωf; approximation = laplace, method = :r)
        @test constrained.evaluations == 2 * 5 + 1
        @test constrained.coordinates ≈ ŷ
        information = -ForwardDiff.hessian(
            v -> growth_exact(inner, vcat(v[1:3], NE._constrained_omega(c, v[4:end]))), ŷ)
        @test constrained.R ≈ information rtol = 1e-6
    end
end

@testset "covariance step: one-compartment FOCEI" begin
    model, _ = simulated_model(seed = 21, nsubjects = 16)
    ω = omega_parameters(nlme_layout(model), Ω_SIM)
    free = [1, 2, 3, 5, 7, 8, 9]
    a = covariance_step(model, θ_SIM, σ_SIM, ω; free, method = :r)
    b = covariance_step(model, θ_SIM, σ_SIM, ω; free, method = :r, step = 5e-4)
    @test a.names == ["log_ka", "log_CL", "log_V", "log_additive_sd",
        "ω[eta_ka,eta_ka]", "ω[eta_CL,eta_CL]", "ω[eta_V,eta_V]"]
    @test issymmetric(a.R)
    @test isposdef(Symmetric(a.R))
    @test a.R ≈ b.R rtol = 1e-4
    @test all(isfinite, a.standard_errors)
    rs = covariance_step(model, θ_SIM, σ_SIM, ω; free)
    @test rs.R == a.R
    @test isposdef(Symmetric(rs.covariance))
end

@testset "covariance step: nlmixr2 FOCEi oracle on theophylline" begin
    data(name) = readdlm(joinpath(@__DIR__, "data", name), ','; header = true)[1]
    estimates = Float64.(data("nlmixr2_focei_theoph_estimates.csv")[:, 2])
    nlmixr2_se(method) = sqrt.(diag(Float64.(data("nlmixr2_focei_theoph_cov_$method.csv")[:, 2:end])))
    model = theoph_model()
    θ, σ = vcat(estimates[1:3], 0.0), [log(estimates[4])]
    ω = log.(sqrt.(estimates[5:7]))
    free = [1, 2, 3, 5, 6, 7, 8]  # the oracle model has no weight effect
    # nlmixr2's OBJF omits the N_obs log(2π) constant that `ofv` keeps.
    nobs = sum(length, observations(model, 1:NE.nsubjects(model)))
    @test focei_evaluate(model, θ, σ, ω).ofv - nobs * log(2π) ≈ estimates[8] atol = 1e-3
    # nlmixr2 reports (tka, tcl, tv, add.sd, Ω variances).
    natural(x) = vcat(x[1:3], exp(x[5]), exp.(2 .* x[6:8]))
    se(method) = delta_method(covariance_step(model, θ, σ, ω; free, method),
        natural).standard_errors
    @test se(:s) ≈ nlmixr2_se("s") rtol = 1e-2  # measured: within 0.2%
    @test maximum(abs, se(:rs) ./ nlmixr2_se("rs") .- 1) < 0.03  # measured: within 2.4%
    # nlmixr2's "r" is not compared: its R exceeds the step-stable Hessian of
    # the shared objective by about N/2 on the η-related diagonal (see the
    # covariance section of docs/src/estimators.md).
end
