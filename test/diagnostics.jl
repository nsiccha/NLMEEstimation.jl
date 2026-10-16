using NLMEEstimation
using LinearAlgebra
using Random
using Statistics
using Test
using DelimitedFiles
import NLMEEstimation as NE
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon

@testset "residual diagnostics: normal quantile (AS 241)" begin
    q = NE._normal_quantile
    # Reference values Φ⁻¹(p) to double precision.
    for (p, x) in ((0.5, 0.0), (0.975, 1.959963984540054), (0.95, 1.6448536269514722),
            (0.9, 1.2815515655446004), (0.99, 2.3263478740408408),
            (0.999, 3.090232306167813), (0.8413447460685429, 1.0),
            (1e-10, -6.361340902404056))
        @test q(p) ≈ x atol = 1e-13
        # `1 - p` itself rounds; at p = 1e-10 that alone moves Φ⁻¹ by ~1e-8.
        @test q(1 - p) ≈ -x atol = 1e-7
    end
    @test_throws DomainError q(0.0)
    @test_throws DomainError q(1.0)
end

# Linear Gaussian: the linearization is exact, so CWRES is the subject's
# observation vector standardized by its exact marginal mean and covariance.
function growth_exact_cwres(m, θ, σ, ω, i)
    Z = hcat(ones(length(m.t[i])), m.t[i])
    C = Z * omega_matrix(GROWTH_LAYOUT, ω) * Z' + exp(2σ[1]) * I
    return cholesky(Symmetric(inv(C))).U * (m.y[i] - Z * θ)
end

@testset "residual diagnostics: exact linear-Gaussian CWRES and NPDE" begin
    model = growth_model(n = 40)
    θ, σ, ω = [1.0, 0.5], [log(0.3)], omega_parameters(GROWTH_LAYOUT, [0.5 0.1; 0.1 0.2])
    d = residual_diagnostics(model, θ, σ, ω; replicates = 4000, rng = Xoshiro(11))
    n = NE.nsubjects(model)
    @test d.subjects == 1:n && d.replicates == 4000
    @test d.observations == model.y
    for i in 1:n
        Z = hcat(ones(length(model.t[i])), model.t[i])
        @test d.pred[i] ≈ Z * θ
        @test d.ipred[i] ≈ Z * (θ + d.modes[:, i])
        @test d.iwres[i] ≈ (model.y[i] - d.ipred[i]) ./ 0.3
        @test d.cpred[i] ≈ Z * θ atol = 1e-10
        @test d.cwres[i] ≈ growth_exact_cwres(model, θ, σ, ω, i) atol = 1e-8
        C = Z * omega_matrix(GROWTH_LAYOUT, ω) * Z' + 0.3^2 * I
        @test d.cwres_per_observation[i] ≈ (model.y[i] - Z * θ) ./ sqrt.(diag(C)) atol = 1e-8
    end
    # With the exact mean and covariance, the decorrelated discrepancy is
    # Φ(CWRES); the simulation reproduces it up to Monte Carlo error.
    gap = reduce(vcat, d.npde) - reduce(vcat, d.cwres)
    @test mean(abs, gap) < 0.06
    @test maximum(abs, gap) < 0.4
    @test all(p -> 1 / 8000 <= p <= 1 - 1 / 8000, reduce(vcat, d.pde))
    # The simulated mean is PRED up to Monte Carlo error (~0.016 per value).
    epred_gap = reduce(vcat, d.epred) - reduce(vcat, d.pred)
    @test mean(abs, epred_gap) < 0.03 && maximum(abs, epred_gap) < 0.1
    # Data simulated from the model: CWRES and NPDE look standard normal.
    all_cwres = reduce(vcat, d.cwres)
    @test abs(mean(all_cwres)) < 0.25 && 0.75 < std(all_cwres) < 1.25
    @test abs(mean(reduce(vcat, d.npde))) < 0.25

    quick = residual_diagnostics(model, θ, σ, ω; replicates = 0)
    @test quick.npde === nothing && quick.epred === nothing && quick.replicates == 0
    @test quick.cwres == d.cwres
    @test_throws ArgumentError residual_diagnostics(model, θ, σ, ω; replicates = 3)

    fit = focei(model, θ, σ, ω; free = [1, 2], options = FOCEOptions(maxeval = 200))
    from_fit = residual_diagnostics(model, fit; replicates = 0)
    @test from_fit.cwres[1] ≈ growth_exact_cwres(model, fit.theta, fit.sigma, fit.omega, 1) atol = 1e-8
end

@testset "residual diagnostics: ragged nested effects" begin
    model = occasions_model(gaussian = true)
    layout = nlme_layout(model)
    θ, σ, ω = [2.0], [log(0.25)], [log(0.6), log(0.4)]
    d = residual_diagnostics(model, θ, σ, ω; replicates = 2000, rng = Xoshiro(12))
    for i in 1:NE.nsubjects(model)
        sizes = length.(model.y[i])
        N = sum(sizes)
        Z = zeros(N, length(sizes))
        offset = 0
        for (j, s) in enumerate(sizes)
            Z[offset+1:offset+s, j] .= 1
            offset += s
        end
        C = 0.6^2 * ones(N, N) + 0.4^2 * Z * Z' + 0.25^2 * I
        y = reduce(vcat, model.y[i])
        @test d.cwres[i] ≈ cholesky(Symmetric(inv(C))).U * (y .- 2.0) atol = 1e-7
        @test length(d.modes[i]) == neta(layout, i)
        @test length(d.npde[i]) == N
    end
    gap = reduce(vcat, d.npde) - reduce(vcat, d.cwres)
    @test mean(abs, gap) < 0.08
    @test_throws ArgumentError residual_diagnostics(occasions_model(), θ, σ, ω)
end

@testset "residual diagnostics: nlmixr2 FOCEi oracle on theophylline" begin
    e = Float64.(readdlm(joinpath(@__DIR__, "data", "nlmixr2_focei_theoph_estimates.csv"), ',';
        header = true)[1][:, 2])
    table, header = readdlm(joinpath(@__DIR__, "data", "nlmixr2_residuals_theoph.csv"), ','; header = true)
    column(name) = Float64.(table[:, findfirst(==(name), vec(header))])
    model = theoph_model()
    θ, σ, ω = vcat(e[1:3], 0.0), [log(e[4])], log.(sqrt.(e[5:7]))
    d = residual_diagnostics(model, θ, σ, ω; replicates = 1000, rng = Xoshiro(13),
        ebe_options = EBEOptions(gtol = 1e-10))
    flat(field) = reduce(vcat, getfield(d, field))
    @test flat(:observations) == column("DV")
    @test flat(:pred) ≈ column("PRED") atol = 1e-10
    # nlmixr2's modes stop at its inner tolerance (~6e-4 in η; empirical_bayes.jl).
    @test flat(:ipred) ≈ column("IPRED") atol = 1e-3
    @test flat(:iwres) ≈ column("IWRES") atol = 1e-3
    @test flat(:cpred) ≈ column("CPRED") atol = 1e-3
    # nlmixr2's CWRES standardizes each observation without decorrelation.
    @test flat(:cwres_per_observation) ≈ column("CWRES") atol = 1e-3
    # Simulation-based columns agree up to Monte Carlo error; nlmixr2 used 300
    # replicates. Measured correlations: NPD 0.995, NPDE 0.83.
    @test cor(flat(:epred), column("EPRED")) > 0.99
    @test cor(flat(:npd), column("NPD")) > 0.98
    @test cor(flat(:npde), column("NPDE")) > 0.7
end
