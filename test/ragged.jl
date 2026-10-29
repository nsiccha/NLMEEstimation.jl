using NLMEEstimation
using LinearAlgebra
using Random
using Test
using DifferentiationInterface: AutoEnzyme
import Enzyme
import NLMEEstimation as NE
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon: DoubleWell, SteepWell, GradientOnly

struct RaggedGaussianFixture
    layout::RaggedNLMELayout
    observed::Vector{Vector{Float64}}
end
NE.nsubjects(m::RaggedGaussianFixture) = length(m.observed)
NE.nlme_layout(m::RaggedGaussianFixture) = m.layout
NE.conditional_loglikelihood(m::RaggedGaussianFixture, i, θ, σ, η) =
    -sum(abs2, η - m.observed[i]) / 2
NE.conditional_loglikelihood_and_gradient(m::RaggedGaussianFixture, i, θ, σ, η) =
    (NE.conditional_loglikelihood(m, i, θ, σ, η), m.observed[i] - η)
NE.conditional_loglikelihood_gradient_and_hessian(m::RaggedGaussianFixture, i, θ, σ, η) =
    (NE.conditional_loglikelihood(m, i, θ, σ, η), m.observed[i] - η,
        -Matrix{Float64}(I, length(η), length(η)))
NE.provides_eta_hessian(::RaggedGaussianFixture) = true

struct BatchedRaggedGaussianFixture
    source::RaggedGaussianFixture
    batch_sizes::Vector{Int}
end
NE.nsubjects(m::BatchedRaggedGaussianFixture) = NE.nsubjects(m.source)
NE.nlme_layout(m::BatchedRaggedGaussianFixture) = NE.nlme_layout(m.source)
NE.evaluation_style(::BatchedRaggedGaussianFixture) = BatchedEvaluation()
function NE.conditional_loglikelihoods(m::BatchedRaggedGaussianFixture, subjects, θ, σ, H)
    push!(m.batch_sizes, length(subjects))
    return [NE.conditional_loglikelihood(m.source, i, θ, σ, H[k])
        for (k, i) in enumerate(subjects)]
end
function NE.conditional_loglikelihoods_and_gradients(m::BatchedRaggedGaussianFixture,
        subjects, θ, σ, H)
    return NE.conditional_loglikelihoods(m, subjects, θ, σ, H),
        [m.source.observed[i] - H[k] for (k, i) in enumerate(subjects)]
end
NE.conditional_loglikelihood_hessian_blocks(m::BatchedRaggedGaussianFixture,
    subjects, θ, σ, H) =
    [-Matrix{Float64}(I, length(H[k]), length(H[k])) for k in eachindex(subjects)]
NE.provides_eta_hessian(::BatchedRaggedGaussianFixture) = true

struct RaggedADFixture
    source::RaggedGaussianFixture
end
NE.nsubjects(m::RaggedADFixture) = NE.nsubjects(m.source)
NE.nlme_layout(m::RaggedADFixture) = NE.nlme_layout(m.source)
NE.evaluation_style(::RaggedADFixture) = BatchedEvaluation()
NE.ad_backend(::RaggedADFixture) = AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))
function NE.conditional_loglikelihoods(m::RaggedADFixture, subjects, θ, σ, H)
    values = zeros(eltype(H[1]), length(subjects))
    for (k, i) in enumerate(subjects)
        for j in eachindex(H[k])
            values[k] -= (H[k][j] - m.source.observed[i][j])^2 / 2
        end
    end
    return values
end

struct RaggedMuFixture
    layout::RaggedNLMELayout
    observed::Vector{Vector{Float64}}
end
NE.nsubjects(m::RaggedMuFixture) = length(m.observed)
NE.nlme_layout(m::RaggedMuFixture) = m.layout
NE.conditional_loglikelihood(m::RaggedMuFixture, i, θ, σ, η) =
    -sum(abs2, θ[1] .+ η .- m.observed[i]) / 2
NE.conditional_loglikelihood_and_gradient(m::RaggedMuFixture, i, θ, σ, η) =
    (NE.conditional_loglikelihood(m, i, θ, σ, η), m.observed[i] .- θ[1] .- η)
NE.conditional_loglikelihood_and_population_gradient(m::RaggedMuFixture,
        i, θ, σ, η) =
    (NE.conditional_loglikelihood(m, i, θ, σ, η),
        [sum(m.observed[i] .- θ[1] .- η)], Float64[])
NE.mu_referencing(m::RaggedMuFixture) =
    MuReferencing([1], [ones(neta(m.layout, i), 1) for i in 1:NE.nsubjects(m)])

struct RaggedScaleFixture
    layout::RaggedNLMELayout
    observed::Vector{Vector{Float64}}
end
NE.nsubjects(m::RaggedScaleFixture) = length(m.observed)
NE.nlme_layout(m::RaggedScaleFixture) = m.layout
function NE.conditional_loglikelihood(m::RaggedScaleFixture, i, θ, σ, η)
    residual = η - m.observed[i]
    return -length(residual) * σ[1] - sum(abs2, residual) * exp(-2σ[1]) / 2
end
NE.conditional_loglikelihood_and_gradient(m::RaggedScaleFixture, i, θ, σ, η) =
    (NE.conditional_loglikelihood(m, i, θ, σ, η),
        (m.observed[i] - η) * exp(-2σ[1]))
function NE.conditional_loglikelihood_and_population_gradient(m::RaggedScaleFixture,
        i, θ, σ, η)
    residual = η - m.observed[i]
    value = NE.conditional_loglikelihood(m, i, θ, σ, η)
    return value, Float64[], [-length(residual) + sum(abs2, residual) * exp(-2σ[1])]
end

struct RaggedMAPFixture
    source::RaggedScaleFixture
end
NE.nsubjects(m::RaggedMAPFixture) = NE.nsubjects(m.source)
NE.nlme_layout(m::RaggedMAPFixture) = NE.nlme_layout(m.source)
NE.conditional_loglikelihood(m::RaggedMAPFixture, i, θ, σ, η) =
    NE.conditional_loglikelihood(m.source, i, θ, σ, η)
NE.conditional_loglikelihood_and_gradient(m::RaggedMAPFixture, i, θ, σ, η) =
    NE.conditional_loglikelihood_and_gradient(m.source, i, θ, σ, η)
NE.conditional_loglikelihood_and_population_gradient(m::RaggedMAPFixture,
    i, θ, σ, η) =
    NE.conditional_loglikelihood_and_population_gradient(m.source, i, θ, σ, η)
NE.population_logprior(m::RaggedMAPFixture, θ, σ, ω) =
    -(σ[1]^2 + sum(abs2, ω)) / (2 * 0.5^2)
NE.population_logprior_and_gradient(m::RaggedMAPFixture, θ, σ, ω) =
    (NE.population_logprior(m, θ, σ, ω), Float64[], [-4σ[1]], -4ω)
NE.population_prior_provenance(::RaggedMAPFixture) = "synthetic Normal(0, 0.5) coordinates"

@testset "ragged random-effect families" begin
    layout = RaggedNLMELayout(ntheta=0, nsigma=0, family_sizes=[1, 1],
        subject_families=[[1, 2], [1, 2, 2, 2], [1, 2, 2]])
    model = RaggedGaussianFixture(layout,
        [[0.2, 1.0], [-0.3, -0.5, 0.8, 1.2], [0.4, 0.3, -0.7]])
    ω = log.([0.5, 1.0])
    @test neta(layout, 1) == 2
    @test neta(layout, 2) == 4
    @test nomega(layout) == 2
    @test omega_parameters(layout, omega_matrix(layout, ω)) ≈ ω
    dense_layout = RaggedNLMELayout(ntheta=0, nsigma=0, family_sizes=[2, 1],
        subject_families=[[1, 2], [1, 2, 2]])
    dense_covariances = [[1.0 0.25; 0.25 2.0], reshape([0.6], 1, 1)]
    dense_ω = omega_parameters(dense_layout, dense_covariances)
    @test all(isapprox.(omega_matrix(dense_layout, dense_ω), dense_covariances))
    dense_prior = RaggedNormalREPrior(dense_layout, dense_ω)
    _, _, dense_hessian = re_logprior_gradient_and_hessian(dense_prior, 2, zeros(4))
    @test dense_hessian[1:2, 1:2] ≈ -inv(dense_covariances[1])
    @test dense_hessian[3, 4] == 0
    dense_design = [zeros(3, 0), zeros(4, 0)]
    dense_mean = [zeros(3), zeros(4)]
    # Second moments are kept per block, as `_saem_statistics` lays them out.
    dense_second = [[D[r, r] for (_, r) in NE._ragged_subject_blocks(dense_layout, i)]
        for (i, D) in enumerate([Matrix{Float64}(I, 3, 3), Matrix{Float64}(I, 4, 4)])]
    _, _, dense_gradient = NE._saem_gaussian_map_value_gradient(dense_layout,
        Float64[], dense_ω, Int[], dense_design, dense_mean, dense_second, 1:2)
    for j in eachindex(dense_ω)
        step = zeros(length(dense_ω))
        step[j] = 1e-6
        plus = first(NE._saem_gaussian_map_value_gradient(dense_layout,
            Float64[], dense_ω + step, Int[], dense_design, dense_mean, dense_second, 1:2))
        minus = first(NE._saem_gaussian_map_value_gradient(dense_layout,
            Float64[], dense_ω - step, Int[], dense_design, dense_mean, dense_second, 1:2))
        @test dense_gradient[j] ≈ (plus - minus) / 2e-6 atol=1e-5
    end
    prior = re_prior(model, ω)
    @test prior isa RaggedNormalREPrior
    @test isapprox(re_logprior(prior, 2, zeros(4)) - re_logprior(prior, 1, zeros(2)),
        -2log(2π) / 2; atol=1e-12)
    H = [zeros(neta(layout, i)) for i in 1:3]
    values, gradients = conditional_loglikelihoods_and_gradients(model, 1:3,
        Float64[], Float64[], H)
    @test values == [-sum(abs2, y) / 2 for y in model.observed]
    @test gradients == model.observed
    @test check_protocol(model, Float64[], Float64[], ω)
    @test length.(parameter_names(model).eta) == [2, 4, 3]

    batched = BatchedRaggedGaussianFixture(model, Int[])
    @test check_protocol(batched, Float64[], Float64[], ω)
    @test conditional_loglikelihood(batched, 2, Float64[], Float64[], zeros(4)) ==
        conditional_loglikelihood(model, 2, Float64[], Float64[], zeros(4))

    result = empirical_bayes(model, Float64[], Float64[], ω; method=QuasiNewtonEBE())
    @test result isa RaggedEBEResult
    @test length.(result.modes) == [2, 4, 3]
    @test all(result.converged)
    reordered = empirical_bayes(model, Float64[], Float64[], ω;
        subjects=[3, 1], method=QuasiNewtonEBE())
    @test reordered.subjects == [3, 1]
    @test length.(reordered.modes) == [3, 2]
    @test length.(parameter_names(model).eta[reordered.subjects]) == [3, 2]
    for i in 1:3
        variances = [0.25; ones(neta(layout, i) - 1)]
        @test result.modes[i] ≈ model.observed[i] .* variances ./ (variances .+ 1) atol=1e-6
    end
    newton = empirical_bayes(model, Float64[], Float64[], ω; method=NewtonEBE())
    @test all(newton.converged)
    @test all(isapprox.(newton.modes, result.modes; atol=1e-6))
    @test length.(ebe_covariances(newton)) == [4, 16, 9]

    # Newton leaves a start where the density is not concave (see the equal-size
    # test) instead of accepting a zero step.
    well_layout = RaggedNLMELayout(ntheta=0, nsigma=0, family_sizes=[1],
        subject_families=[[1], [1], [1]])
    well = DoubleWell(well_layout, 1.5, [0.05, 0.1, 0.2])
    well_ω = omega_parameters(well_layout, [[1.0;;]])
    well_newton = empirical_bayes(well, Float64[], Float64[], well_ω;
        η0=[[0.1] for _ in 1:3], method=NewtonEBE())
    @test all(well_newton.converged)
    well_η = only.(well_newton.modes)
    @test maximum(abs, -well_η .^ 3 .+ 0.5 .* well_η .+ well.b) < 1e-12
    @test all(H -> H[1, 1] < 0, well_newton.hessians)
    well_equal = DoubleWell(NLMELayout(0, 0, [1]), 1.5, well.b)
    @test well_η ≈ vec(empirical_bayes(well_equal, Float64[], Float64[],
        omega_parameters(nlme_layout(well_equal), [1.0;;]); η0=fill(0.1, 1, 3),
        method=NewtonEBE()).modes) atol=1e-12
    # The trust region is measured in prior standard deviations of each block
    # instance, so it leaves a saddle as the equal-size layout with the same Ω does.
    steep_layout = RaggedNLMELayout(ntheta=0, nsigma=0, family_sizes=[1, 1],
        subject_families=[[1, 2], [1, 2], [1, 2]])
    steep = SteepWell(steep_layout, 1.0e4, 1.5, [0.0, 1.0e-6, 1.0e-2], 0.3)
    steep_newton = empirical_bayes(steep, Float64[], Float64[],
        omega_parameters(steep_layout, [[1.0;;], [4.0;;]]);
        η0=[[0.0, 1.0e-6] for _ in 1:3], method=NewtonEBE())
    @test all(steep_newton.converged)
    @test maximum(steep_newton.iterations) <= 10
    steep_η = reduce(hcat, steep_newton.modes)
    @test maximum(abs, -steep_η[2, :] .^ 3 .+ 1.25 .* steep_η[2, :] .+ steep.b) < 1e-12
    # BFGS leaves it as well, by the Hessian block or, gradient-only, by forward
    # differences on the compact vectors.
    for well_model in (steep, GradientOnly(steep))
        steep_bfgs = empirical_bayes(well_model, Float64[], Float64[],
            omega_parameters(steep_layout, [[1.0;;], [4.0;;]]);
            η0=[[0.0, 1.0e-6] for _ in 1:3], method=QuasiNewtonEBE())
        @test all(steep_bfgs.converged)
        @test reduce(hcat, steep_bfgs.modes) ≈ steep_η atol=1e-8
    end
    steep_equal = SteepWell(NLMELayout(0, 0, [1, 1]), steep.D, steep.a, steep.b, steep.c)
    @test steep_η ≈ empirical_bayes(steep_equal, Float64[], Float64[],
        omega_parameters(nlme_layout(steep_equal), [1.0 0.0; 0.0 4.0]);
        η0=[zeros(1, 3); fill(1.0e-6, 1, 3)], method=NewtonEBE()).modes atol=1e-12
    batched_result = empirical_bayes(batched, Float64[], Float64[], ω;
        method=QuasiNewtonEBE())
    @test all(batched_result.converged)
    @test all(isapprox.(batched_result.modes, result.modes; atol=1e-6))
    @test 3 in batched.batch_sizes
    ad_model = RaggedADFixture(model)
    ad_values, ad_gradients = conditional_loglikelihoods_and_gradients(ad_model,
        1:3, Float64[], Float64[], H)
    @test ad_values == values
    @test ad_gradients ≈ gradients

    # Second moments are kept per block, as `_saem_statistics` lays them out.
    family_moments = [[D[r, r] for (_, r) in NE._ragged_subject_blocks(layout, i)]
        for (i, D) in enumerate([Diagonal([4.0, 1.0]) |> Matrix,
            Diagonal([9.0, 4.0, 9.0, 16.0]) |> Matrix,
            Diagonal([16.0, 25.0, 36.0]) |> Matrix])]
    _, pooled = NE._ragged_gaussian_mstep(layout, Float64[], omega_matrix(layout, ω),
        Int[], [zeros(neta(layout, i), 0) for i in 1:3],
        [zeros(neta(layout, i)) for i in 1:3], family_moments, 1e-8, 1:3)
    @test pooled[1][1, 1] ≈ (4 + 9 + 16) / 3
    @test pooled[2][1, 1] ≈ (1 + 4 + 9 + 16 + 25 + 36) / 6

    fitted = saem(model, Float64[], Float64[], ω; rng=Xoshiro(17),
        options=SAEMOptions(iterations=30, burnin=10, mcmc_burnin=2,
            mh_rounds=2, numeric_mstep=:online))
    @test fitted isa RaggedSAEMResult
    @test length.(fitted.eta) == [2, 4, 3]
    @test length(fitted.omega_matrix) == 2
    @test all(M -> isposdef(Symmetric(M)), fitted.omega_matrix)
    @test 0 <= fitted.acceptance.prior <= 1
    @test 0 <= fitted.acceptance.random_walk <= 1
    @test 0 < fitted.acceptance.coordinates < 1

    # The coordinate sweep proposes a family's coordinate only to subjects that
    # have the family, moving it in each of their instances.
    sparse_layout = RaggedNLMELayout(ntheta=0, nsigma=0, family_sizes=[1, 2],
        subject_families=[[1], [1, 2, 2], [1, 2]])
    sparse = RaggedGaussianFixture(sparse_layout, [[0.2], [0.1, 0.4, -0.3, 0.2, 0.5], [-0.2, 0.3, 0.1]])
    sparse_prior = RaggedNormalREPrior(sparse_layout, zeros(nomega(sparse_layout)))
    Hs = [zeros(neta(sparse_layout, i)) for i in 1:3]
    ℓs = NE.conditional_loglikelihoods(sparse, 1:3, Float64[], Float64[], Hs)
    accepted, proposed = NE._saem_mh_component_sweep!(Xoshiro(2), sparse, [1, 2, 3],
        Float64[], Float64[], sparse_prior, Hs, ℓs, [0.5, 0.5, 0.5])
    @test proposed == [3, 2, 2]
    @test all(0 .<= accepted .<= proposed) && sum(accepted) > 0
    @test length.(Hs) == [1, 5, 3]
    @test ℓs ≈ NE.conditional_loglikelihoods(sparse, 1:3, Float64[], Float64[], Hs)

    mu_layout = RaggedNLMELayout(ntheta=1, nsigma=0, family_sizes=[1],
        subject_families=[[1], [1, 1], [1, 1, 1]])
    mu_model = RaggedMuFixture(mu_layout, [[1.8], [2.0, 1.7], [1.9, 2.2, 1.6]])
    @test check_protocol(mu_model, [1.5], Float64[], [log(0.4)])
    mu_fit = saem(mu_model, [1.5], Float64[], [log(0.4)]; rng=Xoshiro(6),
        options=SAEMOptions(iterations=30, burnin=10, mcmc_burnin=2,
            mh_rounds=2, numeric_mstep=:online))
    @test isfinite(mu_fit.theta[1])
    @test 1.0 < mu_fit.theta[1] < 3.0
    @test length.(mu_fit.eta) == [1, 2, 3]

    scale_layout = RaggedNLMELayout(ntheta=0, nsigma=1, family_sizes=[1],
        subject_families=[[1], [1, 1], [1, 1, 1]])
    scale_model = RaggedScaleFixture(scale_layout,
        [[-0.4], [0.5, -0.2], [0.3, -0.7, 0.1]])
    scale_fit = saem(scale_model, Float64[], [log(0.8)], [log(0.5)];
        rng=Xoshiro(5), options=SAEMOptions(iterations=25, burnin=10,
            mcmc_burnin=2, numeric_mstep=:online, numeric_maxiter=4))
    @test isfinite(scale_fit.sigma[1])
    @test length.(scale_fit.eta) == [1, 2, 3]
    mixture_fit = saem(scale_model, Float64[], [log(0.8)], [log(0.5)];
        rng=Xoshiro(8), options=SAEMOptions(iterations=6, burnin=2,
            mcmc_burnin=0, numeric_mstep=:mixture, numeric_maxiter=2))
    @test mixture_fit.numeric_mstep == :mixture
    @test isfinite(mixture_fit.sigma[1])
    map_fit = saem(RaggedMAPFixture(scale_model), Float64[], [log(0.8)], [log(0.5)];
        rng=Xoshiro(9), options=SAEMOptions(iterations=6, burnin=2,
            mcmc_burnin=0, numeric_mstep=:mixture, numeric_maxiter=3))
    @test map_fit.objective == :MAP
    @test isfinite(map_fit.population_logprior)
    @test map_fit.prior_provenance == "synthetic Normal(0, 0.5) coordinates"
end
