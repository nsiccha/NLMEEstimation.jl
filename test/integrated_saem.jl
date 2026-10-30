using NLMEEstimation
using Test
using Random
using LinearAlgebra
import NLMEEstimation as NE
using DifferentiationInterface: AutoForwardDiff
import ForwardDiff
import Optim
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon
using .TestCommon: GROWTH_LAYOUT, LinearGrowth

# A model linear and Gaussian in its random effects that integrates any set of
# its covariance families itself, the shape of a joint program with an embedded
# Laplace approximation (BRM): given the sampled effects s, subject i's
# observations are Gaussian with mean μ₀ + Z_s s and covariance v I + Z_u D Z_uᵀ,
# where D is the integrated instances' prior covariance. Its Laplace
# conditionals, modes and covariance blocks are then exact closed forms; the
# population gradient is ForwardDiff's. Everything else forwards to `inner`.
# `shift`, `scale` and `inflate` distort the values, the sampled gradients and the
# covariance blocks for check_protocol's negative controls; `calls` counts the
# calls of each method.
struct LinearIntegrated{M}
    inner::M
    integrated::Vector{Int}
    shift::Float64
    scale::Float64
    inflate::Float64
    calls::Dict{Symbol, Int}
end
LinearIntegrated(inner, integrated; shift = 0.0, scale = 1.0, inflate = 1.0) =
    LinearIntegrated(inner, collect(Int, integrated), shift, scale, inflate, Dict{Symbol, Int}())
for f in (:nsubjects, :nlme_layout, :mu_referencing)
    @eval NE.$f(m::LinearIntegrated) = NE.$f(m.inner)
end
NE.conditional_loglikelihood(m::LinearIntegrated, i::Integer, θ, σ, η) =
    NE.conditional_loglikelihood(m.inner, i, θ, σ, η)
NE.conditional_loglikelihood_and_gradient(m::LinearIntegrated, i::Integer, θ, σ, η) =
    NE.conditional_loglikelihood_and_gradient(m.inner, i, θ, σ, η)
NE.conditional_loglikelihood_and_population_gradient(m::LinearIntegrated, i::Integer, θ, σ, η) =
    NE.conditional_loglikelihood_and_population_gradient(m.inner, i, θ, σ, η)
NE.population_logprior(m::LinearIntegrated, θ, σ, ω) = NE.population_logprior(m.inner, θ, σ, ω)
NE.ad_backend(::LinearIntegrated) = AutoForwardDiff()
NE.integrated_families(m::LinearIntegrated) = m.integrated

counted!(m, what) = (m.calls[what] = get(m.calls, what, 0) + 1)

# Subject i's observations y, their mean μ₀ at η = 0, the design Z of η and the
# residual variance v.
function linear_parts(m::RaggedOccasions, i, θ, σ)
    sizes = length.(m.y[i])
    Z = zeros(sum(sizes), 1 + length(sizes))
    Z[:, 1] .= 1
    offset = 0
    for (j, s) in enumerate(sizes)
        Z[offset+1:offset+s, 1+j] .= 1
        offset += s
    end
    return reduce(vcat, m.y[i]), fill(θ[1], sum(sizes)), Z, exp(2σ[1])
end
function linear_parts(m::LinearGrowth, i, θ, σ)
    Z = hcat(ones(length(m.t[i])), m.t[i])
    return m.y[i], Z * θ, Z, exp(2σ[1])
end

family_covariances(layout::RaggedNLMELayout, ω) = omega_matrix(layout, ω)
family_covariances(layout::NLMELayout, ω) = [omega_matrix(NLMELayout(0, 0, [k]), ω[r])
    for (k, r) in zip(layout.eta_blocks, last(NE._block_ranges(layout)))]

# Subject i's integrated entries U, sampled entries S, the observations' mean μ
# given the sampled effects and covariance C with the integrated ones integrated
# out, and the integrated instances' prior covariance D.
function integrated_parts(m::LinearIntegrated, F, i, θ, σ, ω, η)
    layout = NE.nlme_layout(m)
    y, μ0, Z, v = linear_parts(m.inner, i, θ, σ)
    Ωf = family_covariances(layout, ω)
    blocks = [(f, r) for (f, r) in NE._subject_blocks(layout, i) if f in F]
    U = reduce(vcat, [collect(r) for (_, r) in blocks]; init = Int[])
    S = setdiff(eachindex(η), U)
    T = promote_type(eltype(μ0), typeof(v), eltype(ω), eltype(η))
    D = zeros(T, length(U), length(U))
    offset = 0
    for (f, r) in blocks
        D[offset .+ (1:length(r)), offset .+ (1:length(r))] = Ωf[f]
        offset += length(r)
    end
    μ = μ0 .+ Z[:, S] * η[S]
    C = v * I + Z[:, U] * D * Z[:, U]'
    return (; y, μ, C, Z, U, S, D, v, blocks)
end

function integrated_value(m, F, i, θ, σ, ω, η)
    p = integrated_parts(m, F, i, θ, σ, ω, η)
    L = cholesky(Symmetric(p.C)).L
    z = L \ (p.y .- p.μ)
    return -(length(z) * log(2π) + 2sum(log, diag(L)) + dot(z, z)) / 2
end

function NE.laplace_conditionals_and_gradients(m::LinearIntegrated, F, subjects, θ, σ, ω, H)
    counted!(m, :gradients)
    layout = NE.nlme_layout(m)
    values = Float64[]
    gradients = Vector{Float64}[]
    for (k, i) in enumerate(subjects)
        η = NE._eta_column(H, k)
        p = integrated_parts(m, F, i, θ, σ, ω, η)
        g = zeros(length(η))
        g[p.S] = m.scale .* (p.Z[:, p.S]' * (cholesky(Symmetric(p.C)) \ (p.y .- p.μ)))
        push!(values, integrated_value(m, F, i, θ, σ, ω, η) + m.shift)
        push!(gradients, g)
    end
    return values, NE._eta_batch(layout, gradients)
end

function NE.laplace_conditionals(m::LinearIntegrated, F, subjects, θ, σ, ω, H)
    counted!(m, :values)
    return [integrated_value(m, F, i, θ, σ, ω, NE._eta_column(H, k)) + m.shift
        for (k, i) in enumerate(subjects)]
end

function NE.laplace_conditional_moments(m::LinearIntegrated, F, subjects, θ, σ, ω, H)
    counted!(m, :moments)
    layout = NE.nlme_layout(m)
    values = Float64[]
    modes = Vector{Float64}[]
    covariances = Vector{Matrix{Float64}}[]
    for (k, i) in enumerate(subjects)
        η = Vector{Float64}(NE._eta_column(H, k))
        p = integrated_parts(m, F, i, θ, σ, ω, η)
        ZU = p.Z[:, p.U]
        Σ = inv(Symmetric(ZU' * ZU / p.v + inv(Symmetric(p.D))))
        η[p.U] = Σ * (ZU' * (p.y .- p.μ)) / p.v
        offset = 0
        blocks = map(p.blocks) do (_, r)
            b = offset .+ (1:length(r))
            offset += length(r)
            m.inflate .* Matrix(Σ[b, b])
        end
        push!(values, integrated_value(m, F, i, θ, σ, ω, η) + m.shift)
        push!(modes, η)
        push!(covariances, blocks)
    end
    return values, NE._eta_batch(layout, modes), covariances
end

function NE.laplace_conditionals_and_population_gradient(m::LinearIntegrated, F, subjects,
        θ, σ, ω, H)
    counted!(m, :population)
    nθ, nσ = length(θ), length(σ)
    total(x) = sum(integrated_value(m, F, i, x[1:nθ], x[nθ+1:nθ+nσ], x[nθ+nσ+1:end],
        NE._eta_column(H, k)) for (k, i) in enumerate(subjects))
    x = vcat(θ, σ, ω)
    g = ForwardDiff.gradient(total, x)
    values = NE.laplace_conditionals(m, F, subjects, θ, σ, ω, H)
    return values, g[1:nθ], g[nθ+1:nθ+nσ], g[nθ+nσ+1:end]
end

# The maximum of `f` and its value.
function integrated_maximum(f, x0)
    result = Optim.optimize(x -> -f(x), (G, x) -> (G .= .-ForwardDiff.gradient(f, x)), x0,
        Optim.LBFGS(), Optim.Options(g_tol = 1e-10))
    @assert Optim.converged(result)
    return Optim.minimizer(result), -Optim.minimum(result)
end

const INTEGRATED_OCCASIONS_START = ([2.0], [log(0.25)], [log(0.6), log(0.4)])
const INTEGRATED_GROWTH_START = ([1.0, 0.5], [log(0.3)],
    omega_parameters(GROWTH_LAYOUT, [0.5 0.1; 0.1 0.2]))

@testset "integrated families: check_protocol" begin
    occasions = occasions_model(n = 6)
    θ, σ, ω = [2.1], [log(0.3)], [log(0.5), log(0.35)]
    η = [0.1 .* randn(Xoshiro(1), NE.neta(NE.nlme_layout(occasions), i)) for i in 1:6]
    for F in ([2], [1], [1, 2])
        @test check_protocol(LinearIntegrated(occasions, F), θ, σ, ω; η)
    end
    growth = growth_model(n = 5)
    θg, σg, ωg = INTEGRATED_GROWTH_START
    @test check_protocol(LinearIntegrated(growth, [1]), θg, σg, ωg)
    # Negative controls: a shifted value, a scaled gradient, an inflated covariance.
    for (kw, what) in (((shift = 1e-3,), "differs from the package's Laplace"),
            ((scale = 1.01,), "central difference"), ((inflate = 1.01,), "covariance block"))
        bad = LinearIntegrated(occasions, [2]; kw...)
        err = try
            check_protocol(bad, θ, σ, ω; η)
            nothing
        catch e
            e
        end
        @test err isa ProtocolViolation && occursin(what, err.message)
    end
    # A model that integrates nothing is not asked for the hooks.
    @test check_protocol(LinearIntegrated(occasions, Int[]), θ, σ, ω; η)
    # refused: an integrated family the layout does not have is not a family to integrate
    @test_throws ArgumentError saem(LinearIntegrated(occasions, [3]), INTEGRATED_OCCASIONS_START...)
    # A model without the hooks names the missing method.
    err = try
        saem(occasions, INTEGRATED_OCCASIONS_START...; integrated = [2],
            options = SAEMOptions(iterations = 2, burnin = 1, mcmc_burnin = 0))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("laplace_conditional_moments", err.msg)
end

@testset "integrated families: Laplace-EM reaches the exact maximum" begin
    # The models are Gaussian in their effects, so the Laplace moments are exact
    # and Laplace-EM is EM: it converges to the maximum-likelihood estimate.
    occasions = occasions_model(n = 30)
    x̂, best = integrated_maximum(x -> occasions_exact_nested(occasions, x),
        vcat(INTEGRATED_OCCASIONS_START...))
    model = LinearIntegrated(occasions, [1, 2])
    fit = saem(model, INTEGRATED_OCCASIONS_START...; rng = Xoshiro(1),
        options = SAEMOptions(iterations = 400))
    @test fit isa RaggedSAEMResult
    @test fit.kernel == :laplace_em && fit.integrated == [1, 2]
    @test fit.iterations == 400 && !fit.interrupted
    x = vcat(fit.theta, fit.sigma, fit.omega)
    @test best - occasions_exact_nested(occasions, x) < 1e-6
    @test maximum(abs, x - x̂) < 1e-3
    # Nothing is sampled: no conditional log-likelihood, no η gradient.
    @test !haskey(model.calls, :gradients)
    # eta holds the exact posterior means at the estimate.
    @test all(eachindex(fit.subjects)) do k
        means, _ = occasions_unit_posterior(occasions, fit.subjects[k], x)
        isapprox(fit.eta[k][2:end], means; atol = 1e-8)
    end

    # An equal-size layout with a dense 2 × 2 covariance block: the closed-form
    # step uses the instances' 2 × 2 covariance blocks.
    growth = growth_model(n = 40)
    x̂, best = integrated_maximum(x -> growth_exact(growth, x), vcat(INTEGRATED_GROWTH_START...))
    fit = saem(LinearIntegrated(growth, [1]), INTEGRATED_GROWTH_START...;
        options = SAEMOptions(iterations = 400))
    @test fit isa SAEMResult && fit.kernel == :laplace_em
    @test best - growth_exact(growth, vcat(fit.theta, fit.sigma, fit.omega)) < 1e-6

    # Under a population prior, the maximum a posteriori estimate.
    prior_sd = 2.0
    posterior(x) = growth_exact(growth, x) - sum(abs2, x) / (2prior_sd^2)
    _, best = integrated_maximum(posterior, vcat(INTEGRATED_GROWTH_START...))
    fit = saem(LinearIntegrated(growth_model(n = 40; prior_sd), [1]), INTEGRATED_GROWTH_START...;
        options = SAEMOptions(iterations = 400))
    @test fit.objective == :MAP
    @test best - posterior(vcat(fit.theta, fit.sigma, fit.omega)) < 1e-6
end

@testset "integrated families: SAEM sampling the rest" begin
    # Occasion effects integrated, subject effects sampled: SAEM samples one effect
    # per subject and ends as close to the maximum as plain SAEM sampling all.
    occasions = occasions_model(n = 30)
    exact(x) = occasions_exact_nested(occasions, x)
    _, best = integrated_maximum(exact, vcat(INTEGRATED_OCCASIONS_START...))
    gap(fit) = best - exact(vcat(fit.theta, fit.sigma, fit.omega))
    for kernel in (StandardSAEMMH(), WarmupHMCNUTS(max_tree_depth = 6), DualAveragingMALA())
        for seed in 1:3
            model = LinearIntegrated(occasions, [2])
            fit = saem(model, INTEGRATED_OCCASIONS_START...; rng = Xoshiro(seed), kernel)
            plain = saem(occasions, INTEGRATED_OCCASIONS_START...; rng = Xoshiro(seed), kernel)
            @test fit.integrated == [2] && fit.kernel == plain.kernel
            @test gap(fit) < 0.5
            @test gap(plain) < 1.0
        end
    end
    # Under a minibatch, with the same calibration.
    model = LinearIntegrated(occasions, [2])
    fit = saem(model, INTEGRATED_OCCASIONS_START...; rng = Xoshiro(1),
        options = SAEMOptions(iterations = 1200, burnin = 600, adapt_iterations = 600,
            minibatch_size = 8))
    @test gap(fit) < 0.5
    @test fit.numeric_mstep == :minibatch
    # With the running mixture of every retained draw as the numerical objective.
    fit = saem(LinearIntegrated(occasions, [2]), INTEGRATED_OCCASIONS_START...;
        rng = Xoshiro(1), options = SAEMOptions(numeric_mstep = :mixture))
    @test gap(fit) < 0.5
    @test fit.numeric_mstep == :mixture
end

@testset "integrated families: nothing integrated is the plain fit" begin
    occasions = occasions_model(n = 12)
    options = SAEMOptions(iterations = 60, burnin = 30, mcmc_burnin = 5)
    for kernel in (StandardSAEMMH(), WarmupHMCNUTS(max_tree_depth = 5), DualAveragingMALA())
        a = saem(occasions, INTEGRATED_OCCASIONS_START...; rng = Xoshiro(3), kernel, options)
        b = saem(LinearIntegrated(occasions, Int[]), INTEGRATED_OCCASIONS_START...;
            rng = Xoshiro(3), kernel, options)
        @test a.integrated == b.integrated == Int[]
        @test (a.theta, a.sigma, a.omega, a.eta) == (b.theta, b.sigma, b.omega, b.eta)
    end
end
