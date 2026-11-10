using NLMEEstimation
using Test
using Random
import NLMEEstimation as NE
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon

# `conditional_loglikelihood_sum_and_population_gradient` (NLMEEstimation 0.2.11):
# the batched θ/σ gradient of SAEM's numerical M-step and of `LouisInformation`, as
# a public method that a model implements and a wrapper forwards (snag
# `nlme-use-does-no-88813224`: a protocol-forwarding wrapper could not forward the
# internal BRM overrode, so the wrapped fit took another gradient route).

# y_ij ~ N(θ + η_i, exp(2σ)), evaluated as one batch; θ is mu-referenced, so the
# numerical M-step updates σ.
struct BatchedLinearPG
    y::Vector{Vector{Float64}}
end
NE.nsubjects(m::BatchedLinearPG) = length(m.y)
NE.nlme_layout(::BatchedLinearPG) = NLMELayout(1, 1, [1])
NE.evaluation_style(::BatchedLinearPG) = BatchedEvaluation()
NE.ad_backend(::BatchedLinearPG) = RUNTIME_REVERSE
NE.hessian_ad_backend(::BatchedLinearPG) = nothing
NE.mu_referencing(m::BatchedLinearPG) = MuReferencing([1], [ones(1, 1) for _ in m.y])
_residuals(m::BatchedLinearPG, subjects, θ, H) =
    [y - θ[1] - H[1, k] for (k, i) in enumerate(subjects) for y in m.y[i]]

# One nested unit per observation: y_ij ~ N(θ + η_ij, exp(2σ)) on a ragged layout.
struct RaggedLinearPG
    y::Vector{Vector{Float64}}
end
NE.nsubjects(m::RaggedLinearPG) = length(m.y)
NE.nlme_layout(m::RaggedLinearPG) = RaggedNLMELayout(ntheta = 1, nsigma = 1,
    family_sizes = [1], subject_families = [fill(1, length(y)) for y in m.y])
NE.evaluation_style(::RaggedLinearPG) = BatchedEvaluation()
NE.ad_backend(::RaggedLinearPG) = RUNTIME_REVERSE
NE.hessian_ad_backend(::RaggedLinearPG) = nothing
_residuals(m::RaggedLinearPG, subjects, θ, H) =
    [m.y[i][j] - θ[1] - H[k][j] for (k, i) in enumerate(subjects) for j in eachindex(m.y[i])]

function _linear_logdensities(m, subjects, θ, σ, H)
    v = exp(2σ[1])
    rows = m isa RaggedLinearPG ?
        [[m.y[i][j] - θ[1] - H[k][j] for j in eachindex(m.y[i])] for (k, i) in enumerate(subjects)] :
        [[y - θ[1] - H[1, k] for y in m.y[i]] for (k, i) in enumerate(subjects)]
    return [sum(r -> -(log(2π * v) + r^2 / v) / 2, residuals) for residuals in rows]
end
NE.conditional_loglikelihoods(m::Union{BatchedLinearPG, RaggedLinearPG}, subjects, θ, σ, H) =
    _linear_logdensities(m, subjects, θ, σ, H)

# The closed-form summed density and its θ/σ gradient.
function _linear_sum_and_gradient(m, subjects, θ, σ, H)
    v = exp(2σ[1])
    ℓ, gθ, gσ = 0.0, 0.0, 0.0
    for r in _residuals(m, subjects, θ, H)
        ℓ -= (log(2π * v) + r^2 / v) / 2
        gθ += r / v
        gσ += r^2 / v - 1
    end
    return ℓ, [gθ], [gσ]
end

const FORWARDED_PROTOCOL = (:nsubjects, :nlme_layout, :evaluation_style, :ad_backend,
    :hessian_ad_backend, :mu_referencing)
const FORWARDED_BATCH = (:conditional_loglikelihoods, :conditional_loglikelihoods_and_gradients)

# A model with its own batched population gradient (closed form, its θ part scaled
# by `scale`, so `scale = 2` is a wrong one), counting its calls.
struct Hooked{M}
    inner::M
    calls::Base.RefValue{Int}
    scale::Float64
end
Hooked(inner; scale = 1.0) = Hooked(inner, Ref(0), scale)
function NE.conditional_loglikelihood_sum_and_population_gradient(m::Hooked, subjects, θ, σ, H)
    m.calls[] += 1
    ℓ, gθ, gσ = _linear_sum_and_gradient(m.inner, subjects, θ, σ, H)
    return ℓ, m.scale .* gθ, gσ
end

# A model that overrides the internal of NLMEEstimation 0.2.10 and before, as BRM's
# extension does, counting its calls.
struct Legacy{M}
    inner::M
    calls::Base.RefValue{Int}
end
Legacy(inner) = Legacy(inner, Ref(0))
function NE._saem_population_value_gradient(style::BatchedEvaluation, m::Legacy, subjects, θ, σ, H)
    m.calls[] += 1
    return NE._saem_population_value_gradient(style, m.inner, subjects, θ, σ, H)
end

# A batched model that routes every numerical M-step mode per subject, so SAEM never
# asks it for the batched gradient; asking would throw.
struct PerSubjectRouted{M}
    inner::M
end
NE.saem_population_gradient_style(::PerSubjectRouted) = PerSubjectEvaluation()
NE.conditional_loglikelihood_sum_and_population_gradient(::PerSubjectRouted, subjects, θ, σ, H) =
    error("PerSubjectRouted is never asked for its batched population gradient")

# Protocol-forwarding wrappers (as a counting wrapper is): `Forwarding` forwards the
# batched population gradient with the rest, `Unforwarded` everything but it.
struct Forwarding{M}
    inner::M
end
struct Unforwarded{M}
    inner::M
end
for W in (:Hooked, :Legacy, :PerSubjectRouted, :Forwarding, :Unforwarded), f in FORWARDED_PROTOCOL
    @eval NE.$f(m::$W) = NE.$f(m.inner)
end
for W in (:Hooked, :Legacy, :PerSubjectRouted, :Forwarding, :Unforwarded), f in FORWARDED_BATCH
    @eval NE.$f(m::$W, subjects, θ, σ, H) = NE.$f(m.inner, subjects, θ, σ, H)
end
for W in (:Forwarding, :Unforwarded)
    @eval NE.saem_population_gradient_style(m::$W) = saem_population_gradient_style(m.inner)
    @eval NE.saem_population_gradient_style(m::$W, mode::Val) =
        saem_population_gradient_style(m.inner, mode)
end
NE.conditional_loglikelihood_sum_and_population_gradient(m::Forwarding, subjects, θ, σ, H) =
    conditional_loglikelihood_sum_and_population_gradient(m.inner, subjects, θ, σ, H)

same_fit(a, b) = a.theta == b.theta && a.sigma == b.sigma && a.omega == b.omega && a.eta == b.eta

@testset "conditional_loglikelihood_sum_and_population_gradient" begin
    y = [[-0.4, -0.2, 0.0], [0.1, 0.2], [0.8, 0.9, 1.0], [1.1, 1.3], [0.3, 0.5]]
    base = BatchedLinearPG(y)
    θ, σ, ω = [0.3], [log(0.4)], [log(0.5)]
    subjects = [4, 1, 3]
    H = reshape([0.2, -0.1, 0.05], 1, 3)
    online = SAEMOptions(iterations = 12, burnin = 4, mcmc_burnin = 2, numeric_maxiter = 2,
        mstep_interval = 2)
    fit(model; seed = 5, options = online) =
        saem(model, θ, σ, ω; rng = MersenneTwister(seed), options)

    @testset "the default is the route of 0.2.10, bit for bit" begin
        ℓ, gθ, gσ = conditional_loglikelihood_sum_and_population_gradient(base, subjects, θ, σ, H)
        value, gradient = NE._saem_population_value_gradient(BatchedEvaluation(), base,
            subjects, θ, σ, H)
        @test ℓ === value
        @test vcat(gθ, gσ) == gradient
        cℓ, cθ, cσ = _linear_sum_and_gradient(base, subjects, θ, σ, H)
        @test ℓ ≈ cℓ rtol = 1e-12
        @test gθ ≈ cθ rtol = 1e-12
        @test gσ ≈ cσ rtol = 1e-12
    end

    @testset "SAEM calls a model's own method, and a forwarding wrapper reaches it" begin
        hooked = Hooked(base)
        result = fit(hooked)
        @test hooked.calls[] > 0
        @test result.numeric_mstep == :online
        @test all(isfinite, (result.theta..., result.sigma..., result.omega...))
        wrapped = Forwarding(Hooked(base))
        @test same_fit(fit(wrapped), result)
        @test wrapped.inner.calls[] == hooked.calls[]
        score = SAEMOptions(iterations = 12, burnin = 4, mcmc_burnin = 2, mstep_interval = 2,
            numeric_mstep = :score)
        hooked_score = Hooked(base)
        @test same_fit(fit(Forwarding(hooked_score); options = score), fit(Hooked(base); options = score))
        @test hooked_score.calls[] > 0
    end

    @testset "minibatch SAEM calls it for the batch" begin
        minibatch = SAEMOptions(iterations = 12, burnin = 4, mcmc_burnin = 2, mstep_interval = 2,
            minibatch_size = 3)
        hooked = Hooked(base)
        result = fit(hooked; seed = 7, options = minibatch)
        @test result.numeric_mstep == :minibatch
        @test hooked.calls[] > 0
        wrapped = Forwarding(Hooked(base))
        @test same_fit(fit(wrapped; seed = 7, options = minibatch), result)
        @test wrapped.inner.calls[] == hooked.calls[]
    end

    @testset "a model overriding the 0.2.10 internal is reached, through a wrapper too" begin
        legacy = Legacy(base)
        direct = fit(legacy)
        @test legacy.calls[] > 0
        @test same_fit(direct, fit(base))
        forwarded = Forwarding(Legacy(base))
        @test same_fit(fit(forwarded), direct)
        @test forwarded.inner.calls[] == legacy.calls[]
        # A wrapper that does not forward the method gets the default, AD over the
        # wrapper itself, and never reaches the inner model's own gradient.
        unforwarded = Unforwarded(Legacy(base))
        fit(unforwarded)
        @test unforwarded.inner.calls[] == 0
    end

    @testset "LouisInformation differences the same gradient" begin
        legacy = Legacy(base)
        result = fit(legacy)
        before = legacy.calls[]
        louis = LouisInformation(draws = 200, burnin = 20)
        direct = covariance_step(legacy, result; approximation = louis, rng = Xoshiro(3))
        @test legacy.calls[] > before
        forwarded = Forwarding(Legacy(base))
        wrapped = covariance_step(forwarded, result; approximation = louis, rng = Xoshiro(3))
        @test forwarded.inner.calls[] == legacy.calls[] - before
        @test wrapped.covariance == direct.covariance
    end

    @testset "check_protocol compares it with the per-subject population gradients" begin
        η = reshape([0.2, -0.1, 0.05, 0.3, -0.2], 1, 5)
        hooked = Hooked(base)
        @test check_protocol(hooked, θ, σ, ω; η, population_gradient = true)
        @test hooked.calls[] == 1
        @test check_protocol(Forwarding(Hooked(base)), θ, σ, ω; η, population_gradient = true)
        wrong = Hooked(base; scale = 2.0)
        @test check_protocol(wrong, θ, σ, ω; η)
        @test wrong.calls[] == 0
        err = try
            check_protocol(wrong, θ, σ, ω; η, population_gradient = true)
        catch e
            e
        end
        @test err isa ProtocolViolation
        @test occursin("conditional_loglikelihood_sum_and_population_gradient's θ/σ gradient",
            err.message)
        # Not asked where SAEM never calls it.
        @test check_protocol(PerSubjectRouted(base), θ, σ, ω; η, population_gradient = true)

        ragged = RaggedLinearPG([[0.1], [0.4, -0.3], [0.2, 0.5, 0.9]])
        Hr = [[0.1], [-0.2, 0.3], [0.05, -0.1, 0.2]]
        ℓ, gθ, gσ = conditional_loglikelihood_sum_and_population_gradient(ragged, [3, 1], θ, σ,
            Hr[[3, 1]])
        cℓ, cθ, cσ = _linear_sum_and_gradient(ragged, [3, 1], θ, σ, Hr[[3, 1]])
        @test ℓ ≈ cℓ rtol = 1e-12
        @test gθ ≈ cθ rtol = 1e-12
        @test gσ ≈ cσ rtol = 1e-12
        @test check_protocol(Hooked(ragged), θ, σ, ω; η = Hr, population_gradient = true)
        @test_throws ProtocolViolation check_protocol(Hooked(ragged; scale = 2.0), θ, σ, ω;
            η = Hr, population_gradient = true)
    end
end
