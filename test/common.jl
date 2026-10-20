# Shared fixtures: the reference model on simulated data and on R's Theoph data, a
# batched joint-program view of it, and the brute-force joint-optimization reference
# for the EBEs. The large joint oracle uses ForwardDiff; production paths use Enzyme.
module TestCommon

using NLMEEstimation
using NLMEEstimation.ReferenceModels
import NLMEEstimation as NE
import DifferentiationInterface as DI
using DifferentiationInterface: AutoEnzyme, AutoForwardDiff
import Enzyme
import ForwardDiff
using DelimitedFiles
using LinearAlgebra
using Random
import Optim

export θ_SIM, σ_SIM, Ω_SIM, simulated_model, θ_THEOPH, σ_THEOPH, Ω_THEOPH, theoph_model,
    JointView, joint_reference, joint_logdensity, joint_conditional, REVERSE, RUNTIME_REVERSE, RUNTIME_HESSIAN, JOINT_HESSIAN,
    LinearGrowth, GROWTH_LAYOUT, growth_model, growth_exact, growth_exact_subject,
    RaggedOccasions, occasions_model, occasions_exact, occasions_exact_subject,
    DoubleWell, SteepWell, GradientOnly

# The promised configuration: Enzyme reverse mode with static activity analysis (DI
# selects forward-over-reverse for Hessians of `AutoEnzyme()`). The package's own
# code and the reference model are tested with it.
const REVERSE = AutoEnzyme()
# Test fixtures written as plain Julia that Enzyme's static activity analysis cannot
# prove (closures and comprehensions mixing constant model data with active η:
# `JointView`, the coupled negative control, the joint reference) declare Enzyme's
# documented runtime-activity mode. The same backends serve as an independently
# configured reference for the promised path.
const RUNTIME_REVERSE = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
const RUNTIME_HESSIAN = DI.SecondOrder(AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Forward)), RUNTIME_REVERSE)
# An independent joint-Hessian oracle uses forward-mode AD over all subjects.
# Enzyme's forward-over-reverse compiler on Julia prereleases can fail on this
# large reference expression even when the package's per-subject Hessians work.
const JOINT_HESSIAN = AutoForwardDiff()

const θ_SIM = [log(1.5), log(2.8), log(32.0), 0.75]
const σ_SIM = [log(0.3), log(0.1)]
const Ω_SIM = Matrix(Diagonal([0.4, 0.3, 0.2] .^ 2))

function simulated_model(; seed = 11, nsubjects = 24, Ω = Ω_SIM, eta_blocks = [1, 1, 1])
    subjects, η = simulate_one_compartment_oral(Xoshiro(seed), θ_SIM, σ_SIM, Ω; nsubjects)
    return OneCompartmentOral(subjects; backend = REVERSE, eta_blocks), η
end

# One η per subject, ℓ_i(η) = -(η² - a)²/4 + b_i η, with closed-form derivatives, for
# either layout. With a unit prior variance and a > 1 the complete-data density is
# convex near η = 0 and concave at its maximum, so the Newton step does not exist at
# a start near 0.
struct DoubleWell{L}
    layout::L
    a::Float64
    b::Vector{Float64}
end
NE.nsubjects(m::DoubleWell) = length(m.b)
NE.nlme_layout(m::DoubleWell) = m.layout
NE.conditional_loglikelihood(m::DoubleWell, i, θ, σ, η) = -(η[1]^2 - m.a)^2 / 4 + m.b[i] * η[1]
NE.conditional_loglikelihood_and_gradient(m::DoubleWell, i, θ, σ, η) =
    (NE.conditional_loglikelihood(m, i, θ, σ, η), [-(η[1]^2 - m.a) * η[1] + m.b[i]])
NE.conditional_loglikelihood_gradient_and_hessian(m::DoubleWell, i, θ, σ, η) =
    (NE.conditional_loglikelihood_and_gradient(m, i, θ, σ, η)..., fill(m.a - 3 * η[1]^2, 1, 1))
NE.provides_eta_hessian(::DoubleWell) = true

# Two η per subject, a steep concave coordinate and a double well,
# ℓ_i(η) = -D(η₁ - c)²/2 - (η₂² - a)²/4 + b_i η₂, for either layout. With unit prior
# variances, a > 1 and a small |b_i|, a start with η₂ at or near 0 lies at or next to
# a saddle point of the complete-data density, which curves upward along η₂ with
# curvature a - 1. A Levenberg–Marquardt step (-H + λI)⁻¹g whose λ scales with the
# steep curvature D leaves that saddle only by a factor λ/(λ - a + 1) per iteration.
struct SteepWell{L}
    layout::L
    D::Float64
    a::Float64
    b::Vector{Float64}
    c::Float64
end
NE.nsubjects(m::SteepWell) = length(m.b)
NE.nlme_layout(m::SteepWell) = m.layout
NE.conditional_loglikelihood(m::SteepWell, i, θ, σ, η) =
    -m.D * (η[1] - m.c)^2 / 2 - (η[2]^2 - m.a)^2 / 4 + m.b[i] * η[2]
NE.conditional_loglikelihood_and_gradient(m::SteepWell, i, θ, σ, η) =
    (NE.conditional_loglikelihood(m, i, θ, σ, η), [-m.D * (η[1] - m.c), -(η[2]^2 - m.a) * η[2] + m.b[i]])
NE.conditional_loglikelihood_gradient_and_hessian(m::SteepWell, i, θ, σ, η) =
    (NE.conditional_loglikelihood_and_gradient(m, i, θ, σ, η)..., [-m.D 0.0; 0.0 m.a - 3 * η[2]^2])
NE.provides_eta_hessian(::SteepWell) = true

# A gradient-only view of a per-subject model with closed-form gradients (such as
# `DoubleWell` or `SteepWell`): the same values and gradients, no Hessian blocks.
struct GradientOnly{M}
    inner::M
end
NE.nsubjects(m::GradientOnly) = NE.nsubjects(m.inner)
NE.nlme_layout(m::GradientOnly) = NE.nlme_layout(m.inner)
NE.conditional_loglikelihood(m::GradientOnly, i, θ, σ, η) = NE.conditional_loglikelihood(m.inner, i, θ, σ, η)
NE.conditional_loglikelihood_and_gradient(m::GradientOnly, i, θ, σ, η) =
    NE.conditional_loglikelihood_and_gradient(m.inner, i, θ, σ, η)
NE.provides_eta_hessian(::GradientOnly) = false

# Typical-value parameters of the standard one-compartment Theoph fit (no weight
# effect), additive error.
const θ_THEOPH = [log(1.57), log(2.75), log(31.5), 0.0]
const σ_THEOPH = [log(0.7)]
const Ω_THEOPH = Matrix(Diagonal([0.6, 0.26, 0.14] .^ 2))

function theoph_model()
    data, header = readdlm(joinpath(@__DIR__, "data", "theoph.csv"), ','; header = true)
    col(name) = data[:, findfirst(==(name), vec(header))]
    id, wt, dose, time, conc = col("Subject"), col("Wt"), col("Dose"), col("Time"), col("conc")
    subjects = map(sort(unique(id))) do s
        rows = findall(==(s), id)
        PKSubject(dose[rows[1]] * wt[rows[1]], wt[rows[1]], time[rows], conc[rows])
    end
    return OneCompartmentOral(subjects; backend = REVERSE, error_model = AdditiveError())
end

# A joint-program view of a per-subject model, shaped like a BRM/RK backend: it
# declares BatchedEvaluation and implements only the batched log-density, by placing
# the requested subjects' η into a full N-subject matrix (other subjects at zero) and
# evaluating every subject. `hessian_backend = nothing` makes it gradient-only.
# `calls` counts batched evaluations.
struct JointView{M, B, HB}
    inner::M
    backend::B
    hessian_backend::HB
    calls::Base.RefValue{Int}
end
JointView(inner; backend = RUNTIME_REVERSE, hessian_backend = RUNTIME_HESSIAN) =
    JointView(inner, backend, hessian_backend, Ref(0))
NE.evaluation_style(::JointView) = BatchedEvaluation()
NE.nsubjects(v::JointView) = nsubjects(v.inner)
NE.nlme_layout(v::JointView) = nlme_layout(v.inner)
NE.ad_backend(v::JointView) = v.backend
NE.hessian_ad_backend(v::JointView) = v.hessian_backend
function NE.conditional_loglikelihoods(v::JointView, subjects, θ, σ, H)
    v.calls[] += 1
    n = nsubjects(v.inner)
    # Scalar scatter avoids Julia 1.13 Enzyme's unsupported indexed slice write.
    full = zeros(eltype(H), size(H, 1), n)
    for (k, i) in enumerate(subjects), r in axes(H, 1)
        full[r, i] = H[r, k]
    end
    all = [conditional_loglikelihood(v.inner, i, θ, σ, view(full, :, i)) for i in 1:n]
    return all[subjects]
end

# The joint complete-data log-density over every subject's η, stacked in x.
function joint_logdensity(x, model, θ, σ, prior, m)
    n = nsubjects(model)
    return sum(i -> complete_data_loglikelihood(model, i, θ, σ, prior, x[((i - 1) * m + 1):(i * m)]), 1:n)
end
_negative_joint(x, model, θ, σ, prior, m) = -joint_logdensity(x, model, θ, σ, prior, m)

# The joint conditional log-density (no η prior) over every subject's η, stacked in x.
# Its Hessian plus the prior's closed-form block-diagonal -Ω⁻¹ is the joint Hessian;
# AD never differentiates the prior's triangular solve in forward mode (Enzyme has no
# forward-mode rule for LAPACK `trtrs`).
function joint_conditional(x, model, θ, σ, m)
    n = nsubjects(model)
    return sum(i -> conditional_loglikelihood(model, i, θ, σ, x[((i - 1) * m + 1):(i * m)]), 1:n)
end

# The EBEs as one joint optimization: maximize the sum of all subjects' complete-data
# log-densities over all η at once with L-BFGS and an Enzyme gradient (runtime
# activity, see `RUNTIME_REVERSE`). It shares no
# code with the per-subject solvers beyond the log-density itself.
function joint_reference(model, θ, σ, ω)
    n = nsubjects(model)
    m = neta(nlme_layout(model))
    prior = re_prior(model, ω)
    contexts = (DI.Constant(model), DI.Constant(θ), DI.Constant(σ), DI.Constant(prior), DI.Constant(m))
    objective(x) = _negative_joint(x, model, θ, σ, prior, m)
    gradient!(G, x) = DI.gradient!(_negative_joint, G, RUNTIME_REVERSE, x, contexts...)
    result = Optim.optimize(objective, gradient!, zeros(n * m), Optim.LBFGS(),
        Optim.Options(g_tol = 1.0e-11, iterations = 100_000))
    Optim.converged(result) || error("joint reference did not converge: $result")
    return reshape(Optim.minimizer(result), m, n), prior
end

# Random intercepts and slopes with a dense 2 × 2 Ω and additive error. The model
# is linear and Gaussian in η, so FOCE(I) is exact and every subject's marginal
# log-likelihood is known in closed form: an oracle for both R and S.
struct LinearGrowth
    t::Vector{Vector{Float64}}
    y::Vector{Vector{Float64}}
    prior_sd::Union{Nothing, Float64}
end

const GROWTH_LAYOUT = NLMELayout(2, 1, [2])
NE.nsubjects(m::LinearGrowth) = length(m.y)
NE.nlme_layout(::LinearGrowth) = GROWTH_LAYOUT
NE.observation_model(::LinearGrowth) = GaussianObservations()
NE.observations(m::LinearGrowth, i::Integer) = m.y[i]
NE.provides_eta_hessian(::LinearGrowth) = true
NE.mu_referencing(m::LinearGrowth) =
    MuReferencing([1, 2], [Matrix(1.0I, 2, 2) for _ in m.y])
growth_mean(m, i, θ, η) = (θ[1] + η[1]) .+ (θ[2] + η[2]) .* m.t[i]

function NE.conditional_loglikelihood(m::LinearGrowth, i::Integer, θ, σ, η)
    r = m.y[i] .- growth_mean(m, i, θ, η)
    return -sum(abs2, r) / (2exp(2σ[1])) - length(r) * (log(2π) / 2 + σ[1])
end

function NE.conditional_loglikelihood_and_gradient(m::LinearGrowth, i::Integer, θ, σ, η)
    r = m.y[i] .- growth_mean(m, i, θ, η)
    v = exp(2σ[1])
    return NE.conditional_loglikelihood(m, i, θ, σ, η), [sum(r), dot(r, m.t[i])] ./ v
end

function NE.conditional_loglikelihood_gradient_and_hessian(m::LinearGrowth, i::Integer, θ, σ, η)
    ℓ, g = NE.conditional_loglikelihood_and_gradient(m, i, θ, σ, η)
    t = m.t[i]
    return ℓ, g, -[length(t) sum(t); sum(t) dot(t, t)] ./ exp(2σ[1])
end

function NE.conditional_loglikelihood_and_population_gradient(m::LinearGrowth, i::Integer, θ, σ, η)
    ℓ, g = NE.conditional_loglikelihood_and_gradient(m, i, θ, σ, η)
    r = m.y[i] .- growth_mean(m, i, θ, η)
    return ℓ, g, [sum(abs2, r) / exp(2σ[1]) - length(r)]
end

NE.observation_moments(m::LinearGrowth, i::Integer, θ, σ, η) =
    (growth_mean(m, i, θ, η), fill(exp(2σ[1]), length(m.y[i])))

function NE.observation_moments_and_jacobians(m::LinearGrowth, i::Integer, θ, σ, η)
    μ, v = NE.observation_moments(m, i, θ, σ, η)
    n = length(μ)
    return μ, v, hcat(ones(n), m.t[i]), zeros(n, 2)
end

NE.population_logprior(m::LinearGrowth, θ, σ, ω) = m.prior_sd === nothing ? nothing :
    -sum(abs2, vcat(θ, σ, ω)) / (2m.prior_sd^2)

function growth_exact_subject(m, i, x)
    θ, σ, ω = x[1:2], x[3], x[4:6]
    Z = hcat(ones(length(m.t[i])), m.t[i])
    C = Z * omega_matrix(GROWTH_LAYOUT, ω) * Z' + exp(2σ) * I
    L = cholesky(Symmetric(C)).L
    z = L \ (m.y[i] - Z * θ)
    return -(length(z) * log(2π) + 2sum(log, diag(L)) + dot(z, z)) / 2
end

growth_exact(m, x) = sum(i -> growth_exact_subject(m, i, x), 1:NE.nsubjects(m))

function growth_model(; seed = 5, n = 12, prior_sd = nothing)
    rng = Xoshiro(seed)
    Ω = [0.5 0.1; 0.1 0.2]
    L = cholesky(Ω).L
    t = [sort(3rand(rng, 3 + rand(rng, 0:3))) for _ in 1:n]
    y = map(t) do ti
        η = L * randn(rng, 2)
        (1.0 + η[1]) .+ (0.5 + η[2]) .* ti .+ 0.3 .* randn(rng, length(ti))
    end
    return LinearGrowth(t, y, prior_sd)
end

# Nested intercepts with unequal subject sizes: one subject-level effect a_i and
# one occasion-level effect b_ij per occasion, from two covariance families, with
# additive error. The θ intercept is mu-referenced through a_i. The model declares
# no observation model and no η Hessian, so the marginal approximations use
# finite-difference curvature. Every subject's marginal is Gaussian and exact.
struct RaggedOccasions
    y::Vector{Vector{Vector{Float64}}}
    calls::Base.RefValue{Int}
    gaussian::Bool
end

RaggedOccasions(y; gaussian = false) = RaggedOccasions(y, Ref(0), gaussian)
# `gaussian = true` declares the Gaussian observation protocol, for FOCE-style
# residual diagnostics on a ragged layout.
NE.observation_model(m::RaggedOccasions) = m.gaussian ? GaussianObservations() : nothing
NE.observations(m::RaggedOccasions, i::Integer) = reduce(vcat, m.y[i])
NE.observation_moments(m::RaggedOccasions, i::Integer, θ, σ, η) =
    (reduce(vcat, [fill(θ[1] + η[1] + η[1+j], length(o)) for (j, o) in enumerate(m.y[i])]),
        fill(exp(2σ[1]), sum(length, m.y[i])))
function NE.observation_moments_and_jacobians(m::RaggedOccasions, i::Integer, θ, σ, η)
    μ, v = NE.observation_moments(m, i, θ, σ, η)
    J = zeros(length(μ), length(η))
    J[:, 1] .= 1
    offset = 0
    for (j, o) in enumerate(m.y[i])
        J[offset+1:offset+length(o), 1+j] .= 1
        offset += length(o)
    end
    return μ, v, J, zeros(size(J))
end
NE.nsubjects(m::RaggedOccasions) = length(m.y)
NE.nlme_layout(m::RaggedOccasions) = RaggedNLMELayout(1, 1, [1, 1],
    [vcat(1, fill(2, length(occasions))) for occasions in m.y])
NE.mu_referencing(m::RaggedOccasions) =
    MuReferencing([1], [reshape(vcat(1.0, zeros(length(o))), :, 1) for o in m.y])

function occasion_residuals(m::RaggedOccasions, i, θ, η)
    return [m.y[i][j] .- (θ[1] + η[1] + η[1+j]) for j in eachindex(m.y[i])]
end

function NE.conditional_loglikelihood(m::RaggedOccasions, i::Integer, θ, σ, η)
    m.calls[] += 1
    r = reduce(vcat, occasion_residuals(m, i, θ, η))
    return -sum(abs2, r) / (2exp(2σ[1])) - length(r) * (log(2π) / 2 + σ[1])
end

function NE.conditional_loglikelihood_and_gradient(m::RaggedOccasions, i::Integer, θ, σ, η)
    rs = occasion_residuals(m, i, θ, η)
    v = exp(2σ[1])
    g = vcat(sum(sum, rs), [sum(r) for r in rs]) ./ v
    return NE.conditional_loglikelihood(m, i, θ, σ, η), g
end

function NE.conditional_loglikelihood_and_population_gradient(m::RaggedOccasions, i::Integer, θ, σ, η)
    ℓ, g = NE.conditional_loglikelihood_and_gradient(m, i, θ, σ, η)
    r = reduce(vcat, occasion_residuals(m, i, θ, η))
    return ℓ, [g[1]], [sum(abs2, r) / exp(2σ[1]) - length(r)]
end

function occasions_exact_subject(m::RaggedOccasions, i, x)
    θ, σ, ω = x[1], x[2], x[3:4]
    sizes = length.(m.y[i])
    n = sum(sizes)
    Z = zeros(n, length(sizes))
    offset = 0
    for (j, s) in enumerate(sizes)
        Z[offset+1:offset+s, j] .= 1
        offset += s
    end
    C = exp(2ω[1]) * ones(n, n) + exp(2ω[2]) * Z * Z' + exp(2σ) * I
    L = cholesky(Symmetric(C)).L
    z = L \ (reduce(vcat, m.y[i]) .- θ)
    return -(n * log(2π) + 2sum(log, diag(L)) + dot(z, z)) / 2
end

occasions_exact(m::RaggedOccasions, x) =
    sum(i -> occasions_exact_subject(m, i, x), 1:NE.nsubjects(m))

function occasions_model(; seed = 9, n = 8, gaussian = false)
    rng = Xoshiro(seed)
    y = map(1:n) do _
        a = 0.6 * randn(rng)
        map(1:rand(rng, 1:3)) do _
            b = 0.4 * randn(rng)
            2.0 + a + b .+ 0.25 .* randn(rng, rand(rng, 2:4))
        end
    end
    return RaggedOccasions(y; gaussian)
end

end
