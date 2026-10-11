"""
    ProtocolViolation(model_type, message)

Thrown by [`check_protocol`](@ref) when a model breaks the protocol contract.
"""
struct ProtocolViolation <: Exception
    model_type::Type
    message::String
end
Base.showerror(io::IO, e::ProtocolViolation) =
    print(io, "ProtocolViolation for ", e.model_type, ": ", e.message)

"""
    check_protocol(model, θ, σ, ω; subjects = 1:nsubjects(model), η = nothing,
                   reference_backend = nothing, reference_hessian_backend = reference_backend,
                   population_gradient = false, rtol = 1e-8) -> true

Check that `model` implements the protocol consistently at the given parameters,
and throw a [`ProtocolViolation`](@ref) naming the first violation otherwise.

Checked for the batch `subjects` at random effects `η` (zeros by default; a
`neta × length(subjects)` matrix, one vector for all subjects, or one vector per
subject):

- the layout matches the lengths of θ, σ and ω, and `evaluation_style` is an
  [`EvaluationStyle`](@ref);
- the batched log-densities are finite reals and equal the per-subject ones;
- the batched gradients equal the per-subject gradients, and the values returned
  alongside derivatives equal the plain values;
- subjects are separable: changing one subject's η leaves every other subject's
  log-density unchanged;
- when [`provides_eta_hessian`](@ref): the Hessian blocks are square, symmetric and
  equal the per-subject Hessians;
- a declared [`mu_referencing`](@ref) trait has one design per subject of the right
  size, and the density is invariant under the mu-referencing shift;
- a declared [`observation_model`](@ref) reproduces the conditional log-density.
- a declared Gaussian model's batched moments and first-order η Jacobians have
  the promised per-subject shape and agree with its plain moments.

With `reference_backend` (an AD backend), the model's η gradients are also compared
against that backend, and its Hessian blocks (when provided) against
`reference_hessian_backend` (for Enzyme, a forward-over-reverse
`DifferentiationInterface.SecondOrder`). With
`population_gradient = true`, the optional θ/σ gradient is checked as well (against
`reference_backend` when given).
"""
function check_protocol(model, θ::AbstractVector, σ::AbstractVector, ω::AbstractVector;
        subjects = nothing, η = nothing, reference_backend = nothing,
        reference_hessian_backend = reference_backend,
        population_gradient::Bool = false, rtol::Real = 1.0e-8)
    violation(msg) = throw(ProtocolViolation(typeof(model), msg))
    n = nsubjects(model)
    n isa Integer && n >= 1 || violation("nsubjects returned $(repr(n)); expected a positive integer")
    layout = nlme_layout(model)
    layout isa NLMELayout || violation("nlme_layout returned a $(typeof(layout)); expected an NLMELayout")
    evaluation_style(model) isa EvaluationStyle ||
        violation("evaluation_style returned $(repr(evaluation_style(model))); expected an EvaluationStyle")
    length(θ) == layout.ntheta || violation("θ has length $(length(θ)), the layout declares ntheta = $(layout.ntheta)")
    length(σ) == layout.nsigma || violation("σ has length $(length(σ)), the layout declares nsigma = $(layout.nsigma)")
    length(ω) == nomega(layout) || violation("ω has length $(length(ω)), the layout requires nomega = $(nomega(layout))")
    subjects = subjects === nothing ? collect(1:n) : collect(Int, subjects)
    all(i -> 1 <= i <= n, subjects) || violation("subjects $subjects are outside 1:$n")
    m = neta(layout)
    H = _check_eta(violation, η, m, length(subjects))
    mr = mu_referencing(model)
    _check_mu_layout(violation, mr, n, m)
    reference = reference_backend === nothing ? nothing :
        (gradient = _checked_backend(model, reference_backend, :check_protocol),
            hessian = reference_hessian_backend === nothing ? nothing :
                _checked_backend(model, reference_hessian_backend, :check_protocol))
    agrees(a, b) = isapprox(a, b; rtol = rtol, atol = rtol * (1 + maximum(abs, a; init = 0.0)))

    ℓb = conditional_loglikelihoods(model, subjects, θ, σ, H)
    length(ℓb) == length(subjects) ||
        violation("conditional_loglikelihoods returned $(length(ℓb)) values for $(length(subjects)) subjects")
    for (k, i) in enumerate(subjects)
        ℓb[k] isa Real && isfinite(ℓb[k]) ||
            violation("the conditional log-density of subject $i is $(repr(ℓb[k])) at the checked point; expected a finite real")
    end
    ℓg, G = conditional_loglikelihoods_and_gradients(model, subjects, θ, σ, H)
    agrees(ℓg, ℓb) || violation("conditional_loglikelihoods_and_gradients values $ℓg ≠ conditional_loglikelihoods $ℓb")
    size(G) == size(H) || violation("batched η gradients have size $(size(G)); expected $(size(H))")
    _check_separability(violation, agrees, model, subjects, θ, σ, H, ℓb)
    blocks = provides_eta_hessian(model) ? conditional_loglikelihood_hessian_blocks(model, subjects, θ, σ, H) : nothing
    _check_block_shapes(violation, agrees, blocks, subjects, m)

    for (k, i) in enumerate(subjects)
        ηi = H[:, k]
        ℓ = conditional_loglikelihood(model, i, θ, σ, ηi)
        agrees(ℓ, ℓb[k]) || violation("subject $i: per-subject log-density $ℓ ≠ batched $(ℓb[k])")
        ℓi, g = conditional_loglikelihood_and_gradient(model, i, θ, σ, ηi)
        agrees(ℓi, ℓ) || violation("subject $i: conditional_loglikelihood_and_gradient value $ℓi ≠ $ℓ")
        agrees(g, G[:, k]) || violation("subject $i: per-subject η gradient $g ≠ batched $(G[:, k])")
        _check_subject_hessian(violation, agrees, blocks, model, k, i, θ, σ, ηi, ℓ, g)
        _check_reference(violation, agrees, reference, model, i, θ, σ, ηi, g, blocks === nothing ? nothing : blocks[k])
        _check_mu_invariance(violation, agrees, mr, model, i, θ, σ, ηi, ℓ)
        _check_observation_model(violation, agrees, observation_model(model), model, i, θ, σ, ηi, ℓ)
        population_gradient &&
            _check_population_gradient(violation, agrees, reference, model, i, θ, σ, ηi, ℓ, layout)
    end
    observation_model(model) isa GaussianObservations &&
        _check_observation_batch(violation, agrees, model, subjects, θ, σ, H, m)
    return true
end

_check_eta(violation, ::Nothing, m, n) = zeros(m, n)
function _check_eta(violation, η::AbstractMatrix{<:Real}, m, n)
    size(η) == (m, n) || violation("η has size $(size(η)); expected ($m, $n)")
    return Matrix{Float64}(η)
end
function _check_eta(violation, η::AbstractVector{<:Real}, m, n)
    length(η) == m || violation("η has length $(length(η)); neta = $m")
    return repeat(η, 1, n)
end
function _check_eta(violation, η::AbstractVector{<:AbstractVector}, m, n)
    length(η) == n || violation("η holds $(length(η)) vectors for $n subjects")
    all(v -> length(v) == m, η) || violation("some η vector does not have length neta = $m")
    return reduce(hcat, η)
end

# Conditional independence: one subject's η moves only that subject's density.
function _check_separability(violation, agrees, model, subjects, θ, σ, H, ℓb)
    length(subjects) >= 2 || return nothing
    moved = copy(H)
    moved[:, 1] .+= 0.1
    ℓ1 = conditional_loglikelihoods(model, subjects, θ, σ, moved)
    agrees(ℓ1[2:end], ℓb[2:end]) || violation(
        "changing subject $(subjects[1])'s η changed other subjects' log-densities; subjects must be conditionally independent")
    others = copy(H)
    others[:, 2:end] .+= 0.1
    ℓ2 = conditional_loglikelihoods(model, subjects, θ, σ, others)
    agrees(ℓ2[1], ℓb[1]) || violation(
        "changing the other subjects' η changed subject $(subjects[1])'s log-density; subjects must be conditionally independent")
    return nothing
end

_check_block_shapes(violation, agrees, ::Nothing, subjects, m) = nothing
function _check_block_shapes(violation, agrees, blocks, subjects, m)
    length(blocks) == length(subjects) ||
        violation("conditional_loglikelihood_hessian_blocks returned $(length(blocks)) blocks for $(length(subjects)) subjects")
    for (k, B) in enumerate(blocks)
        size(B) == (m, m) || violation("subject $(subjects[k]): η Hessian block has size $(size(B)); expected ($m, $m)")
        agrees(B, B') || violation("subject $(subjects[k]): η Hessian block is not symmetric")
    end
    return nothing
end

_check_subject_hessian(violation, agrees, ::Nothing, model, k, i, θ, σ, η, ℓ, g) = nothing
function _check_subject_hessian(violation, agrees, blocks, model, k, i, θ, σ, η, ℓ, g)
    ℓh, gh, Hi = conditional_loglikelihood_gradient_and_hessian(model, i, θ, σ, η)
    agrees(ℓh, ℓ) || violation("subject $i: conditional_loglikelihood_gradient_and_hessian value $ℓh ≠ $ℓ")
    agrees(gh, g) || violation("subject $i: the gradients of the two derivative methods disagree: $gh vs $g")
    agrees(Hi, blocks[k]) || violation("subject $i: per-subject η Hessian disagrees with its batched block")
    return nothing
end

_check_reference(violation, agrees, ::Nothing, model, i, θ, σ, η, g, block) = nothing
function _check_reference(violation, agrees, reference, model, i, θ, σ, η, g, block)
    gref = DI.gradient(_eta_objective, reference.gradient, η, _eta_contexts(model, i, θ, σ)...)
    agrees(g, gref) || violation("subject $i: η gradient $g disagrees with the reference backend's $gref")
    (block === nothing || reference.hessian === nothing) && return nothing
    Href = DI.hessian(_eta_objective, reference.hessian, η, _eta_contexts(model, i, θ, σ)...)
    agrees(block, Href) || violation("subject $i: η Hessian disagrees with the reference backend's")
    return nothing
end

_check_mu_layout(violation, ::Nothing, n, m) = nothing
function _check_mu_layout(violation, mr::MuReferencing, n, m)
    length(mr.design) == n ||
        violation("mu_referencing declares $(length(mr.design)) designs for $n subjects")
    for (i, X) in enumerate(mr.design)
        size(X, 1) == m || violation("mu_referencing design[$i] has $(size(X, 1)) rows; neta = $m")
    end
    return nothing
end

_check_mu_invariance(violation, agrees, ::Nothing, model, i, θ, σ, η, ℓ) = nothing
function _check_mu_invariance(violation, agrees, mr::MuReferencing, model, i, θ, σ, η, ℓ)
    k = length(mr.theta_indices)
    k == 0 && return nothing
    δ = [0.1 * (-1)^j * j / k for j in 1:k]
    θs = collect(float.(θ))
    θs[mr.theta_indices] .+= δ
    ηs = η - mr.design[i] * δ
    ℓs = conditional_loglikelihood(model, i, θs, σ, ηs)
    agrees(ℓs, ℓ) || violation(
        "subject $i: the conditional log-density changes from $ℓ to $ℓs under the declared mu-referencing " *
        "shift (θ[$(mr.theta_indices)] += δ, η -= X_i δ), so it does not depend on them only through X_i θ + η")
    return nothing
end

_check_observation_model(violation, agrees, ::Nothing, model, i, θ, σ, η, ℓ) = nothing
function _check_observation_model(violation, agrees, ::GaussianObservations, model, i, θ, σ, η, ℓ)
    y = observations(model, i)
    μ, v = observation_moments(model, i, θ, σ, η)
    length(μ) == length(y) == length(v) || violation(
        "subject $i: observation_moments returned $(length(μ)) means and $(length(v)) variances for $(length(y)) observations")
    all(>(0), v) || violation("subject $i: observation_moments returned non-positive variances")
    ℓobs = sum(j -> -(log(2π * v[j]) + (y[j] - μ[j])^2 / v[j]) / 2, eachindex(y); init = zero(ℓ))
    agrees(ℓobs, ℓ) || violation(
        "subject $i: the declared Gaussian observation model gives log-density $ℓobs, conditional_loglikelihood gives $ℓ")
    return nothing
end

function _check_observation_batch(violation, agrees, model, subjects, θ, σ, H, m)
    ys = observations(model, subjects)
    parts = observation_moments_and_jacobians(model, subjects, θ, σ, H)
    length(ys) == length(subjects) || violation("batched observations returned $(length(ys)) entries for $(length(subjects)) subjects")
    length(parts) == length(subjects) || violation("batched observation moments and Jacobians returned $(length(parts)) entries for $(length(subjects)) subjects")
    for (k, i) in enumerate(subjects)
        y = observations(model, i)
        agrees(ys[k], y) || violation("subject $i: batched observations disagree with per-subject observations")
        μ, v, Jμ, Jv = parts[k]
        μi, vi = observation_moments(model, i, θ, σ, view(H, :, k))
        agrees(μ, μi) && agrees(v, vi) || violation("subject $i: batched observation moments disagree with per-subject moments")
        size(Jμ) == size(Jv) == (length(y), m) || violation(
            "subject $i: observation Jacobians have sizes $(size(Jμ)) and $(size(Jv)); expected ($(length(y)), $m)")
        all(isfinite, Jμ) && all(isfinite, Jv) || violation("subject $i: observation Jacobians contain non-finite entries")
    end
    return nothing
end

function _check_population_gradient(violation, agrees, reference, model, i, θ, σ, η, ℓ, layout)
    ℓp, gθ, gσ = conditional_loglikelihood_and_population_gradient(model, i, θ, σ, η)
    agrees(ℓp, ℓ) || violation("subject $i: conditional_loglikelihood_and_population_gradient value $ℓp ≠ $ℓ")
    length(gθ) == layout.ntheta || violation("subject $i: θ gradient has length $(length(gθ))")
    length(gσ) == layout.nsigma || violation("subject $i: σ gradient has length $(length(gσ))")
    reference === nothing && return nothing
    x = vcat(θ, σ)
    gref = DI.gradient(_population_objective, reference.gradient, x,
        DI.Constant(model), DI.Constant(i), DI.Constant(η), DI.Constant(length(θ)))
    agrees(vcat(gθ, gσ), gref) || violation("subject $i: θ/σ gradient disagrees with the reference backend's")
    return nothing
end
