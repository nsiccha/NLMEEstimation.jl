"""
    LouisInformation(; draws = 1000, burnin = 200, kernel = WarmupHMCNUTS())

[`covariance_step`](@ref)'s information matrices by Louis's missing-information
identity (Louis 1982), from posterior draws of the random effects at the point:
SAEM's stochastic-approximation Fisher information (Kuhn & Lavielle 2005;
Monolix's default, nlmixr2's SAEM `"sa"`). For subject `i`, with ψ the estimated
coordinates and draws `η_i ~ p(η_i | y_i, ψ)`,

    -∂²ψ log p(y_i | ψ) = E[-∂²ψ log p(y_i, η_i | ψ)] - Var[∂ψ log p(y_i, η_i | ψ)]
     ∂ψ log p(y_i | ψ)  = E[∂ψ log p(y_i, η_i | ψ)]

so `R` (the summed first line) and `S` (the outer products of the second) need
neither a marginal likelihood nor the curvature in η. The complete-data log
density is taken in the mu-referenced coordinates `φ = η + Xθ_μ`, in which the
model depends on the mu-referenced θ only through the Gaussian prior. The
derivatives in those θ and in ω are closed form. Those in the other θ and in σ
come from [`conditional_loglikelihood_and_population_gradient`](@ref) (one call
per subject and draw), and their curvature from central differences of the
summed population gradient (two batched calls per coordinate and draw, with
[`covariance_step`](@ref)'s `step`).

`kernel`, a [`WarmupHMCNUTS`](@ref), draws each subject's η: Pathfinder from the
starting η (a fit's chain states or modes), `burnin` discarded transitions that
adapt the kernel, then `draws` kept ones under the kernel frozen. `R` and `S`
carry Monte Carlo error that falls as `1/√draws`. It applies to any likelihood
and to both layouts, asks the model for no η Hessian, and keeps no `n × n`
matrix per subject. It is not a marginal likelihood:
[`marginal_loglikelihood`](@ref) refuses it.
"""
Base.@kwdef struct LouisInformation{K <: WarmupHMCNUTS}
    draws::Int = 1000
    burnin::Int = 200
    kernel::K = WarmupHMCNUTS()
end

function _check_approximation(a::LouisInformation)
    a.draws >= 2 || throw(ArgumentError("LouisInformation needs at least 2 draws"))
    a.burnin >= 0 || throw(ArgumentError("LouisInformation burnin must be nonnegative"))
    _saem_check_nuts(a.kernel)
    return nothing
end

# What a covariance step with Louis's identity starts from; the draws are taken
# in `_covariance_matrices`, where the difference steps and the estimated
# coordinates are known, and are never stored (a subject with thousands of
# nested effects would keep `draws` copies of them).
struct _PreparedLouis{M, A <: LouisInformation, L, E, R}
    model::M
    approximation::A
    layout::L
    subjects::Vector{Int}
    η0::E
    rng::R
end

function _prepare_marginal(approximation::LouisInformation, model, θ, σ, ω, subjects, η0,
        rng, progress = nothing)
    _check_approximation(approximation)
    return _PreparedLouis(model, approximation, nlme_layout(model),
        _covariance_subjects(model, subjects), η0, rng)
end

function _covariance_subjects(model, subjects)
    n = nsubjects(model)
    ids = subjects === nothing ? collect(1:n) : collect(Int, subjects)
    !isempty(ids) && allunique(ids) && all(i -> 1 <= i <= n, ids) ||
        throw(ArgumentError("subjects must be distinct valid model subject indices"))
    return ids
end

_marginal_subjects(prepared::_PreparedLouis) = prepared.subjects
_marginal_terms(::_PreparedLouis, θ, σ, ω, progress) = throw(ArgumentError(
    "LouisInformation gives covariance_step's information matrices, not a marginal " *
    "likelihood; use ImportanceSampling or AdaptiveGaussHermite"))
_marginal_calls(::_PreparedLouis) = missing

_covariance_progress(prepared, method, p) =
    (method == :s ? 2p + 1 : 2p^2 + 1, "Marginal likelihoods")
_covariance_progress(prepared::_PreparedLouis, method, p) =
    (prepared.approximation.burnin + prepared.approximation.draws, "Posterior draws")

# The starting η: a fit's chain states or EBEs, or zeros.
_starting_batch(layout::NLMELayout, η0, subjects) =
    Matrix{Float64}(_eta_start(η0, subjects, neta(layout)))
_starting_batch(layout::RaggedNLMELayout, η0, subjects) = _ragged_start(η0, layout, subjects)

# Draw, accumulate, and assemble R and S in the full estimated coordinates
# ŷ = [θ; σ; free ω], then keep the `free` ones.
function _covariance_matrices(prepared::_PreparedLouis, split, ŷ, free, step, method, lp0,
        subject_ids, model, counter)
    a, layout, ids = prepared.approximation, prepared.layout, prepared.subjects
    θ, σ, ω = split(ŷ)
    θ, σ, ω = Float64.(θ), Float64.(σ), Float64.(ω)
    nθ, nσ, p, n = layout.ntheta, layout.nsigma, length(ŷ), length(ids)
    constraints = _omega_constraints(model)
    indices, design = _saem_mu_trait(model, ids, layout)
    # Coordinates of the conditional likelihood (θ outside the mu-referenced ones,
    # and σ) and of the Gaussian prior (mu-referenced θ, free ω).
    conditional = vcat(setdiff(1:nθ, indices), nθ .+ (1:nσ))
    gaussian = vcat(indices, (nθ + nσ + 1):p)
    h = step .* max.(1.0, abs.(ŷ))
    H = _starting_batch(layout, prepared.η0, ids)
    prior = _re_prior(layout, ω)
    style = saem_population_gradient_style(model)
    sums, squares = zeros(p, n), zeros(p, p, n)
    curvature = zeros(p, p)
    mean_phi, second_phi = _saem_statistics(layout, ids, H)
    states = _saem_nuts_start!(prepared.rng, a.kernel, model, ids, θ, σ, prior, H)
    for round in 1:(a.burnin + a.draws)
        throw_if_interrupted(counter)
        _saem_nuts_round!(states, model, ids, θ, σ, prior, H; adapt = round <= a.burnin)
        counter === nothing || update_progress!(counter, round)
        round <= a.burnin && continue
        _louis_add_moments!(mean_phi, second_phi, _saem_phi(H, θ, indices, design))
        for (k, i) in enumerate(ids)
            s = _louis_score(model, layout, i, θ, σ, ω, prior, constraints, indices,
                design[k], _column(H, k))
            sums[:, k] .+= s
            squares[:, :, k] .+= s * s'
        end
        method == :s && continue
        for j in conditional
            gradient(sign) = last(_saem_population_value_gradient(style, model, ids,
                split(_louis_shift(ŷ, j, sign * h[j]))[1:2]..., H))
            column = (gradient(1) - gradient(-1)) / (2h[j])
            curvature[conditional, j] .-= column[conditional]
        end
    end
    K = a.draws
    curvature ./= K
    _louis_scale_moments!(mean_phi, second_phi, 1 / K)
    # The Gaussian prior's expected curvature is linear in the draws' moments, so
    # it is the curvature at the mean moments.
    prior_gradient(y) = _louis_gaussian_gradient(layout, split(y)..., constraints, indices,
        design, mean_phi, second_phi, ids)
    for j in gaussian
        column = (prior_gradient(_louis_shift(ŷ, j, h[j])) -
            prior_gradient(_louis_shift(ŷ, j, -h[j]))) / (2h[j])
        curvature[gaussian, j] .-= column[gaussian]
    end
    scores = sums ./ K
    variance = sum(k -> (view(squares, :, :, k) - K * scores[:, k] * scores[:, k]') / (K - 1), 1:n)
    R = (curvature + curvature') / 2 - variance
    gradient = vec(sum(scores; dims = 2))
    if lp0 !== nothing
        lp_gradient, lp_curvature = _louis_prior_derivatives(model, split, ŷ, h)
        R += lp_curvature
        gradient += lp_gradient
    end
    G = scores[free, :]
    S = G * G'
    return method == :s ? nothing : R[free, free], S, G, gradient[free], h[free],
        a.burnin + a.draws
end

_louis_shift(y, j, δ) = (z = copy(y); z[j] += δ; z)

# ∂ψ log p(y_i, η_i | ψ) for one draw, in the full estimated coordinates: the
# conditional's gradient in the non-mu-referenced θ and σ, and the Gaussian
# prior's in the mu-referenced θ and the free ω (with η = φ - Xθ_μ held at its φ).
# With every θ mu-referenced and no σ, the conditional takes no part, and the model
# need not provide its population gradient (SAEM does not need it either).
function _louis_score(model, layout, i, θ, σ, ω, prior, constraints, indices, X, η)
    s = _louis_prior_score(layout, θ, σ, ω, prior, constraints, indices, X, η, i)
    nθ = layout.ntheta
    length(indices) == nθ && layout.nsigma == 0 && return s
    _, gθ, gσ = conditional_loglikelihood_and_population_gradient(model, i, θ, σ, η)
    gθ = copy(gθ)
    gθ[indices] .= 0
    s[1:nθ] .+= gθ
    s[nθ .+ (1:layout.nsigma)] .+= gσ
    return s
end

# One draw's Gaussian-prior score: from the draw's own moments on an equal-size
# layout (a few random effects), and block by block on a ragged one, where the
# per-family sums Σ η_b η_bᵀ and Σ X_bᵀ Ω_f⁻¹ η_b are all `_ragged_gaussian_*`
# would read from the subject's moments, without forming them.
function _louis_prior_score(layout::NLMELayout, θ, σ, ω, prior, constraints, indices, X, η, i)
    H = _single_eta_batch(layout, η)
    mean_phi, second_phi = _saem_statistics(layout, [i], H)
    _louis_add_moments!(mean_phi, second_phi, _saem_phi(H, θ, indices, [X]))
    return _louis_gaussian_gradient(layout, θ, σ, ω, constraints, indices, [X],
        mean_phi, second_phi, [i])
end

function _louis_prior_score(layout::RaggedNLMELayout, θ, σ, ω, prior, constraints, indices,
        X, η, i)
    sums = [zeros(k, k) for k in layout.family_sizes]
    counts = zeros(Int, length(layout.family_sizes))
    gμ = zeros(length(indices))
    for (family, r) in _ragged_subject_blocks(layout, i)
        ηr = view(η, r)
        mul!(sums[family], ηr, ηr', 1.0, 1.0)
        counts[family] += 1
        isempty(indices) || mul!(gμ, view(X, r, :)', prior.precisions[family] * ηr, 1.0, 1.0)
    end
    gω = zeros(length(ω))
    for (f, range) in enumerate(_ragged_family_ranges(layout))
        counts[f] == 0 && continue
        gω[range] = last(_gaussian_family_value_gradient(prior.cholesky_factors[f],
            prior.precisions[f], sums[f], counts[f]))
    end
    g = zeros(layout.ntheta + layout.nsigma)
    g[indices] = gμ
    return vcat(g, _free_omega_gradient(constraints, ω, gω))
end

# Sums of the draws' φ moments, in place (then scaled to their means): per subject
# the vector φ and, per block of a ragged subject (the whole η of an equal-size
# one), φ_b φ_bᵀ.
function _louis_add_moments!(mean_phi::AbstractMatrix, second_phi, Φ)
    mean_phi .+= Φ
    for k in axes(Φ, 2)
        φ = view(Φ, :, k)
        mul!(view(second_phi, :, :, k), φ, φ', 1.0, 1.0)
    end
    return nothing
end
function _louis_add_moments!(mean_phi::AbstractVector, second_phi, Φ)
    for k in eachindex(Φ)
        mean_phi[k] .+= Φ[k]
        for (b, r) in enumerate(_moment_block_ranges(second_phi[k]))
            φ = view(Φ[k], r)
            mul!(second_phi[k][b], φ, φ', 1.0, 1.0)
        end
    end
    return nothing
end
_louis_scale_moments!(mean_phi::AbstractMatrix, second_phi, c) =
    (mean_phi .*= c; second_phi .*= c; nothing)
function _louis_scale_moments!(mean_phi::AbstractVector, second_phi, c)
    foreach(m -> m .*= c, mean_phi)
    foreach(blocks -> foreach(S -> S .*= c, blocks), second_phi)
    return nothing
end

# The Gaussian random-effect prior's gradient in the full estimated coordinates
# (nonzero in the mu-referenced θ and the free ω) at the given φ moments.
function _louis_gaussian_gradient(layout, θ, σ, ω, constraints, indices, design,
        mean_phi, second_phi, subjects)
    _, gμ, gω = _saem_gaussian_map_value_gradient(layout, θ, ω, indices, design,
        mean_phi, second_phi, subjects)
    g = zeros(layout.ntheta + layout.nsigma)
    g[indices] = gμ
    return vcat(g, _free_omega_gradient(constraints, ω, gω))
end

# The population log prior's gradient and minus its Hessian, by central
# differences of its values (it costs no model evaluations).
function _louis_prior_derivatives(model, split, ŷ, h)
    lp(y) = Float64(population_logprior(model, split(y)...))
    p = length(ŷ)
    center = lp(ŷ)
    gradient = [(lp(_louis_shift(ŷ, j, h[j])) - lp(_louis_shift(ŷ, j, -h[j]))) / (2h[j])
        for j in 1:p]
    curvature = zeros(p, p)
    for j in 1:p
        curvature[j, j] = -(lp(_louis_shift(ŷ, j, h[j])) - 2center +
            lp(_louis_shift(ŷ, j, -h[j]))) / h[j]^2
        for k in 1:j-1
            corner(a, b) = lp(_louis_shift(_louis_shift(ŷ, j, a * h[j]), k, b * h[k]))
            curvature[j, k] = curvature[k, j] = -(corner(1, 1) - corner(1, -1) -
                corner(-1, 1) + corner(-1, -1)) / (4h[j] * h[k])
        end
    end
    return gradient, curvature
end
