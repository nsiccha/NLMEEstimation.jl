"""
    ImportanceSampling(; draws = 2000, df = 4, scale = 1.0)

Importance-sampling estimate of each subject's marginal log-likelihood
`log ∫ p(y_i | θ, σ, η) N(η; 0, Ω) dη`. The proposal for subject `i` is a
multivariate Student t with `df` degrees of freedom (a positive integer, or
`Inf` for a Gaussian), centred at the subject's conditional mode, with scale
matrix `scale²` times the inverse curvature of the complete-data log-density
there. The curvature comes from the η Hessian blocks when the model provides
them, and otherwise from central differences of batched η gradients. Each draw
costs one batched conditional evaluation over all subjects. The estimate
reports its Monte Carlo standard error and per-subject effective sample sizes.
"""
Base.@kwdef struct ImportanceSampling
    draws::Int = 2000
    df::Float64 = 4.0
    scale::Float64 = 1.0
end

"""
    AdaptiveGaussHermite(; nodes = 5, max_points = 100_000)

Adaptive Gauss–Hermite quadrature of each subject's marginal log-likelihood:
`nodes` points per random-effect dimension, as a tensor product, centred at
the subject's conditional mode and scaled by its inverse curvature (computed as
for [`ImportanceSampling`](@ref)). `nodes = 1` is the Laplace approximation. A
subject with `d` random effects costs `nodes^d` batched evaluation points; a
subject needing more than `max_points` is refused.
"""
Base.@kwdef struct AdaptiveGaussHermite
    nodes::Int = 5
    max_points::Int = 100_000
end

function _check_approximation(a::ImportanceSampling)
    a.draws >= 2 || throw(ArgumentError("ImportanceSampling needs at least 2 draws"))
    a.df > 0 && (isinf(a.df) || isinteger(a.df)) || throw(ArgumentError(
        "ImportanceSampling df must be a positive integer or Inf, got $(a.df)"))
    isfinite(a.scale) && a.scale > 0 || throw(ArgumentError(
        "ImportanceSampling scale must be finite and positive"))
    return nothing
end

function _check_approximation(a::AdaptiveGaussHermite)
    a.nodes >= 1 || throw(ArgumentError("AdaptiveGaussHermite needs at least one node"))
    a.max_points >= 1 || throw(ArgumentError("max_points must be positive"))
    return nothing
end

# Quadrature or sampling points for each subject, held fixed in the
# mu-referenced coordinates φ = η + X θ_μ. `offset[k][j]` is the log weight of
# point j minus its proposal log density, so each subject's marginal is
# `logsumexp(conditional + prior + offset)`.
struct _PreparedPoints{M, A, L}
    model::M
    approximation::A
    layout::L
    subjects::Vector{Int}
    indices::Vector{Int}
    design::Vector{Matrix{Float64}}
    phi::Vector{Matrix{Float64}}
    offset::Vector{Vector{Float64}}
    reference::Vector{Float64}
    cache::Dict{Vector{Float64}, Vector{Vector{Float64}}}
    calls::Base.RefValue{Int}
end

_marginal_subjects(prepared::_PreparedPoints) = prepared.subjects

# Adaptive Gauss–Hermite quadrature adapts its points to the integrand at every
# point (θ, σ, ω) it is evaluated at: the subject modes and their curvature are
# found again there, warm-started from the estimate's modes with the tightened
# tolerance `FOCEApproximation` uses, so the terms are smooth in the coordinates.
# Points held fixed at the estimate would not adapt: with one node (Laplace) the
# covariance step's R would then be the complete-data information at the
# estimate's modes, larger than the observed information and without its cross
# terms. `points` are the estimate's own, which `marginal_loglikelihood` reports.
struct _PreparedAdaptive{M, A <: AdaptiveGaussHermite, E, P <: _PreparedPoints}
    model::M
    approximation::A
    center::E
    points::P
    calls::Base.RefValue{Int}
end

_marginal_subjects(prepared::_PreparedAdaptive) = prepared.points.subjects

function _prepare_marginal(approximation::ImportanceSampling, model, θ, σ, ω, subjects,
        η0, rng, progress = nothing)
    _check_approximation(approximation)
    ebe = _run_empirical_bayes(model, θ, σ, ω, subjects, η0, default_ebe_method(model),
        EBEOptions(), true, progress, false)
    return _points_at_modes(approximation, model, θ, σ, ω, ebe, rng)
end

function _prepare_marginal(approximation::AdaptiveGaussHermite, model, θ, σ, ω, subjects,
        η0, rng, progress = nothing)
    _check_approximation(approximation)
    ebe = _run_empirical_bayes(model, θ, σ, ω, subjects, η0, default_ebe_method(model),
        EBEOptions(), true, progress, false)
    return _PreparedAdaptive(model, approximation, ebe,
        _points_at_modes(approximation, model, θ, σ, ω, ebe, rng), Ref(0))
end

# The approximation's points around each subject's conditional mode in `ebe`,
# scaled by the inverse curvature there.
function _points_at_modes(approximation, model, θ, σ, ω, ebe, rng)
    layout = nlme_layout(model)
    ids = collect(Int, ebe.subjects)
    indices, design = _saem_mu_trait(model, ids, layout)
    modes = [Vector{Float64}(_eta_column(ebe.modes, k)) for k in eachindex(ids)]
    hessians = ebe.hessians === nothing ?
        _difference_eta_hessians(model, layout, ids, θ, σ, re_prior(model, ω), modes) :
        [convert(AbstractMatrix{Float64}, H) for H in ebe.hessians]
    θμ = Float64.(θ[indices])
    phi = Matrix{Float64}[]
    offset = Vector{Float64}[]
    for (k, i) in enumerate(ids)
        F = cholesky(Symmetric(-hessians[k]); check = false)
        issuccess(F) || throw(DomainError(hessians[k],
            "subject $i: the complete-data curvature at its conditional mode is not positive definite"))
        scale, logdet_A = _curvature_scale(F)
        Z, w = _marginal_points(approximation, rng, length(modes[k]), logdet_A)
        push!(phi, (modes[k] .+ design[k] * θμ) .+ scale(Z))
        push!(offset, w)
    end
    nonmu = setdiff(1:layout.ntheta, indices)
    reference = Float64.(vcat(θ[nonmu], σ))
    return _PreparedPoints(model, approximation, layout, ids, indices, design, phi,
        offset, reference, Dict{Vector{Float64}, Vector{Vector{Float64}}}(), Ref(0))
end

# The map from standardized points Z to proposal offsets AZ, with AAᵀ = (-H)⁻¹ for
# the Cholesky factor F of -H, and log|det A|. A sparse (CHOLMOD) factor with
# P(-H)Pᵀ = LLᵀ applies A = PᵀL⁻ᵀ by one triangular solve, never forming a dense
# matrix.
_curvature_scale(F::Cholesky) = (Z -> inv(UpperTriangular(F.U)) * Z, -sum(log, diag(F.U)))
_curvature_scale(F::SparseArrays.CHOLMOD.Factor) = (Z -> F.UP \ Z, -logdet(F) / 2)

# Standardized points Z (d × K, before the factor A) and their offsets, given
# log|det A|.
function _marginal_points(a::ImportanceSampling, rng, d, logdet_A)
    K = a.draws
    Z = randn(rng, d, K)
    offset = zeros(K)
    ν = a.df
    for j in 1:K
        z = view(Z, :, j)
        if isinf(ν)
            logq = -(d * log(2π) + sum(abs2, z)) / 2
        else
            s = sqrt(ν / sum(abs2, randn(rng, Int(ν))))
            z .*= s
            logq = _loggamma_half(Int(ν) + d) - _loggamma_half(Int(ν)) -
                d * log(ν * π) / 2 - (ν + d) * log1p(sum(abs2, z) / ν) / 2
        end
        offset[j] = -log(K) - (logq - logdet_A - d * log(a.scale))
    end
    return a.scale .* Z, offset
end

function _marginal_points(a::AdaptiveGaussHermite, rng, d, logdet_A)
    # In BigInt: nodes^d overflows Int for large d (2^64 wraps to 0 points).
    big(a.nodes)^d <= a.max_points || throw(ArgumentError(
        "AdaptiveGaussHermite with $(a.nodes) nodes needs $(a.nodes)^$d points for a subject with $d random effects (max_points = $(a.max_points)); use fewer nodes or ImportanceSampling"))
    npoints = a.nodes^d
    x, w = _gauss_hermite(a.nodes)
    Z = zeros(d, npoints)
    offset = zeros(npoints)
    # Point j's node per dimension, the first dimension varying fastest; no tuple of
    # length d, so a single Laplace point serves thousands of random effects.
    nodes(j) = a.nodes == 1 ? ones(Int, d) : digits(j - 1; base = a.nodes, pad = d) .+ 1
    for j in 1:npoints
        index = nodes(j)
        z = [x[index[c]] for c in 1:d]
        Z[:, j] = sqrt(2) .* z
        offset[j] = sum(c -> log(w[index[c]]), 1:d; init = 0.0) + sum(abs2, z) +
            d * log(2) / 2 + logdet_A
    end
    return Z, offset
end

# Gauss–Hermite nodes and weights for ∫ f(x) exp(-x²) dx (Golub–Welsch).
function _gauss_hermite(n::Integer)
    E = eigen(SymTridiagonal(zeros(n), [sqrt(k / 2) for k in 1:n-1]))
    return E.values, sqrt(π) .* E.vectors[1, :] .^ 2
end

# log Γ(k/2) for a positive integer k.
function _loggamma_half(k::Integer)
    k > 0 || throw(DomainError(k, "log Γ(k/2) needs a positive integer k"))
    iseven(k) && return sum(log, 1:(k ÷ 2 - 1); init = 0.0)
    m = (k - 1) ÷ 2
    return sum(log, 1:2m; init = 0.0) - m * log(4) - sum(log, 1:m; init = 0.0) + log(π) / 2
end

# η Hessian blocks of the complete-data log-density at `points` (one vector per
# subject) from central differences of batched gradients, with steps
# h = 1e-4·max(1, |η_c|). Component c moves in every subject that has it at once,
# which conditional independence makes one batched call per direction and side.
function _difference_eta_hessians(model, layout, ids, θ, σ, prior, points)
    dims = length.(points)
    hessians = [zeros(d, d) for d in dims]
    for c in 1:maximum(dims)
        ks = findall(>=(c), dims)
        h = [1.0e-4 * max(1.0, abs(points[k][c])) for k in ks]
        shifted(sign) = _eta_batch(layout, [begin
            η = copy(points[k])
            η[c] += sign * h[j]
            η
        end for (j, k) in enumerate(ks)])
        _, _, Gup = _complete_gradients(model, ids[ks], θ, σ, prior, shifted(1))
        _, _, Gdown = _complete_gradients(model, ids[ks], θ, σ, prior, shifted(-1))
        for (j, k) in enumerate(ks)
            hessians[k][:, c] = (_eta_column(Gup, j) - _eta_column(Gdown, j)) / (2 * h[j])
        end
    end
    return [(H + H') / 2 for H in hessians]
end

_eta_batch(::NLMELayout, columns) = reduce(hcat, columns)
_eta_batch(::RaggedNLMELayout, columns) = columns

# Conditional log-likelihoods at every point. By mu-referencing they depend on
# (θ, σ) only through the non-mu-referenced θ and σ, so evaluations that change
# only mu-referenced θ or ω reuse them. Keys within one coordinate of the
# reference are cached: those are the ones a covariance step revisits. A point
# outside the model's domain (its evaluation throws `_InfeasibleEvaluation`) has
# zero density. Under `progress` a transient counter tracks the batched calls.
function _conditional_point_values!(prepared::_PreparedPoints, θ, σ, progress = nothing)
    nonmu = setdiff(1:prepared.layout.ntheta, prepared.indices)
    key = Float64.(vcat(θ[nonmu], σ))
    cached = get(prepared.cache, key, nothing)
    cached === nothing || return cached
    θμ = θ[prepared.indices]
    counts = [size(Φ, 2) for Φ in prepared.phi]
    values = [fill(NaN, n) for n in counts]
    with_progress(progress, maximum(counts);
            description = _points_label(prepared.approximation), transient = true) do points
        for j in 1:maximum(counts)
            throw_if_interrupted(points)
            ks = findall(>=(j), counts)
            batch = _eta_batch(prepared.layout, [prepared.phi[k][:, j] - prepared.design[k] * θμ
                for k in ks])
            v = _proposal_loglikelihoods(prepared.model, prepared.subjects[ks], θ, σ, batch)
            prepared.calls[] += 1
            length(v) == length(ks) || throw(DimensionMismatch(
                "conditional_loglikelihoods returned $(length(v)) values for $(length(ks)) subjects"))
            for (m, k) in enumerate(ks)
                (isnan(v[m]) || v[m] == Inf) && throw(DomainError(v[m],
                    "subject $(prepared.subjects[k]): conditional log-likelihood is NaN or +Inf at a quadrature point"))
                values[k][j] = v[m]
            end
            points === nothing || update_progress!(points, j)
        end
    end
    count(!=(0), key .- prepared.reference) <= 1 && (prepared.cache[key] = values)
    return values
end

_points_label(::ImportanceSampling) = "Importance sampling draws"
_points_label(::AdaptiveGaussHermite) = "Quadrature points"

# Per-subject marginal log-likelihoods, Monte Carlo standard errors and
# effective sample sizes at (θ, σ, ω).
function _point_marginals(prepared::_PreparedPoints, θ, σ, ω, progress = nothing)
    conditional = _conditional_point_values!(prepared, θ, σ, progress)
    prior = re_prior(prepared.model, ω)
    θμ = θ[prepared.indices]
    n = length(prepared.subjects)
    ℓ, se, ess = zeros(n), zeros(n), zeros(n)
    for (k, i) in enumerate(prepared.subjects)
        terms = [conditional[k][j] + prepared.offset[k][j] +
            _subject_logprior(prior, i, prepared.phi[k][:, j] - prepared.design[k] * θμ)
            for j in eachindex(conditional[k])]
        top = maximum(terms)
        isfinite(top) || throw(DomainError(top,
            "subject $i: every quadrature point has zero density; the marginal log-likelihood is not finite"))
        r = exp.(terms .- top)
        total = sum(r)
        ℓ[k] = top + log(total)
        ess[k] = total^2 / sum(abs2, r)
        K = length(r)
        mean_r = total / K
        se[k] = sqrt(sum(abs2, r .- mean_r) / (K - 1) / K) / mean_r
    end
    return ℓ, se, ess
end

_subject_marginals(prepared::_PreparedPoints, θ, σ, ω, progress = nothing) =
    first(_point_marginals(prepared, θ, σ, ω, progress))

function _subject_marginals(prepared::_PreparedAdaptive, θ, σ, ω, progress = nothing)
    model = prepared.model
    ebe = _run_empirical_bayes(model, θ, σ, ω, prepared.center.subjects, prepared.center,
        default_ebe_method(model), EBEOptions(gtol = 1.0e-10), true, progress, true)
    points = _points_at_modes(prepared.approximation, model, θ, σ, ω, ebe, nothing)
    terms = _subject_marginals(points, θ, σ, ω, progress)
    prepared.calls[] += points.calls[]
    return terms
end

# Marginal terms with Monte Carlo error: `nothing` effective sample sizes and
# zero standard errors for the deterministic approximations.
_marginal_terms(prepared::_PreparedPoints{<:Any, ImportanceSampling}, θ, σ, ω, progress) =
    _point_marginals(prepared, θ, σ, ω, progress)

function _marginal_terms(prepared::_PreparedAdaptive, θ, σ, ω, progress)
    ℓ, _, _ = _point_marginals(prepared.points, θ, σ, ω, progress)
    return ℓ, zero(ℓ), nothing
end

function _marginal_terms(prepared::_PreparedFOCE, θ, σ, ω, progress)
    ℓ = _subject_marginals(prepared, θ, σ, ω, progress)
    return ℓ, zero(ℓ), nothing
end

_marginal_calls(prepared::_PreparedPoints) = prepared.calls[]
_marginal_calls(prepared::_PreparedAdaptive) = prepared.points.calls[] + prepared.calls[]
_marginal_calls(prepared::_PreparedFOCE) = missing

"""
    MarginalLikelihoodResult

Result of [`marginal_loglikelihood`](@ref). `point` is `[θ; σ; ω]`.
`loglikelihood` is the marginal log-likelihood with every constant kept, and
`subject_loglikelihoods[k]` is subject `subjects[k]`'s term.
`standard_error` and `subject_standard_errors` are Monte Carlo standard errors.
They are zero for the deterministic approximations, which report no
approximation error. `effective_sample_size` holds the per-subject importance
sampling effective sample sizes, or `nothing`.

`logposterior` adds a model [`population_logprior`](@ref) once (`objective`
`:MAP`). `ofv` is `-2 logposterior`, as in [`FOCEResult`](@ref), and equals
`-2 loglikelihood` under `:ML`. nlmixr2's `OBJF` omits the Gaussian constant:
it is `ofv - nobs * log(2π)`. `aic = -2 loglikelihood + 2 nparameters` and
`bic = -2 loglikelihood + nparameters * log(nobs)`, where `nobs` counts the
observations of the listed subjects. `nobs`, and so `bic`, is `missing` for a
model that declares no [`observation_model`](@ref) unless it is passed
explicitly. `evaluations` counts batched conditional evaluations (`missing` for
`FOCEApproximation`).
"""
struct MarginalLikelihoodResult{T, A, E}
    subjects::Vector{Int}
    point::Vector{T}
    approximation::A
    loglikelihood::T
    subject_loglikelihoods::Vector{T}
    standard_error::T
    subject_standard_errors::Vector{T}
    effective_sample_size::E
    population_logprior::T
    logposterior::T
    ofv::T
    objective::Symbol
    nobs::Union{Int, Missing}
    nparameters::Int
    aic::T
    bic::Union{T, Missing}
    evaluations::Union{Int, Missing}
end

"""
    marginal_loglikelihood(model, fit; approximation = <see below>,
                           rng = Random.default_rng(), nobs = nothing, progress = nothing)
    marginal_loglikelihood(model, θ, σ, ω; subjects = 1:nsubjects(model), η0 = nothing,
                           approximation = <see below>, rng = Random.default_rng(),
                           nparameters = nothing, nobs = nothing, progress = nothing)
        -> MarginalLikelihoodResult

The marginal log-likelihood `Σᵢ log ∫ p(y_i | θ, σ, η) N(η; 0, Ω) dη` at a fit's
estimate or at given population coordinates, with OFV, AIC and BIC. This gives
SAEM fits a likelihood-based objective. `approximation` is
[`ImportanceSampling`](@ref), [`AdaptiveGaussHermite`](@ref) or
[`FOCEApproximation`](@ref). The first two need only conditional values and
η gradients, so they apply to any likelihood and to ragged layouts. The default
is `ImportanceSampling()`, except on a [`RaggedNLMELayout`](@ref) whose model
reports η Hessians (its own blocks, or a declared [`eta_hessian_sparsity`](@ref)):
there it is the Laplace approximation, `AdaptiveGaussHermite(nodes = 1)`, because
importance sampling loses effective draws as a subject grows. Subject
modes warm-start from the fit's EBEs or final chain states (`η0`).
`nparameters`, the AIC/BIC parameter count, defaults to the estimated
coordinates: θ, σ and the Ω coordinates that [`fixed_omega`](@ref) leaves free,
or `fit.free` for a FOCE(I) fit. `nobs` overrides the observation count used by
BIC. `progress` (a Treebars node or backend symbol) gets a `Marginal likelihood`
node with the subject modes and, while they are evaluated, a counter over the
sampling or quadrature points; an interrupt raises `Treebars.ProgressInterrupted`.
"""
function marginal_loglikelihood(model, fit::Union{SAEMResult, RaggedSAEMResult};
        nobs = nothing, kwargs...)
    return marginal_loglikelihood(model, fit.theta, fit.sigma, fit.omega;
        subjects = fit.subjects, η0 = fit.eta, nobs, kwargs...)
end

function marginal_loglikelihood(model, fit::FOCEResult; nobs = nothing, kwargs...)
    return marginal_loglikelihood(model, fit.theta, fit.sigma, fit.omega;
        subjects = fit.ebes.subjects, η0 = fit.ebes, nparameters = length(fit.free),
        nobs, kwargs...)
end

# Laplace where a ragged subject's η Hessians are cheap: importance sampling's
# effective draws collapse as a subject grows, and Laplace is what lme4 and
# glmmTMB use for nested effects at that scale (decision 1jf8d4p).
_default_marginal_approximation(model) =
    _default_marginal_approximation(model, nlme_layout(model))
_default_marginal_approximation(model, ::RaggedNLMELayout) =
    _reports_eta_hessians(model) ? AdaptiveGaussHermite(nodes = 1) : ImportanceSampling()
_default_marginal_approximation(model, layout) = ImportanceSampling()

function marginal_loglikelihood(model, θ::AbstractVector, σ::AbstractVector, ω::AbstractVector;
        subjects = nothing, η0 = nothing, approximation = _default_marginal_approximation(model),
        rng::AbstractRNG = Random.default_rng(), nparameters = nothing, nobs = nothing,
        progress = nothing)
    layout = nlme_layout(model)
    _check_length("θ", θ, layout.ntheta)
    _check_length("σ", σ, layout.nsigma)
    _check_length("ω", ω, nomega(layout))
    nparameters = nparameters === nothing ? layout.ntheta + layout.nsigma +
        length(_free_omega(_omega_constraints(layout, fixed_omega(model)), ω)) :
        Int(nparameters)
    nparameters >= 0 || throw(ArgumentError("nparameters must be nonnegative"))
    prepared, (ℓ, se, ess) = with_progress(progress; description = "Marginal likelihood") do node
        marginal = _prepare_marginal(approximation, model, θ, σ, ω, subjects, η0, rng, node)
        return marginal, _marginal_terms(marginal, θ, σ, ω, node)
    end
    ids = collect(Int, _marginal_subjects(prepared))
    all(isfinite, ℓ) || throw(DomainError(ℓ, "marginal log-likelihood is not finite"))
    lp = population_logprior(model, θ, σ, ω)
    lp === nothing || lp isa Real || throw(ArgumentError(
        "population_logprior must return a real log density or nothing"))
    objective = lp === nothing ? :ML : :MAP
    lp = lp === nothing ? 0.0 : Float64(lp)
    ll = sum(ℓ)
    observed = nobs === nothing ? _observation_count(observation_model(model), model, ids) :
        Int(nobs)
    return MarginalLikelihoodResult(ids, Float64.(vcat(θ, σ, ω)), approximation, ll,
        Vector{Float64}(ℓ), sqrt(sum(abs2, se)), Vector{Float64}(se), ess, lp, ll + lp,
        -2(ll + lp), objective, observed, Int(nparameters), -2ll + 2nparameters,
        observed === missing ? missing : -2ll + nparameters * log(observed),
        _marginal_calls(prepared))
end

_observation_count(::Nothing, model, subjects) = missing
_observation_count(::GaussianObservations, model, subjects) =
    sum(length, observations(model, subjects))
