"""
    EBEMethod

Per-subject mode search of [`empirical_bayes`](@ref):

- [`NewtonEBE`](@ref): Newton on the η Hessian blocks, needs
  [`provides_eta_hessian`](@ref).
- [`QuasiNewtonEBE`](@ref): BFGS on η gradients only.

[`default_ebe_method`](@ref) picks Newton when Hessian blocks are available.
"""
abstract type EBEMethod end

"""
    NewtonEBE()

Newton's method on the complete-data Hessian blocks with Levenberg–Marquardt
damping (λ increased tenfold) when `-H` is not positive definite or a step fails the
Armijo test. A subject converges once `gᵀ(-H)⁻¹g / 2 ≤ decrement_tol`; the Newton
step from there is taken as a final polishing step, so the returned mode is
stationary to roundoff.
"""
struct NewtonEBE <: EBEMethod end

"""
    QuasiNewtonEBE()

BFGS per subject on η gradients only, starting from the prior covariance Ω as the
inverse-Hessian approximation (rescaled to the observed curvature at the first
update), with backtracking (halving) to the Armijo condition.
A subject converges once its standardized gradient `‖Lᵀg‖∞ ≤ gtol`, with `Ω = LLᵀ`
(the gradient with respect to the standardized random effects `L⁻¹η`).
"""
struct QuasiNewtonEBE <: EBEMethod end

"""
    default_ebe_method(model) -> EBEMethod

[`NewtonEBE`](@ref) if [`provides_eta_hessian`](@ref), else [`QuasiNewtonEBE`](@ref).
"""
default_ebe_method(model) = provides_eta_hessian(model) ? NewtonEBE() : QuasiNewtonEBE()

"""
    EBEOptions(; maxiter = 200, decrement_tol = 1e-14, gtol = 1e-8, max_backtracks = 30)

Settings of [`empirical_bayes`](@ref).

- `maxiter`: maximum number of iterations (accepted steps) per subject.
- `decrement_tol`: [`NewtonEBE`](@ref) convergence, half the Newton decrement in
  log-density units.
- `gtol`: [`QuasiNewtonEBE`](@ref) convergence, the standardized gradient `‖Lᵀg‖∞`.
- `max_backtracks`: maximum damping increases or step halvings per iteration before
  a subject is reported as not converged.
"""
Base.@kwdef struct EBEOptions
    maxiter::Int = 200
    decrement_tol::Float64 = 1.0e-14
    gtol::Float64 = 1.0e-8
    max_backtracks::Int = 30
end

abstract type AbstractEBEResult end

"""
    EBEResult

Result of [`empirical_bayes`](@ref). Column/entry `k` of every field belongs to
subject `subjects[k]`.

- `modes`: `neta × length(subjects)` matrix of empirical Bayes estimates η̂_i.
- `hessians`: the `neta × neta` Hessian in η of the complete-data log-density
  `log p(y_i, η_i | θ, σ, Ω)` at η̂_i per subject, or `nothing` when the model
  provides no second order (see `ebe_covariances`).
- `conditional_loglikelihood`, `complete_data_loglikelihood`: the two log-densities
  at η̂_i.
- `converged`, `iterations`: per-subject solver diagnostics.
- `gradient_norm`: the standardized complete-data gradient `‖Lᵀg‖∞` at η̂_i.
- `method`: the [`EBEMethod`](@ref) used.
"""
struct EBEResult{T <: Real, HS <: Union{Nothing, Vector{Matrix{T}}}, M <: EBEMethod} <: AbstractEBEResult
    subjects::Vector{Int}
    modes::Matrix{T}
    hessians::HS
    conditional_loglikelihood::Vector{T}
    complete_data_loglikelihood::Vector{T}
    converged::Vector{Bool}
    iterations::Vector{Int}
    gradient_norm::Vector{T}
    method::M
end

"""
    RaggedEBEResult

Empirical Bayes result for a [`RaggedNLMELayout`](@ref). `modes[k]` and
`hessians[k]` have the actual dimension of `subjects[k]`, without padding.
"""
struct RaggedEBEResult{T <: Real, M <: EBEMethod} <: AbstractEBEResult
    subjects::Vector{Int}
    modes::Vector{Vector{T}}
    hessians::Union{Nothing,Vector{Matrix{T}}}
    conditional_loglikelihood::Vector{T}
    complete_data_loglikelihood::Vector{T}
    converged::Vector{Bool}
    iterations::Vector{Int}
    gradient_norm::Vector{T}
    method::M
end

"""
    EBEConvergenceError(subjects, result)

Thrown by [`empirical_bayes`](@ref) when no mode was found for some subjects;
`result` holds the partial [`EBEResult`](@ref).
"""
struct EBEConvergenceError{R <: AbstractEBEResult} <: Exception
    subjects::Vector{Int}
    result::R
end
function Base.showerror(io::IO, e::EBEConvergenceError)
    r = e.result
    print(io, "EBEConvergenceError: no mode found for subject(s) ", e.subjects, " (", nameof(typeof(r.method)), ").")
    for i in e.subjects
        k = findfirst(==(i), r.subjects)
        print(io, "\n  subject ", i, ": ", r.iterations[k], " iterations, standardized gradient ",
            r.gradient_norm[k], ", complete-data log-density ", r.complete_data_loglikelihood[k])
    end
    print(io, "\nPass `require_convergence = false` to inspect the partial result, or a warm start `η0`.")
end

"""
    empirical_bayes(model, θ, σ, ω; subjects = 1:nsubjects(model), η0 = nothing,
                    method = default_ebe_method(model), options = EBEOptions(),
                    require_convergence = true) -> EBEResult

Empirical Bayes estimates (posthoc): for fixed population parameters θ, σ and Ω
parameters ω, the per-subject mode

    η̂_i = argmax_η  log p(y_i | θ, σ, η) + log N(η | 0, Ω).

All subjects are solved in lockstep: each iteration makes one batched gradient call
([`conditional_loglikelihoods_and_gradients`](@ref)) for every subject still
iterating, plus batched value calls for its trial steps, so a joint-program model
pays O(N) per iteration rather than O(N) per subject. Models are never called
concurrently.

`η0` is the starting point: `nothing` (zeros), a `neta × length(subjects)` matrix,
one vector per subject, or an earlier [`EBEResult`](@ref) covering these subjects
(warm start). `method` is [`NewtonEBE`](@ref) or [`QuasiNewtonEBE`](@ref).

Throws an [`EBEConvergenceError`](@ref) if some subject's mode is not found, unless
`require_convergence = false`, in which case the result records the failure in
`converged`. A non-finite complete-data log-density at the starting point is always
an error.
"""
function empirical_bayes(model, θ::AbstractVector, σ::AbstractVector, ω::AbstractVector;
        subjects = nothing, η0 = nothing, method::EBEMethod = default_ebe_method(model),
        options::EBEOptions = EBEOptions(), require_convergence::Bool = true)
    layout = nlme_layout(model)
    return _empirical_bayes(layout, model, θ, σ, ω, subjects, η0, method, options,
        require_convergence)
end

function _empirical_bayes(layout::NLMELayout, model, θ, σ, ω, subjects, η0,
        method, options, require_convergence)
    _check_length("θ", θ, layout.ntheta)
    _check_length("σ", σ, layout.nsigma)
    _check_length("ω", ω, nomega(layout))
    n = nsubjects(model)
    subjects = subjects === nothing ? collect(1:n) : collect(Int, subjects)
    all(i -> 1 <= i <= n, subjects) || throw(ArgumentError("subjects $subjects are outside 1:$n"))
    allunique(subjects) || throw(ArgumentError("subjects $subjects are not unique"))
    prior = re_prior(model, ω)
    H0 = _eta_start(η0, subjects, neta(layout))
    result = _ebe_lockstep(method, model, subjects, θ, σ, prior, H0, options)
    failed = subjects[.!result.converged]
    require_convergence && !isempty(failed) && throw(EBEConvergenceError(failed, result))
    return result
end

"""
    posthoc(model, θ, σ, ω; kwargs...)

Alias of [`empirical_bayes`](@ref).
"""
const posthoc = empirical_bayes

"""
    ebe_covariances(result::EBEResult) -> Vector{Matrix}

The Laplace (Gaussian) approximation of each subject's η posterior covariance,
`(-H_i)⁻¹`, from the Hessians at the modes. Throws if the result carries no
Hessians or some `-H_i` is not positive definite.
"""
ebe_covariances(result::EBEResult) = _ebe_covariances(result.hessians)
ebe_covariances(result::RaggedEBEResult) = _ebe_covariances(result.hessians)
_ebe_covariances(hessians::Vector) = [inv(cholesky(Symmetric(-H))) for H in hessians]
_ebe_covariances(::Nothing) = throw(ArgumentError(
    "this EBEResult carries no Hessians: the model provides no η second order (provides_eta_hessian)"))

_eta_start(::Nothing, subjects, m) = zeros(m, length(subjects))
_eta_start(η0::AbstractMatrix{<:Real}, subjects, m) = _checked_start(η0, subjects, m)
_eta_start(η0::AbstractVector{<:AbstractVector}, subjects, m) =
    _checked_start(reduce(hcat, η0; init = zeros(m, 0)), subjects, m)
function _eta_start(η0::EBEResult, subjects, m)
    columns = map(subjects) do i
        j = findfirst(==(i), η0.subjects)
        j === nothing && throw(ArgumentError("η0: the warm-start EBEResult has no mode for subject $i"))
        j
    end
    return _checked_start(η0.modes[:, columns], subjects, m)
end
_eta_start(η0, subjects, m) = throw(ArgumentError(
    "η0 must be `nothing`, a matrix, one vector per subject, or an EBEResult; got a $(typeof(η0))"))
function _checked_start(H, subjects, m)
    size(H) == (m, length(subjects)) || throw(DimensionMismatch(
        "η0 has size $(size(H)); expected ($m, $(length(subjects))) for neta = $m and $(length(subjects)) subjects"))
    return H
end

# ── The lockstep solver ───────────────────────────────────────────────────────
#
# State per subject k: current η (column k of H, or H[k] for a ragged layout),
# conditional and complete-data log-densities ℓ[k], f[k], complete-data gradient (in
# the same storage as H), and the method's curvature (Newton: Hessian block;
# quasi-Newton: inverse-Hessian approximation of -f). Each iteration proposes a
# direction for every active subject, evaluates all trial points in batched value
# calls (retrying rejected subjects with more damping or a shorter step), then
# evaluates gradients (and Newton's Hessian blocks) at the accepted points in one
# batched call. The prior's type selects the layout's complete-data evaluations.

function _ebe_lockstep(method::EBEMethod, model, subjects, θ, σ, prior, H0, opt::EBEOptions)
    T = float(promote_type(_eta_eltype(H0), eltype(θ), eltype(σ), _prior_eltype(prior)))
    n = length(subjects)
    H = _eta_state(T, H0)
    ℓ, f, G = _complete_gradients(model, subjects, θ, σ, prior, H)
    bad = findall(!isfinite, f)
    isempty(bad) || throw(DomainError(f[bad], "empirical_bayes: the complete-data log-density is " *
        "not finite at the starting point for subject(s) $(subjects[bad])"))
    curvature = _initial_curvature(method, model, subjects, θ, σ, prior, H)
    converged = falses(n)
    failed = falses(n)
    iterations = zeros(Int, n)
    while true
        active = findall(k -> !converged[k] && !failed[k], 1:n)
        done = filter(k -> _is_converged(method, curvature, k, _column(G, k), prior, subjects[k], opt), active)
        converged[done] .= true
        _polish!(method, model, subjects, θ, σ, prior, H, ℓ, f, G, curvature, iterations, done)
        active = setdiff(active, done)
        isempty(active) && break
        exhausted = filter(k -> iterations[k] >= opt.maxiter, active)
        failed[exhausted] .= true
        active = setdiff(active, exhausted)
        isempty(active) && break
        accepted, steps, rejected = _search_steps(method, model, subjects, θ, σ, prior, H, f, G,
            curvature, active, opt)
        failed[rejected] .= true
        isempty(accepted) && continue
        previous_G = _columns(G, accepted)
        Ha = _from_columns(H, [_column(H, k) + step for (k, step) in zip(accepted, steps)])
        _set_columns!(H, accepted, Ha)
        ℓa, fa, Ga = _complete_gradients(model, subjects[accepted], θ, σ, prior, Ha)
        ℓ[accepted] = ℓa
        f[accepted] = fa
        _set_columns!(G, accepted, Ga)
        iterations[accepted] .+= 1
        _update_curvature!(method, curvature, model, subjects, θ, σ, prior, H, accepted,
            steps, previous_G, Ga)
    end
    hessians = _final_hessians(method, curvature, model, subjects, θ, σ, prior, H)
    gradient_norm = [_standardized_norm(prior, i, _column(G, k)) for (k, i) in enumerate(subjects)]
    return _ebe_result(H, collect(subjects), hessians, ℓ, f, collect(converged), iterations,
        gradient_norm, method)
end

# Trial steps for the active subjects: a direction each, then batched value calls
# until every subject either passes the sufficient-increase test or exhausts its
# retries. Returns the subjects that found an ascent step (in `active` order), their
# steps, and the subjects that found none.
function _search_steps(method, model, subjects, θ, σ, prior, H, f, G, curvature, active, opt)
    trials = Dict(k => _first_trial(method, curvature, k, _column(G, k)) for k in active)
    pending = collect(active)
    for _ in 0:opt.max_backtracks
        isempty(pending) && break
        proposals = filter(k -> trials[k].step !== nothing, pending)
        if !isempty(proposals)
            trial_H = _from_columns(H, [_column(H, k) + trials[k].step for k in proposals])
            _, ft = _complete_values(model, subjects[proposals], θ, σ, prior, trial_H)
            for (j, k) in enumerate(proposals)
                _accepts(trials[k], f[k], ft[j], _column(G, k)) && filter!(!=(k), pending)
            end
        end
        for k in pending
            trials[k] = _next_trial(method, trials[k], curvature, k, _column(G, k))
        end
    end
    accepted = setdiff(active, pending)
    return accepted, [trials[k].step for k in accepted], pending
end

# η storage: a neta × n matrix for an equal-size layout, one compact vector per
# subject for a ragged one. `k` indexes the solver's subjects.
_eta_eltype(H::AbstractMatrix) = eltype(H)
_eta_eltype(H::AbstractVector{<:AbstractVector}) = eltype(eltype(H))
_eta_state(::Type{T}, H::AbstractMatrix) where {T} = Matrix{T}(H)
_eta_state(::Type{T}, H::AbstractVector{<:AbstractVector}) where {T} = [Vector{T}(h) for h in H]
_column(X::AbstractMatrix, k) = view(X, :, k)
_column(X::AbstractVector{<:AbstractVector}, k) = X[k]
_columns(X::AbstractMatrix, ks) = X[:, ks]
_columns(X::AbstractVector{<:AbstractVector}, ks) = X[ks]
_from_columns(::AbstractMatrix, columns) = reduce(hcat, columns)
_from_columns(::AbstractVector{<:AbstractVector}, columns) = columns
_set_columns!(X::AbstractMatrix, ks, Y) = (X[:, ks] = Y; X)
_set_columns!(X::AbstractVector{<:AbstractVector}, ks, Y) = (X[ks] = Y; X)
_set_column!(X::AbstractMatrix, k, y) = (X[:, k] = y; X)
_set_column!(X::AbstractVector{<:AbstractVector}, k, y) = (X[k] = y; X)

_ebe_result(H::AbstractMatrix, subjects, hessians, ℓ, f, converged, iterations, gradient_norm, method) =
    EBEResult(subjects, H, hessians, ℓ, f, converged, iterations, gradient_norm, method)
function _ebe_result(H::AbstractVector{<:AbstractVector}, subjects, hessians, ℓ, f, converged,
        iterations, gradient_norm, method)
    T = _eta_eltype(H)
    return RaggedEBEResult(subjects, H, hessians, Vector{T}(ℓ), Vector{T}(f), converged, iterations,
        Vector{T}(gradient_norm), method)
end

# The prior's covariance of subject i's η, and the standardized gradient ‖Lᵀg‖∞.
_prior_eltype(prior::NormalREPrior) = eltype(prior.precision)
_prior_covariance(prior::NormalREPrior, i) = Matrix(prior.cholesky_factor * prior.cholesky_factor')
_standardized_norm(prior::NormalREPrior, i, g) =
    maximum(abs, prior.cholesky_factor' * g; init = zero(eltype(g)))

# Sufficient increase (Armijo, c = 1e-4) — or, for an undamped full step whose
# predicted increase is below what f resolves, no measurable decrease.
function _accepts(trial, f, ft, g)
    isfinite(ft) || return false
    predicted = dot(g, trial.step)
    ft - f >= 1.0e-4 * predicted && return true
    return trial.full && predicted <= sqrt(eps(typeof(f))) * max(one(f), abs(f)) && ft >= f - _roundoff(f)
end

# Changes of f below this are not resolved by evaluating f.
_roundoff(f) = 16 * eps(typeof(f)) * max(one(f), abs(f))

# ── Newton ────────────────────────────────────────────────────────────────────

_initial_curvature(::NewtonEBE, model, subjects, θ, σ, prior, H) =
    _complete_hessians(model, subjects, θ, σ, prior, H)

function _is_converged(::NewtonEBE, hessians, k, g, prior, i, opt)
    p, decrement = _newton_step(g, hessians[k])
    return p !== nothing && decrement / 2 <= opt.decrement_tol
end

# The Newton step (-H)⁻¹g and decrement gᵀ(-H)⁻¹g, or (nothing, NaN) when -H is not
# positive definite.
function _newton_step(g, Hk)
    ch = cholesky(Symmetric(-Hk); check = false)
    issuccess(ch) || return nothing, oftype(float(zero(eltype(g))), NaN)
    p = ch \ g
    return p, dot(g, p)
end

# A Newton trial: the full step (λ = 0) or a Levenberg–Marquardt step (-H + λI)⁻¹g.
# `step` is `nothing` while -H + λI is not positive definite (the next retry raises
# λ). Every state has this one concrete type, so `_search_steps` can store any retry
# in the table it builds from the first trials.
struct DampedTrial{T}
    step::Union{Nothing, Vector{T}}
    full::Bool
    λ::T
    λ0::T
end
DampedTrial(step, full, λ::T, λ0::T) where {T} = DampedTrial{T}(step, full, λ, λ0)

function _first_trial(::NewtonEBE, hessians, k, g)
    p, _ = _newton_step(g, hessians[k])
    λ0 = 1.0e-3 * max(one(eltype(hessians[k])), maximum(abs, diag(hessians[k])))
    p === nothing && return _damped_trial(hessians[k], g, λ0, λ0)
    return DampedTrial(p, true, zero(λ0), λ0)
end

function _next_trial(::NewtonEBE, trial, hessians, k, g)
    λ = iszero(trial.λ) ? trial.λ0 : 10 * trial.λ
    return _damped_trial(hessians[k], g, λ, trial.λ0)
end

function _damped_trial(Hk, g, λ, λ0)
    ch = cholesky(Symmetric(-Hk + λ * I); check = false)
    return DampedTrial(issuccess(ch) ? ch \ g : nothing, false, λ, λ0)
end

function _update_curvature!(::NewtonEBE, hessians, model, subjects, θ, σ, prior, H, accepted, steps, previous_G, Ga)
    hessians[accepted] = _complete_hessians(model, subjects[accepted], θ, σ, prior, _columns(H, accepted))
    return hessians
end

# Converged subjects take the Newton step from the converged point (accurate to
# second order); it is kept unless f drops measurably or -H stops being positive
# definite at the new point.
function _polish!(::NewtonEBE, model, subjects, θ, σ, prior, H, ℓ, f, G, hessians, iterations, done)
    isempty(done) && return nothing
    Hp = _from_columns(H, [_column(H, k) + first(_newton_step(_column(G, k), hessians[k])) for k in done])
    ℓp, fp, Gp = _complete_gradients(model, subjects[done], θ, σ, prior, Hp)
    Hessp = _complete_hessians(model, subjects[done], θ, σ, prior, Hp)
    for (j, k) in enumerate(done)
        keep = isfinite(fp[j]) && fp[j] >= f[k] - _roundoff(f[k]) &&
            issuccess(cholesky(Symmetric(-Hessp[j]); check = false))
        keep || continue
        _set_column!(H, k, _column(Hp, j))
        ℓ[k], f[k] = ℓp[j], fp[j]
        _set_column!(G, k, _column(Gp, j))
        hessians[k] = Hessp[j]
        iterations[k] += 1
    end
    return nothing
end

_final_hessians(::NewtonEBE, hessians, model, subjects, θ, σ, prior, H) = hessians

# ── Quasi-Newton (BFGS) ───────────────────────────────────────────────────────

# Per-subject inverse-Hessian approximations of -f. They start from the prior
# covariance Ω (the exact inverse Hessian when the data are uninformative), rescaled
# at the first update by sᵀy / yᵀΩy (Shanno–Phua) to the curvature actually observed.
struct BFGSState{T}
    inverse_hessians::Vector{Matrix{T}}
    rescaled::Vector{Bool}
end

_initial_curvature(::QuasiNewtonEBE, model, subjects, θ, σ, prior, H) =
    BFGSState([_prior_covariance(prior, i) for i in subjects], fill(false, length(subjects)))

_is_converged(::QuasiNewtonEBE, state, k, g, prior, i, opt) = _standardized_norm(prior, i, g) <= opt.gtol

_first_trial(::QuasiNewtonEBE, state, k, g) = (step = state.inverse_hessians[k] * g, full = true)
_next_trial(::QuasiNewtonEBE, trial, state, k, g) = (step = trial.step / 2, full = false)

# BFGS update of the inverse Hessian of -f with s = Δη, y = Δ(-g); skipped when the
# curvature condition sᵀy > 0 fails.
function _update_curvature!(::QuasiNewtonEBE, state, model, subjects, θ, σ, prior, H, accepted, steps, previous_G, Ga)
    for (j, k) in enumerate(accepted)
        s = steps[j]
        y = _column(previous_G, j) - _column(Ga, j)
        sy = dot(s, y)
        sy > 0 || continue
        B = state.inverse_hessians[k]
        if !state.rescaled[k]
            B = (sy / dot(y, B * y)) * B
            state.rescaled[k] = true
        end
        ρ = 1 / sy
        V = I - ρ * y * s'
        state.inverse_hessians[k] = V' * B * V + ρ * s * s'
    end
    return state
end

_polish!(::QuasiNewtonEBE, model, subjects, θ, σ, prior, H, ℓ, f, G, state, iterations, done) = nothing

_final_hessians(::QuasiNewtonEBE, state, model, subjects, θ, σ, prior, H) =
    provides_eta_hessian(model) ? _complete_hessians(model, subjects, θ, σ, prior, H) : nothing
