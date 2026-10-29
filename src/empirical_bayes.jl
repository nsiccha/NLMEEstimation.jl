"""
    EBEMethod

Per-subject mode search of [`empirical_bayes`](@ref):

- [`NewtonEBE`](@ref): Newton on the η Hessian blocks, needs
  [`provides_eta_hessian`](@ref) or a declared [`eta_hessian_sparsity`](@ref).
- [`QuasiNewtonEBE`](@ref): BFGS on η gradients only.

[`default_ebe_method`](@ref) picks Newton when Hessian blocks are available.
"""
abstract type EBEMethod end

"""
    NewtonEBE()

Trust-region Newton on the complete-data Hessian blocks. Each step maximizes the
local quadratic model of the complete-data log-density over a ball in the
standardized coordinates `L⁻¹η`, with `Ω = LLᵀ`. Inside the ball that is the full
Newton step when `-H` is positive definite; where `-H` is indefinite (a saddle or a
non-concave region) the step follows the directions in which the density curves
upward, out to the radius. The radius shrinks after a rejected step or a poor
prediction and grows after a good step that reached it. A subject converges once
`gᵀ(-H)⁻¹g / 2 ≤ decrement_tol`; the Newton step from there is taken as a final
polishing step, so the returned mode is stationary to roundoff.

The Hessian blocks are the model's own ([`provides_eta_hessian`](@ref)), or sparse
matrices computed from its batched gradients when it declares
[`eta_hessian_sparsity`](@ref). On a sparse block the trust-region step is found by
sparse Cholesky factorizations of `-H + νΩ⁻¹` (Moré & Sorensen 1983) instead of an
eigendecomposition, so a subject with thousands of nested effects costs time and
memory proportional to the Hessian's nonzeros.
"""
struct NewtonEBE <: EBEMethod end

"""
    QuasiNewtonEBE(; memory = 20)

Limited-memory BFGS per subject on η gradients, with backtracking (halving) to the
Armijo condition. The inverse-Hessian approximation applies the last `memory`
curvature pairs to the prior covariance Ω, rescaled at every update to the curvature
last observed. A subject with `n` random effects therefore stores `O(memory · n)`
numbers, and Ω is applied block by block for a [`RaggedNLMELayout`](@ref).
A subject converges once its standardized gradient `‖Lᵀg‖∞ ≤ gtol`, with `Ω = LLᵀ`
(the gradient with respect to the standardized random effects `L⁻¹η`), and the
complete-data density is concave there.

BFGS keeps a positive-definite curvature model, so on its own it cannot follow a
direction in which the density curves upward: a saddle point passes the gradient
test, and next to one the steps along that direction stay tiny. The solver therefore
checks the complete-data Hessian block where BFGS meets upward curvature (a curvature
pair with `sᵀy ≤ 0`, which BFGS skips) and, once no subject is still iterating, at
every subject that passed the gradient test. The block is sparse when the model
declares [`eta_hessian_sparsity`](@ref) (two batched gradient calls per color of
its pattern), else the model's own when it [`provides_eta_hessian`](@ref) (the one
the result reports, so the check adds no model calls), and otherwise forward
differences of batched gradients (`neta` batched gradient calls per check, not
reported in the result). Where `-H` is not positive
definite, the next step is [`NewtonEBE`](@ref)'s trust-region step with a radius of
one prior standard deviation, halved until the density increases, and BFGS restarts
from Ω.
"""
struct QuasiNewtonEBE <: EBEMethod
    memory::Int
    function QuasiNewtonEBE(; memory::Integer = 20)
        memory >= 1 || throw(ArgumentError("QuasiNewtonEBE memory must be positive, got $memory"))
        return new(memory)
    end
end

"""
    default_ebe_method(model) -> EBEMethod

[`NewtonEBE`](@ref) if [`provides_eta_hessian`](@ref) or the model declares
[`eta_hessian_sparsity`](@ref), else [`QuasiNewtonEBE`](@ref).
"""
default_ebe_method(model) = _reports_eta_hessians(model) ? NewtonEBE() : QuasiNewtonEBE()

"""
    EBEOptions(; maxiter = 200, decrement_tol = 1e-14, gtol = 1e-8, max_backtracks = 30)

Settings of [`empirical_bayes`](@ref).

- `maxiter`: maximum number of iterations (accepted steps) per subject.
- `decrement_tol`: [`NewtonEBE`](@ref) convergence, half the Newton decrement in
  log-density units.
- `gtol`: [`QuasiNewtonEBE`](@ref) convergence, the standardized gradient `‖Lᵀg‖∞`.
- `max_backtracks`: maximum trust-region reductions ([`NewtonEBE`](@ref)) or step
  halvings ([`QuasiNewtonEBE`](@ref)) per iteration before a subject is reported as
  not converged.
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
- With a declared [`eta_hessian_sparsity`](@ref), `hessians` holds `SparseMatrixCSC`s.
- `conditional_loglikelihood`, `complete_data_loglikelihood`: the two log-densities
  at η̂_i.
- `converged`, `iterations`: per-subject solver diagnostics.
- `gradient_norm`: the standardized complete-data gradient `‖Lᵀg‖∞` at η̂_i.
- `method`: the [`EBEMethod`](@ref) used.
"""
struct EBEResult{T <: Real, HS <: Union{Nothing, Vector{Matrix{T}}, Vector{SparseMatrixCSC{T, Int}}},
        M <: EBEMethod} <: AbstractEBEResult
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
`hessians[k]` have the actual dimension of `subjects[k]`, without padding; the
Hessians are `SparseMatrixCSC`s when the model declares
[`eta_hessian_sparsity`](@ref).
"""
struct RaggedEBEResult{T <: Real, M <: EBEMethod} <: AbstractEBEResult
    subjects::Vector{Int}
    modes::Vector{Vector{T}}
    hessians::Union{Nothing, Vector{Matrix{T}}, Vector{SparseMatrixCSC{T, Int}}}
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
        r.hessians === nothing || isposdef(Symmetric(-r.hessians[k])) ||
            print(io, "; the density is not concave there (its η Hessian is not negative definite)")
    end
    print(io, "\nPass `require_convergence = false` to inspect the partial result, or a warm start `η0`.")
end

"""
    empirical_bayes(model, θ, σ, ω; subjects = 1:nsubjects(model), η0 = nothing,
                    method = default_ebe_method(model), options = EBEOptions(),
                    require_convergence = true, progress = nothing) -> EBEResult

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
an error. A trial point at which the model returns `-Inf` or throws a `DomainError`,
`OverflowError` or `PosDefException` (a point outside its domain) is rejected, and
a shorter step is tried.

Evaluating the complete-data log-density f can round off by many ulps of f when f
sums terms much larger than itself, such as observations far from zero with a small
residual SD. Near a mode, a full step whose predicted increase is below
`√eps · max(1, |f|)` is therefore accepted unless f decreases by more than that. The
convergence tests use gradients and Hessian blocks, which stay accurate there.

`progress` is a Treebars node (or a backend symbol such as `:term`) under which an
`Empirical Bayes (Newton)` or `(BFGS)` node reports the solver's iteration and how
many subjects have converged. `Treebars.request_interrupt!` on it raises
`Treebars.ProgressInterrupted` at the next iteration.
"""
function empirical_bayes(model, θ::AbstractVector, σ::AbstractVector, ω::AbstractVector;
        subjects = nothing, η0 = nothing, method::EBEMethod = default_ebe_method(model),
        options::EBEOptions = EBEOptions(), require_convergence::Bool = true,
        progress = nothing)
    return _run_empirical_bayes(model, θ, σ, ω, subjects, η0, method, options,
        require_convergence, progress, false)
end

# `empirical_bayes` under a new progress node. A search that an outer estimator
# repeats (a FOCE(I) objective call, a covariance-step evaluation) gets a
# transient node, which leaves the tree once it finishes.
function _run_empirical_bayes(model, θ, σ, ω, subjects, η0, method, options,
        require_convergence, progress, transient::Bool)
    return with_progress(progress; description = "Empirical Bayes ($(_ebe_label(method)))",
            transient) do node
        _empirical_bayes(nlme_layout(model), model, θ, σ, ω, subjects, η0, method,
            options, require_convergence, node)
    end
end

_ebe_label(::NewtonEBE) = "Newton"
_ebe_label(::QuasiNewtonEBE) = "BFGS"

function _empirical_bayes(layout::NLMELayout, model, θ, σ, ω, subjects, η0,
        method, options, require_convergence, progress)
    _check_length("θ", θ, layout.ntheta)
    _check_length("σ", σ, layout.nsigma)
    _check_length("ω", ω, nomega(layout))
    n = nsubjects(model)
    subjects = subjects === nothing ? collect(1:n) : collect(Int, subjects)
    all(i -> 1 <= i <= n, subjects) || throw(ArgumentError("subjects $subjects are outside 1:$n"))
    allunique(subjects) || throw(ArgumentError("subjects $subjects are not unique"))
    prior = re_prior(model, ω)
    H0 = _eta_start(η0, subjects, neta(layout))
    result = _ebe_lockstep(method, model, subjects, θ, σ, prior, H0, options, progress)
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
_ebe_covariances(hessians::Vector) = [_laplace_covariance(H) for H in hessians]
_laplace_covariance(H::AbstractMatrix) = inv(cholesky(Symmetric(-H)))
# Dense: the covariance of a subject with a sparse Hessian is in general dense.
_laplace_covariance(H::SparseMatrixCSC) = cholesky(Symmetric(-H)) \ Matrix{eltype(H)}(I, size(H)...)
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
# batched call. Once no subject iterates, the method may reopen converged subjects
# that are not at a mode. The prior's type selects the layout's complete-data
# evaluations.

function _ebe_lockstep(method::EBEMethod, model, subjects, θ, σ, prior, H0, opt::EBEOptions,
        progress = nothing)
    T = float(promote_type(_eta_eltype(H0), eltype(θ), eltype(σ), _prior_eltype(prior)))
    n = length(subjects)
    H = _eta_state(T, H0)
    ℓ, f, G = _complete_gradients(model, subjects, θ, σ, prior, H)
    bad = findall(!isfinite, f)
    isempty(bad) || throw(DomainError(f[bad], "empirical_bayes: the complete-data log-density is " *
        "not finite at the starting point for subject(s) $(subjects[bad])"))
    curvature = _initial_curvature(method, model, subjects, θ, σ, prior, H, G)
    converged = falses(n)
    failed = falses(n)
    iterations = zeros(Int, n)
    sweep = 0
    while true
        throw_if_interrupted(progress)
        active = findall(k -> !converged[k] && !failed[k], 1:n)
        done = filter(k -> _is_converged(method, curvature, k, _column(G, k), prior, subjects[k], opt), active)
        converged[done] .= true
        progress === nothing || update_progress!(progress,
            "iteration $sweep · $(count(converged))/$n subjects converged")
        sweep += 1
        _polish!(method, model, subjects, θ, σ, prior, H, ℓ, f, G, curvature, iterations, done)
        active = setdiff(active, done)
        if isempty(active)
            active = _reopen_nonconcave!(method, curvature, model, subjects, θ, σ, prior, H, G, converged)
            converged[active] .= false
            isempty(active) && break
        end
        exhausted = filter(k -> iterations[k] >= opt.maxiter, active)
        failed[exhausted] .= true
        active = setdiff(active, exhausted)
        isempty(active) && break
        accepted, trials, gains, rejected = _search_steps(method, model, subjects, θ, σ, prior, H, f, G,
            curvature, active, opt)
        failed[rejected] .= true
        isempty(accepted) && continue
        previous_G = _columns(G, accepted)
        Ha = _from_columns(H, [_column(H, k) + trial.step for (k, trial) in zip(accepted, trials)])
        _set_columns!(H, accepted, Ha)
        ℓa, fa, Ga = _complete_gradients(model, subjects[accepted], θ, σ, prior, Ha)
        ℓ[accepted] = ℓa
        f[accepted] = fa
        _set_columns!(G, accepted, Ga)
        iterations[accepted] .+= 1
        _update_curvature!(method, curvature, model, subjects, θ, σ, prior, H, accepted,
            trials, gains, previous_G, Ga)
    end
    hessians = _final_hessians(method, curvature, model, subjects, θ, σ, prior, H)
    gradient_norm = [_standardized_norm(prior, i, _column(G, k)) for (k, i) in enumerate(subjects)]
    return _ebe_result(H, collect(subjects), hessians, ℓ, f, collect(converged), iterations,
        gradient_norm, method)
end

# Trial steps for the active subjects: a step each, then batched value calls until
# every subject either passes the sufficient-increase test or exhausts its retries.
# Returns the subjects that found an ascent step (in `active` order), their accepted
# trials and increases of f, and the subjects that found none.
function _search_steps(method, model, subjects, θ, σ, prior, H, f, G, curvature, active, opt)
    trials = Dict(k => _first_trial(method, curvature, k, _column(G, k)) for k in active)
    gains = Dict{Int, eltype(f)}()
    pending = collect(active)
    for _ in 0:opt.max_backtracks
        isempty(pending) && break
        trial_H = _from_columns(H, [_column(H, k) + trials[k].step for k in pending])
        _, ft = _complete_values(model, subjects[pending], θ, σ, prior, trial_H)
        for (j, k) in enumerate(copy(pending))
            _accepts(trials[k], f[k], ft[j], _column(G, k)) || continue
            gains[k] = ft[j] - f[k]
            filter!(!=(k), pending)
        end
        for k in pending
            trials[k] = _next_trial(method, trials[k], curvature, k, _column(G, k))
        end
    end
    accepted = setdiff(active, pending)
    return accepted, [trials[k] for k in accepted], [gains[k] for k in accepted], pending
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

# The lower Cholesky factor L of the prior covariance of subject i's η, Ω_i = LLᵀ;
# from it the covariance and the standardized gradient ‖Lᵀg‖∞, which is taken block
# by block for a ragged subject.
_prior_eltype(prior::NormalREPrior) = eltype(prior.precision)
_prior_factor(prior::NormalREPrior, i) = prior.cholesky_factor
_prior_factor(prior::NormalREPrior) = prior.cholesky_factor
_prior_covariance(prior, i) = (L = _prior_factor(prior, i); Matrix(L * L'))
_standardized_norm(prior, i, g) =
    maximum(abs, _factor_transpose_times(_subject_prior(prior, i), g); init = zero(eltype(g)))

# Sufficient increase: at least 1e-4 of the predicted increase (the Armijo test for
# a BFGS step's linear prediction, the ratio test for a quadratic model) — or, for a
# full step whose predicted increase is below what f resolves, no decrease beyond that.
function _accepts(trial, f, ft, g)
    isfinite(ft) || return false
    predicted = _predicted(trial, g)
    ft - f >= 1.0e-4 * predicted && return true
    return trial.full && predicted <= _resolution(f) && ft >= f - _resolution(f)
end

# Changes of f below this are not resolved by evaluating f. Its rounding error is
# relative to the terms it sums, which can be far larger than f itself (observations
# far from zero with a small residual SD), and then reaches many ulps of f. Near a
# mode the derivatives stay accurate, so a full step predicted to gain less than this
# is judged by its prediction, and f only rejects a drop beyond this.
_resolution(f) = sqrt(eps(typeof(f))) * max(one(f), abs(f))

# ── Newton ────────────────────────────────────────────────────────────────────

# Newton's state per subject: the complete-data Hessian block (sparse for a model
# that declares `eta_hessian_sparsity`), the subject's prior (the trust region is a
# ball in the standardized coordinates L⁻¹η, with Ω = LLᵀ, where the prior is
# N(0, I)), and the trust-region radius. The first radius is the length of the full
# Newton step where that step exists, so a concave start takes it, and one prior
# standard deviation elsewhere.
struct NewtonState{HS, PS, R}
    hessians::HS
    priors::PS
    radii::R
end

function _initial_curvature(::NewtonEBE, model, subjects, θ, σ, prior, H, G)
    hessians = _complete_hessians(model, subjects, θ, σ, prior, H)
    priors = [_subject_prior(prior, i) for i in subjects]
    radii = map(eachindex(subjects)) do k
        p, _ = _newton_step(_column(G, k), hessians[k])
        p === nothing ? one(eltype(hessians[k])) : norm(_factor_solve(priors[k], p))
    end
    return NewtonState(hessians, priors, radii)
end

function _is_converged(::NewtonEBE, state, k, g, prior, i, opt)
    p, decrement = _newton_step(g, state.hessians[k])
    return p !== nothing && decrement / 2 <= opt.decrement_tol
end

# The Newton step (-H)⁻¹g and decrement gᵀ(-H)⁻¹g, or (nothing, NaN) when -H is not
# positive definite.
function _newton_step(g, Hk)
    ch = cholesky(Symmetric(-Hk); check = false)
    issuccess(ch) || return nothing, oftype(float(zero(eltype(g))), NaN)
    p = ch \ collect(g)
    return p, dot(g, p)
end

# A Newton trial: the maximizer `step` of the quadratic model f + gᵀp + pᵀHp/2 over
# the trust region ‖L⁻¹p‖ ≤ `radius`, the model's increase `predicted` there, whether
# it is the full Newton step from inside the region, and its standardized length.
struct TrustRegionTrial{T}
    step::Vector{T}
    predicted::T
    full::Bool
    radius::T
    length::T
end
_predicted(trial::TrustRegionTrial, g) = trial.predicted

_first_trial(::NewtonEBE, state, k, g) =
    _trust_region_trial(state.hessians[k], state.priors[k], g, state.radii[k])

# A rejected step shrinks the region to a quarter of that step's length.
_next_trial(::NewtonEBE, trial, state, k, g) =
    _trust_region_trial(state.hessians[k], state.priors[k], g, trial.length / 4)

# The trust-region subproblem in standardized coordinates q = L⁻¹p: maximize
# aᵀc - Σⱼ λⱼcⱼ²/2 over ‖c‖ ≤ Δ in the eigenbasis -LᵀHL = Q Diagonal(λ) Qᵀ, with
# a = QᵀLᵀg and q = Qc. Outside the full-Newton case the maximizer is
# cⱼ = aⱼ / (λⱼ + ν) on the boundary, for the ν > max(0, -λ₁) solving ‖c‖ = Δ (Moré &
# Sorensen 1983). In t = ν - max(0, -λ₁) the shifted eigenvalues dⱼ = λⱼ + max(0, -λ₁)
# are ≥ 0, with d₁ = 0 when -H is not positive definite, so the root stays resolved
# however close it lies to that bound. If a has no component where d = 0 and the
# other components fit inside the region (the hard case, e.g. a start exactly at a
# saddle point), the remaining length goes along the first eigenvector. A sparse
# Hessian has its own method, by sparse factorizations (`sparse_curvature.jl`).
_trust_region_trial(Hk::AbstractMatrix, prior, g, Δ) =
    _eigen_trust_region_trial(Hk, _prior_factor(prior), g, Δ)

function _eigen_trust_region_trial(Hk, L, g, Δ)
    λ, Q = eigen(Symmetric(-(L' * Hk * L)))
    a = Q' * (L' * g)
    if first(λ) > 0
        c = a ./ λ
        norm(c) <= Δ && return _eigen_trial(L, Q, λ, a, c, Δ, true)
    end
    d = λ .+ max(zero(eltype(λ)), -first(λ))
    flat = iszero.(d)
    if all(iszero, a[flat])
        c = [iszero(dⱼ) ? zero(aⱼ) : aⱼ / dⱼ for (aⱼ, dⱼ) in zip(a, d)]
        if norm(c) <= Δ
            c[1] = sqrt(max(Δ^2 - sum(abs2, c), zero(Δ)))
            return _eigen_trial(L, Q, λ, a, c, Δ, false)
        end
    end
    c = a ./ (d .+ _boundary_shift(a, d, Δ))
    return _eigen_trial(L, Q, λ, a, c, Δ, false)
end

function _eigen_trial(L, Q, λ, a, c, Δ, full)
    predicted = dot(a, c) - sum(λ .* c .^ 2) / 2
    return TrustRegionTrial(L * (Q * c), predicted, full, Δ, norm(c))
end

# The t ≥ 0 with ‖a ./ (d .+ t)‖ = Δ, by Newton's method on 1/‖c(t)‖ - 1/Δ, which is
# concave and increasing in t: from a start left of the root the iterates increase
# monotonically to it (Moré & Sorensen 1983). Zero is left of the root unless some
# dⱼ = 0, and then ‖a_flat‖ / 2Δ is.
function _boundary_shift(a, d, Δ)
    flat = iszero.(d)
    t = any(flat) ? norm(a[flat]) / (2 * Δ) : zero(Δ)
    for _ in 1:100
        φ = norm(a ./ (d .+ t))
        φ - Δ <= 1.0e-12 * Δ && break
        slope = sum(abs2.(a) ./ (d .+ t) .^ 3) / φ^3
        next = t + (φ - Δ) / (Δ * φ * slope)
        next <= t && break
        t = next
    end
    return t
end

# Nocedal & Wright (2006), Algorithm 4.1, on the ratio ρ of the actual to the
# predicted increase: shrink after a poor prediction, grow after a good step that
# reached the boundary. A step accepted only because its prediction is below what f
# resolves (ρ < 1e-4, see `_accepts`) leaves the radius as it was.
function _updated_radius(trial::TrustRegionTrial, gain)
    ρ = gain / trial.predicted
    ρ < 1.0e-4 && return trial.radius
    ρ < 0.25 && return trial.length / 4
    ρ > 0.75 && !trial.full && return 2 * trial.radius
    return trial.radius
end

function _update_curvature!(::NewtonEBE, state, model, subjects, θ, σ, prior, H, accepted, trials, gains, previous_G, Ga)
    state.hessians[accepted] = _complete_hessians(model, subjects[accepted], θ, σ, prior, _columns(H, accepted))
    state.radii[accepted] = _updated_radius.(trials, gains)
    return state
end

# Converged subjects take the Newton step from the converged point (accurate to
# second order); it is kept unless f drops by more than it resolves (`_resolution`)
# or -H stops being positive definite at the new point. A step that leaves the
# model's domain is dropped too: if the batch throws `_InfeasibleEvaluation`, the
# subjects whose step values are not finite are left at their converged points.
function _polish!(::NewtonEBE, model, subjects, θ, σ, prior, H, ℓ, f, G, state, iterations, done)
    isempty(done) && return nothing
    hessians = state.hessians
    Hp = _from_columns(H, [_column(H, k) + first(_newton_step(_column(G, k), hessians[k])) for k in done])
    evaluate(ks, Hk) = (_complete_gradients(model, subjects[ks], θ, σ, prior, Hk)...,
        _complete_hessians(model, subjects[ks], θ, σ, prior, Hk))
    evaluated = _feasible(Hk -> evaluate(done, Hk), Hp)
    if evaluated === nothing
        _, fp = _complete_values(model, subjects[done], θ, σ, prior, Hp)
        inside = findall(isfinite, fp)
        isempty(inside) && return nothing
        done, Hp = done[inside], _columns(Hp, inside)
        evaluated = evaluate(done, Hp)
    end
    ℓp, fp, Gp, Hessp = evaluated
    for (j, k) in enumerate(done)
        keep = isfinite(fp[j]) && fp[j] >= f[k] - _resolution(f[k]) &&
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

_final_hessians(::NewtonEBE, state, model, subjects, θ, σ, prior, H) = state.hessians

# Newton converges only where -H is positive definite, so every converged subject is
# at a mode.
_reopen_nonconcave!(::NewtonEBE, state, model, subjects, θ, σ, prior, H, G, converged) = Int[]

# ── Quasi-Newton (BFGS) ───────────────────────────────────────────────────────

# Per-subject limited-memory inverse-Hessian approximations of -f (Nocedal & Wright
# 2006, Algorithm 7.4): the last `memory` curvature pairs (s, y, 1/sᵀy) applied to
# the initial matrix γΩ. It starts at γ = 1, the prior covariance Ω (the exact
# inverse Hessian when the data are uninformative), and every stored pair sets
# γ = sᵀy / yᵀΩy (Shanno–Phua), the curvature last observed. Rescaling at every
# update rather than only the first halved the iterations on nested subjects of
# 125–4,000 units with 20 pairs, and kept them below the full BFGS update's.
# Ω is applied through the subject's prior, block by block for a ragged subject.
# Alongside: the complete-data Hessian block at the subject's current point where
# it has been computed (`nothing` otherwise; a step discards it), sparse for a model
# that declares `eta_hessian_sparsity`.
struct BFGSState{T, P}
    priors::Vector{P}
    scales::Vector{T}
    pairs::Vector{Vector{Tuple{Vector{T}, Vector{T}, T}}}
    memory::Int
    hessians::Vector{Union{Nothing, Matrix{T}, SparseMatrixCSC{T, Int}}}
end

function _initial_curvature(method::QuasiNewtonEBE, model, subjects, θ, σ, prior, H, G)
    T = _eta_eltype(H)
    n = length(subjects)
    return BFGSState([_subject_prior(prior, i) for i in subjects], ones(T, n),
        [Tuple{Vector{T}, Vector{T}, T}[] for _ in 1:n], method.memory,
        Vector{Union{Nothing, Matrix{T}, SparseMatrixCSC{T, Int}}}(nothing, n))
end

# Back to γΩ with no pairs, after an escape step.
function _restart!(state::BFGSState, k)
    state.scales[k] = one(eltype(state.scales))
    empty!(state.pairs[k])
    return state
end

# The approximate inverse Hessian of subject k times g, by the two-loop recursion.
function _inverse_hessian_times(state::BFGSState, k, g)
    pairs = state.pairs[k]
    q = Vector{eltype(state.scales)}(g)
    α = similar(q, length(pairs))
    for j in reverse(eachindex(pairs))
        s, y, ρ = pairs[j]
        α[j] = ρ * dot(s, q)
        axpy!(-α[j], y, q)
    end
    r = state.scales[k] * _covariance_times(state.priors[k], q)
    for (j, (s, y, ρ)) in enumerate(pairs)
        axpy!(α[j] - ρ * dot(y, r), s, r)
    end
    return r
end

# Whether subject k's Hessian block shows that the density is not concave at its
# current point.
function _nonconcave(state::BFGSState, k)
    Hk = state.hessians[k]
    return Hk !== nothing && !issuccess(cholesky(Symmetric(-Hk); check = false))
end

_is_converged(::QuasiNewtonEBE, state, k, g, prior, i, opt) =
    !_nonconcave(state, k) && _standardized_norm(prior, i, g) <= opt.gtol

# A quasi-Newton trial: the step p and the predicted increase of f along it,
# `slope + curvature / 2` with slope gᵀp and curvature pᵀHp. A BFGS step predicts with
# the slope alone (the Armijo test). An escape step from a non-concave point uses the
# quadratic model of the Hessian block, and BFGS restarts after it. `full` marks the
# unhalved step.
struct QuasiNewtonTrial{T}
    step::Vector{T}
    slope::T
    curvature::T
    full::Bool
    escape::Bool
end
_predicted(trial::QuasiNewtonTrial, g) = trial.slope + trial.curvature / 2

# The escape step is Newton's trust-region step over one prior standard deviation:
# where -H is not positive definite it follows the directions in which the density
# curves upward, out to the radius (along the most upward one at a zero gradient).
function _first_trial(::QuasiNewtonEBE, state, k, g)
    if _nonconcave(state, k)
        Hk = state.hessians[k]
        step = _trust_region_trial(Hk, state.priors[k], g, one(eltype(Hk))).step
        return QuasiNewtonTrial(step, dot(g, step), dot(step, Hk * step), true, true)
    end
    step = _inverse_hessian_times(state, k, g)
    return QuasiNewtonTrial(step, dot(g, step), zero(eltype(step)), true, false)
end
_next_trial(::QuasiNewtonEBE, trial, state, k, g) =
    QuasiNewtonTrial(trial.step / 2, trial.slope / 2, trial.curvature / 4, false, trial.escape)

# BFGS update of the inverse Hessian of -f with s = Δη, y = Δ(-g): the pair is stored,
# dropping the oldest beyond `memory`. A pair with sᵀy ≤ 0 shows the density curving
# upward along s: BFGS cannot represent that, so the update is skipped and the Hessian
# block at the new point is checked. After an escape step BFGS restarts from Ω.
function _update_curvature!(::QuasiNewtonEBE, state, model, subjects, θ, σ, prior, H, accepted, trials, gains, previous_G, Ga)
    upward = Int[]
    for (j, k) in enumerate(accepted)
        state.hessians[k] = nothing
        if trials[j].escape
            _restart!(state, k)
            continue
        end
        s = trials[j].step
        y = _column(previous_G, j) - _column(Ga, j)
        sy = dot(s, y)
        if sy <= 0
            push!(upward, j)
            continue
        end
        state.scales[k] = sy / dot(y, _covariance_times(state.priors[k], y))
        pairs = state.pairs[k]
        length(pairs) == state.memory && popfirst!(pairs)
        push!(pairs, (copy(s), y, 1 / sy))
    end
    ks = accepted[upward]
    _store_hessians!(state, model, subjects, θ, σ, prior, ks, _columns(H, ks), _columns(Ga, upward))
    return state
end

# Once no subject iterates, the converged subjects whose Hessian block is not yet
# known are checked; those where the density is not concave are reopened and take an
# escape step.
function _reopen_nonconcave!(::QuasiNewtonEBE, state, model, subjects, θ, σ, prior, H, G, converged)
    ks = findall(k -> converged[k] && state.hessians[k] === nothing, eachindex(subjects))
    _store_hessians!(state, model, subjects, θ, σ, prior, ks, _columns(H, ks), _columns(G, ks))
    return filter(k -> _nonconcave(state, k), ks)
end

# The complete-data Hessian blocks of subjects `ks` at their points `Hks` (with
# complete-data gradients `Gks` there): sparse from the declared patterns, the
# model's own blocks, or else forward differences of batched gradients.
function _store_hessians!(state::BFGSState, model, subjects, θ, σ, prior, ks, Hks, Gks)
    isempty(ks) && return state
    state.hessians[ks] = _declares_eta_hessian_sparsity(model) ?
        _sparse_complete_hessians(model, subjects[ks], θ, σ, prior, Hks) :
        provides_eta_hessian(model) ? _model_complete_hessians(model, subjects[ks], θ, σ, prior, Hks) :
        _difference_blocks(model, subjects[ks], θ, σ, prior,
            [Vector(_column(Hks, j)) for j in eachindex(ks)], [Vector(_column(Gks, j)) for j in eachindex(ks)])
    return state
end

# Forward-difference blocks (`_difference_eta_hessians`) take gradients a step away
# from each point. If that batch throws `_InfeasibleEvaluation` (a step left the
# model's domain), each subject is differenced alone, and one whose step still
# throws gets no block: its curvature is unknown, so it converges on the gradient
# test alone.
function _difference_blocks(model, subjects, θ, σ, prior, points, gradients)
    layout = nlme_layout(model)
    blocks(ks) = _difference_eta_hessians(model, layout, subjects[ks], θ, σ, prior, points[ks], gradients[ks])
    batch = _feasible(blocks, eachindex(subjects))
    batch === nothing || return batch
    return [_feasible(k -> only(blocks([k])), k) for k in eachindex(subjects)]
end

_polish!(::QuasiNewtonEBE, model, subjects, θ, σ, prior, H, ℓ, f, G, state, iterations, done) = nothing

# The Hessian blocks at the final points (the model's, or sparse from its patterns),
# computing those not yet known; none for a gradient-only model without patterns,
# whose difference blocks served only the checks.
function _final_hessians(::QuasiNewtonEBE, state::BFGSState, model, subjects, θ, σ, prior, H)
    _reports_eta_hessians(model) || return nothing
    ks = findall(isnothing, state.hessians)
    isempty(ks) || (state.hessians[ks] = _complete_hessians(model, subjects[ks], θ, σ, prior, _columns(H, ks)))
    return identity.(state.hessians)
end
