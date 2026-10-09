"""
    EBEOptions(; maxiter = 100, decrement_tol = 1e-14, max_damping_steps = 30)

Settings of the per-subject mode search in [`empirical_bayes`](@ref).

- `maxiter`: maximum number of Newton iterations per subject.
- `decrement_tol`: a mode is accepted once the Hessian of the complete-data
  log-density is negative definite and half the Newton decrement,
  `gᵀ(-H)⁻¹g / 2` (the predicted remaining increase of the log-density), is at most
  this value. The Newton step from that point is then taken as a final polishing
  step (counted in `iterations`), so the returned mode is stationary to roundoff.
- `max_damping_steps`: maximum number of Levenberg–Marquardt damping increases
  per iteration before the subject is reported as not converged.
"""
Base.@kwdef struct EBEOptions
    maxiter::Int = 100
    decrement_tol::Float64 = 1.0e-14
    max_damping_steps::Int = 30
end

"""
    EBEResult

Result of [`empirical_bayes`](@ref). Entry `k` of every field belongs to subject
`subjects[k]`.

- `modes`: the empirical Bayes estimates η̂_i (posterior modes of η_i).
- `hessians`: the Hessian in η of the complete-data log-density
  `log p(y_i, η_i | θ, σ, Ω)` at η̂_i (negative definite at a converged mode); see
  [`ebe_covariances`](@ref).
- `conditional_loglikelihood`, `complete_data_loglikelihood`: the two
  log-densities at η̂_i.
- `converged`, `iterations`, `newton_decrement`: per-subject solver diagnostics;
  `newton_decrement` is `gᵀ(-H)⁻¹g`, or `NaN` where `-H` is not positive definite.
"""
struct EBEResult{T <: Real}
    subjects::Vector{Int}
    modes::Vector{Vector{T}}
    hessians::Vector{Matrix{T}}
    conditional_loglikelihood::Vector{T}
    complete_data_loglikelihood::Vector{T}
    converged::Vector{Bool}
    iterations::Vector{Int}
    newton_decrement::Vector{T}
end

"""
    EBEConvergenceError(subjects, result)

Thrown by [`empirical_bayes`](@ref) when no mode was found for some subjects;
`result` holds the partial [`EBEResult`](@ref).
"""
struct EBEConvergenceError{R <: EBEResult} <: Exception
    subjects::Vector{Int}
    result::R
end
function Base.showerror(io::IO, e::EBEConvergenceError)
    r = e.result
    print(io, "EBEConvergenceError: no mode found for subject(s) ", e.subjects, ".")
    for i in e.subjects
        k = findfirst(==(i), r.subjects)
        print(io, "\n  subject ", i, ": ", r.iterations[k], " iterations, Newton decrement ",
            r.newton_decrement[k], ", complete-data log-density ", r.complete_data_loglikelihood[k])
    end
    print(io, "\nPass `require_convergence = false` to inspect the partial result, or a warm start `η0`.")
end

"""
    empirical_bayes(model, θ, σ, ω; subjects = 1:nsubjects(model), η0 = nothing,
                    options = EBEOptions(), require_convergence = true) -> EBEResult

Empirical Bayes estimates (posthoc): for fixed population parameters θ, σ and Ω
parameters ω, the per-subject mode

    η̂_i = argmax_η  log p(y_i | θ, σ, η) + log N(η | 0, Ω)

found by Newton's method with Levenberg–Marquardt damping on the AD gradient and
Hessian ([`conditional_loglikelihood_gradient_and_hessian`](@ref)). Subjects are
solved one after another; models need not be thread-safe.

`η0` is the starting point: `nothing` (zeros), one vector per subject in
`subjects`, or an earlier [`EBEResult`](@ref) covering these subjects (warm start).

Throws an [`EBEConvergenceError`](@ref) if some subject's mode is not found, unless
`require_convergence = false`, in which case the result records the failure in
`converged`. A non-finite complete-data log-density at the starting point is always
an error.
"""
function empirical_bayes(model, θ::AbstractVector, σ::AbstractVector, ω::AbstractVector;
        subjects = nothing, η0 = nothing, options::EBEOptions = EBEOptions(),
        require_convergence::Bool = true)
    layout = nlme_layout(model)
    _check_length("θ", θ, layout.ntheta)
    _check_length("σ", σ, layout.nsigma)
    _check_length("ω", ω, nomega(layout))
    n = nsubjects(model)
    subjects = subjects === nothing ? collect(1:n) : collect(Int, subjects)
    all(i -> 1 <= i <= n, subjects) || throw(ArgumentError("subjects $subjects are outside 1:$n"))
    prior = re_prior(model, ω)
    m = neta(layout)
    fits = map(enumerate(subjects)) do (k, i)
        _subject_mode(model, i, θ, σ, prior, _eta_start(η0, k, i, m), options)
    end
    result = EBEResult(subjects, [f.mode for f in fits], [f.hessian for f in fits],
        [f.ℓ for f in fits], [f.f for f in fits], [f.converged for f in fits],
        [f.iterations for f in fits], [f.decrement for f in fits])
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
`(-H_i)⁻¹`, from the Hessians at the modes. Throws if some `-H_i` is not positive
definite.
"""
ebe_covariances(result::EBEResult) = [inv(cholesky(Symmetric(-H))) for H in result.hessians]

_eta_start(::Nothing, k, i, m) = zeros(m)
_eta_start(η0::AbstractVector{<:AbstractVector}, k, i, m) = _checked_start(η0[k], m)
function _eta_start(η0::EBEResult, k, i, m)
    j = findfirst(==(i), η0.subjects)
    j === nothing && throw(ArgumentError("η0: the warm-start EBEResult has no mode for subject $i"))
    return _checked_start(η0.modes[j], m)
end
_eta_start(η0, k, i, m) = throw(ArgumentError(
    "η0 must be `nothing`, one vector per subject, or an EBEResult; got a $(typeof(η0))"))
function _checked_start(η, m)
    _check_length("η0", η, m)
    return η
end

function _subject_mode(model, i, θ, σ, prior, η0, opt::EBEOptions)
    T = float(promote_type(eltype(η0), eltype(θ), eltype(σ), eltype(prior.precision)))
    η = Vector{T}(η0)
    ℓ, f, g, H = complete_data_loglikelihood_gradient_and_hessian(model, i, θ, σ, prior, η)
    isfinite(f) || throw(DomainError(f,
        "empirical_bayes: the complete-data log-density of subject $i is $f at the starting point η = $η"))
    decrement = T(NaN)
    iterations = 0
    while true
        newton_step, decrement = _newton_step(g, H)
        if newton_step !== nothing && decrement / 2 <= opt.decrement_tol
            # Converged. The Newton step from here is accurate to second order, so take
            # it (the mode and Hessian then sit at a stationary point to roundoff, which
            # FOCEI's envelope gradient relies on) unless it measurably lowers f.
            ηp = η + newton_step
            ℓp, fp, gp, Hp = complete_data_loglikelihood_gradient_and_hessian(model, i, θ, σ, prior, ηp)
            stepp, decp = _newton_step(gp, Hp)
            if stepp !== nothing && isfinite(fp) && fp >= f - _roundoff(f)
                η, ℓ, f, H, decrement = ηp, ℓp, fp, Hp, decp
                iterations += 1
            end
            return (mode = η, hessian = H, ℓ = ℓ, f = f, converged = true,
                iterations = iterations, decrement = decrement)
        end
        iterations == opt.maxiter && break
        step = _ascent_step(model, i, θ, σ, prior, η, f, g, H, newton_step, opt)
        step === nothing && break
        η = η + step
        ℓ, f, g, H = complete_data_loglikelihood_gradient_and_hessian(model, i, θ, σ, prior, η)
        iterations += 1
    end
    return (mode = η, hessian = H, ℓ = ℓ, f = f, converged = false,
        iterations = iterations, decrement = decrement)
end

# The Newton step (-H)⁻¹g and decrement gᵀ(-H)⁻¹g, or (nothing, NaN) when -H is not
# positive definite.
function _newton_step(g, H)
    ch = cholesky(Symmetric(-H); check = false)
    issuccess(ch) || return nothing, oftype(float(zero(eltype(g))), NaN)
    p = ch \ g
    return p, dot(g, p)
end

# Changes of f below this are not resolved by evaluating f.
_roundoff(f) = 16 * eps(typeof(f)) * max(one(f), abs(f))

# One accepted step that increases the complete-data log-density f: the Newton step
# when -H is positive definite and it increases f sufficiently, otherwise the
# Levenberg–Marquardt step (-H + λI)⁻¹ g with λ raised tenfold until it does.
# Returns `nothing` when no such step is found within `max_damping_steps`.
function _ascent_step(model, i, θ, σ, prior, η, f, g, H, newton_step, opt)
    T = eltype(η)
    # Near the mode the increase of f is below what f itself resolves; a Newton step
    # whose predicted increase is that small is accepted unless f drops measurably.
    roundoff = _roundoff(f)
    λ0 = T(1.0e-3) * max(one(T), maximum(abs, diag(H)))
    λ = zero(T)
    for _ in 0:opt.max_damping_steps
        p = if iszero(λ)
            newton_step
        else
            ch = cholesky(Symmetric(-H + λ * I); check = false)
            issuccess(ch) ? ch \ g : nothing
        end
        if p !== nothing
            predicted = dot(g, p)
            fnew = complete_data_loglikelihood(model, i, θ, σ, prior, η + p)
            increase = fnew - f
            isfinite(fnew) && increase >= T(1.0e-4) * predicted && return p
            iszero(λ) && predicted <= sqrt(eps(T)) * max(one(T), abs(f)) &&
                isfinite(fnew) && increase >= -roundoff && return p
        end
        λ = iszero(λ) ? λ0 : 10λ
    end
    return nothing
end
