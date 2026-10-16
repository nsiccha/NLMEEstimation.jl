"""
    FOCEOptions(; maxeval = 500, rhobeg = 0.2, rhoend = 1e-5,
                 interaction = true, ebe_options = EBEOptions())

Settings for the first-order conditional estimation objective and its
derivative-free population search. `interaction = false` omits the variance
Jacobian term (FOCE without interaction). The radii are in the unconstrained
`(θ, σ, ω)` coordinates.
"""
Base.@kwdef struct FOCEOptions
    maxeval::Int = 500
    rhobeg::Float64 = 0.2
    rhoend::Float64 = 1.0e-5
    interaction::Bool = true
    ebe_options::EBEOptions = EBEOptions()
end

"""
    FOCEEvaluation

FOCE(I) log-likelihood at fixed population parameters. `ebes.modes` contains
the subject EBEs, and `curvatures[k]` is the positive η curvature for subject
`subjects[k]`. `subject_loglikelihoods[k]` is that subject's term of
`loglikelihood`. `logposterior` adds the optional population log prior exactly
once; `ofv` is minus twice `logposterior` (equal to minus twice the likelihood
when `objective == :ML`).
"""
struct FOCEEvaluation{T, E, P}
    subjects::Vector{Int}
    loglikelihood::T
    subject_loglikelihoods::Vector{T}
    logposterior::T
    population_logprior::T
    ofv::T
    objective::Symbol
    prior_provenance::P
    ebes::E
    curvatures::Vector{Matrix{T}}
end

"""
    FOCEResult

Population estimate from [`focei`](@ref). `theta`, `sigma`, and `omega` use the
model's declared coordinates; `omega_matrix` is the estimated Ω on covariance
scale. `loglikelihood` remains the approximate marginal likelihood;
`logposterior` includes a model-provided population prior once, and `ofv` is
minus twice that fitted objective. `objective` distinguishes `:ML` from `:MAP`.
`free` lists the fitted positions in the search vector `[θ; σ; free Ω
coordinates]`, which is the concatenated `(θ, σ, ω)` unless the model declares
[`fixed_omega`](@ref) (see [`CovarianceResult`](@ref) for those coordinates),
and `options` the [`FOCEOptions`](@ref) of the fit; [`covariance_step`](@ref)
reuses both.
"""
struct FOCEResult{T, E, I, P}
    theta::Vector{T}
    sigma::Vector{T}
    omega::Vector{T}
    omega_matrix::Matrix{T}
    loglikelihood::T
    logposterior::T
    population_logprior::T
    ofv::T
    objective::Symbol
    prior_provenance::P
    ebes::E
    curvatures::Vector{Matrix{T}}
    evaluations::Int
    converged::Bool
    optimizer_info::I
    free::Vector{Int}
    options::FOCEOptions
end

"""
    focei_evaluate(model, θ, σ, ω; subjects = 1:nsubjects(model), η0 = nothing,
                   interaction = true, ebe_method = QuasiNewtonEBE(),
                   ebe_options = EBEOptions())

Evaluate the first-order conditional (with interaction by default) marginal
log-likelihood. The EBEs maximize the exact complete-data log-density; their
Laplace curvature uses only the first η derivatives of the Gaussian observation
means and variances. Supply `η0` as for [`empirical_bayes`](@ref) to warm-start
the mode search; `ebe_method` and `ebe_options` configure it. A model's
[`population_logprior`](@ref), if present, is added once after the subject
likelihood terms; it does not alter the subject EBEs.
"""
function focei_evaluate(model, θ::AbstractVector, σ::AbstractVector, ω::AbstractVector;
        subjects = nothing, η0 = nothing, interaction::Bool = true,
        ebe_method::EBEMethod = QuasiNewtonEBE(), ebe_options::EBEOptions = EBEOptions())
    _require_focei_layout(nlme_layout(model))
    observation_model(model) isa GaussianObservations || throw(ArgumentError(
        "FOCE(I) requires observation_model(model) = GaussianObservations()"))
    ebes = empirical_bayes(model, θ, σ, ω; subjects, η0,
        method = ebe_method, options = ebe_options)
    parts = observation_moments_and_jacobians(model, ebes.subjects, θ, σ, ebes.modes)
    length(parts) == length(ebes.subjects) || throw(DimensionMismatch(
        "observation_moments_and_jacobians returned $(length(parts)) entries for $(length(ebes.subjects)) subjects"))
    ys = observations(model, ebes.subjects)
    length(ys) == length(parts) || throw(DimensionMismatch(
        "observations returned $(length(ys)) entries for $(length(parts)) subjects"))
    prior = re_prior(model, ω)
    m = neta(prior)
    T = float(promote_type(eltype(θ), eltype(σ), eltype(ω), eltype(ebes.modes)))
    curvatures = Matrix{T}[]
    terms = T[]
    for (k, (i, part)) in enumerate(zip(ebes.subjects, parts))
        μ, v, Jμ, Jv = part
        n = length(ys[k])
        length(μ) == length(v) == n || throw(DimensionMismatch(
            "subject $i has $n observations, $(length(μ)) means, and $(length(v)) variances"))
        size(Jμ) == size(Jv) == (n, m) || throw(DimensionMismatch(
            "subject $i Jacobians have sizes $(size(Jμ)) and $(size(Jv)); expected ($n, $m)"))
        all(x -> isfinite(x) && x > 0, v) || throw(DomainError(v,
            "subject $i has non-positive or non-finite Gaussian observation variance"))
        C = prior.precision + Jμ' * (Jμ ./ v)
        interaction && (C += Jv' * (Jv ./ (2 .* v.^2)))
        factor = cholesky(Symmetric(C))
        push!(curvatures, Matrix{T}(C))
        push!(terms, ebes.complete_data_loglikelihood[k] + m * log(2π) / 2 -
            sum(log, diag(factor.L)))
    end
    ll = sum(terms; init = zero(T))
    lp = population_logprior(model, θ, σ, ω)
    lp === nothing || lp isa Real || throw(ArgumentError(
        "population_logprior must return a real log density or nothing"))
    active = lp !== nothing
    lp = active ? lp : zero(ll)
    return FOCEEvaluation(ebes.subjects, ll, terms, ll + lp, lp, -2(ll + lp),
        active ? :MAP : :ML, active ? population_prior_provenance(model) : nothing,
        ebes, curvatures)
end

"""
    focei(model, θ0, σ0, ω0; subjects = 1:nsubjects(model), free = nothing,
          options = FOCEOptions()) -> FOCEResult

Estimate population parameters by minimizing the FOCE(I) OFV with NLopt's
BOBYQA, a quadratic-interpolation trust-region algorithm.
The model's θ and σ and the package's log-Cholesky ω are already unconstrained.
Each objective call warm-starts subject EBEs from the preceding successful call.
No population derivatives or η Hessians are taken.
`free` may select one-based positions in the concatenated `(θ, σ, ω)` vector;
other coordinates stay at their starting values.
With a population prior, the outer objective is the MAP point's negative twice
log posterior; the result still reports its likelihood separately.
Covariance families declared by [`fixed_omega`](@ref) keep their fixed parts.
The search covers the estimated standard deviations and correlation of a
partially fixed family; a fully fixed family is not searched. `free` may not
name a position inside a constrained family's ω segment. `ω0` must hold the
fixed values ([`apply_fixed_omega`](@ref)).
"""
function focei(model, θ0::AbstractVector, σ0::AbstractVector, ω0::AbstractVector;
        subjects = nothing, free = nothing, options::FOCEOptions = FOCEOptions())
    layout = nlme_layout(model)
    _require_focei_layout(layout)
    _check_length("θ", θ0, layout.ntheta)
    _check_length("σ", σ0, layout.nsigma)
    _check_length("ω", ω0, nomega(layout))
    constraints = _checked_omega_constraints(model, layout, ω0)
    options.maxeval > 0 || throw(ArgumentError("maxeval must be positive"))
    0 < options.rhoend < options.rhobeg || throw(ArgumentError("require 0 < rhoend < rhobeg"))
    subjects = subjects === nothing ? collect(1:nsubjects(model)) : collect(Int, subjects)
    nθσ = layout.ntheta + layout.nsigma
    # The search vector holds θ, σ and the free Ω coordinates (`_free_omega`).
    x0 = Float64.(vcat(θ0, σ0, _free_omega(constraints, ω0)))
    free_indices = _focei_free_indices(constraints, free, nθσ, nomega(layout), length(x0))
    function assemble(z)
        x = copy(x0)
        x[free_indices] = z
        return x
    end
    split(x) = (view(x, 1:layout.ntheta),
        view(x, layout.ntheta + 1:nθσ),
        _constrained_omega(constraints, view(x, nθσ + 1:length(x))))
    θ, σ, ω = split(x0)
    initial = focei_evaluate(model, θ, σ, ω; subjects,
        interaction = options.interaction, ebe_options = options.ebe_options)
    isfinite(initial.ofv) || throw(DomainError(initial.ofv,
        "FOCE(I) objective is not finite at the starting population parameters"))
    warm = Ref(initial.ebes)
    best_x = Ref(copy(x0))
    best_value = Ref(initial.ofv)
    best_ebes = Ref(initial.ebes)
    calls = Ref(0)
    function objective(z)
        calls[] += 1
        x = assemble(z)
        θx, σx, ωx = split(x)
        value = try
            focei_evaluate(model, θx, σx, ωx; subjects, η0 = warm[],
                interaction = options.interaction, ebe_options = options.ebe_options)
        catch err
            err isa Union{DomainError, OverflowError, LinearAlgebra.PosDefException,
                EBEConvergenceError} || rethrow()
            return 1.0e12 + sum(abs2, x .- x0)
        end
        isfinite(value.ofv) || return 1.0e12 + sum(abs2, x .- x0)
        warm[] = value.ebes
        if value.ofv < best_value[]
            best_x[] = copy(x)
            best_value[] = value.ofv
            best_ebes[] = value.ebes
        end
        return value.ofv
    end
    opt = NLopt.Opt(:LN_BOBYQA, length(free_indices))
    NLopt.min_objective!(opt, (x, grad) -> objective(x))
    NLopt.maxeval!(opt, options.maxeval)
    NLopt.initial_step!(opt, options.rhobeg)
    NLopt.xtol_abs!(opt, options.rhoend)
    min_ofv, zbest, status = NLopt.optimize(opt, x0[free_indices])
    optimum = assemble(zbest)
    if min_ofv > best_value[]
        optimum = best_x[]
    end
    θ̂, σ̂, ω̂ = split(optimum)
    evaluated = focei_evaluate(model, θ̂, σ̂, ω̂; subjects, η0 = best_ebes[],
        interaction = options.interaction, ebe_options = options.ebe_options)
    T = eltype(optimum)
    return FOCEResult(Vector{T}(θ̂), Vector{T}(σ̂), Vector{T}(ω̂),
        Matrix{T}(omega_matrix(layout, ω̂)), evaluated.loglikelihood,
        evaluated.logposterior, evaluated.population_logprior, evaluated.ofv,
        evaluated.objective, evaluated.prior_provenance,
        evaluated.ebes, evaluated.curvatures, calls[],
        status in (:SUCCESS, :STOPVAL_REACHED, :FTOL_REACHED, :XTOL_REACHED),
        (minimum_ofv = min_ofv, status = status), free_indices, options)
end

function _focei_free_indices(::Nothing, free, nθσ, nω, n)
    free_indices = free === nothing ? collect(1:n) : collect(Int, free)
    !isempty(free_indices) && allunique(free_indices) &&
        all(i -> 1 <= i <= n, free_indices) || throw(ArgumentError(
            "free must contain unique positions in 1:$n"))
    return free_indices
end

# Map `free` positions in `(θ, σ, ω)` onto the search vector; every free
# coordinate of a partially fixed family is searched.
function _focei_free_indices(c::_OmegaConstraints, free, nθσ, nω, n)
    position = collect(Int, 1:nθσ)
    partial = Int[]
    cursor = nθσ
    family_of = zeros(Int, nω)
    for (f, fam) in enumerate(c.families)
        r = c.ranges[f]
        family_of[r] .= f
        if fam === nothing
            append!(position, (cursor + 1):(cursor + length(r)))
            cursor += length(r)
        else
            append!(position, zeros(Int, length(r)))
            width = fam.fully_fixed ? 0 :
                count(!, fam.sd_fixed) + (fam.correlation_fixed ? 0 : length(fam.m))
            append!(partial, (cursor + 1):(cursor + width))
            cursor += width
        end
    end
    free === nothing && return collect(1:n)
    requested = collect(Int, free)
    allunique(requested) && all(i -> 1 <= i <= nθσ + nω, requested) || throw(ArgumentError(
        "free must contain unique positions in 1:$(nθσ + nω)"))
    for i in requested
        position[i] == 0 && throw(ArgumentError(
            "free position $i lies in covariance family $(family_of[i - nθσ]), which fixed_omega constrains"))
    end
    free_indices = vcat(position[requested], partial)
    isempty(free_indices) && throw(ArgumentError("free selects no coordinates to estimate"))
    return free_indices
end

_require_focei_layout(::NLMELayout) = nothing
_require_focei_layout(::RaggedNLMELayout) = throw(ArgumentError(
    "FOCE(I) does not yet support RaggedNLMELayout; use posthoc or standard-MH SAEM for nested random effects"))
