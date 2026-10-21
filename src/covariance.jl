"""
    FOCEApproximation(; interaction = true, ebe_method = nothing,
                        ebe_options = EBEOptions(gtol = 1e-10))

The FOCE(I) approximation of each subject's marginal log-likelihood, the
per-subject terms of [`focei_evaluate`](@ref), as used by
[`covariance_step`](@ref). `interaction = false` gives FOCE without interaction.
`ebe_method = nothing` selects [`default_ebe_method`](@ref): Newton when the model
provides η Hessian blocks, otherwise BFGS. The default mode tolerance is tighter
than a fit's because finite differences of the objective amplify the error of the
inner mode search. Requires [`GaussianObservations`](@ref) and an
[`NLMELayout`](@ref).
"""
Base.@kwdef struct FOCEApproximation{M <: Union{Nothing, EBEMethod}}
    interaction::Bool = true
    ebe_method::M = nothing
    ebe_options::EBEOptions = EBEOptions(gtol = 1.0e-10)
end

# A marginal approximation prepared at the point of a covariance step. Every
# evaluation warm-starts from the same state, so the per-subject terms are a
# deterministic function of the population coordinates.
struct _PreparedFOCE{M, A <: FOCEApproximation, EM <: EBEMethod, E <: EBEResult}
    model::M
    approximation::A
    method::EM
    center::E
end

# The subject modes at the point report under `progress`; each later
# evaluation's mode search is a transient child of the evaluation's `progress`.
function _prepare_marginal(approximation::FOCEApproximation, model, θ, σ, ω, subjects, η0, rng,
        progress = nothing)
    _require_focei_layout(nlme_layout(model))
    observation_model(model) isa GaussianObservations || throw(ArgumentError(
        "FOCEApproximation requires observation_model(model) = GaussianObservations()"))
    method = approximation.ebe_method === nothing ? default_ebe_method(model) :
        approximation.ebe_method
    center = _run_empirical_bayes(model, θ, σ, ω, subjects, η0, method,
        approximation.ebe_options, true, progress, false)
    return _PreparedFOCE(model, approximation, method, center)
end

_marginal_subjects(prepared::_PreparedFOCE) = prepared.center.subjects

function _subject_marginals(prepared::_PreparedFOCE, θ, σ, ω, progress = nothing)
    a = prepared.approximation
    return _focei_evaluate(prepared.model, θ, σ, ω, prepared.center.subjects,
        prepared.center, a.interaction, prepared.method, a.ebe_options, progress,
        true).subject_loglikelihoods
end

"""
    CovarianceResult

Result of [`covariance_step`](@ref). `point` is the full concatenated population
vector `[θ; σ; ω]` at which the step was taken, in the model's unconstrained
coordinates and the package's log-Cholesky ω. `coordinates` is the same point in
the coordinates the estimators fit: `[θ; σ; free Ω coordinates]`. It equals
`point` unless the model declares [`fixed_omega`](@ref); then a fully fixed
family has no coordinates, and a partially fixed family contributes its
estimated log standard deviations (named `log sd(…)`) and, when its
correlation is estimated, its correlation coordinates `m_jl = L_jl / L_jj` of
the Cholesky factor `L` (named `m(…)`). Every other vector and matrix field is
indexed by the estimated coordinates `free` (positions in `coordinates`),
named in `names`:

- `covariance`, `standard_errors`: the estimated sampling covariance of those
  coordinates and the square roots of its diagonal, for `method` `:rs`
  (`R⁻¹ S R⁻¹`), `:r` (`R⁻¹`) or `:s` (`S⁻¹`).
- `R`: minus the Hessian of the fitted objective, the summed marginal
  log-likelihood plus any [`population_logprior`](@ref) (`objective` is `:ML` or
  `:MAP`); `nothing` for `method = :s`, which does not compute it.
- `S`: `Σᵢ gᵢ gᵢᵀ` over the subjects' marginal log-likelihood gradients `gᵢ`,
  the columns of `subject_gradients`. A population prior is not a subject term
  and enters `R` only.
- `gradient`: the gradient of the fitted objective, close to zero at a converged
  estimate.
- `step`: the central-difference step per coordinate; `evaluations` counts the
  marginal-approximation evaluations.

Use [`delta_method`](@ref) for natural-unit quantities.
"""
struct CovarianceResult{T, A}
    subjects::Vector{Int}
    point::Vector{T}
    coordinates::Vector{T}
    free::Vector{Int}
    names::Vector{String}
    method::Symbol
    approximation::A
    objective::Symbol
    covariance::Matrix{T}
    standard_errors::Vector{T}
    R::Union{Nothing, Matrix{T}}
    S::Matrix{T}
    gradient::Vector{T}
    subject_gradients::Matrix{T}
    step::Vector{T}
    evaluations::Int
    omega_constraints::Union{Nothing, _OmegaConstraints}
end

"""
    CovarianceStepError

Thrown by [`covariance_step`](@ref) when the requested covariance does not exist:
`R` is not positive definite (methods `:rs` and `:r`) or `S` is singular (method
`:s`, for example with fewer subjects than estimated coordinates). The fields
`R`, `S`, `gradient` and `names` hold what was computed, for diagnosis.
"""
struct CovarianceStepError{T} <: Exception
    message::String
    R::Union{Nothing, Matrix{T}}
    S::Matrix{T}
    gradient::Vector{T}
    names::Vector{String}
end

function Base.showerror(io::IO, e::CovarianceStepError)
    print(io, "CovarianceStepError: ", e.message)
    matrix, label = e.R === nothing ? (e.S, "S") : (e.R, "R")
    values = eigen(Symmetric(matrix))
    k = argmin(values.values)
    direction = values.vectors[:, k]
    j = argmax(abs.(direction))
    print(io, "\nSmallest eigenvalue of ", label, ": ", values.values[k],
        "; its eigenvector loads most on ", e.names[j], ".")
end

"""
    covariance_step(model, fit::FOCEResult; method = :rs, step = 1e-3,
                    approximation = FOCEApproximation(interaction = fit.options.interaction))
    covariance_step(model, fit::Union{SAEMResult, RaggedSAEMResult}; method = :rs,
                    step = 1e-3, approximation = <see below>, rng = Random.default_rng())
    covariance_step(model, θ, σ, ω; subjects = 1:nsubjects(model), free = nothing,
                    η0 = nothing, method = :rs, step = 1e-3,
                    approximation = FOCEApproximation(), rng = Random.default_rng(),
                    progress = nothing)
        -> CovarianceResult

The covariance step: the sampling covariance and standard errors of the estimated
population coordinates `(θ, σ, ω)`, in the unconstrained coordinates in which they
are fitted.

`R` is minus the Hessian of the fitted objective and `S` the sum over subjects of
the outer products of their marginal log-likelihood gradients. `method = :rs`
(the default) returns the sandwich `R⁻¹ S R⁻¹`, which remains consistent when the
model's random-effect or residual distribution is misspecified; `:r` returns the
inverse observed information `R⁻¹`, and `:s` the inverse cross-product `S⁻¹`.
With a [`population_logprior`](@ref), `R` includes the prior and `R⁻¹` is the
Laplace approximation of the population posterior covariance at the MAP point.

Both matrices are central finite differences of the per-subject marginal
log-likelihoods of `approximation`, which needs objective values only: no
population derivatives of the model. Each coordinate `j` moves by
`step * max(1, |x_j|)`. `:s` takes `2p + 1` evaluations for `p` estimated
coordinates; `:r` and `:rs` take `2p² + 1`. Every evaluation warm-starts the
subject modes from the same solution at the estimate.

For a FOCE(I) fit, the defaults reuse its subjects, its fitted coordinates
(`fit.free`) and its interaction setting, so the step differentiates the fitted
objective. For an SAEM fit, all `(θ, σ, ω)` are estimated. A model with
[`GaussianObservations`](@ref) and an [`NLMELayout`](@ref) defaults to the FOCE(I)
linearization at the SAEM estimate. Any other model, including every
[`RaggedNLMELayout`](@ref), defaults to [`ImportanceSampling`](@ref). Its draws,
from `rng`, are held fixed across the difference points, so the estimated
marginal is a smooth function of the coordinates. [`AdaptiveGaussHermite`](@ref)
is the deterministic alternative for few random effects per subject. For those
two approximations, perturbations of ω and of mu-referenced θ reuse the
conditional values and cost no model evaluations. `free` selects positions in the
estimated coordinates `[θ; σ; free Ω coordinates]`, which are the concatenated
`(θ, σ, ω)` unless the model declares [`fixed_omega`](@ref) (see
[`CovarianceResult`](@ref)). Fixed covariance parts carry no uncertainty and are
never differentiated; `ω` must hold their values. Throws
[`CovarianceStepError`](@ref) when the requested matrix is not positive definite.

`progress` (a Treebars node or backend symbol) gets a `Covariance step` node: the
subject modes at the point, then a `Marginal likelihoods` counter over the `2p² + 1`
(or `2p + 1`) evaluations. An interrupt raises `Treebars.ProgressInterrupted`.
"""
function covariance_step(model, fit::FOCEResult; approximation = FOCEApproximation(
            interaction = fit.options.interaction), kwargs...)
    return covariance_step(model, fit.theta, fit.sigma, fit.omega;
        subjects = fit.ebes.subjects, free = fit.free, η0 = fit.ebes,
        approximation, kwargs...)
end

function covariance_step(model, fit::Union{SAEMResult, RaggedSAEMResult};
        approximation = _default_saem_approximation(nlme_layout(model), observation_model(model)),
        kwargs...)
    return covariance_step(model, fit.theta, fit.sigma, fit.omega;
        subjects = fit.subjects, η0 = fit.eta, approximation, kwargs...)
end

_default_saem_approximation(::NLMELayout, ::GaussianObservations) = FOCEApproximation()
_default_saem_approximation(layout, observation_model) = ImportanceSampling()

function covariance_step(model, θ::AbstractVector, σ::AbstractVector, ω::AbstractVector;
        subjects = nothing, free = nothing, η0 = nothing, method::Symbol = :rs,
        step::Real = 1.0e-3, approximation = FOCEApproximation(),
        rng::AbstractRNG = Random.default_rng(), progress = nothing)
    method in (:rs, :r, :s) || throw(ArgumentError("method must be :rs, :r or :s, got :$method"))
    isfinite(step) && step > 0 || throw(ArgumentError("step must be finite and positive"))
    layout = nlme_layout(model)
    nθ, nσ = layout.ntheta, layout.nsigma
    _check_length("θ", θ, nθ)
    _check_length("σ", σ, nσ)
    _check_length("ω", ω, nomega(layout))
    constraints = _checked_omega_constraints(model, layout, ω)
    x̂ = Float64.(vcat(θ, σ, ω))
    # The estimated coordinates; `point_of` maps them back to [θ; σ; ω].
    ŷ = Float64.(vcat(θ, σ, _free_omega(constraints, ω)))
    free = free === nothing ? collect(eachindex(ŷ)) : collect(Int, free)
    !isempty(free) && allunique(free) && all(i -> 1 <= i <= length(ŷ), free) ||
        throw(ArgumentError("free must contain unique positions in 1:$(length(ŷ))"))
    names = _estimated_coordinate_names(model, constraints)[free]
    return with_progress(progress; description = "Covariance step") do node
        _covariance_step(model, θ, σ, ω, subjects, η0, method, step, approximation, rng,
            layout, constraints, x̂, ŷ, free, names, node)
    end
end

function _covariance_step(model, θ, σ, ω, subjects, η0, method, step, approximation, rng,
        layout, constraints, x̂, ŷ, free, names, progress)
    nθ, nσ = layout.ntheta, layout.nsigma
    prepared = _prepare_marginal(approximation, model, θ, σ, ω, subjects, η0, rng, progress)
    subject_ids = collect(Int, _marginal_subjects(prepared))
    point_of(y) = _point_from_coordinates(constraints, y, nθ + nσ)
    split(y) = (x = point_of(y); (x[1:nθ], x[nθ+1:nθ+nσ], x[nθ+nσ+1:end]))
    lp0 = population_logprior(model, split(ŷ)...)
    lp0 === nothing || lp0 isa Real || throw(ArgumentError(
        "population_logprior must return a real log density or nothing"))
    p = length(free)
    R, S, G, gradient, h, evaluations = with_progress(progress,
            method == :s ? 2p + 1 : 2p^2 + 1;
            description = "Marginal likelihoods") do counter
        _covariance_matrices(prepared, split, ŷ, free, step, method, lp0, subject_ids,
            model, counter)
    end
    covariance = _covariance_from(Val(method), R, S, G, gradient, names)
    return CovarianceResult(subject_ids, x̂, ŷ, free, names, method, approximation,
        lp0 === nothing ? :ML : :MAP, covariance, sqrt.(diag(covariance)), R, S,
        gradient, G, h, evaluations, constraints)
end

# R and S from central differences of the per-subject marginals, with their
# gradient, steps and evaluation count; each evaluation advances `counter`.
function _covariance_matrices(prepared, split, ŷ, free, step, method, lp0, subject_ids,
        model, counter)
    p = length(free)
    evaluations = Ref(0)
    # Per-subject marginal terms and the population log prior (zero under ML).
    function evaluate(x)
        throw_if_interrupted(counter)
        evaluations[] += 1
        terms = _subject_marginals(prepared, split(x)..., counter)
        length(terms) == length(subject_ids) && all(isfinite, terms) ||
            throw(DomainError(terms, "covariance step: the marginal log-likelihood is " *
                "not finite at population coordinates $x; try a smaller step"))
        lp = lp0 === nothing ? 0.0 : population_logprior(model, split(x)...)
        isfinite(lp) || throw(DomainError(lp,
            "covariance step: the population log prior is not finite at $x"))
        counter === nothing || update_progress!(counter, evaluations[])
        return Vector{Float64}(terms), Float64(lp)
    end
    h = step .* max.(1.0, abs.(ŷ[free]))
    total((terms, lp)) = sum(terms) + lp
    function shifted(signs)
        y = copy(ŷ)
        for (j, s) in signs
            y[free[j]] += s * h[j]
        end
        return y
    end
    center = evaluate(ŷ)
    plus = [evaluate(shifted(((j, 1),))) for j in 1:p]
    minus = [evaluate(shifted(((j, -1),))) for j in 1:p]
    G = Matrix{Float64}(reduce(hcat, [(plus[j][1] - minus[j][1]) / (2h[j]) for j in 1:p])')
    gradient = vec(sum(G; dims = 2)) .+ [(plus[j][2] - minus[j][2]) / (2h[j]) for j in 1:p]
    S = G * G'
    R = nothing
    if method != :s
        R = zeros(p, p)
        for j in 1:p
            R[j, j] = -(total(plus[j]) - 2total(center) + total(minus[j])) / h[j]^2
            for k in 1:j-1
                corners = [total(evaluate(shifted(((j, a), (k, b))))) for (a, b) in
                    ((1, 1), (1, -1), (-1, 1), (-1, -1))]
                R[j, k] = R[k, j] = -(corners[1] - corners[2] - corners[3] + corners[4]) /
                    (4h[j] * h[k])
            end
        end
    end
    return R, S, G, gradient, h, evaluations[]
end

_point_from_coordinates(::Nothing, y, nθσ) = y
_point_from_coordinates(c::_OmegaConstraints, y, nθσ) =
    vcat(y[1:nθσ], _constrained_omega(c, y[nθσ+1:end]))

function _covariance_from(::Val{:s}, R, S, G, gradient, names)
    F = cholesky(Symmetric(S); check = false)
    issuccess(F) || throw(CovarianceStepError(
        "S is singular; method :s needs at least as many informative subjects as estimated coordinates",
        R, S, gradient, names))
    return Matrix(inv(F))
end

function _inverse_R(R, S, gradient, names)
    F = cholesky(Symmetric(R); check = false)
    issuccess(F) || throw(CovarianceStepError(
        "R is not positive definite; the estimate is not a strict local optimum of the objective in the estimated coordinates",
        R, S, gradient, names))
    return Matrix(inv(F))
end

_covariance_from(::Val{:r}, R, S, G, gradient, names) = _inverse_R(R, S, gradient, names)

function _covariance_from(::Val{:rs}, R, S, G, gradient, names)
    A = _inverse_R(R, S, gradient, names) * G
    return A * A'
end

"""
    delta_method(result::CovarianceResult, f; jacobian = nothing)
        -> (estimate, covariance, standard_errors)

First-order (delta-method) uncertainty of natural-unit quantities `f(x)`, a real
vector function of the full population vector `x = [θ; σ; ω]` (`result.point`).
`jacobian(x)` may supply `∂f/∂x` as a `length(f(x)) × length(x)` matrix;
otherwise the Jacobian is taken by central differences. Coordinates outside
`result.free`, and covariance parts held by [`fixed_omega`](@ref), carry no
uncertainty; the Jacobian is taken with respect to `result.coordinates`, mapped
through the fixed parts. For example, with `nθ = length(θ)` and
`nσ = length(σ)`, the covariance of `exp.(θ)` and of the Ω entries is

    delta_method(result, x -> exp.(x[1:nθ]))
    delta_method(result, x -> omega_elements(layout, x[nθ+nσ+1:end]))
"""
function delta_method(result::CovarianceResult, f; jacobian = nothing)
    x = result.point
    value = f(x)
    value isa AbstractVector{<:Real} || throw(ArgumentError(
        "delta_method: f must return a real vector, got $(typeof(value))"))
    J = jacobian === nothing ? _central_jacobian(result.omega_constraints, f, result, length(value)) :
        _coordinate_jacobian(result.omega_constraints, jacobian(x), result)
    size(J) == (length(value), length(result.free)) || throw(DimensionMismatch(
        "delta_method: the Jacobian has size $(size(J)); expected ($(length(value)), $(length(x))) before selecting free coordinates"))
    covariance = J * result.covariance * J'
    covariance = (covariance + covariance') / 2
    return (estimate = collect(value), covariance = covariance,
        standard_errors = sqrt.(diag(covariance)))
end

_central_jacobian(::Nothing, f, result, m) = _central_jacobian(f, result.point, result.free, m)
function _central_jacobian(c::_OmegaConstraints, f, result, m)
    nθσ = length(result.point) - last(last(c.ranges))
    return _central_jacobian(y -> f(_point_from_coordinates(c, y, nθσ)),
        result.coordinates, result.free, m)
end

_coordinate_jacobian(::Nothing, J, result) = J[:, result.free]
# Chain rule through the fixed parts: ∂f/∂y = ∂f/∂x ∂x/∂y.
function _coordinate_jacobian(c::_OmegaConstraints, J, result)
    x, y = result.point, result.coordinates
    nω = last(last(c.ranges))
    nθσ = length(x) - nω
    size(J, 2) == length(x) || throw(DimensionMismatch(
        "delta_method: the Jacobian has $(size(J, 2)) columns; expected $(length(x))"))
    D = zeros(length(x), length(y))
    for i in 1:nθσ
        D[i, i] = 1
    end
    ω = x[(nθσ + 1):end]
    for r in 1:nω
        D[nθσ + r, (nθσ + 1):end] = _free_omega_gradient(c, ω, [i == r ? 1.0 : 0.0 for i in 1:nω])
    end
    return J * D[:, result.free]
end

function _central_jacobian(f, x, free, m)
    J = zeros(m, length(free))
    for (c, j) in enumerate(free)
        h = cbrt(eps(Float64)) * max(1.0, abs(x[j]))
        up, down = copy(x), copy(x)
        up[j] += h
        down[j] -= h
        J[:, c] = (f(up) - f(down)) / (2h)
    end
    return J
end

"""
    omega_elements(layout, ω) -> Vector

The entries of Ω on the covariance scale, in one-to-one correspondence with the
log-Cholesky coordinates ω: block by block (family by family for a
[`RaggedNLMELayout`](@ref)), the lower triangle in row-major order. Diagonal
positions are variances and the others covariances. Pair it with
[`delta_method`](@ref) for standard errors of Ω.
"""
function omega_elements(layout::NLMELayout, ω::AbstractVector)
    Ω = omega_matrix(layout, ω)
    eta_ranges, _ = _block_ranges(layout)
    return [Ω[er[j], er[l]] for er in eta_ranges for j in eachindex(er) for l in 1:j]
end

omega_elements(layout::RaggedNLMELayout, ω::AbstractVector) =
    [Ω[j, l] for (k, Ω) in zip(layout.family_sizes, omega_matrix(layout, ω))
        for j in 1:k for l in 1:j]

_estimated_coordinate_names(model, ::Nothing) = _population_coordinate_names(model)
function _estimated_coordinate_names(model, c::_OmegaConstraints)
    layout = nlme_layout(model)
    full = _population_coordinate_names(model)
    nθσ = layout.ntheta + layout.nsigma
    labels = _omega_margin_labels(layout, parameter_names(model).eta)
    names = full[1:nθσ]
    for (f, fam) in enumerate(c.families)
        if fam === nothing
            append!(names, full[nθσ .+ c.ranges[f]])
        elseif !fam.fully_fixed
            k = c.sizes[f]
            append!(names, ["log sd($(labels[f][j]))" for j in 1:k if !fam.sd_fixed[j]])
            fam.correlation_fixed || append!(names,
                ["m($(labels[f][j]),$(labels[f][l]))" for j in 1:k for l in 1:(j - 1)])
        end
    end
    return names
end

_omega_margin_labels(layout::NLMELayout, eta) =
    [[string(eta[i]) for i in r] for r in first(_block_ranges(layout))]
_omega_margin_labels(layout::RaggedNLMELayout, eta) =
    [["$f[$j]" for j in 1:k] for (f, k) in enumerate(layout.family_sizes)]

# Names of the concatenated population coordinates [θ; σ; ω]. The ω entry for
# Cholesky position (j, l) of a block is named by the η components it couples.
function _population_coordinate_names(model)
    layout = nlme_layout(model)
    names = parameter_names(model)
    _check_length("parameter_names(model).theta", names.theta, layout.ntheta)
    _check_length("parameter_names(model).sigma", names.sigma, layout.nsigma)
    return vcat(String.(names.theta), String.(names.sigma),
        _omega_coordinate_names(layout, names.eta))
end

function _omega_coordinate_names(layout::NLMELayout, eta)
    _check_length("parameter_names(model).eta", eta, neta(layout))
    eta_ranges, _ = _block_ranges(layout)
    return ["ω[$(eta[er[j]]),$(eta[er[l]])]" for er in eta_ranges
        for j in eachindex(er) for l in 1:j]
end

_omega_coordinate_names(layout::RaggedNLMELayout, eta) =
    ["ω$f[$j,$l]" for (f, k) in enumerate(layout.family_sizes) for j in 1:k for l in 1:j]
