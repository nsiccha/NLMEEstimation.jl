# The model protocol. A model is any Julia value; it takes part by adding methods
# to the functions below. Subjects are indexed `1:nsubjects(model)`. θ, σ and η may
# be any `AbstractVector` (views, AD-tracked values), and a batch of random effects
# is an `AbstractMatrix` with one column per subject for `NLMELayout`, or a
# vector of compact subject vectors for `RaggedNLMELayout`. Implementations
# must not restrict element types.
#
# Two evaluation styles exist, chosen by `evaluation_style(model)`. Estimators always
# call the batched entry points (`conditional_loglikelihoods`, …); every entry point
# has a default derived from the other style, so a model implements one style only.

# ── Required ──────────────────────────────────────────────────────────────────

"""
    nsubjects(model) -> Int

**Required.** Number of subjects (independent units carrying their own η_i).
"""
function nsubjects end

"""
    nlme_layout(model) -> NLMELayout or RaggedNLMELayout

**Required.** The parameter layout: lengths of θ and σ and the η block structure.
See [`NLMELayout`](@ref) for the coordinate conventions.
"""
function nlme_layout end

"""
    EvaluationStyle

How a model evaluates its conditional log-density, declared by
[`evaluation_style`](@ref):

- [`PerSubjectEvaluation`](@ref) (default): the model implements
  [`conditional_loglikelihood`](@ref) for one subject at a time.
- [`BatchedEvaluation`](@ref): the model implements
  [`conditional_loglikelihoods`](@ref) for a batch of subjects in one pass, as
  joint-program backends (one log-density over all subjects) do.
"""
abstract type EvaluationStyle end

"""
    PerSubjectEvaluation()

The model evaluates one subject per call: it implements
[`conditional_loglikelihood`](@ref). Batched entry points loop over subjects.
"""
struct PerSubjectEvaluation <: EvaluationStyle end

"""
    BatchedEvaluation()

The model evaluates a batch of subjects per call: it implements
[`conditional_loglikelihoods`](@ref). Per-subject entry points evaluate a batch of
one; derivatives of a batch come from AD of the batch's summed log-density, which
yields every subject's η-gradient at once because subjects are conditionally
independent.
"""
struct BatchedEvaluation <: EvaluationStyle end

"""
    evaluation_style(model) -> EvaluationStyle

**Optional.** Defaults to [`PerSubjectEvaluation`](@ref).
"""
evaluation_style(model) = PerSubjectEvaluation()

"""
    conditional_loglikelihood(model, i, θ, σ, η) -> Real

The conditional log-density `log p(y_i | θ, σ, η_i)` of subject `i`'s observations,
*without* the random-effect prior (which this package owns, see
[`re_logprior`](@ref)). θ and σ are in the model's unconstrained coordinates.

**Required** under [`PerSubjectEvaluation`](@ref); derived from
[`conditional_loglikelihoods`](@ref) under [`BatchedEvaluation`](@ref).

Return `-Inf` where the density is zero (a failed solve, for example); estimators
treat non-finite values as infeasible and report them, never as success.
"""
conditional_loglikelihood(model, i, θ, σ, η) =
    _conditional_loglikelihood(evaluation_style(model), model, i, θ, σ, η)

"""
    conditional_loglikelihoods(model, subjects, θ, σ, H) -> Vector

The conditional log-densities of a batch of subjects: entry `k` is
`log p(y_i | θ, σ, η_i)` for `i = subjects[k]` and `η_i = H[:, k]`, with `H` of
size `neta × length(subjects)` for `NLMELayout`. For `RaggedNLMELayout`, `H[k]`
has length `neta(layout, subjects[k])`.

**Required** under [`BatchedEvaluation`](@ref): a joint-program model places each
column of `H` into its subject's random-effect slots (the other subjects' slots are
irrelevant, by conditional independence) and returns the per-subject terms.
Derived from [`conditional_loglikelihood`](@ref) under
[`PerSubjectEvaluation`](@ref).
"""
conditional_loglikelihoods(model, subjects, θ, σ, H) =
    _conditional_loglikelihoods(evaluation_style(model), model, subjects, θ, σ, H)

_conditional_loglikelihood(::PerSubjectEvaluation, model, i, θ, σ, η) = throw(ArgumentError(
    "$(typeof(model)) declares PerSubjectEvaluation but implements no " *
    "`NLMEEstimation.conditional_loglikelihood(::$(typeof(model)), i, θ, σ, η)`"))
_conditional_loglikelihood(::BatchedEvaluation, model, i, θ, σ, η) =
    only(conditional_loglikelihoods(model, [i], θ, σ, _single_eta_batch(nlme_layout(model), η)))

_single_eta_batch(::NLMELayout, η) = reshape(η, :, 1)
_single_eta_batch(::RaggedNLMELayout, η) = [η]
_eta_column(H::AbstractMatrix, k) = H[:, k]
_eta_column(H::AbstractVector{<:AbstractVector}, k) = H[k]

_conditional_loglikelihoods(::PerSubjectEvaluation, model, subjects, θ, σ, H) =
    [conditional_loglikelihood(model, i, θ, σ, _eta_column(H, k)) for (k, i) in enumerate(subjects)]
_conditional_loglikelihoods(::BatchedEvaluation, model, subjects, θ, σ, H) = throw(ArgumentError(
    "$(typeof(model)) declares BatchedEvaluation but implements no " *
    "`NLMEEstimation.conditional_loglikelihoods(::$(typeof(model)), subjects, θ, σ, H)`"))

# A model may signal that a point lies outside its domain by throwing one of
# these, as a Stan-semantics model does (a matrix exponential of an overflowed
# rate, a Cholesky factor of an indefinite matrix). Estimators reject such a
# trial point like one with a non-finite density; at the point a search starts
# from it remains an error.
const _InfeasibleEvaluation = Union{DomainError, OverflowError, PosDefException}

# `f(x)`, or `nothing` when the evaluation throws `_InfeasibleEvaluation`.
function _feasible(f, x)
    try
        return f(x)
    catch err
        err isa _InfeasibleEvaluation || rethrow()
        return nothing
    end
end

# Conditional log densities at proposed latent effects (MCMC proposals, line-search
# trial points, quadrature points). If the batch throws `_InfeasibleEvaluation`,
# each subject is evaluated alone, and one whose evaluation still throws scores
# -Inf, so only its proposal is rejected.
function _proposal_loglikelihoods(model, subjects, θ, σ, proposal)
    batch = _feasible(H -> conditional_loglikelihoods(model, subjects, θ, σ, H), proposal)
    batch === nothing || return batch
    layout = nlme_layout(model)
    single(i) = η -> only(conditional_loglikelihoods(model, [i], θ, σ,
        _single_eta_batch(layout, η)))
    return [something(_feasible(single(i), _eta_column(proposal, k)), -Inf)
        for (k, i) in enumerate(subjects)]
end

# ── Derivatives: implement them, or declare an AD backend ─────────────────────

"""
    ad_backend(model) -> Union{Nothing, ADTypes.AbstractADType}

**Optional.** The automatic-differentiation backend this package uses, through
DifferentiationInterface, for every derivative method the model does not implement
itself. The default `nothing` declares no backend; calling a missing derivative
method is then an error.

What this package promises and tests is reverse-mode AD with Enzyme,
`ADTypes.AutoEnzyme()`; second-order defaults are forward-over-reverse
Hessian-vector products. Other backends are passed through untested.
"""
ad_backend(model) = nothing

"""
    hessian_ad_backend(model) -> Union{Nothing, ADTypes.AbstractADType}

**Optional.** The backend used for default second-order derivatives (η Hessian
blocks). Defaults to [`ad_backend`](@ref). Return `nothing` to declare that the
model provides no second order, so estimators use their gradient-only paths.
"""
hessian_ad_backend(model) = ad_backend(model)

"""
    provides_eta_hessian(model) -> Bool

**Optional.** Whether η Hessian blocks are available
([`conditional_loglikelihood_hessian_blocks`](@ref)). Defaults to whether a
[`hessian_ad_backend`](@ref) is declared; a model implementing the Hessian methods
itself returns `true`. Second order is only ever requested as per-subject
`neta × neta` blocks, never as a joint matrix.
"""
provides_eta_hessian(model) = hessian_ad_backend(model) !== nothing

"""
    eta_hessian_sparsity(model, i) -> Union{Nothing, AbstractMatrix{Bool}}

**Optional.** Which entries of subject `i`'s conditional η Hessian `∇²η ℓ_i` can be
nonzero: an `n × n` Boolean matrix, usually a `SparseMatrixCSC{Bool}`, with
`n = neta(nlme_layout(model), i)`. Entry `(j, k)` is `true` when η_j and η_k can
enter the same observation's likelihood. The default `nothing` declares nothing,
and the Hessian is treated as dense. A model declares a pattern for every subject
or for none.

With a pattern, the estimators compute the subject's η Hessian as a sparse matrix
from a few batched gradient evaluations, one pair per color of a star coloring
of the pattern (central differences of
[`conditional_loglikelihoods_and_gradients`](@ref)), and factor it sparsely. A
subject of nested random effects, whose units interact only through the effects
above them, then costs memory and time linear in its number of effects; dense
Hessian blocks cost their square. [`empirical_bayes`](@ref) uses
[`NewtonEBE`](@ref) by default for such a model, its result's `hessians` are
sparse, and [`marginal_loglikelihood`](@ref) and [`covariance_step`](@ref) take
their curvature from them. The pattern supersedes
[`conditional_loglikelihood_hessian_blocks`](@ref): a model that declares one is
not asked for dense blocks. [`check_protocol`](@ref) checks a declared pattern
against Hessian-vector products along random directions.

The diagonal and symmetry are implied: `P .| P' .| I` is used.
"""
eta_hessian_sparsity(model, i) = nothing

"""
    conditional_loglikelihood_and_gradient(model, i, θ, σ, η) -> (ℓ, ∇η ℓ)

The conditional log-density of subject `i` and its gradient in η_i. Implement it,
or let it derive from `ad_backend` (per-subject models) or from the batched
gradient (batched models).
"""
conditional_loglikelihood_and_gradient(model, i, θ, σ, η) =
    _and_gradient(evaluation_style(model), model, i, θ, σ, η)

"""
    conditional_loglikelihoods_and_gradients(model, subjects, θ, σ, H) -> (ℓ::Vector, G::Matrix)

Batched values and η-gradients: `G[:, k]` is `∇η ℓ_i` for `i = subjects[k]` at
`H[:, k]`. Under [`BatchedEvaluation`](@ref) the default takes the AD gradient of
`sum(conditional_loglikelihoods(model, subjects, θ, σ, H))` with respect to `H`,
which by conditional independence is every subject's gradient in one pass;
implement it directly to fuse that pass with the values.
"""
conditional_loglikelihoods_and_gradients(model, subjects, θ, σ, H) =
    _and_gradients(evaluation_style(model), model, subjects, θ, σ, H)

"""
    conditional_loglikelihood_gradient_and_hessian(model, i, θ, σ, η) -> (ℓ, ∇η ℓ, ∇²η ℓ)

The conditional log-density of subject `i` with its η gradient and `neta × neta`
η Hessian. Available when [`provides_eta_hessian`](@ref).
"""
conditional_loglikelihood_gradient_and_hessian(model, i, θ, σ, η) =
    _gradient_and_hessian(evaluation_style(model), model, i, θ, σ, η)

"""
    conditional_loglikelihood_hessian_blocks(model, subjects, θ, σ, H) -> Vector{Matrix}

The per-subject η Hessian blocks of a batch: entry `k` is `∇²η ℓ_i` for
`i = subjects[k]` at `H[:, k]`. Available when [`provides_eta_hessian`](@ref).
Under [`BatchedEvaluation`](@ref) the default takes `neta` Hessian-vector products
of the batch's summed log-density, one per η component across all subjects at once.
"""
conditional_loglikelihood_hessian_blocks(model, subjects, θ, σ, H) =
    _hessian_blocks(evaluation_style(model), model, subjects, θ, σ, H)

"""
    conditional_loglikelihood_and_population_gradient(model, i, θ, σ, η) -> (ℓ, ∇θ ℓ, ∇σ ℓ)

**Optional, or provided by [`ad_backend`](@ref).** The conditional log-density of
subject `i` with its gradients in θ and σ at fixed η_i. Used by population-level
estimators, not by `empirical_bayes`.
"""
function conditional_loglikelihood_and_population_gradient(model, i, θ, σ, η)
    backend = _checked_backend(model, ad_backend(model), :conditional_loglikelihood_and_population_gradient)
    ntheta = length(θ)
    x = vcat(θ, σ)
    ℓ, g = DI.value_and_gradient(_population_objective, backend, x,
        DI.Constant(model), DI.Constant(i), DI.Constant(η), DI.Constant(ntheta))
    return ℓ, g[1:ntheta], g[(ntheta + 1):end]
end

"""
    saem_population_gradient_style(model[, numeric_mode]) -> EvaluationStyle

**Optional.** Selects how SAEM computes the conditional-likelihood value and
population-parameter gradient for its numerical M-step. Defaults to
[`evaluation_style`](@ref). A model with [`BatchedEvaluation`](@ref) can return
[`PerSubjectEvaluation`](@ref) here when it implements an efficient
[`conditional_loglikelihood_and_population_gradient`](@ref) for one subject;
SAEM sums those subject values and gradients. This choice affects only that
M-step gradient. Likelihood, η-gradient, and η-Hessian evaluation continue to
use `evaluation_style(model)`.

The optional second argument is `Val(:online)` or `Val(:mixture)`, the resolved
SAEM numerical M-step mode. Override it when one gradient route is efficient
for online updates but costly for the retained-draw mixture. By default both
modes use the one-argument style, preserving existing model methods.
"""
saem_population_gradient_style(model) = evaluation_style(model)
saem_population_gradient_style(model, ::Val) = saem_population_gradient_style(model)

"""
    population_logprior(model, θ, σ, ω) -> Union{Nothing, Real}

Optional log prior on the population coordinates, evaluated **once per fit**, not
once per subject or retained SAEM draw. Return `nothing` (the default) for
maximum likelihood. A model with a prior returns its log density with respect
to the unconstrained `(θ, σ, ω)` coordinates. If the original prior is defined
on transformed parameters, the model must include the transformation's log
absolute Jacobian here. `ω` is the package's log-Cholesky coordinate vector.
The random-effect prior `η ~ N(0, Ω)` is already owned by this package and must
not be repeated here.
"""
population_logprior(model, θ, σ, ω) = nothing

"""
    population_logprior_and_gradient(model, θ, σ, ω)
        -> (logprior, ∇θ logprior, ∇σ logprior, ∇ω logprior)

Optional gradient for a model's [`population_logprior`](@ref). The default uses
`ad_backend(model)` when a prior is present and returns zero gradients when
there is none. Models may implement this method directly, including when their
conditional-likelihood gradient uses a different backend.
"""
function population_logprior_and_gradient(model, θ, σ, ω)
    lp = population_logprior(model, θ, σ, ω)
    lp === nothing && return (zero(promote_type(eltype(θ), eltype(σ), eltype(ω))),
        zero(θ), zero(σ), zero(ω))
    backend = _checked_backend(model, ad_backend(model), :population_logprior_and_gradient)
    nθ, nσ = length(θ), length(σ)
    value, g = DI.value_and_gradient(_population_logprior_objective, backend,
        vcat(θ, σ, ω), DI.Constant(model), DI.Constant(nθ), DI.Constant(nσ))
    return value, g[1:nθ], g[nθ+1:nθ+nσ], g[nθ+nσ+1:end]
end

_population_logprior_objective(x, model, nθ, nσ) =
    population_logprior(model, view(x, 1:nθ), view(x, nθ+1:nθ+nσ),
        view(x, nθ+nσ+1:length(x)))

"""
    population_prior_provenance(model)

Optional, model-supplied description of the population prior. Returned unchanged
in SAEM and FOCE results when a prior is present; defaults to `nothing`.
"""
population_prior_provenance(model) = nothing

# Per-subject models: derivatives of the per-subject density.
function _and_gradient(::PerSubjectEvaluation, model, i, θ, σ, η)
    backend = _checked_backend(model, ad_backend(model), :conditional_loglikelihood_and_gradient)
    return DI.value_and_gradient(_eta_objective, backend, η, _eta_contexts(model, i, θ, σ)...)
end
function _gradient_and_hessian(::PerSubjectEvaluation, model, i, θ, σ, η)
    backend = _checked_backend(model, hessian_ad_backend(model), :conditional_loglikelihood_gradient_and_hessian)
    return DI.value_gradient_and_hessian(_eta_objective, backend, η, _eta_contexts(model, i, θ, σ)...)
end
function _and_gradients(::PerSubjectEvaluation, model, subjects, θ, σ, H)
    pairs = [conditional_loglikelihood_and_gradient(model, i, θ, σ, H[:, k]) for (k, i) in enumerate(subjects)]
    G = similar(H, promote_type(eltype(H), (eltype(last(p)) for p in pairs)...), size(H))
    for (k, p) in enumerate(pairs)
        G[:, k] = last(p)
    end
    return [first(p) for p in pairs], G
end
function _and_gradients(::PerSubjectEvaluation, model, subjects, θ, σ,
        H::AbstractVector{<:AbstractVector})
    pairs = [conditional_loglikelihood_and_gradient(model, i, θ, σ, H[k])
        for (k, i) in enumerate(subjects)]
    return [first(p) for p in pairs], [last(p) for p in pairs]
end
_hessian_blocks(::PerSubjectEvaluation, model, subjects, θ, σ, H) =
    [last(conditional_loglikelihood_gradient_and_hessian(model, i, θ, σ, _eta_column(H, k))) for (k, i) in enumerate(subjects)]

# Batched models: derivatives of the batch's summed density, then per-subject views.
function _and_gradients(::BatchedEvaluation, model, subjects, θ, σ, H)
    backend = _checked_backend(model, ad_backend(model), :conditional_loglikelihoods_and_gradients)
    ℓ = conditional_loglikelihoods(model, subjects, θ, σ, H)
    G = DI.gradient(_batch_objective, backend, H, _batch_contexts(model, subjects, θ, σ)...)
    return ℓ, G
end
function _ragged_ranges(H::AbstractVector{<:AbstractVector})
    ranges = UnitRange{Int}[]
    offset = 0
    for h in H
        push!(ranges, (offset + 1):(offset + length(h)))
        offset += length(h)
    end
    return ranges
end
_ragged_unpack(x, ranges) = [view(x, r) for r in ranges]
_ragged_batch_objective(x, model, subjects, θ, σ, ranges) =
    sum(conditional_loglikelihoods(model, subjects, θ, σ, _ragged_unpack(x, ranges)))
_ragged_batch_contexts(model, subjects, θ, σ, ranges) =
    (DI.Constant(model), DI.Constant(subjects), DI.Constant(θ), DI.Constant(σ), DI.Constant(ranges))

function _and_gradients(::BatchedEvaluation, model, subjects, θ, σ,
        H::AbstractVector{<:AbstractVector})
    backend = _checked_backend(model, ad_backend(model), :conditional_loglikelihoods_and_gradients)
    ranges = _ragged_ranges(H)
    flat = reduce(vcat, H; init=Float64[])
    ℓ = conditional_loglikelihoods(model, subjects, θ, σ, H)
    gradient = DI.gradient(_ragged_batch_objective, backend, flat,
        _ragged_batch_contexts(model, subjects, θ, σ, ranges)...)
    return ℓ, [gradient[r] for r in ranges]
end
function _hessian_blocks(::BatchedEvaluation, model, subjects, θ, σ, H)
    backend = _checked_backend(model, hessian_ad_backend(model), :conditional_loglikelihood_hessian_blocks)
    m = size(H, 1)
    # Tangent k moves η component k of every subject at once; by conditional
    # independence its HVP holds column k of every subject's block.
    tangents = ntuple(m) do c
        V = zero(H)
        V[c, :] .= 1
        V
    end
    hvps = DI.hvp(_batch_objective, backend, H, tangents, _batch_contexts(model, subjects, θ, σ)...)
    return [reduce(hcat, [hvps[c][:, k] for c in 1:m]) for k in eachindex(subjects)]
end
function _hessian_blocks(::BatchedEvaluation, model, subjects, θ, σ,
        H::AbstractVector{<:AbstractVector})
    backend = _checked_backend(model, hessian_ad_backend(model), :conditional_loglikelihood_hessian_blocks)
    ranges = _ragged_ranges(H)
    flat = reduce(vcat, H; init=Float64[])
    m = maximum(length, H; init=0)
    tangents = ntuple(m) do c
        V = zero(flat)
        for r in ranges
            c <= length(r) && (V[r[c]] = 1)
        end
        V
    end
    hvps = DI.hvp(_ragged_batch_objective, backend, flat, tangents,
        _ragged_batch_contexts(model, subjects, θ, σ, ranges)...)
    return [reduce(hcat, [hvps[c][r] for c in 1:length(r)]) for r in ranges]
end
function _and_gradient(::BatchedEvaluation, model, i, θ, σ, η)
    ℓ, G = conditional_loglikelihoods_and_gradients(model, [i], θ, σ,
        _single_eta_batch(nlme_layout(model), η))
    return only(ℓ), _eta_column(G, 1)
end
function _gradient_and_hessian(::BatchedEvaluation, model, i, θ, σ, η)
    ℓ, g = conditional_loglikelihood_and_gradient(model, i, θ, σ, η)
    return ℓ, g, only(conditional_loglikelihood_hessian_blocks(model, [i], θ, σ,
        _single_eta_batch(nlme_layout(model), η)))
end

_eta_objective(η, model, i, θ, σ) = conditional_loglikelihood(model, i, θ, σ, η)
_eta_contexts(model, i, θ, σ) = (DI.Constant(model), DI.Constant(i), DI.Constant(θ), DI.Constant(σ))
_batch_objective(H, model, subjects, θ, σ) = sum(conditional_loglikelihoods(model, subjects, θ, σ, H))
_batch_contexts(model, subjects, θ, σ) =
    (DI.Constant(model), DI.Constant(subjects), DI.Constant(θ), DI.Constant(σ))
_population_objective(x, model, i, η, ntheta) =
    conditional_loglikelihood(model, i, view(x, 1:ntheta), view(x, (ntheta + 1):length(x)), η)

# A declared backend, or an error naming both remedies when none is declared.
_checked_backend(model, ::Nothing, op) = throw(ArgumentError(
    "$(typeof(model)) has no method for `$op` and declares no AD backend. Either implement " *
    "`NLMEEstimation.$op` for it or declare a backend with `NLMEEstimation.ad_backend` " *
    "(or `hessian_ad_backend` for second order)."))
_checked_backend(model, backend::ADTypes.NoAutoDiff, op) = _checked_backend(model, nothing, op)
_checked_backend(model, backend::ADTypes.AbstractADType, op) = backend
_checked_backend(model, backend, op) = throw(ArgumentError(
    "the AD backend declared for $(typeof(model)) is $(repr(backend)), which is not an ADTypes backend"))

# ── Reporting ─────────────────────────────────────────────────────────────────

"""
    parameter_names(model) -> (theta = Vector{String}, sigma = Vector{String}, eta = Vector{String})

**Optional.** Human-readable coordinate names for reporting. Defaults to
`θ1, θ2, …`, `σ1, …`, `η1, …`.
"""
function parameter_names(model)
    layout = nlme_layout(model)
    return _default_parameter_names(layout)
end
function _default_parameter_names(layout::NLMELayout)
    return (theta = ["θ$k" for k in 1:layout.ntheta],
        sigma = ["σ$k" for k in 1:layout.nsigma],
        eta = ["η$k" for k in 1:neta(layout)])
end
function _default_parameter_names(layout::RaggedNLMELayout)
    return (theta = ["θ$k" for k in 1:layout.ntheta],
        sigma = ["σ$k" for k in 1:layout.nsigma],
        eta = [["η$(i).$(k)" for k in 1:neta(layout, i)]
            for i in eachindex(layout.subject_families)])
end

# ── Optional structural traits (used by later estimators) ─────────────────────

"""
    MuReferencing(theta_indices, design)

Mu-referencing declaration (see [`mu_referencing`](@ref)).

- `theta_indices`: the θ coordinates that are mu-referenced.
- `design`: one `neta × length(theta_indices)` matrix `X_i` per subject.

Contract: for every subject `i`, [`conditional_loglikelihood`](@ref) depends on
`θ[theta_indices]` and `η` only through the individual linear predictor
`ϕ_i = X_i * θ[theta_indices] + η`. Equivalently, it is unchanged under
`θ[theta_indices] .+= δ`, `η .-= X_i * δ`. The model applies its own inverse link to
ϕ_i. A zero row of `X_i` marks an η component without a mu-referenced typical value.
"""
struct MuReferencing{M <: AbstractMatrix{<:Real}}
    theta_indices::Vector{Int}
    design::Vector{M}
    function MuReferencing(theta_indices::AbstractVector{<:Integer}, design::AbstractVector{M}) where {M <: AbstractMatrix{<:Real}}
        allunique(theta_indices) ||
            throw(ArgumentError("MuReferencing: theta_indices $theta_indices are not unique"))
        for (i, X) in enumerate(design)
            size(X, 2) == length(theta_indices) || throw(DimensionMismatch(
                "MuReferencing: design[$i] has $(size(X, 2)) columns for $(length(theta_indices)) mu-referenced θ coordinates"))
        end
        return new{M}(collect(Int, theta_indices), collect(design))
    end
end

"""
    mu_referencing(model) -> Union{Nothing, MuReferencing}

**Optional trait.** Declares which θ coordinates enter the model as typical values
of the random effects, `ϕ_i = X_i θ_μ + η_i`. SAEM's closed-form M-step needs it.
Default: `nothing` (not declared).
"""
mu_referencing(model) = nothing

"""
    GaussianObservations()

Observation-model declaration (see [`observation_model`](@ref)): the observations
of subject `i` are independent Gaussians. The model then implements
[`observations`](@ref) and [`observation_moments`](@ref).

Contract: `conditional_loglikelihood(model, i, θ, σ, η)` equals
`Σ_j log N(y_ij | μ_ij, v_ij)` with `y_i = observations(model, i)` and
`(μ_i, v_i) = observation_moments(model, i, θ, σ, η)`.
"""
struct GaussianObservations end

"""
    observation_model(model) -> Union{Nothing, GaussianObservations}

**Optional trait.** Declares the observation-model structure FOCE(I) linearizes
(first order: the Jacobians of the predicted means and variances in η). Default:
`nothing`.
"""
observation_model(model) = nothing

"""
    observations(model, i) -> AbstractVector

Subject `i`'s observations; required when [`observation_model`](@ref) is declared.
"""
function observations end

function observations(model, index)
    index isa Integer && return _observations(evaluation_style(model), model, index)
    index isa AbstractVector{<:Integer} &&
        return _observations_batch(evaluation_style(model), model, index)
    throw(ArgumentError("observations index must be an integer or vector of integers"))
end
_observations(::PerSubjectEvaluation, model, i) = throw(ArgumentError(
    "$(typeof(model)) declares PerSubjectEvaluation but implements no `NLMEEstimation.observations` for subject $i"))
_observations(::BatchedEvaluation, model, i) = only(observations(model, [i]))

"""
    observations(model, subjects::AbstractVector{<:Integer})

Observed vectors in `subjects` order. A batched model implements this method to
obtain all observations in one joint-program query.
"""
_observations_batch(::PerSubjectEvaluation, model, subjects) =
    [observations(model, i) for i in subjects]
_observations_batch(::BatchedEvaluation, model, subjects) = throw(ArgumentError(
    "$(typeof(model)) declares BatchedEvaluation but implements no `NLMEEstimation.observations` for a batch"))

"""
    observation_moments(model, i, θ, σ, η) -> (mean::AbstractVector, variance::AbstractVector)

Predicted means and variances of subject `i`'s observations, differentiable in η;
required when [`observation_model`](@ref) is declared.
"""
function observation_moments end

function observation_moments(model, index, θ, σ, eta)
    index isa Integer && return _observation_moments(evaluation_style(model), model, index, θ, σ, eta)
    index isa AbstractVector{<:Integer} &&
        return _observation_moments(evaluation_style(model), model, index, θ, σ, eta)
    throw(ArgumentError("observation_moments index must be an integer or vector of integers"))
end
_observation_moments(::PerSubjectEvaluation, model, i, θ, σ, η) = throw(ArgumentError(
    "$(typeof(model)) declares PerSubjectEvaluation but implements no `NLMEEstimation.observation_moments` for subject $i"))
_observation_moments(::BatchedEvaluation, model, i, θ, σ, η) =
    only(observation_moments(model, [i], θ, σ, reshape(η, :, 1)))

"""
    observation_moments(model, subjects, θ, σ, H)

One `(mean, variance)` pair per subject in `subjects` order. Column `k` of `H`
is the random effect for `subjects[k]` (entry `k`, a compact vector, under a
[`RaggedNLMELayout`](@ref)). A batched model implements this method for a
joint-program pass.
"""
_observation_moments(::PerSubjectEvaluation, model, subjects::AbstractVector, θ, σ, H) =
    [observation_moments(model, i, θ, σ, _eta_column(H, k)) for (k, i) in enumerate(subjects)]
_observation_moments(::BatchedEvaluation, model, subjects::AbstractVector, θ, σ, H) = throw(ArgumentError(
    "$(typeof(model)) declares BatchedEvaluation but implements no `NLMEEstimation.observation_moments` for a batch"))

"""
    observation_moments_and_jacobians(model, i, θ, σ, η) -> (μ, v, Jμ, Jv)

One subject's observation means, variances, and their Jacobians in η. Jacobian
rows follow observation order. The default derives them with reverse-mode AD
from [`observation_moments`](@ref); a model may implement this directly.
"""
function observation_moments_and_jacobians(model, index, θ, σ, eta)
    index isa Integer && return _single_observation_moments_and_jacobians(model, index, θ, σ, eta)
    index isa AbstractVector{<:Integer} &&
        return _observation_moments_and_jacobians(evaluation_style(model), model, index, θ, σ, eta)
    throw(ArgumentError("observation_moments_and_jacobians index must be an integer or vector of integers"))
end

function _single_observation_moments_and_jacobians(model, i, θ, σ, η)
    μ, v = observation_moments(model, i, θ, σ, η)
    n = length(μ)
    length(v) == n || throw(DimensionMismatch("subject $i has $n means but $(length(v)) variances"))
    backend = _checked_backend(model, ad_backend(model), :observation_moments_and_jacobians)
    T = promote_type(eltype(η), eltype(μ), eltype(v))
    Jμ, Jv = zeros(T, n, length(η)), zeros(T, n, length(η))
    contexts = _eta_contexts(model, i, θ, σ)
    for j in 1:n
        seed = zeros(T, 2n)
        seed[j] = one(T)
        Jμ[j, :] = only(DI.pullback(_single_moment_vector, backend, η, (seed,), contexts...))
        seed = zeros(T, 2n)
        seed[n + j] = one(T)
        Jv[j, :] = only(DI.pullback(_single_moment_vector, backend, η, (seed,), contexts...))
    end
    return μ, v, Jμ, Jv
end

"""
    observation_moments_and_jacobians(model, subjects, θ, σ, H)
        -> Vector{Tuple{μ_i, v_i, Jμ_i, Jv_i}}

Batched observation moments and first-order η Jacobians. Result entry `k`
belongs to `subjects[k]`, with `H[:, k]` as its random effect (`H[k]` under a
[`RaggedNLMELayout`](@ref)). Each Jacobian has
`length(observations(model, subjects)[k])` rows and one column per η component
of that subject. A
joint-program model provides this directly in one batched computation; a
per-subject model uses the per-subject derivative method for each entry.
"""
_observation_moments_and_jacobians(::PerSubjectEvaluation, model, subjects, θ, σ, H) =
    [observation_moments_and_jacobians(model, i, θ, σ, _eta_column(H, k))
        for (k, i) in enumerate(subjects)]
_observation_moments_and_jacobians(::BatchedEvaluation, model, subjects, θ, σ, H) =
    throw(ArgumentError("$(typeof(model)) declares BatchedEvaluation with GaussianObservations " *
        "but implements no `NLMEEstimation.observation_moments_and_jacobians` for a batch"))

_single_moment_vector(η, model, i, θ, σ) =
    vcat(observation_moments(model, i, θ, σ, η)...)
