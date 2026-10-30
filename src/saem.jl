"""
    SAEMOptions(; iterations=300, burnin=150, mcmc_burnin=20,
                 mh_rounds=1, rw_scale=0.5, adapt_iterations=150,
                 step_exponent=0.8, numeric_maxiter=12, mstep_interval=1,
                 numeric_mstep=:auto,
                 omega_floor=1e-8, omega_annealing=0.95,
                 minibatch_size=nothing)

Settings for [`saem`](@ref). The first `burnin` stochastic-approximation
iterations use step size one; subsequent sizes are `(k-burnin)^(-step_exponent)`.
`mcmc_burnin` runs kernel sweeps at the starting population parameters without
updating them. With `StandardSAEMMH`, each round makes one prior-independence
and one joint random-walk proposal for every subject, then one random-walk
proposal per η coordinate, each evaluated in one batch. The joint random-walk
scale adapts toward 0.3 acceptance and each coordinate's scale toward 0.4 for
the first `adapt_iterations` sweeps; then they are frozen. With
`WarmupHMCNUTS`, each round makes one NUTS transition per subject with a kernel
that adapts on every transition.
`omega_floor` is a covariance eigenvalue floor in the M-step. With few subjects
for the mu-referenced coefficients, the covariance that maximizes an M-step's
Gaussian term can be singular; its closed-form update moves towards the floor
by at most 100 coordinate updates per M-step, so Ω may approach it over several
iterations.
`omega_annealing=0.95` anneals Ω while γ = 1, as Monolix and saemix do: an
M-step may shrink no covariance family by more than that factor in any
direction (its new Ω is at least `omega_annealing` times the previous one in
the matrix order). Each such M-step sets Ω from one posterior draw per subject,
so Ω wanders, and where it wanders towards a singular correlation, kernels
whose proposals follow Ω cannot leave it. `omega_annealing=0` turns it off
and reproduces fits from before it was introduced.
`numeric_mstep=:auto` uses an online update for remaining σ coordinates when
all θ coordinates have a closed-form `MuReferencing` update; otherwise it uses
the historical running-mixture objective. Either numerical update also moves
the mu-referenced θ, holding each sampled η fixed. `:online` explicitly applies
the online update to every remaining θ and σ coordinate: it optimizes the
current complete-data conditional likelihood, then moves toward that optimizer by
the SA step size, retaining one sampled latent-effect matrix. Its use for a
structural θ without a declared sufficient statistic is an approximation.
`:mixture` optimizes the exact running conditional-density mixture and retains
every sampled matrix. The default
`mstep_interval=1` updates population parameters every SA iteration; a larger
interval is an approximate, lower-frequency variant in `:online` mode.
With `DualAveragingMALA`, each round makes one batched MALA proposal and
updates a separate subject step size throughout the run. Set `mcmc_burnin=0`
to omit a separate initial MCMC phase; `burnin` still controls the SA step sizes.

`minibatch_size=nothing` moves every subject's chain at every iteration. An
integer `b` runs minibatch SAEM (Kuhn, Matias & Rebafka 2020): each SA
iteration draws `b` of the `n` fitted subjects uniformly at random without
replacement, and only their chains move. The `mcmc_burnin` sweeps still move
every chain (and the first iteration does when `mcmc_burnin = 0`), so that each
starts from its conditional posterior at the starting estimate; from chains
left at their start, the first γ = 1 M-steps would shrink Ω, and with it every
proposal scaled by Ω. The other subjects
keep their last sampled latent effects, which stay in the
stochastic-approximation statistics, so the Gaussian block uses every subject
without evaluating the model. The numerical update evaluates the batch alone:
each M-step takes one Robbins–Monro step `min(γ, 4b/n) M ĝ` along the batch's
score `ĝ`, its conditional log density scaled to all `n` subjects. The cap
limits a step to four times the batch's share of a Newton step, so one pass over
the subjects nearly solves the update, as a full-batch search does, without the
batch's whole sampling error. `M` is the inverse curvature of the full
objective, estimated from every subject's chain at M-steps 1, 2, 4, 8, …. The
step's mean is the full score, so the estimates converge to the same fixed
point as full-batch SAEM, where a search to each batch's optimum would converge
to a point biased by order `1/b`. `numeric_mstep` must be `:auto`; the result
reports `:minibatch`, and `numeric_maxiter` is unused. Each chain moves in only
a fraction `b / n` of the iterations. `b` equal to `n` is the full-batch
algorithm.
"""
Base.@kwdef struct SAEMOptions
    iterations::Int = 300
    burnin::Int = 150
    mcmc_burnin::Int = 20
    mh_rounds::Int = 1
    rw_scale::Float64 = 0.5
    adapt_iterations::Int = 150
    step_exponent::Float64 = 0.8
    numeric_maxiter::Int = 12
    mstep_interval::Int = 1
    numeric_mstep::Symbol = :auto
    omega_floor::Float64 = 1.0e-8
    omega_annealing::Float64 = 0.95
    minibatch_size::Union{Nothing, Int} = nothing
end

abstract type AbstractSAEMKernel end

"""
    StandardSAEMMH()

The standard SAEM per-subject kernel: per subject and MH round, one
prior-independence proposal, one covariance-scaled random-walk Metropolis
proposal, and one random-walk proposal per η coordinate, each coordinate with
its own adapted scale (saemix's componentwise kernel). The coordinate sweep
keeps a subject's chain moving when its posterior widths differ widely across
coordinates, which a single joint scale cannot follow.
"""
struct StandardSAEMMH <: AbstractSAEMKernel end

"""
    DualAveragingMALA(; metric=:prior, initial_stepsize=0.1,
                        target_acceptance_rate=0.574,
                        gamma=0.05, t0=10.0,
                        min_stepsize=1e-5, max_stepsize=10.0,
                        curvature_floor=1e-4)

MALA with one independent dual-averaging step size per subject. `metric=:prior`
uses the current Ω as proposal covariance. `metric=:hessian` obtains a fixed
per-subject covariance from the initial regularized local posterior Hessian,
falling back to Ω if the model has no Hessian. That fallback stays the starting
Ω: when a variance component then shrinks, the step size shrinks with its
posterior, and the chains can stop moving. Gradients are evaluated in
subject batches. The source gradient is reused after each accept or reject
until a population M-step changes the target. Adaptation continues at every
transition, including during population updates. This changing-target, continuously adapted mode is
experimental; its transition is Metropolis corrected at each fixed population
state, but a full SAEM convergence guarantee is not implied.
"""
Base.@kwdef struct DualAveragingMALA <: AbstractSAEMKernel
    metric::Symbol = :prior
    initial_stepsize::Float64 = 0.1
    target_acceptance_rate::Float64 = 0.574
    gamma::Float64 = 0.05
    t0::Float64 = 10.0
    min_stepsize::Float64 = 1.0e-5
    max_stepsize::Float64 = 10.0
    curvature_floor::Float64 = 1.0e-4
end

mutable struct _MALADAState
    count::Int
    hbar::Float64
    log_stepsize::Float64
    log_mu::Float64
    accepted::Int
end

# `valid[j]` says whether column `j` holds subject `j`'s complete-data value and
# gradient at the current population parameters. A minibatch round moves some
# subjects only, so validity is per subject. `G` has the layout's η shape.
mutable struct _MALACache{G}
    f::Vector{Float64}
    G::G
    valid::BitVector
end

_saem_mala_cache(H::AbstractMatrix) =
    _MALACache(zeros(size(H, 2)), zeros(size(H)), falses(size(H, 2)))

function _saem_mala_refresh_prior!(cache::_MALACache, H::AbstractMatrix, old_prior,
        new_prior, subjects)
    any(cache.valid) || return nothing
    for j in axes(H, 2)
        cache.valid[j] || continue
        cache.f[j] += re_logprior(new_prior, view(H, :, j)) -
            re_logprior(old_prior, view(H, :, j))
    end
    # Invalid columns are recomputed before they are used again.
    cache.G .+= (old_prior.precision - new_prior.precision) * H
    return nothing
end

function _saem_check_mala(k::DualAveragingMALA)
    k.metric in (:prior, :hessian) || throw(ArgumentError(
        "MALA metric must be :prior or :hessian"))
    isfinite(k.initial_stepsize) && k.initial_stepsize > 0 ||
        throw(ArgumentError("initial_stepsize must be finite and positive"))
    isfinite(k.target_acceptance_rate) && 0 < k.target_acceptance_rate < 1 ||
        throw(ArgumentError("target_acceptance_rate must be strictly between 0 and 1"))
    isfinite(k.gamma) && k.gamma > 0 || throw(ArgumentError("gamma must be finite and positive"))
    isfinite(k.t0) && k.t0 > 0 || throw(ArgumentError("t0 must be finite and positive"))
    isfinite(k.min_stepsize) && 0 < k.min_stepsize < k.max_stepsize &&
        isfinite(k.max_stepsize) || throw(ArgumentError(
        "require finite 0 < min_stepsize < max_stepsize"))
    isfinite(k.curvature_floor) && k.curvature_floor > 0 || throw(ArgumentError(
        "curvature_floor must be finite and positive"))
    return nothing
end

function _saem_mala_states(k::DualAveragingMALA, n::Int)
    e = clamp(k.initial_stepsize, k.min_stepsize, k.max_stepsize)
    return [_MALADAState(0, 0.0, log(e), log(10 * e), 0) for _ in 1:n]
end

function _saem_mala_metrics(k::DualAveragingMALA, model, ids, θ, σ, prior, H)
    k.metric == :prior && return nothing
    return [_saem_mala_metric(model, i, θ, σ, prior, copy(_column(H, j)), k.curvature_floor)
        for (j, i) in enumerate(ids)]
end

_saem_mala_metric(model, i, θ, σ, prior::NormalREPrior, η, floor) =
    cholesky(Symmetric(_saem_hessian_metric(model, i, θ, σ, prior, η;
        curvature_floor = floor))).L

# A MALA proposal covariance Σ = L Lᵀ, given by its lower factor L: a dense
# triangular matrix, or a ragged subject's block-diagonal prior factor
# (`_RaggedSubjectPrior`, applied block by block).
_proposal_factor(prior::NormalREPrior) = prior.cholesky_factor
_proposal_factor(prior::_RaggedSubjectPrior) = prior
_metric_times(L::AbstractMatrix, g) = L * (L' * g)
_factor_times(L::AbstractMatrix, z) = L * z
_factor_solve(L::AbstractMatrix, x) = L \ x
_metric_times(p::_RaggedSubjectPrior, g) = _covariance_times(p, g)
_factor_solve(p::_RaggedSubjectPrior, x) = _blockwise(\, p, p.prior.cholesky_factors, x)
_factor_solve(p::NormalREPrior, x) = p.cholesky_factor \ x

_saem_mala_noise(rng, H::AbstractMatrix) = randn(rng, size(H)...)

# Shape checks on a batch of η-shaped values: a matrix with one column per
# subject, or one vector per subject.
_nchains(H::AbstractMatrix) = size(H, 2)
_nchains(H::AbstractVector{<:AbstractVector}) = length(H)
_same_shape(A::AbstractMatrix, B::AbstractMatrix) = size(A) == size(B)
_same_shape(A::AbstractVector{<:AbstractVector}, B::AbstractVector{<:AbstractVector}) =
    length(A) == length(B) && all(((a, b),) -> length(a) == length(b), zip(A, B))
_same_shape(A, B) = false
_all_finite(G::AbstractMatrix) = all(isfinite, G)
_all_finite(G::AbstractVector{<:AbstractVector}) = all(g -> all(isfinite, g), G)

function _saem_mala_adapt!(state::_MALADAState, k::DualAveragingMALA, alpha)
    state.count += 1
    t = state.count
    state.hbar += (k.target_acceptance_rate - alpha - state.hbar) / (t + k.t0)
    raw = clamp(state.log_mu - sqrt(t) * state.hbar / k.gamma,
        log(k.min_stepsize), log(k.max_stepsize))
    state.log_stepsize = raw
    return nothing
end

function _saem_mala_logratio(f0, f1, η0, η1, g0, g1, L, stepsize)
    half_e2 = 0.5 * stepsize^2
    forward = _factor_solve(L, η1 - η0 - half_e2 * _metric_times(L, g0))
    reverse = _factor_solve(L, η0 - η1 - half_e2 * _metric_times(L, g1))
    return f1 - f0 + (sum(abs2, forward) - sum(abs2, reverse)) /
        (2 * stepsize^2)
end

function _saem_mala_proposal_gradients(model, subjects, θ, σ, prior, proposal, H)
    n = _nchains(H)
    rejected = falses(n)
    for j in 1:n
        if !all(isfinite, _column(proposal, j))
            rejected[j] = true
            _set_column!(proposal, j, copy(_column(H, j)))
        end
    end
    first_error = nothing
    f1 = G1 = nothing
    try
        _, f1, G1 = _complete_gradients(model, subjects, θ, σ, prior, proposal)
    catch err
        first_error = err
    end
    if first_error !== nothing || f1 === nothing || length(f1) != n ||
            !_same_shape(G1, H) || !all(isfinite, f1) || !_all_finite(G1)
        # Most proposals need only the two batched value/gradient calls. If one
        # proposal leaves the model's domain, identify it with a value batch,
        # replace it by its current position, then recompute the safe batch.
        values = _proposal_loglikelihoods(model, subjects, θ, σ, proposal)
        length(values) == n || throw(DimensionMismatch(
            "MALA proposal returned $(length(values)) values for $n subjects"))
        for j in 1:n
            invalid = !isfinite(values[j]) ||
                !isfinite(_subject_logprior(prior, subjects[j], _column(proposal, j)))
            if !invalid && G1 !== nothing && _same_shape(G1, H)
                invalid = !all(isfinite, _column(G1, j))
            end
            if invalid
                rejected[j] = true
                _set_column!(proposal, j, copy(_column(H, j)))
            end
        end
        any(rejected) || (first_error === nothing ?
            throw(DomainError(f1, "MALA proposed values or gradients are invalid")) :
            throw(first_error))
        _, f1, G1 = _complete_gradients(model, subjects, θ, σ, prior, proposal)
    end
    length(f1) == n && _same_shape(G1, H) && all(isfinite, f1) &&
        _all_finite(G1) || throw(DomainError(f1,
        "MALA proposed values or gradients remain invalid after rejection repair"))
    return f1, G1, rejected
end

function _saem_mala_round!(rng, kernel::DualAveragingMALA, states,
        model, subjects, θ, σ, prior, H; adapt::Bool=true,
        metrics=nothing, cache=nothing)
    n = _nchains(H)
    if cache !== nothing && all(cache.valid)
        f0, G0 = cache.f, cache.G
    else
        _, f0, G0 = _complete_gradients(model, subjects, θ, σ, prior, H)
        if cache !== nothing
            cache.f .= f0
            cache.G .= G0
            cache.valid .= true
            f0, G0 = cache.f, cache.G
        end
    end
    _same_shape(G0, H) && length(f0) == n && all(isfinite, f0) && _all_finite(G0) ||
        throw(DomainError(f0, "MALA current values or gradients are invalid"))
    noise = _saem_mala_noise(rng, H)
    factors = [metrics === nothing ? _proposal_factor(_subject_prior(prior, i)) : metrics[j]
        for (j, i) in enumerate(subjects)]
    proposal = similar(H)
    for j in 1:n
        L = factors[j]
        e = exp(states[j].log_stepsize)
        _set_column!(proposal, j, _column(H, j) + 0.5 * e^2 *
            _metric_times(L, _column(G0, j)) +
            e * _factor_times(L, _column(noise, j)))
    end
    f1, G1, rejected = _saem_mala_proposal_gradients(model, subjects, θ, σ,
        prior, proposal, H)
    for j in 1:n
        L = factors[j]
        e = exp(states[j].log_stepsize)
        lr = rejected[j] ? -Inf : _saem_mala_logratio(f0[j], f1[j],
            _column(H, j), _column(proposal, j), _column(G0, j),
            _column(G1, j), L, e)
        isnan(lr) && throw(DomainError(lr, "MALA log acceptance ratio is invalid"))
        if log(rand(rng)) < min(0.0, lr)
            _set_column!(H, j, _column(proposal, j))
            states[j].accepted += 1
            if cache !== nothing
                cache.f[j] = f1[j]
                _set_column!(cache.G, j, _column(G1, j))
            end
        end
        adapt && _saem_mala_adapt!(states[j], kernel, exp(min(0.0, lr)))
    end
    return nothing
end

# One MALA round over the chains at positions `active`: their columns, step-size
# states, metrics and cached values are taken out, moved and put back.
function _saem_mala_batch_round!(rng, kernel::DualAveragingMALA, states, model, ids,
        θ, σ, prior, H, active; metrics, cache)
    Hb = _columns(H, active)
    sub = _MALACache(cache.f[active], _columns(cache.G, active), cache.valid[active])
    _saem_mala_round!(rng, kernel, states[active], model, ids[active], θ, σ, prior, Hb;
        metrics = metrics === nothing ? nothing : metrics[active], cache = sub)
    _set_columns!(H, active, Hb)
    cache.f[active] = sub.f
    _set_columns!(cache.G, active, sub.G)
    cache.valid[active] = sub.valid
    return nothing
end

"""
    WarmupHMCNUTS(; target_acceptance_rate=0.8, max_tree_depth=10)

One continuously adapting NUTS chain per subject, WarmupHMC's
`ContinuousNUTSState`. Pathfinder from the subject's starting η at the starting
population parameters gives the first position and the first diagonal metric,
and every transition then adapts the metric and the step size from its leapfrog
leaves, for the whole fit, as θ, σ and Ω move. There is no warmup phase and no
η Hessian. `target_acceptance_rate` must lie strictly between zero and one.
"""
Base.@kwdef struct WarmupHMCNUTS <: AbstractSAEMKernel
    target_acceptance_rate::Float64 = 0.8
    max_tree_depth::Int = 10
end

"""
    SAEMResult

Population estimate from [`saem`](@ref). `eta` holds the final subject-chain
states, one column per entry of `subjects`. `acceptance` gives prior and
joint random-walk and per-coordinate random-walk acceptance fractions
(`prior`, `random_walk`, `coordinates`); `rw_scale` is the final joint proposal
scale.
These MH-only fields are `missing` for other kernels. `kernel` identifies the
method and `nuts_divergences` counts its NUTS transitions that diverged.
For MALA, `mala_stepsizes` and `mala_acceptance` hold per-subject final values;
`mala_adaptation_steps` counts updates per subject, the fewest any subject
received under `minibatch_size`. The acceptance fractions are over every
subject's proposals, so under `minibatch_size` over the batches'. `numeric_mstep` records
the resolved numerical M-step mode (`:online` or `:mixture`, `:minibatch` under
`minibatch_size`). `objective` is
`:ML` without a model population prior and `:MAP` otherwise. The result is a
population point estimate, not a draw from a population posterior.
`iterations` counts the completed SA iterations; `interrupted` is `true` when a
Treebars interrupt request (see `progress` in [`saem`](@ref)) stopped the run before
`SAEMOptions.iterations`, so the estimate is that of the last completed iteration.
The acceptance and MALA fields are `NaN` when no kernel sweep ran.
"""
struct SAEMResult{T, A, R, P}
    subjects::Vector{Int}
    theta::Vector{T}
    sigma::Vector{T}
    omega::Vector{T}
    omega_matrix::Matrix{T}
    eta::Matrix{T}
    acceptance::A
    rw_scale::R
    kernel::Symbol
    nuts_divergences::Int
    mala_stepsizes::Union{Missing, Vector{Float64}}
    mala_acceptance::Union{Missing, Vector{Float64}}
    mala_adaptation_steps::Int
    numeric_mstep::Symbol
    objective::Symbol
    population_logprior::T
    prior_provenance::P
    iterations::Int
    interrupted::Bool
end

"""
    RaggedSAEMResult

SAEM estimate for a [`RaggedNLMELayout`](@ref). `eta[k]` is subject
`subjects[k]`'s compact latent vector. `omega_matrix[f]` is the shared
covariance of random-effect family `f`. The other fields are those of
[`SAEMResult`](@ref): `acceptance` and `rw_scale` for the MH kernel, whose
per-coordinate acceptance is per family coordinate, `nuts_divergences` for
NUTS, the `mala_` fields for MALA, `objective` (`:MAP` with a population prior,
`:ML` without one), `iterations` and `interrupted`.
"""
struct RaggedSAEMResult{T, A, R, P}
    subjects::Vector{Int}
    theta::Vector{T}
    sigma::Vector{T}
    omega::Vector{T}
    omega_matrix::Vector{Matrix{T}}
    eta::Vector{Vector{T}}
    acceptance::A
    rw_scale::R
    kernel::Symbol
    nuts_divergences::Int
    mala_stepsizes::Union{Missing, Vector{Float64}}
    mala_acceptance::Union{Missing, Vector{Float64}}
    mala_adaptation_steps::Int
    numeric_mstep::Symbol
    objective::Symbol
    population_logprior::T
    prior_provenance::P
    iterations::Int
    interrupted::Bool
end

# A subject's NUTS target: its conditional density at θ and σ (`conditional`) plus
# its own η prior (`_subject_prior`).
struct _SAEMSubjectTarget{C, P}
    conditional::C
    prior::P
end

LDP.dimension(target::_SAEMSubjectTarget) = neta(target.prior)
LDP.capabilities(::Type{<:_SAEMSubjectTarget}) = LDP.LogDensityOrder{1}()
LDP.logdensity(target::_SAEMSubjectTarget, η) =
    _conditional_value(target.conditional, η) + re_logprior(target.prior, η)
function LDP.logdensity_and_gradient(target::_SAEMSubjectTarget, η)
    ℓ, g = _conditional_value_and_gradient(target.conditional, η)
    return ℓ + re_logprior(target.prior, η), g - _precision_times(target.prior, η)
end

# Subject `subject`'s conditional density, from its own model call. A position
# outside the model's domain (its evaluation throws `_InfeasibleEvaluation`) has
# conditional log-density -Inf and a zero conditional gradient, as if the model had
# returned -Inf there.
struct _DirectConditional{M}
    model::M
    subject::Int
    θ::Vector{Float64}
    σ::Vector{Float64}
end
_conditional_value(c::_DirectConditional, η) = something(_feasible(η ->
    conditional_loglikelihood(c.model, c.subject, c.θ, c.σ, η), η), -Inf)
_conditional_value_and_gradient(c::_DirectConditional, η) = something(_feasible(η ->
    conditional_loglikelihood_and_gradient(c.model, c.subject, c.θ, c.σ, η), η),
    (-Inf, zero(η)))

_saem_target(model, subject, θ, σ, prior) = _SAEMSubjectTarget(
    _DirectConditional(model, subject, copy(θ), copy(σ)), _subject_prior(prior, subject))

# Chain `chain`'s conditional density, evaluated by `_saem_chains` in a batch with
# the other chains': the request goes to `requests`, the answer comes from `reply`.
struct _LockstepConditional
    chain::Int
    requests::Channel{Tuple{Int, Symbol, Any}}
    reply::Channel{Any}
end
function _conditional_value(c::_LockstepConditional, η)
    put!(c.requests, (c.chain, :value, η))
    return take!(c.reply)
end
function _conditional_value_and_gradient(c::_LockstepConditional, η)
    put!(c.requests, (c.chain, :gradient, η))
    return take!(c.reply)
end

# Run `chain(k, target)` for each chain position `k` in `positions`, with `target`
# subject `ids[k]`'s `_SAEMSubjectTarget` at θ and σ, and return the results in the
# order of `positions`, or `nothing` once `stop()` holds before an evaluation.
# `finished()` is called each time a chain returns.
_saem_chains(chain, model, ids, θ, σ, prior, positions;
        stop = () -> false, finished = () -> nothing) =
    _saem_chains(evaluation_style(model), chain, model, ids, θ, σ, prior, positions;
        stop, finished)

# A per-subject model gains nothing from a batch: the chains run one after another,
# each with its own model calls, and `stop` is checked between them.
function _saem_chains(::PerSubjectEvaluation, chain, model, ids, θ, σ, prior, positions;
        stop, finished)
    results = []
    for k in positions
        stop() && return nothing
        push!(results, chain(k, _saem_target(model, ids[k], θ, σ, prior)))
        finished()
    end
    return results
end

# A batched model evaluates the chains in lockstep. Each chain's WarmupHMC call runs
# in its own task on this thread, and its target hands every evaluation to this task
# and waits. Once every unfinished chain waits, one batched call answers all their
# η-gradient requests (and one more any density-only ones), so a leapfrog step costs
# one `conditional_loglikelihoods_and_gradients` call over the subjects whose trees
# are still growing, not one call per subject. Model code runs only in this task,
# and each chain's draws are those its own model calls would give.
function _saem_chains(::BatchedEvaluation, chain, model, ids, θ, σ, prior, positions;
        stop, finished)
    positions = collect(positions)
    θ, σ = Vector{Float64}(θ), Vector{Float64}(σ)
    requests = Channel{Tuple{Int, Symbol, Any}}(Inf)
    replies = [Channel{Any}(1) for _ in positions]
    results = Vector{Any}(undef, length(positions))
    try
        for (j, k) in enumerate(positions)
            target = _SAEMSubjectTarget(_LockstepConditional(j, requests, replies[j]),
                _subject_prior(prior, ids[k]))
            @async try
                put!(requests, (j, :returned, chain(k, target)))
            catch err
                put!(requests, (j, :threw, err))
            end
        end
        unfinished = length(positions)
        waiting = Vector{Any}(nothing, length(positions))
        while unfinished > 0
            nwaiting = 0
            while nwaiting < unfinished
                j, kind, value = take!(requests)
                if kind === :returned
                    results[j] = value
                    unfinished -= 1
                    finished()
                elseif kind === :threw
                    throw(value)
                else
                    waiting[j] === nothing || error("chain $j made a second density ",
                        "request before the first was answered")
                    waiting[j] = (kind, value)
                    nwaiting += 1
                end
            end
            unfinished == 0 && break
            stop() && return nothing
            for kind in (:gradient, :value)
                chains = [j for j in eachindex(waiting)
                    if waiting[j] !== nothing && first(waiting[j]) === kind]
                isempty(chains) && continue
                answers = _saem_batch_conditionals(Val(kind), model,
                    ids[positions[chains]], θ, σ, [last(waiting[j]) for j in chains])
                for (j, answer) in zip(chains, answers)
                    put!(replies[j], answer)
                end
            end
            fill!(waiting, nothing)
        end
        return results
    finally
        foreach(close, replies)
    end
end

# The conditional densities (with their η gradients) of `subjects` at `ηs`, from one
# batched call; a batch that throws `_InfeasibleEvaluation` is evaluated subject by
# subject, so that only the positions outside the domain get -Inf.
function _saem_batch_conditionals(::Val{:gradient}, model, subjects, θ, σ, ηs)
    batch = _feasible(H -> conditional_loglikelihoods_and_gradients(model, subjects,
        θ, σ, H), _eta_batch(nlme_layout(model), ηs))
    batch === nothing && return [_conditional_value_and_gradient(
        _DirectConditional(model, i, θ, σ), η) for (i, η) in zip(subjects, ηs)]
    ℓ, G = batch
    return [(ℓ[k], _eta_column(G, k)) for k in eachindex(subjects)]
end
function _saem_batch_conditionals(::Val{:value}, model, subjects, θ, σ, ηs)
    batch = _feasible(H -> conditional_loglikelihoods(model, subjects, θ, σ, H),
        _eta_batch(nlme_layout(model), ηs))
    batch === nothing && return [_conditional_value(_DirectConditional(model, i, θ, σ), η)
        for (i, η) in zip(subjects, ηs)]
    return batch
end

function _saem_check_nuts(o::WarmupHMCNUTS)
    o.max_tree_depth > 0 || throw(ArgumentError("max_tree_depth must be positive"))
    isfinite(o.target_acceptance_rate) && 0 < o.target_acceptance_rate < 1 ||
        throw(ArgumentError("target_acceptance_rate must be strictly between 0 and 1"))
    return nothing
end

# Each subject's chain starts with a Pathfinder run from its η at the starting
# population parameters (WarmupHMC's `ContinuousNUTSState`), counted on `progress`,
# the start phase, as each subject's start completes. An interrupt request ends the
# start early and returns `nothing`: checked between subjects, or for a batched
# model, whose starts run in lockstep (`_saem_chains`), between batched evaluations.
function _saem_nuts_start!(rng, kernel::WarmupHMCNUTS, model, ids, θ, σ, prior, H,
        progress = nothing)
    start_progress!(progress)
    seeds = [rand(rng, UInt64) for _ in ids]
    started = Ref(0)
    states = _saem_chains(model, ids, θ, σ, prior, eachindex(ids);
            stop = () -> interrupt_requested(progress),
            finished = () -> progress === nothing ||
                update_progress!(progress, started[] += 1)) do k, target
        WarmupHMC.ContinuousNUTSState(Xoshiro(seeds[k]), target; init = copy(_column(H, k)),
            kernel.target_acceptance_rate, kernel.max_tree_depth)
    end
    finalize_progress!(progress)
    return states === nothing ? nothing : identity.(states)
end

# Move each listed subject's chain to its re-centred or replaced η. The kernel is
# unchanged: a translation of the target leaves its metric and step size valid.
function _saem_nuts_reposition!(states, H, positions = eachindex(states))
    for k in positions
        WarmupHMC.reposition!(states[k], _column(H, k))
    end
    return states
end

# One NUTS transition for each subject at positions `active`, adapting the kernel
# from its leapfrog leaves; `adapt = false` takes the transition under the current
# kernel (posterior draws at a fixed point). A batched model takes the transitions in
# lockstep (`_saem_chains`).
function _saem_nuts_round!(states, model, ids, θ, σ, prior, H, active = eachindex(ids);
        adapt::Bool = true)
    steps = _saem_chains(model, ids, θ, σ, prior, active) do k, target
        WarmupHMC.continuous_nuts_step(states[k], target; adapt)
    end
    divergences = 0
    for (k, step) in zip(active, steps)
        _set_column!(H, k, copy(step.state.position))
        divergences += step.divergent
    end
    return divergences
end

# MALA's `:hessian` metric: the inverse local curvature. The Hessian is taken only in
# Initial covariance for a per-subject HMC warmup. The Hessian is taken only in
# η_i; an indefinite local negative Hessian is regularized before inversion.
# Without η Hessians it is the prior covariance of subject i, or its diagonal.
# A model that declares `eta_hessian_sparsity` gives a sparse Hessian, so the
# diagonal metric of a subject with thousands of nested effects never forms a
# dense matrix; the dense metric still does.
function _saem_hessian_metric(model, i, θ, σ, prior, η;
        curvature_floor::Real = 1e-4, diagonal::Bool = false)
    isfinite(curvature_floor) && curvature_floor > 0 || throw(ArgumentError(
        "curvature_floor must be finite and positive"))
    if !_reports_eta_hessians(model)
        diagonal && return _prior_variances(prior, i)
        return _prior_covariance(prior, i)
    end
    H = if _declares_eta_hessian_sparsity(model)
        only(_complete_hessians(model, [i], θ, σ, prior, _single_eta_batch(nlme_layout(model), η)))
    else
        last(complete_data_loglikelihood_gradient_and_hessian(model, i, θ, σ, prior, η))
    end
    diagonal && return inv.(max.(-diag(H), curvature_floor))
    eig = eigen(Symmetric(-Matrix(H)))
    scale = eig.vectors * Diagonal(inv.(max.(eig.values, curvature_floor))) * eig.vectors'
    return Matrix(Symmetric(scale))
end

function _saem_check_options(o::SAEMOptions)
    o.iterations > 0 || throw(ArgumentError("iterations must be positive"))
    0 <= o.burnin < o.iterations || throw(ArgumentError("require 0 ≤ burnin < iterations"))
    o.mcmc_burnin >= 0 || throw(ArgumentError("mcmc_burnin must be nonnegative"))
    o.mh_rounds > 0 || throw(ArgumentError("mh_rounds must be positive"))
    isfinite(o.rw_scale) && o.rw_scale > 0 || throw(ArgumentError("rw_scale must be finite and positive"))
    o.adapt_iterations >= 0 || throw(ArgumentError("adapt_iterations must be nonnegative"))
    0.5 < o.step_exponent <= 1 || throw(ArgumentError("require 0.5 < step_exponent ≤ 1"))
    o.numeric_maxiter > 0 || throw(ArgumentError("numeric_maxiter must be positive"))
    o.mstep_interval > 0 || throw(ArgumentError("mstep_interval must be positive"))
    o.numeric_mstep in (:auto, :online, :mixture) ||
        throw(ArgumentError("numeric_mstep must be :auto, :online, or :mixture"))
    isfinite(o.omega_floor) && o.omega_floor > 0 || throw(ArgumentError("omega_floor must be finite and positive"))
    0 <= o.omega_annealing < 1 || throw(ArgumentError("omega_annealing must be in [0, 1)"))
    o.minibatch_size === nothing || o.minibatch_size > 0 ||
        throw(ArgumentError("minibatch_size must be positive or nothing"))
    return nothing
end

# The number of subjects whose chains move in one iteration, or `nothing` when
# that is every fitted subject (the full-batch algorithm).
_saem_minibatch(::Nothing, n) = nothing
function _saem_minibatch(size::Int, n)
    size <= n || throw(ArgumentError(
        "minibatch_size $size exceeds the $n fitted subjects"))
    return size == n ? nothing : size
end

# Positions among the fitted subjects whose chains move in one iteration: all of
# them, or `batch` distinct positions drawn uniformly at random, in ascending
# order.
_saem_active(rng, n, ::Nothing) = Base.OneTo(n)
_saem_active(rng, n, batch::Int) = sort!(randperm(rng, n)[1:batch])

function _saem_mh_round!(rng, model, subjects, θ, σ, prior, H, ℓ, scale,
        component_scales)
    m, n = size(H)
    proposal = Matrix(prior.cholesky_factor * randn(rng, m, n))
    proposed_ℓ = _proposal_loglikelihoods(model, subjects, θ, σ, proposal)
    length(proposed_ℓ) == n || throw(DimensionMismatch("prior MH proposal returned $(length(proposed_ℓ)) log densities for $n subjects"))
    all(x -> !isnan(x) && x != Inf, proposed_ℓ) || throw(DomainError(proposed_ℓ,
        "prior MH proposal returned NaN or +Inf log density"))
    prior_accepted = 0
    for k in 1:n
        # The prior is the independence proposal, so its density cancels.
        if log(rand(rng)) < proposed_ℓ[k] - ℓ[k]
            H[:, k] = proposal[:, k]
            ℓ[k] = proposed_ℓ[k]
            prior_accepted += 1
        end
    end
    proposal = H + scale * (prior.cholesky_factor * randn(rng, m, n))
    proposed_ℓ = _proposal_loglikelihoods(model, subjects, θ, σ, proposal)
    length(proposed_ℓ) == n || throw(DimensionMismatch("random-walk MH proposal returned $(length(proposed_ℓ)) log densities for $n subjects"))
    all(x -> !isnan(x) && x != Inf, proposed_ℓ) || throw(DomainError(proposed_ℓ,
        "random-walk MH proposal returned NaN or +Inf log density"))
    rw_accepted = 0
    for k in 1:n
        logratio = proposed_ℓ[k] - ℓ[k] +
            re_logprior(prior, view(proposal, :, k)) - re_logprior(prior, view(H, :, k))
        if log(rand(rng)) < logratio
            H[:, k] = proposal[:, k]
            ℓ[k] = proposed_ℓ[k]
            rw_accepted += 1
        end
    end
    component_accepted, component_proposed = _saem_mh_component_sweep!(rng, model,
        subjects, θ, σ, prior, H, ℓ, component_scales)
    return prior_accepted, rw_accepted, component_accepted, component_proposed
end

# One random-walk proposal per η coordinate for every subject, each scaled by
# its own adapted factor times the coordinate's prior standard deviation.
# Returns the accepted and proposed counts per coordinate.
function _saem_mh_component_sweep!(rng, model, subjects, θ, σ, prior, H, ℓ, scales)
    m, n = size(H)
    sd = sqrt.(vec(sum(abs2, Matrix(prior.cholesky_factor); dims = 2)))
    accepted = zeros(Int, m)
    for c in 1:m
        proposal = copy(H)
        proposal[c, :] .+= (scales[c] * sd[c]) .* randn(rng, n)
        proposed_ℓ = _proposal_loglikelihoods(model, subjects, θ, σ, proposal)
        length(proposed_ℓ) == n && all(x -> !isnan(x) && x != Inf, proposed_ℓ) ||
            throw(DomainError(proposed_ℓ,
                "coordinate random-walk MH proposal returned invalid densities"))
        for k in 1:n
            logratio = proposed_ℓ[k] - ℓ[k] +
                re_logprior(prior, view(proposal, :, k)) - re_logprior(prior, view(H, :, k))
            if log(rand(rng)) < logratio
                H[:, k] = proposal[:, k]
                ℓ[k] = proposed_ℓ[k]
                accepted[c] += 1
            end
        end
    end
    return accepted, fill(n, m)
end

# The MH random-walk scales adapt during the first `adapt_iterations` sweeps:
# the joint proposal's toward 0.3 acceptance, each coordinate's toward 0.4.
# Returns the joint log scale; the coordinate log scales update in place.
function _saem_adapt_mh_scales!(logscale, component_logscales, rw_accepted, n,
        component_accepted, component_proposed, sweeps)
    logscale = clamp(logscale + (rw_accepted / n - 0.3) / sqrt(sweeps),
        log(1.0e-3), log(1.0e2))
    for c in eachindex(component_logscales)
        component_proposed[c] > 0 || continue
        component_logscales[c] = clamp(component_logscales[c] +
            (component_accepted[c] / component_proposed[c] - 0.4) / sqrt(sweeps),
            log(1.0e-3), log(1.0e2))
    end
    return logscale
end

function _saem_mu_trait(model, subjects, layout)
    mr = mu_referencing(model)
    if mr === nothing
        return Int[], [zeros(neta(layout), 0) for _ in subjects]
    end
    length(mr.design) == nsubjects(model) || throw(DimensionMismatch(
        "mu_referencing has $(length(mr.design)) designs for $(nsubjects(model)) subjects"))
    allunique(mr.theta_indices) && all(i -> 1 <= i <= layout.ntheta, mr.theta_indices) ||
        throw(ArgumentError("mu_referencing has invalid theta_indices"))
    for i in subjects
        size(mr.design[i]) == (neta(layout), length(mr.theta_indices)) || throw(DimensionMismatch(
            "mu_referencing design[$i] has size $(size(mr.design[i]))"))
    end
    return mr.theta_indices, [mr.design[i] for i in subjects]
end

function _saem_phi(H, θ, indices, design)
    μ = view(θ, indices)
    Φ = copy(H)
    for k in axes(Φ, 2)
        Φ[:, k] += design[k] * μ
    end
    return Φ
end

function _saem_eta(Φ, θ, indices, design)
    μ = view(θ, indices)
    H = copy(Φ)
    for k in axes(H, 2)
        H[:, k] -= design[k] * μ
    end
    return H
end

# Conditional log densities of the chains' current latent effects at population
# parameters an M-step just set. The update keeps every chain's φ (re-centring)
# and its numerical part rejects trial points where a current chain leaves the
# model's domain, so this is the safety net for a chain that still ends up there
# (a non-finite density, or an `_InfeasibleEvaluation` throw). The MH and NUTS
# kernels then stop here with one error naming those subjects; MALA checks its
# own values each round.
function _saem_current_loglikelihoods(model, ids, θ, σ, H)
    ℓ = _proposal_loglikelihoods(model, ids, θ, σ, H)
    length(ℓ) == length(ids) || throw(DimensionMismatch(
        "returned $(length(ℓ)) conditional log densities for $(length(ids)) subjects"))
    outside = [i for (i, l) in zip(ids, ℓ) if !isfinite(l)]
    isempty(outside) || throw(DomainError(ℓ, "SAEM conditional log densities became " *
        "non-finite after population update (subjects $(join(outside, ", ")))"))
    return ℓ
end

_saem_prior_draw(rng, prior::NormalREPrior, i) =
    prior.cholesky_factor * randn(rng, neta(prior))

# The minibatch form of `_saem_current_loglikelihoods`, for the chains at
# positions `active`, written into `ℓ[active]`. A minibatch population update
# screens only its own batch's chains, so a chain drawn now can lie outside the
# model's domain at the current population parameters. Its state then has zero
# density, and it is replaced by a draw from the random-effect prior: an
# independence Metropolis–Hastings move, which a zero-density state accepts
# whenever the draw is feasible. Returns the positions whose chains were
# replaced. A chain still outside after ten draws raises the `DomainError`.
function _saem_batch_loglikelihoods!(rng, model, ids, θ, σ, prior, H, ℓ, active)
    values = _proposal_loglikelihoods(model, ids[active], θ, σ, _columns(H, active))
    length(values) == length(active) || throw(DimensionMismatch(
        "returned $(length(values)) conditional log densities for $(length(active)) subjects"))
    ℓ[active] = values
    replaced = Int[]
    for _ in 1:10
        outside = [k for k in active if ℓ[k] == -Inf]
        isempty(outside) && break
        proposal = _from_columns(H, [_saem_prior_draw(rng, prior, ids[k]) for k in outside])
        proposed = _proposal_loglikelihoods(model, ids[outside], θ, σ, proposal)
        for (j, k) in enumerate(outside)
            isfinite(proposed[j]) || continue
            _set_column!(H, k, copy(_column(proposal, j)))
            ℓ[k] = proposed[j]
            push!(replaced, k)
        end
    end
    bad = [ids[k] for k in active if !isfinite(ℓ[k])]
    isempty(bad) || throw(DomainError(ℓ[active], "SAEM conditional log densities " *
        "are non-finite at the current population parameters (subjects $(join(bad, ", ")))"))
    return unique!(replaced)
end

# Keep each chain's φ = Dθ + η fixed across a population update, as saemix does:
# shift η by −D(θ_new − θ_old) on the mu-referenced coordinates. Each subject's
# likelihood is then unchanged and its kernel continues where the chain had
# settled. Holding η fixed instead moved every φ by DΔθ, and at gain 1 a slowly
# mixing kernel then drifted θ by about mean(η) at every M-step (brief
# `dqtdys`). Returns whether any chain moved.
function _saem_recenter!(H::AbstractMatrix, θ_old, θ_new, indices, design)
    Δ = θ_new[indices] - θ_old[indices]
    all(iszero, Δ) && return false
    for k in axes(H, 2)
        H[:, k] .-= design[k] * Δ
    end
    return true
end

# The mu-referenced coefficient directions that the selected subjects' designs
# identify: an orthonormal basis of the stacked designs' row space, with the rank
# counted as `LinearAlgebra.rank` counts it, or `nothing` when they identify every
# coefficient. A combination δ with X_k δ = 0 for every subject moves no φ, so
# the likelihood does not depend on it: more coefficients on a random effect
# than subjects that carry it (three subjects with an intercept and three
# covariates), or collinear covariates. The closed-form step's matrix
# Σ_k X_kᵀ P X_k has the stacked designs' null space for every positive-definite
# precision P, so one basis serves the whole fit.
function _saem_identified(design, indices)
    isempty(indices) && return nothing
    D = reduce(vcat, design)
    F = svd(D)
    tolerance = minimum(size(D)) * eps() * maximum(F.S; init = 0.0)
    rank = count(>(tolerance), F.S)
    return rank == length(indices) ? nothing : F.V[:, 1:rank]
end

# The generalized-least-squares update of the mu-referenced θ from μ. With
# identified directions Q it solves A Δ = b - A μ for Δ in their span, which
# maximizes the expected Gaussian density (b and A μ lie in that span), and keeps
# μ along the other directions, which the likelihood does not inform; a MAP
# M-step's prior then moves them (`_saem_gaussian_map_block`). A Cholesky solve
# of the whole system failed there, or returned arbitrary values.
_saem_gls(A, b, μ, ::Nothing) = cholesky(Symmetric(A)) \ b
_saem_gls(A, b, μ, Q) = μ + Q * (cholesky(Symmetric(Q' * A * Q)) \ (Q' * (b - A * μ)))

# Without a population prior, a combination of mu-referenced θ that the designs
# do not identify leaves the likelihood flat, so the maximum-likelihood estimate
# is not unique.
function _saem_require_identified(model, indices, identified)
    identified === nothing && return nothing
    outside = 1 .- vec(sum(abs2, identified; dims = 2))
    names = parameter_names(model).theta[indices[outside .> sqrt(eps())]]
    combinations = length(indices) - size(identified, 2)
    throw(ArgumentError("the mu-referencing designs of the selected subjects do not " *
        "identify every mu-referenced θ: $combinations combination(s) of " *
        "$(join(names, ", ")) move no subject's individual parameters, so the " *
        "likelihood does not depend on them and the maximum-likelihood estimate is " *
        "not unique. Give the model a population_logprior (a MAP fit estimates " *
        "them from the prior), remove or recode covariates, or fit more subjects"))
end

function _saem_moments!(mean_phi, second_phi, Φ, γ)
    mean_phi .*= 1 - γ
    second_phi .*= 1 - γ
    for k in axes(Φ, 2)
        φ = view(Φ, :, k)
        mean_phi[:, k] .+= γ .* φ
        second_phi[:, :, k] .+= γ .* (φ * φ')
    end
    return nothing
end

# The closed-form Gaussian block alternates the generalized-least-squares
# mu-referenced θ at fixed Ω with the floored covariance at fixed θ. Each update
# maximizes its block, so the expected Gaussian density never decreases. Where
# Ω's maximizer is singular, as when few subjects carry the mu-referenced
# coefficients and the γ = 1 statistics come from one draw, the updates approach
# `floor` ever more slowly (7659 updates to settle at 1e-8 on three PK subjects
# with a weight slope). After 100 updates the step returns its iterate, as the budgeted
# searches of the other blocks do, and the next M-step continues from it; a MAP
# M-step only starts its Gaussian block there. Where the designs do not identify
# every mu-referenced coefficient (`identified`), the update keeps θ along the
# combinations they leave out (`_saem_gls`).
function _saem_gaussian_mstep(layout, θ, Ω, indices, design, mean_phi, second_phi, floor,
        constraints = nothing; identified = _saem_identified(design, indices))
    m, n = size(mean_phi)
    μ = copy(θ[indices])
    Ωnew = copy(Ω)
    ranges, _ = _block_ranges(layout)
    for _ in 1:100
        if !isempty(indices)
            P = inv(Symmetric(Ωnew))
            A = zeros(Float64, length(indices), length(indices))
            b = zeros(Float64, length(indices))
            for k in 1:n
                X = design[k]
                A += X' * P * X
                b += X' * P * view(mean_phi, :, k)
            end
            μ = _saem_gls(A, b, μ, identified)
        end
        second_residual = zeros(Float64, m, m)
        for k in 1:n
            xμ = design[k] * μ
            φ = view(mean_phi, :, k)
            second_residual += view(second_phi, :, :, k) - φ * xμ' - xμ * φ' + xμ * xμ'
        end
        second_residual ./= n
        nextΩ = zeros(Float64, m, m)
        for (b, r) in enumerate(ranges)
            fam = _omega_family(constraints, b)
            if fam === nothing
                E = eigen(Symmetric(second_residual[r, r]))
                nextΩ[r, r] = E.vectors * Diagonal(max.(E.values, floor)) * E.vectors'
            else
                nextΩ[r, r] = _fixed_family_update(fam, second_residual[r, r], Ωnew[r, r], floor)
            end
        end
        converged = maximum(abs, nextΩ - Ωnew) <= 1.0e-8 * max(1.0, maximum(abs, Ωnew))
        Ωnew = nextΩ
        converged && break
    end
    θnew = copy(θ)
    θnew[indices] = μ
    return θnew, Ωnew
end

# Expected Gaussian random-effect log density under the running sufficient
# statistics, with derivatives in mu-referenced θ and log-Cholesky ω.
function _saem_gaussian_map_value_gradient(layout, θ, ω, indices, design,
        mean_phi, second_phi)
    m, n = size(mean_phi)
    L = Matrix(omega_cholesky(layout, ω))
    Linv = inv(LowerTriangular(L))
    P = Matrix(Linv' * Linv)
    μ = view(θ, indices)
    S = zeros(Float64, m, m)
    gμ = zeros(Float64, length(indices))
    for k in 1:n
        X = design[k]
        xμ = X * μ
        φ = view(mean_phi, :, k)
        S += view(second_phi, :, :, k) - φ * xμ' - xμ * φ' + xμ * xμ'
        gμ += X' * P * (φ - xμ)
    end
    value = -(n * (m * log(2π) + 2sum(log, diag(L))) + sum(P .* S)) / 2
    GL = (P * S * P - n * P) * L
    gω = zeros(Float64, length(ω))
    ranges, omega_ranges = _block_ranges(layout)
    for (er, or) in zip(ranges, omega_ranges)
        c = first(or)
        for j in 1:length(er), l in 1:j
            row, col = er[j], er[l]
            gω[c] = j == l ? GL[row, col] * L[row, col] : GL[row, col]
            c += 1
        end
    end
    return value, gμ, gω
end

_saem_gaussian_map_value_gradient(layout::NLMELayout, θ, ω, indices, design,
    mean_phi, second_phi, subjects) = _saem_gaussian_map_value_gradient(
        layout, θ, ω, indices, design, mean_phi, second_phi)

# The historical mixture optimizes the running conditional-density objective.
function _saem_record!(::Val{:mixture}, bank, weights, Φ, γ)
    if γ == 1
        empty!(bank)
        empty!(weights)
    else
        weights .*= 1 - γ
    end
    push!(bank, copy(Φ))
    push!(weights, γ)
    return nothing
end

# Practical SAEM stores only the current latent draw. The optimizer itself is
# updated stochastically when the M-step is due.
function _saem_record!(::Union{Val{:online}, Val{:minibatch}}, bank, weights, Φ, γ)
    empty!(bank)
    empty!(weights)
    push!(bank, copy(Φ))
    push!(weights, 1.0)
    return nothing
end

_saem_numeric_mode(mode::Symbol, free, ntheta) =
    Val(mode === :auto ? (any(i -> i <= ntheta, free) ? :mixture : :online) : mode)
_saem_mode_name(::Val{M}) where {M} = M

# Minibatch SAEM has its own numerical update (`_SAEMScoreStep`), which needs no
# sufficient statistic for any coordinate. The explicit full-batch modes are
# refused: `:mixture` re-evaluates every subject's retained draws at every
# M-step, and `:online` searches each draw's objective to its optimum, which on
# a batch jumps by the batch's whole sampling error (a weakly identified σ
# landed where its score vanishes).
_saem_numeric_mode(mode::Symbol, free, ntheta, ::Nothing) =
    _saem_numeric_mode(mode, free, ntheta)
function _saem_numeric_mode(mode::Symbol, free, ntheta, batch::Int)
    mode === :auto || throw(ArgumentError("minibatch SAEM has its own numerical " *
        "update and needs numeric_mstep = :auto; :$mode is a full-batch update"))
    return Val(:minibatch)
end

# The numerical M-step's objective adds a population prior at the M-step's ω;
# `ω = nothing` is maximum likelihood.
function _saem_numeric_objective(z, model, subjects, θ, σ, free, indices, design, bank,
        weights, ω = nothing)
    x = vcat(θ, σ)
    x[free] = z
    θz = x[1:length(θ)]
    σz = x[length(θ)+1:end]
    value = zero(eltype(x))
    for (Φ, w) in zip(bank, weights)
        H = _saem_eta(Φ, θz, indices, design)
        value += w * sum(conditional_loglikelihoods(model, subjects, θz, σz, H))
    end
    return value + _saem_prior_value(model, θz, σz, ω)
end

_saem_prior_value(model, θ, σ, ::Nothing) = 0.0
function _saem_prior_value(model, θ, σ, ω)
    lp = population_logprior(model, θ, σ, ω)
    lp isa Real || throw(ArgumentError("population_logprior must return a real log density"))
    return lp
end

_saem_prior_value_gradient(model, θ, σ, ::Nothing) = (0.0, zeros(length(θ) + length(σ)))
function _saem_prior_value_gradient(model, θ, σ, ω)
    lp, gθ, gσ, _ = _saem_prior_terms(model, θ, σ, ω)
    return lp, vcat(gθ, gσ)
end

function _saem_prior_terms(model, θ, σ, ω)
    lp, gθ, gσ, gω = population_logprior_and_gradient(model, θ, σ, ω)
    lp isa Real || throw(ArgumentError("population_logprior_and_gradient must return a real value"))
    _check_length("population prior θ gradient", gθ, length(θ))
    _check_length("population prior σ gradient", gσ, length(σ))
    _check_length("population prior ω gradient", gω, length(ω))
    return lp, gθ, gσ, gω
end

function _saem_population_value_gradient(::PerSubjectEvaluation, model, subjects, θ, σ, H)
    value = 0.0
    gθ = zeros(Float64, length(θ))
    gσ = zeros(Float64, length(σ))
    for (k, i) in enumerate(subjects)
        ℓ, a, b = conditional_loglikelihood_and_population_gradient(model, i, θ, σ, _eta_column(H, k))
        value += ℓ
        gθ += a
        gσ += b
    end
    return value, vcat(gθ, gσ)
end

_saem_batch_population_objective(x, model, subjects, H, ntheta) =
    sum(conditional_loglikelihoods(model, subjects, view(x, 1:ntheta),
        view(x, ntheta+1:length(x)), H))

function _saem_population_value_gradient(::BatchedEvaluation, model, subjects, θ, σ, H)
    backend = _checked_backend(model, ad_backend(model), :saem)
    return DI.value_and_gradient(_saem_batch_population_objective, backend, vcat(θ, σ),
        DI.Constant(model), DI.Constant(subjects), DI.Constant(H), DI.Constant(length(θ)))
end

function _saem_numeric_value_gradient(z, model, subjects, θ, σ, free, indices, design,
        bank, weights, mode::Val, ω = nothing)
    x = vcat(θ, σ)
    x[free] = z
    θz = x[1:length(θ)]
    σz = x[length(θ)+1:end]
    value = 0.0
    gradient = zeros(Float64, length(x))
    style = saem_population_gradient_style(model, mode)
    for (Φ, w) in zip(bank, weights)
        H = _saem_eta(Φ, θz, indices, design)
        v, g = _saem_population_value_gradient(style, model, subjects, θz, σz, H)
        value += w * v
        gradient += w * g
    end
    lp, glp = _saem_prior_value_gradient(model, θz, σz, ω)
    return value + lp, (gradient + glp)[free]
end

# `metric` carries the search's inverse-Hessian approximation from one M-step
# to the next; the first search starts from a difference estimate of the
# curvature, because the coordinates' scales can differ by many orders of
# magnitude (an intercept pinned by thousands of observations next to a
# rate constant).
function _saem_numeric_mstep(model, subjects, θ, σ, free, indices, design, bank, weights,
        maxiter, mode::Val, ω, metric)
    isempty(free) && return θ, σ
    objective(z) = _saem_numeric_value_gradient(z, model, subjects, θ, σ,
        free, indices, design, bank, weights, mode, ω)
    screen(z) = _saem_numeric_objective(z, model, subjects, θ, σ,
        free, indices, design, bank, weights, ω)
    z0 = vcat(θ, σ)[free]
    metric[] === nothing && (metric[] = _saem_search_metric(objective, z0))
    z, metric[] = _bfgs_ascent_state(objective, z0, maxiter, _ -> true;
        screen, inverse_hessian = metric[], what = "SAEM numerical M-step")
    x = vcat(θ, σ)
    x[free] = z
    return x[1:length(θ)], x[length(θ)+1:end]
end

# Whether `Ω = L Lᵀ` is finite with smallest eigenvalue at least `floor`, read
# from the Cholesky factor `L`: the eigenvalues of the formed product are only
# resolved to about eps·‖Ω‖, which at a large scale exceeds the floor and passes
# an exactly singular Ω (a diagonal of L underflowed to zero), whose density then
# throws. σ_min(L)² is resolved to about (eps·‖L‖)². A diagonal entry of L is one
# of its eigenvalues and so bounds σ_min(L) exactly, at any scale.
function _factor_above_floor(L, floor)
    all(isfinite, L) && all(isfinite, L * L') || return false
    minimum(abs, diag(L))^2 >= floor || return false
    return minimum(svdvals(Matrix(L)))^2 >= floor
end

function _saem_omega_above_floor(layout, ω, floor)
    L = omega_cholesky(layout, ω)
    ranges, _ = _block_ranges(layout)
    return all(r -> _factor_above_floor(L[r, r], floor), ranges)
end

# The Gaussian block of a MAP M-step: the mu-referenced θ and the free Ω
# coordinates (`_free_omega`) maximize the expected Gaussian random-effect
# density plus the population prior at the current remaining θ and σ. Neither
# term evaluates the model. The search starts where the Gaussian term's
# closed-form updates (`_saem_gaussian_mstep`) stopped, with that term's inverse
# curvature as its quasi-Newton matrix, so a prior that is weak against the data
# moves it in a few steps. Along combinations that the designs do not identify
# (`_saem_identified`) the Gaussian term has no curvature, and only the prior
# gives the search a scale there, so it starts with the whole objective's.
function _saem_gaussian_map_block(model, layout, θ, σ, ω, indices, design,
        mean_phi, second_phi, subjects, floor, constraints;
        identified = _saem_identified(design, indices))
    nμ = length(indices)
    function point(z)
        θz = copy(θ)
        θz[indices] = view(z, 1:nμ)
        return θz, Vector{Float64}(_constrained_omega(constraints, view(z, nμ+1:length(z))))
    end
    function gaussian(z)
        θz, ωz = point(z)
        value, gμ, gω = _saem_gaussian_map_value_gradient(layout, θz, ωz, indices,
            design, mean_phi, second_phi, subjects)
        return value, vcat(gμ, _free_omega_gradient(constraints, ωz, gω))
    end
    function objective(z)
        θz, ωz = point(z)
        value, gradient = gaussian(z)
        lp, gθ, _, gω = _saem_prior_terms(model, θz, σ, ωz)
        return value + lp, gradient + vcat(gθ[indices],
            _free_omega_gradient(constraints, ωz, gω))
    end
    function screen(z)
        θz, ωz = point(z)
        return first(gaussian(z)) + _saem_prior_value(model, θz, σ, ωz)
    end
    admissible(z) = _omega_admissible(layout, constraints, last(point(z)), floor)
    z0 = vcat(θ[indices], _free_omega(constraints, ω))
    curvature = identified === nothing ? gaussian : objective
    z = _bfgs_ascent(objective, z0, 100, admissible; screen,
        inverse_hessian = _saem_inverse_curvature(curvature, z0))
    return point(z)
end

# Minus the Hessian of a value-and-gradient function at z, symmetrized, from
# central differences of its gradient; `nothing` where a difference point is
# outside the model's domain or not finite.
function _saem_difference_curvature(f, z; step = 1.0e-5)
    d = length(z)
    H = zeros(d, d)
    for j in 1:d
        e = zeros(d)
        e[j] = step
        plus, minus = _feasible(f, z + e), _feasible(f, z - e)
        (plus === nothing || minus === nothing) && return nothing
        isfinite(first(plus)) && isfinite(first(minus)) || return nothing
        H[:, j] = (last(plus) - last(minus)) / 2step
    end
    C = Symmetric(-(H + H') / 2)
    return all(isfinite, C) ? C : nothing
end

# The inverse of that curvature; `nothing` (the identity) where it is not
# positive definite or not available.
function _saem_inverse_curvature(f, z; step = 1.0e-5)
    isempty(z) && return nothing
    C = _saem_difference_curvature(f, z; step)
    C === nothing && return nothing
    F = cholesky(C; check = false)
    return issuccess(F) ? inv(F) : nothing
end

# A positive-definite quasi-Newton metric from that curvature: the inverse of
# its absolute eigenvalues, floored relative to the largest, so that a start
# where the objective is not concave still gets each direction's scale.
# `nothing` (the identity) where the curvature is not available.
function _saem_search_metric(f, z; step = 1.0e-5)
    isempty(z) && return nothing
    C = _saem_difference_curvature(f, z; step)
    C === nothing && return nothing
    E = eigen(C)
    λ = abs.(E.values)
    λ = max.(λ, eps()^(2 / 3) * maximum(λ))
    all(>(0), λ) || return nothing
    return Matrix(E.vectors * Diagonal(inv.(λ)) * E.vectors')
end

_saem_numeric_update(::Val{:mixture}, model, subjects, θ, σ, free, indices,
    design, bank, weights, maxiter, γ, ω, metric) =
    _saem_numeric_mstep(model, subjects, θ, σ, free, indices, design,
        bank, weights, maxiter, Val(:mixture), ω, metric)

function _saem_numeric_update(::Val{:online}, model, subjects, θ, σ, free,
        indices, design, bank, weights, maxiter, γ, ω, metric)
    isempty(free) && return θ, σ
    θstar, σstar = _saem_numeric_mstep(model, subjects, θ, σ, free, indices,
        design, bank, weights, maxiter, Val(:online), ω, metric)
    x = vcat(θ, σ)
    xstar = vcat(θstar, σstar)
    x[free] .+= γ .* (xstar[free] .- x[free])
    return x[1:length(θ)], x[length(θ)+1:end]
end

# Resolve the selected M-step once at runtime. Keeping its callable abstract in
# saem prevents Julia from compiling the unused mode's model-specific gradient
# while compiling the outer SAEM loop.
# It also holds the numerical search's metric across the fit's M-steps.
struct _SAEMNumericUpdate{M<:Val} <: Function
    mode::M
    metric::Base.RefValue{Union{Nothing, Matrix{Float64}}}
end
_SAEMNumericUpdate(mode::Val) =
    _SAEMNumericUpdate(mode, Ref{Union{Nothing, Matrix{Float64}}}(nothing))

function (update::_SAEMNumericUpdate)(model, subjects, θ, σ, free, indices,
        design, bank, weights, maxiter, γ, ω = nothing, active = nothing)
    return _saem_numeric_update(update.mode, model, subjects, θ, σ, free,
        indices, design, bank, weights, maxiter, γ, ω, update.metric)
end

# The numerical update of minibatch SAEM, built like `_SAEMNumericUpdate` once
# per fit. `share` is the batch's fraction of the fitted subjects; `metric`
# holds the quasi-Newton metric across M-steps and `steps` counts them
# (`_saem_score_step`).
struct _SAEMScoreStep <: Function
    share::Float64
    metric::Base.RefValue{Union{Nothing, Matrix{Float64}}}
    steps::Base.RefValue{Int}
end
_SAEMScoreStep(share::Real) =
    _SAEMScoreStep(share, Ref{Union{Nothing, Matrix{Float64}}}(nothing), Ref(0))

function (step::_SAEMScoreStep)(model, subjects, θ, σ, free, indices, design,
        bank, weights, maxiter, γ, ω = nothing, active = eachindex(subjects))
    return _saem_score_step(model, subjects, θ, σ, free, indices, design, bank,
        weights, γ, ω, active, step)
end

# One Robbins–Monro step z ← z + t g M ĝ along ĝ, the score of the batch at
# positions `active` with its conditional log density scaled to all `subjects`.
# The gain g = min(γ, 4b/n) caps the step at four times the batch's share of a
# Newton step. One pass over the subjects then shrinks the distance to the
# optimum by about e⁻⁴, as a full-batch search nearly solves each M-step, and
# the steps' noise keeps the iterate within about 1.4 standard errors of it,
# like the single draws of a full-batch iteration. An uncapped step at γ = 1, or
# a search to the batch's optimum, moves by the batch's whole sampling error: on
# 10 of 400 PK subjects, a weakly identified additive error SD jumped to where
# its score vanishes and never came back. A cap of b/n left e⁻¹ of the distance
# per pass, and the lagging σ let Ω drift onto a singular, absorbing
# correlation in 1 of 8 NUTS fits of a 40-subject growth model. The gains still
# sum to infinity with finite squares, and the step's mean is g M times the full
# score, so the iterates converge where that score vanishes; a search to each
# batch's optimum would average optimizers whose mean is off by order 1/b. M is the inverse absolute curvature of the full objective, from
# every subject's chain, estimated at M-steps 1, 2, 4, 8, …: the curvature
# changes most while the parameters travel, and a metric from a far start
# (where σ's curvature is inflated by residuals that still hold the random
# effects) made σ lag the chains. The doubling schedule costs a logarithmic
# number of passes over every subject. Between estimates M is held, independent of the batches it
# multiplies. t is halved from 1 until the batch objective passes the Armijo
# test (with the searches' rounding allowance), which so small a step near a
# smooth optimum passes at once.
function _saem_score_step(model, subjects, θ, σ, free, indices, design, bank,
        weights, γ, ω, active, step::_SAEMScoreStep)
    isempty(free) && return θ, σ
    z0 = vcat(θ, σ)[free]
    step.steps[] += 1
    if ispow2(step.steps[])
        full(z) = _saem_numeric_value_gradient(z, model, subjects, θ, σ, free,
            indices, design, bank, weights, Val(:online), ω)
        step.metric[] = _saem_search_metric(full, z0)
    end
    batch_subjects, batch_design = subjects[active], design[active]
    batch = [_columns(H, active) for H in bank]
    scaled = (length(subjects) / length(active)) .* weights
    objective(z) = _saem_numeric_value_gradient(z, model, batch_subjects, θ, σ, free,
        indices, batch_design, batch, scaled, Val(:online), ω)
    screen(z) = _saem_numeric_objective(z, model, batch_subjects, θ, σ, free, indices,
        batch_design, batch, scaled, ω)
    value, gradient = objective(z0)
    isfinite(value) && all(isfinite, gradient) || throw(DomainError(value,
        "SAEM numerical M-step has non-finite objective or gradient"))
    metric = step.metric[]
    direction = min(γ, 4step.share) .* (metric === nothing ? gradient : metric * gradient)
    z = z0
    t = 1.0
    while t >= 1.0e-8
        candidate = z0 + t * direction
        predicted = t * dot(gradient, direction)
        threshold = value + 1.0e-4 * predicted
        if t == 1 && predicted <= _resolution(value)
            threshold = min(threshold, value - _resolution(value))
        end
        if _armijo(_feasible(screen, candidate), threshold)
            z = candidate
            break
        end
        t /= 2
    end
    x = vcat(θ, σ)
    x[free] = z
    return x[1:length(θ)], x[length(θ)+1:end]
end

# Simulated annealing of Ω during SAEM's exploration phase (as in Monolix and
# saemix): no covariance family shrinks by more than the factor τ per M-step in
# any direction, Ω_new ⪰ τ Ω_old in the matrix order, and fixed parts are then
# re-applied. While γ = 1 each M-step sets Ω from one posterior draw per subject,
# so Ω wanders; where it wanders towards a singular correlation, kernels whose
# proposals follow Ω cannot leave it. On the `fixed Ω` test fixture, 3 of 20
# fits ended there, up to 3.5 log-likelihood units below the maximum, and on
# the growth model prior-metric MALA collapsed in 3 of 20 full-batch and 4 of
# 20 minibatch fits; with τ = 0.95 none did.
function _saem_anneal(layout::NLMELayout, ω_previous, ω, τ, constraints)
    Ω_previous, Ω = omega_matrix(layout, ω_previous), omega_matrix(layout, ω)
    ranges, _ = _block_ranges(layout)
    for r in ranges
        Ω[r, r] = _saem_anneal_floor(Ω[r, r], Ω_previous[r, r], τ)
    end
    return _apply_omega_constraints(constraints, omega_parameters(layout, Ω))
end

# The covariance closest to Ω (in the eigenbasis of Ω - τ Ω_previous) that is at
# least τ Ω_previous: Ω itself when it already is, and never below Ω.
function _saem_anneal_floor(Ω, Ω_previous, τ)
    A = τ * Ω_previous
    E = eigen(Symmetric(Ω - A))
    return Matrix(Symmetric(A + E.vectors * Diagonal(max.(E.values, 0)) * E.vectors'))
end

_saem_closed_form_mstep(layout::NLMELayout, θ, Ω, indices, design, mean_phi,
    second_phi, floor, subjects, constraints; require_instances = true,
    identified = _saem_identified(design, indices)) =
    _saem_gaussian_mstep(layout, θ, Ω, indices, design, mean_phi, second_phi,
        floor, constraints; identified)

# One population update, for either layout. At fixed Φ the conditional density
# does not depend on the mu-referenced θ, and Ω enters only the expected Gaussian
# random-effect density, so the update starts with a Gaussian block
# (mu-referenced θ and Ω). Without a population prior it takes closed-form
# coordinate updates (`_saem_gaussian_mstep`); a prior moves it on to its
# maximizer with the prior, without evaluating the model. The numerical block
# then holds every draw's η fixed instead of its φ, so it moves the
# mu-referenced θ jointly with the remaining θ and σ, adding the prior once.
# Holding φ fixed, a σ coordinate that trades off against a random effect (a
# rate against every subject's amplitude) could only move as fast as the chains
# followed it, and SAEM stopped hundreds of iterations short (snag
# `saem-population-8975f340`). Returns θ, σ, ω and the θ of the Gaussian block,
# whose φ the chains keep (`_saem_recenter!`). With `:online`, only the
# numerical block is damped by the SA step size γ; the Gaussian block already
# uses the stochastically averaged statistics. While γ = 1 the Gaussian block's
# Ω is annealed (`_saem_anneal`). Under minibatch SAEM the numerical update
# evaluates only the subjects at positions `active`.
function _saem_mstep(numeric_update, model, subjects, layout, θ, σ, ω, indices,
        design, free, bank, weights, mean_phi, second_phi, options, γ,
        constraints, map_objective, active = eachindex(subjects);
        identified = _saem_identified(design, indices))
    ω_previous = ω
    θ, Ω = _saem_closed_form_mstep(layout, θ, omega_matrix(layout, ω), indices,
        design, mean_phi, second_phi, options.omega_floor, subjects, constraints;
        require_instances = !map_objective, identified)
    ω = _apply_omega_constraints(constraints, omega_parameters(layout, Ω))
    if map_objective
        θ, ω = _saem_gaussian_map_block(model, layout, θ, σ, ω, indices, design,
            mean_phi, second_phi, subjects, options.omega_floor, constraints; identified)
    end
    if options.omega_annealing > 0 && γ == 1
        ω = _saem_anneal(layout, ω_previous, ω, options.omega_annealing, constraints)
    end
    θ_centered = θ
    # Without other numerical coordinates there is no ridge to follow, and the
    # model needs no population gradient.
    isempty(free) && return θ, σ, ω, θ_centered
    etas = [_saem_eta(Φ, θ, indices, design) for Φ in bank]
    nodesign = [zeros(size(X, 1), 0) for X in design]
    θ, σ = numeric_update(model, subjects, θ, σ, sort!(vcat(indices, free)), Int[],
        nodesign, etas, weights, options.numeric_maxiter, γ,
        map_objective ? ω : nothing, active)::Tuple{Vector{Float64},Vector{Float64}}
    _saem_transport!(mean_phi, second_phi, bank, design, θ[indices] - θ_centered[indices])
    return θ, σ, ω, θ_centered
end

# A non-centred step moves every φ = Dθ + η by D δ. Shift the running
# statistics and the retained draws with them, so that they describe the same η
# and the next closed-form step does not pull the mu-referenced θ back.
function _saem_transport!(mean_phi::AbstractMatrix, second_phi, bank, design, δ)
    all(iszero, δ) && return nothing
    for k in axes(mean_phi, 2)
        d = design[k] * δ
        m = mean_phi[:, k]
        second_phi[:, :, k] .+= m * d' + d * m' + d * d'
        mean_phi[:, k] .+= d
        for Φ in bank
            Φ[:, k] .+= d
        end
    end
    return nothing
end

function _saem_transport!(mean_phi::AbstractVector, second_phi, bank, design, δ)
    all(iszero, δ) && return nothing
    for k in eachindex(mean_phi)
        d = design[k] * δ
        m = mean_phi[k]
        for (b, r) in enumerate(_moment_block_ranges(second_phi[k]))
            mr, dr = m[r], d[r]
            second_phi[k][b] = second_phi[k][b] + mr * dr' + dr * mr' + dr * dr'
        end
        mean_phi[k] = m + d
        for Φ in bank
            Φ[k] = Φ[k] + d
        end
    end
    return nothing
end

# Progress phases of an SAEM run under its node, prepared up front so that the
# whole run shows from the start: the Pathfinder start, the MCMC burn-in and the SA
# iterations. A phase is `nothing` when the run has none or progress is off.
function _saem_phases(node, kernel, options, nsubjects)
    warmup = kernel isa WarmupHMCNUTS ?
        prepare_progress!(node, nsubjects; description = "Pathfinder start") : nothing
    burnin = options.mcmc_burnin > 0 ?
        prepare_progress!(node, options.mcmc_burnin; description = "MCMC burn-in") : nothing
    sa = prepare_progress!(node, options.iterations; description = "Stochastic approximation")
    return (; warmup, burnin, sa)
end

# Start the phase that `iteration` belongs to, finishing the burn-in where the SA
# iterations begin, and return it.
function _saem_phase!(phases, iteration, burnin)
    if iteration == 1 && burnin > 0
        start_progress!(phases.burnin)
    elseif iteration == burnin + 1
        finalize_progress!(phases.burnin)
        start_progress!(phases.sa)
    end
    return iteration <= burnin ? phases.burnin : phases.sa
end

# Finish whichever phase was running when the iterations ended. Phases never
# reached (after an interrupt) are skipped when the run's node finishes.
_saem_finish_phase!(::Nothing) = nothing
_saem_finish_phase!(phase) = is_running(phase) ? finalize_progress!(phase) : nothing

# A phase's running message: the current stage, the SA step size γ, the summed
# conditional log-density of the chains at the current population estimate,
# and the kernel's statistics. Absent parts are `nothing`.
function _saem_report!(phase, stage, γ, loglikelihood, kernel_stats)
    parts = String[]
    stage === nothing || push!(parts, stage)
    γ === nothing || push!(parts, γ == 1 ? "γ = 1 (exploration)" :
        "γ = $(short_string(γ; sigdigits = 3))")
    loglikelihood === nothing || push!(parts, _saem_loglikelihood_part(loglikelihood))
    push!(parts, kernel_stats)
    update_progress!(phase, join(parts, " · "))
    return nothing
end

# A minibatch iteration evaluates only its batch, so the chains' summed
# conditional log-density is the batch's sum scaled to all fitted subjects.
struct _SAEMBatchEstimate
    value::Float64
    size::Int
end

_saem_loglikelihood_part(value::Real) =
    "Σ log p(y|η) = $(short_string(value; sigdigits = 7))"
_saem_loglikelihood_part(estimate::_SAEMBatchEstimate) =
    "Σ log p(y|η) ≈ $(short_string(estimate.value; sigdigits = 7)) " *
    "(batch of $(estimate.size))"

_saem_batch_estimate(ℓ, active, n) =
    _SAEMBatchEstimate(sum(view(ℓ, active)) * n / length(active), length(active))

_saem_kernel_stats(::StandardSAEMMH, nprior, nrw, ncw, ncw_proposed, denom, ndivergent,
        mala_states) =
    "acceptance $(short_string(Fraction(nprior / denom))) prior, " *
    "$(short_string(Fraction(nrw / denom))) random walk, " *
    "$(short_string(Fraction(ncw / ncw_proposed))) per coordinate"
_saem_kernel_stats(::WarmupHMCNUTS, nprior, nrw, ncw, ncw_proposed, denom, ndivergent,
    mala_states) = "$ndivergent divergent transitions"
_saem_kernel_stats(::DualAveragingMALA, nprior, nrw, ncw, ncw_proposed, denom, ndivergent,
        mala_states) =
    "mean MALA step size $(short_string(
        sum(s -> exp(s.log_stepsize), mala_states) / length(mala_states); sigdigits = 3))"

_saem_stop_message(completed, options) =
    "stopped by interrupt after $completed of $(options.iterations) SA iterations"

"""
    saem(model, θ0, σ0, ω0; rng=Random.default_rng(), subjects=nothing,
         η0=nothing, options=SAEMOptions(), kernel=StandardSAEMMH(),
         progress=nothing) -> SAEMResult

Estimate population parameters using MCMC stochastic approximation. Subject
chains persist across iterations. The Gaussian random-effect prior supplies a
closed-form covariance M-step; [`MuReferencing`](@ref) also supplies the
generalized-least-squares fixed-effect M-step. Remaining θ and σ coordinates
use a first-order AD numerical M-step, which holds each chain's η fixed and so
also moves the mu-referenced θ: a σ coordinate that trades off against a random
effect then moves together with it. Its quasi-Newton search starts from a
difference estimate of the objective's curvature, carried from one M-step to
the next. By default, σ coordinates are updated
online only when all structural θ coordinates have a `MuReferencing`
sufficient-statistic update. Otherwise the historical running
conditional-density objective is used and retains one latent-effect matrix
per SA iteration. Set `numeric_mstep=:online` to force an online update for
undeclared structural θ as a model-specific approximation.
`SAEMResult.numeric_mstep` records the selected mode. `kernel` selects the
standard lockstep MH proposals, batched continuously adapted MALA proposals,
or per-subject in-memory WarmupHMC NUTS transitions.
When [`population_logprior`](@ref) is supplied, the M-step maximizes the same
objectives plus the prior, added once. The mu-referenced θ and Ω move on from
their closed-form update to the maximizer with the prior, a search that does
not evaluate the model, and the numerical update of the remaining θ and σ
includes the prior; so a prior adds no model evaluations to the step. The
model supplies a population-prior gradient or an AD backend from which to
derive it.
Covariance families declared by [`fixed_omega`](@ref) keep their fixed parts:
a fully fixed family is not updated, and a partially fixed family maximizes
the expected Gaussian random-effect density over its estimated standard
deviations and correlation. `ω0` must hold the fixed values
([`apply_fixed_omega`](@ref)); `omega_floor` applies to estimated parts only.
A model rejects a point outside its domain by returning a log density of
`-Inf` or by throwing a `DomainError`, `OverflowError` or `PosDefException`.
The MH and MALA kernels then reject that subject's proposal alone, and the
numerical and MAP M-steps halve their line-search step. The point each M-step
starts from must evaluate finitely.

`SAEMOptions(minibatch_size = b)` moves only `b` randomly drawn subjects'
chains per iteration and evaluates only those subjects in the numerical
M-step (see [`SAEMOptions`](@ref)). That M-step screens only the batch's chains,
so a chain drawn later can lie outside the model's domain at the population
parameters of its next iteration; it is then replaced by a feasible draw from
the random-effect prior. The progress message's `Σ log p(y|η)` is then the
batch's sum scaled to all subjects.

`progress` is a Treebars node (or a backend symbol such as `:term`) under which a
`SAEM` node shows the run's phases: `Pathfinder start` (one subject per step),
`MCMC burn-in` and `Stochastic approximation`, each counting its
iterations. The running message gives the stage (`M-step` while it runs), the SA
step size γ, the chains' summed conditional log-density at the current population
estimate (MH and NUTS) and the kernel's acceptance, divergences or mean step size.
`Treebars.request_interrupt!` on that node stops the run at the next iteration
boundary: `saem` returns the estimate of the last completed iteration with
`interrupted = true` and `iterations` counting the completed SA iterations.
"""
function saem(model, θ0::AbstractVector, σ0::AbstractVector, ω0::AbstractVector;
        rng::AbstractRNG = Random.default_rng(), subjects = nothing, η0 = nothing,
        options::SAEMOptions = SAEMOptions(),
        kernel::AbstractSAEMKernel = StandardSAEMMH(), progress = nothing)
    return with_progress(progress; description = "SAEM") do node
        _saem(nlme_layout(model), model, θ0, σ0, ω0, rng, subjects, η0,
            options, kernel, node)
    end
end

# The equal-size parts of `_saem`; `src/ragged.jl` has the ragged ones.
_saem_check_subjects(::NLMELayout, model) = nothing
function _saem_start(layout::NLMELayout, η0, subjects)
    H = η0 === nothing ? zeros(neta(layout), length(subjects)) : Matrix{Float64}(η0)
    size(H) == (neta(layout), length(subjects)) || throw(DimensionMismatch(
        "η0 has size $(size(H)); expected ($(neta(layout)), $(length(subjects)))"))
    return H
end
_saem_statistics(layout::NLMELayout, subjects, H) =
    (zeros(neta(layout), length(subjects)), zeros(neta(layout), neta(layout), length(subjects)))
_saem_ncoordinates(layout::NLMELayout) = neta(layout)
_saem_result(::NLMELayout, fields...) = SAEMResult(fields...)

# One loop for both layouts and every kernel. The layout decides the η storage
# (a matrix, or one compact vector per subject), the running statistics, the
# prior and the result type; each kernel works subject by subject on either.
function _saem(layout, model, θ0, σ0, ω0, rng, subjects, η0, options, kernel, node)
    _saem_check_options(options)
    kernel isa WarmupHMCNUTS && _saem_check_nuts(kernel)
    kernel isa DualAveragingMALA && _saem_check_mala(kernel)
    _saem_check_subjects(layout, model)
    _check_length("θ", θ0, layout.ntheta)
    _check_length("σ", σ0, layout.nsigma)
    _check_length("ω", ω0, nomega(layout))
    constraints = _checked_omega_constraints(model, layout, ω0)
    ids = subjects === nothing ? collect(1:nsubjects(model)) : collect(Int, subjects)
    !isempty(ids) && allunique(ids) && all(i -> 1 <= i <= nsubjects(model), ids) ||
        throw(ArgumentError("subjects must be distinct valid model subject indices"))
    indices, design = _saem_mu_trait(model, ids, layout)
    θ, σ, ω = Float64.(θ0), Float64.(σ0), Float64.(ω0)
    initial_lp = population_logprior(model, θ, σ, ω)
    initial_lp === nothing || initial_lp isa Real || throw(ArgumentError(
        "population_logprior must return a real log density or nothing"))
    map_objective = initial_lp !== nothing
    map_objective && !isfinite(initial_lp) && throw(DomainError(initial_lp,
        "SAEM initial population log prior must be finite"))
    if map_objective
        gradient_lp = first(population_logprior_and_gradient(model, θ, σ, ω))
        isapprox(gradient_lp, initial_lp; rtol = 1e-10, atol = 1e-10) ||
            throw(ArgumentError("population_logprior_and_gradient value differs from population_logprior"))
    end
    identified = _saem_identified(design, indices)
    map_objective || _saem_require_identified(model, indices, identified)
    H = _saem_start(layout, η0, ids)
    free = [i for i in 1:(layout.ntheta + layout.nsigma) if i > layout.ntheta || !(i in indices)]
    prior = _re_prior(layout, ω)
    ℓ = conditional_loglikelihoods(model, ids, θ, σ, H)
    length(ℓ) == length(ids) && all(isfinite, ℓ) || throw(DomainError(ℓ,
        "SAEM initial conditional log densities must be finite and one per subject"))
    bank = typeof(H)[]
    weights = Float64[]
    batch = _saem_minibatch(options.minibatch_size, length(ids))
    numeric_mode = _saem_numeric_mode(options.numeric_mstep, free, layout.ntheta, batch)
    numeric_update = batch === nothing ?
        Base.invokelatest(_SAEMNumericUpdate, numeric_mode)::Function :
        Base.invokelatest(_SAEMScoreStep, batch / length(ids))::Function
    mean_phi, second_phi = _saem_statistics(layout, ids, H)
    logscale = log(options.rw_scale)
    component_logscales = fill(log(options.rw_scale), _saem_ncoordinates(layout))
    nprior = nrw = ncw = ncw_proposed = sweeps = moved = 0
    ndivergent = 0
    mala_states = kernel isa DualAveragingMALA ?
        _saem_mala_states(kernel, length(ids)) : nothing
    mala_cache = kernel isa DualAveragingMALA ? _saem_mala_cache(H) : nothing
    mala_metrics = kernel isa DualAveragingMALA ?
        _saem_mala_metrics(kernel, model, ids, θ, σ, prior, H) : nothing
    # Chains moved (re-centred or replaced) since their last NUTS transition; each
    # moves its chain to the new position before its next transition.
    nuts_moved = falses(length(ids))
    phases = _saem_phases(node, kernel, options, length(ids))
    completed = 0
    interrupted = false
    loglikelihood = nothing
    nuts = kernel isa WarmupHMCNUTS ? _saem_nuts_start!(rng, kernel,
        model, ids, θ, σ, prior, H, phases.warmup) : nothing
    interrupted = kernel isa WarmupHMCNUTS && nuts === nothing
    for iteration in 1:(interrupted ? 0 : options.mcmc_burnin + options.iterations)
        if interrupt_requested(node)
            interrupted = true
            break
        end
        phase = _saem_phase!(phases, iteration, options.mcmc_burnin)
        active = iteration <= max(options.mcmc_burnin, 1) ? Base.OneTo(length(ids)) :
            _saem_active(rng, length(ids), batch)
        if iteration > options.mcmc_burnin
            prior = _re_prior(layout, ω)
            if batch !== nothing
                replaced = _saem_batch_loglikelihoods!(rng, model, ids, θ, σ,
                    prior, H, ℓ, active)
                mala_cache === nothing || (mala_cache.valid[replaced] .= false)
                nuts_moved[replaced] .= true
                loglikelihood = _saem_batch_estimate(ℓ, active, length(ids))
            elseif kernel isa StandardSAEMMH || kernel isa WarmupHMCNUTS
                ℓ = _saem_current_loglikelihoods(model, ids, θ, σ, H)
                loglikelihood = sum(ℓ)
            end
        end
        for _ in 1:options.mh_rounds
            if kernel isa StandardSAEMMH
                Hb, ℓb = _columns(H, active), ℓ[active]
                ap, ar, ac, apc = _saem_mh_round!(rng, model, ids[active], θ, σ, prior,
                    Hb, ℓb, exp(logscale), exp.(component_logscales))
                _set_columns!(H, active, Hb)
                ℓ[active] = ℓb
                nprior += ap
                nrw += ar
                ncw += sum(ac)
                ncw_proposed += sum(apc)
            elseif kernel isa WarmupHMCNUTS
                restart = [k for k in active if nuts_moved[k]]
                _saem_nuts_reposition!(nuts, H, restart)
                nuts_moved[restart] .= false
                ndivergent += _saem_nuts_round!(nuts, model, ids, θ, σ,
                    prior, H, active)
            elseif kernel isa DualAveragingMALA
                _saem_mala_batch_round!(rng, kernel, mala_states, model, ids, θ, σ,
                    prior, H, active; metrics=mala_metrics, cache=mala_cache)
            else
                throw(ArgumentError("unsupported SAEM kernel $(typeof(kernel))"))
            end
            sweeps += 1
            moved += length(active)
            if kernel isa StandardSAEMMH && sweeps <= options.adapt_iterations
                logscale = _saem_adapt_mh_scales!(logscale, component_logscales, ar,
                    length(active), ac, apc, sweeps)
            end
        end
        if iteration <= options.mcmc_burnin
            if phase !== nothing
                update_progress!(phase, iteration)
                _saem_report!(phase, nothing, nothing, nothing, _saem_kernel_stats(kernel,
                    nprior, nrw, ncw, ncw_proposed, moved, ndivergent, mala_states))
            end
            continue
        end
        k = iteration - options.mcmc_burnin
        γ = k <= options.burnin ? 1.0 : (k - options.burnin)^(-options.step_exponent)
        Φ = _saem_phi(H, θ, indices, design)
        _saem_moments!(mean_phi, second_phi, Φ, γ)
        _saem_record!(numeric_mode, bank, weights, Φ, γ)
        if k % options.mstep_interval == 0 || k == options.iterations
            phase === nothing || _saem_report!(phase, "M-step", γ, loglikelihood,
                _saem_kernel_stats(kernel, nprior, nrw, ncw, ncw_proposed,
                    moved, ndivergent, mala_states))
            θ_before, σ_before = copy(θ), copy(σ)
            θ, σ, ω, θ_kept = _saem_mstep(numeric_update, model, ids, layout, θ, σ, ω,
                indices, design, free, bank, weights, mean_phi, second_phi,
                options, γ, constraints, map_objective, active;
                identified)
            if mala_cache !== nothing && any(mala_cache.valid)
                if θ == θ_before && σ == σ_before
                    _saem_mala_refresh_prior!(mala_cache, H, prior,
                        _re_prior(layout, ω), ids)
                else
                    mala_cache.valid .= false
                end
            end
            _saem_recenter!(H, θ_before, θ_kept, indices, design) && (nuts_moved .= true)
        end
        completed = k
        if phase !== nothing
            update_progress!(phase, k)
            _saem_report!(phase, nothing, γ, loglikelihood, _saem_kernel_stats(kernel,
                nprior, nrw, ncw, ncw_proposed, moved, ndivergent, mala_states))
        end
    end
    _saem_finish_phase!(phases.burnin)
    _saem_finish_phase!(phases.sa)
    interrupted && update_progress!(node, _saem_stop_message(completed, options))
    denom = moved
    is_mh = kernel isa StandardSAEMMH
    is_mala = kernel isa DualAveragingMALA
    final_lp = map_objective ? population_logprior(model, θ, σ, ω) : 0.0
    final_lp isa Real && isfinite(final_lp) || throw(DomainError(final_lp,
        "SAEM final population log prior must be finite"))
    return _saem_result(layout, ids, θ, σ, ω, omega_matrix(layout, ω), H,
        (prior = is_mh ? nprior / denom : missing,
            random_walk = is_mh ? nrw / denom : missing,
            coordinates = is_mh ? ncw / ncw_proposed : missing),
        is_mh ? exp(logscale) : missing,
        is_mh ? :standard_mh : is_mala ? :dual_averaging_mala : :warmuphmc_nuts,
        ndivergent,
        is_mala ? exp.([s.log_stepsize for s in mala_states]) : missing,
        is_mala ? [s.accepted / s.count for s in mala_states] : missing,
        is_mala ? minimum(s -> s.count, mala_states) : 0,
        _saem_mode_name(numeric_mode), map_objective ? :MAP : :ML,
        final_lp, map_objective ? population_prior_provenance(model) : nothing,
        completed, interrupted)
end
