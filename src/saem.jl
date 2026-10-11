"""
    SAEMOptions(; iterations=300, burnin=150, mcmc_burnin=20,
                 mh_rounds=1, rw_scale=0.5, adapt_iterations=150,
                 step_exponent=0.8, numeric_maxiter=12, mstep_interval=1,
                 numeric_mstep=:auto,
                 omega_floor=1e-8)

Settings for [`saem`](@ref). The first `burnin` stochastic-approximation
iterations use step size one; subsequent sizes are `(k-burnin)^(-step_exponent)`.
`mcmc_burnin` runs kernel sweeps at the starting population parameters without
updating them. With `StandardSAEMMH`, each round makes one prior-independence
and one random-walk proposal for every subject, each evaluated in one batch.
The random-walk scale adapts toward 0.3 acceptance for the first
`adapt_iterations` sweeps and is then frozen. With `WarmupHMCNUTS`, each round
makes one fixed-kernel NUTS transition per subject. `omega_floor` is a
covariance eigenvalue floor in the M-step.
`numeric_mstep=:auto` uses an online update for remaining σ coordinates when
all θ coordinates have a closed-form `MuReferencing` update; otherwise it uses
the historical running-mixture objective. `:online` explicitly applies the
online update to every remaining θ and σ coordinate: it optimizes the current
complete-data conditional likelihood, then moves toward that optimizer by
the SA step size, retaining one sampled latent-effect matrix. Its use for a
structural θ without a declared sufficient statistic is an approximation.
`:mixture` optimizes the exact running conditional-density mixture and retains
every sampled matrix. The default
`mstep_interval=1` updates population parameters every SA iteration; a larger
interval is an approximate, lower-frequency variant in `:online` mode.
With `DualAveragingMALA`, each round makes one batched MALA proposal and
updates a separate subject step size throughout the run. Set `mcmc_burnin=0`
to omit a separate initial MCMC phase; `burnin` still controls the SA step sizes.
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
end

abstract type AbstractSAEMKernel end

"""
    StandardSAEMMH()

The standard SAEM per-subject kernel: one prior-independence and one
covariance-scaled random-walk Metropolis proposal per subject and MH round.
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
falling back to Ω if the model has no Hessian. Gradients are evaluated in
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

mutable struct _MALACache
    f::Vector{Float64}
    G::Matrix{Float64}
    valid::Bool
end

_saem_mala_cache(H) = _MALACache(zeros(size(H, 2)), zeros(size(H)), false)

function _saem_mala_refresh_prior!(cache::_MALACache, H, old_prior, new_prior)
    cache.valid || return nothing
    for j in axes(H, 2)
        cache.f[j] += re_logprior(new_prior, view(H, :, j)) -
            re_logprior(old_prior, view(H, :, j))
    end
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
    return [cholesky(Symmetric(_saem_hessian_metric(model, i, θ, σ, prior,
        copy(view(H, :, j)); curvature_floor=k.curvature_floor))).L
        for (j, i) in enumerate(ids)]
end

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
    forward = L \ (η1 - η0 - half_e2 * (L * (L' * g0)))
    reverse = L \ (η0 - η1 - half_e2 * (L * (L' * g1)))
    return f1 - f0 + (sum(abs2, forward) - sum(abs2, reverse)) /
        (2 * stepsize^2)
end

function _saem_mala_proposal_gradients(model, subjects, θ, σ, prior, proposal, H)
    n = size(H, 2)
    rejected = falses(n)
    for j in 1:n
        if !all(isfinite, view(proposal, :, j))
            rejected[j] = true
            proposal[:, j] = H[:, j]
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
            size(G1) != size(H) || !all(isfinite, f1) || !all(isfinite, G1)
        # Most proposals need only the two batched value/gradient calls. If one
        # proposal leaves the model's domain, identify it with a value batch,
        # replace it by its current position, then recompute the safe batch.
        values = conditional_loglikelihoods(model, subjects, θ, σ, proposal)
        length(values) == n || throw(DimensionMismatch(
            "MALA proposal returned $(length(values)) values for $n subjects"))
        for j in 1:n
            invalid = !isfinite(values[j]) ||
                !isfinite(re_logprior(prior, view(proposal, :, j)))
            if !invalid && G1 !== nothing && size(G1) == size(H)
                invalid = !all(isfinite, view(G1, :, j))
            end
            if invalid
                rejected[j] = true
                proposal[:, j] = H[:, j]
            end
        end
        any(rejected) || (first_error === nothing ?
            throw(DomainError(f1, "MALA proposed values or gradients are invalid")) :
            throw(first_error))
        _, f1, G1 = _complete_gradients(model, subjects, θ, σ, prior, proposal)
    end
    length(f1) == n && size(G1) == size(H) && all(isfinite, f1) &&
        all(isfinite, G1) || throw(DomainError(f1,
        "MALA proposed values or gradients remain invalid after rejection repair"))
    return f1, G1, rejected
end

function _saem_mala_round!(rng, kernel::DualAveragingMALA, states,
        model, subjects, θ, σ, prior, H; adapt::Bool=true,
        metrics=nothing, cache=nothing)
    m, n = size(H)
    if cache !== nothing && cache.valid
        f0, G0 = cache.f, cache.G
    else
        _, f0, G0 = _complete_gradients(model, subjects, θ, σ, prior, H)
        if cache !== nothing
            cache.f .= f0
            cache.G .= G0
            cache.valid = true
            f0, G0 = cache.f, cache.G
        end
    end
    size(G0) == size(H) && length(f0) == n && all(isfinite, f0) && all(isfinite, G0) ||
        throw(DomainError(f0, "MALA current values or gradients are invalid"))
    noise = randn(rng, m, n)
    proposal = similar(H)
    for j in 1:n
        L = metrics === nothing ? prior.cholesky_factor : metrics[j]
        e = exp(states[j].log_stepsize)
        proposal[:, j] = view(H, :, j) + 0.5 * e^2 *
            (L * (L' * view(G0, :, j))) +
            e * (L * view(noise, :, j))
    end
    f1, G1, rejected = _saem_mala_proposal_gradients(model, subjects, θ, σ,
        prior, proposal, H)
    for j in 1:n
        L = metrics === nothing ? prior.cholesky_factor : metrics[j]
        e = exp(states[j].log_stepsize)
        lr = rejected[j] ? -Inf : _saem_mala_logratio(f0[j], f1[j],
            view(H, :, j), view(proposal, :, j), view(G0, :, j),
            view(G1, :, j), L, e)
        isnan(lr) && throw(DomainError(lr, "MALA log acceptance ratio is invalid"))
        if log(rand(rng)) < min(0.0, lr)
            H[:, j] = proposal[:, j]
            states[j].accepted += 1
            if cache !== nothing
                cache.f[j] = f1[j]
                cache.G[:, j] = G1[:, j]
            end
        end
        adapt && _saem_mala_adapt!(states[j], kernel, exp(min(0.0, lr)))
    end
    return nothing
end

"""
    WarmupHMCNUTS(; warmup_draws=20, warmup_evaluations=100,
                   stepsize_adaptation_limit=20, max_tree_depth=8,
                   target_acceptance_rate=0.8,
                   curvature_floor=1e-4, metric=:diagonal)

Warm up a separate NUTS chain for each subject with WarmupHMC. Its initial
covariance comes from the inverse local negative log-posterior Hessian when
available, and otherwise from Ω. Subsequent SAEM rounds use fixed-kernel
in-memory NUTS transitions with the current population parameters. Warmup
checkpoints are temporary and removed when `saem` returns. The default
`:diagonal` metric uses Hessian diagonal curvatures; `metric=:dense` requests
the inverse regularized Hessian as a full initial covariance.
`target_acceptance_rate` controls WarmupHMC's step-size adaptation during
the one-time subject warmup; it must be strictly between zero and one.
"""
Base.@kwdef struct WarmupHMCNUTS <: AbstractSAEMKernel
    warmup_draws::Int = 20
    warmup_evaluations::Int = 100
    stepsize_adaptation_limit::Int = 20
    max_tree_depth::Int = 8
    target_acceptance_rate::Float64 = 0.8
    curvature_floor::Float64 = 1e-4
    metric::Symbol = :diagonal
end

"""
    SAEMResult

Population estimate from [`saem`](@ref). `eta` holds the final subject-chain
states, one column per entry of `subjects`. `acceptance` gives prior and
random-walk acceptance fractions; `rw_scale` is the final proposal scale.
These MH-only fields are `missing` for other kernels. `kernel` identifies the
method and `nuts_divergences` counts its fixed-kernel transitions that diverged.
For MALA, `mala_stepsizes` and `mala_acceptance` hold per-subject final values;
`mala_adaptation_steps` counts updates per subject. `numeric_mstep` records
the resolved numerical M-step mode (`:online` or `:mixture`).
"""
struct SAEMResult{T, A, R}
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
end

struct _SAEMSubjectTarget{M, P}
    model::M
    subject::Int
    θ::Vector{Float64}
    σ::Vector{Float64}
    prior::P
end

LDP.dimension(target::_SAEMSubjectTarget) = neta(target.prior)
LDP.capabilities(::Type{<:_SAEMSubjectTarget}) = LDP.LogDensityOrder{1}()
LDP.logdensity(target::_SAEMSubjectTarget, η) =
    complete_data_loglikelihood(target.model, target.subject,
        target.θ, target.σ, target.prior, η)
function LDP.logdensity_and_gradient(target::_SAEMSubjectTarget, η)
    ℓ, g = conditional_loglikelihood_and_gradient(target.model, target.subject,
        target.θ, target.σ, η)
    return ℓ + re_logprior(target.prior, η), g - target.prior.precision * η
end

_saem_target(model, subject, θ, σ, prior) =
    _SAEMSubjectTarget(model, subject, copy(θ), copy(σ), prior)

function _saem_check_nuts(o::WarmupHMCNUTS)
    o.warmup_draws > 0 || throw(ArgumentError("warmup_draws must be positive"))
    o.warmup_evaluations > 0 || throw(ArgumentError("warmup_evaluations must be positive"))
    o.stepsize_adaptation_limit > 0 || throw(ArgumentError(
        "stepsize_adaptation_limit must be positive"))
    o.max_tree_depth > 0 || throw(ArgumentError("max_tree_depth must be positive"))
    isfinite(o.target_acceptance_rate) && 0 < o.target_acceptance_rate < 1 ||
        throw(ArgumentError("target_acceptance_rate must be strictly between 0 and 1"))
    isfinite(o.curvature_floor) && o.curvature_floor > 0 || throw(ArgumentError(
        "curvature_floor must be finite and positive"))
    o.metric in (:diagonal, :dense) || throw(ArgumentError(
        "metric must be :diagonal or :dense"))
    return nothing
end

function _saem_nuts_warmup!(rng, kernel::WarmupHMCNUTS,
        root, model, ids, θ, σ, prior, H)
    return [begin
        target = _saem_target(model, i, θ, σ, prior)
        η = copy(view(H, :, k))
        Σ = _saem_hessian_metric(model, i, θ, σ, prior, η;
            curvature_floor=kernel.curvature_floor,
            diagonal=kernel.metric == :diagonal)
        warm_dir = joinpath(root, "warm_$(k)")
        subject_rng = Xoshiro(rand(rng, UInt64))
        WarmupHMC.adaptive_warmup_mcmc(subject_rng, target;
            init=(position=η, squared_scale=Σ),
            checkpoint_dir=warm_dir,
            n_draws=kernel.warmup_draws,
            n_evaluations=kernel.warmup_evaluations,
            recording_target=kernel.warmup_draws,
            stepsize_adaptation_limit=kernel.stepsize_adaptation_limit,
            max_tree_depth=kernel.max_tree_depth,
            target_acceptance_rate=kernel.target_acceptance_rate,
            nonlinear_adapt=false,
            monitor_ess=false)
        WarmupHMC.FixedKernelNUTSState(joinpath(warm_dir, "cp_latest.jls");
            max_tree_depth=kernel.max_tree_depth)
    end for (k, i) in enumerate(ids)]
end

function _saem_nuts_round!(states, model, ids, θ, σ, prior, H)
    divergences = 0
    for (k, i) in enumerate(ids)
        target = _saem_target(model, i, θ, σ, prior)
        step = WarmupHMC.fixed_kernel_nuts_step(states[k], target)
        states[k] = step.state
        H[:, k] = step.state.position
        divergences += step.divergent
    end
    return divergences
end

# Initial covariance for a per-subject HMC warmup. The Hessian is taken only in
# η_i; an indefinite local negative Hessian is regularized before inversion.
function _saem_hessian_metric(model, i, θ, σ, prior::NormalREPrior, η;
        curvature_floor::Real = 1e-4, diagonal::Bool = false)
    isfinite(curvature_floor) && curvature_floor > 0 || throw(ArgumentError(
        "curvature_floor must be finite and positive"))
    if !provides_eta_hessian(model)
        L = prior.cholesky_factor
        diagonal && return [sum(abs2(L[j, k]) for k in 1:j)
            for j in 1:neta(prior)]
        return Matrix(L * L')
    end
    _, _, _, H = complete_data_loglikelihood_gradient_and_hessian(
        model, i, θ, σ, prior, η)
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
    return nothing
end

function _saem_mh_round!(rng, model, subjects, θ, σ, prior, H, ℓ, scale)
    m, n = size(H)
    proposal = Matrix(prior.cholesky_factor * randn(rng, m, n))
    proposed_ℓ = conditional_loglikelihoods(model, subjects, θ, σ, proposal)
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
    proposed_ℓ = conditional_loglikelihoods(model, subjects, θ, σ, proposal)
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
    return prior_accepted, rw_accepted
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

function _saem_gaussian_mstep(layout, θ, Ω, indices, design, mean_phi, second_phi, floor)
    m, n = size(mean_phi)
    μ = copy(θ[indices])
    Ωnew = copy(Ω)
    ranges, _ = _block_ranges(layout)
    converged = false
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
            μ = cholesky(Symmetric(A)) \ b
        end
        second_residual = zeros(Float64, m, m)
        for k in 1:n
            xμ = design[k] * μ
            φ = view(mean_phi, :, k)
            second_residual += view(second_phi, :, :, k) - φ * xμ' - xμ * φ' + xμ * xμ'
        end
        second_residual ./= n
        nextΩ = zeros(Float64, m, m)
        for r in ranges
            E = eigen(Symmetric(second_residual[r, r]))
            nextΩ[r, r] = E.vectors * Diagonal(max.(E.values, floor)) * E.vectors'
        end
        if maximum(abs, nextΩ - Ωnew) <= 1.0e-8 * max(1.0, maximum(abs, Ωnew))
            Ωnew = nextΩ
            converged = true
            break
        end
        Ωnew = nextΩ
    end
    converged || throw(ErrorException("SAEM Gaussian M-step did not converge in 100 coordinate updates"))
    θnew = copy(θ)
    θnew[indices] = μ
    return θnew, Ωnew
end

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
function _saem_record!(::Val{:online}, bank, weights, Φ, γ)
    empty!(bank)
    empty!(weights)
    push!(bank, copy(Φ))
    push!(weights, 1.0)
    return nothing
end

_saem_numeric_mode(mode::Symbol, free, ntheta) =
    Val(mode === :auto ? (any(i -> i <= ntheta, free) ? :mixture : :online) : mode)
_saem_mode_name(::Val{M}) where {M} = M

function _saem_numeric_objective(z, model, subjects, θ, σ, free, indices, design, bank, weights)
    x = vcat(θ, σ)
    x[free] = z
    θz = x[1:length(θ)]
    σz = x[length(θ)+1:end]
    value = zero(eltype(x))
    for (Φ, w) in zip(bank, weights)
        H = _saem_eta(Φ, θz, indices, design)
        value += w * sum(conditional_loglikelihoods(model, subjects, θz, σz, H))
    end
    return value
end

function _saem_population_value_gradient(::PerSubjectEvaluation, model, subjects, θ, σ, H)
    value = 0.0
    gθ = zeros(Float64, length(θ))
    gσ = zeros(Float64, length(σ))
    for (k, i) in enumerate(subjects)
        ℓ, a, b = conditional_loglikelihood_and_population_gradient(model, i, θ, σ, view(H, :, k))
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

function _saem_numeric_value_gradient(z, model, subjects, θ, σ, free, indices, design, bank, weights)
    x = vcat(θ, σ)
    x[free] = z
    θz = x[1:length(θ)]
    σz = x[length(θ)+1:end]
    value = 0.0
    gradient = zeros(Float64, length(x))
    style = saem_population_gradient_style(model)
    for (Φ, w) in zip(bank, weights)
        H = _saem_eta(Φ, θz, indices, design)
        v, g = _saem_population_value_gradient(style, model, subjects, θz, σz, H)
        value += w * v
        gradient += w * g
    end
    return value, gradient[free]
end

function _saem_numeric_mstep(model, subjects, θ, σ, free, indices, design, bank, weights, maxiter)
    isempty(free) && return θ, σ
    z = vcat(θ, σ)[free]
    objective(z) = _saem_numeric_value_gradient(z, model, subjects, θ, σ,
        free, indices, design, bank, weights)
    value, gradient = objective(z)
    isfinite(value) && all(isfinite, gradient) || throw(DomainError(value,
        "SAEM numerical M-step has non-finite objective or gradient"))
    B = Matrix{Float64}(I, length(z), length(z))
    for _ in 1:maxiter
        maximum(abs, gradient) < 1.0e-6 && break
        direction = B * gradient
        dot(direction, gradient) > 0 || (direction = copy(gradient); B = Matrix{Float64}(I, length(z), length(z)))
        step = 1.0
        accepted = false
        while step >= 1.0e-8
            candidate = z + step * direction
            candidate_value = _saem_numeric_objective(candidate, model, subjects, θ, σ,
                free, indices, design, bank, weights)
            if isfinite(candidate_value) && candidate_value >= value + 1.0e-4 * step * dot(gradient, direction)
                next_value, next_gradient = objective(candidate)
                isfinite(next_value) && all(isfinite, next_gradient) || throw(DomainError(next_value,
                    "SAEM numerical M-step has non-finite objective or gradient"))
                s = candidate - z
                y = gradient - next_gradient
                ys = dot(y, s)
                if ys > 1.0e-12
                    ρ = inv(ys)
                    C = Matrix{Float64}(I, length(z), length(z)) - ρ * s * y'
                    B = C * B * C' + ρ * s * s'
                end
                z, value, gradient = candidate, next_value, next_gradient
                accepted = true
                break
            end
            step /= 2
        end
        accepted || break
    end
    x = vcat(θ, σ)
    x[free] = z
    return x[1:length(θ)], x[length(θ)+1:end]
end

_saem_numeric_update(::Val{:mixture}, model, subjects, θ, σ, free, indices,
    design, bank, weights, maxiter, γ) =
    _saem_numeric_mstep(model, subjects, θ, σ, free, indices, design,
        bank, weights, maxiter)

function _saem_numeric_update(::Val{:online}, model, subjects, θ, σ, free,
        indices, design, bank, weights, maxiter, γ)
    isempty(free) && return θ, σ
    θstar, σstar = _saem_numeric_mstep(model, subjects, θ, σ, free, indices,
        design, bank, weights, maxiter)
    x = vcat(θ, σ)
    xstar = vcat(θstar, σstar)
    x[free] .+= γ .* (xstar[free] .- x[free])
    return x[1:length(θ)], x[length(θ)+1:end]
end

"""
    saem(model, θ0, σ0, ω0; rng=Random.default_rng(), subjects=nothing,
         η0=nothing, options=SAEMOptions(), kernel=StandardSAEMMH()) -> SAEMResult

Estimate population parameters using MCMC stochastic approximation. Subject
chains persist across iterations. The Gaussian random-effect prior supplies a
closed-form covariance M-step; [`MuReferencing`](@ref) also supplies the
generalized-least-squares fixed-effect M-step. Remaining θ and σ coordinates
use a first-order AD numerical M-step. By default, σ coordinates are updated
online only when all structural θ coordinates have a `MuReferencing`
sufficient-statistic update. Otherwise the historical running
conditional-density objective is used and retains one latent-effect matrix
per SA iteration. Set `numeric_mstep=:online` to force an online update for
undeclared structural θ as a model-specific approximation.
`SAEMResult.numeric_mstep` records the selected mode. `kernel` selects the
standard lockstep MH proposals, batched continuously adapted MALA proposals,
or per-subject in-memory WarmupHMC NUTS transitions.
"""
function saem(model, θ0::AbstractVector, σ0::AbstractVector, ω0::AbstractVector;
        rng::AbstractRNG = Random.default_rng(), subjects = nothing, η0 = nothing,
        options::SAEMOptions = SAEMOptions(),
        kernel::AbstractSAEMKernel = StandardSAEMMH())
    _saem_check_options(options)
    kernel isa WarmupHMCNUTS && _saem_check_nuts(kernel)
    kernel isa DualAveragingMALA && _saem_check_mala(kernel)
    layout = nlme_layout(model)
    _check_length("θ", θ0, layout.ntheta)
    _check_length("σ", σ0, layout.nsigma)
    _check_length("ω", ω0, nomega(layout))
    ids = subjects === nothing ? collect(1:nsubjects(model)) : collect(Int, subjects)
    !isempty(ids) && allunique(ids) && all(i -> 1 <= i <= nsubjects(model), ids) ||
        throw(ArgumentError("subjects must be distinct valid model subject indices"))
    indices, design = _saem_mu_trait(model, ids, layout)
    θ, σ, ω = Float64.(θ0), Float64.(σ0), Float64.(ω0)
    H = η0 === nothing ? zeros(neta(layout), length(ids)) : Matrix{Float64}(η0)
    size(H) == (neta(layout), length(ids)) || throw(DimensionMismatch(
        "η0 has size $(size(H)); expected ($(neta(layout)), $(length(ids)))"))
    free = [i for i in 1:(layout.ntheta + layout.nsigma) if i > layout.ntheta || !(i in indices)]
    prior = re_prior(model, ω)
    ℓ = conditional_loglikelihoods(model, ids, θ, σ, H)
    length(ℓ) == length(ids) && all(isfinite, ℓ) || throw(DomainError(ℓ,
        "SAEM initial conditional log densities must be finite and one per subject"))
    bank = Matrix{Float64}[]
    weights = Float64[]
    numeric_mode = _saem_numeric_mode(options.numeric_mstep, free, layout.ntheta)
    mean_phi = zeros(neta(layout), length(ids))
    second_phi = zeros(neta(layout), neta(layout), length(ids))
    logscale = log(options.rw_scale)
    nprior = nrw = sweeps = 0
    ndivergent = 0
    root = kernel isa WarmupHMCNUTS ? mktempdir() : nothing
    mala_states = kernel isa DualAveragingMALA ?
        _saem_mala_states(kernel, length(ids)) : nothing
    mala_cache = kernel isa DualAveragingMALA ? _saem_mala_cache(H) : nothing
    mala_metrics = kernel isa DualAveragingMALA ?
        _saem_mala_metrics(kernel, model, ids, θ, σ, prior, H) : nothing
    try
        nuts_state = kernel isa WarmupHMCNUTS ? _saem_nuts_warmup!(rng, kernel,
            root, model, ids, θ, σ, prior, H) : nothing
        for iteration in 1:(options.mcmc_burnin + options.iterations)
            if iteration > options.mcmc_burnin
                prior = re_prior(model, ω)
                if kernel isa StandardSAEMMH
                    ℓ = conditional_loglikelihoods(model, ids, θ, σ, H)
                    length(ℓ) == length(ids) && all(isfinite, ℓ) || throw(DomainError(ℓ,
                        "SAEM conditional log densities became non-finite after population update"))
                end
            end
            for _ in 1:options.mh_rounds
                if kernel isa StandardSAEMMH
                    ap, ar = _saem_mh_round!(rng, model, ids, θ, σ, prior, H, ℓ, exp(logscale))
                    nprior += ap
                    nrw += ar
                elseif kernel isa WarmupHMCNUTS
                    ndivergent += _saem_nuts_round!(nuts_state, model, ids, θ, σ, prior, H)
                elseif kernel isa DualAveragingMALA
                    _saem_mala_round!(rng, kernel, mala_states, model, ids, θ, σ,
                        prior, H; metrics=mala_metrics, cache=mala_cache)
                else
                    throw(ArgumentError("unsupported SAEM kernel $(typeof(kernel))"))
                end
                sweeps += 1
                if kernel isa StandardSAEMMH && sweeps <= options.adapt_iterations
                    logscale = clamp(logscale + (ar / length(ids) - 0.3) / sqrt(sweeps),
                        log(1.0e-3), log(1.0e2))
                end
            end
            iteration <= options.mcmc_burnin && continue
            k = iteration - options.mcmc_burnin
            γ = k <= options.burnin ? 1.0 : (k - options.burnin)^(-options.step_exponent)
            Φ = _saem_phi(H, θ, indices, design)
            _saem_moments!(mean_phi, second_phi, Φ, γ)
            _saem_record!(numeric_mode, bank, weights, Φ, γ)
            if k % options.mstep_interval == 0 || k == options.iterations
                θ_before, σ_before = θ, σ
                θ, Ω = _saem_gaussian_mstep(layout, θ, omega_matrix(layout, ω), indices,
                    design, mean_phi, second_phi, options.omega_floor)
                ω = omega_parameters(layout, Ω)
                θ, σ = _saem_numeric_update(numeric_mode, model, ids, θ, σ,
                    free, indices, design, bank, weights, options.numeric_maxiter, γ)
                if mala_cache !== nothing && mala_cache.valid
                    if θ == θ_before && σ == σ_before
                        _saem_mala_refresh_prior!(mala_cache, H, prior,
                            re_prior(model, ω))
                    else
                        mala_cache.valid = false
                    end
                end
            end
        end
    finally
        root === nothing || _saem_remove_nuts_tempdir(root)
    end
    denom = sweeps * length(ids)
    is_mh = kernel isa StandardSAEMMH
    is_mala = kernel isa DualAveragingMALA
    return SAEMResult(ids, θ, σ, ω, omega_matrix(layout, ω), H,
        (prior = is_mh ? nprior / denom : missing,
            random_walk = is_mh ? nrw / denom : missing),
        is_mh ? exp(logscale) : missing,
        is_mh ? :standard_mh : is_mala ? :dual_averaging_mala : :warmuphmc_nuts,
        ndivergent,
        is_mala ? exp.([s.log_stepsize for s in mala_states]) : missing,
        is_mala ? [s.accepted / s.count for s in mala_states] : missing,
        is_mala ? first(mala_states).count : 0,
        _saem_mode_name(numeric_mode))
end

function _saem_remove_nuts_tempdir(root::AbstractString)
    for attempt in 1:10
        try
            rm(root; recursive=true, force=true)
            return nothing
        catch err
            # Retry only ENOTEMPTY for a bounded interval; preserve all other
            # cleanup failures and surface an ENOTEMPTY that persists.
            if !(err isa Base.IOError && err.code == Base.UV_ENOTEMPTY && attempt < 10)
                rethrow()
            end
            sleep(0.05 * attempt)
        end
    end
end
