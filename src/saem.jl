"""
    SAEMOptions(; iterations=300, burnin=150, mcmc_burnin=20,
                 mh_rounds=1, rw_scale=0.5, adapt_iterations=150,
                 step_exponent=0.8, numeric_maxiter=12, mstep_interval=1,
                 omega_floor=1e-8)

Settings for [`saem`](@ref). The first `burnin` stochastic-approximation
iterations use step size one; subsequent sizes are `(k-burnin)^(-step_exponent)`.
`mcmc_burnin` runs MH sweeps at the starting population parameters without
updating them. Each MH round makes one prior-independence and one random-walk
proposal for every subject, each evaluated in one batch. The random-walk scale
adapts toward 0.3 acceptance for the first `adapt_iterations` sweeps and is
then frozen. `omega_floor` is a covariance eigenvalue floor in the M-step.
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
    omega_floor::Float64 = 1.0e-8
end

"""
    SAEMResult

Population estimate from [`saem`](@ref). `eta` holds the final subject-chain
states, one column per entry of `subjects`. `acceptance` gives prior and
random-walk acceptance fractions; `rw_scale` is the final proposal scale.
"""
struct SAEMResult{T}
    subjects::Vector{Int}
    theta::Vector{T}
    sigma::Vector{T}
    omega::Vector{T}
    omega_matrix::Matrix{T}
    eta::Matrix{T}
    acceptance::NamedTuple{(:prior, :random_walk), Tuple{Float64, Float64}}
    rw_scale::Float64
end

# Initial covariance for a per-subject HMC warmup. The Hessian is taken only in
# η_i; an indefinite local negative Hessian is regularized before inversion.
function _saem_hessian_metric(model, i, θ, σ, prior::NormalREPrior, η;
        curvature_floor::Real = 1e-4)
    isfinite(curvature_floor) && curvature_floor > 0 || throw(ArgumentError(
        "curvature_floor must be finite and positive"))
    provides_eta_hessian(model) ||
        return Matrix(prior.cholesky_factor * prior.cholesky_factor')
    _, _, _, H = complete_data_loglikelihood_gradient_and_hessian(
        model, i, θ, σ, prior, η)
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

# The conditional density need not have finite-dimensional sufficient statistics.
# Retain the stochastic-approximation mixture of sampled latent effects so its
# numerical M-step optimizes the same running Q function as the Gaussian step.
function _saem_record!(bank, weights, Φ, γ)
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
    style = evaluation_style(model)
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

"""
    saem(model, θ0, σ0, ω0; rng=Random.default_rng(), subjects=nothing,
         η0=nothing, options=SAEMOptions()) -> SAEMResult

Estimate population parameters using MCMC stochastic approximation. Subject
chains persist across iterations. The Gaussian random-effect prior supplies a
closed-form covariance M-step; [`MuReferencing`](@ref) also supplies the
generalized-least-squares fixed-effect M-step. Remaining θ and σ coordinates
use a first-order AD numerical M-step over the running conditional-density
mixture. The mixture retains one latent-effect matrix per SA iteration, so
memory and numerical M-step cost grow with `iterations`.
"""
function saem(model, θ0::AbstractVector, σ0::AbstractVector, ω0::AbstractVector;
        rng::AbstractRNG = Random.default_rng(), subjects = nothing, η0 = nothing,
        options::SAEMOptions = SAEMOptions())
    _saem_check_options(options)
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
    mean_phi = zeros(neta(layout), length(ids))
    second_phi = zeros(neta(layout), neta(layout), length(ids))
    logscale = log(options.rw_scale)
    nprior = nrw = sweeps = 0
    for iteration in 1:(options.mcmc_burnin + options.iterations)
        if iteration > options.mcmc_burnin
            prior = re_prior(model, ω)
            ℓ = conditional_loglikelihoods(model, ids, θ, σ, H)
            length(ℓ) == length(ids) && all(isfinite, ℓ) || throw(DomainError(ℓ,
                "SAEM conditional log densities became non-finite after population update"))
        end
        for _ in 1:options.mh_rounds
            ap, ar = _saem_mh_round!(rng, model, ids, θ, σ, prior, H, ℓ, exp(logscale))
            nprior += ap
            nrw += ar
            sweeps += 1
            if sweeps <= options.adapt_iterations
                logscale = clamp(logscale + (ar / length(ids) - 0.3) / sqrt(sweeps),
                    log(1.0e-3), log(1.0e2))
            end
        end
        iteration <= options.mcmc_burnin && continue
        k = iteration - options.mcmc_burnin
        γ = k <= options.burnin ? 1.0 : (k - options.burnin)^(-options.step_exponent)
        Φ = _saem_phi(H, θ, indices, design)
        _saem_moments!(mean_phi, second_phi, Φ, γ)
        _saem_record!(bank, weights, Φ, γ)
        if k % options.mstep_interval == 0 || k == options.iterations
            θ, Ω = _saem_gaussian_mstep(layout, θ, omega_matrix(layout, ω), indices,
                design, mean_phi, second_phi, options.omega_floor)
            ω = omega_parameters(layout, Ω)
            θ, σ = _saem_numeric_mstep(model, ids, θ, σ, free, indices,
                design, bank, weights, options.numeric_maxiter)
        end
    end
    denom = sweeps * length(ids)
    return SAEMResult(ids, θ, σ, ω, omega_matrix(layout, ω), H,
        (prior = nprior / denom, random_walk = nrw / denom), exp(logscale))
end
