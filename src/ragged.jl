# Nested, variable-size random-effect batches. The vector-of-vectors protocol
# keeps each subject's η compact; families carry one covariance shared by all
# their instances, including repeated instances within a subject.

function _ragged_validate_batch(layout::RaggedNLMELayout, subjects, H)
    length(H) == length(subjects) || throw(DimensionMismatch(
        "η batch has $(length(H)) vectors for $(length(subjects)) subjects"))
    for (k, i) in enumerate(subjects)
        _check_length("subject $i η", H[k], neta(layout, i))
    end
    return nothing
end

function _complete_values(model, subjects, θ, σ, prior::RaggedNormalREPrior, H)
    ℓ = conditional_loglikelihoods(model, subjects, θ, σ, H)
    length(ℓ) == length(subjects) || throw(DimensionMismatch(
        "conditional likelihood returned $(length(ℓ)) values for $(length(subjects)) subjects"))
    return ℓ, [ℓ[k] + re_logprior(prior, i, H[k]) for (k, i) in enumerate(subjects)]
end

function _complete_gradients(model, subjects, θ, σ, prior::RaggedNormalREPrior, H)
    ℓ, Gc = conditional_loglikelihoods_and_gradients(model, subjects, θ, σ, H)
    length(ℓ) == length(Gc) == length(subjects) || throw(DimensionMismatch(
        "conditional gradient batch must return one value and gradient per subject"))
    f = similar(ℓ)
    G = Vector{Vector{Float64}}(undef, length(subjects))
    for (k, i) in enumerate(subjects)
        _check_length("subject $i conditional gradient", Gc[k], neta(prior.layout, i))
        lp, gp, _ = re_logprior_gradient_and_hessian(prior, i, H[k])
        f[k] = ℓ[k] + lp
        G[k] = Gc[k] + gp
    end
    return ℓ, f, G
end

function _complete_hessians(model, subjects, θ, σ, prior::RaggedNormalREPrior, H)
    blocks = conditional_loglikelihood_hessian_blocks(model, subjects, θ, σ, H)
    length(blocks) == length(subjects) || throw(DimensionMismatch(
        "conditional Hessian batch must return one block per subject"))
    return [begin
        _, _, Hp = re_logprior_gradient_and_hessian(prior, i, H[k])
        size(blocks[k]) == size(Hp) || throw(DimensionMismatch(
            "subject $i conditional Hessian has size $(size(blocks[k])), expected $(size(Hp))"))
        blocks[k] + Hp
    end for (k, i) in enumerate(subjects)]
end

function _ragged_start(::Nothing, layout, subjects)
    return [zeros(neta(layout, i)) for i in subjects]
end
function _ragged_start(H::AbstractVector{<:AbstractVector}, layout, subjects)
    _ragged_validate_batch(layout, subjects, H)
    return [Float64.(h) for h in H]
end
function _ragged_start(result::RaggedEBEResult, layout, subjects)
    indices = [begin
        k = findfirst(==(i), result.subjects)
        k === nothing && throw(ArgumentError("η0 has no mode for subject $i"))
        k
    end for i in subjects]
    return _ragged_start(result.modes[indices], layout, subjects)
end
_ragged_start(η0, layout, subjects) = throw(ArgumentError(
    "ragged η0 must be nothing, a vector of subject vectors, or a RaggedEBEResult"))

_prior_eltype(::RaggedNormalREPrior{T}) where {T} = T

function _prior_covariance(prior::RaggedNormalREPrior, i)
    n = neta(prior.layout, i)
    Ω = zeros(Float64, n, n)
    for (family, r) in _ragged_subject_blocks(prior.layout, i)
        L = prior.cholesky_factors[family]
        Ω[r, r] = L * L'
    end
    return Ω
end

function _standardized_norm(prior::RaggedNormalREPrior, i, g)
    norm = 0.0
    for (family, r) in _ragged_subject_blocks(prior.layout, i)
        norm = max(norm, maximum(abs, prior.cholesky_factors[family]' * view(g, r); init=0.0))
    end
    return norm
end

function _empirical_bayes(layout::RaggedNLMELayout, model, θ, σ, ω, subjects, η0,
        method::EBEMethod, opt::EBEOptions, require_convergence)
    length(layout.subject_families) == nsubjects(model) || throw(DimensionMismatch(
        "ragged layout has $(length(layout.subject_families)) subjects; model has $(nsubjects(model))"))
    _check_length("θ", θ, layout.ntheta)
    _check_length("σ", σ, layout.nsigma)
    _check_length("ω", ω, nomega(layout))
    ids = subjects === nothing ? collect(1:nsubjects(model)) : collect(Int, subjects)
    !isempty(ids) && allunique(ids) && all(i -> 1 <= i <= nsubjects(model), ids) ||
        throw(ArgumentError("subjects must be distinct valid model subject indices"))
    H = _ragged_start(η0, layout, ids)
    result = _ebe_lockstep(method, model, ids, θ, σ, RaggedNormalREPrior(layout, ω), H, opt)
    bad = ids[.!result.converged]
    require_convergence && !isempty(bad) && throw(EBEConvergenceError(bad, result))
    return result
end

function _saem_mu_trait(model, subjects, layout::RaggedNLMELayout)
    mr = mu_referencing(model)
    if mr === nothing
        return Int[], [zeros(neta(layout, i), 0) for i in subjects]
    end
    length(mr.design) == nsubjects(model) || throw(DimensionMismatch(
        "mu_referencing has $(length(mr.design)) designs for $(nsubjects(model)) subjects"))
    allunique(mr.theta_indices) && all(i -> 1 <= i <= layout.ntheta, mr.theta_indices) ||
        throw(ArgumentError("mu_referencing has invalid theta_indices"))
    for i in subjects
        size(mr.design[i]) == (neta(layout, i), length(mr.theta_indices)) ||
            throw(DimensionMismatch("mu_referencing design[$i] has size $(size(mr.design[i]))"))
    end
    return mr.theta_indices, [mr.design[i] for i in subjects]
end

_saem_phi(H::AbstractVector{<:AbstractVector}, θ, indices, design) =
    [H[k] + design[k] * view(θ, indices) for k in eachindex(H)]
_saem_eta(Φ::AbstractVector{<:AbstractVector}, θ, indices, design) =
    [Φ[k] - design[k] * view(θ, indices) for k in eachindex(Φ)]

function _ragged_moments!(mean_phi, second_phi, Φ, γ)
    for k in eachindex(Φ)
        mean_phi[k] = (1 - γ) * mean_phi[k] + γ * Φ[k]
        second_phi[k] = (1 - γ) * second_phi[k] + γ * (Φ[k] * Φ[k]')
    end
    return nothing
end

function _ragged_sample_prior(rng, prior::RaggedNormalREPrior, i)
    η = zeros(Float64, neta(prior.layout, i))
    for (family, r) in _ragged_subject_blocks(prior.layout, i)
        η[r] = prior.cholesky_factors[family] * randn(rng, length(r))
    end
    return η
end

function _saem_mh_round!(rng, model, subjects, θ, σ, prior::RaggedNormalREPrior,
        H::AbstractVector{<:AbstractVector}, ℓ, scale)
    n = length(subjects)
    proposal = [_ragged_sample_prior(rng, prior, i) for i in subjects]
    proposed_ℓ = _proposal_loglikelihoods(model, subjects, θ, σ, proposal)
    length(proposed_ℓ) == n && all(x -> !isnan(x) && x != Inf, proposed_ℓ) ||
        throw(DomainError(proposed_ℓ, "prior MH proposal returned invalid densities"))
    prior_accepted = 0
    for k in 1:n
        if log(rand(rng)) < proposed_ℓ[k] - ℓ[k]
            H[k] = proposal[k]
            ℓ[k] = proposed_ℓ[k]
            prior_accepted += 1
        end
    end
    proposal = [H[k] + scale * _ragged_sample_prior(rng, prior, i)
        for (k, i) in enumerate(subjects)]
    proposed_ℓ = _proposal_loglikelihoods(model, subjects, θ, σ, proposal)
    length(proposed_ℓ) == n && all(x -> !isnan(x) && x != Inf, proposed_ℓ) ||
        throw(DomainError(proposed_ℓ, "random-walk MH proposal returned invalid densities"))
    rw_accepted = 0
    for (k, i) in enumerate(subjects)
        ratio = proposed_ℓ[k] - ℓ[k] + re_logprior(prior, i, proposal[k]) -
            re_logprior(prior, i, H[k])
        if log(rand(rng)) < ratio
            H[k] = proposal[k]
            ℓ[k] = proposed_ℓ[k]
            rw_accepted += 1
        end
    end
    return prior_accepted, rw_accepted
end

function _ragged_gaussian_mstep(layout, θ, Ω, indices, design, mean_phi,
        second_phi, floor, subjects, constraints = nothing)
    μ = copy(θ[indices])
    Ωnew = deepcopy(Ω)
    for _ in 1:100
        if !isempty(indices)
            A = zeros(length(indices), length(indices))
            b = zeros(length(indices))
            precisions = [inv(Symmetric(matrix)) for matrix in Ωnew]
            for (k, i) in enumerate(subjects)
                X = design[k]
                for (family, r) in _ragged_subject_blocks(layout, i)
                    Xr = view(X, r, :)
                    P = precisions[family]
                    A += Xr' * P * Xr
                    b += Xr' * P * view(mean_phi[k], r)
                end
            end
            μ = cholesky(Symmetric(A)) \ b
        end
        sums = [zeros(k, k) for k in layout.family_sizes]
        counts = zeros(Int, length(layout.family_sizes))
        for (k, i) in enumerate(subjects)
            xμ = design[k] * μ
            φ = mean_phi[k]
            S = second_phi[k]
            for (family, r) in _ragged_subject_blocks(layout, i)
                a = view(xμ, r)
                p = view(φ, r)
                sums[family] += view(S, r, r) - p * a' - a * p' + a * a'
                counts[family] += 1
            end
        end
        all(f -> counts[f] > 0 || _fully_fixed_family(constraints, f), eachindex(counts)) ||
            throw(ArgumentError(
                "SAEM selected subjects contain no instances of at least one estimated covariance family"))
        nextΩ = [begin
            fam = _omega_family(constraints, f)
            if fam === nothing
                E = eigen(Symmetric(sums[f] / counts[f]))
                Matrix(Symmetric(E.vectors * Diagonal(max.(E.values, floor)) * E.vectors'))
            elseif fam.fully_fixed
                copy(fam.covariance)
            else
                _fixed_family_update(fam, sums[f] / counts[f], Ωnew[f], floor)
            end
        end for f in eachindex(sums)]
        if all(f -> maximum(abs, nextΩ[f] - Ωnew[f]) <=
                1e-8 * max(1.0, maximum(abs, Ωnew[f])), eachindex(Ωnew))
            Ωnew = nextΩ
            θnew = copy(θ)
            θnew[indices] = μ
            return θnew, Ωnew
        end
        Ωnew = nextΩ
    end
    throw(ErrorException("ragged SAEM Gaussian M-step did not converge in 100 coordinate updates"))
end

function _saem_gaussian_map_value_gradient(layout::RaggedNLMELayout, θ, ω,
        indices, design, mean_phi, second_phi, subjects)
    factors = omega_cholesky(layout, ω)
    precisions = [begin
        Linv = inv(L)
        Matrix(Linv' * Linv)
    end for L in factors]
    μ = view(θ, indices)
    sums = [zeros(k, k) for k in layout.family_sizes]
    counts = zeros(Int, length(layout.family_sizes))
    gμ = zeros(Float64, length(indices))
    for (k, i) in enumerate(subjects)
        X = design[k]
        xμ = X * μ
        φ = mean_phi[k]
        S = second_phi[k]
        for (family, r) in _ragged_subject_blocks(layout, i)
            a = view(xμ, r)
            p = view(φ, r)
            Xr = view(X, r, :)
            P = precisions[family]
            sums[family] += view(S, r, r) - p * a' - a * p' + a * a'
            counts[family] += 1
            gμ += Xr' * P * (p - a)
        end
    end
    value = 0.0
    gω = zeros(Float64, length(ω))
    omega_ranges = _ragged_family_ranges(layout)
    for f in eachindex(factors)
        v, g = _gaussian_family_value_gradient(factors[f], precisions[f], sums[f], counts[f])
        value += v
        gω[omega_ranges[f]] = g
    end
    return value, gμ, gω
end

function _saem_omega_above_floor(layout::RaggedNLMELayout, ω, floor)
    covariances = omega_matrix(layout, ω)
    return all(Ω -> all(isfinite, Ω) && eigmin(Symmetric(Ω)) >= floor, covariances)
end

function _saem_floor_omega(layout::RaggedNLMELayout, ω, floor)
    covariances = omega_matrix(layout, ω)
    all(Ω -> all(isfinite, Ω), covariances) || throw(DomainError(ω,
        "SAEM MAP M-step starts with non-finite covariance"))
    floored = [begin
        E = eigen(Symmetric(Ω))
        Matrix(Symmetric(E.vectors * Diagonal(max.(E.values, floor)) * E.vectors'))
    end for Ω in covariances]
    return omega_parameters(layout, floored)
end

function _saem(layout::RaggedNLMELayout, model, θ0, σ0, ω0, rng, subjects, η0,
        options::SAEMOptions, kernel::StandardSAEMMH)
    _saem_check_options(options)
    length(layout.subject_families) == nsubjects(model) || throw(DimensionMismatch(
        "ragged layout has $(length(layout.subject_families)) subjects; model has $(nsubjects(model))"))
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
        isapprox(gradient_lp, initial_lp; rtol=1e-10, atol=1e-10) || throw(ArgumentError(
            "population_logprior_and_gradient value differs from population_logprior"))
    end
    H = _ragged_start(η0, layout, ids)
    free = [i for i in 1:(layout.ntheta + layout.nsigma) if i > layout.ntheta || !(i in indices)]
    prior = RaggedNormalREPrior(layout, ω)
    ℓ = conditional_loglikelihoods(model, ids, θ, σ, H)
    length(ℓ) == length(ids) && all(isfinite, ℓ) || throw(DomainError(ℓ,
        "SAEM initial conditional log densities must be finite and one per subject"))
    bank = Vector{Vector{Float64}}[]
    weights = Float64[]
    numeric_mode = _saem_numeric_mode(options.numeric_mstep, free, layout.ntheta)
    numeric_update = Base.invokelatest(_SAEMNumericUpdate, numeric_mode)::Function
    mean_phi = [zeros(length(h)) for h in H]
    second_phi = [zeros(length(h), length(h)) for h in H]
    logscale = log(options.rw_scale)
    nprior = nrw = sweeps = 0
    for iteration in 1:(options.mcmc_burnin + options.iterations)
        if iteration > options.mcmc_burnin
            prior = RaggedNormalREPrior(layout, ω)
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
                    log(1e-3), log(1e2))
            end
        end
        iteration <= options.mcmc_burnin && continue
        k = iteration - options.mcmc_burnin
        γ = k <= options.burnin ? 1.0 : (k - options.burnin)^(-options.step_exponent)
        Φ = _saem_phi(H, θ, indices, design)
        _ragged_moments!(mean_phi, second_phi, Φ, γ)
        _saem_record!(numeric_mode, bank, weights, Φ, γ)
        if k % options.mstep_interval == 0 || k == options.iterations
            if map_objective
                θ, σ, ω = _saem_map_mstep(model, ids, layout, θ, σ, ω,
                    indices, design, bank, weights, mean_phi, second_phi,
                    options.numeric_maxiter, numeric_mode, γ, options.omega_floor,
                    constraints)
            else
                θ, Ω = _ragged_gaussian_mstep(layout, θ, omega_matrix(layout, ω), indices,
                    design, mean_phi, second_phi, options.omega_floor, ids, constraints)
                ω = _apply_omega_constraints(constraints, omega_parameters(layout, Ω))
                θ, σ = numeric_update(model, ids, θ, σ, free, indices, design,
                    bank, weights, options.numeric_maxiter, γ)::Tuple{Vector{Float64},Vector{Float64}}
            end
        end
    end
    denom = sweeps * length(ids)
    final_lp = map_objective ? population_logprior(model, θ, σ, ω) : 0.0
    final_lp isa Real && isfinite(final_lp) || throw(DomainError(final_lp,
        "SAEM final population log prior must be finite"))
    return RaggedSAEMResult(ids, θ, σ, ω, omega_matrix(layout, ω), H,
        (prior=nprior / denom, random_walk=nrw / denom), exp(logscale),
        :standard_mh, _saem_mode_name(numeric_mode), map_objective ? :MAP : :ML,
        final_lp, map_objective ? population_prior_provenance(model) : nothing)
end

_saem(layout::RaggedNLMELayout, model, θ0, σ0, ω0, rng, subjects, η0,
    options::SAEMOptions, kernel::AbstractSAEMKernel) = throw(ArgumentError(
    "ragged SAEM currently supports StandardSAEMMH; got $(typeof(kernel))"))

function _check_ragged_protocol(model, layout::RaggedNLMELayout, θ, σ, ω;
        subjects, η, reference_backend, reference_hessian_backend,
        population_gradient, rtol)
    violation(msg) = throw(ProtocolViolation(typeof(model), msg))
    n = nsubjects(model)
    length(layout.subject_families) == n || violation(
        "ragged layout has $(length(layout.subject_families)) subjects; model has $n")
    evaluation_style(model) isa EvaluationStyle || violation("invalid evaluation_style")
    saem_population_gradient_style(model) isa EvaluationStyle || violation(
        "invalid saem_population_gradient_style")
    length(θ) == layout.ntheta || violation("θ length differs from layout")
    length(σ) == layout.nsigma || violation("σ length differs from layout")
    length(ω) == nomega(layout) || violation("ω length differs from layout")
    _check_fixed_omega(violation, model, layout, ω)
    ids = subjects === nothing ? collect(1:n) : collect(Int, subjects)
    !isempty(ids) && allunique(ids) && all(i -> 1 <= i <= n, ids) || violation(
        "subjects must be distinct valid model subject indices")
    H = η === nothing ? _ragged_start(nothing, layout, ids) :
        η isa AbstractVector{<:AbstractVector} ? _ragged_start(η, layout, ids) :
        violation("ragged η must be one vector per subject")
    mr = mu_referencing(model)
    _saem_mu_trait(model, ids, layout)
    reference = reference_backend === nothing ? nothing :
        (gradient=_checked_backend(model, reference_backend, :check_protocol),
            hessian=reference_hessian_backend === nothing ? nothing :
                _checked_backend(model, reference_hessian_backend, :check_protocol))
    agrees(a, b) = isapprox(a, b; rtol=rtol,
        atol=rtol * (1 + maximum(abs, a; init=0.0)))
    ℓ = conditional_loglikelihoods(model, ids, θ, σ, H)
    length(ℓ) == length(ids) && all(x -> x isa Real && isfinite(x), ℓ) || violation(
        "batched conditional log densities must be finite and one per subject")
    ℓg, G = conditional_loglikelihoods_and_gradients(model, ids, θ, σ, H)
    agrees(ℓ, ℓg) || violation("batched values disagree with gradient values")
    length(G) == length(ids) || violation("batched gradients need one vector per subject")
    for (k, i) in enumerate(ids)
        length(G[k]) == neta(layout, i) || violation("subject $i gradient has wrong length")
    end
    if length(ids) >= 2
        moved = copy(H)
        moved[1] = H[1] .+ 0.1
        changed = conditional_loglikelihoods(model, ids, θ, σ, moved)
        agrees(changed[2:end], ℓ[2:end]) || violation(
            "changing one subject's η changed another subject's density")
    end
    blocks = provides_eta_hessian(model) ?
        conditional_loglikelihood_hessian_blocks(model, ids, θ, σ, H) : nothing
    if blocks !== nothing
        length(blocks) == length(ids) || violation("Hessian batch needs one block per subject")
        for (k, i) in enumerate(ids)
            size(blocks[k]) == (neta(layout, i), neta(layout, i)) || violation(
                "subject $i Hessian has wrong size")
            agrees(blocks[k], blocks[k]') || violation("subject $i Hessian is not symmetric")
        end
    end
    for (k, i) in enumerate(ids)
        ηi = H[k]
        value = conditional_loglikelihood(model, i, θ, σ, ηi)
        agrees(value, ℓ[k]) || violation("subject $i value disagrees with batch")
        vi, gi = conditional_loglikelihood_and_gradient(model, i, θ, σ, ηi)
        agrees(vi, value) && agrees(gi, G[k]) || violation(
            "subject $i gradient disagrees with batch")
        _check_subject_hessian(violation, agrees, blocks, model, k, i, θ, σ, ηi, value, gi)
        _check_reference(violation, agrees, reference, model, i, θ, σ, ηi, gi,
            blocks === nothing ? nothing : blocks[k])
        _check_mu_invariance(violation, agrees, mr, model, i, θ, σ, ηi, value)
        _check_observation_model(violation, agrees, observation_model(model),
            model, i, θ, σ, ηi, value)
        population_gradient && _check_population_gradient(violation, agrees,
            reference, model, i, θ, σ, ηi, value, layout)
    end
    observation_model(model) isa GaussianObservations &&
        _check_observation_batch(violation, agrees, model, ids, θ, σ, H, layout)
    return true
end

function _check_observation_batch(violation, agrees, model, subjects, θ, σ,
        H::AbstractVector{<:AbstractVector}, layout::RaggedNLMELayout)
    ys = observations(model, subjects)
    parts = observation_moments_and_jacobians(model, subjects, θ, σ, H)
    length(ys) == length(parts) == length(subjects) || violation(
        "batched observations and moments need one entry per subject")
    for (k, i) in enumerate(subjects)
        y = observations(model, i)
        agrees(ys[k], y) || violation("subject $i observations disagree with batch")
        μ, v, Jμ, Jv = parts[k]
        μi, vi = observation_moments(model, i, θ, σ, H[k])
        agrees(μ, μi) && agrees(v, vi) || violation("subject $i moments disagree with batch")
        shape = (length(y), neta(layout, i))
        size(Jμ) == size(Jv) == shape || violation("subject $i Jacobian has wrong size")
    end
    return nothing
end
