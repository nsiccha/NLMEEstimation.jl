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
    ℓ = _proposal_loglikelihoods(model, subjects, θ, σ, H)
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
        p = _subject_prior(prior, i)
        f[k] = ℓ[k] + re_logprior(p, H[k])
        G[k] = Gc[k] - _precision_times(p, H[k])
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

# Block diagonal: one family factor per block instance.
function _prior_factor(prior::RaggedNormalREPrior{T}, i) where {T}
    n = neta(prior.layout, i)
    L = zeros(T, n, n)
    for (family, r) in _ragged_subject_blocks(prior.layout, i)
        L[r, r] = prior.cholesky_factors[family]
    end
    return LowerTriangular(L)
end

function _empirical_bayes(layout::RaggedNLMELayout, model, θ, σ, ω, subjects, η0,
        method::EBEMethod, opt::EBEOptions, require_convergence, progress)
    length(layout.subject_families) == nsubjects(model) || throw(DimensionMismatch(
        "ragged layout has $(length(layout.subject_families)) subjects; model has $(nsubjects(model))"))
    _check_length("θ", θ, layout.ntheta)
    _check_length("σ", σ, layout.nsigma)
    _check_length("ω", ω, nomega(layout))
    ids = subjects === nothing ? collect(1:nsubjects(model)) : collect(Int, subjects)
    !isempty(ids) && allunique(ids) && all(i -> 1 <= i <= nsubjects(model), ids) ||
        throw(ArgumentError("subjects must be distinct valid model subject indices"))
    H = _ragged_start(η0, layout, ids)
    result = _ebe_lockstep(method, model, ids, θ, σ, RaggedNormalREPrior(layout, ω), H, opt,
        progress)
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

function _saem_recenter!(H::AbstractVector{<:AbstractVector}, θ_old, θ_new, indices, design)
    Δ = θ_new[indices] - θ_old[indices]
    all(iszero, Δ) && return false
    for k in eachindex(H)
        H[k] -= design[k] * Δ
    end
    return true
end

# The running statistics of a ragged fit: per subject the mean of φ and, for
# each of its blocks in η order, the second moment of that block alone. The
# M-step reads only those blocks (a family's covariance pools its instances), so
# the subject's off-block second moments, n_k² entries for a subject with n_k
# nested effects, are never formed.
function _saem_statistics(layout::RaggedNLMELayout, subjects, H)
    mean_phi = [zeros(length(h)) for h in H]
    second_phi = [[zeros(length(r), length(r)) for (_, r) in _ragged_subject_blocks(layout, i)]
        for i in subjects]
    return mean_phi, second_phi
end

# The η ranges of a subject's blocks, from its block second moments.
function _moment_block_ranges(blocks::AbstractVector{<:AbstractMatrix})
    ends = cumsum(size.(blocks, 1))
    return [(e - size(S, 1) + 1):e for (S, e) in zip(blocks, ends)]
end

function _saem_moments!(mean_phi::AbstractVector{<:AbstractVector}, second_phi, Φ, γ)
    for k in eachindex(Φ)
        mean_phi[k] = (1 - γ) * mean_phi[k] + γ * Φ[k]
        for (b, r) in enumerate(_moment_block_ranges(second_phi[k]))
            φ = Φ[k][r]
            second_phi[k][b] = (1 - γ) * second_phi[k][b] + γ * (φ * φ')
        end
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

_saem_prior_draw(rng, prior::RaggedNormalREPrior, i) = _ragged_sample_prior(rng, prior, i)

function _saem_mh_round!(rng, model, subjects, θ, σ, prior::RaggedNormalREPrior,
        H::AbstractVector{<:AbstractVector}, ℓ, scale, component_scales)
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
    component_accepted, component_proposed = _saem_mh_component_sweep!(rng, model,
        subjects, θ, σ, prior, H, ℓ, component_scales)
    return prior_accepted, rw_accepted, component_accepted, component_proposed
end

# The coordinate sweep of a ragged layout has one scale per family coordinate,
# shared by every instance of the family (coordinate `r` of family `f` is entry
# `sum(family_sizes[1:f-1]) + r`). Each proposal moves that coordinate in all of
# a subject's instances of the family, among the subjects that have one.
function _saem_mh_component_sweep!(rng, model, subjects, θ, σ,
        prior::RaggedNormalREPrior, H::AbstractVector{<:AbstractVector}, ℓ, scales)
    layout = prior.layout
    offsets = cumsum(vcat(0, layout.family_sizes[1:end-1]))
    accepted = zeros(Int, sum(layout.family_sizes))
    proposed = zeros(Int, length(accepted))
    for (f, width) in enumerate(layout.family_sizes)
        sd = sqrt.(vec(sum(abs2, Matrix(prior.cholesky_factors[f]); dims = 2)))
        members = [k for (k, i) in enumerate(subjects) if f in layout.subject_families[i]]
        isempty(members) && continue
        for r in 1:width
            c = offsets[f] + r
            proposal = map(members) do k
                η = copy(H[k])
                for (family, block) in _ragged_subject_blocks(layout, subjects[k])
                    family == f && (η[block[r]] += scales[c] * sd[r] * randn(rng))
                end
                η
            end
            proposed_ℓ = _proposal_loglikelihoods(model, subjects[members], θ, σ, proposal)
            length(proposed_ℓ) == length(members) &&
                all(x -> !isnan(x) && x != Inf, proposed_ℓ) || throw(DomainError(proposed_ℓ,
                    "coordinate random-walk MH proposal returned invalid densities"))
            for (j, k) in enumerate(members)
                i = subjects[k]
                ratio = proposed_ℓ[j] - ℓ[k] + re_logprior(prior, i, proposal[j]) -
                    re_logprior(prior, i, H[k])
                if log(rand(rng)) < ratio
                    H[k] = proposal[j]
                    ℓ[k] = proposed_ℓ[j]
                    accepted[c] += 1
                end
            end
            proposed[c] += length(members)
        end
    end
    return accepted, proposed
end

# The ragged form of `_saem_gaussian_mstep`, which returns its iterate after 100
# updates and keeps θ along unidentified combinations in the same way. Without
# `require_instances`, an estimated family with no instances among `subjects`
# keeps its covariance `Ω[f]` (a MAP step then moves it by the prior).
function _ragged_gaussian_mstep(layout, θ, Ω, indices, design, mean_phi,
        second_phi, floor, subjects, constraints = nothing; require_instances = true,
        identified = _saem_identified(design, indices))
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
            μ = _saem_gls(A, b, μ, identified)
        end
        sums = [zeros(k, k) for k in layout.family_sizes]
        counts = zeros(Int, length(layout.family_sizes))
        for (k, i) in enumerate(subjects)
            xμ = design[k] * μ
            φ = mean_phi[k]
            for (b, (family, r)) in enumerate(_ragged_subject_blocks(layout, i))
                a = view(xμ, r)
                p = view(φ, r)
                sums[family] += second_phi[k][b] - p * a' - a * p' + a * a'
                counts[family] += 1
            end
        end
        !require_instances ||
            all(f -> counts[f] > 0 || _fully_fixed_family(constraints, f), eachindex(counts)) ||
            throw(ArgumentError(
                "SAEM selected subjects contain no instances of at least one estimated covariance family"))
        nextΩ = [begin
            fam = _omega_family(constraints, f)
            if counts[f] == 0 && !_fully_fixed_family(constraints, f)
                copy(Ωnew[f])
            elseif fam === nothing
                E = eigen(Symmetric(sums[f] / counts[f]))
                Matrix(Symmetric(E.vectors * Diagonal(max.(E.values, floor)) * E.vectors'))
            elseif fam.fully_fixed
                copy(fam.covariance)
            else
                _fixed_family_update(fam, sums[f] / counts[f], Ωnew[f], floor)
            end
        end for f in eachindex(sums)]
        converged = all(f -> maximum(abs, nextΩ[f] - Ωnew[f]) <=
            1e-8 * max(1.0, maximum(abs, Ωnew[f])), eachindex(Ωnew))
        Ωnew = nextΩ
        converged && break
    end
    θnew = copy(θ)
    θnew[indices] = μ
    return θnew, Ωnew
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
        for (b, (family, r)) in enumerate(_ragged_subject_blocks(layout, i))
            a = view(xμ, r)
            p = view(φ, r)
            Xr = view(X, r, :)
            P = precisions[family]
            sums[family] += second_phi[k][b] - p * a' - a * p' + a * a'
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

_saem_omega_above_floor(layout::RaggedNLMELayout, ω, floor) =
    all(L -> _factor_above_floor(L, floor), omega_cholesky(layout, ω))

function _saem_anneal(layout::RaggedNLMELayout, ω_previous, ω, τ, constraints)
    Ω_previous, Ω = omega_matrix(layout, ω_previous), omega_matrix(layout, ω)
    annealed = [_saem_anneal_floor(Ω[f], Ω_previous[f], τ) for f in eachindex(Ω)]
    return _apply_omega_constraints(constraints, omega_parameters(layout, annealed))
end

_saem_closed_form_mstep(layout::RaggedNLMELayout, θ, Ω, indices, design, mean_phi,
    second_phi, floor, subjects, constraints; require_instances = true,
    identified = _saem_identified(design, indices)) =
    _ragged_gaussian_mstep(layout, θ, Ω, indices, design, mean_phi, second_phi,
        floor, subjects, constraints; require_instances, identified)

# The ragged parts of `_saem`, which runs every kernel on both layouts.
function _saem_check_subjects(layout::RaggedNLMELayout, model)
    length(layout.subject_families) == nsubjects(model) || throw(DimensionMismatch(
        "ragged layout has $(length(layout.subject_families)) subjects; model has $(nsubjects(model))"))
    return nothing
end
_saem_start(layout::RaggedNLMELayout, η0, subjects) = _ragged_start(η0, layout, subjects)
# The MH coordinate sweep has one scale per family coordinate.
_saem_ncoordinates(layout::RaggedNLMELayout) = sum(layout.family_sizes)
_saem_result(::RaggedNLMELayout, fields...) = RaggedSAEMResult(fields...)

_saem_mala_cache(H::AbstractVector{<:AbstractVector}) =
    _MALACache(zeros(length(H)), [zeros(length(h)) for h in H], falses(length(H)))

# One standard normal vector per subject, drawn as one vector of all their
# coordinates; for subjects of equal size these are the columns the matrix
# layout draws.
function _saem_mala_noise(rng, H::AbstractVector{<:AbstractVector})
    z = randn(rng, sum(length, H))
    ends = cumsum(length.(H))
    return [z[(e - length(h) + 1):e] for (h, e) in zip(H, ends)]
end

function _saem_mala_refresh_prior!(cache::_MALACache, H::AbstractVector{<:AbstractVector},
        old_prior, new_prior, subjects)
    any(cache.valid) || return nothing
    for (j, i) in enumerate(subjects)
        cache.valid[j] || continue
        old, new = _subject_prior(old_prior, i), _subject_prior(new_prior, i)
        cache.f[j] += re_logprior(new, H[j]) - re_logprior(old, H[j])
        cache.G[j] = cache.G[j] + _precision_times(old, H[j]) - _precision_times(new, H[j])
    end
    return nothing
end

# Without η Hessians the `:hessian` metric is the starting Ω, whose factor is
# the subject's block-diagonal prior factor, never formed as a dense matrix.
_saem_mala_metric(model, i, θ, σ, prior::RaggedNormalREPrior, η, floor) =
    provides_eta_hessian(model) ? cholesky(Symmetric(_saem_hessian_metric(model, i, θ, σ,
        prior, η; curvature_floor = floor))).L : _subject_prior(prior, i)

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
