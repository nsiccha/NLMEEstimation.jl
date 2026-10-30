# Families a model integrates out by its own Laplace approximation
# (`integrated_families`). SAEM's chains sample only the other entries of each
# subject's η, against the Laplace conditional of the integrated ones (an
# `_EntriesView` of `_IntegratedModel`); the M-step takes the integrated
# families' covariances from the Laplace moments `E[u uᵀ] ≈ û ûᵀ + C` and moves
# θ and σ on the Laplace conditionals (`_IntegratedModel`). With every family
# integrated nothing is sampled, and the iterations are Laplace-EM.

function _checked_integrated(layout, integrated)
    families = sort!(collect(Int, integrated))
    allunique(families) && all(in(_families(layout)), families) || throw(ArgumentError(
        "integrated must list distinct family indices of the layout " *
        "(1:$(length(_families(layout)))), got $(collect(integrated))"))
    return families
end

# The blocks of subject i's η in η order, as `family => range`; the blocks of an
# `NLMELayout` are its families, once each.
_subject_blocks(layout::NLMELayout, i) =
    [b => r for (b, r) in enumerate(first(_block_ranges(layout)))]
_subject_blocks(layout::RaggedNLMELayout, i) = _ragged_subject_blocks(layout, i)

_family_omega_ranges(layout::NLMELayout) = last(_block_ranges(layout))
_family_omega_ranges(layout::RaggedNLMELayout) = _ragged_family_ranges(layout)

# The part of `layout` made of `families`, for the subjects in `subjects` with an
# instance of one: its own layout (families renumbered in order, the ragged one
# over the covered subjects only), the positions in `subjects` of the subjects
# it covers, each covered subject's entries of its full η, and the ω coordinates
# of its families. `layout === nothing` when no subject has such an entry.
function _restriction(layout::NLMELayout, families, subjects)
    isempty(families) && return (layout = nothing, positions = Int[],
        slots = Vector{Int}[], omega = Int[])
    ranges = first(_block_ranges(layout))
    slots = reduce(vcat, [collect(ranges[f]) for f in families])
    return (layout = NLMELayout(layout.ntheta, layout.nsigma, layout.eta_blocks[families]),
        positions = collect(eachindex(subjects)), slots = [slots for _ in subjects],
        omega = reduce(vcat, [collect(r) for r in _family_omega_ranges(layout)[families]]))
end
function _restriction(layout::RaggedNLMELayout, families, subjects)
    has(i) = any(in(families), layout.subject_families[i])
    positions = [k for (k, i) in enumerate(subjects) if has(i)]
    isempty(positions) && return (layout = nothing, positions = positions,
        slots = Vector{Int}[], omega = Int[])
    present = sort!(unique(f for k in positions
        for f in layout.subject_families[subjects[k]] if f in families))
    number = Dict(f => j for (j, f) in enumerate(present))
    restricted = RaggedNLMELayout(layout.ntheta, layout.nsigma, layout.family_sizes[present],
        [[number[f] for f in layout.subject_families[subjects[k]] if f in families]
            for k in positions])
    slots = [reduce(vcat, [collect(r) for (f, r) in _subject_blocks(layout, subjects[k])
        if f in families]) for k in positions]
    return (layout = restricted, positions = positions, slots = slots,
        omega = reduce(vcat, [collect(r) for r in _family_omega_ranges(layout)[present]]))
end

# How an SAEM fit splits each subject's η: `sampled` is the restriction to the
# families it samples (its positions are the subjects with a chain), and
# `blocks[k]` / `ranges[k]` the indices and η ranges, among the blocks of the
# subject at position k, of its integrated instances.
struct _IntegrationPlan{S}
    integrated::Vector{Int}
    sampled::S
    blocks::Vector{Vector{Int}}
    ranges::Vector{Vector{UnitRange{Int}}}
end

function _integration_plan(layout, integrated, subjects)
    sampled = _restriction(layout, setdiff(_families(layout), integrated), subjects)
    instances = [[(b, r) for (b, (f, r)) in enumerate(_subject_blocks(layout, i))
        if f in integrated] for i in subjects]
    return _IntegrationPlan(integrated, sampled, [first.(x) for x in instances],
        [last.(x) for x in instances])
end

_nchains(plan::_IntegrationPlan) = length(plan.sampled.positions)

# The sampled entries of the chains' subjects, and the reverse: write them back.
_sampled_batch(H::AbstractMatrix, plan) = _nchains(plan) == 0 ? H[Int[], Int[]] :
    H[only(unique(plan.sampled.slots)), plan.sampled.positions]
_sampled_batch(H::AbstractVector{<:AbstractVector}, plan) =
    [H[k][s] for (k, s) in zip(plan.sampled.positions, plan.sampled.slots)]
function _write_sampled!(H::AbstractMatrix, plan, S)
    _nchains(plan) == 0 && return H
    H[only(unique(plan.sampled.slots)), plan.sampled.positions] = S
    return H
end
function _write_sampled!(H::AbstractVector{<:AbstractVector}, plan, S)
    for (j, (k, s)) in enumerate(zip(plan.sampled.positions, plan.sampled.slots))
        h = copy(H[k])
        h[s] = S[j]
        H[k] = h
    end
    return H
end

# Subject k's full η with the entries `slots` set to `x` in `base`.
_merged(base, slots, x) = (h = Vector{Float64}(base); h[slots] = x; h)

# The model `inner` seen on only some entries of each subject's η (`slots[j]` of
# subject `subjects[j]`, with layout `layout`), the others held at `base`: a
# batched model whose subject j is `inner`'s subject `subjects[j]`. SAEM's kernels
# run on it with `inner` the Laplace conditionals of a model's integrated
# families, which leaves only the sampled entries; their `base` entries start
# each mode search and stay fixed between M-steps, so the target of a transition
# does not depend on the order of its evaluations. `check_protocol` runs the
# package's own Laplace approximation on it with `inner` the model itself and only
# the integrated entries, the others given.
struct _EntriesView{M, L, B}
    inner::M
    layout::L
    subjects::Vector{Int}
    slots::Vector{Vector{Int}}
    base::B
end

nsubjects(v::_EntriesView) = length(v.subjects)
nlme_layout(v::_EntriesView) = v.layout
evaluation_style(::_EntriesView) = BatchedEvaluation()

function _full_batch(v::_EntriesView, js, X)
    return _from_columns(v.base, [_merged(_column(v.base, j), v.slots[j], _column(X, c))
        for (c, j) in enumerate(js)])
end

conditional_loglikelihoods(v::_EntriesView, js, θ, σ, X) = (js = collect(Int, js);
    conditional_loglikelihoods(v.inner, v.subjects[js], θ, σ, _full_batch(v, js, X)))

function conditional_loglikelihoods_and_gradients(v::_EntriesView, js, θ, σ, X)
    js = collect(Int, js)
    ℓ, G = conditional_loglikelihoods_and_gradients(v.inner, v.subjects[js], θ, σ,
        _full_batch(v, js, X))
    length(ℓ) == _nchains(G) == length(js) || throw(DimensionMismatch(
        "$(length(ℓ)) values and $(_nchains(G)) gradients for $(length(js)) subjects"))
    return ℓ, _from_columns(X, [_column(G, c)[v.slots[j]] for (c, j) in enumerate(js)])
end

# A model's Laplace conditionals at ω as a model of its own, on the full layout:
# its conditional log-likelihood is the Laplace conditional of the integrated
# families given the other entries of η, and its population gradient the
# conditional hook's. SAEM's kernels see it through an `_EntriesView` of the
# sampled entries, its numerical M-step directly, at the M-step's new
# covariances (`_saem_numeric_model`).
struct _IntegratedModel{M}
    model::M
    integrated::Vector{Int}
    ω::Vector{Float64}
end

nsubjects(m::_IntegratedModel) = nsubjects(m.model)
nlme_layout(m::_IntegratedModel) = nlme_layout(m.model)
evaluation_style(::_IntegratedModel) = BatchedEvaluation()
saem_population_gradient_style(::_IntegratedModel) = BatchedEvaluation()
saem_population_gradient_style(::_IntegratedModel, ::Val) = BatchedEvaluation()
population_logprior(m::_IntegratedModel, θ, σ, ω) = population_logprior(m.model, θ, σ, ω)
population_logprior_and_gradient(m::_IntegratedModel, θ, σ, ω) =
    population_logprior_and_gradient(m.model, θ, σ, ω)

conditional_loglikelihoods(m::_IntegratedModel, subjects, θ, σ, H) =
    laplace_conditionals(m.model, m.integrated, subjects, θ, σ, m.ω, H)
conditional_loglikelihoods_and_gradients(m::_IntegratedModel, subjects, θ, σ, H) =
    laplace_conditionals_and_gradients(m.model, m.integrated, subjects, θ, σ, m.ω, H)

function _saem_population_value_gradient(::BatchedEvaluation, m::_IntegratedModel,
        subjects, θ, σ, H)
    ℓ, gθ, gσ, _ = _laplace_population_gradient(m.model, m.integrated, subjects,
        θ, σ, m.ω, H)
    _check_length("the Laplace conditionals' gradient in θ", gθ, length(θ))
    _check_length("the Laplace conditionals' gradient in σ", gσ, length(σ))
    return sum(ℓ), vcat(gθ, gσ)
end

# With every family integrated the conditionals are the marginals, which a model
# may supply through the covariance step's hook alone; that hook's default is the
# conditional one, so each method reaches the other without a cycle.
function _laplace_population_gradient(model, integrated, subjects, θ, σ, ω, H)
    integrated == _families(nlme_layout(model)) &&
        return laplace_marginals_and_gradient(model, subjects, θ, σ, ω, H)
    return laplace_conditionals_and_population_gradient(model, integrated, subjects,
        θ, σ, ω, H)
end

_saem_numeric_model(model, ω) = model
_saem_numeric_model(m::_IntegratedModel, ω) =
    _IntegratedModel(m.model, m.integrated, collect(Float64, ω))

# The Laplace moments of the subjects at `positions`: their integrated entries of
# `H` move to the modes, `covariances[k]` takes their covariance blocks and
# `values[k]` their Laplace conditionals.
function _saem_laplace_moments!(H, covariances, values, model, plan, ids, positions,
        θ, σ, ω)
    positions = collect(Int, positions)
    ℓ, M, C = laplace_conditional_moments(model, plan.integrated, ids[positions], θ, σ, ω,
        _columns(H, positions))
    length(ℓ) == _nchains(M) == length(C) == length(positions) || throw(DimensionMismatch(
        "laplace_conditional_moments returned $(length(ℓ)) values, $(_nchains(M)) modes " *
        "and $(length(C)) covariance lists for $(length(positions)) subjects"))
    all(isfinite, ℓ) || throw(DomainError(ℓ,
        "SAEM Laplace conditional log densities must be finite"))
    for (c, k) in enumerate(positions)
        blocks = C[c]
        length(blocks) == length(plan.ranges[k]) || throw(DimensionMismatch(
            "subject $(ids[k]): laplace_conditional_moments returned $(length(blocks)) " *
            "covariance blocks for $(length(plan.ranges[k])) integrated instances"))
        for (B, r) in zip(blocks, plan.ranges[k])
            size(B) == (length(r), length(r)) || throw(DimensionMismatch(
                "subject $(ids[k]): an integrated instance's covariance block has size " *
                "$(size(B)); expected $((length(r), length(r)))"))
        end
        _check_length("subject $(ids[k]) η", _column(M, c), length(_column(H, k)))
        covariances[k] = [Matrix{Float64}(B) for B in blocks]
        values[k] = ℓ[c]
    end
    _set_columns!(H, positions, M)
    return H
end

# Add the integrated instances' Laplace covariances, weighted by the SA step γ,
# to the running second moments `_saem_moments!` has just updated.
function _saem_add_covariances!(second_phi::AbstractArray{<:Real, 3}, plan, covariances, γ)
    for k in axes(second_phi, 3), (B, r) in zip(covariances[k], plan.ranges[k])
        second_phi[r, r, k] .+= γ .* B
    end
    return second_phi
end
function _saem_add_covariances!(second_phi::AbstractVector, plan, covariances, γ)
    for k in eachindex(second_phi), (B, b) in zip(covariances[k], plan.blocks[k])
        second_phi[k][b] = second_phi[k][b] + γ * B
    end
    return second_phi
end

# The kernels' model, chain subjects, prior and chain states (`_saem`).
_saem_estep(model, ::Nothing, ids, layout, ω, H) = (model, ids, _re_prior(layout, ω), H)
function _saem_estep(model, plan::_IntegrationPlan, ids, layout, ω, H)
    _nchains(plan) == 0 && return (model, Int[], nothing, H)
    positions = plan.sampled.positions
    view = _EntriesView(_IntegratedModel(model, plan.integrated, collect(Float64, ω)),
        plan.sampled.layout, ids[positions], plan.sampled.slots, _columns(H, positions))
    return view, collect(1:_nchains(plan)),
        _re_prior(plan.sampled.layout, ω[plan.sampled.omega]), _sampled_batch(H, plan)
end

# The chains among the positions `active` of an SA iteration.
_saem_chain_active(::Nothing, active, n) = active
_saem_chain_active(plan::_IntegrationPlan, ::Base.OneTo, n) = Base.OneTo(n)
_saem_chain_active(plan::_IntegrationPlan, active, n) =
    [j for (j, k) in enumerate(plan.sampled.positions) if insorted(k, active)]

# The package's own Laplace approximation of the entries of the families
# `families` given the other entries of `H`: per subject its value, its full η
# at the mode, and the covariance blocks of its integrated instances. It is
# `check_protocol`'s reference on small subjects: a gradient-only search, and each
# covariance from a dense difference Hessian.
function _conditioned_laplace(model, families, ids, θ, σ, ω, H)
    r = _restriction(nlme_layout(model), families, ids)
    values = Float64.(conditional_loglikelihoods(model, ids, θ, σ, H))
    modes = [Vector{Float64}(_column(H, k)) for k in eachindex(ids)]
    covariances = [Matrix{Float64}[] for _ in ids]
    r.layout === nothing && return values, modes, covariances
    view = _EntriesView(model, r.layout, ids[r.positions], r.slots, _columns(H, r.positions))
    ωr = ω[r.omega]
    start = _eta_batch(r.layout, [modes[k][s] for (k, s) in zip(r.positions, r.slots)])
    ebe = empirical_bayes(view, θ, σ, ωr; η0 = start)
    laplace = marginal_loglikelihood(view, θ, σ, ωr; η0 = ebe,
        approximation = AdaptiveGaussHermite(nodes = 1))
    at = [Vector{Float64}(_eta_column(ebe.modes, j)) for j in eachindex(r.positions)]
    hessians = _difference_eta_hessians(view, r.layout, collect(eachindex(r.positions)),
        θ, σ, re_prior(view, ωr), at)
    for (j, k) in enumerate(r.positions)
        values[k] = laplace.subject_loglikelihoods[j]
        modes[k][r.slots[j]] = at[j]
        Σ = inv(Symmetric(-hessians[j]))
        covariances[k] = [Matrix(Σ[b, b]) for (_, b) in _subject_blocks(r.layout, j)]
    end
    return values, modes, covariances
end

# `check_protocol`'s checks of the methods `integrated_families` asks for.
function _check_integrated(violation, model, integrated, subjects, θ, σ, ω, H, rtol)
    isempty(integrated) && return nothing
    layout = nlme_layout(model)
    families = try
        _checked_integrated(layout, integrated)
    catch err
        err isa ArgumentError || rethrow()
        violation(sprint(showerror, err))
    end
    ids = collect(Int, subjects)
    plan = _integration_plan(layout, families, ids)
    tol = max(rtol, 1.0e-6)
    close(a, b) = isapprox(a, b; rtol = tol, atol = tol)
    ℓ, G = laplace_conditionals_and_gradients(model, families, ids, θ, σ, ω, H)
    length(ℓ) == _nchains(G) == length(ids) || violation(
        "laplace_conditionals_and_gradients returned $(length(ℓ)) values and " *
        "$(_nchains(G)) gradients for $(length(ids)) subjects")
    all(x -> x isa Real && isfinite(x), ℓ) || violation(
        "laplace_conditionals_and_gradients returned the values $ℓ; expected finite reals")
    values = laplace_conditionals(model, families, ids, θ, σ, ω, H)
    close(values, ℓ) || violation(
        "laplace_conditionals values $values ≠ laplace_conditionals_and_gradients values $ℓ")
    ℓm, M, C = laplace_conditional_moments(model, families, ids, θ, σ, ω, H)
    close(ℓm, ℓ) || violation(
        "laplace_conditional_moments values $ℓm ≠ laplace_conditionals_and_gradients values $ℓ")
    _nchains(M) == length(C) == length(ids) || violation(
        "laplace_conditional_moments returned $(_nchains(M)) modes and $(length(C)) " *
        "covariance lists for $(length(ids)) subjects")
    reference = _conditioned_laplace(model, families, ids, θ, σ, ω, M)
    directions = Vector{Vector{Float64}}(undef, length(ids))
    rng = Xoshiro(0)
    for (k, i) in enumerate(ids)
        h, g, m = _column(H, k), _column(G, k), _column(M, k)
        integrated_entries = reduce(vcat, collect.(plan.ranges[k]); init = Int[])
        sampled = setdiff(eachindex(h), integrated_entries)
        length(g) == length(h) || violation("subject $i: laplace_conditionals_and_gradients " *
            "returned a gradient of length $(length(g)); expected $(length(h))")
        all(iszero, g[integrated_entries]) || violation("subject $i: " *
            "laplace_conditionals_and_gradients is nonzero at integrated entries " *
            "$(integrated_entries[findall(!iszero, g[integrated_entries])]), which only " *
            "start the mode search")
        length(m) == length(h) && m[sampled] == h[sampled] || violation("subject $i: " *
            "laplace_conditional_moments changed sampled entries of η")
        length(C[k]) == length(plan.ranges[k]) || violation("subject $i: " *
            "laplace_conditional_moments returned $(length(C[k])) covariance blocks for " *
            "$(length(plan.ranges[k])) integrated instances")
        for (B, R, r) in zip(C[k], reference[3][k], plan.ranges[k])
            size(B) == (length(r), length(r)) || violation("subject $i: an integrated " *
                "instance's covariance block has size $(size(B)); expected $((length(r), length(r)))")
            isapprox(B, R; rtol = max(rtol, 1.0e-5), atol = max(rtol, 1.0e-5)) ||
                violation("subject $i: laplace_conditional_moments covariance block $B " *
                    "differs from the package's Laplace approximation $R")
        end
        close(ℓ[k], reference[1][k]) || violation("subject $i: " *
            "laplace_conditionals_and_gradients value $(ℓ[k]) differs from the package's " *
            "Laplace approximation $(reference[1][k])")
        isapprox(m, reference[2][k]; rtol = max(rtol, 1.0e-5), atol = max(rtol, 1.0e-5)) ||
            violation("subject $i: laplace_conditional_moments modes $(m[integrated_entries]) " *
                "differ from the package's $(reference[2][k][integrated_entries])")
        d = zeros(length(h))
        d[sampled] = randn(rng, length(sampled))
        directions[k] = d
    end
    step = 1.0e-4
    shifted(t) = _from_columns(M, [_column(M, k) .+ (t * step) .* directions[k]
        for k in eachindex(ids)])
    summed(t) = sum(laplace_conditionals(model, families, ids, θ, σ, ω, shifted(t)))
    tol = max(rtol, 1.0e-5)
    difference = (summed(1) - summed(-1)) / 2step
    directional = sum(k -> dot(_column(G, k), directions[k]), eachindex(ids))
    isapprox(directional, difference; rtol = tol, atol = tol) || violation(
        "laplace_conditionals_and_gradients' gradient along a random direction of the " *
        "sampled entries is $directional, but a central difference of its values gives $difference")
    mr = mu_referencing(model)
    mu = mr === nothing ? Int[] : mr.theta_indices
    layout.nsigma > 0 || length(mu) < layout.ntheta || return nothing
    ℓp, gθ, gσ, gω = _laplace_population_gradient(model, families, ids, θ, σ, ω, M)
    close(ℓp, ℓ) || violation(
        "laplace_conditionals_and_population_gradient values $ℓp ≠ laplace_conditionals values $ℓ")
    map(length, (gθ, gσ, gω)) == map(length, (θ, σ, ω)) || violation(
        "laplace_conditionals_and_population_gradient returned gradients of lengths " *
        "$(map(length, (gθ, gσ, gω))); expected $(map(length, (θ, σ, ω)))")
    dθ, dσ, dω = randn(rng, length(θ)), randn(rng, length(σ)), randn(rng, length(ω))
    population(t) = sum(laplace_conditionals(model, families, ids, θ .+ (t * step) .* dθ,
        σ .+ (t * step) .* dσ, ω .+ (t * step) .* dω, M))
    difference = (population(1) - population(-1)) / 2step
    directional = dot(gθ, dθ) + dot(gσ, dσ) + dot(gω, dω)
    isapprox(directional, difference; rtol = tol, atol = tol) || violation(
        "laplace_conditionals_and_population_gradient's gradient along a random direction " *
        "is $directional, but a central difference of its values gives $difference")
    return nothing
end
