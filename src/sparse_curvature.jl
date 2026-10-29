# Sparse η Hessians for models that declare which entries can be nonzero
# (`eta_hessian_sparsity`). A subject's conditional Hessian is recovered from one
# pair of batched gradient evaluations per color of a star coloring of its pattern
# (Gebremedhin, Manne & Pothen 2005, via SparseMatrixColorings.jl): perturbing
# every coordinate of one color at once, each nonzero entry is read from a row in
# which no other coordinate of that color appears. Nested random effects need a
# handful of colors however many units a subject has, so the Hessian costs a few
# batched gradient calls and memory proportional to its nonzeros, where dense
# blocks cost one call per coordinate and memory n².

const _STAR_COLORING = (SMC.ColoringProblem(; structure = :symmetric, partition = :column),
    SMC.GreedyColoringAlgorithm(; decompression = :direct))

# A model declares patterns for every subject or for none, so subject 1 answers
# for all.
_declares_eta_hessian_sparsity(model) = eta_hessian_sparsity(model, 1) !== nothing

# Subject i's declared pattern, symmetrized, with the diagonal.
function _eta_hessian_pattern(model, i, n)
    P = eta_hessian_sparsity(model, i)
    P === nothing && throw(ArgumentError("eta_hessian_sparsity(model, $i) is nothing, " *
        "but subject 1 declares a pattern; declare one for every subject or for none"))
    size(P) == (n, n) || throw(DimensionMismatch(
        "eta_hessian_sparsity(model, $i) has size $(size(P)); subject $i has $n random effects"))
    S = sparse(Bool.(P))
    return S .| permutedims(S) .| sparse(I, n, n)
end

# The greedy star coloring takes time quadratic in a subject's effects when one
# effect shares observations with all the others (1.2 s at 35,783 nested units,
# with every ordering SparseMatrixColorings offers), and Newton's iterations, the
# Laplace marginal and the covariance step's re-adaptations ask for the same
# subjects' patterns again and again. The coloring depends only on the pattern,
# so it is cached by the pattern's content: the same colors, so the same Hessians.
# The cache holds at most `_COLORING_CACHE_LIMIT` pattern nonzeros and is emptied
# when a new pattern would exceed that.
const _COLORING_CACHE = Dict{UInt, Vector{Tuple{SparseMatrixCSC, Any}}}()
const _COLORING_CACHE_LOCK = ReentrantLock()
const _COLORING_CACHE_LIMIT = 50_000_000
const _COLORING_CACHE_NNZ = Ref(0)

_same_pattern(P, Q) = size(P) == size(Q) && P.colptr == Q.colptr && P.rowval == Q.rowval

function _eta_hessian_coloring(pattern::SparseMatrixCSC)
    key = hash(pattern.rowval, hash(pattern.colptr, hash(size(pattern))))
    cached = lock(_COLORING_CACHE_LOCK) do
        for (P, coloring) in get(_COLORING_CACHE, key, ())
            _same_pattern(P, pattern) && return coloring
        end
        return nothing
    end
    cached === nothing || return cached
    coloring = SMC.coloring(pattern, _STAR_COLORING...)
    lock(_COLORING_CACHE_LOCK) do
        if _COLORING_CACHE_NNZ[] + SparseArrays.nnz(pattern) > _COLORING_CACHE_LIMIT
            empty!(_COLORING_CACHE)
            _COLORING_CACHE_NNZ[] = 0
        end
        push!(get!(Vector{Tuple{SparseMatrixCSC, Any}}, _COLORING_CACHE, key),
            (copy(pattern), coloring))
        _COLORING_CACHE_NNZ[] += SparseArrays.nnz(pattern)
    end
    return coloring
end

# Central differences of the batched conditional η gradients along one direction
# per subject, (g(η + hd) - g(η - hd)) / 2h ≈ ∇²ℓ d, with h = cbrt(eps)·max(1, ‖η‖∞)
# / ‖d‖∞, so that no coordinate moves by more than cbrt(eps)·max(1, ‖η‖∞).
function _gradient_differences(model, layout, subjects, θ, σ, points, directions)
    steps = [cbrt(eps(eltype(η))) * max(one(eltype(η)), maximum(abs, η; init = zero(eltype(η)))) /
        maximum(abs, d) for (η, d) in zip(points, directions)]
    shifted(sign) = _eta_batch(layout, [η .+ (sign * h) .* d
        for (η, d, h) in zip(points, directions, steps)])
    _, up = conditional_loglikelihoods_and_gradients(model, subjects, θ, σ, shifted(1))
    _, down = conditional_loglikelihoods_and_gradients(model, subjects, θ, σ, shifted(-1))
    return [(_eta_column(up, j) - _eta_column(down, j)) / (2 * steps[j]) for j in eachindex(subjects)]
end

# The conditional η Hessians from their colorings: column c of a subject's
# compressed Hessian is the difference along the indicator of its color c.
function _colored_differences(model, layout, subjects, θ, σ, points, colorings)
    counts = SMC.ncolors.(colorings)
    compressed = [zeros(eltype(η), length(η), c) for (η, c) in zip(points, counts)]
    for c in 1:maximum(counts; init = 0)
        ks = findall(>=(c), counts)
        columns = _gradient_differences(model, layout, subjects[ks], θ, σ, points[ks],
            [eltype(points[k]).(SMC.column_colors(colorings[k]) .== c) for k in ks])
        for (j, k) in enumerate(ks)
            compressed[k][:, c] = columns[j]
        end
    end
    return [SMC.decompress(B, r) for (B, r) in zip(compressed, colorings)]
end

# The conditional η Hessians of `subjects` at their points `H`. If the batch
# throws `_InfeasibleEvaluation` (a step left the model's domain), each subject is
# differenced alone, and one whose steps still throw gets `nothing`.
function _colored_conditional_hessians(model, subjects, θ, σ, H, colorings)
    layout = nlme_layout(model)
    points = [Vector(_column(H, k)) for k in eachindex(subjects)]
    hessians(ks) = _colored_differences(model, layout, subjects[ks], θ, σ, points[ks], colorings[ks])
    batch = _feasible(hessians, eachindex(subjects))
    batch === nothing || return batch
    return [_feasible(k -> only(hessians([k])), k) for k in eachindex(subjects)]
end

# The complete-data η Hessians from the declared patterns: the colored conditional
# Hessians minus the prior precision, block by block (`nothing` where the
# differences left the model's domain).
function _sparse_complete_hessians(model, subjects, θ, σ, prior, H)
    colorings = [_eta_hessian_coloring(_eta_hessian_pattern(model, i, length(_column(H, k))))
        for (k, i) in enumerate(subjects)]
    conditional = _colored_conditional_hessians(model, subjects, θ, σ, H, colorings)
    return [C === nothing ? nothing : C - _precision_matrix(_subject_prior(prior, i))
        for (C, i) in zip(conditional, subjects)]
end

# The complete-data η Hessians that Newton steps on and a result reports: sparse
# from the declared patterns, else the model's own blocks.
function _complete_hessians(model, subjects, θ, σ, prior, H)
    _declares_eta_hessian_sparsity(model) ||
        return _model_complete_hessians(model, subjects, θ, σ, prior, H)
    hessians = _sparse_complete_hessians(model, subjects, θ, σ, prior, H)
    outside = findall(isnothing, hessians)
    isempty(outside) || error("the central differences for the sparse η Hessian of subject(s) " *
        "$(subjects[outside]) leave the model's domain, so their curvature is unknown there")
    return identity.(hessians)
end

# Whether the solver's result carries Hessians: the model's blocks, or sparse ones
# from its patterns.
_reports_eta_hessians(model) = provides_eta_hessian(model) || _declares_eta_hessian_sparsity(model)

# ── The trust-region subproblem on a sparse Hessian ──────────────────────────────
#
# Maximize gᵀp + pᵀHp/2 over pᵀMp ≤ Δ², with M = Ω⁻¹ the prior precision (the dense
# method's ball ‖L⁻¹p‖ ≤ Δ), by sparse Cholesky factorizations of A(ν) = -H + νM in
# place of an eigendecomposition (Moré & Sorensen 1983, in the metric M). M is block
# diagonal, so A(ν) keeps H's sparsity. The maximizer is p(ν) = A(ν)⁻¹g for the
# smallest ν ≥ 0 with A(ν) positive definite and ‖p(ν)‖_M ≤ Δ, on the boundary
# unless ν = 0. In ν, ψ(ν) = 1/‖p(ν)‖_M - 1/Δ is concave and increasing, so Newton's
# method from a ν left of the root rises monotonically to it.
function _trust_region_trial(Hk::SparseMatrixCSC, prior, g, Δ)
    g = collect(g)
    M = _precision_matrix(prior)
    factor(ν) = cholesky(Symmetric(iszero(ν) ? -Hk : ν * M - Hk); check = false)
    F = factor(zero(Δ))
    if issuccess(F)
        p = F \ g
        _metric_norm(M, p) <= Δ && return _sparse_trial(Hk, M, g, p, Δ, true)
        return _boundary_trial(Hk, M, g, Δ, zero(Δ), F, p, factor)
    end
    # -H is not positive definite: bracket the smallest positive-definite shift.
    ν_bad, ν = zero(Δ), one(Δ)
    F = factor(ν)
    while !issuccess(F)
        ν_bad, ν = ν, 4ν
        F = factor(ν)
    end
    p = F \ g
    _metric_norm(M, p) >= Δ && return _boundary_trial(Hk, M, g, Δ, ν, F, p, factor)
    # The root lies between the bracket's ends, or there is none (the hard case).
    while ν - ν_bad > 1.0e-12 * ν
        mid = (ν_bad + ν) / 2
        Fm = factor(mid)
        if !issuccess(Fm)
            ν_bad = mid
            continue
        end
        pm = Fm \ g
        _metric_norm(M, pm) >= Δ && return _boundary_trial(Hk, M, g, Δ, mid, Fm, pm, factor)
        ν, F, p = mid, Fm, pm
    end
    # The hard case (for instance a start exactly at a saddle point): at the
    # smallest positive-definite shift p(ν) stays inside the region, and the
    # remaining length goes along the eigenvector of (-H, M) with the smallest
    # eigenvalue, from inverse iteration with the nearly singular factor.
    z = _inverse_iteration(F, M, length(g))
    a, b, c = one(Δ), 2 * dot(p, M * z), _metric_norm(M, p)^2 - Δ^2
    root = sqrt(max(b^2 - 4a * c, zero(Δ)))
    steps = (p + ((-b - root) / 2a) * z, p + ((-b + root) / 2a) * z)
    step = argmax(s -> _model_increase(Hk, g, s), steps)
    return _sparse_trial(Hk, M, g, step, Δ, false)
end

_metric_norm(M, p) = sqrt(max(dot(p, M * p), zero(eltype(p))))
_model_increase(Hk, g, p) = dot(g, p) + dot(p, Hk * p) / 2

# Newton's method on ψ(ν) from (ν, p(ν)) left of the root: with w = A(ν)⁻¹Mp,
# d‖p‖²_M/dν = -2pᵀMw, so ν ← ν + ‖p‖²_M (‖p‖_M - Δ) / (Δ pᵀMw).
function _boundary_trial(Hk, M, g, Δ, ν, F, p, factor)
    for _ in 1:100
        φ = _metric_norm(M, p)
        φ - Δ <= 1.0e-12 * Δ && break
        Mp = M * p
        next = ν + φ^2 * (φ - Δ) / (Δ * dot(Mp, F \ Mp))
        next <= ν && break
        Fn = factor(next)
        issuccess(Fn) || break
        ν, F, p = next, Fn, Fn \ g
    end
    return _sparse_trial(Hk, M, g, p, Δ, false)
end

# The eigenvector of the pencil (A, M) with the eigenvalue nearest zero, M-normalized.
function _inverse_iteration(F, M, n)
    z = randn(Xoshiro(0), n)
    for _ in 1:20
        z = F \ (M * z)
        z ./= _metric_norm(M, z)
    end
    return z
end

_sparse_trial(Hk, M, g, p, Δ, full) =
    TrustRegionTrial(p, _model_increase(Hk, g, p), full, Δ, _metric_norm(M, p))
