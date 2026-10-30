# The curvature check of `QuasiNewtonEBE` for a model with gradients only and no
# declared pattern, from a bounded number of Hessian-vector products: a dense block
# would cost one batched gradient call per coordinate and memory n². Each product
# is a forward difference of the batched complete-data gradients, so subjects share
# their calls and only the current vectors are kept, O(steps · n) numbers per
# subject during a check.
#
# The operator is the negative complete-data Hessian in the standardized coordinates
# L⁻¹η, A = -LᵀHL, whose eigenvalues are 1 where the data say nothing. Lanczos finds
# its lowest eigenvalue in about √(spread of the spectrum) · log n products, which is
# hundreds where nested units carry information from 0.1 to 10⁴ prior precisions (a
# single stiff effect costs a step or two, not the spread). So it runs on
# B = S⁻¹AS⁻¹, with S² a diagonal estimated from `_DIAGONAL_PROBES` random sign
# vectors, E[z ⊙ Az] = diag(A) (Bekas, Kokiopoulou & Saad 2007). B is congruent to
# A, so it has the same number of negative eigenvalues (Sylvester), and the spread
# of the units' information is gone from it. The estimate's error is a unit's
# coupling to the effects above it, as large as its information in a nested model,
# over √probes: on 36,000 units with a saddle hidden from the diagonal, 16 probes found
# it within 14 steps and 4 probes (or none) not within 150; the exact diagonal, in 3.
#
# A negative Ritz value proves upward curvature, along the Ritz vector. Its absence
# proves concavity only when the Krylov space fills up, which it does for a subject
# of at most `steps + _DIAGONAL_PROBES` effects: there the check is as exact as a
# dense block and takes at most `n` calls, with no probes. Above that, curvature
# that `steps` products did not reach goes unseen.

const _DIAGONAL_PROBES = 16

# The two findings of a check; `nothing` in `BFGSState.hessians` means no check, or
# that its steps left the model's domain, so the curvature is unknown.
struct _NoUpwardCurvature end

# `direction`, of unit length in L⁻¹η, and the density's curvature `direction'H*direction > 0`.
struct _UpwardCurvature{T}
    direction::Vector{T}
    curvature::T
end

# The complete-data Hessian times one direction per subject,
# (G(η + hd) - G(η)) / h, moving no coordinate by more than 1e-4 · max(1, ‖η‖∞) (the
# dense blocks' step). If the batch throws `_InfeasibleEvaluation`, each subject is
# differenced alone, and one whose step still throws gets `nothing`.
function _hessian_times(model, subjects, θ, σ, prior, points, gradients, directions)
    layout = nlme_layout(model)
    function products(ks)
        steps = [1.0e-4 * max(one(eltype(η)), maximum(abs, η; init = zero(eltype(η)))) / maximum(abs, d)
            for (η, d) in zip(points[ks], directions[ks])]
        batch = _eta_batch(layout, [η .+ h .* d for (η, d, h) in zip(points[ks], directions[ks], steps)])
        _, _, G = _complete_gradients(model, subjects[ks], θ, σ, prior, batch)
        return [(_eta_column(G, j) - gradients[k]) / steps[j] for (j, k) in enumerate(ks)]
    end
    batch = _feasible(products, eachindex(subjects))
    batch === nothing || return batch
    return [_feasible(k -> only(products([k])), k) for k in eachindex(subjects)]
end

# One subject's Lanczos run on B: `basis` holds the vectors so far, `α` and `β` the
# tridiagonal matrix, and `q` the vector to multiply next.
struct _LanczosRun{T, P}
    prior::P
    scale::Vector{T}
    basis::Vector{Vector{T}}
    α::Vector{T}
    β::Vector{T}
    q::Vector{T}
    limit::Int
end

function _LanczosRun(prior, scale::Vector{T}, limit) where {T}
    q = randn(Xoshiro(0), T, length(scale))
    return _LanczosRun(prior, scale, Vector{T}[], T[], T[], q ./ norm(q), limit)
end

# Takes w = Bq, with full reorthogonalization, O(j · n) for step j. Returns the
# finding, or `nothing` to go on.
function _lanczos_step!(run::_LanczosRun{T}, w) where {T}
    q = copy(run.q)
    push!(run.basis, q)
    push!(run.α, dot(q, w))
    for _ in 1:2, b in run.basis
        axpy!(-dot(b, w), b, w)
    end
    β = norm(w)
    ritz = eigen(SymTridiagonal(run.α, run.β))
    if first(ritz.values) < 0
        y = sum(c * b for (c, b) in zip(view(ritz.vectors, :, 1), run.basis)) ./ run.scale
        len = norm(y)
        return _UpwardCurvature(_factor_times(run.prior, y ./ len), -first(ritz.values) / len^2)
    end
    (length(run.α) >= run.limit || β <= sqrt(eps(T)) * maximum(abs, ritz.values)) &&
        return _NoUpwardCurvature()
    push!(run.β, β)
    run.q .= w ./ β
    return nothing
end

# The findings of the check at `points`, with complete-data `gradients` there: at
# most `steps + _DIAGONAL_PROBES` batched gradient calls.
function _lanczos_curvature(model, subjects, θ, σ, prior, points, gradients, steps)
    T = eltype(first(points))
    priors = [_subject_prior(prior, i) for i in subjects]
    A_times(ks, vectors) = map(_hessian_times(model, subjects[ks], θ, σ, prior, points[ks], gradients[ks],
            [_factor_times(priors[k], v) for (k, v) in zip(ks, vectors)]), ks) do Hd, k
        Hd === nothing ? nothing : -_factor_transpose_times(priors[k], Hd)
    end
    findings = Vector{Any}(nothing, length(subjects))
    scales = [ones(T, length(η)) for η in points]
    active = collect(eachindex(subjects))
    large = filter(k -> length(points[k]) > steps + _DIAGONAL_PROBES, active)
    diagonals = Dict(k => zeros(T, length(points[k])) for k in large)
    rngs = Dict(k => Xoshiro(1) for k in large)
    for _ in 1:_DIAGONAL_PROBES
        isempty(large) && break
        signs = [rand(rngs[k], (-one(T), one(T)), length(points[k])) for k in large]
        products = A_times(large, signs)
        unknown = findall(isnothing, products)
        for (j, k) in enumerate(large)
            products[j] === nothing || (diagonals[k] .+= signs[j] .* products[j])
        end
        setdiff!(active, large[unknown])
        deleteat!(large, unknown)
    end
    for k in large
        scales[k] = sqrt.(max.(abs.(diagonals[k]) ./ _DIAGONAL_PROBES, one(T)))
    end
    runs = Dict(k => _LanczosRun(priors[k], scales[k],
        length(points[k]) > steps + _DIAGONAL_PROBES ? steps : length(points[k])) for k in active)
    while !isempty(active)
        products = A_times(active, [runs[k].q ./ runs[k].scale for k in active])
        for (j, k) in enumerate(copy(active))
            findings[k] = products[j] === nothing ? nothing :
                _lanczos_step!(runs[k], products[j] ./ runs[k].scale)
            (products[j] === nothing || findings[k] !== nothing) && filter!(!=(k), active)
        end
    end
    return findings
end
