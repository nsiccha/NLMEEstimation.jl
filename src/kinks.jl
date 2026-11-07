# ── Modes at kinks ────────────────────────────────────────────────────────────
#
# A complete-data density that is only piecewise smooth (a running minimum, ordered
# thresholds taken by `min`, a clamp) can have its maximum on a kink, where the
# gradient does not vanish. Near such a point the density is the minimum of smooth
# pieces f_i, and the smooth solvers stall there: every trial step that their local
# model predicts to climb crosses the kink, where the density drops, until none is
# accepted. The subjects for which that happens enter this phase.
#
# **Search.** Gradient sampling with a variable metric (Burke, Lewis & Overton 2005;
# Kiwiel 2007; Curtis & Que 2015). Around the current point, gradients are sampled
# uniformly in a ball of radius ε prior standard deviations (in L⁻¹η, Ω = LLᵀ). The
# step is d = W z, where z is the element of the convex hull of the sampled gradients
# nearest to zero in the metric W, the method's inverse curvature: (-H)⁻¹ from
# Newton's Hessian at the current point, or BFGS's limited-memory approximation. For
# two pieces with gradients g₁, g₂ that z satisfies (g₁ - g₂)ᵀWz = 0, so the step
# follows the linearized kink instead of crossing it, and where every sample lies on
# one piece it is that piece's Newton or BFGS step. A step is accepted by the solver's
# sufficient-increase test; when the predicted step is shorter than ε, or no step is
# accepted, ε shrinks tenfold. The search stops at ε = `_KINK_RADIUS_MIN` with a
# predicted step shorter than ε, or where the smooth convergence test passes.
#
# **Certificate.** Maximizing min_i f_i is the smooth problem max t subject to
# t ≤ f_i, whose second-order sufficient condition certifies a strict local maximum:
# 0 = Σλ_i ∇f_i with λ ≥ 0, Σλ_i = 1 (the hull's weights, so z ≈ 0), and the Hessian
# Σλ_i ∇²f_i negative definite on U, the directions along which every active piece
# changes alike (the orthogonal complement of the differences of their gradients).
# The pieces are the hull's support. With η Hessians (the model's blocks or a
# declared pattern) the Hessians are evaluated at its sample points; with gradients
# only, their products along U are forward differences there, and the jump between
# two pieces lies outside U, so a difference that crosses between them is projected
# away. Lanczos on U gives the smallest eigenvalue, exactly when the Krylov space
# fills U (at most `_KINK_CURVATURE_STEPS` dimensions). Last, the density is
# evaluated one `_KINK_PROBE` prior standard deviation along the coordinates, the
# sampled gradients and a few random directions, both ways: a point that gains more
# than f resolves there restarts the search from it, at most `_KINK_RESTARTS` times,
# which guards against a piece that the samples missed.

const _KINK_RADIUS = 0.1
const _KINK_RADIUS_MIN = 1.0e-6
const _KINK_SHRINK = 0.1
const _KINK_SAMPLES = 64
const _KINK_PROBE = 1.0e-3
const _KINK_RESTARTS = 3
const _KINK_CURVATURE_STEPS = 36

# One stalled subject's search. `k` indexes the solver's subjects; `points` and
# `gradients` are the samples, the current point first; `weights` and `step` are the
# last hull's weights over them and its step W z, and `jumps` an orthonormal basis of
# the differences of the standardized gradients on that hull's support (the
# directions across the kinks there).
mutable struct _KinkSearch{T, P, R}
    k::Int
    prior::P
    x::Vector{T}
    ℓ::T
    f::T
    g::Vector{T}
    hessian::Any
    ε::T
    points::Vector{Vector{T}}
    values::Vector{T}
    gradients::Vector{Vector{T}}
    weights::Vector{T}
    z::Vector{T}
    step::Vector{T}
    jumps::Matrix{T}
    rng::R
    iterations::Int
    restarts::Int
    failures::Int
    status::Symbol
end

# Runs the phase on the solver's subjects `ks` and writes each one's end point into
# H, ℓ, f, G. Returns, per subject in `ks`, `:smooth` (a mode where the smooth test
# passes), `:kink` (a certified mode on a kink) or `:none`, and for a kink the
# standardized norm ‖Lᵀz‖∞ of the element z of the hull of its last samples' gradients
# nearest zero in L⁻¹η (a measure of Clarke stationarity at the sampling radius).
function _kink_search!(method, curvature, model, subjects, θ, σ, prior, H, ℓ, f, G, ks,
        iterations, opt, progress)
    T = eltype(f)
    searches = map(ks) do k
        x = Vector{T}(_column(H, k))
        s = _KinkSearch(k, _subject_prior(prior, subjects[k]), x, ℓ[k], f[k], Vector{T}(_column(G, k)),
            nothing, T(_KINK_RADIUS), Vector{T}[], T[], Vector{T}[], T[], Vector{T}(_column(G, k)), T[],
            zeros(T, length(x), 0),
            Xoshiro(subjects[k]), 0, 0, 0, :search)
        s.hessian = _kink_hessian(method, curvature, k)
        s
    end
    while any(s -> s.status == :search, searches)
        while true
            throw_if_interrupted(progress)
            active = filter(s -> s.status == :search, searches)
            isempty(active) && break
            progress === nothing || update_progress!(progress,
                "kink search · $(length(active)) of $(length(searches)) subject(s) searching")
            _kink_iteration!(method, curvature, model, subjects, θ, σ, prior, H, active, opt)
        end
        candidates = filter(s -> s.status == :stationary, searches)
        _kink_certify!(method, curvature, model, subjects, θ, σ, prior, H, candidates, opt)
    end
    outcomes = Symbol[]
    norms = T[]
    for s in searches
        k = s.k
        _set_column!(H, k, s.x)
        ℓ[k], f[k] = s.ℓ, s.f
        _set_column!(G, k, s.g)
        iterations[k] += s.iterations
        _kink_store_curvature!(method, curvature, s)
        push!(outcomes, s.status == :smooth ? :smooth : s.status == :kink ? :kink : :none)
        push!(norms, s.status == :kink ? _kink_stationarity(s) : T(NaN))
    end
    return outcomes, norms
end

function _kink_stationarity(s)
    T = eltype(s.x)
    std = reduce(hcat, [_factor_transpose_times(s.prior, g) for g in s.gradients])
    z = std * _hull_weights(Symmetric(std' * std), zeros(T, size(std, 2)))
    return maximum(abs, z; init = zero(T))
end

_kink_hessian(::NewtonEBE, state, k) = state.hessians[k]
_kink_hessian(::QuasiNewtonEBE, state, k) = nothing

# W times the columns of `Gs`: Newton's inverse curvature (-H)⁻¹ where -H is positive
# definite, else the prior covariance; BFGS's limited-memory inverse Hessian.
function _kink_metric(::NewtonEBE, state, s, Gs)
    F = cholesky(Symmetric(-s.hessian); check = false)
    issuccess(F) && return Matrix(F \ Gs)
    return reduce(hcat, [_covariance_times(s.prior, Gs[:, j]) for j in axes(Gs, 2)])
end
_kink_metric(::QuasiNewtonEBE, state, s, Gs) =
    reduce(hcat, [_inverse_hessian_times(state, s.k, Gs[:, j]) for j in axes(Gs, 2)])

# The smooth solver's own convergence test at the search's point.
function _kink_smooth(::NewtonEBE, state, s, opt)
    p, decrement = _newton_step(s.g, s.hessian)
    return p !== nothing && decrement / 2 <= opt.decrement_tol
end
_kink_smooth(::QuasiNewtonEBE, state, s, opt) =
    maximum(abs, _factor_transpose_times(s.prior, s.g); init = zero(eltype(s.g))) <= opt.gtol

# After an accepted step: Newton's Hessian at the new points, or a BFGS pair.
function _kink_update!(::NewtonEBE, state, model, subjects, θ, σ, prior, H, moved, steps, previous)
    hessians = _complete_hessians(model, subjects[[s.k for s in moved]], θ, σ, prior,
        _from_columns(H, [s.x for s in moved]))
    for (s, Hk) in zip(moved, hessians)
        s.hessian = Hk
    end
end
function _kink_update!(::QuasiNewtonEBE, state, model, subjects, θ, σ, prior, H, moved, steps, previous)
    for (s, step, g) in zip(moved, steps, previous)
        _store_pair!(state, s.k, step, g - s.g)
    end
end

# The solver's curvature state at the search's end point: Newton's Hessian there; for
# BFGS, unknown unless the certificate computed it, so that the result's Hessian
# (when the model reports Hessians) is formed there.
_kink_store_curvature!(::NewtonEBE, state, s) = (state.hessians[s.k] = s.hessian; state)
function _kink_store_curvature!(::QuasiNewtonEBE, state, s)
    s.status == :smooth || (state.hessians[s.k] = nothing)
    return state
end

_kink_distance(s, a) = norm(_factor_solve(s.prior, a - s.x))

# Complete-data values and gradients of the searches' subjects at one point each:
# `(ℓ, f, g)`, or `nothing` where the point is outside the model's domain (the batch
# is then retried one subject at a time).
function _kink_gradients(model, subjects, θ, σ, prior, H, searches, points)
    function evaluate(js)
        ℓs, fs, Gs = _complete_gradients(model, subjects[[searches[j].k for j in js]], θ, σ, prior,
            _from_columns(H, points[js]))
        return [isfinite(fs[c]) ? (ℓs[c], fs[c], Vector(_column(Gs, c))) : nothing for c in eachindex(js)]
    end
    batch = _feasible(evaluate, collect(eachindex(searches)))
    batch === nothing || return batch
    return [_feasible(j -> only(evaluate([j])), j) for j in eachindex(searches)]
end

# A point uniform in the ball of radius ε around the search's point (in L⁻¹η).
function _kink_sample(s)
    n = length(s.x)
    u = randn(s.rng, eltype(s.x), n)
    u .*= rand(s.rng)^(1 / n) / norm(u)
    return s.x + s.ε * _factor_times(s.prior, u)
end

# One iteration for every searching subject: samples, hulls, then a line search.
function _kink_iteration!(method, curvature, model, subjects, θ, σ, prior, H, active, opt)
    T = eltype(first(active).x)
    for s in active
        keep = [j for j in 2:length(s.points) if _kink_distance(s, s.points[j]) <= s.ε]
        keep = keep[max(1, end - _kink_samples(s) + 2):end]
        s.points = [[s.x]; s.points[keep]]
        s.values = [s.f; s.values[keep]]
        s.gradients = [[s.g]; s.gradients[keep]]
    end
    needs = [max(_kink_samples(s) + 1 - length(s.points), 1) * 2^min(s.failures, 4) for s in active]
    for r in 1:maximum(needs; init = 0)
        group = active[needs .>= r]
        points = [_kink_sample(s) for s in group]
        for (s, p, e) in zip(group, points, _kink_gradients(model, subjects, θ, σ, prior, H, group, points))
            e === nothing && continue
            push!(s.points, p)
            push!(s.values, e[2])
            push!(s.gradients, e[3])
        end
    end
    stepping = eltype(active)[]
    for s in active
        if _kink_smooth(method, curvature, s, opt)
            s.status = :stationary
            s.weights = [one(T); zeros(T, length(s.points) - 1)]
            s.z = s.g
            s.jumps = zeros(T, length(s.x), 0)
            continue
        end
        Gs = reduce(hcat, s.gradients)
        WG = _kink_metric(method, curvature, s, Gs)
        s.weights = _hull_weights(Symmetric((Gs' * WG + WG' * Gs) / 2), _kink_errors(s))
        s.step = WG * s.weights
        s.z = Gs * s.weights
        s.jumps = _kink_jumps(s)
        if norm(_factor_solve(s.prior, s.step)) <= s.ε
            if !_kink_smallest(s)
                _kink_shrink!(s)
                continue
            elseif !isempty(s.jumps)
                s.status = :stationary
                continue
            end
            # One smooth piece at the smallest radius: its own step.
        end
        push!(stepping, s)
    end
    isempty(stepping) && return nothing
    decrements = [dot(s.z, s.step) + dot(s.weights, _kink_errors(s))
        for s in stepping]
    t = ones(T, length(stepping))
    pending = collect(eachindex(stepping))
    for _ in 0:opt.max_backtracks
        isempty(pending) && break
        _, ft = _complete_values(model, subjects[[stepping[j].k for j in pending]], θ, σ, prior,
            _from_columns(H, [stepping[j].x + t[j] * stepping[j].step for j in pending]))
        accepted = Int[]
        for (j, value) in zip(pending, ft)
            s, gain = stepping[j], value - stepping[j].f
            isfinite(value) || continue
            gain >= 1.0e-4 * t[j] * decrements[j] ||
                (t[j] == 1 && decrements[j] <= _resolution(s.f) && gain >= -_resolution(s.f)) || continue
            push!(accepted, j)
        end
        setdiff!(pending, accepted)
        t[pending] ./= 2
    end
    for j in pending
        s = stepping[j]
        _kink_failed_step!(s)
    end
    moved = setdiff(eachindex(stepping), pending)
    isempty(moved) && return nothing
    _kink_move!(method, curvature, model, subjects, θ, σ, prior, H, stepping[moved],
        [stepping[j].x + t[j] * stepping[j].step for j in moved])
    for s in stepping[moved]
        s.failures = 0
        s.iterations += 1
        s.iterations >= opt.maxiter && s.status == :search && (s.status = :failed)
    end
    return nothing
end

# How far each sample's tangent plane lies above the density at the current point,
# f_j + g_jᵀ(x - x_j) - f(x), floored at 0: a sample from another piece than the
# current point's is as far above as the current point lies from the kink between
# them, so its weight pulls the step back onto that kink.
_kink_errors(s) = [max(v + dot(g, s.x - p) - s.f, zero(v)) for (p, v, g) in zip(s.points, s.values, s.gradients)]

_kink_smallest(s) = s.ε <= _KINK_RADIUS_MIN * (1 + sqrt(eps(eltype(s.x))))

# Fresh gradients sampled per iteration while the search advances (Burke, Lewis &
# Overton take n + 1 besides the current point); earlier samples still inside the
# ball are kept beside them.
_kink_samples(s) = min(length(s.x) + 1, _KINK_SAMPLES)

_kink_shrink!(s) = (s.ε *= _KINK_SHRINK; s.failures = 0; s)

# No step accepted. Where the current point lies on a kink, a step that every sample
# put on one side of it crosses it at once, so the samples missed a piece: the next
# iteration draws a fresh set at the same radius, twice as large after each failure.
# A second failure in a row shrinks the radius; at the smallest one the fifth ends
# the search, after 31 times the usual number of fresh samples.
function _kink_failed_step!(s)
    s.failures += 1
    s.points, s.values, s.gradients = s.points[1:1], s.values[1:1], s.gradients[1:1]
    if _kink_smallest(s)
        s.failures >= 5 && (s.status = :failed)
    elseif s.failures >= 2
        _kink_shrink!(s)
    end
    return s
end

# Moves the searches to `points`: their values and gradients there, and the method's
# curvature (Newton's Hessian, or a BFGS pair). One outside the model's domain fails.
function _kink_move!(method, curvature, model, subjects, θ, σ, prior, H, searches, points)
    values = _kink_gradients(model, subjects, θ, σ, prior, H, searches, points)
    moved = eltype(searches)[]
    steps = Vector{eltype(first(points))}[]
    previous = Vector{eltype(first(points))}[]
    for (s, p, e) in zip(searches, points, values)
        if e === nothing
            s.status = :failed
            continue
        end
        push!(steps, p - s.x)
        push!(previous, s.g)
        s.x = p
        s.ℓ, s.f, s.g = e
        push!(moved, s)
    end
    isempty(moved) || _kink_update!(method, curvature, model, subjects, θ, σ, prior, H, moved, steps, previous)
    return nothing
end

# An orthonormal basis of the differences of the standardized gradients on the hull's
# support: the directions across the kinks that meet there. Differences below
# √(`_KINK_RADIUS_MIN`) of the gradients' scale are the change of one smooth piece
# across the sampling ball, not a kink.
function _kink_jumps(s)
    T = eltype(s.x)
    support = findall(>(sqrt(eps(T)) * maximum(s.weights)), s.weights)
    std = [_factor_transpose_times(s.prior, s.gradients[j]) for j in support]
    length(std) <= 1 && return zeros(T, length(s.x), 0)
    D = reduce(hcat, [v - std[1] for v in std[2:end]])
    F = svd(D)
    scale = max(one(T), maximum(norm, std))
    return F.U[:, F.S .> sqrt(T(_KINK_RADIUS_MIN)) * scale]
end

# The weights λ ≥ 0, Σλ = 1 minimizing ½λᵀQλ + eᵀλ, by the active-set method of
# Wolfe (1976), whose minimum-norm point is the case e = 0: on a support S it solves
# the equality-constrained problem, and moves to the boundary and drops a point when
# that leaves the simplex; it adds the point of steepest descent while one improves.
function _hull_weights(Q::AbstractMatrix{T}, e::AbstractVector{T}) where {T}
    m = size(Q, 1)
    scale = max(maximum(abs, diag(Q)), maximum(abs, e; init = zero(T)), floatmin(T))
    tol = 1.0e3 * eps(T) * scale
    S = [argmin(diag(Q) ./ 2 .+ e)]
    λ = [one(T)]
    for _ in 1:(10 * m + 10)
        gradient = Q[:, S] * λ .+ e
        level = dot(λ, gradient[S])
        j = argmin(gradient)
        (gradient[j] >= level - tol || j in S) && break
        push!(S, j)
        push!(λ, zero(T))
        for _ in 1:(m + 1)
            k = length(S)
            K = [Q[S, S] ones(T, k); ones(T, 1, k) zero(T)]
            μ = (pinv(K) * [-e[S]; one(T)])[1:k]
            if all(>(0), μ)
                λ = μ
                break
            end
            θ = minimum((λ[i] / (λ[i] - μ[i]) for i in 1:k if μ[i] <= 0 && λ[i] > μ[i]); init = one(T))
            λ = (1 - θ) .* λ .+ θ .* μ
            keep = findall(>(eps(T)), λ)
            S, λ = S[keep], λ[keep] ./ sum(λ[keep])
        end
    end
    w = zeros(T, m)
    w[S] = λ
    return w
end

# The certificate for the subjects whose search stopped (`:stationary`): the smooth
# test with the method's own curvature check, else the second-order condition on U
# and the probes. Each ends `:smooth`, `:kink`, `:failed`, or `:search` again from a
# point a probe found higher.
function _kink_certify!(method, curvature, model, subjects, θ, σ, prior, H, candidates, opt)
    isempty(candidates) && return nothing
    smooth = filter(s -> _kink_smooth(method, curvature, s, opt), candidates)
    smooth = _kink_confirm_smooth!(method, curvature, model, subjects, θ, σ, prior, H, smooth)
    _kink_smooth_curvature!(method, curvature, model, subjects, θ, σ, prior, H, smooth)
    kinked = filter(s -> !(s in smooth), candidates)
    for s in kinked
        isempty(s.jumps) && (s.status = :failed)
    end
    kinked = filter(s -> s.status == :stationary, kinked)
    minima = _kink_curvatures(model, subjects, θ, σ, prior, H, kinked)
    for (s, μ) in zip(kinked, minima)
        μ !== nothing && μ > 0 || (s.status = :failed)
    end
    _kink_probe!(method, curvature, model, subjects, θ, σ, prior, H,
        filter(s -> s.status == :stationary, kinked))
    return nothing
end

# A point where the smooth test passes can still lie on a kink, where the gradient the
# model reports is one piece's or a subgradient between them that happens to vanish:
# `neta + 1` gradients sampled within the smallest radius show whether they jump. The
# point's own gradient is left out of them, since on a kink it need not be a piece's.
# Returns the searches whose samples agree; the others become kink candidates.
function _kink_confirm_smooth!(method, curvature, model, subjects, θ, σ, prior, H, smooth)
    isempty(smooth) && return smooth
    T = eltype(first(smooth).x)
    for s in smooth
        s.ε = T(_KINK_RADIUS_MIN)
        s.points, s.values, s.gradients = Vector{T}[], T[], Vector{T}[]
    end
    for _ in 1:maximum(_kink_samples, smooth)
        points = [_kink_sample(s) for s in smooth]
        for (s, p, e) in zip(smooth, points, _kink_gradients(model, subjects, θ, σ, prior, H, smooth, points))
            e === nothing && continue
            push!(s.points, p)
            push!(s.values, e[2])
            push!(s.gradients, e[3])
        end
    end
    agreeing = eltype(smooth)[]
    for s in smooth
        if length(s.points) < 2
            push!(agreeing, s)
            continue
        end
        Gs = reduce(hcat, s.gradients)
        WG = _kink_metric(method, curvature, s, Gs)
        s.weights = ones(T, length(s.points)) ./ length(s.points)
        s.jumps = _kink_jumps(s)
        if isempty(s.jumps)
            push!(agreeing, s)
            continue
        end
        s.weights = _hull_weights(Symmetric((Gs' * WG + WG' * Gs) / 2), _kink_errors(s))
        s.z = Gs * s.weights
        s.jumps = _kink_jumps(s)
        isempty(s.jumps) && push!(agreeing, s)
    end
    return agreeing
end

function _kink_smooth_curvature!(::NewtonEBE, state, model, subjects, θ, σ, prior, H, smooth)
    for s in smooth
        s.status = :smooth
    end
end
function _kink_smooth_curvature!(::QuasiNewtonEBE, state, model, subjects, θ, σ, prior, H, smooth)
    isempty(smooth) && return nothing
    ks = [s.k for s in smooth]
    _store_hessians!(state, model, subjects, θ, σ, prior, ks, _from_columns(H, [s.x for s in smooth]),
        _from_columns(H, [s.g for s in smooth]))
    for s in smooth
        s.status = state.hessians[s.k] === nothing || _nonconcave(state, s.k) ? :failed : :smooth
    end
end

# The smallest eigenvalue on U of -LᵀHL, with H = Σλ_j ∇²f(x_j) over the hull's
# support, per subject (`nothing` where its products left the model's domain).
function _kink_curvatures(model, subjects, θ, σ, prior, H, searches)
    isempty(searches) && return []
    T = eltype(first(searches).x)
    supports = [findall(>(sqrt(eps(T)) * maximum(s.weights)), s.weights) for s in searches]
    if _reports_eta_hessians(model)
        hessians = [Vector{Any}(undef, length(sp)) for sp in supports]
        for r in 1:maximum(length, supports)
            group = findall(j -> length(supports[j]) >= r, eachindex(searches))
            Hs = _complete_hessians(model, subjects[[searches[j].k for j in group]], θ, σ, prior,
                _from_columns(H, [searches[j].points[supports[j][r]] for j in group]))
            for (j, Hk) in zip(group, Hs)
                hessians[j][r] = Hk
            end
        end
        weighted = [sum(s.weights[j] * Hk for (j, Hk) in zip(sp, hs)) for (s, sp, hs) in zip(searches, supports, hessians)]
        products = (js, vs) -> [-_factor_transpose_times(searches[j].prior,
            weighted[j] * _factor_times(searches[j].prior, v)) for (j, v) in zip(js, vs)]
    else
        products = (js, vs) -> _kink_difference_products(model, subjects, θ, σ, prior, H,
            searches, supports, js, vs)
    end
    return _restricted_min_eigenvalues(products, [s.jumps for s in searches],
        [length(s.x) for s in searches], _KINK_CURVATURE_STEPS)
end

# Σλ_j (∇f(x_j + h Lv) - ∇f(x_j)) / h in standardized coordinates, by forward
# differences with h far below the sampling radius, so that they stay on the piece of
# x_j or cross only between the support's pieces, whose jump U excludes.
function _kink_difference_products(model, subjects, θ, σ, prior, H, searches, supports, js, vs)
    T = eltype(first(searches).x)
    h = T(1.0e-4 * _KINK_RADIUS_MIN)
    totals = [zeros(T, length(searches[j].x)) for j in js]
    unknown = falses(length(js))
    for r in 1:maximum(c -> length(supports[js[c]]), eachindex(js))
        group = filter(c -> length(supports[js[c]]) >= r && !unknown[c], eachindex(js))
        isempty(group) && break
        points = [searches[js[c]].points[supports[js[c]][r]] + h * _factor_times(searches[js[c]].prior, vs[c])
            for c in group]
        values = _kink_gradients(model, subjects, θ, σ, prior, H, searches[js[group]], points)
        for (c, e) in zip(group, values)
            if e === nothing
                unknown[c] = true
                continue
            end
            s = searches[js[c]]
            j = supports[js[c]][r]
            totals[c] .+= s.weights[j] .* (e[3] - s.gradients[j]) ./ h
        end
    end
    return [unknown[c] ? nothing : -_factor_transpose_times(searches[js[c]].prior, totals[c]) for c in eachindex(js)]
end

# The smallest eigenvalue of each subject's symmetric operator restricted to U, the
# orthogonal complement of the columns of `jumps[j]`, by Lanczos with full
# reorthogonalization; exact when the Krylov space fills U (dim U ≤ `limit`), `Inf`
# when U is empty, `nothing` when a product is unknown.
function _restricted_min_eigenvalues(products, jumps, dims, limit)
    T = isempty(jumps) ? Float64 : eltype(first(jumps))
    results = Vector{Any}(nothing, length(dims))
    project(V, w) = w - V * (V' * w)
    basis = [Vector{T}[] for _ in dims]
    α = [T[] for _ in dims]
    β = [T[] for _ in dims]
    q = Vector{Vector{T}}(undef, length(dims))
    steps = zeros(Int, length(dims))
    active = Int[]
    for j in eachindex(dims)
        steps[j] = min(dims[j] - size(jumps[j], 2), limit)
        if steps[j] <= 0
            results[j] = T(Inf)
            continue
        end
        v = project(jumps[j], randn(Xoshiro(0), T, dims[j]))
        q[j] = v ./ norm(v)
        push!(active, j)
    end
    while !isempty(active)
        ws = products(active, [q[j] for j in active])
        for (j, w) in zip(copy(active), ws)
            if w === nothing
                filter!(!=(j), active)
                continue
            end
            w = project(jumps[j], w)
            push!(basis[j], q[j])
            push!(α[j], dot(q[j], w))
            for _ in 1:2, b in basis[j]
                w = w - dot(b, w) * b
            end
            w = project(jumps[j], w)
            b = norm(w)
            if length(α[j]) >= steps[j] || b <= sqrt(eps(T)) * maximum(abs, α[j])
                # Not `eigmin`: on Julia 1.10.12 its index-range path (`eigvals(T, 1:1)`)
                # returns the larger eigenvalue of some 2 × 2 tridiagonals; 1.13 does not.
                results[j] = minimum(eigvals(SymTridiagonal(α[j], β[j])))
                filter!(!=(j), active)
                continue
            end
            push!(β[j], b)
            q[j] = w ./ b
        end
    end
    return results
end

# Evaluates the density one `_KINK_PROBE` prior SD from each candidate along ± its
# coordinates (at most 16 of them, else random ones), ± its support's standardized
# gradients and its kink directions, and ± 4 random directions. A candidate where one
# gains more than f resolves restarts its search from the highest; the others are
# certified.
function _kink_probe!(method, curvature, model, subjects, θ, σ, prior, H, candidates)
    isempty(candidates) && return nothing
    T = eltype(first(candidates).x)
    directions = map(candidates) do s
        n = length(s.x)
        support = findall(>(sqrt(eps(T)) * maximum(s.weights)), s.weights)
        coordinates = n <= 16 ? collect(1:n) : randperm(s.rng, n)[1:16]
        D = Vector{T}[]
        append!(D, [setindex!(zeros(T, n), one(T), i) for i in coordinates])
        append!(D, [normalize(_factor_transpose_times(s.prior, s.gradients[j])) for j in support])
        append!(D, eachcol(s.jumps))
        append!(D, [normalize(randn(s.rng, T, n)) for _ in 1:4])
        [D; -D]
    end
    best = [(s.f, s.x) for s in candidates]
    for r in 1:maximum(length, directions)
        group = filter(j -> length(directions[j]) >= r, eachindex(candidates))
        points = [candidates[j].x + T(_KINK_PROBE) * _factor_times(candidates[j].prior, directions[j][r]) for j in group]
        _, ft = _complete_values(model, subjects[[candidates[j].k for j in group]], θ, σ, prior,
            _from_columns(H, points))
        for (j, p, value) in zip(group, points, ft)
            isfinite(value) && value > first(best[j]) && (best[j] = (value, p))
        end
    end
    restart = Int[]
    for (j, s) in enumerate(candidates)
        if first(best[j]) - s.f > _resolution(s.f)
            s.restarts += 1
            s.restarts > _KINK_RESTARTS ? (s.status = :failed) : push!(restart, j)
        else
            s.status = :kink
        end
    end
    isempty(restart) && return nothing
    moved = candidates[restart]
    _kink_move!(method, curvature, model, subjects, θ, σ, prior, H, moved, [last(best[j]) for j in restart])
    for s in moved
        s.status == :failed && continue
        s.ε = T(_KINK_RADIUS)
        s.points, s.values, s.gradients = Vector{T}[], T[], Vector{T}[]
        s.status = :search
    end
    return nothing
end
