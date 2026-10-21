"""
    FixedOmega(Ω::AbstractMatrix)
    FixedOmega(; sd = nothing, correlation = nothing)

A known value for all or part of one random-effect covariance family, declared
through [`fixed_omega`](@ref). `FixedOmega(Ω)` holds the whole family at the
covariance matrix `Ω`, like nlmixr2's `fix()` on an ω block or NONMEM's
`OMEGA … FIX`.

The keyword form fixes parts of a family in standard-deviation and correlation
terms, `Ω = D R D` with `D = Diagonal(sd)`:

- `sd` is `nothing` (every margin estimated) or a vector with one entry per
  margin: a positive standard deviation for a fixed margin, and `nothing` or
  `missing` for an estimated one;
- `correlation` is `nothing` (estimated) or the known correlation matrix `R`,
  symmetric positive definite with unit diagonal.

The estimators fit the remaining parts with the fixed parts held at their
values. A family of size one has only its standard deviation.
"""
struct FixedOmega
    sd::Union{Nothing, Vector{Union{Nothing, Float64}}}
    correlation::Union{Nothing, Matrix{Float64}}
    function FixedOmega(sd, correlation)
        sd = sd === nothing ? nothing :
            Union{Nothing, Float64}[x === nothing || x === missing ? nothing : Float64(x) for x in sd]
        sd === nothing || all(x -> x === nothing || (isfinite(x) && x > 0), sd) ||
            throw(ArgumentError("FixedOmega: fixed standard deviations must be finite and positive, got $sd"))
        if correlation !== nothing
            R = Matrix{Float64}(correlation)
            k = size(R, 1)
            size(R) == (k, k) && k >= 1 || throw(ArgumentError(
                "FixedOmega: correlation must be a square matrix, got size $(size(R))"))
            all(j -> isapprox(R[j, j], 1; atol = 1.0e-12), 1:k) || throw(ArgumentError(
                "FixedOmega: correlation must have a unit diagonal"))
            isapprox(R, R'; atol = 1.0e-12) || throw(ArgumentError(
                "FixedOmega: correlation must be symmetric"))
            R = (R + R') / 2
            for j in 1:k
                R[j, j] = 1
            end
            isposdef(Symmetric(R)) || throw(ArgumentError(
                "FixedOmega: correlation must be positive definite"))
            sd === nothing || length(sd) == k || throw(DimensionMismatch(
                "FixedOmega: $(length(sd)) standard deviations for a $k×$k correlation"))
            correlation = R
        end
        sd !== nothing && any(!isnothing, sd) || correlation !== nothing ||
            throw(ArgumentError("FixedOmega fixes nothing: give a standard deviation or a correlation"))
        return new(sd, correlation)
    end
end

FixedOmega(; sd = nothing, correlation = nothing) = FixedOmega(sd, correlation)

function FixedOmega(Ω::AbstractMatrix{<:Real})
    k = size(Ω, 1)
    size(Ω) == (k, k) && k >= 1 || throw(ArgumentError(
        "FixedOmega: Ω must be a square matrix, got size $(size(Ω))"))
    isapprox(Ω, Ω'; rtol = 1.0e-12) || throw(ArgumentError("FixedOmega: Ω must be symmetric"))
    isposdef(Symmetric(Matrix{Float64}(Ω))) || throw(ArgumentError(
        "FixedOmega: Ω must be positive definite"))
    sd = sqrt.(Float64.(diag(Ω)))
    R = Matrix{Float64}(Ω) ./ (sd * sd')
    R = (R + R') / 2
    for j in 1:k
        R[j, j] = 1
    end
    return FixedOmega(sd, R)
end

"""
    fixed_omega(model) -> nothing, or family => FixedOmega pairs

Optional declaration that some random-effect covariance families of `model`
are known, wholly or in part. Return `nothing` (the default) when every family
is estimated, or a collection of `family => FixedOmega(…)` pairs (a vector of
pairs, a single pair, or a dictionary). A family is an entry of the layout's
`eta_blocks` ([`NLMELayout`](@ref)) or `family_sizes`
([`RaggedNLMELayout`](@ref)), by position.

[`focei`](@ref) and [`saem`](@ref) estimate only the free parts and keep the
fixed parts at their values, so the fitted `omega` and `omega_matrix` report
them. Their starting `ω0` must already hold the fixed values;
[`apply_fixed_omega`](@ref) builds such a vector. [`empirical_bayes`](@ref)
evaluates at whatever `ω` it is given, and [`check_protocol`](@ref) verifies
that its `ω` holds the declared values.
"""
fixed_omega(model) = nothing

"""
    apply_fixed_omega(model, ω) -> Vector

The Ω parameters `ω` with each family declared by [`fixed_omega`](@ref) moved
onto its fixed values. The estimated parts of a constrained family keep their
values in `ω`, in standard-deviation and correlation terms: an estimated
margin keeps its standard deviation, and an estimated correlation keeps the
correlation of `ω`. Unconstrained families are unchanged.
"""
function apply_fixed_omega(model, ω::AbstractVector)
    layout = nlme_layout(model)
    _check_length("ω", ω, nomega(layout))
    constraints = _omega_constraints(layout, fixed_omega(model))
    return Vector{Float64}(_apply_omega_constraints(constraints, Float64.(ω)))
end

# A validated FixedOmega for a family of size k. Standard deviations are held as
# log values; the correlation by the coordinates `m` of `_scale_coordinates`.
struct _FixedFamily
    sd_fixed::BitVector
    sd::Vector{Float64}
    log_sd::Vector{Float64}
    correlation_fixed::Bool
    correlation::Matrix{Float64}
    m::Vector{Float64}
    fully_fixed::Bool
    segment::Vector{Float64}
    covariance::Matrix{Float64}
end

struct _OmegaConstraints
    sizes::Vector{Int}
    ranges::Vector{UnitRange{Int}}
    families::Vector{Union{Nothing, _FixedFamily}}
end

_omega_family_sizes(layout::NLMELayout) = layout.eta_blocks
_omega_family_sizes(layout::RaggedNLMELayout) = layout.family_sizes
_omega_family_ranges(layout::NLMELayout) = last(_block_ranges(layout))
_omega_family_ranges(layout::RaggedNLMELayout) = _ragged_family_ranges(layout)

_omega_constraints(model) = _omega_constraints(nlme_layout(model), fixed_omega(model))
_omega_constraints(layout, ::Nothing) = nothing
_omega_constraints(layout, spec::Pair) = _omega_constraints(layout, [spec])
_omega_constraints(layout, spec::AbstractDict) = _omega_constraints(layout, collect(spec))
function _omega_constraints(layout, spec)
    sizes = _omega_family_sizes(layout)
    families = Vector{Union{Nothing, _FixedFamily}}(nothing, length(sizes))
    for entry in spec
        entry isa Pair || throw(ArgumentError(
            "fixed_omega must return family => FixedOmega pairs, got $(repr(entry))"))
        f, constraint = entry.first, entry.second
        f isa Integer && 1 <= f <= length(sizes) || throw(ArgumentError(
            "fixed_omega names family $(repr(f)); the layout has families 1:$(length(sizes))"))
        constraint isa FixedOmega || throw(ArgumentError(
            "fixed_omega for family $f must be a FixedOmega, got $(typeof(constraint))"))
        families[f] === nothing || throw(ArgumentError("fixed_omega names family $f twice"))
        families[f] = _fixed_family(constraint, sizes[f], f)
    end
    all(isnothing, families) && return nothing
    return _OmegaConstraints(collect(sizes), _omega_family_ranges(layout), families)
end

function _fixed_family(c::FixedOmega, k, f)
    c.sd === nothing || length(c.sd) == k || throw(DimensionMismatch(
        "FixedOmega for family $f has $(length(c.sd)) standard deviations; the family has size $k"))
    c.correlation === nothing || size(c.correlation) == (k, k) || throw(DimensionMismatch(
        "FixedOmega for family $f has a $(size(c.correlation)) correlation; the family has size $k"))
    sd_fixed = c.sd === nothing ? falses(k) : BitVector([x !== nothing for x in c.sd])
    correlation_fixed = k == 1 || c.correlation !== nothing
    any(sd_fixed) || k > 1 || throw(ArgumentError(
        "FixedOmega for family $f fixes nothing: a family of size one has no correlation"))
    sd = c.sd === nothing ? ones(k) : [x === nothing ? 1.0 : x for x in c.sd]
    R = c.correlation === nothing ? Matrix{Float64}(I, k, k) : c.correlation
    m = _correlation_coordinates(R)
    fully_fixed = all(sd_fixed) && correlation_fixed
    segment = _logcholesky_coordinates(log.(sd), m, k)
    L = omega_cholesky(NLMELayout(0, 0, [k]), segment)
    return _FixedFamily(sd_fixed, sd, log.(sd), correlation_fixed, R, m,
        fully_fixed, segment, Matrix(L * L'))
end

# Log-Cholesky coordinates of one family ↔ its log standard deviations `s` and
# correlation coordinates `m`. Row `j` of the Cholesky factor `L` of `Ω` is
# `exp(s_j) (m_j1, …, m_j,j-1, 1) / ‖(m_j1, …, m_j,j-1, 1)‖`, so `s_j` is the
# log of the row norm and `m_jl = L_jl / L_jj`. The `m` are unconstrained and
# determine the correlation matrix alone.
function _scale_coordinates(segment::AbstractVector, k::Integer)
    T = float(eltype(segment))
    s = zeros(T, k)
    m = zeros(T, k * (k - 1) ÷ 2)
    c = 1
    mc = 1
    for j in 1:k
        diagonal = exp(segment[c + j - 1])
        q = zero(T)
        for l in 1:(j - 1)
            m[mc] = segment[c + l - 1] / diagonal
            q += m[mc]^2
            mc += 1
        end
        s[j] = segment[c + j - 1] + log1p(q) / 2
        c += j
    end
    return s, m
end

function _logcholesky_coordinates(s::AbstractVector, m::AbstractVector, k::Integer)
    T = float(promote_type(eltype(s), eltype(m)))
    segment = zeros(T, k * (k + 1) ÷ 2)
    c = 1
    mc = 1
    for j in 1:k
        q = zero(T)
        for l in 1:(j - 1)
            q += m[mc + l - 1]^2
        end
        logn = log1p(q) / 2
        scale = exp(s[j] - logn)
        for l in 1:(j - 1)
            segment[c + l - 1] = scale * m[mc + l - 1]
        end
        segment[c + j - 1] = s[j] - logn
        mc += j - 1
        c += j
    end
    return segment
end

# The gradient in (s, m) of a function whose gradient in the log-Cholesky
# segment is `g`.
function _scale_gradient(g::AbstractVector, s::AbstractVector, m::AbstractVector, k::Integer)
    gs = zeros(Float64, k)
    gm = zeros(Float64, length(m))
    c = 1
    mc = 1
    for j in 1:k
        n2 = one(Float64)
        for l in 1:(j - 1)
            n2 += m[mc + l - 1]^2
        end
        scale = exp(s[j]) / sqrt(n2)
        gdiagonal = g[c + j - 1]
        projection = zero(Float64)
        for l in 1:(j - 1)
            projection += g[c + l - 1] * m[mc + l - 1]
        end
        gs[j] = gdiagonal + scale * projection
        for a in 1:(j - 1)
            ma = m[mc + a - 1]
            gm[mc + a - 1] = -gdiagonal * ma / n2 + scale * (g[c + a - 1] - ma * projection / n2)
        end
        mc += j - 1
        c += j
    end
    return gs, gm
end

function _correlation_coordinates(R::AbstractMatrix)
    k = size(R, 1)
    L = cholesky(Symmetric(Matrix{Float64}(R))).L
    return [L[j, l] / L[j, j] for j in 1:k for l in 1:(j - 1)]
end

function _correlation_from_coordinates(m::AbstractVector, k::Integer)
    L = omega_cholesky(NLMELayout(0, 0, [k]), _logcholesky_coordinates(zeros(k), m, k))
    return Matrix(L * L')
end

# Free coordinates of a partially fixed family: its estimated log standard
# deviations, then its correlation coordinates when the correlation is estimated.
function _family_free(fam::_FixedFamily, segment, k)
    s, m = _scale_coordinates(segment, k)
    z = s[.!fam.sd_fixed]
    return fam.correlation_fixed ? z : vcat(z, m)
end

function _family_assemble(fam::_FixedFamily, z, k)
    nfree = count(!, fam.sd_fixed)
    s = Vector{eltype(z)}(fam.log_sd)
    s[.!fam.sd_fixed] = view(z, 1:nfree)
    m = fam.correlation_fixed ? fam.m : view(z, (nfree + 1):length(z))
    return _logcholesky_coordinates(s, m, k)
end

function _family_free_gradient(fam::_FixedFamily, segment, g, k)
    s, m = _scale_coordinates(segment, k)
    gs, gm = _scale_gradient(g, s, m, k)
    z = gs[.!fam.sd_fixed]
    return fam.correlation_fixed ? z : vcat(z, gm)
end

# The estimators' Ω coordinates: the log-Cholesky segment of an unconstrained
# family, the free (s, m) coordinates of a partially fixed one, and nothing for
# a fully fixed one. Without constraints this is ω itself.
_free_omega(::Nothing, ω) = ω
function _free_omega(c::_OmegaConstraints, ω)
    z = Float64[]
    for (f, fam) in enumerate(c.families)
        segment = view(ω, c.ranges[f])
        if fam === nothing
            append!(z, segment)
        elseif !fam.fully_fixed
            append!(z, _family_free(fam, segment, c.sizes[f]))
        end
    end
    return z
end

_constrained_omega(::Nothing, z) = z
function _constrained_omega(c::_OmegaConstraints, z)
    ω = zeros(eltype(z), last(last(c.ranges)))
    cursor = 0
    for (f, fam) in enumerate(c.families)
        r = c.ranges[f]
        if fam === nothing
            ω[r] = view(z, (cursor + 1):(cursor + length(r)))
            cursor += length(r)
        elseif fam.fully_fixed
            ω[r] = fam.segment
        else
            width = count(!, fam.sd_fixed) + (fam.correlation_fixed ? 0 : length(fam.m))
            ω[r] = _family_assemble(fam, view(z, (cursor + 1):(cursor + width)), c.sizes[f])
            cursor += width
        end
    end
    cursor == length(z) || throw(DimensionMismatch(
        "free Ω coordinates have length $(length(z)); the constraints use $cursor"))
    return ω
end

_free_omega_gradient(::Nothing, ω, g) = g
function _free_omega_gradient(c::_OmegaConstraints, ω, g)
    gz = Float64[]
    for (f, fam) in enumerate(c.families)
        r = c.ranges[f]
        if fam === nothing
            append!(gz, view(g, r))
        elseif !fam.fully_fixed
            append!(gz, _family_free_gradient(fam, view(ω, r), view(g, r), c.sizes[f]))
        end
    end
    return gz
end

_apply_omega_constraints(c, ω) = _constrained_omega(c, _free_omega(c, ω))

_omega_family(::Nothing, f) = nothing
_omega_family(c::_OmegaConstraints, f) = c.families[f]
_fully_fixed_family(c, f) = (fam = _omega_family(c, f); fam !== nothing && fam.fully_fixed)

# The first constrained family whose fixed parts `ω` does not hold.
function _unsatisfied_family(c::_OmegaConstraints, ω)
    for (f, fam) in enumerate(c.families)
        fam === nothing && continue
        k = c.sizes[f]
        L = omega_cholesky(NLMELayout(0, 0, [k]), view(ω, c.ranges[f]))
        Ω = Matrix(L * L')
        sd = sqrt.(diag(Ω))
        all(j -> !fam.sd_fixed[j] || isapprox(sd[j], fam.sd[j]; rtol = 1.0e-8), 1:k) &&
            (!fam.correlation_fixed || isapprox(Ω ./ (sd * sd'), fam.correlation; atol = 1.0e-8)) ||
            return f
    end
    return nothing
end

_checked_omega_constraints(model, layout, ω) = _checked_omega_constraints(
    _omega_constraints(layout, fixed_omega(model)), ω)
_checked_omega_constraints(::Nothing, ω) = nothing
function _checked_omega_constraints(c::_OmegaConstraints, ω)
    f = _unsatisfied_family(c, ω)
    f === nothing || throw(ArgumentError(
        "ω0 does not hold covariance family $f at its fixed_omega(model) value; start from apply_fixed_omega(model, ω0)"))
    return c
end

function _check_fixed_omega(violation, model, layout, ω)
    c = try
        _omega_constraints(layout, fixed_omega(model))
    catch err
        err isa Union{ArgumentError, DimensionMismatch} || rethrow()
        violation("fixed_omega(model) is invalid: " * sprint(showerror, err))
    end
    c === nothing && return nothing
    f = _unsatisfied_family(c, ω)
    f === nothing || violation(
        "ω does not hold covariance family $f at its fixed_omega(model) value; see apply_fixed_omega")
    return nothing
end

# The SAEM covariance floor applies to estimated parts only: the eigenvalues of
# an unconstrained family, the estimated variances of a partially fixed one, and
# the eigenvalues of its estimated correlation. A fully fixed family is the
# model's value.
_omega_admissible(layout, ::Nothing, ω, floor) = _saem_omega_above_floor(layout, ω, floor)
function _omega_admissible(layout, c::_OmegaConstraints, ω, floor)
    all(isfinite, ω) || return false
    for (f, fam) in enumerate(c.families)
        k = c.sizes[f]
        segment = view(ω, c.ranges[f])
        if fam === nothing
            L = omega_cholesky(NLMELayout(0, 0, [k]), segment)
            Ω = Matrix(L * L')
            all(isfinite, Ω) && eigmin(Symmetric(Ω)) >= floor || return false
        elseif !fam.fully_fixed
            _partial_family_admissible(fam, segment, k, floor) || return false
        end
    end
    return true
end

function _partial_family_admissible(fam::_FixedFamily, segment, k, floor)
    s, m = _scale_coordinates(segment, k)
    all(isfinite, s) && all(isfinite, m) || return false
    all(j -> fam.sd_fixed[j] || exp(2s[j]) >= floor, 1:k) || return false
    fam.correlation_fixed && return true
    R = _correlation_from_coordinates(m, k)
    return all(isfinite, R) && eigmin(Symmetric(R)) >= floor
end

function _floor_partial_family(fam::_FixedFamily, segment, k, floor)
    s, m = _scale_coordinates(segment, k)
    for j in 1:k
        fam.sd_fixed[j] || (s[j] = max(s[j], log(floor) / 2))
    end
    if !fam.correlation_fixed
        R = _correlation_from_coordinates(m, k)
        E = eigen(Symmetric(R))
        if minimum(E.values) < floor
            F = E.vectors * Diagonal(max.(E.values, floor)) * E.vectors'
            d = sqrt.(diag(F))
            m = _correlation_coordinates(F ./ (d * d'))
        end
    end
    return _logcholesky_coordinates(s, fam.correlation_fixed ? fam.m : m, k)
end

# Expected Gaussian log density of one family under its pooled sufficient
# statistics: `count` instances with sum of outer products `S`. Returns the
# value and its gradient in the family's log-Cholesky coordinates; `P` is the
# precision of `L Lᵀ`.
function _gaussian_family_value_gradient(L, P, S, count)
    k = size(L, 1)
    value = -(count * (k * log(2π) + 2sum(log, diag(L))) + sum(P .* S)) / 2
    GL = (P * S * P - count * P) * L
    g = zeros(Float64, k * (k + 1) ÷ 2)
    cursor = 1
    for j in 1:k, l in 1:j
        g[cursor] = j == l ? GL[j, l] * L[j, l] : GL[j, l]
        cursor += 1
    end
    return value, g
end

# Quasi-Newton ascent with backtracking Armijo steps. `admissible` rejects a
# trial point before it is evaluated. `screen`, if given, is a value-only form
# of the objective that decides the Armijo test before the gradient is taken.
# A trial point whose evaluation is non-finite or infeasible
# (`_InfeasibleEvaluation`) halves the step; at `x0` either is an error.
# `inverse_hessian`, if given, is a positive-definite initial (and reset)
# inverse curvature in place of the identity. Near the optimum a full step can
# be predicted to gain less than f resolves; as in the EBE solver (`_accepts`),
# such a step is judged by its prediction and f only rejects a drop beyond that
# resolution, instead of halving to the minimum step through rounding noise.
function _bfgs_ascent(objective, x0, maxiter, admissible; screen = nothing,
        gtol = 1.0e-6, inverse_hessian = nothing, what = "SAEM MAP M-step")
    x = copy(x0)
    isempty(x) && return x
    value, gradient = objective(x)
    isfinite(value) && all(isfinite, gradient) || throw(DomainError(value,
        "$what has non-finite objective or gradient"))
    B0 = inverse_hessian === nothing ? Matrix{Float64}(I, length(x), length(x)) :
        Matrix{Float64}(inverse_hessian)
    B = copy(B0)
    for _ in 1:maxiter
        maximum(abs, gradient) < gtol && break
        direction = B * gradient
        if dot(direction, gradient) <= 0
            direction = B0 * gradient
            B = copy(B0)
        end
        step = 1.0
        accepted = false
        while step >= 1.0e-8
            candidate = x + step * direction
            predicted = step * dot(gradient, direction)
            threshold = value + 1.0e-4 * predicted
            if step == 1 && predicted <= _resolution(value)
                threshold = min(threshold, value - _resolution(value))
            end
            screened = admissible(candidate) &&
                (screen === nothing || _armijo(_feasible(screen, candidate), threshold))
            trial = screened ? _feasible(objective, candidate) : nothing
            # A screened trial already passed the Armijo test on its value.
            if trial !== nothing && all(isfinite, last(trial)) &&
                    _armijo(first(trial), screen === nothing ? threshold : -Inf)
                next_value, next_gradient = trial
                s = candidate - x
                y = gradient - next_gradient
                ys = dot(y, s)
                if ys > 1.0e-12
                    ρ = inv(ys)
                    C = Matrix{Float64}(I, length(x), length(x)) - ρ * s * y'
                    B = C * B * C' + ρ * s * s'
                end
                x, value, gradient = candidate, next_value, next_gradient
                accepted = true
                break
            end
            step /= 2
        end
        accepted || break
    end
    return x
end

# Whether a trial value passes the Armijo test; `nothing` is an infeasible trial.
_armijo(value, threshold) = value !== nothing && isfinite(value) && value >= threshold

# The covariance M-step of one constrained family from its mean outer product
# `S̄`: a fully fixed family keeps its value; a partially fixed one maximizes the
# expected Gaussian log density over its free coordinates, starting from
# `current`.
function _fixed_family_update(fam::_FixedFamily, S̄, current, floor)
    fam.fully_fixed && return copy(fam.covariance)
    k = size(S̄, 1)
    block = NLMELayout(0, 0, [k])
    start = _floor_partial_family(fam, omega_parameters(block, current), k, floor)
    function objective(z)
        segment = _family_assemble(fam, z, k)
        L = omega_cholesky(block, segment)
        Linv = inv(L)
        value, g = _gaussian_family_value_gradient(L, Matrix(Linv' * Linv), S̄, 1)
        return value, _family_free_gradient(fam, segment, g, k)
    end
    admissible(z) = _partial_family_admissible(fam, _family_assemble(fam, z, k), k, floor)
    z = _bfgs_ascent(objective, _family_free(fam, start, k), 500, admissible;
        gtol = 1.0e-11, what = "SAEM fixed-Ω family M-step")
    L = omega_cholesky(block, _family_assemble(fam, z, k))
    return Matrix(L * L')
end
