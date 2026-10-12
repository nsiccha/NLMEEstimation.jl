"""
    NLMELayout(; ntheta, nsigma, eta_blocks)

Parameter layout of an NLME model, returned by [`nlme_layout`](@ref).

- `ntheta`: length of θ, the population (fixed-effect) coordinates.
- `nsigma`: length of σ, the other scalar coordinates the conditional density
  depends on (residual-error parameters, for example).
- `eta_blocks`: sizes of the mutually independent blocks of the per-subject
  random-effect vector η_i, in η order. `sum(eta_blocks)` is the η dimension
  (`neta`). A diagonal Ω is `ones(Int, neta)`; one dense Ω is `[neta]`.

θ and σ are in the model's own *unconstrained* coordinates: every real vector of
the right length is a valid input to [`conditional_loglikelihood`](@ref).

Ω is owned by this package. Its unconstrained parameter vector ω is the
concatenation, block by block, of the log-Cholesky coordinates of each block
`Ω_b = L_b L_bᵀ`: the entries of the lower triangle of `L_b` in row-major order
(`L[1,1], L[2,1], L[2,2], L[3,1], …`), with each diagonal entry stored as its
logarithm. For a block of size one, the coordinate is `log(sd)`.
[`nomega`](@ref) gives the length of ω.
"""
struct NLMELayout
    ntheta::Int
    nsigma::Int
    eta_blocks::Vector{Int}
    function NLMELayout(ntheta::Integer, nsigma::Integer, eta_blocks::AbstractVector{<:Integer})
        ntheta >= 0 || throw(ArgumentError("NLMELayout: ntheta must be ≥ 0, got $ntheta"))
        nsigma >= 0 || throw(ArgumentError("NLMELayout: nsigma must be ≥ 0, got $nsigma"))
        isempty(eta_blocks) &&
            throw(ArgumentError("NLMELayout: eta_blocks is empty; an NLME model needs at least one random effect"))
        all(>=(1), eta_blocks) ||
            throw(ArgumentError("NLMELayout: every η block needs size ≥ 1, got $eta_blocks"))
        return new(Int(ntheta), Int(nsigma), collect(Int, eta_blocks))
    end
end

"""
    RaggedNLMELayout(; ntheta, nsigma, family_sizes, subject_families)

A nested random-effect layout with one shared covariance per family. `family_sizes[f]`
is the dimension of family `f`; `subject_families[i]` lists the family of each
successive block in subject `i`'s η vector. A family may occur more than once in
a subject (for example, once per lesion). No absent block is padded into η.
The model's subject order must agree with `subject_families`.
"""
struct RaggedNLMELayout
    ntheta::Int
    nsigma::Int
    family_sizes::Vector{Int}
    subject_families::Vector{Vector{Int}}
    function RaggedNLMELayout(ntheta::Integer, nsigma::Integer,
            family_sizes::AbstractVector{<:Integer}, subject_families::AbstractVector{<:AbstractVector{<:Integer}})
        ntheta >= 0 || throw(ArgumentError("RaggedNLMELayout: ntheta must be nonnegative"))
        nsigma >= 0 || throw(ArgumentError("RaggedNLMELayout: nsigma must be nonnegative"))
        !isempty(family_sizes) && all(>=(1), family_sizes) || throw(ArgumentError(
            "RaggedNLMELayout: family sizes must be positive and nonempty"))
        !isempty(subject_families) || throw(ArgumentError(
            "RaggedNLMELayout: at least one subject is required"))
        nf = length(family_sizes)
        all(!isempty, subject_families) || throw(ArgumentError(
            "RaggedNLMELayout: each subject needs at least one random-effect block"))
        all(fs -> all(f -> 1 <= f <= nf, fs), subject_families) || throw(ArgumentError(
            "RaggedNLMELayout: subject family index is outside 1:$nf"))
        used = reduce(vcat, subject_families)
        all(f -> f in used, 1:nf) || throw(ArgumentError(
            "RaggedNLMELayout: every covariance family must occur in at least one subject"))
        return new(Int(ntheta), Int(nsigma), collect(Int, family_sizes),
            [collect(Int, fs) for fs in subject_families])
    end
end
RaggedNLMELayout(; ntheta, nsigma, family_sizes, subject_families) =
    RaggedNLMELayout(ntheta, nsigma, family_sizes, subject_families)
Base.:(==)(a::RaggedNLMELayout, b::RaggedNLMELayout) =
    a.ntheta == b.ntheta && a.nsigma == b.nsigma &&
    a.family_sizes == b.family_sizes && a.subject_families == b.subject_families

neta(layout::RaggedNLMELayout, i::Integer) =
    sum(f -> layout.family_sizes[f], layout.subject_families[i]; init=0)
nomega(layout::RaggedNLMELayout) =
    sum(k -> k * (k + 1) ÷ 2, layout.family_sizes)

function _ragged_family_ranges(layout::RaggedNLMELayout)
    ranges = UnitRange{Int}[]
    offset = 0
    for k in layout.family_sizes
        width = k * (k + 1) ÷ 2
        push!(ranges, (offset + 1):(offset + width))
        offset += width
    end
    return ranges
end

function _ragged_subject_blocks(layout::RaggedNLMELayout, i::Integer)
    blocks = Pair{Int,UnitRange{Int}}[]
    offset = 0
    for family in layout.subject_families[i]
        width = layout.family_sizes[family]
        push!(blocks, family => ((offset + 1):(offset + width)))
        offset += width
    end
    return blocks
end

function omega_cholesky(layout::RaggedNLMELayout, ω::AbstractVector)
    _check_length("ω", ω, nomega(layout))
    return [omega_cholesky(NLMELayout(0, 0, [k]), view(ω, r))
        for (k, r) in zip(layout.family_sizes, _ragged_family_ranges(layout))]
end

omega_matrix(layout::RaggedNLMELayout, ω::AbstractVector) =
    [Matrix(L * L') for L in omega_cholesky(layout, ω)]

function omega_parameters(layout::RaggedNLMELayout, Ω::AbstractVector{<:AbstractMatrix})
    length(Ω) == length(layout.family_sizes) || throw(DimensionMismatch(
        "Ω needs one covariance matrix per family"))
    return reduce(vcat, [omega_parameters(NLMELayout(0, 0, [k]), matrix)
        for (k, matrix) in zip(layout.family_sizes, Ω)])
end
NLMELayout(; ntheta::Integer, nsigma::Integer, eta_blocks::AbstractVector{<:Integer}) =
    NLMELayout(ntheta, nsigma, eta_blocks)

Base.:(==)(a::NLMELayout, b::NLMELayout) =
    a.ntheta == b.ntheta && a.nsigma == b.nsigma && a.eta_blocks == b.eta_blocks

"""
    neta(layout::NLMELayout) -> Int

Dimension of the per-subject random-effect vector η_i.
"""
neta(layout::NLMELayout) = sum(layout.eta_blocks)

"""
    nomega(layout::NLMELayout) -> Int

Length of the package-owned Ω parameter vector ω (see [`NLMELayout`](@ref)).
"""
nomega(layout::NLMELayout) = sum(k -> k * (k + 1) ÷ 2, layout.eta_blocks)

# Index ranges of each η block, and of each block's log-Cholesky coordinates in ω.
function _block_ranges(layout::NLMELayout)
    eta_ranges = UnitRange{Int}[]
    omega_ranges = UnitRange{Int}[]
    e = 0
    o = 0
    for k in layout.eta_blocks
        push!(eta_ranges, (e + 1):(e + k))
        m = k * (k + 1) ÷ 2
        push!(omega_ranges, (o + 1):(o + m))
        e += k
        o += m
    end
    return eta_ranges, omega_ranges
end

function _check_length(what, x, n)
    length(x) == n || throw(DimensionMismatch("$what has length $(length(x)), the layout requires $n"))
    return nothing
end

"""
    omega_cholesky(layout, ω) -> LowerTriangular

Block-diagonal lower-triangular Cholesky factor `L` of `Ω = L Lᵀ`, built from the
unconstrained Ω parameters ω (see [`NLMELayout`](@ref)). Generic in the element
type of ω, so it can be differentiated.
"""
function omega_cholesky(layout::NLMELayout, ω::AbstractVector)
    _check_length("ω", ω, nomega(layout))
    T = float(eltype(ω))
    n = neta(layout)
    L = zeros(T, n, n)
    eta_ranges, omega_ranges = _block_ranges(layout)
    for (er, or) in zip(eta_ranges, omega_ranges)
        c = first(or)
        for j in 1:length(er), l in 1:j
            L[er[j], er[l]] = j == l ? exp(ω[c]) : ω[c]
            c += 1
        end
    end
    return LowerTriangular(L)
end

"""
    omega_matrix(layout, ω) -> Matrix

The random-effect covariance `Ω = L Lᵀ` for the unconstrained Ω parameters ω.
"""
function omega_matrix(layout::NLMELayout, ω::AbstractVector)
    L = omega_cholesky(layout, ω)
    return Matrix(L * L')
end

"""
    omega_parameters(layout, Ω::AbstractMatrix) -> Vector{Float64}

Inverse of [`omega_matrix`](@ref): the unconstrained Ω parameters ω of a covariance
matrix Ω. Throws if Ω has nonzero entries outside the layout's η blocks or if a
block is not positive definite.
"""
function omega_parameters(layout::NLMELayout, Ω::AbstractMatrix)
    n = neta(layout)
    size(Ω) == (n, n) || throw(DimensionMismatch("Ω has size $(size(Ω)), the layout requires ($n, $n)"))
    eta_ranges, omega_ranges = _block_ranges(layout)
    block_of = zeros(Int, n)
    for (b, er) in enumerate(eta_ranges)
        block_of[er] .= b
    end
    for j in 1:n, l in 1:n
        block_of[j] != block_of[l] && !iszero(Ω[j, l]) && throw(ArgumentError(
            "omega_parameters: Ω[$j, $l] = $(Ω[j, l]) couples η components in different blocks of $(layout.eta_blocks)"))
    end
    ω = zeros(float(eltype(Ω)), nomega(layout))
    for (er, or) in zip(eta_ranges, omega_ranges)
        block = Symmetric(Matrix(Ω[er, er]))
        ch = cholesky(block; check = false)
        issuccess(ch) || throw(ArgumentError(
            "omega_parameters: the Ω block for η components $er is not positive definite"))
        L = ch.L
        c = first(or)
        for j in 1:length(er), l in 1:j
            ω[c] = j == l ? log(L[j, l]) : L[j, l]
            c += 1
        end
    end
    return ω
end
