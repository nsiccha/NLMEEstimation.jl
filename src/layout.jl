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
