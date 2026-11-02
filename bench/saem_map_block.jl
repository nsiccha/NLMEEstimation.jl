# SAEM's MAP Gaussian block (the mu-referenced θ and Ω under a population
# prior): population-prior calls per M-step, its wall time, and how far each
# M-step's result is from the block's maximum, for the 0.2.7 block and the 0.2.6
# one (copied below as `legacy_map_block`), snag `saem-population-1ff48ec8`.
#
# Run from the repository root with the test environment:
#   julia --project=test bench/saem_map_block.jl [seeds] [output]
#
# Model: three subjects, four random effects with a dense Ω, seven
# mu-referenced θ of which two (standardized covariates on the first effect)
# are exactly collinear, so the designs leave one combination unidentified, one
# structural θ (a covariate coefficient without a random effect, so
# `numeric_mstep = :auto` resolves to `:score`), one residual SD, and a Gaussian
# population prior. The shape of a 3-subject joint PK/QT BRM model on a small
# scale: there d = 39 (11 mu-referenced θ, a 7 × 7 Ω), here d = 17.
#
# Reference: the block's objective (the expected Gaussian random-effect density
# plus the prior) maximized by Optim's LBFGS with ForwardDiff gradients over
# μ and a square L with Ω = floor·I + L Lᵀ, which holds the floor for every L,
# from the block's start and its result, each with floored eigenvalues raised by
# 1e-6 and by 1e-3; the gap is the best of those maxima minus the value the
# block returned. The odd M-steps of the exploration phase (γ = 1, 1 to 9) and
# every fourth of the SA phase (21 to 37) are scored. (A triangular L can hold a
# floored direction at a coordinate axis, so it is not a reference.)
using NLMEEstimation
import NLMEEstimation as NE
using LinearAlgebra
using Printf
using Random
using Statistics
import Optim
import ForwardDiff
using ADTypes: AutoForwardDiff

struct MapProbe{Legacy}
    t::Vector{Vector{Float64}}
    x::Vector{Vector{Float64}}
    w::Matrix{Float64}
    y::Vector{Vector{Float64}}
    calls::Dict{Symbol, Int}
end

map_probe_design(w, i) = [1.0 w[i, 1] w[i, 2] w[i, 3] 0 0 0;
                          0 0 0 0 1.0 0 0;
                          0 0 0 0 0 1.0 0;
                          0 0 0 0 0 0 1.0]
map_probe_basis(t) = hcat(ones(length(t)), t, sin.(t), cos.(t))

function map_probe(legacy::Bool; seed = 5, n = 8)
    rng = Xoshiro(seed)
    standardize(v) = (v .- mean(v)) ./ std(v)
    w = hcat([0.0, 1.0, 1.0], standardize([2.0, 5.0, 3.0]), standardize([8.0, 5.0, 7.0]))
    t = [collect(range(0.25, 4.0; length = n)) for _ in 1:3]
    x = [rand(rng, n) for _ in 1:3]
    θtrue = [1.0, 0.3, -0.2, 0.1, 0.5, -0.4, 0.2, 0.7]
    y = map(1:3) do i
        φ = map_probe_design(w, i) * θtrue[1:7] .+ [0.4, 0.2, 0.3, 0.25] .* randn(rng, 4)
        map_probe_basis(t[i]) * φ .+ θtrue[8] .* x[i] .+ 0.1 .* randn(rng, n)
    end
    return MapProbe{legacy}(t, x, w, y, Dict{Symbol, Int}())
end
count!(m, k) = (m.calls[k] = get(m.calls, k, 0) + 1)

NE.nsubjects(m::MapProbe) = length(m.y)
NE.nlme_layout(::MapProbe) = NLMELayout(8, 1, [4])
NE.mu_referencing(m::MapProbe) = MuReferencing(collect(1:7),
    [map_probe_design(m.w, i) for i in 1:3])
function map_probe_residuals(m, i, θ, η)
    φ = map_probe_design(m.w, i) * θ[1:7] .+ η
    return m.y[i] .- map_probe_basis(m.t[i]) * φ .- θ[8] .* m.x[i]
end
function NE.conditional_loglikelihood(m::MapProbe, i::Integer, θ, σ, η)
    r = map_probe_residuals(m, i, θ, η)
    return -sum(abs2, r) / (2exp(2σ[1])) - length(r) * (log(2π) / 2 + σ[1])
end
function NE.conditional_loglikelihood_and_gradient(m::MapProbe, i::Integer, θ, σ, η)
    r = map_probe_residuals(m, i, θ, η)
    return NE.conditional_loglikelihood(m, i, θ, σ, η),
        map_probe_basis(m.t[i])' * r ./ exp(2σ[1])
end
function NE.conditional_loglikelihood_and_population_gradient(m::MapProbe, i::Integer,
        θ, σ, η)
    r = map_probe_residuals(m, i, θ, η)
    v = exp(2σ[1])
    gφ = map_probe_basis(m.t[i])' * r ./ v
    return NE.conditional_loglikelihood(m, i, θ, σ, η),
        vcat(map_probe_design(m.w, i)' * gφ, dot(r, m.x[i]) / v), [sum(abs2, r) / v - length(r)]
end
map_probe_logprior(θ, σ, ω) = -sum(abs2, vcat(θ, σ, ω)) / 8
function NE.population_logprior(m::MapProbe, θ, σ, ω)
    count!(m, :value)
    return map_probe_logprior(θ, σ, ω)
end
function NE.population_logprior_and_gradient(m::MapProbe, θ, σ, ω)
    count!(m, :gradient)
    return map_probe_logprior(θ, σ, ω), -θ ./ 4, -σ ./ 4, -ω ./ 4
end

# NLMEEstimation 0.2.6's block: one BFGS search on the objective itself from the
# closed-form start, rejecting every trial below the Ω floor, with a new
# difference curvature at every M-step (of the whole objective, two prior
# gradients per coordinate, where the designs leave a combination unidentified).
function legacy_map_block(model, layout, θ, σ, ω, indices, design, mean_phi,
        second_phi, subjects, floor, constraints)
    identified = NE._saem_identified(design, indices)
    nμ = length(indices)
    function point(z)
        θz = copy(θ)
        θz[indices] = view(z, 1:nμ)
        return θz, Vector{Float64}(NE._constrained_omega(constraints, view(z, nμ+1:length(z))))
    end
    function gaussian(z)
        θz, ωz = point(z)
        gv, gμ, gω = NE._saem_gaussian_map_value_gradient(layout, θz, ωz, indices,
            design, mean_phi, second_phi, subjects)
        return gv, vcat(gμ, NE._free_omega_gradient(constraints, ωz, gω))
    end
    function objective(z)
        θz, ωz = point(z)
        gv, gradient = gaussian(z)
        lp, gθ, _, gω = NE._saem_prior_terms(model, θz, σ, ωz)
        return gv + lp, gradient + vcat(gθ[indices],
            NE._free_omega_gradient(constraints, ωz, gω))
    end
    function screen(z)
        θz, ωz = point(z)
        return first(gaussian(z)) + NE._saem_prior_value(model, θz, σ, ωz)
    end
    admissible(z) = NE._omega_admissible(layout, constraints, last(point(z)), floor)
    z0 = vcat(θ[indices], NE._free_omega(constraints, ω))
    C = NE._saem_difference_curvature(identified === nothing ? gaussian : objective, z0)
    F = C === nothing ? nothing : cholesky(C; check = false)
    H = F === nothing || !issuccess(F) ? nothing : inv(F)
    z = NE._bfgs_ascent(objective, z0, 100, admissible; screen, inverse_hessian = H)
    return point(z)
end

# Every M-step's block call, recorded: prior calls, wall time and what the
# reference needs.
const RECORDS = Any[]
function NE._saem_gaussian_map_block(model::MapProbe{Legacy}, layout, θ, σ, ω, indices,
        design, mean_phi, second_phi, subjects, floor, constraints;
        kwargs...) where {Legacy}
    g0, v0 = get(model.calls, :gradient, 0), get(model.calls, :value, 0)
    t = time_ns()
    θn, ωn = Legacy ?
        legacy_map_block(model, layout, θ, σ, ω, indices, design, mean_phi, second_phi,
            subjects, floor, constraints) :
        invoke(NE._saem_gaussian_map_block, NTuple{12, Any}, model, layout, θ, σ, ω,
            indices, design, mean_phi, second_phi, subjects, floor, constraints; kwargs...)
    push!(RECORDS, (; seconds = (time_ns() - t) / 1e9,
        gradients = get(model.calls, :gradient, 0) - g0,
        values = get(model.calls, :value, 0) - v0,
        layout, θ = copy(θ), σ = copy(σ), indices = copy(indices), design,
        mean_phi = copy(mean_phi), second_phi = copy(second_phi), floor,
        start = vcat(θ[indices], ω), result = vcat(θn[indices], ωn)))
    return θn, ωn
end

function block_objective(r, μ, Ω)
    m, n = size(r.mean_phi)
    θ = convert(Vector{eltype(μ)}, copy(r.θ))
    θ[r.indices] = μ
    S = zeros(eltype(μ), m, m)
    for k in 1:n
        xμ = r.design[k] * μ
        φ = r.mean_phi[:, k]
        S += r.second_phi[:, :, k] - φ * xμ' - xμ * φ' + xμ * xμ'
    end
    C = cholesky(Symmetric(Ω)).L
    gaussian = -(n * (m * log(2π) + 2sum(log, diag(C))) + sum(inv(Symmetric(Ω)) .* S)) / 2
    return gaussian + map_probe_logprior(θ, r.σ, NE.omega_parameters(r.layout, Matrix(Ω)))
end

function reference_gap(r)
    m, nμ = size(r.mean_phi, 1), length(r.indices)
    unpack(x) = (x[1:nμ], r.floor * I + reshape(x[nμ+1:end], m, m) * reshape(x[nμ+1:end], m, m)')
    value(z) = block_objective(r, z[1:nμ], NE.omega_matrix(r.layout, z[nμ+1:end]))
    best = -Inf
    for z in (r.start, r.result), jitter in (1e-6, 1e-3)
        E = eigen(Symmetric(NE.omega_matrix(r.layout, z[nμ+1:end]) - r.floor * I))
        L0 = cholesky(Symmetric(E.vectors * Diagonal(max.(E.values, jitter)) * E.vectors')).L
        x0 = vcat(z[1:nμ], vec(Matrix(L0)))
        fit = Optim.optimize(x -> -block_objective(r, unpack(x)...), x0, Optim.LBFGS(),
            Optim.Options(g_tol = 1e-10, iterations = 10_000); autodiff = AutoForwardDiff())
        best = max(best, -Optim.minimum(fit))
    end
    return best - value(r.result), r.start == r.result
end

function main(seeds, output)
    N = nullspace(reduce(vcat, NE.mu_referencing(map_probe(false)).design))
    θ0 = zeros(8)
    θ0[1:7] .+= 0.8 .* vec(N)   # the unidentified combination off the prior mode
    σ0, ω0 = [log(0.3)], NE.omega_parameters(NLMELayout(8, 1, [4]), Matrix(0.25I, 4, 4))
    io = output === nothing ? stdout : open(output, "w")
    println(io, "SAEM MAP Gaussian block benchmark: $(length(seeds)) seeds, Julia $(VERSION), ",
        "$(Threads.nthreads()) threads, NLMEEstimation $(pkgversion(NLMEEstimation))")
    println(io, "per M-step of SA iterations 21-40 (fit iterations 21-40 of 40, burn-in 10): ",
        "prior gradients and values inside the block, median block milliseconds; largest ",
        "gap to the reference over M-steps 1:2:9 (exploration) and 21:4:37 (SA); M-steps ",
        "that returned their start")
    println(io, "block\tkernel\tfloor\tseed\tfirst M-step gradients\tgradients\tvalues\t",
        "block ms\texploration max gap\tSA max gap\tno-op M-steps")
    for legacy in (true, false), (kname, kernel) in (("mh", StandardSAEMMH()),
            ("nuts", WarmupHMCNUTS())), floor in (0.05, 1e-8), seed in seeds
        empty!(RECORDS)
        saem(map_probe(legacy), θ0, σ0, ω0; rng = Xoshiro(seed), kernel,
            options = SAEMOptions(; iterations = 40, burnin = 10, omega_floor = floor))
        records = copy(RECORDS)
        sa = records[21:40]
        exploration = [first(reference_gap(records[k])) for k in 1:2:9]
        approximation = [first(reference_gap(records[k])) for k in 21:4:37]
        @printf(io, "%s\t%s\t%g\t%d\t%d\t%.2f\t%.2f\t%.3f\t%.2e\t%.2e\t%d/40\n",
            legacy ? "0.2.6" : "0.2.7", kname, floor, seed, records[1].gradients,
            mean(r.gradients for r in sa), mean(r.values for r in sa),
            1e3median(r.seconds for r in sa), maximum(exploration), maximum(approximation),
            count(r -> r.start == r.result, records))
        flush(io)
    end
    output === nothing || close(io)
end

seeds = length(ARGS) >= 1 ? (1:parse(Int, ARGS[1])) : (1:3)
main(seeds, length(ARGS) >= 2 ? ARGS[2] : nothing)
