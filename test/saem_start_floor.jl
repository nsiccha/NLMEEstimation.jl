using Test
using NLMEEstimation
using Random
using LinearAlgebra
import NLMEEstimation as NE

# SAEM starts from ω0 raised to `omega_floor`, as every M-step leaves Ω (snag
# tgi-nlme-point-u-fbe50c66). A start whose family variance a maximum a posteriori
# point collapsed (Bruno's joint TGI start had an SD of 4.4e-12 beside SDs near 0.4)
# gave each NUTS chain a target whose Pathfinder start stalled at once; the draws of
# that fit lay ~1e23 log-density units below the start yet were finite, so 13 of 24
# chains started there and never came back, and the first M-step took Ω from them.

# y_ij ~ N(η₁ + η₂, 0.5²): only the sum is informed, so η₁'s variance stays at the
# floor in a correct fit, and a chain stranded off the start shows in both.
struct CollapsedSum
    y::Matrix{Float64}
end
NE.nsubjects(m::CollapsedSum) = size(m.y, 2)
NE.nlme_layout(::CollapsedSum) = NLMELayout(0, 0, [2])
NE.conditional_loglikelihood(m::CollapsedSum, i, θ, σ, η) =
    -sum(abs2, view(m.y, :, i) .- (η[1] + η[2])) / (2 * 0.25)
function NE.conditional_loglikelihood_and_gradient(m::CollapsedSum, i, θ, σ, η)
    g = sum(view(m.y, :, i) .- (η[1] + η[2])) / 0.25
    return NE.conditional_loglikelihood(m, i, θ, σ, η), [g, g]
end

@testset "SAEM start on omega_floor: the helper" begin
    floor = 1e-8
    layout = NLMELayout(0, 0, [2, 1])
    Ωs = ([1e-24 0.0; 0.0 0.25], fill(0.04, 1, 1))
    ω = omega_parameters(layout, cat(Ωs...; dims = (1, 2)))
    raised, families = NE._saem_floored_start(layout, nothing, ω, floor)
    @test families == [1]
    Ω = omega_matrix(layout, raised)
    @test eigmin(Symmetric(Ω[1:2, 1:2])) ≈ floor rtol = 1e-6
    @test Ω[2, 2] ≈ 0.25
    @test raised[4] == ω[4]                         # family 2 untouched, bit for bit
    @test NE._omega_admissible(layout, nothing, raised, floor * (1 - 1e-6))
    # On the floor up to rounding reads as on it: raising again changes nothing.
    again, none_again = NE._saem_floored_start(layout, nothing, raised, floor)
    @test again === raised && isempty(none_again)

    # A start that holds the floor is returned as it is: those fits are unchanged.
    above = omega_parameters(layout, cat([0.04 0.01; 0.01 0.25], fill(0.09, 1, 1); dims = (1, 2)))
    same, none = NE._saem_floored_start(layout, nothing, above, floor)
    @test same === above && isempty(none)

    # Fixed parts are kept; only estimated ones are raised.
    single = NLMELayout(0, 0, [2])
    partial = NE._omega_constraints(single, [1 => FixedOmega(sd = [nothing, 0.5])])
    ωp = omega_parameters(single, [1e-24 0.0; 0.0 0.25])
    raisedp, familiesp = NE._saem_floored_start(single, partial, ωp, floor)
    @test familiesp == [1]
    sd = sqrt.(diag(omega_matrix(single, raisedp)))
    @test sd[1] ≈ sqrt(floor) rtol = 1e-8
    @test sd[2] ≈ 0.5 rtol = 1e-12
    fixed = NE._omega_constraints(single, [1 => FixedOmega([1e-24 0.0; 0.0 0.25])])
    kept, nonef = NE._saem_floored_start(single, fixed, ωp, floor)
    @test kept === ωp && isempty(nonef)

    # A ragged layout floors family by family.
    ragged = RaggedNLMELayout(ntheta = 0, nsigma = 0, family_sizes = [1, 2],
        subject_families = [[1, 2], [1, 2, 2]])
    ωr = omega_parameters(ragged, [fill(1e-24, 1, 1), [0.3 0.1; 0.1 0.2]])
    raisedr, familiesr = NE._saem_floored_start(ragged, nothing, ωr, floor)
    @test familiesr == [1]
    @test only(omega_matrix(ragged, raisedr)[1]) ≈ floor rtol = 1e-8
    @test raisedr[2:end] == ωr[2:end]
end

@testset "SAEM start on omega_floor: a collapsed start variance" begin
    rng = MersenneTwister(3)
    n = 12
    model = CollapsedSum((0.5 .* randn(rng, n))' .+ 0.5 .* randn(rng, 4, n))
    layout = nlme_layout(model)
    collapsed = omega_parameters(layout, [1e-24 0.0; 0.0 0.25])
    options = SAEMOptions(iterations = 6, burnin = 3, mcmc_burnin = 5)
    floored = first(NE._saem_floored_start(layout, nothing, collapsed, options.omega_floor))
    kernels = Any[WarmupHMCNUTS(max_tree_depth = 6), StandardSAEMMH()]
    Threads.nthreads() > 1 && push!(kernels, WarmupHMCNUTS(max_tree_depth = 6, threads = true))
    for kernel in kernels
        fit = @test_logs (:warn, r"ω0 lies below omega_floor = 1.0e-8 in covariance family 1") match_mode = :any saem(
            model, Float64[], Float64[], collapsed; rng = MersenneTwister(1), options, kernel)
        # The same fit as from the floored start given explicitly.
        same = saem(model, Float64[], Float64[], floored; rng = MersenneTwister(1), options, kernel)
        @test fit.omega == same.omega && fit.eta == same.eta && fit.iterations == same.iterations
        # Before the floor, NUTS (serial and on threads) ended this fit at
        # Ω = [0.199 3.08; 3.08 74.2] with 33 divergences and |η| up to 21.
        @test fit.omega_matrix[1, 1] < 1e-6
        @test 0.05 < fit.omega_matrix[2, 2] < 1.0
        @test maximum(abs, fit.eta) < 3
        kernel isa WarmupHMCNUTS && @test fit.nuts_divergences <= 3
        # The fit's Ω holds the floor (η₁'s variance just above it): a warm start
        # from it stays as it is.
        warm, none_warm = NE._saem_floored_start(layout, nothing, fit.omega, options.omega_floor)
        @test warm === fit.omega && isempty(none_warm)
    end
end
