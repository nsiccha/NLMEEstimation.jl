using NLMEEstimation
using LinearAlgebra
using Random
using SparseArrays
using Test
import ForwardDiff
import NLMEEstimation as NE
import Optim
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon: occasions_model, occasions_unit_posterior, occasions_exact_nested,
    occasions_pattern, occasions_sparse, SparseDeclared, SteepWell

# The exact mode of an occasions subject: for this linear-Gaussian model, the
# posterior means of its subject effect and of its occasion effects.
function occasions_exact_mode(m, i, x)
    θ, s2, a2, b2 = x[1], exp(2x[2]), exp(2x[3]), exp(2x[4])
    counts = length.(m.y[i])
    means = [sum(o) / length(o) for o in m.y[i]]
    w = counts ./ (s2 .+ counts .* b2)
    va = inv(inv(a2) + sum(w))
    return vcat(va * sum(w .* (means .- θ)), first(occasions_unit_posterior(m, i, x)))
end

@testset "limited-memory BFGS modes of subjects with hundreds of nested effects" begin
    # Subjects of 150, 300 and 600 occasions (151 to 601 η). The subject effect's
    # curvature is hundreds of times the occasions' in prior standard deviations,
    # and the BFGS approximation keeps only `memory` pairs (20 by default, 10 here as
    # well, fewer than the iterations it takes), yet every subject reaches its exact
    # mode. Over 30 seeds of this design, memory 10 took at most 88 iterations and
    # memory 20 at most 49 (maxiter 200). Memory 3 is too little: it ran out of
    # iterations on some seeds, and on macOS CI on this seed.
    model = occasions_model(seed = 3, units = [150, 300, 600])
    x = [2.0, log(0.25), log(0.6), log(0.4)]
    for method in (QuasiNewtonEBE(), QuasiNewtonEBE(memory = 10))
        r = empirical_bayes(model, x[1:1], x[2:2], x[3:4]; method)
        @test r.method === method
        @test all(r.converged)
        @test all(<=(EBEOptions().gtol), r.gradient_norm)
        for (k, i) in enumerate(r.subjects)
            @test r.modes[k] ≈ occasions_exact_mode(model, i, x) atol = 1e-8
        end
    end
    # refused: limited-memory BFGS stores at least one curvature pair
    @test_throws ArgumentError QuasiNewtonEBE(memory = 0)
end

# The occasions model's complete-data η Hessian, in closed form.
function occasions_exact_hessian(m, i, x)
    s2, a2, b2 = exp(2x[2]), exp(2x[3]), exp(2x[4])
    counts = length.(m.y[i])
    n = length(counts) + 1
    H = zeros(n, n)
    H[1, 1] = -sum(counts) / s2 - 1 / a2
    for (j, c) in enumerate(counts)
        H[1, j + 1] = H[j + 1, 1] = -c / s2
        H[j + 1, j + 1] = -c / s2 - 1 / b2
    end
    return H
end

@testset "sparse η Hessians from a declared pattern" begin
    # The occasions model declares its arrow pattern (the subject effect shares
    # observations with every occasion effect, occasion effects share none). Two
    # colors cover it however many occasions a subject has, so the sparse Hessians
    # cost four batched gradient calls.
    inner = occasions_model(seed = 3, units = [150, 300, 600])
    model = occasions_sparse(inner)
    x = [2.0, log(0.25), log(0.6), log(0.4)]
    prior = re_prior(model, x[3:4])
    H = [occasions_exact_mode(inner, i, x) for i in 1:3]
    inner.calls[] = 0
    hessians = NE._complete_hessians(model, [1, 2, 3], x[1:1], x[2:2], prior, H)
    @test inner.calls[] == 4 * 3
    @test all(h -> h isa SparseMatrixCSC, hessians)
    for i in 1:3
        exact = occasions_exact_hessian(inner, i, x)
        @test hessians[i] ≈ exact rtol = 1e-9
        @test SparseArrays.nnz(hessians[i]) == 3 * size(exact, 1) - 2
    end
    # A pattern is colored once (the greedy star coloring is quadratic in time on
    # nested patterns): an equal pattern, rebuilt, gets the cached coloring, and a
    # different one its own.
    P = NE._eta_hessian_pattern(model, 2, 301)
    coloring = NE._eta_hessian_coloring(P)
    @test NE._eta_hessian_coloring(NE._eta_hessian_pattern(model, 2, 301)) === coloring
    @test NE._eta_hessian_coloring(NE._eta_hessian_pattern(model, 1, 151)) !== coloring
    @test NE.SMC.column_colors(coloring) ==
        NE.SMC.column_colors(NE.SMC.coloring(P, NE._STAR_COLORING...))
    # On small subjects the sparse Hessian equals the dense difference Hessian.
    small = occasions_model()
    points = [zeros(1 + length(o)) .+ 0.1 for o in small.y]
    sparse_blocks = NE._complete_hessians(occasions_sparse(small), collect(1:8), x[1:1], x[2:2],
        re_prior(small, x[3:4]), points)
    dense_blocks = NE._difference_eta_hessians(small, nlme_layout(small), collect(1:8), x[1:1],
        x[2:2], re_prior(small, x[3:4]), points)
    @test all(Matrix(s) ≈ d for (s, d) in zip(sparse_blocks, dense_blocks))
end

@testset "empirical Bayes on sparse Hessians" begin
    # A model that declares its pattern gets Newton by default, on sparse Hessians
    # computed from its gradients, and the result reports them.
    inner = occasions_model(seed = 3, units = [150, 300, 600])
    model = occasions_sparse(inner)
    x = [2.0, log(0.25), log(0.6), log(0.4)]
    @test default_ebe_method(model) === NewtonEBE()
    for method in (NewtonEBE(), QuasiNewtonEBE())
        r = empirical_bayes(model, x[1:1], x[2:2], x[3:4]; method)
        @test all(r.converged)
        @test r.hessians isa Vector{<:SparseMatrixCSC}
        for (k, i) in enumerate(r.subjects)
            @test r.modes[k] ≈ occasions_exact_mode(inner, i, x) atol = 1e-8
            @test r.hessians[k] ≈ occasions_exact_hessian(inner, i, x) rtol = 1e-8
        end
        @test ebe_covariances(r)[1] ≈ inv(-occasions_exact_hessian(inner, r.subjects[1], x)) rtol = 1e-6
    end
end

@testset "the sparse trust-region step equals the dense one" begin
    # Random indefinite sparse Hessians in a ragged subject's prior metric, and the
    # hard case: a zero gradient at a saddle, where the step must go along the most
    # upward direction (the dense method's first eigenvector; either sign increases
    # the model equally, so the comparison is of the increase and the length).
    inner = occasions_model(seed = 3, units = [150])
    p = NE._subject_prior(re_prior(inner, [log(0.6), log(0.4)]), 1)
    rng = Xoshiro(1)
    for trial in 1:6
        B = sprandn(rng, 151, 151, 0.02)
        Hk = sparse(-(B + B') / 2 + (trial == 6 ? -0.5I : 0.5I))
        g = trial == 5 ? zeros(151) : randn(rng, 151)
        for Δ in (0.3, 3.0)
            s = NE._trust_region_trial(Hk, p, g, Δ)
            d = NE._trust_region_trial(Matrix(Hk), p, g, Δ)
            @test s.full == d.full
            @test s.predicted ≈ d.predicted rtol = 1e-8
            @test s.length ≈ d.length rtol = 1e-8
            trial == 5 || @test s.step ≈ d.step rtol = 1e-6
        end
    end
end

@testset "sparse Hessians leave saddle points like dense ones" begin
    # The double well of `test/empirical_bayes.jl`, declared with a diagonal pattern
    # and so given sparse difference Hessians: Newton and BFGS leave the exact saddle
    # start (subjects 1 and 2) and reach the same modes as with the model's own blocks.
    model = SteepWell(NLMELayout(0, 0, [1, 1]), 1.0e4, 1.5, [0.0, 0.0, 1.0e-6, 1.0e-4, 1.0e-2], 0.3)
    declared = SparseDeclared(model, i -> sparse(I, 2, 2))
    ω = omega_parameters(nlme_layout(model), Matrix(1.0I, 2, 2))
    η0 = [0.0 0.3 0.0 0.0 0.0; 0.0 0.0 1.0e-6 1.0e-6 1.0e-6]
    dense = empirical_bayes(model, Float64[], Float64[], ω; η0, method = NewtonEBE())
    for method in (NewtonEBE(), QuasiNewtonEBE())
        r = empirical_bayes(declared, Float64[], Float64[], ω; η0, method)
        @test all(r.converged)
        @test r.hessians isa Vector{<:SparseMatrixCSC}
        @test abs.(r.modes) ≈ abs.(dense.modes) rtol = 1e-8
        @test all(>(0.7), abs.(r.modes[2, :]))
    end
end

@testset "marginal likelihood and covariance step on sparse Hessians" begin
    # The occasions posterior is Gaussian, so Laplace (one Gauss–Hermite node) and a
    # Gaussian importance-sampling proposal are exact at any subject size; the
    # proposal draws go through the sparse Cholesky factor.
    inner = occasions_model(seed = 3, units = [150, 300, 600])
    model = occasions_sparse(inner)
    x = [2.0, log(0.25), log(0.6), log(0.4)]
    exact = occasions_exact_nested(inner, x)
    laplace = marginal_loglikelihood(model, x[1:1], x[2:2], x[3:4];
        approximation = AdaptiveGaussHermite(nodes = 1))
    @test laplace.loglikelihood ≈ exact atol = 1e-8
    draws = marginal_loglikelihood(model, x[1:1], x[2:2], x[3:4];
        approximation = ImportanceSampling(draws = 20, df = Inf), rng = Xoshiro(2))
    @test draws.loglikelihood ≈ exact atol = 1e-8
    @test all(≈(20), draws.effective_sample_size)
    # The covariance step's R is minus the Hessian of the exact marginal.
    small = occasions_model(seed = 4, units = [20, 40, 60, 30])
    x̂ = Optim.minimizer(Optim.optimize(y -> -occasions_exact_nested(small, y),
        y -> ForwardDiff.gradient(z -> -occasions_exact_nested(small, z), y), x,
        Optim.LBFGS(); inplace = false))
    step = covariance_step(occasions_sparse(small), x̂[1:1], x̂[2:2], x̂[3:4];
        approximation = AdaptiveGaussHermite(nodes = 1), method = :r)
    exact_R = -ForwardDiff.hessian(y -> occasions_exact_nested(small, y), x̂)
    @test step.R ≈ exact_R rtol = 1e-4
end

@testset "check_protocol checks a declared η Hessian pattern" begin
    inner = occasions_model()
    x = [2.0, log(0.25), log(0.6), log(0.4)]
    @test check_protocol(occasions_sparse(inner), x[1:1], x[2:2], x[3:4])
    # refused: a pattern that omits the subject-occasion couplings would make the
    # sparse Hessian wrong (decision 1qwpx6g: the declaration is the contract)
    diagonal = SparseDeclared(inner, i -> sparse(I, 1 + length(inner.y[i]), 1 + length(inner.y[i])))
    @test_throws ProtocolViolation check_protocol(diagonal, x[1:1], x[2:2], x[3:4])
    # refused: a pattern must have the subject's size
    wrong_size = SparseDeclared(inner, i -> sparse(I, 2, 2))
    @test_throws ProtocolViolation check_protocol(wrong_size, x[1:1], x[2:2], x[3:4])
    # refused: patterns are declared for every subject or for none
    partial = SparseDeclared(inner, i -> i == 1 ? occasions_pattern(inner, 1) : nothing)
    @test_throws ProtocolViolation check_protocol(partial, x[1:1], x[2:2], x[3:4])
end

@testset "adaptive Gauss–Hermite counts its points without overflow" begin
    # refused: 2^70 points exceed max_points; the Int count used to wrap to 0
    @test_throws ArgumentError NE._marginal_points(AdaptiveGaussHermite(nodes = 2), Xoshiro(1), 70, 0.0)
    Z, offset = NE._marginal_points(AdaptiveGaussHermite(nodes = 1), Xoshiro(1), 25_000, 0.0)
    @test size(Z) == (25_000, 1) && iszero(Z)
    @test only(offset) ≈ 25_000 * log(2π) / 2
end
