using NLMEEstimation
import NLMEEstimation as NE
using LinearAlgebra
using Random
using Test
import ForwardDiff
import Optim

# Empirical Bayes modes on kinks of a piecewise-smooth complete-data density
# (NLMEEstimation 0.2.10, snag `empirical-bayes-8e0ba5a2`): there the gradient does
# not vanish and the smooth solvers stop with no step accepted.

# ℓ_i(η) = -c|y_i - η₁|, one Laplace-distributed observation per subject; with the
# unit prior the mode is the kink η₁ = y_i whenever |y_i| < c. `calls` counts
# per-subject evaluations, `batches` the subjects in each batched gradient call.
struct LaplaceObservation
    y::Vector{Float64}
    c::Float64
    hessian::Bool
end
NE.nsubjects(m::LaplaceObservation) = length(m.y)
NE.nlme_layout(::LaplaceObservation) = NLMELayout(0, 0, [1])
NE.conditional_loglikelihood(m::LaplaceObservation, i, θ, σ, η) = -m.c * abs(m.y[i] - η[1])
NE.conditional_loglikelihood_and_gradient(m::LaplaceObservation, i, θ, σ, η) =
    (NE.conditional_loglikelihood(m, i, θ, σ, η), [m.c * sign(m.y[i] - η[1])])
NE.conditional_loglikelihood_gradient_and_hessian(m::LaplaceObservation, i, θ, σ, η) =
    (NE.conditional_loglikelihood_and_gradient(m, i, θ, σ, η)..., zeros(1, 1))
NE.provides_eta_hessian(m::LaplaceObservation) = m.hessian

# Two effects, ℓ_i(η) = -c|η₁ - η₂²/2 - y_i| + bη₂ - D(η₂ - 1)²/2: the kink is the
# parabola η₁ = η₂²/2 + y_i, and the mode lies on it away from where a search first
# meets it, so a point where the smooth search stalls still has an ascent direction
# along the kink.
struct KinkedRidge
    y::Vector{Float64}
    c::Float64
    b::Float64
    D::Float64
    hessian::Bool
end
KinkedRidge(y; hessian = true) = KinkedRidge(y, 2.0, 0.5, 1.0, hessian)
NE.nsubjects(m::KinkedRidge) = length(m.y)
NE.nlme_layout(::KinkedRidge) = NLMELayout(0, 0, [2])
ridge(m, i, η) = η[1] - η[2]^2 / 2 - m.y[i]
NE.conditional_loglikelihood(m::KinkedRidge, i, θ, σ, η) =
    -m.c * abs(ridge(m, i, η)) + m.b * η[2] - m.D * (η[2] - 1)^2 / 2
function NE.conditional_loglikelihood_and_gradient(m::KinkedRidge, i, θ, σ, η)
    s = sign(ridge(m, i, η))
    return NE.conditional_loglikelihood(m, i, θ, σ, η), [-m.c * s, m.c * s * η[2] + m.b - m.D * (η[2] - 1)]
end
function NE.conditional_loglikelihood_gradient_and_hessian(m::KinkedRidge, i, θ, σ, η)
    s = sign(ridge(m, i, η))
    return NE.conditional_loglikelihood_and_gradient(m, i, θ, σ, η)..., [0.0 0.0; 0.0 m.c * s - m.D]
end
NE.provides_eta_hessian(m::KinkedRidge) = m.hessian

# The exact mode of a KinkedRidge subject with unit prior variances: on the kink,
# maximizing over η₂ alone.
function ridge_mode(m::KinkedRidge, i)
    along(e) = (e1 = e^2 / 2 + m.y[i]; m.b * e - m.D * (e - 1)^2 / 2 - (e1^2 + e^2) / 2)
    e = Optim.minimizer(Optim.optimize(e -> -along(e), -3.0, 3.0, Optim.Brent(); abs_tol = 1.0e-14))
    return [e^2 / 2 + m.y[i], e]
end

# A batched view of a per-subject model that records how many subjects each call lists.
struct Batched{M}
    inner::M
    batches::Vector{Int}
end
Batched(inner) = Batched(inner, Int[])
NE.evaluation_style(::Batched) = BatchedEvaluation()
NE.nsubjects(m::Batched) = nsubjects(m.inner)
NE.nlme_layout(m::Batched) = nlme_layout(m.inner)
NE.provides_eta_hessian(m::Batched) = provides_eta_hessian(m.inner)
NE.conditional_loglikelihoods(m::Batched, subjects, θ, σ, H) =
    [conditional_loglikelihood(m.inner, i, θ, σ, view(H, :, k)) for (k, i) in enumerate(subjects)]
function NE.conditional_loglikelihoods_and_gradients(m::Batched, subjects, θ, σ, H)
    push!(m.batches, length(subjects))
    parts = [conditional_loglikelihood_and_gradient(m.inner, i, θ, σ, H[:, k]) for (k, i) in enumerate(subjects)]
    return first.(parts), reduce(hcat, last.(parts))
end
NE.conditional_loglikelihood_hessian_blocks(m::Batched, subjects, θ, σ, H) =
    [last(conditional_loglikelihood_gradient_and_hessian(m.inner, i, θ, σ, H[:, k])) for (k, i) in enumerate(subjects)]

# A tumour-response model shaped as Bruno's RECIST category law: tumour size
# S(t) = S₀(e^{gt} + e^{-dt} - 1) with g, d, S₀ log-normal in η, observed log sizes, and
# a category per visit from the change r = S/S₀ - 1 against its running nadir (a
# running minimum): pd = (c_pd + nadir - r)/τ, pr = min((c_pr - r)/τ, pd),
# cr = min((c_cr - r)/τ, pr), an ordered logit with an ε floor.
struct RecistLike
    times::Vector{Float64}
    logsize::Vector{Vector{Float64}}
    category::Vector{Vector{Int}}
end
NE.nsubjects(m::RecistLike) = length(m.logsize)
NE.nlme_layout(::RecistLike) = NLMELayout(0, 0, [3])
logistic_lcdf(x) = x < 0 ? x - log1p(exp(x)) : -log1p(exp(-x))
logaddexp(a, b) = (top = max(a, b); top + log1p(exp(min(a, b) - top)))
function logit_interval(a, b)
    b <= a && return oftype(a, -Inf)
    return a > 0 ? logistic_lcdf(-a) + log1p(-exp(logistic_lcdf(-b) - logistic_lcdf(-a))) :
        logistic_lcdf(b) + log1p(-exp(logistic_lcdf(a) - logistic_lcdf(b)))
end
tumour_sizes(η, times) = (s0 = 50 * exp(η[3]);
    ([s0 * (exp(0.05 * exp(η[1]) * t) + exp(-0.3 * exp(η[2]) * t) - 1) for t in times], s0))
function recist_loglikelihood(m::RecistLike, i, η)
    S, s0 = tumour_sizes(η, m.times)
    lp = zero(eltype(η))
    nadir = zero(eltype(η))
    for (j, v) in enumerate(m.category[i])
        lp += -((log(max(S[j], 1.0e-8)) - m.logsize[i][j]) / 0.3)^2 / 2
        r = S[j] / s0 - 1
        pd = (0.2 + nadir - r) / 0.05
        pr = min((-0.3 - r) / 0.05, pd)
        cr = min((-1.0 - r) / 0.05, pr)
        p = v == 1 ? logit_interval(oftype(cr, -30), cr) : v == 2 ? logit_interval(cr, pr) :
            v == 3 ? logit_interval(pr, pd) : logit_interval(pd, oftype(pd, 30))
        floor = oftype(p, log(1.0e-3) - log(4.0))
        lp += p == -Inf ? floor : logaddexp(floor, log1p(-1.0e-3) + p)
        nadir = min(nadir, r)
    end
    return lp
end
NE.conditional_loglikelihood(m::RecistLike, i, θ, σ, η) = recist_loglikelihood(m, i, η)
NE.conditional_loglikelihood_and_gradient(m::RecistLike, i, θ, σ, η) =
    (recist_loglikelihood(m, i, η), ForwardDiff.gradient(e -> recist_loglikelihood(m, i, e), η))
NE.conditional_loglikelihood_gradient_and_hessian(m::RecistLike, i, θ, σ, η) =
    (NE.conditional_loglikelihood_and_gradient(m, i, θ, σ, η)...,
        ForwardDiff.hessian(e -> recist_loglikelihood(m, i, e), η))
NE.provides_eta_hessian(::RecistLike) = true
function recist_like(rng, n)
    times = collect(6.0:6.0:48.0)
    logsize, category = Vector{Float64}[], Vector{Int}[]
    for _ in 1:n
        η = 0.5 .* randn(rng, 3)
        S, s0 = tumour_sizes(η, times)
        push!(logsize, log.(S) .+ 0.3 .* randn(rng, length(times)))
        nadir = 0.0
        push!(category, map(eachindex(times)) do j
            r = S[j] / s0 - 1 + 0.1 * randn(rng)
            v = r - nadir >= 0.2 ? 4 : r <= -1.0 ? 1 : r <= -0.3 ? 2 : 3
            nadir = min(nadir, S[j] / s0 - 1)
            v
        end)
    end
    return RecistLike(times, logsize, category)
end

# Gradients only, without blocks or a declared pattern.
struct GradientsOnly{M}
    inner::M
end
NE.nsubjects(m::GradientsOnly) = nsubjects(m.inner)
NE.nlme_layout(m::GradientsOnly) = nlme_layout(m.inner)
NE.conditional_loglikelihood(m::GradientsOnly, i, θ, σ, η) = conditional_loglikelihood(m.inner, i, θ, σ, η)
NE.conditional_loglikelihood_and_gradient(m::GradientsOnly, i, θ, σ, η) =
    conditional_loglikelihood_and_gradient(m.inner, i, θ, σ, η)
NE.provides_eta_hessian(::GradientsOnly) = false

# A Gaussian observation whose mean has a kink: y_i ~ N(-s|η₁|, 1), y_i > 0, so the
# complete-data density's mode is the kink η₁ = 0.
struct KinkedMean
    y::Vector{Float64}
    s::Float64
end
NE.nsubjects(m::KinkedMean) = length(m.y)
NE.nlme_layout(::KinkedMean) = NLMELayout(0, 0, [1])
NE.observation_model(::KinkedMean) = GaussianObservations()
NE.observations(m::KinkedMean, i) = [m.y[i]]
NE.observation_moments(m::KinkedMean, i, θ, σ, η) = ([-m.s * abs(η[1])], [1.0])
NE.observation_moments_and_jacobians(m::KinkedMean, i, θ, σ, η) =
    ([-m.s * abs(η[1])], [1.0], fill(-m.s * sign(η[1]), 1, 1), zeros(1, 1))
NE.conditional_loglikelihood(m::KinkedMean, i, θ, σ, η) = -(m.y[i] + m.s * abs(η[1]))^2 / 2 - log(2π) / 2
NE.conditional_loglikelihood_and_gradient(m::KinkedMean, i, θ, σ, η) =
    (NE.conditional_loglikelihood(m, i, θ, σ, η), [-(m.y[i] + m.s * abs(η[1])) * m.s * sign(η[1])])
NE.conditional_loglikelihood_gradient_and_hessian(m::KinkedMean, i, θ, σ, η) =
    (NE.conditional_loglikelihood_and_gradient(m, i, θ, σ, η)..., fill(-m.s^2, 1, 1))
NE.provides_eta_hessian(::KinkedMean) = true

# Throws a DomainError beyond `limit` above each subject's kink.
struct RightBounded
    inner::LaplaceObservation
    limit::Float64
end
NE.nsubjects(m::RightBounded) = nsubjects(m.inner)
NE.nlme_layout(m::RightBounded) = nlme_layout(m.inner)
check_domain(m::RightBounded, i, η) =
    η[1] > m.inner.y[i] + m.limit && throw(DomainError(η[1], "outside the model's domain"))
NE.conditional_loglikelihood(m::RightBounded, i, θ, σ, η) =
    (check_domain(m, i, η); conditional_loglikelihood(m.inner, i, θ, σ, η))
NE.conditional_loglikelihood_and_gradient(m::RightBounded, i, θ, σ, η) =
    (check_domain(m, i, η); conditional_loglikelihood_and_gradient(m.inner, i, θ, σ, η))
NE.conditional_loglikelihood_gradient_and_hessian(m::RightBounded, i, θ, σ, η) =
    (check_domain(m, i, η); conditional_loglikelihood_gradient_and_hessian(m.inner, i, θ, σ, η))
NE.provides_eta_hessian(::RightBounded) = true

# A ragged layout: family 1 is a KinkedRidge pair, family 2 smooth units with one
# observation y_u ~ N(η_u, 1) each.
struct RaggedRidge
    layout::RaggedNLMELayout
    ridge::KinkedRidge
    units::Vector{Vector{Float64}}
end
NE.nsubjects(::RaggedRidge) = 3
NE.nlme_layout(m::RaggedRidge) = m.layout
NE.conditional_loglikelihood(m::RaggedRidge, i, θ, σ, η) =
    conditional_loglikelihood(m.ridge, i, θ, σ, η[1:2]) - sum(abs2, η[3:end] - m.units[i]; init = 0.0) / 2
function NE.conditional_loglikelihood_and_gradient(m::RaggedRidge, i, θ, σ, η)
    _, g = conditional_loglikelihood_and_gradient(m.ridge, i, θ, σ, η[1:2])
    return NE.conditional_loglikelihood(m, i, θ, σ, η), [g; m.units[i] - η[3:end]]
end
function NE.conditional_loglikelihood_gradient_and_hessian(m::RaggedRidge, i, θ, σ, η)
    _, _, Hr = conditional_loglikelihood_gradient_and_hessian(m.ridge, i, θ, σ, η[1:2])
    n = length(η)
    Hk = zeros(n, n)
    Hk[1:2, 1:2] = Hr
    Hk[3:n, 3:n] = -I(n - 2)
    return NE.conditional_loglikelihood_and_gradient(m, i, θ, σ, η)..., Hk
end
NE.provides_eta_hessian(::RaggedRidge) = true

const METHODS = (NewtonEBE(), QuasiNewtonEBE())

@testset "modes at kinks" begin
    @testset "a Laplace observation: the mode is the kink" begin
        y = [0.5, -0.3, 0.9]
        for hessian in (true, false), method in METHODS
            hessian || method isa QuasiNewtonEBE || continue
            model = LaplaceObservation(y, 1.0, hessian)
            r = empirical_bayes(model, Float64[], Float64[], [0.0]; method)
            @test all(r.converged)
            @test all(r.nonsmooth)
            @test vec(r.modes) ≈ y atol = 1.0e-8
            @test all(<(1.0e-6), r.gradient_norm)
            # Without the kink phase, as before 0.2.10: no mode, with the same error.
            err = try
                empirical_bayes(model, Float64[], Float64[], [0.0]; method,
                    options = EBEOptions(nonsmooth = false))
            catch e
                e
            end
            @test err isa EBEConvergenceError
            @test sort(err.subjects) == 1:3
            @test !any(err.result.nonsmooth)
        end
    end

    @testset "a kink whose mode lies along it" begin
        y = [0.0, 0.3, -0.2]
        for hessian in (true, false), method in METHODS
            hessian || method isa QuasiNewtonEBE || continue
            model = KinkedRidge(y; hessian)
            smooth_only = empirical_bayes(model, Float64[], Float64[], zeros(3); method,
                options = EBEOptions(nonsmooth = false), require_convergence = false)
            @test !any(smooth_only.converged)
            r = empirical_bayes(model, Float64[], Float64[], zeros(3); method)
            @test all(r.converged) && all(r.nonsmooth)
            for k in 1:3
                truth = ridge_mode(model, k)
                @test maximum(abs, r.modes[:, k] - truth) < 1.0e-5
                f(η) = complete_data_loglikelihood(model, k, Float64[], Float64[], zeros(3), η)
                @test f(truth) - f(r.modes[:, k]) < 1.0e-9
            end
            # The kink is not a mode where the smooth search stopped on it.
            @test maximum(abs, smooth_only.modes - r.modes) > 1.0e-3 || method isa QuasiNewtonEBE
            @test r.hessians === nothing || length(r.hessians) == 3
        end
    end

    @testset "lockstep: the stalled subjects share batched calls" begin
        y = [0.0, 0.3, -0.2]
        batched = Batched(KinkedRidge(y))
        r = empirical_bayes(batched, Float64[], Float64[], zeros(3); method = NewtonEBE())
        single = empirical_bayes(KinkedRidge(y), Float64[], Float64[], zeros(3); method = NewtonEBE())
        @test all(r.nonsmooth)
        @test r.modes ≈ single.modes atol = 1.0e-12
        @test count(>(1), batched.batches) > length(batched.batches) / 2
    end

    @testset "ragged layout" begin
        y = [0.0, 0.3, -0.2]
        units = [Float64[], [1.0, -2.0], [0.5]]
        layout = RaggedNLMELayout(ntheta = 0, nsigma = 0, family_sizes = [2, 1],
            subject_families = [[1], [1, 2, 2], [1, 2]])
        ridgepart = KinkedRidge(y)
        model = RaggedRidge(layout, ridgepart, units)
        ω = omega_parameters(layout, [Matrix(1.0I, 2, 2), ones(1, 1)])
        for method in METHODS
            r = empirical_bayes(model, Float64[], Float64[], ω; method)
            @test r isa RaggedEBEResult
            @test all(r.converged) && all(r.nonsmooth)
            for k in 1:3
                @test maximum(abs, r.modes[k][1:2] - ridge_mode(ridgepart, k)) < 1.0e-5
                @test r.modes[k][3:end] ≈ units[k] ./ 2 atol = 1.0e-5
            end
        end
    end

    @testset "a RECIST-shaped category law" begin
        model = recist_like(Xoshiro(2026), 40)
        ω = omega_parameters(nlme_layout(model), Matrix(Diagonal(fill(0.25, 3))))
        smooth_only = empirical_bayes(model, Float64[], Float64[], ω; method = NewtonEBE(),
            options = EBEOptions(nonsmooth = false), require_convergence = false)
        stalled = findall(!, smooth_only.converged)
        @test !isempty(stalled)
        for method in METHODS, m in (model, GradientsOnly(model))
            method isa NewtonEBE && m isa GradientsOnly && continue
            r = empirical_bayes(m, Float64[], Float64[], ω; method)
            @test all(r.converged)
            kinked = findall(r.nonsmooth)
            @test !isempty(kinked)
            for k in kinked
                f(η) = complete_data_loglikelihood(model, k, Float64[], Float64[], ω, η)
                local_search = Optim.optimize(η -> -f(η), r.modes[:, k], Optim.NelderMead(),
                    Optim.Options(g_tol = 1.0e-14, iterations = 10_000))
                @test -Optim.minimum(local_search) - f(r.modes[:, k]) < 1.0e-8
                @test maximum(abs, Optim.minimizer(local_search) - r.modes[:, k]) < 1.0e-4
            end
            # Subjects the smooth search settled are left as they were.
            if method isa NewtonEBE
                settled = findall(smooth_only.converged)
                @test kinked ⊆ stalled
                @test r.modes[:, settled] == smooth_only.modes[:, settled]
                @test r.iterations[settled] == smooth_only.iterations[settled]
            end
        end
    end

    @testset "trial points outside the model's domain" begin
        model = RightBounded(LaplaceObservation([0.5, 0.2], 1.0, true), 0.02)
        for method in METHODS
            r = empirical_bayes(model, Float64[], Float64[], [0.0]; method)
            @test all(r.converged) && all(r.nonsmooth)
            @test vec(r.modes) ≈ [0.5, 0.2] atol = 1.0e-8
        end
    end

    @testset "curvature at a kink is refused by name" begin
        model = LaplaceObservation([0.5, 0.0, 0.9], 1.0, true)
        r = empirical_bayes(model, Float64[], Float64[], [0.0]; subjects = [1, 3])
        @test all(r.nonsmooth)
        err = try
            ebe_covariances(r)
        catch e
            e
        end
        @test err isa NonsmoothModeError
        @test err.subjects == [1, 3]
        @test occursin("kink", sprint(showerror, err))
        for approximation in (ImportanceSampling(draws = 50), AdaptiveGaussHermite(nodes = 1))
            err = try
                marginal_loglikelihood(model, Float64[], Float64[], [0.0]; subjects = [1, 3], approximation)
            catch e
                e
            end
            @test err isa NonsmoothModeError
            @test err.subjects == [1, 3]
            @test occursin(string(nameof(typeof(approximation))), sprint(showerror, err))
        end
        # A Gaussian observation whose mean has the kink: FOCE(I) and CWRES refuse it,
        # and FOCE(I)'s search scores such a point as infeasible.
        kinked = KinkedMean([1.0, 2.0], 1.0)
        r = empirical_bayes(kinked, Float64[], Float64[], [0.0]; η0 = [0.3 -0.4])
        @test all(r.nonsmooth)
        @test vec(r.modes) ≈ [0.0, 0.0] atol = 1.0e-8
        @test_throws NonsmoothModeError focei_evaluate(kinked, Float64[], Float64[], [0.0]; η0 = [0.3 -0.4])
        @test_throws NonsmoothModeError residual_diagnostics(kinked, Float64[], Float64[], [0.0];
            η0 = [0.3 -0.4], replicates = 0)
    end

    @testset "the hull weights" begin
        rng = Xoshiro(7)
        for _ in 1:30
            m, n = rand(rng, 2:9), rand(rng, 1:6)
            P = randn(rng, n, m) .+ randn(rng, n)
            e = rand(rng) < 0.5 ? zeros(m) : abs.(randn(rng, m))
            Q = P' * P
            λ = NE._hull_weights(Q, e)
            @test all(>=(0), λ) && sum(λ) ≈ 1
            # Optimality: no vertex lowers ½λᵀQλ + eᵀλ to first order.
            gradient = Q * λ + e
            @test minimum(gradient) >= dot(λ, gradient) - 1.0e-10 * max(1, maximum(abs, Q))
        end
    end

    @testset "the restricted smallest eigenvalue" begin
        rng = Xoshiro(8)
        for _ in 1:20
            n, v = rand(rng, 2:12), rand(rng, 0:2)
            A = Symmetric(randn(rng, n, n))
            V = Matrix(qr(randn(rng, n, v)).Q)[:, 1:v]
            U = nullspace(V')
            products = (js, vs) -> [A * w for w in vs]
            μ = only(NE._restricted_min_eigenvalues(products, [V], [n], 36))
            @test μ ≈ eigmin(Symmetric(U' * A * U)) atol = 1.0e-8
        end
        @test only(NE._restricted_min_eigenvalues((js, vs) -> vs, [Matrix(1.0I, 2, 2)], [2], 36)) == Inf
    end
end
