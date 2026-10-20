using NLMEEstimation
using NLMEEstimation.ReferenceModels
import NLMEEstimation as NE
import DifferentiationInterface as DI
using LinearAlgebra
using Random
using Statistics: cor
using DelimitedFiles: readdlm
using Test
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon

# The EBEs must coincide with the brute-force joint optimum over all subjects' η; the
# returned Hessians (when the model provides second order) with the diagonal blocks
# of the joint Hessian, whose off-diagonal blocks vanish because subjects are
# independent given θ, σ, Ω.
function check_against_joint_reference(model, θ, σ, ω; method = default_ebe_method(model), atol = 1.0e-7)
    r = empirical_bayes(model, θ, σ, ω; method)
    n = nsubjects(model)
    m = neta(nlme_layout(model))
    @test all(r.converged)
    @test r.subjects == 1:n
    @test r.method === method
    reference, prior = joint_reference(model, θ, σ, ω)
    @test maximum(abs, r.modes - reference) < atol
    contexts = (DI.Constant(model), DI.Constant(θ), DI.Constant(σ), DI.Constant(prior), DI.Constant(m))
    x = vec(r.modes)
    @test maximum(abs, DI.gradient(joint_logdensity, RUNTIME_REVERSE, x, contexts...)) < 1.0e-6
    for i in 1:n
        @test r.complete_data_loglikelihood[i] ≈ complete_data_loglikelihood(model, i, θ, σ, ω, r.modes[:, i])
        @test r.conditional_loglikelihood[i] ≈ conditional_loglikelihood(model, i, θ, σ, r.modes[:, i])
    end
    if r.hessians === nothing
        @test !provides_eta_hessian(model)
    else
        Hcond = DI.hessian(joint_conditional, JOINT_HESSIAN, x,
            DI.Constant(model), DI.Constant(θ), DI.Constant(σ), DI.Constant(m))
        Hjoint = Hcond - kron(Matrix(I, n, n), prior.precision)
        for i in 1:n, j in 1:n
            block = Hjoint[((i - 1) * m + 1):(i * m), ((j - 1) * m + 1):(j * m)]
            if i == j
                @test block ≈ r.hessians[i] rtol = 1.0e-10
                @test isposdef(Symmetric(-r.hessians[i]))
            else
                @test iszero(block)
            end
        end
    end
    return r
end

# Observations near 400 with a residual SD of 0.3, as for a QT interval in ms:
# y_ij ~ N(30 exp(η₁) x_j / (1 + x_j) + 400 exp(η₂), 0.3²). Each residual y - μ rounds
# off at about 400 eps, so evaluating f, about -15 per subject, rounds off by tens of
# ulps of f.
struct HighBaseline
    y::Vector{Vector{Float64}}
    x::Vector{Float64}
end
const HIGH_BASELINE_SD = 0.3
NE.nsubjects(m::HighBaseline) = length(m.y)
NE.nlme_layout(::HighBaseline) = NLMELayout(0, 0, [1, 1])
NE.provides_eta_hessian(::HighBaseline) = true
# The two mean terms; each is also its own first and second derivative in its η.
_high_baseline_terms(η, x) = (30 * exp(η[1]) * x / (1 + x), 400 * exp(η[2]))
function NE.conditional_loglikelihood(m::HighBaseline, i, θ, σ, η)
    return sum(zip(m.y[i], m.x)) do (y, x)
        -((y - sum(_high_baseline_terms(η, x))) / HIGH_BASELINE_SD)^2 / 2 - log(HIGH_BASELINE_SD) - log(2π) / 2
    end
end
function NE.conditional_loglikelihood_gradient_and_hessian(m::HighBaseline, i, θ, σ, η)
    g = zeros(2)
    H = zeros(2, 2)
    for (y, x) in zip(m.y[i], m.x)
        J = collect(_high_baseline_terms(η, x))
        r = (y - sum(J)) / HIGH_BASELINE_SD^2
        g .+= r .* J
        H .+= r .* Diagonal(J) .- J * J' ./ HIGH_BASELINE_SD^2
    end
    return conditional_loglikelihood(m, i, θ, σ, η), g, H
end
NE.conditional_loglikelihood_and_gradient(m::HighBaseline, i, θ, σ, η) =
    conditional_loglikelihood_gradient_and_hessian(m, i, θ, σ, η)[1:2]
const Ω_HIGH_BASELINE = Matrix(Diagonal([0.3, 0.03] .^ 2))
function high_baseline_model(rng, n)
    x = collect(range(0.1, 5.0; length = 12))
    L = cholesky(Ω_HIGH_BASELINE).L
    y = map(1:n) do _
        η = L * randn(rng, 2)
        [sum(_high_baseline_terms(η, xj)) + HIGH_BASELINE_SD * randn(rng) for xj in x]
    end
    return HighBaseline(y, x)
end

@testset "empirical_bayes" begin
    @testset "simulated data, diagonal Ω" begin
        model, η_true = simulated_model()
        ω = omega_parameters(nlme_layout(model), Ω_SIM)
        @test default_ebe_method(model) === NewtonEBE()
        r = check_against_joint_reference(model, θ_SIM, σ_SIM, ω)
        @test all(<=(1.0e-8), r.gradient_norm)
        # The clearance random effect is well identified by ten samples per subject.
        @test cor(r.modes[2, :], [η[2] for η in η_true]) > 0.9
        @test posthoc === empirical_bayes

        # Gradient-only BFGS reaches the same modes; this model still provides Hessians.
        q = check_against_joint_reference(model, θ_SIM, σ_SIM, ω; method = QuasiNewtonEBE())
        @test maximum(abs, q.modes - r.modes) < 1.0e-7
        @test all(<=(EBEOptions().gtol), q.gradient_norm)

        # Laplace posterior covariances of η.
        C = ebe_covariances(r)
        @test C[3] ≈ inv(-r.hessians[3])
        @test all(isposdef, C)

        # A warm start from the result is already at the modes (at most the polishing
        # Newton step is taken).
        warm = empirical_bayes(model, θ_SIM, σ_SIM, ω; η0 = r)
        @test maximum(abs, warm.modes - r.modes) < 1.0e-12
        @test all(<=(1), warm.iterations)
        subset = empirical_bayes(model, θ_SIM, σ_SIM, ω; subjects = [4, 9], η0 = r)
        @test subset.subjects == [4, 9]
        @test maximum(abs, subset.modes - r.modes[:, [4, 9]]) < 1.0e-12
        @test empirical_bayes(model, θ_SIM, σ_SIM, ω; subjects = [4, 9], η0 = r.modes[:, [4, 9]]).modes ≈ subset.modes
        @test_throws ArgumentError empirical_bayes(model, θ_SIM, σ_SIM, ω; subjects = [nsubjects(model) + 1])
        @test_throws ArgumentError empirical_bayes(model, θ_SIM, σ_SIM, ω; subjects = [2, 2])
        @test_throws ArgumentError empirical_bayes(model, θ_SIM, σ_SIM, ω; η0 = subset)
        @test_throws DimensionMismatch empirical_bayes(model, θ_SIM, σ_SIM, ω; η0 = zeros(3, 2))
        @test_throws DimensionMismatch empirical_bayes(model, θ_SIM[1:3], σ_SIM, ω)
    end

    @testset "joint-program models: batched, lockstep" begin
        model, _ = simulated_model()
        ω = omega_parameters(nlme_layout(model), Ω_SIM)
        r = empirical_bayes(model, θ_SIM, σ_SIM, ω)

        # Gradient-only joint program (the shape of BRM's RK backend today): BFGS by
        # default, no Hessians in the result.
        gradient_only = JointView(model; hessian_backend = nothing)
        @test default_ebe_method(gradient_only) === QuasiNewtonEBE()
        q = check_against_joint_reference(gradient_only, θ_SIM, σ_SIM, ω)
        @test q.hessians === nothing
        @test_throws ArgumentError ebe_covariances(q)
        @test maximum(abs, q.modes - r.modes) < 1.0e-7
        # Lockstep: the number of batched evaluations follows the iteration count, not
        # the number of subjects (each one is an O(N) pass for a joint program).
        gradient_only.calls[] = 0
        q = empirical_bayes(gradient_only, θ_SIM, σ_SIM, ω)
        @info "lockstep batched evaluations" calls = gradient_only.calls[] max_iterations = maximum(q.iterations) total_iterations = sum(q.iterations)
        @test gradient_only.calls[] < 2 * nsubjects(model) * maximum(q.iterations)

        # With HVP-based Hessian blocks the joint program runs Newton.
        joint = JointView(model)
        @test default_ebe_method(joint) === NewtonEBE()
        j = check_against_joint_reference(joint, θ_SIM, σ_SIM, ω)
        @test maximum(abs, j.modes - r.modes) < 1.0e-10
        @test j.hessians ≈ r.hessians
    end

    @testset "simulated data, dense Ω" begin
        R = [1 0.3 0.1; 0.3 1 -0.2; 0.1 -0.2 1]
        D = Diagonal([0.4, 0.3, 0.2])
        Ω = Matrix(D * R * D)
        model, _ = simulated_model(; seed = 12, Ω, eta_blocks = [3])
        ω = omega_parameters(nlme_layout(model), Ω)
        check_against_joint_reference(model, θ_SIM, σ_SIM, ω)
        check_against_joint_reference(model, θ_SIM, σ_SIM, ω; method = QuasiNewtonEBE())
    end

    @testset "Theoph data" begin
        model = theoph_model()
        @test nsubjects(model) == 12
        ω = omega_parameters(nlme_layout(model), Ω_THEOPH)
        r = check_against_joint_reference(model, θ_THEOPH, σ_THEOPH, ω)
        check_against_joint_reference(JointView(model; hessian_backend = nothing), θ_THEOPH, σ_THEOPH, ω)
        # Test-only oracle: nlmixr2 posthoc at the same fixed parameters on the same data
        # (`nlmixr2data::theo_sd` equals test/data/theoph.csv), generated by
        # test/oracles/nlmixr2_posthoc_theoph.R.
        oracle, oracle_header = readdlm(joinpath(@__DIR__, "data", "nlmixr2_posthoc_theoph.csv"), ','; header = true)
        @test vec(oracle_header) == ["ID", "eta.ka", "eta.cl", "eta.v"]
        @test oracle[:, 1] == 1:12
        nlmixr2_modes = permutedims(Float64.(oracle[:, 2:4]))
        # The structural model is the same: our predictions equal nlmixr2's individual
        # predictions at its EBEs (IPRED) and its typical-value predictions (PRED).
        predictions, predictions_header = readdlm(joinpath(@__DIR__, "data", "nlmixr2_posthoc_theoph_predictions.csv"), ','; header = true)
        @test vec(predictions_header) == ["ID", "TIME", "IPRED", "PRED"]
        for i in 1:12
            rows = findall(==(i), predictions[:, 1])
            @test Float64.(predictions[rows, 2]) == model.subjects[i].times
            @test observation_moments(model, i, θ_THEOPH, σ_THEOPH, nlmixr2_modes[:, i])[1] ≈ predictions[rows, 3] atol = 1.0e-10
            @test observation_moments(model, i, θ_THEOPH, σ_THEOPH, zeros(3))[1] ≈ predictions[rows, 4] atol = 1.0e-10
        end
        # nlmixr2's modes are converged only to its own inner tolerance (see the oracle
        # script): they agree with ours to 1e-3, and on the shared objective ours are at
        # least as good and within 1e-5 of them, i.e. the same optimum.
        @info "EBE agreement with nlmixr2 posthoc" max_abs_difference = maximum(abs, r.modes - nlmixr2_modes)
        @test maximum(abs, r.modes - nlmixr2_modes) < 1.0e-3
        for i in 1:12
            ours = complete_data_loglikelihood(model, i, θ_THEOPH, σ_THEOPH, ω, r.modes[:, i])
            theirs = complete_data_loglikelihood(model, i, θ_THEOPH, σ_THEOPH, ω, nlmixr2_modes[:, i])
            @test theirs <= ours + 1.0e-12
            @test ours - theirs < 1.0e-5
        end
        # The individual fits are better than the typical-value fit.
        sse(i, η) = sum(abs2, observations(model, i) - observation_moments(model, i, θ_THEOPH, σ_THEOPH, η)[1])
        @test sum(i -> sse(i, r.modes[:, i]), 1:12) < sum(i -> sse(i, zeros(3)), 1:12) / 2
    end

    @testset "Newton from a start where the density is not concave" begin
        model = DoubleWell(NLMELayout(0, 0, [1]), 1.5, [0.05, 0.1, 0.2])
        ω = omega_parameters(nlme_layout(model), [1.0;;])
        η0 = fill(0.1, 1, 3)
        # At η0 the complete-data Hessian is +0.47 for every subject: the Newton step
        # does not exist, so the first step goes to the trust region's boundary.
        for i in 1:3
            _, _, _, H = complete_data_loglikelihood_gradient_and_hessian(model, i, Float64[], Float64[], ω, η0[:, i])
            @test H[1, 1] ≈ 0.47
        end
        newton = empirical_bayes(model, Float64[], Float64[], ω; η0, method = NewtonEBE())
        bfgs = empirical_bayes(model, Float64[], Float64[], ω; η0, method = QuasiNewtonEBE())
        @test all(newton.converged)
        # The modes solve -η³ + (a - 1)η + b = 0 where the density is concave.
        η = newton.modes[1, :]
        @test maximum(abs, -η .^ 3 .+ 0.5 .* η .+ model.b) < 1.0e-12
        @test all(H -> H[1, 1] < 0, newton.hessians)
        @test maximum(abs, newton.modes - bfgs.modes) < 1.0e-7
    end

    @testset "Newton leaves saddle points along their upward curvature" begin
        # η₂ starts at the saddle (subjects 1 and 2, where b = 0 makes its gradient
        # exactly zero) or next to it. Levenberg–Marquardt damping with λ ≈ 1e-3·D
        # left it by a factor of about 1.05 per iteration and ran out of iterations
        # with a small gradient (snag newton-ebe-stall-9d3fb9ab).
        model = SteepWell(NLMELayout(0, 0, [1, 1]), 1.0e4, 1.5, [0.0, 0.0, 1.0e-6, 1.0e-4, 1.0e-2], 0.3)
        ω = omega_parameters(nlme_layout(model), Matrix(1.0I, 2, 2))
        η0 = [0.0 0.3 0.0 0.0 0.0; 0.0 0.0 1.0e-6 1.0e-6 1.0e-6]
        r = empirical_bayes(model, Float64[], Float64[], ω; η0, method = NewtonEBE())
        @test all(r.converged)
        @test maximum(r.iterations) <= 10
        # The modes are stationary and the density is concave there: η₁ = Dc/(D + 1),
        # and η₂ solves -η³ + (a - 1)η + b = 0 away from the saddle.
        @test r.modes[1, :] ≈ fill(model.D * model.c / (model.D + 1), 5) rtol = 1.0e-12
        η₂ = r.modes[2, :]
        @test maximum(abs, -η₂ .^ 3 .+ 0.5 .* η₂ .+ model.b) < 1.0e-12
        @test all(>(0.7), abs.(η₂))
        @test all(H -> isposdef(Symmetric(-H)), r.hessians)
    end

    @testset "BFGS leaves saddle points, with or without Hessian blocks" begin
        # BFGS keeps a positive-definite curvature model. It reported the exact saddle
        # start (subject 1, where b = 0 makes the gradient zero) as converged after 0
        # iterations, and likewise the start η = 0 (subject 2), from which η₂ stays
        # exactly 0 while η₁ converges. Next to the saddle (the others) its steps along
        # η₂ stayed about 1e-4 of the gradient, since every curvature pair along η₂ has
        # sᵀy < 0 and is skipped, until maxiter (todo 2026-10-20T07-55-53-251-1569zus).
        # The solver checks the Hessian block at skipped pairs and at convergence (the
        # model's, or forward differences for a gradient-only model) and leaves by a
        # trust-region step.
        model = SteepWell(NLMELayout(0, 0, [1, 1]), 1.0e4, 1.5,
            [0.0, 0.0, 0.0, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2], 0.3)
        ω = omega_parameters(nlme_layout(model), Matrix(1.0I, 2, 2))
        η★ = model.D * model.c / (model.D + 1)
        η0 = [η★ 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 1.0e-6 1.0e-6 1.0e-6 1.0e-6 1.0e-6]
        newton = empirical_bayes(model, Float64[], Float64[], ω; η0, method = NewtonEBE())
        bfgs = empirical_bayes(model, Float64[], Float64[], ω; η0, method = QuasiNewtonEBE())
        gradient_only = GradientOnly(model)
        @test default_ebe_method(gradient_only) === QuasiNewtonEBE()
        differences = empirical_bayes(gradient_only, Float64[], Float64[], ω; η0)
        @test differences.hessians === nothing
        for r in (bfgs, differences)
            @test all(r.converged)
            @test minimum(r.iterations) > 0
            @test maximum(r.iterations) <= 20
            # Stationary where the density is concave: η₁ = Dc/(D + 1), and η₂ solves
            # -η³ + (a - 1)η + b = 0 away from the saddle; from η₂ > 0 the same mode
            # as Newton's.
            @test r.modes[1, :] ≈ fill(η★, 7) rtol = 1.0e-10
            η₂ = r.modes[2, :]
            @test maximum(abs, -η₂ .^ 3 .+ 0.5 .* η₂ .+ model.b) < 1.0e-8
            @test all(>(0.7), abs.(η₂))
            @test maximum(abs, r.modes[:, 3:end] - newton.modes[:, 3:end]) < 1.0e-8
        end
        @test all(H -> isposdef(Symmetric(-H)), bfgs.hessians)
    end

    @testset "modes are found where evaluating f rounds off by many ulps" begin
        # A decrease of f was tolerated only up to 16 ulps of f, so a point whose
        # rounding came out high rejected every later step, and subjects at their
        # modes were reported unconverged with standardized gradients near 1e-6:
        # subject 5 for Newton, 18 of 40 for BFGS (snag run-empirical-ba-6d5b89f0).
        model = high_baseline_model(Xoshiro(1), 40)
        ω = omega_parameters(nlme_layout(model), Ω_HIGH_BASELINE)
        newton = empirical_bayes(model, Float64[], Float64[], ω; method = NewtonEBE(), require_convergence = false)
        bfgs = empirical_bayes(model, Float64[], Float64[], ω; method = QuasiNewtonEBE(), require_convergence = false)
        @test all(newton.converged)
        @test all(bfgs.converged)
        # f rounds off by more than 16 ulps around the modes: its values a 1e-9 step
        # away depart from its quadratic model by more than that.
        η = newton.modes[:, 5]
        _, f, g, H = complete_data_loglikelihood_gradient_and_hessian(model, 5, Float64[], Float64[], ω, η)
        departures = map(1:50) do j
            δ = 1.0e-9 .* [cos(j), sin(j)]
            complete_data_loglikelihood(model, 5, Float64[], Float64[], ω, η + δ) - (f + dot(g, δ) + dot(δ, H * δ) / 2)
        end
        @test maximum(abs, departures) > 16 * eps(abs(f))
        # Newton's polishing step is kept, so its modes are stationary to roundoff;
        # BFGS's agree with them to its gradient tolerance.
        for k in 1:40
            _, _, g, H = complete_data_loglikelihood_gradient_and_hessian(model, k, Float64[], Float64[], ω, newton.modes[:, k])
            @test dot(g, -H \ g) / 2 < 1.0e-20
        end
        @test maximum(abs, newton.modes - bfgs.modes) < 1.0e-8
    end

    @testset "failures are reported, never silent" begin
        model, _ = simulated_model()
        ω = omega_parameters(nlme_layout(model), Ω_SIM)
        for method in (NewtonEBE(), QuasiNewtonEBE())
            options = EBEOptions(; maxiter = 1)
            err = try
                empirical_bayes(model, θ_SIM, σ_SIM, ω; method, options)
            catch e
                e
            end
            @test err isa EBEConvergenceError
            @test !isempty(err.subjects)
            @test occursin("subject", sprint(showerror, err))
            partial = empirical_bayes(model, θ_SIM, σ_SIM, ω; method, options, require_convergence = false)
            @test partial.subjects[.!partial.converged] == err.subjects
            @test all(==(1), partial.iterations[.!partial.converged])
        end
        # The report says when the density is not concave at the last point.
        well = DoubleWell(NLMELayout(0, 0, [1]), 1.5, [0.05])
        err = try
            empirical_bayes(well, Float64[], Float64[], [0.0]; η0 = [0.1;;],
                method = NewtonEBE(), options = EBEOptions(; maxiter = 0))
        catch e
            e
        end
        @test err isa EBEConvergenceError
        @test occursin("not concave", sprint(showerror, err))
        # A zero residual variance makes the log-density non-finite at the start.
        @test_throws DomainError empirical_bayes(model, θ_SIM, [-400.0, -400.0], ω)
    end
end
