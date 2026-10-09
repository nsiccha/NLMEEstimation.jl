using NLMEEstimation
using NLMEEstimation.ReferenceModels
using LinearAlgebra
using Random
using Statistics: cor
using Test
import ForwardDiff
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon

# The EBEs must coincide with the brute-force joint optimum over all subjects' η, and
# the returned Hessians with the diagonal blocks of the joint Hessian (whose
# off-diagonal blocks vanish, because subjects are independent given θ, σ, Ω).
function check_against_joint_reference(model, θ, σ, ω)
    r = empirical_bayes(model, θ, σ, ω)
    n = nsubjects(model)
    m = neta(nlme_layout(model))
    @test all(r.converged)
    @test r.subjects == 1:n
    reference, joint = joint_reference(model, θ, σ, ω)
    @test maximum(abs, stacked(r.modes) - stacked(reference)) < 1.0e-7
    x = stacked(r.modes)
    @test maximum(abs, ForwardDiff.gradient(joint, x)) < 1.0e-8
    H = ForwardDiff.hessian(joint, x)
    for i in 1:n, j in 1:n
        block = H[((i - 1) * m + 1):(i * m), ((j - 1) * m + 1):(j * m)]
        if i == j
            @test block ≈ r.hessians[i] rtol = 1.0e-10
        else
            @test iszero(block)
        end
    end
    for i in 1:n
        @test r.complete_data_loglikelihood[i] ≈ complete_data_loglikelihood(model, i, θ, σ, ω, r.modes[i])
        @test r.conditional_loglikelihood[i] ≈ conditional_loglikelihood(model, i, θ, σ, r.modes[i])
        @test isposdef(Symmetric(-r.hessians[i]))
        @test r.newton_decrement[i] / 2 <= EBEOptions().decrement_tol
    end
    return r
end

@testset "empirical_bayes" begin
    @testset "simulated data, diagonal Ω" begin
        model, η_true = simulated_model()
        ω = omega_parameters(nlme_layout(model), Ω_SIM)
        r = check_against_joint_reference(model, θ_SIM, σ_SIM, ω)
        # The clearance random effect is well identified by ten samples per subject.
        @test cor([η[2] for η in r.modes], [η[2] for η in η_true]) > 0.9
        @test posthoc === empirical_bayes

        # Laplace posterior covariances of η.
        C = ebe_covariances(r)
        @test C[3] ≈ inv(-r.hessians[3])
        @test all(isposdef, C)

        # A warm start from the result is already at the modes (at most the polishing
        # Newton step is taken).
        warm = empirical_bayes(model, θ_SIM, σ_SIM, ω; η0 = r)
        @test maximum(abs, stacked(warm.modes) - stacked(r.modes)) < 1.0e-12
        @test all(<=(1), warm.iterations)
        subset = empirical_bayes(model, θ_SIM, σ_SIM, ω; subjects = [4, 9], η0 = r)
        @test subset.subjects == [4, 9]
        @test maximum(abs, stacked(subset.modes) - stacked(r.modes[[4, 9]])) < 1.0e-12
        @test_throws ArgumentError empirical_bayes(model, θ_SIM, σ_SIM, ω; subjects = [nsubjects(model) + 1])
        @test_throws ArgumentError empirical_bayes(model, θ_SIM, σ_SIM, ω; η0 = subset)
        @test_throws DimensionMismatch empirical_bayes(model, θ_SIM[1:3], σ_SIM, ω)
    end

    @testset "simulated data, dense Ω" begin
        R = [1 0.3 0.1; 0.3 1 -0.2; 0.1 -0.2 1]
        D = Diagonal([0.4, 0.3, 0.2])
        Ω = Matrix(D * R * D)
        model, _ = simulated_model(; seed = 12, Ω, eta_blocks = [3])
        ω = omega_parameters(nlme_layout(model), Ω)
        check_against_joint_reference(model, θ_SIM, σ_SIM, ω)
    end

    @testset "Theoph data" begin
        model = theoph_model()
        @test nsubjects(model) == 12
        ω = omega_parameters(nlme_layout(model), Ω_THEOPH)
        r = check_against_joint_reference(model, θ_THEOPH, σ_THEOPH, ω)
        # The individual fits are better than the typical-value fit.
        sse(i, η) = sum(abs2, observations(model, i) - observation_moments(model, i, θ_THEOPH, σ_THEOPH, η)[1])
        @test sum(i -> sse(i, r.modes[i]), 1:12) < sum(i -> sse(i, zeros(3)), 1:12) / 2
    end

    @testset "failures are reported, never silent" begin
        model, _ = simulated_model()
        ω = omega_parameters(nlme_layout(model), Ω_SIM)
        options = EBEOptions(; maxiter = 1)
        err = try
            empirical_bayes(model, θ_SIM, σ_SIM, ω; options)
        catch e
            e
        end
        @test err isa EBEConvergenceError
        @test !isempty(err.subjects)
        @test occursin("subject", sprint(showerror, err))
        partial = empirical_bayes(model, θ_SIM, σ_SIM, ω; options, require_convergence = false)
        @test partial.subjects[.!partial.converged] == err.subjects
        @test all(==(1), partial.iterations[.!partial.converged])

        # A zero residual variance makes the log-density non-finite at the start.
        @test_throws DomainError empirical_bayes(model, θ_SIM, [-400.0, -400.0], ω)
    end
end
