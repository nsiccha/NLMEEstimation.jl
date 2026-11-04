using Test
using NLMEEstimation
using Random
using LinearAlgebra
import NLMEEstimation as NE
import Enzyme
import Treebars
using Treebars: render_text, request_interrupt!
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon
using .TestCommon: GROWTH_LAYOUT

# `WarmupHMCNUTS(threads = true)`. The suite runs on one Julia thread, so there this
# file tests the one-thread behaviour and reruns itself in a process with two threads.
# Every model here has hand-written gradients: the test environment has registered
# Enzyme, whose gradients are unsafe beside other threads (the refusal tested below).
const THREADS_NUTS = WarmupHMCNUTS(max_tree_depth = 6, threads = true)
const SERIAL_NUTS = WarmupHMCNUTS(max_tree_depth = 6)
const THREADS_OPTIONS = SAEMOptions(iterations = 12, burnin = 5, mcmc_burnin = 2)
const THREADS_GROWTH = ([1.0, 0.5], [log(0.4)],
    omega_parameters(GROWTH_LAYOUT, [0.5 0.1; 0.1 0.2]))
const THREADS_OCCASIONS = ([1.9], [log(0.27)], [log(0.55), log(0.35)])

same_chains(a, b) = a.theta == b.theta && a.sigma == b.sigma && a.omega == b.omega &&
    a.eta == b.eta && a.nuts_divergences == b.nuts_divergences

threads_fit(model, start, kernel; options = THREADS_OPTIONS, kwargs...) =
    saem(model, start...; rng = Xoshiro(5), options, kernel, kwargs...)

# A nonlinear, five-effect MAP model with structural θ. Both evaluation routes
# use exactly the same arithmetic, including away from the starting point.
# Equality at just the start is insufficient: last-bit gradient differences can
# change Pathfinder's path and the subsequent NUTS draws.
struct NonlinearThreadToy
    y::Vector{Vector{Float64}}
end
NE.nsubjects(m::NonlinearThreadToy) = length(m.y)
NE.nlme_layout(::NonlinearThreadToy) = NLMELayout(2, 1, [5])
NE.evaluation_style(::NonlinearThreadToy) = BatchedEvaluation()
NE.saem_population_gradient_style(::NonlinearThreadToy) = PerSubjectEvaluation()
function thread_toy_residual(m, i, θ, σ, η)
    r = m.y[i] .- (θ[1] .+ θ[2] .* η .+ 0.1 .* η .^ 2)
    v = exp(2σ[1])
    ℓ = -sum(abs2, r) / (2v) - length(r) * (log(2π) / 2 + σ[1])
    return ℓ, r, v
end
NE.conditional_loglikelihood(m::NonlinearThreadToy, i, θ, σ, η) =
    first(thread_toy_residual(m, i, θ, σ, η))
function NE.conditional_loglikelihood_and_gradient(m::NonlinearThreadToy, i, θ, σ, η)
    ℓ, r, v = thread_toy_residual(m, i, θ, σ, η)
    return ℓ, r .* (θ[2] .+ 0.2 .* η) ./ v
end
function NE.conditional_loglikelihood_and_population_gradient(m::NonlinearThreadToy,
        i, θ, σ, η)
    ℓ, r, v = thread_toy_residual(m, i, θ, σ, η)
    return ℓ, [sum(r), dot(r, η)] ./ v, [sum(abs2, r) / v - length(r)]
end
NE.conditional_loglikelihoods(m::NonlinearThreadToy, ids, θ, σ, H) =
    [NE.conditional_loglikelihood(m, i, θ, σ, H[:, k]) for (k, i) in enumerate(ids)]
function NE.conditional_loglikelihoods_and_gradients(m::NonlinearThreadToy, ids, θ, σ, H)
    pairs = [NE.conditional_loglikelihood_and_gradient(m, i, θ, σ, H[:, k])
        for (k, i) in enumerate(ids)]
    return first.(pairs), stack(last.(pairs))
end
NE.population_logprior(::NonlinearThreadToy, θ, σ, ω) = -sum(abs2, vcat(θ, σ, ω)) / 8
NE.population_logprior_and_gradient(m::NonlinearThreadToy, θ, σ, ω) =
    (NE.population_logprior(m, θ, σ, ω), -θ / 4, -σ / 4, -ω / 4)

# Forwards to `inner`; subclasses of behaviour below overload one method each.
abstract type Forwarding end
for f in (:nsubjects, :nlme_layout, :mu_referencing)
    @eval NE.$f(m::Forwarding) = NE.$f(m.inner)
end
for f in (:conditional_loglikelihood, :conditional_loglikelihood_and_population_gradient)
    @eval NE.$f(m::Forwarding, i::Integer, θ, σ, η) = NE.$f(m.inner, i, θ, σ, η)
end

# Holds the first two gradient calls until both are running, spinning without yielding,
# so that on threads they can only meet by overlapping on two different threads.
# `patience` bounds the wait, so a serial run passes through and shows one thread.
struct Rendezvous{M} <: Forwarding
    inner::M
    patience::Float64
    arrived::Threads.Atomic{Int}
    thread::Vector{Threads.Atomic{Int}}
end
Rendezvous(inner, patience) = Rendezvous(inner, patience, Threads.Atomic{Int}(0),
    [Threads.Atomic{Int}(0), Threads.Atomic{Int}(0)])
function NE.conditional_loglikelihood_and_gradient(m::Rendezvous, i::Integer, θ, σ, η)
    if m.arrived[] < 2
        arrival = Threads.atomic_add!(m.arrived, 1) + 1
        if arrival <= 2
            m.thread[arrival][] = Threads.threadid()
            deadline = time() + m.patience
            while m.arrived[] < 2 && time() < deadline
                GC.safepoint()
            end
        end
    end
    return NE.conditional_loglikelihood_and_gradient(m.inner, i, θ, σ, η)
end

# A model that declares Enzyme as its backend; its gradients stay hand-written.
struct OnEnzyme{M} <: Forwarding
    inner::M
end
NE.ad_backend(::OnEnzyme) = REVERSE
NE.conditional_loglikelihood_and_gradient(m::OnEnzyme, i::Integer, θ, σ, η) =
    NE.conditional_loglikelihood_and_gradient(m.inner, i, θ, σ, η)

# Not thread-safe: each evaluation puts η in one shared buffer and reads it back after
# yielding, so an overlapping evaluation of another subject replaces it.
struct SharedBuffer{M} <: Forwarding
    inner::M
    buffer::Vector{Float64}
end
function NE.conditional_loglikelihood_and_gradient(m::SharedBuffer, i::Integer, θ, σ, η)
    copyto!(m.buffer, η)
    yield()
    return NE.conditional_loglikelihood_and_gradient(m.inner, i, θ, σ, copy(m.buffer))
end

if Threads.nthreads() == 1
    @testset "WarmupHMCNUTS(threads = true) in a one-thread process" begin
        model = growth_model()
        threaded = @test_logs (:warn, r"one thread") threads_fit(model, THREADS_GROWTH, THREADS_NUTS)
        @test same_chains(threaded, threads_fit(model, THREADS_GROWTH, SERIAL_NUTS))
        # This check cannot run here, so it must not pass.
        @test_throws ArgumentError check_protocol(model, THREADS_GROWTH...; threads = true)
    end
    @testset "WarmupHMCNUTS(threads = true) in a two-thread process" begin
        child = `$(Base.julia_cmd()) -t 2 --project=$(Base.active_project()) $(@__FILE__)`
        @test success(pipeline(child; stdout, stderr))
    end
else
    @testset "WarmupHMCNUTS(threads = true)" begin
        @test Threads.nthreads() > 1
        growth, occasions = growth_model(), occasions_model()

        @testset "chains overlap on different threads" begin
            threaded = Rendezvous(growth, 120.0)
            threads_fit(threaded, THREADS_GROWTH, THREADS_NUTS)
            @test threaded.arrived[] >= 2
            @test threaded.thread[1][] != threaded.thread[2][]
            # Positive control: serial calls never meet, whatever the patience.
            serial = Rendezvous(growth, 0.05)
            threads_fit(serial, THREADS_GROWTH, SERIAL_NUTS)
            @test serial.thread[1][] == serial.thread[2][]
        end

        @testset "fits equal the serial route's, whatever the schedule" begin
            cases = (("equal layout", growth, THREADS_GROWTH),
                ("ragged layout", occasions, THREADS_OCCASIONS),
                ("positions outside the domain", BoundedFirst(growth, 0.6), THREADS_GROWTH),
                ("integrated families", LinearIntegrated(occasions, [2]), THREADS_OCCASIONS))
            for (name, model, start) in cases, options in (THREADS_OPTIONS,
                    SAEMOptions(iterations = 12, burnin = 5, mcmc_burnin = 2, minibatch_size = 3))
                serial = threads_fit(model, start, SERIAL_NUTS; options)
                @testset "$name, minibatch $(options.minibatch_size)" begin
                    for _ in 1:3
                        @test same_chains(threads_fit(model, start, THREADS_NUTS; options), serial)
                    end
                end
            end
        end

        @testset "Louis's information" begin
            louis(kernel) = covariance_step(growth, THREADS_GROWTH...; method = :s,
                approximation = LouisInformation(; draws = 40, burnin = 10, kernel),
                rng = Xoshiro(2)).covariance
            serial = louis(SERIAL_NUTS)
            @test all(louis(THREADS_NUTS) == serial for _ in 1:2)
        end

        @testset "threads overrides lockstep for a batched model" begin
            model = Lockstep(growth)
            threaded = threads_fit(model, THREADS_GROWTH, THREADS_NUTS)
            @test isempty(gradient_batches(model))
            lockstep = threads_fit(model, THREADS_GROWTH, SERIAL_NUTS)
            @test !isempty(gradient_batches(model))
            @test same_chains(threaded, lockstep)
        end

        @testset "nonlinear batched MAP, structural θ, no burn-in or adaptation delay" begin
            data_rng = Xoshiro(71)
            model = NonlinearThreadToy([randn(data_rng, 5) for _ in 1:2])
            θ, σ = [0.2, 1.3], [log(0.7)]
            ω = omega_parameters(nlme_layout(model), Matrix(0.7I, 5, 5))
            options = SAEMOptions(iterations = 2, burnin = 1, mcmc_burnin = 0,
                adapt_iterations = 0, numeric_maxiter = 2, omega_floor = 0.05)
            for seed in (5, 20261013)
                fit(threads) = saem(model, θ, σ, ω; options, rng = Xoshiro(seed),
                    kernel = WarmupHMCNUTS(; threads))
                serial, threaded = fit(false), fit(true)
                @test serial.objective == threaded.objective == :MAP
                @test serial.numeric_mstep == threaded.numeric_mstep == :score
                @test same_chains(serial, threaded)
            end
        end

        @testset "a chain's own error reaches the caller" begin
            model = BoundedFirst(growth, 0.6)
            ids = collect(1:nsubjects(model))
            θ, σ = [1.0, 0.5], [log(0.5)]
            prior = re_prior(model, omega_parameters(GROWTH_LAYOUT, [0.8 0.0; 0.0 0.3]))
            H = zeros(2, length(ids))
            states = NE._saem_nuts_start!(Xoshiro(1), THREADS_NUTS, model, ids, θ, σ, prior, H)
            H[1, 1] = 0.9
            NE._saem_nuts_reposition!(states, H, [1])
            @test_throws ArgumentError NE._saem_nuts_round!(states, model, ids, θ, σ, prior,
                copy(H); threads = true)
        end

        @testset "progress and interrupts" begin
            root = Treebars.initialize_progress!(:state; description = "test")
            threads_fit(growth, THREADS_GROWTH, THREADS_NUTS; progress = root)
            Treebars.finalize_progress!(root)
            @test occursin("✓ Pathfinder start (12/12)", render_text(root))

            root = Treebars.initialize_progress!(:state; description = "stopped")
            request_interrupt!(root)
            fit = threads_fit(growth, THREADS_GROWTH, THREADS_NUTS; progress = root)
            @test fit.interrupted && fit.iterations == 0 && fit.theta == THREADS_GROWTH[1]
        end

        # enzyme-use: "Reverse gradients beside other threads". A registered Enzyme
        # gives silently wrong gradients in exactly this shape of process.
        @testset "an Enzyme model needs the concurrent-GC repair" begin
            repaired = isdefined(Enzyme.Compiler, :RestoreSafepointEffectsPass)
            @test NE._enzyme_has_gc_repair() == repaired
            model = OnEnzyme(growth)
            if repaired
                @test threads_fit(model, THREADS_GROWTH, THREADS_NUTS) isa SAEMResult
            else
                refused = try
                    threads_fit(model, THREADS_GROWTH, THREADS_NUTS)
                    nothing
                catch err
                    err
                end
                @test refused isa ArgumentError
                @test occursin("dogfood", sprint(showerror, refused))
                @test_throws ArgumentError covariance_step(model, THREADS_GROWTH...;
                    method = :s, approximation = LouisInformation(draws = 4, burnin = 1,
                        kernel = THREADS_NUTS), rng = Xoshiro(2))
            end
            @test threads_fit(model, THREADS_GROWTH, SERIAL_NUTS) isa SAEMResult
        end

        @testset "check_protocol tries concurrent evaluation" begin
            @test check_protocol(growth, THREADS_GROWTH...; threads = true)
            @test check_protocol(occasions, THREADS_OCCASIONS...; threads = true)
            racy = SharedBuffer(growth, zeros(2))
            η = [Float64(k) / 12 for _ in 1:2, k in 1:12]
            @test check_protocol(racy, THREADS_GROWTH...; η)
            err = try
                check_protocol(racy, THREADS_GROWTH...; η, threads = true)
                nothing
            catch caught
                caught
            end
            @test err isa ProtocolViolation
            @test err isa ProtocolViolation && occursin("concurrent", err.message)
        end
    end
end
