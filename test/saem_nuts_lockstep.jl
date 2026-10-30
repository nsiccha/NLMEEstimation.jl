using Test
using NLMEEstimation
using Random
import NLMEEstimation as NE
import Treebars
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon

# A per-subject model behind the batched protocol, as a joint program such as BRM's
# whole-model sweep presents one: each batched call evaluates the listed subjects
# with the inner model's per-subject functions and logs the subjects it was given,
# `:gradient` for values with η gradients and `:value` for values alone. Its numbers
# are the inner model's, so fitting either gives the same result.
struct Lockstep{M}
    inner::M
    log::Vector{Pair{Symbol, Vector{Int}}}
end
Lockstep(inner) = Lockstep(inner, Pair{Symbol, Vector{Int}}[])
gradient_batches(m::Lockstep) = [s for (kind, s) in m.log if kind === :gradient]
NE.evaluation_style(::Lockstep) = BatchedEvaluation()
NE.saem_population_gradient_style(::Lockstep) = PerSubjectEvaluation()
for f in (:nsubjects, :nlme_layout, :mu_referencing)
    @eval NE.$f(m::Lockstep) = NE.$f(m.inner)
end
for f in (:conditional_loglikelihood, :conditional_loglikelihood_and_gradient,
        :conditional_loglikelihood_and_population_gradient)
    @eval NE.$f(m::Lockstep, i::Integer, θ, σ, η) = NE.$f(m.inner, i, θ, σ, η)
end
function NE.conditional_loglikelihoods(m::Lockstep, subjects, θ, σ, H)
    push!(m.log, :value => collect(subjects))
    return [conditional_loglikelihood(m.inner, i, θ, σ, NE._eta_column(H, k))
        for (k, i) in enumerate(subjects)]
end
function NE.conditional_loglikelihoods_and_gradients(m::Lockstep, subjects, θ, σ, H)
    push!(m.log, :gradient => collect(subjects))
    pairs = [conditional_loglikelihood_and_gradient(m.inner, i, θ, σ, NE._eta_column(H, k))
        for (k, i) in enumerate(subjects)]
    G = last.(pairs)
    return first.(pairs), H isa AbstractMatrix ? stack(G) : G
end

# The inner model, except that subject 1's density throws a `DomainError` where its
# first random effect exceeds `bound` in magnitude, as a Stan-semantics model does
# outside its domain; `hits` counts those evaluations. It declares no mu-referencing,
# so a population update never moves a chain's η.
struct BoundedFirst{M}
    inner::M
    bound::Float64
    hits::Base.RefValue{Int}
end
BoundedFirst(inner, bound) = BoundedFirst(inner, bound, Ref(0))
NE.nsubjects(m::BoundedFirst) = nsubjects(m.inner)
NE.nlme_layout(m::BoundedFirst) = nlme_layout(m.inner)
for f in (:conditional_loglikelihood, :conditional_loglikelihood_and_gradient,
        :conditional_loglikelihood_and_population_gradient)
    @eval function NE.$f(m::BoundedFirst, i::Integer, θ, σ, η)
        if i == 1 && abs(η[1]) > m.bound
            m.hits[] += 1
            throw(DomainError(η[1], "outside the model's domain"))
        end
        return NE.$f(m.inner, i, θ, σ, η)
    end
end

const LOCKSTEP_NUTS = WarmupHMCNUTS(max_tree_depth = 6)

function same_saem(a, b)
    @test a.theta == b.theta
    @test a.sigma == b.sigma
    @test a.omega == b.omega
    @test a.eta == b.eta
    @test a.nuts_divergences == b.nuts_divergences
end

@testset "SAEM NUTS: a batched model's chains in lockstep" begin
    growth = (growth_model(), [1.0, 0.5], [log(0.4)],
        omega_parameters(nlme_layout(growth_model()), [0.5 0.1; 0.1 0.2]))
    occasions = (occasions_model(), [1.9], [log(0.27)], [log(0.55), log(0.35)])
    @testset "$(nameof(typeof(first(case)))) fits equal the per-subject ones" for case in
            (growth, occasions)
        inner, θ0, σ0, ω0 = case
        for options in (SAEMOptions(iterations = 12, burnin = 5, mcmc_burnin = 2),
                SAEMOptions(iterations = 12, burnin = 5, mcmc_burnin = 2,
                    minibatch_size = 3))
            model = Lockstep(inner)
            fit(m) = saem(m, θ0, σ0, ω0; rng = Xoshiro(5), options, kernel = LOCKSTEP_NUTS)
            same_saem(fit(model), fit(inner))
            @test maximum(length, gradient_batches(model)) > 1
        end
    end

    @testset "Louis's information equals the per-subject one" begin
        inner, θ0, σ0, ω0 = growth
        model = Lockstep(inner)
        approximation = LouisInformation(draws = 40, burnin = 10, kernel = LOCKSTEP_NUTS)
        # `:s`: 40 draws are too few for a positive-definite `R` at this point.
        louis(m) = covariance_step(m, θ0, σ0, ω0; method = :s, approximation,
            rng = Xoshiro(2))
        a, b = louis(model), louis(inner)
        @test a.covariance == b.covariance
        @test maximum(length, gradient_batches(model)) == nsubjects(inner)
    end

    @testset "a leapfrog step is one batched call over the unfinished trees" begin
        inner = occasions_model(units = [2, 6, 15, 40])
        model = Lockstep(inner)
        ids = collect(1:nsubjects(model))
        θ, σ, ω = [1.9], [log(0.27)], [log(0.55), log(0.35)]
        prior = re_prior(model, ω)
        H = [zeros(NE.neta(nlme_layout(model), i)) for i in ids]
        states = NE._saem_nuts_start!(Xoshiro(1), LOCKSTEP_NUTS, model, ids, θ, σ, prior, H)
        serial_states, serial_H = deepcopy(states), deepcopy(H)
        for round in 1:5
            empty!(model.log)
            calls = inner.calls[]
            NE._saem_nuts_round!(states, model, ids, θ, σ, prior, H)
            lockstep_calls = inner.calls[] - calls
            batches = gradient_batches(model)
            @test length(batches) == length(model.log)
            # Subject i's chain takes part in the first c_i batches, and the round
            # makes as many batched calls as the longest chain needs.
            c = [count(b -> i in b, batches) for i in ids]
            @test length(batches) == maximum(c)
            @test all(b -> batches[b] == [i for i in ids if c[i] >= b], eachindex(batches))
            # The same transitions, evaluated per subject, make the same calls.
            calls = inner.calls[]
            NE._saem_nuts_round!(serial_states, inner, ids, θ, σ, prior, serial_H)
            @test inner.calls[] - calls == lockstep_calls == sum(c)
            @test H == serial_H
        end
    end

    @testset "a minibatch's chains evaluate within the batch" begin
        n, b, iterations = 24, 6, 20
        model = Lockstep(growth_model(n = n))
        _, θ0, σ0, ω0 = growth
        saem(model, θ0, σ0, ω0; rng = Xoshiro(1), kernel = LOCKSTEP_NUTS,
            options = SAEMOptions(; iterations, burnin = 8, mcmc_burnin = 2,
                minibatch_size = b))
        log = model.log
        # The start is checked on every subject, and the Pathfinder start and the
        # burn-in transitions move every chain, each burn-in round opening with all
        # of them. The numerical M-step's evaluations are per subject. So after the
        # last round over all chains, each SA iteration opens with the domain check
        # of exactly its batch (densities alone), and every transition of the
        # iteration evaluates within it.
        @test first(log) == (:value => 1:n)
        last_full = findlast(e -> e == (:gradient => 1:n), log)
        start = findnext(e -> first(e) === :value, log, last_full)
        current, checks, contained = Int[], 0, true
        for (kind, s) in log[start:end]
            if kind === :value
                length(s) == b && (current = s; checks += 1)
            end
            contained &= issubset(s, current)
        end
        @test contained
        @test checks == iterations
        @test all(e -> length(last(e)) <= b, log[start:end])
        @test sort!(unique(reduce(vcat, last.(log[start:end])))) == 1:n
    end

    @testset "a batch outside the domain is answered subject by subject" begin
        θ0, σ0 = [1.0, 0.5], [log(0.5)]
        ω0 = omega_parameters(GROWTH_LAYOUT, [0.8 0.0; 0.0 0.3])
        # Subject 1's random effect is excluded beyond 0.6, so its trajectories
        # reach positions that throw inside batches with the other subjects.
        excluded = BoundedFirst(growth_model(), 0.6)
        model = Lockstep(excluded)
        options = SAEMOptions(iterations = 12, burnin = 5, mcmc_burnin = 2)
        fit(m) = saem(m, θ0, σ0, ω0; rng = Xoshiro(3), options, kernel = LOCKSTEP_NUTS)
        lockstep, serial = fit(model), fit(excluded)
        @test excluded.hits[] > 0
        same_saem(lockstep, serial)
    end

    @testset "an error in a chain surfaces from the lockstep" begin
        inner = BoundedFirst(growth_model(), 0.6)
        model = Lockstep(inner)
        ids = collect(1:nsubjects(model))
        θ, σ = [1.0, 0.5], [log(0.5)]
        prior = re_prior(model, omega_parameters(GROWTH_LAYOUT, [0.8 0.0; 0.0 0.3]))
        H = zeros(2, length(ids))
        states = NE._saem_nuts_start!(Xoshiro(1), LOCKSTEP_NUTS, model, ids, θ, σ, prior, H)
        # A chain moved outside the domain cannot start a transition: WarmupHMC's
        # `ArgumentError`, as for a per-subject model.
        H[1, 1] = 0.9
        NE._saem_nuts_reposition!(states, H, [1])
        serial = deepcopy(states)
        @test_throws ArgumentError NE._saem_nuts_round!(states, model, ids, θ, σ, prior,
            copy(H))
        @test_throws ArgumentError NE._saem_nuts_round!(serial, inner, ids, θ, σ, prior,
            copy(H))
    end

    @testset "an interrupt ends the lockstep start" begin
        inner, θ0, σ0, ω0 = growth
        root = Treebars.initialize_progress!(:state; description = "stopped")
        Treebars.request_interrupt!(root)
        fit = saem(Lockstep(inner), θ0, σ0, ω0; rng = Xoshiro(1), kernel = LOCKSTEP_NUTS,
            options = SAEMOptions(iterations = 4, burnin = 2), progress = root)
        @test fit.interrupted && fit.iterations == 0 && fit.theta == θ0
    end
end
