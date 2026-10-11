# Fixed-parameter comparison of per-subject SAEM kernels on the public Theoph
# model. Run from the repository root with `julia --project=test bench/saem_kernels.jl`.
# Compilation and model construction are excluded from the timed region.
using NLMEEstimation
using Random
using LinearAlgebra
using Statistics
import NLMEEstimation as NE
include(joinpath(@__DIR__, "..", "test", "common.jl"))
using .TestCommon

BLAS.set_num_threads(1)

function ess_initial_positive(x)
    n = length(x)
    centered = x .- mean(x)
    variance = sum(abs2, centered) / n
    variance == 0 && return 0.0
    tau = 1.0
    for lag in 1:2:(n - 2)
        pair = (dot(view(centered, 1:(n-lag)), view(centered, (lag+1):n)) +
            dot(view(centered, 1:(n-lag-1)), view(centered, (lag+2):n))) / (n * variance)
        pair <= 0 && break
        tau += 2pair
    end
    return n / tau
end

function summarize_draws(draws)
    m, n, _ = size(draws)
    return median([ess_initial_positive(vec(view(draws, j, i, :)))
        for j in 1:m, i in 1:n])
end

function standard_mh_draws(rng, model, θ, σ, ω; burnin=200, n_draws=800)
    ids = collect(1:nsubjects(model))
    prior = re_prior(model, ω)
    m, n = neta(nlme_layout(model)), length(ids)
    H = zeros(m, n)
    ℓ = conditional_loglikelihoods(model, ids, θ, σ, H)
    draws = Array{Float64}(undef, m, n, n_draws)
    logscale = log(0.5)
    accepted_prior = accepted_rw = 0
    for iteration in 1:(burnin + n_draws)
        ap, ar = NE._saem_mh_round!(rng, model, ids, θ, σ, prior, H, ℓ, exp(logscale))
        accepted_prior += ap
        accepted_rw += ar
        if iteration <= burnin
            logscale = clamp(logscale + (ar / n - 0.3) / sqrt(iteration),
                log(1e-3), log(1e2))
        else
            draws[:, :, iteration - burnin] = H
        end
    end
    return (draws=draws, acceptance=(prior=accepted_prior / (n * (burnin + n_draws)),
        random_walk=accepted_rw / (n * (burnin + n_draws))), scale=exp(logscale))
end

if abspath(PROGRAM_FILE) == @__FILE__
    model = theoph_model()
    ω = omega_parameters(nlme_layout(model), Ω_THEOPH)
    standard_mh_draws(MersenneTwister(1), model, θ_THEOPH, σ_THEOPH, ω;
        burnin=5, n_draws=5) # compile
    trial = @timed standard_mh_draws(MersenneTwister(2026), model,
        θ_THEOPH, σ_THEOPH, ω)
    result = trial.value
    ess = summarize_draws(result.draws)
    println("kernel\tseconds\tmedian_ess\tmedian_ess_per_second\tprior_acceptance\trw_acceptance")
    println(join(("standard_MH", trial.time, ess, ess / trial.time,
        result.acceptance.prior, result.acceptance.random_walk), '\t'))
end
