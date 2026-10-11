# Synthetic one-compartment IV-bolus NLME comparison. See pinned commits in the
# accompanying results page. Run with the docs/examples environment.
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using NLMEEstimation, WarmupHMC, Distributions, Random, Statistics, LinearAlgebra, Printf
using DifferentiationInterface: AutoEnzyme
using MCMCDiagnosticTools: rhat, ess
import Enzyme

const AD = AutoEnzyme(; mode=Enzyme.Reverse)
report(args...) = (println(args...); flush(stdout))
rng = Xoshiro(20261011)
n = 8
weight = collect(range(-0.8, 0.8; length=n))
dose = fill(100.0, n)
t = [Float64[0.5, 1, 2, 4, 8] for _ in 1:n]
eta_cl_true = 0.3 .* randn(rng, n)
eta_v_true = 0.15 .* randn(rng, n)
dv = map(1:n) do i
    cl = exp(0.2 + 0.25weight[i] + eta_cl_true[i])
    v = exp(1.6 + eta_v_true[i])
    pred = dose[i] / v .* exp.(-(cl / v) .* t[i])
    pred .+ (0.6 .+ 0.03 .* pred) .* randn(rng, length(t[i]))
end
data = (; subject=["s$i" for i in 1:n], weight, dose, t, dv)

function pk_brmi(data)
    @brm data begin
        sigma ~ Exponential(1)
        log_CL ~ 1 + weight + (1 | pk | subject)
        log_V ~ 1 + (1 | pk | subject)
        effect(log_CL, Intercept) ~ Normal(0.2, 1)
        effect(log_CL, weight) ~ Normal(0, 0.5)
        effect(log_V, Intercept) ~ Normal(1.6, 1)
        sd(:, pk) ~ Exponential(1)
        @plate for i in eachindex(log_CL)
            CL = exp(log_CL[i])
            V = exp(log_V[i])
            pred[i] = dose[i] / V .* exp.(-(CL / V) .* t[i])
            dv[i] ~ normal(pred[i], sigma .+ 0.03 .* pred[i])
        end
    end
end
brmi = pk_brmi(data)
artifact = BayesianRegressionModels.emit_rk_artifact(brmi; case_id="nlmee-pk")
# This is the exact production RKPPL emission, including submodel definitions.
# The public translator also verifies that the printed program is executable.
BayesianRegressionModels.rk_translate_artifact(artifact)
emitted_rkppl = sprint(Base.show_unquoted,
    Expr(:block, artifact.defs..., artifact.ast))
backend = RKBRMI(brmi)
m = brm_nlme_model(backend; ad_backend=AD)
name_to_value(θ) = Dict(zip(parameter_names(m).theta, θ))
θ0 = [Dict("log_V_Intercept" => 1.6, "pop_log_CL.beta_pop.1" => 0.2,
    "pop_log_CL.beta_pop.2" => 0.25)[name] for name in parameter_names(m).theta]
σ0 = [log(0.6)]
Ω0 = [0.3^2 0.0; 0.0 0.15^2]
ω0 = omega_parameters(nlme_layout(m), Ω0)
@assert check_protocol(m, θ0, σ0, ω0; η=zeros(2, n))
report("θ names = ", parameter_names(m).theta)
report("η names = ", parameter_names(m).eta)

# Posthoc is conditional on a stated population point; it does not estimate θ.
ebe = empirical_bayes(m, θ0, σ0, ω0)
report("posthoc: converged=", count(ebe.converged), "/", n,
    " subject1_eta=", ebe.modes[:, 1])

foce = focei(m, θ0, σ0, ω0; options=FOCEOptions(maxeval=400, interaction=false))
report("FOCE: converged=", foce.converged, " evaluations=", foce.evaluations,
    " OFV=", foce.ofv)
focei_fit = focei(m, θ0, σ0, ω0; options=FOCEOptions(maxeval=400, interaction=true))
report("FOCEI: converged=", focei_fit.converged, " evaluations=", focei_fit.evaluations,
    " OFV=", focei_fit.ofv)

saem_mh = saem(m, θ0, σ0, ω0; rng=Xoshiro(5),
    options=SAEMOptions(iterations=100, burnin=50, mcmc_burnin=15,
        adapt_iterations=50, numeric_maxiter=8, mstep_interval=3),
    kernel=StandardSAEMMH())
report("SAEM MH: acceptance=", saem_mh.acceptance)
saem_nuts = saem(m, θ0, σ0, ω0; rng=Xoshiro(6),
    options=SAEMOptions(iterations=40, burnin=20, mcmc_burnin=0,
        numeric_maxiter=8, mstep_interval=3),
    kernel=WarmupHMCNUTS(warmup_draws=12, warmup_evaluations=50,
        stepsize_adaptation_limit=10, max_tree_depth=5))
report("SAEM NUTS: divergences=", saem_nuts.nuts_divergences)

problem = rk_logdensity_problem(backend; ad_backend=AD)
chains = [WarmupHMC.adaptive_warmup_mcmc(Xoshiro(seed), problem;
    n_draws=500, target_acceptance_rate=0.9,
    nonlinear_adapt=false, monitor_ess=false) for seed in (7, 8)]
report("joint MCMC: draws=", size(chains[1].posterior_position, 2),
    " per chain; divergences=", [c.n_divergent_samples for c in chains])
population_chains = permutedims(cat(
    (c.posterior_position[m.theta, :] for c in chains)...; dims=3), (2, 3, 1))
report("joint MCMC theta rank Rhat=", rhat(population_chains),
    " bulk ESS=", ess(population_chains))
U = hcat((c.posterior_position for c in chains)...)
θdraw = U[m.theta, :]
σdraw = U[m.sigma, :]
@assert m.view.coordinates[m.theta] == Symbol.(parameter_names(m).theta)
mean_θ = vec(mean(θdraw; dims=2))
# tau is the constrained random-effect SD vector in the emitted RK block.
restored = [ReactiveKernelsPPL.constrain(backend.model.layout, U[:, j]) for j in axes(U, 2)]
mean_tau = vec(mean(reduce(hcat, (c.b_pk_subject.tau for c in restored)); dims=2))
mean_eta1 = vec(mean(reduce(hcat, (begin
    b = c.b_pk_subject
    H = b.z * (b.tau .* b.L)'
    H[1, :]
end for c in restored)); dims=2))
report("joint MCMC subject1_eta posterior mean=", mean_eta1)

# Compare individual random effects without confusing a SAEM chain state with
# an empirical Bayes estimate. These five rows are conditional modes; the
# joint MCMC row averages the natural η effect over posterior draws.
saem_mh_ebe = empirical_bayes(m, saem_mh.theta, saem_mh.sigma,
    saem_mh.omega; subjects=[1]).modes[:, 1]
saem_nuts_ebe = empirical_bayes(m, saem_nuts.theta, saem_nuts.sigma,
    saem_nuts.omega; subjects=[1]).modes[:, 1]
report("subject 1 random effects: log CL, log V")
for (label, eta) in (("posthoc @ truth (mode)", ebe.modes[:, 1]),
        ("FOCE (mode)", foce.ebes.modes[:, 1]),
        ("FOCEI (mode)", focei_fit.ebes.modes[:, 1]),
        ("SAEM MH posthoc (mode)", saem_mh_ebe),
        ("SAEM NUTS posthoc (mode)", saem_nuts_ebe),
        ("joint MCMC (mean)", mean_eta1))
    @printf("%-25s %8.3f %8.3f\n", label, eta[1], eta[2])
end
flush(stdout)

function table_line(label, θ, σ, sd)
    d = name_to_value(θ)
    @printf("%-12s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n", label,
        exp(d["pop_log_CL.beta_pop.1"]), exp(d["log_V_Intercept"]),
        d["pop_log_CL.beta_pop.2"], exp(σ[1]), sd[1], sd[2])
    flush(stdout)
end
report("method           CL0       V0   βweight    σbase    sd_CL     sd_V")
for (label, f) in (("FOCE", foce), ("FOCEI", focei_fit),
        ("SAEM MH", saem_mh), ("SAEM NUTS", saem_nuts))
    table_line(label, f.theta, f.sigma, sqrt.(diag(f.omega_matrix)))
end
# Nonlinear transformed coordinates are averaged per draw, not transformed
# after taking means in unconstrained space.
mean_cl0 = mean(exp.(θdraw[findfirst(==("pop_log_CL.beta_pop.1"), parameter_names(m).theta), :]))
mean_v0 = mean(exp.(θdraw[findfirst(==("log_V_Intercept"), parameter_names(m).theta), :]))
mean_slope = mean(θdraw[findfirst(==("pop_log_CL.beta_pop.2"), parameter_names(m).theta), :])
@printf("%-12s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f\n", "joint MCMC",
    mean_cl0, mean_v0, mean_slope, mean(exp.(σdraw[1, :])), mean_tau[1], mean_tau[2])
