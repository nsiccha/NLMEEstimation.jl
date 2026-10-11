# PK comparison

This example uses one synthetic IV-bolus, one-compartment dataset to compare every implemented NLMEEstimation method with a full joint WarmupHMC fit. The conditional model is written with BayesianRegressionModels.jl (`@brm`), which lowers its plate to ReactiveKernelsPPL.jl. Every fit uses the same observations and subject covariates.

## Model and data

Eight subjects each receive a dose of 100 and have observations at 0.5, 1, 2, 4, and 8 time units. Centered subject weights range from −0.8 to 0.8. The subject-level log clearance has an intercept, a weight slope, and a random effect; log volume has an intercept and a random effect. For subject `i` at time `t`, the predicted concentration is

```math
\widehat c_i(t) = \frac{100}{V_i}\exp\!\left(-\frac{CL_i}{V_i}t\right).
```

The Gaussian observation standard deviation is `σ + 0.03 × predicted concentration`. Its dependence on the prediction makes the FOCE versus FOCEI interaction term visible in this example. Data are simulated with seed `20261011`, log-clearance intercept `0.2`, weight slope `0.25`, log-volume intercept `1.6`, random-effect standard deviations `0.3` and `0.15`, and additive standard deviation `0.6`.

The conditional program and its priors are defined in the BRM model itself. `RKBRMI` lowers the `@plate` body through RKPPL, and `brm_nlme_model` supplies NLMEEstimation's batched model protocol:

```julia
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using NLMEEstimation, Distributions
using DifferentiationInterface: AutoEnzyme
import Enzyme

const AD = AutoEnzyme(; mode=Enzyme.Reverse)

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

backend = RKBRMI(pk_brmi(data))
model = brm_nlme_model(backend; ad_backend=AD)
```

The production lowering emits an RKPPL program with the correlated subject effects and both observation plates. The concentration and likelihood cells in that emitted program are:

```julia
@plate for i = eachindex(dose)
    CL = exp(log_CL[i])
    V = exp(log_V[i])
    pred_cell = (dose[i] / V) .* exp.(-(CL / V) .* t[i])
    pred_cells[i] = pred_cell
    pred_argument_dv_1[i] = pred_cell
    pred_argument_dv_2[i] = sigma .+ 0.03 .* pred_cell
end
pred = brm_flatten_cells(pred_cells)
@plate for i = eachindex(dv)
    dv[i] .~ Normal.(pred_argument_dv_1[i], pred_argument_dv_2[i])
end
```

The [full emitted source](https://github.com/nsiccha/NLMEEstimation.jl/blob/master/docs/examples/pk_emitted_rkppl.jl) includes BRM's helper submodels and population-effect construction. The [comparison script](https://github.com/nsiccha/NLMEEstimation.jl/blob/master/docs/examples/brm_rkppl_pk.jl) generates that source from `emit_rk_artifact`, runs the public translator, and then fits the NLME and full joint models.

## What is compared

| Method | Population treatment | Subject effects |
| --- | --- | --- |
| Posthoc empirical Bayes | Fixed supplied population point | Conditional modes |
| FOCE | Approximate marginal likelihood without interaction | Conditional modes at the fit |
| FOCEI | Approximate marginal likelihood with interaction | Conditional modes at the fit |
| SAEM with MH | Stochastic approximation | Per-subject MH chains |
| SAEM with WarmupHMC NUTS | Stochastic approximation | Per-subject NUTS chains |
| Full joint WarmupHMC | Joint posterior for population and subject parameters | Joint MCMC draws |

The posthoc result is conditional on its supplied population parameters; it does not estimate them. FOCE, FOCEI, and SAEM estimate population parameters but use different approximations and sampling schedules. The full joint fit provides a posterior comparison on the same data. Population point estimates and subject effects are the useful quantities to compare across rows; the methods' raw objective values are not interchangeable.

The recorded SAEM rows used the historical running-mixture numerical M-step every third iteration. The script spells out `numeric_mstep=:mixture` so those pinned results remain reproducible. They do not measure the textbook every-iteration default.

Here are the fitting calls from the [complete script](https://github.com/nsiccha/NLMEEstimation.jl/blob/master/docs/examples/brm_rkppl_pk.jl). The initial `θ0`, `σ0`, and `ω0` encode the generating population point in the model's declared parameter order; `m`, `backend`, and `AD` come from the model setup above.

```julia
using Random, WarmupHMC

θ0 = [Dict("log_V_Intercept" => 1.6, "pop_log_CL.beta_pop.1" => 0.2,
    "pop_log_CL.beta_pop.2" => 0.25)[name] for name in parameter_names(m).theta]
σ0 = [log(0.6)]
ω0 = omega_parameters(nlme_layout(m), [0.3^2 0.0; 0.0 0.15^2])

posthoc = empirical_bayes(m, θ0, σ0, ω0)
foce = focei(m, θ0, σ0, ω0;
    options=FOCEOptions(maxeval=400, interaction=false))
focei_fit = focei(m, θ0, σ0, ω0;
    options=FOCEOptions(maxeval=400, interaction=true))

saem_mh = saem(m, θ0, σ0, ω0; rng=Xoshiro(5),
    options=SAEMOptions(iterations=100, burnin=50, mcmc_burnin=15,
        adapt_iterations=50, numeric_maxiter=8, mstep_interval=3,
        numeric_mstep=:mixture),
    kernel=StandardSAEMMH())
saem_nuts = saem(m, θ0, σ0, ω0; rng=Xoshiro(6),
    options=SAEMOptions(iterations=40, burnin=20, mcmc_burnin=0,
        numeric_maxiter=8, mstep_interval=3, numeric_mstep=:mixture),
    kernel=WarmupHMCNUTS(warmup_draws=12, warmup_evaluations=50,
        stepsize_adaptation_limit=10, max_tree_depth=5))

problem = rk_logdensity_problem(backend; ad_backend=AD)
chains = [WarmupHMC.adaptive_warmup_mcmc(Xoshiro(seed), problem;
    n_draws=500, target_acceptance_rate=0.9,
    nonlinear_adapt=false, monitor_ess=false) for seed in (7, 8)]
```

## Reproduce the comparison

Use Julia 1.10.12 from the repository root. The separate [example project](https://github.com/nsiccha/NLMEEstimation.jl/blob/master/docs/examples/Project.toml) pins its registered direct dependencies; the setup script checks out the source packages at the exact revisions used for this run and develops the nested ReactiveKernels packages together. Choose a fresh directory for those clones:

```sh
julia --project=docs/examples docs/examples/setup_env.jl /path/to/example-deps
julia --project=docs/examples docs/examples/brm_rkppl_pk.jl
```

The [setup script](https://github.com/nsiccha/NLMEEstimation.jl/blob/master/docs/examples/setup_env.jl) pins BayesianRegressionModels `35c2258f`, NLMEEstimation `527a593c`, ReactiveKernels `a2268e47`, WarmupHMC `6cd665de`, StanBlocks `75d9feb4`, and Treebars `7426b9c8`. The comparison script uses `Xoshiro(20261011)` for the data and fixed seeds for the fitting methods. It renders the exact emitted RKPPL program as `emitted_rkppl` and calls the public production translator on the artifact. The rendered program is included in the linked source file above.

## Results from the pinned run

`CL₀` and `V₀` are the typical clearance and volume at centered weight zero and subject effect zero. `σbase` is the residual baseline standard deviation; `sd_CL` and `sd_V` are the subject-effect standard deviations. All four are reported on their natural scales. The posthoc method has no population row because its population point is supplied rather than fitted.

| Method | CL₀ | V₀ | Weight slope | σbase | sd_CL | sd_V |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Generating truth | 1.221 | 4.953 | 0.250 | 0.600 | 0.300 | 0.150 |
| FOCE | 0.968 | 5.433 | 0.092 | 0.459 | 0.234 | 0.166 |
| FOCEI | 0.968 | 5.433 | 0.092 | 0.459 | 0.234 | 0.166 |
| SAEM MH | 0.951 | 5.392 | 0.203 | 0.414 | 0.244 | 0.165 |
| SAEM NUTS | 0.964 | 5.409 | 0.294 | 0.454 | 0.261 | 0.171 |
| Full joint WarmupHMC posterior mean | 0.967 | 5.456 | 0.066 | 0.487 | 0.357 | 0.228 |

The next table compares subject 1 effects on the log-clearance and log-volume scales. The first five rows are conditional modes: posthoc is evaluated at the stated generating population point, FOCE and FOCEI use their fit EBEs, and the SAEM rows recompute EBEs at each fitted population point. The joint MCMC row is a posterior **mean** of each draw's constrained subject effect.

| Method and summary | η log CL | η log V |
| --- | ---: | ---: |
| Posthoc at truth, conditional mode | −0.030 | 0.100 |
| FOCE, conditional mode | 0.064 | 0.018 |
| FOCEI, conditional mode | 0.064 | 0.018 |
| SAEM MH, recomputed conditional mode | 0.165 | 0.024 |
| SAEM NUTS, recomputed conditional mode | 0.219 | 0.021 |
| Full joint WarmupHMC posterior mean | 0.049 | 0.017 |

FOCE used `maxeval=400` and converged after 148 evaluations, with objective value 136.8283. FOCEI enabled `interaction=true`, converged after 144 evaluations, and had objective value 136.8548. Their displayed estimates round to the same three decimals; the objective values retain their respective interaction definitions.

SAEM MH ran 100 iterations with 50 burn-in iterations, 15 MCMC burn-in transitions, 50 proposal-adaptation iterations, and an M-step every three iterations. Its prior-proposal acceptance was 0.0837 and random-walk acceptance was 0.2391. SAEM NUTS ran 40 iterations with 20 burn-in iterations and an M-step every three iterations; each subject kernel used 12 warmup draws, 50 warmup evaluations, step-size adaptation limit 10, and maximum tree depth 5. It recorded zero fixed-kernel divergences.

Full joint WarmupHMC used two chains with 500 posterior draws each and no divergences. The rank-normalized R̂ values for the log-volume intercept, log-clearance intercept, and clearance weight slope were 1.0061, 1.0000, and 1.0109; their bulk effective sample sizes were 564.9, 490.0, and 438.6. The script constrains each joint draw before averaging clearance, volume, residual SD, random-effect SDs, or subject effects. These finite-run point estimates and Bayesian posterior means are different estimands. Eight subjects keep the example runnable; this table is a workflow comparison, not an accuracy ranking.
