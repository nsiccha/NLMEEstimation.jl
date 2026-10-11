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

The complete runnable script, method options, fixed seeds, and measured results will be added here with the BRM adapter validation.
