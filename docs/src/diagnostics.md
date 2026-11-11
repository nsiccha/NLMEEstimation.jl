# Diagnostics and uncertainty

Interpret observation predictions, residuals and uncertainty calculations without conflating them with the population optimizer's stopping criteria. Each diagnostic concerns a different approximation or source of variation.

## Predictions and residuals

[`residual_diagnostics`](@ref) requires a Gaussian observation model and uses subject modes at the supplied population parameters. Its output follows the requested subject order and, within a subject, observation order.

| Field | Interpretation |
|:--|:--|
| `pred` | Observation mean with subject effects set to zero |
| `ipred` | Observation mean at the subject mode |
| `iwres` | Observation minus `ipred`, divided by its conditional standard deviation |
| `cpred` | Population mean of the first-order expansion around the subject mode |
| `cwres` | Residuals decorrelated with the linearized within-subject covariance |
| `cwres_per_observation` | Each conditional residual standardized separately, without decorrelation |
| `npde` | Simulation-based prediction discrepancies after decorrelation |

The distinction between the two conditional residual fields matters when comparing another implementation. A per-observation standardized residual need not be independent of another observation from the same subject.

The cached PK result allows the IWRES formula to be checked without another fit:

```@example diagnostics
using Main.DocsSupport
s = DocsSupport.cache()["fit"]
residual_sd = exp(only(s["sigma"]))
computed = (s["observed"][1] .- s["ipred"][1]) ./ residual_sd
println(isapprox(computed, s["iwres"][1]))
```

This equality checks the meaning of the field for the additive residual model used there. It does not test distributional assumptions. Residual plots should be inspected against time, predictions and relevant covariates, and interpreted with the fitting process in mind.

`replicates=0` skips NPDE simulation. Otherwise, the simulation count must exceed each subject's number of observations. Supply an explicit RNG for reproducibility. Increasing the count reduces simulation noise; it does not repair a misspecified observation model.

## Population uncertainty

[`covariance_step`](@ref) estimates uncertainty in the fitted population coordinates. Its result carries the free-coordinate mapping and the approximation used. Standard errors on logarithmic coordinates are not standard errors in physical units. [`delta_method`](@ref) propagates covariance through a differentiable transformation, and [`omega_elements`](@ref) provides covariance-matrix elements for that purpose.

The covariance step and the optimizer can fail independently. Near a covariance boundary or an unidentified direction, a local quadratic approximation may be singular or unstable even after the optimizer stops. Inspect the failure reported by [`CovarianceStepError`](@ref) and consider the model's parameterization and information content before changing numerical tolerances.

[`ebe_covariances`](@ref) instead describes local uncertainty in subject effects at fixed population parameters. It does not include uncertainty in the population estimates. Its curvature requirement also means that nonsmooth subject modes cannot be treated as ordinary Gaussian approximations.

## Marginal likelihood

[`marginal_loglikelihood`](@ref) evaluates the likelihood using the selected approximation. FOCE, importance sampling and adaptive Gauss–Hermite quadrature are distinct approximations and should be labelled when reported.

```@example diagnostics
using NLMEEstimation
laplace = AdaptiveGaussHermite(nodes=1)
println((method=typeof(laplace), nodes_per_dimension=laplace.nodes))
```

A single adaptive quadrature node gives the Laplace approximation using the exact subject curvature, when it exists. That is different from FOCE's first-order observation-moment curvature. Higher quadrature orders become expensive as the subject-effect dimension grows. Importance sampling introduces Monte Carlo error and needs a proposal with adequate coverage.

The [oral PK example](oral-pk.md) displays population and individual predictions, numerical convergence and IWRES, while stating which uncertainty calculations it omits.
