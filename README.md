# NLMEEstimation.jl

Model-agnostic estimators for nonlinear mixed-effects (NLME) models.

A model is any Julia value that implements a small protocol: a per-subject
conditional log-density with its derivatives in the random effects, and a
parameter layout. The package owns the random-effect prior `η_i ~ N(0, Ω)` and
builds estimators on top: posthoc / empirical Bayes estimates now, FOCEI (with
exact Laplace) and SAEM next.

Every derivative comes from automatic differentiation. A model either implements
its derivative methods (for example with Enzyme) or declares an
[ADTypes](https://github.com/SciML/ADTypes.jl) backend that the package drives
through [DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl).
Finite-difference backends are refused.

## The protocol

| Function | Status | Meaning |
|---|---|---|
| `nsubjects(model)` | required | number of subjects |
| `nlme_layout(model)` | required | `NLMELayout(ntheta, nsigma, eta_blocks)`: lengths of θ and σ, η block structure |
| `conditional_loglikelihood(model, i, θ, σ, η)` | required | `log p(y_i \| θ, σ, η_i)`, without the η prior |
| `conditional_loglikelihood_and_gradient(model, i, θ, σ, η)` | required or via `ad_backend` | `(ℓ, ∇η ℓ)` |
| `conditional_loglikelihood_gradient_and_hessian(model, i, θ, σ, η)` | required or via `hessian_ad_backend` | `(ℓ, ∇η ℓ, ∇²η ℓ)` |
| `conditional_loglikelihood_and_population_gradient(model, i, θ, σ, η)` | optional | `(ℓ, ∇θ ℓ, ∇σ ℓ)` |
| `ad_backend(model)`, `hessian_ad_backend(model)` | optional | AD backend for the derivative methods not implemented |
| `mu_referencing(model)` | optional trait | `MuReferencing`: θ coordinates entering as `ϕ_i = X_i θ + η_i` (SAEM) |
| `observation_model(model)` | optional trait | `GaussianObservations()` plus `observations` / `observation_moments` (FOCE(I)) |
| `parameter_names(model)` | optional | names for reporting |

θ and σ are in the model's unconstrained coordinates. Ω is parameterized by the
package: per block, the log-Cholesky coordinates of `Ω_b = L_b L_bᵀ`
(`omega_matrix`, `omega_parameters`). `check_protocol(model, θ, σ, ω)` checks an
implementation against these contracts.

## Empirical Bayes estimates (posthoc)

```julia
using NLMEEstimation, NLMEEstimation.ReferenceModels
using DifferentiationInterface: AutoForwardDiff
import ForwardDiff, Random

θ = [log(1.5), log(2.8), log(32.0), 0.75]   # log ka, log CL, log V, weight effect on CL
σ = [log(0.3), log(0.1)]                     # log additive sd, log proportional sd
Ω = [0.16 0 0; 0 0.09 0; 0 0 0.04]
subjects, η_true = simulate_one_compartment_oral(Random.Xoshiro(1), θ, σ, Ω; nsubjects = 20)
model = OneCompartmentOral(subjects; backend = AutoForwardDiff())

ω = omega_parameters(nlme_layout(model), Ω)
r = empirical_bayes(model, θ, σ, ω)   # or posthoc(...)
r.modes        # η̂_i
r.hessians     # ∇²η log p(y_i, η_i | θ, σ, Ω) at η̂_i
ebe_covariances(r)
```

`empirical_bayes` runs a Levenberg–Marquardt-damped Newton method per subject on
the AD gradient and Hessian. It throws an `EBEConvergenceError` naming any subject
without a mode, unless `require_convergence = false`.
