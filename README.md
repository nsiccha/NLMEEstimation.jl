# NLMEEstimation.jl

Model-agnostic estimators for nonlinear mixed-effects (NLME) models.

A model is any Julia value that implements a small protocol: a conditional
log-density of the observations given the random effects η, with its derivatives in
η, and a parameter layout. The package owns the random-effect prior
`η_i ~ N(0, Ω)` and builds estimators on top of the protocol. Posthoc / empirical
Bayes and FOCE(I) population estimation are implemented; SAEM is next.

Derivatives come from automatic differentiation. A model either implements its
derivative methods itself, or declares an
[ADTypes](https://github.com/SciML/ADTypes.jl) backend that the package drives
through
[DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl).
What the package promises and tests is reverse-mode AD with
[Enzyme](https://github.com/EnzymeAD/Enzyme.jl) (`AutoEnzyme()`). Second order is
optional, and only ever requested as per-subject `neta × neta` blocks, never as a
joint Hessian.

## The protocol

A model declares how it evaluates, by setting `evaluation_style`:

- **`PerSubjectEvaluation()`** (default): the model implements
  `conditional_loglikelihood(model, i, θ, σ, η)`, the log-density for one subject.
- **`BatchedEvaluation()`**: the model implements
  `conditional_loglikelihoods(model, subjects, θ, σ, H)`, a joint program that
  evaluates many subjects in one pass. `H` holds one column of η per subject.

Estimators always call the batched entry points. Each style derives the other
style's methods. A batched model's η gradients come from one AD pass over the
batch's summed log-density, which works because subjects are conditionally
independent.

| Function | Status | Meaning |
|---|---|---|
| `nsubjects(model)` | required | number of subjects |
| `nlme_layout(model)` | required | `NLMELayout(ntheta, nsigma, eta_blocks)`: lengths of θ and σ, η block structure |
| `evaluation_style(model)` | optional | `PerSubjectEvaluation()` (default) or `BatchedEvaluation()` |
| `conditional_loglikelihood(model, i, θ, σ, η)` | required (per-subject) | `log p(y_i \| θ, σ, η_i)`, without the η prior |
| `conditional_loglikelihoods(model, subjects, θ, σ, H)` | required (batched) | the same for a batch of subjects |
| `conditional_loglikelihoods_and_gradients(model, subjects, θ, σ, H)` | derived from `ad_backend`, or implemented | values and η gradients of a batch |
| `conditional_loglikelihood_hessian_blocks(model, subjects, θ, σ, H)` | optional | per-subject η Hessian blocks (Hessian-vector products for batched models) |
| `ad_backend(model)`, `hessian_ad_backend(model)` | optional | backends for derivatives the model does not implement; `hessian_ad_backend = nothing` declares a gradient-only model |
| `provides_eta_hessian(model)` | optional | whether Hessian blocks are available |
| `conditional_loglikelihood_and_population_gradient(model, i, θ, σ, η)` | optional | `(ℓ, ∇θ ℓ, ∇σ ℓ)` |
| `mu_referencing(model)` | optional trait | `MuReferencing`: θ coordinates entering as `ϕ_i = X_i θ + η_i` (SAEM) |
| `observation_model(model)` | optional trait | `GaussianObservations()` for FOCE(I) |
| `observations(model, subjects)` | required for batched Gaussian models | one observation vector per subject, in batch order |
| `observation_moments(model, subjects, θ, σ, H)` | required for batched Gaussian models | one `(mean, variance)` pair per subject |
| `observation_moments_and_jacobians(model, subjects, θ, σ, H)` | derived per subject or implemented for a batch | one `(μ, v, Jμ, Jv)` tuple per subject; Jacobian rows follow observations and columns follow η |
| `parameter_names(model)` | optional | names for reporting |

Coordinates:

- θ and σ are in the model's unconstrained coordinates.
- Ω is parameterized by the package: for each block, the log-Cholesky coordinates of
  `Ω_b = L_b L_bᵀ` (`omega_matrix`, `omega_parameters`).

`check_protocol(model, θ, σ, ω)` checks an implementation against these contracts,
including batched-vs-per-subject consistency and the independence of subjects.

## Empirical Bayes estimates (posthoc)

```julia
using NLMEEstimation, NLMEEstimation.ReferenceModels
using ADTypes: AutoEnzyme
import Enzyme, Random

θ = [log(1.5), log(2.8), log(32.0), 0.75]   # log ka, log CL, log V, weight effect on CL
σ = [log(0.3), log(0.1)]                     # log additive sd, log proportional sd
Ω = [0.16 0 0; 0 0.09 0; 0 0 0.04]
subjects, η_true = simulate_one_compartment_oral(Random.Xoshiro(1), θ, σ, Ω; nsubjects = 20)
model = OneCompartmentOral(subjects; backend = AutoEnzyme())

ω = omega_parameters(nlme_layout(model), Ω)
r = empirical_bayes(model, θ, σ, ω)   # or posthoc(...)
r.modes        # η̂, one column per subject
r.hessians     # nothing for this gradient-only model
```

`empirical_bayes` solves all subjects in lockstep. Each iteration makes one batched
gradient call, plus batched value calls for its line search.

- **`NewtonEBE()`**: damped Newton on the Hessian blocks. This is the default when
  the model provides Hessian blocks.
- **`QuasiNewtonEBE()`**: BFGS on gradients alone. This is the default for
  gradient-only models.

It throws an `EBEConvergenceError` naming any subject without a mode, unless
`require_convergence = false`.

## FOCE(I) population estimation

```julia
at_start = focei_evaluate(model, θ, σ, ω)
fit = focei(model, θ, σ, ω; options = FOCEOptions(maxeval = 500))
fit.theta, fit.sigma, fit.omega_matrix  # estimated model coordinates and Ω
fit.ofv                                # -2 × approximate marginal log-likelihood
fit.ebes.modes                         # η̂ at the optimum
fit.converged, fit.evaluations
```

FOCE(I) uses only first η derivatives of the observation means and variances.
The outer BOBYQA search uses no gradients. For a joint-program model, implement
the three batched Gaussian methods in the table so one query can return the
moments and η Jacobians of all subjects. `FOCEOptions(interaction = false)`
omits the variance-Jacobian term for FOCE without interaction.
