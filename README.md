# NLMEEstimation.jl

[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://nsiccha.github.io/NLMEEstimation.jl/dev/)

Model-agnostic estimators for nonlinear mixed-effects (NLME) models.

A model is any Julia value that implements a small protocol: a conditional
log-density of the observations given the random effects η, with its derivatives in
η, and a parameter layout. The package owns the random-effect prior
`η_i ~ N(0, Ω)` and builds estimators on top of the protocol. Posthoc empirical
Bayes, FOCE(I), and SAEM population estimation are implemented.

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

Estimator likelihood queries call the batched entry points. Each style derives
the other style's methods. A batched model's η gradients come from one AD pass
over the batch's summed log-density, which works because subjects are
conditionally independent. SAEM's numerical population M-step uses the
per-subject population-gradient method for per-subject models and one batched
AD pass for batched models.

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
| `population_logprior(model, θ, σ, ω)` | optional | log prior on unconstrained population coordinates; `nothing` (default) selects maximum likelihood |
| `population_logprior_and_gradient(model, θ, σ, ω)` | optional | prior value and gradients in θ, σ, ω; derived using `ad_backend` unless implemented |
| `population_prior_provenance(model)` | optional | description carried in population-fit results |
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
fit.ofv                                # -2 × fitted ML or MAP objective
fit.ebes.modes                         # η̂ at the optimum
fit.objective, fit.loglikelihood, fit.logposterior
fit.converged, fit.evaluations
```

FOCE(I) uses only first η derivatives of the observation means and variances.
The outer BOBYQA search uses no gradients. For a joint-program model, implement
the three batched Gaussian methods in the table so one query can return the
moments and η Jacobians of all subjects. `FOCEOptions(interaction = false)`
omits the variance-Jacobian term for FOCE without interaction.

By default FOCE and SAEM estimate by maximum likelihood (`objective == :ML`).
To request a maximum a posteriori (MAP) population point, implement
`population_logprior(model, θ, σ, ω)` and return a real log density. This is a
single prior for the whole population fit; it does not belong in each subject's
conditional likelihood. The density must be expressed with respect to the
unconstrained coordinates. For example, a prior on a positive parameter
`a = exp(θ[1])` needs the `+θ[1]` log-Jacobian term. The package's Gaussian
random-effect prior is already included separately. A model can also implement
`population_logprior_and_gradient` directly and name its prior through
`population_prior_provenance`. FOCE keeps `loglikelihood` separate from
`logposterior`; its `ofv` optimizes `-2logposterior` in MAP mode.

## SAEM population estimation

```julia
using Random

saem_fit = saem(model, θ, σ, ω; rng = Xoshiro(2026),
    options = SAEMOptions(iterations = 300, burnin = 150))
saem_fit.theta, saem_fit.sigma, saem_fit.omega_matrix
saem_fit.eta                 # final subject-chain states
saem_fit.objective, saem_fit.population_logprior, saem_fit.prior_provenance
posthoc_fit = empirical_bayes(model, saem_fit.theta, saem_fit.sigma,
    saem_fit.omega; η0 = saem_fit.eta)  # subject EBEs at the fitted point
saem_fit.acceptance          # prior and random-walk MH acceptance fractions
```

For per-subject NUTS transitions, select `WarmupHMCNUTS()`:

```julia
nuts_fit = saem(model, θ, σ, ω; rng = Xoshiro(2026),
    options = SAEMOptions(iterations = 300, burnin = 150),
    kernel = WarmupHMCNUTS())
nuts_fit.kernel, nuts_fit.nuts_divergences
```

WarmupHMC warms up each subject once, initialized with a covariance derived
from the local η Hessian when the model provides one. The default uses its
diagonal; `WarmupHMCNUTS(metric = :dense)` passes the inverse regularized
Hessian. SAEM then advances each
subject by fixed-kernel NUTS transitions as the population parameters change.
The warmup and stream files are temporary for the duration of the fit. NUTS
uses `Xoshiro` streams internally, seeded from the supplied RNG; its MH-only
`acceptance` and `rw_scale` fields are `missing`. See
[`bench/RESULTS.md`](bench/RESULTS.md) for MH/NUTS sampling and fit benchmarks.

SAEM keeps one η chain per subject. Each MH round evaluates prior-independence
and random-walk proposals in two model batches, with separate accept/reject
decisions per subject. The random-walk scale adapts for a finite initial period.
After an optional MCMC burn-in, stochastic approximation averages latent-effect
statistics with unit steps during exploration and decreasing steps thereafter.

When a population prior is present, SAEM includes it once in a joint numerical
M-step over θ, σ, and ω. The step combines the conditional objective, running
Gaussian random-effect statistics, and the population prior, so priors on
mu-referenced fixed effects and covariance coordinates also affect the estimate.
The `:online` numerical mode blends the joint update with the SA gain; the
`:mixture` mode optimizes the weighted retained-draw objective. The reported
point and final subject states are estimates, not samples from a population
posterior.

`saem_fit.eta` contains the final Markov-chain states, not EBEs; call
`empirical_bayes` as above when subject posterior modes are needed for app
reporting or initialization.

Without a population prior, `mu_referencing(model)` lets the M-step update its
θ coordinates by generalized least squares and Ω within `eta_blocks`; other θ
coordinates and σ use first-order AD. The `:mixture` numerical mode retains
weighted latent draws, so its memory and evaluation cost grow with iterations.
The `:online` mode keeps the current draw and applies the SA gain to its
numerical update. `:auto` selects between them from the declared
mu-referencing coverage. `mstep_interval` can reduce evaluation frequency.
SAEM estimates depend on the RNG stream and run length, so assess recovery
across starts for a new model.
