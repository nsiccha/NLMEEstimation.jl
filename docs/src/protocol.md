# Model protocol

A model is any Julia value that supplies a conditional observation likelihood. NLMEEstimation supplies the subject-level random-effect prior, `ηᵢ ~ N(0, Ω)`, and represents Ω by blockwise log-Cholesky coordinates. The model supplies θ and σ in its own unconstrained coordinates.

Every model implements `nsubjects(model)` and `nlme_layout(model)`. It can use either evaluation style:

| Style | Required conditional likelihood | How estimators evaluate it |
| --- | --- | --- |
| `PerSubjectEvaluation()` | `conditional_loglikelihood(model, i, θ, σ, η)` | The package gathers subject calls into a batch. |
| `BatchedEvaluation()` | `conditional_loglikelihoods(model, subjects, θ, σ, H)` | One call evaluates the listed subjects; column `k` of `H` is that subject's η. |

The likelihood excludes the η prior. Derivatives may be provided directly or obtained through `ad_backend(model)` with DifferentiationInterface. Reverse-mode Enzyme is the tested backend. Per-subject η Hessian blocks are optional; set `hessian_ad_backend(model) = nothing` for a gradient-only model. [PPL and AD requirements](requirements.md) lists which derivatives each estimator uses; every method runs on first-order AD.

A model rejects a point outside its domain by returning `-Inf`, or by throwing a `DomainError`, `OverflowError` or `PosDefException` as a Stan-semantics model does (a matrix exponential of an overflowed rate, the Cholesky factor of an indefinite matrix). At a trial point the estimators treat both alike. Where each subject has its own trial point (EBE steps, MCMC proposals, quadrature points) and a batched evaluation throws, the subjects are evaluated one at a time to find which ones.

- `empirical_bayes` (Newton and BFGS, and so the EBEs inside `focei`) rejects that subject's trial step and tries a shorter one; Newton's final polishing step is dropped.
- `focei`'s search scores the evaluation as a very large objective.
- SAEM's MH and MALA kernels reject that subject's proposal. The NUTS kernel scores that position `-Inf`, exactly as if the model had returned it. SAEM's numerical and MAP M-steps halve their line-search step.
- `marginal_loglikelihood` gives that importance-sampling or quadrature point zero density.

The point a search starts from must evaluate finitely.

SAEM's chains are checked the same way after each population update. Under mu-referencing, the closed-form update moves each subject's φ = Dθ + η while its η stays where it was, so a chain can end up outside the domain. The MH and NUTS kernels then raise a `DomainError` naming those subjects, and the MALA kernel raises its own `DomainError`.

FOCE and FOCEI require `observation_model(model) = GaussianObservations()` and observation means, variances, and first η Jacobians. SAEM can use `mu_referencing(model)` for a closed-form update of the corresponding fixed effects and Ω; remaining population coordinates use a numerical M-step.

A batched model may keep its batched likelihood and η derivatives while using
subject-level population gradients in SAEM. Implement
`conditional_loglikelihood_and_population_gradient(model, i, θ, σ, η)` and
declare `saem_population_gradient_style(model) = PerSubjectEvaluation()`. SAEM
then sums those subject values and gradients for the numerical M-step. This
choice does not change `evaluation_style(model)` or any other estimator step.
For different routes in the two numerical M-step modes, define
`saem_population_gradient_style(model, ::Val{:online})` or
`saem_population_gradient_style(model, ::Val{:mixture})`. A missing mode-specific
method uses the one-argument style. This lets a model use compact subject-cell
gradients for the bounded-memory online update and a batched gradient for the
retained-draw mixture.

A model can hold a random-effect covariance family, or part of one, at a known value by declaring `fixed_omega(model)`. It returns `family => FixedOmega(…)` pairs, where a family is an `eta_blocks` entry of `NLMELayout` (or a `family_sizes` entry of `RaggedNLMELayout`) by position. `FixedOmega(Ω)` fixes the whole family, matching nlmixr2's `fix()` on an ω block and NONMEM's `OMEGA … FIX`. `FixedOmega(sd = [0.3, nothing], correlation = nothing)` fixes the listed standard deviations, and `FixedOmega(correlation = R)` fixes the correlation, in `Ω = D R D` terms. FOCE(I) searches only the free Ω coordinates. SAEM leaves a fully fixed family unchanged and maximizes the expected Gaussian random-effect density over a partially fixed family's free standard deviations and correlation. Both report the fixed values in `omega` and `omega_matrix`. Their starting `ω0` must hold the fixed values; `apply_fixed_omega(model, ω)` moves a vector onto them. `empirical_bayes` evaluates at the `ω` it is given.

Run `check_protocol(model, θ, σ, ω)` while implementing a model. It checks dimensions, agreement between evaluation styles, subject separability, available derivatives, declared observation and mu-referencing traits, and that `ω` holds any declared `fixed_omega` values.

The [PK comparison](pk-comparison.md) shows a complete batched BRM+RKPPL model and the calls used to fit it. The [API reference](api.md) lists every protocol entry point.
