# Model protocol

A model is any Julia value that supplies a conditional observation likelihood. NLMEEstimation supplies the subject-level random-effect prior, `ηᵢ ~ N(0, Ω)`, and represents Ω by blockwise log-Cholesky coordinates. The model supplies θ and σ in its own unconstrained coordinates.

Every model implements `nsubjects(model)` and `nlme_layout(model)`. It can use either evaluation style:

| Style | Required conditional likelihood | How estimators evaluate it |
| --- | --- | --- |
| `PerSubjectEvaluation()` | `conditional_loglikelihood(model, i, θ, σ, η)` | The package gathers subject calls into a batch. |
| `BatchedEvaluation()` | `conditional_loglikelihoods(model, subjects, θ, σ, H)` | One call evaluates the listed subjects; column `k` of `H` is that subject's η. |

The likelihood excludes the η prior. Derivatives may be provided directly or obtained through `ad_backend(model)` with DifferentiationInterface. Reverse-mode Enzyme is the tested backend. Per-subject η Hessian blocks are optional; set `hessian_ad_backend(model) = nothing` for a gradient-only model.

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

Run `check_protocol(model, θ, σ, ω)` while implementing a model. It checks dimensions, agreement between evaluation styles, subject separability, available derivatives, and declared observation and mu-referencing traits.

The [PK comparison](pk-comparison.md) shows a complete batched BRM+RKPPL model and the calls used to fit it. The [API reference](api.md) lists every protocol entry point.
