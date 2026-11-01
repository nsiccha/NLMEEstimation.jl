# Model protocol

A model is any Julia value that supplies a conditional observation likelihood. NLMEEstimation supplies the subject-level random-effect prior, `ηᵢ ~ N(0, Ω)`, and represents Ω by blockwise log-Cholesky coordinates. The model supplies θ and σ in its own unconstrained coordinates.

Every model implements `nsubjects(model)` and `nlme_layout(model)`. It can use either evaluation style:

| Style | Required conditional likelihood | How estimators evaluate it |
| --- | --- | --- |
| `PerSubjectEvaluation()` | `conditional_loglikelihood(model, i, θ, σ, η)` | The package gathers subject calls into a batch. |
| `BatchedEvaluation()` | `conditional_loglikelihoods(model, subjects, θ, σ, H)` | One call evaluates the listed subjects; column `k` of `H` is that subject's η. |

The likelihood excludes the η prior. Derivatives may be provided directly or obtained through `ad_backend(model)` with DifferentiationInterface. Reverse-mode Enzyme is the tested backend. Per-subject η Hessian blocks are optional; set `hessian_ad_backend(model) = nothing` for a gradient-only model. A model whose subjects carry many nested random effects can instead declare `eta_hessian_sparsity(model, i)`, which entries of subject `i`'s η Hessian can be nonzero; the estimators then compute sparse Hessians from its η gradients ([PPL and AD requirements](requirements.md), "Sparse η Hessians"). [PPL and AD requirements](requirements.md) lists which derivatives each estimator uses; every method runs on first-order AD.

A model rejects a point outside its domain by returning `-Inf`, or by throwing a `DomainError`, `OverflowError` or `PosDefException` as a Stan-semantics model does (a matrix exponential of an overflowed rate, the Cholesky factor of an indefinite matrix). At a trial point the estimators treat both alike. Where each subject has its own trial point (EBE steps, MCMC proposals, quadrature points) and a batched evaluation throws, the subjects are evaluated one at a time to find which ones.

- `empirical_bayes` (Newton and BFGS, and so the EBEs inside `focei`) rejects that subject's trial step and tries a shorter one; Newton's final polishing step is dropped.
- `focei`'s search scores the evaluation as a very large objective.
- SAEM's MH and MALA kernels reject that subject's proposal. The NUTS kernel scores that position `-Inf`, exactly as if the model had returned it. SAEM's numerical and MAP M-steps halve their line-search step.
- `marginal_loglikelihood` gives that importance-sampling or quadrature point zero density.

The point a search starts from must evaluate finitely.

A population update keeps SAEM's chains inside the domain: it keeps each subject's φ = Dθ + η where it was, and its numerical M-step rejects trial parameters at which a current chain would leave the domain. If a chain's density is still non-finite after an update, the MH and NUTS kernels raise a `DomainError` naming those subjects, and the MALA kernel raises its own `DomainError`.

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

A model that computes its own Laplace approximation, such as a joint program with an embedded Laplace, can declare `provides_laplace_gradient(model) = true` and implement `laplace_marginals_and_gradient(model, subjects, θ, σ, ω, H)`. It returns each listed subject's Laplace marginal log-likelihood and the gradient of their sum in θ, σ and ω, including the mode's dependence on them; `H` holds starting points for its mode search, in the batch shape of `conditional_loglikelihoods`. `covariance_step` with `AdaptiveGaussHermite(nodes = 1)` then takes `R` from central differences of that gradient, as TMB's `sdreport` does: `2p + 1` calls for `p` estimated coordinates, where differences of marginal values take `2p² + 1`. The package adds the gradient of a `population_logprior` (from `population_logprior_and_gradient`) and maps the ω gradient onto the coordinates `fixed_omega` leaves free.

Such a model can also integrate some covariance families out inside SAEM, so that SAEM samples only the others. `integrated_families(model)` lists them (by default none; `saem(...; integrated = …)` chooses another set for one fit), and the model implements three methods, each taking that set as its second argument and a batch `H` whose integrated entries start the mode search while the other, sampled entries are given:

- `laplace_conditionals_and_gradients(model, integrated, subjects, θ, σ, ω, H)` returns each subject's Laplace approximation of its conditional log-likelihood given its sampled effects, with the integrated ones integrated out against their prior at ω, and its gradient in the sampled entries (zero in the integrated ones). The chains sample against it. `laplace_conditionals` returns the values alone and defaults to this method.
- `laplace_conditional_moments(model, integrated, subjects, θ, σ, ω, H)` returns the same values, `H` with its integrated entries at the mode, and per subject the covariance block of each integrated instance, the diagonal blocks of the inverse negative Hessian. SAEM's M-step takes each integrated family's covariance from them.
- `laplace_conditionals_and_population_gradient(model, integrated, subjects, θ, σ, ω, H)` returns the values and their summed gradient in θ, σ and ω at fixed sampled effects, for the numerical M-step. With every family integrated it is `laplace_marginals_and_gradient`, and either method serves for the other.

The model keeps its own sparse factorization: only the modes and the instances' covariance blocks cross the protocol, never a dense matrix. How SAEM uses them is in [estimators](estimators.md).

Run `check_protocol(model, θ, σ, ω)` while implementing a model. It checks dimensions, agreement between evaluation styles, subject separability, available derivatives, declared observation and mu-referencing traits, that `ω` holds any declared `fixed_omega` values, that a declared Laplace gradient's values equal the package's own Laplace approximation and its gradient a central difference of them, and the same for the methods of `integrated_families` (on small subjects, against a dense Laplace of the integrated effects given the others).

The [PK comparison](pk-comparison.md) shows a complete batched BRM+RKPPL model and the calls used to fit it. The [API reference](api.md) lists every protocol entry point.
