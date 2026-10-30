# Changelog

All notable changes to NLMEEstimation.jl are documented here, newest first.

Every landing that adds or changes public API (an export, a keyword, a method,
or documented behaviour a dependant can rely on) bumps the patch digit in
`Project.toml` and adds its entry under a new heading at the top of this file.
An entry that breaks something a dependant relies on says **Breaking:** and
what to change. The minor digit, the breaking one in
[semantic versioning](https://semver.org/) before 1.0, has not moved yet.
A package that calls that API states the version as its floor, for example
`[compat] NLMEEstimation = "0.1.1"`. An environment that pins an older
NLMEEstimation commit is then refused when it resolves, with
`Unsatisfiable requirements detected for package NLMEEstimation`, instead of
failing or silently behaving differently at run time.

## 0.2.1

- `WarmupHMCNUTS` evaluates a `BatchedEvaluation` model's chains in lockstep.
  Each subject's chain still takes its own transitions (and its own Pathfinder
  start), but every leapfrog step of the subjects whose trajectories are still
  growing reaches the model as one `conditional_loglikelihoods_and_gradients`
  call over those subjects, and Pathfinder's density-only evaluations as one
  `conditional_loglikelihoods` call, instead of one call per subject. A round of
  transitions therefore makes as many batched calls as its longest trajectory
  needs, not the sum over subjects. Before, a batched model received one-subject
  batches, one per subject and leapfrog step. On twelve subjects of
  3,294–35,783 nested units behind a model whose every call evaluates the whole
  model (as BRM evaluates a nested model without subject cells), one round costs
  16 whole-model sweeps instead of 119, an SA iteration 26.7 instead of 189.7,
  and the Pathfinder start 173 gradient and 204 density sweeps instead of 711
  and 1,132 (`bench/saem_nuts_lockstep.jl`). The fits were identical. Each
  chain receives the values its own model calls give, so seeded fits are
  unchanged wherever a model's batched values and gradients equal its
  single-subject ones bit for bit. A batch that throws a domain error is
  re-evaluated one subject at a time. The chains are tasks on the calling
  thread: the model is only ever called from it. `LouisInformation`'s draws
  take the same route. A `PerSubjectEvaluation` model's chains run one after
  another as before. Under `progress`, a batched model's Pathfinder start
  checks for an interrupt between batched evaluations rather than between
  subjects. Each batched call costs what the model's batch of that size costs, so
  a model whose small batches cost much more than its one-subject calls pays
  that for every step with two or more subjects left, most under a small
  `minibatch_size`.

## 0.2.0

- **Breaking:** `WarmupHMCNUTS` is a continuously adapting kernel. Each
  subject's chain starts with Pathfinder from its starting η (no warmup phase,
  no η Hessian), and every transition adapts the diagonal metric and the step
  size from its leapfrog leaves for the whole fit, as θ, σ and Ω move
  (WarmupHMC's `ContinuousNUTSState`, `continuous_nuts_step` and `reposition!`;
  the floor is 0.2.8, whose Pathfinder start works from a fit's modes, stays
  inside a bounded support, and allocates no d × d buffer). The
  keywords `warmup_draws`, `warmup_evaluations`, `stepsize_adaptation_limit`,
  `curvature_floor` and `metric` are gone: remove them from
  `WarmupHMCNUTS(…)` calls. `max_tree_depth` defaults to 10 (was 8).
  `LouisInformation` adapts through its `burnin` and freezes the kernel for its
  `draws`. The `saem` progress phase `NUTS warmup` is now `Pathfinder start`.
  Seeded NUTS fits change. SAEM with NUTS no longer asks the model for an η
  Hessian or its sparsity pattern.
  - On five exact nested subjects of 250–4,000 units, fits end 0.002–0.031
    log-likelihood units below the maximum whether the start's scales are
    right, 4× too wide or 4× too narrow, where 0.1.7 ended 0.79–3.65 below
    from the wide start and cost about 5× the gradient evaluations from the
    narrow one (`bench/saem_nuts_continuous.jl`).
  - On twelve subjects of 3,294–35,783 units (157,808 effects): 563 s and
    1.29 GB peak resident memory, against 704 s and 1.79 GB on 0.1.7
    (`bench/saem_nuts_ragged_scale.jl`).

## 0.1.7

- The star coloring of a declared `eta_hessian_sparsity` pattern is computed
  once per pattern and cached.
  - SparseMatrixColorings' greedy star coloring takes time quadratic in a
    nested subject's effects (1.2 s at 35,783 units), and was repeated at every
    sparse Hessian: each Newton iteration, each Laplace marginal, and each
    point of the covariance step.
  - The colors, and so the Hessians, are unchanged.
  - On `bench/ebe_ragged_scale.jl`, the Laplace covariance step of six subjects
    of 250–8,000 units went from 10.1 s to 2.1 s, and Newton on one subject of
    25,872 effects from 1.8 s to 0.7 s.

## 0.1.6

- **Changed default:** `marginal_loglikelihood` on a `RaggedNLMELayout` whose
  model reports η Hessians (its own blocks, or a declared
  `eta_hessian_sparsity`) uses the Laplace approximation,
  `AdaptiveGaussHermite(nodes = 1)`, instead of `ImportanceSampling()`. This
  matches the 0.1.5 covariance-step default.
  - Why: importance sampling's effective draws collapse as a subject grows
    (4–6 of 2,000 at 1,260 Poisson units).
  - Laplace's bias on non-Gaussian posteriors grows with the number of units
    (−2.19 there); the docs give the measured values.
  - To keep the old behaviour, pass `approximation = ImportanceSampling()`.

## 0.1.5

- **Changed default:** `covariance_step(model, fit)` for an SAEM fit on a
  `RaggedNLMELayout` no longer defaults to `ImportanceSampling()`.
  - It uses the Laplace approximation, `AdaptiveGaussHermite(nodes = 1)`, when
    the model reports η Hessians: its own blocks, or a declared
    `eta_hessian_sparsity`. This is the rule `default_ebe_method` uses for Newton.
  - It uses `LouisInformation()` otherwise.
  - Why: importance sampling loses effective draws as a subject grows. On a
    Poisson nested model with an exact `R`, Laplace's standard errors stayed
    within 4%, while importance sampling's `R` was not positive definite on one
    of two seeds and Louis's identity was furthest off on the unit variance.
  - To keep the old behaviour, pass `approximation = ImportanceSampling()`.
- `LouisInformation` no longer calls
  `conditional_loglikelihood_and_population_gradient` when every θ is
  mu-referenced and there is no σ, so it works for any model SAEM fits.

## 0.1.4

- `covariance_step(…; approximation = LouisInformation(; draws, burnin, kernel))`
  builds `R` and `S` by Louis's missing-information identity from NUTS draws
  of every subject's random effects at the point: SAEM's stochastic-approximation
  Fisher information (Monolix's default, nlmixr2's SAEM `"sa"`). It needs no
  marginal likelihood and no η curvature, works for any likelihood and layout,
  and keeps no `n × n` matrix per subject. `R` and `S` carry Monte Carlo error,
  largest for weakly informed variance components. `marginal_loglikelihood`
  refuses it.
- SAEM's NUTS warmup metric (and MALA's `:hessian` metric) uses the sparse η
  Hessian when the model declares `eta_hessian_sparsity`, so the default
  diagonal metric of a subject with thousands of nested effects no longer
  forms a dense `n × n` matrix. A gradient-only model that declares a pattern
  now gets a curvature-based metric instead of the prior's.

## 0.1.3

- New optional protocol method `eta_hessian_sparsity(model, i)`: a model
  declares which entries of subject `i`'s conditional η Hessian can be
  nonzero, for every subject or for none. The estimators then compute that
  Hessian as a sparse matrix from the model's batched η gradients, one pair of
  evaluations per color of a star coloring of the pattern
  (SparseMatrixColorings.jl), and factor it sparsely (CHOLMOD). A subject with
  thousands of nested random effects costs time and memory linear in its
  size: on the test suite's nested model, one subject of 25,872 random effects
  takes 1.9 s and 265 MB for its mode, and its Laplace marginal equals the
  closed form. New dependencies: SparseArrays (standard library) and
  SparseMatrixColorings.
- With a declared pattern, `default_ebe_method` is `NewtonEBE` even without
  Hessian blocks, and the result's `hessians` are `SparseMatrixCSC`s
  (`EBEResult` and `RaggedEBEResult` accept them). Newton's trust-region step
  on a sparse Hessian is found by sparse Cholesky factorizations of
  `-H + νΩ⁻¹` (Moré & Sorensen) instead of an eigendecomposition.
  `QuasiNewtonEBE`'s curvature check, `ebe_covariances`, and the curvature,
  log-determinant and proposal points of `marginal_loglikelihood` use the
  sparse Hessians too.
- `check_protocol` checks a declared pattern: its size, that every checked
  subject has one, and that the Hessian recovered from it agrees with a
  Hessian-vector product along a random direction.
- `covariance_step` with `AdaptiveGaussHermite` adapts its quadrature at every
  evaluated point: it finds the subject modes and curvature there again
  (warm-started, `gtol = 1e-10`, as `FOCEApproximation` does). It used to keep
  the points at the estimate's modes, and with one node (Laplace) `R` was then
  the complete-data information at those modes, with its cross terms missing
  (on the test model 6.60 against the exact 6.52, and 0 against 25.4).
  `ImportanceSampling` keeps its fixed draws. `marginal_loglikelihood` values
  are unchanged.
- `AdaptiveGaussHermite` counts its points without integer overflow
  (`nodes^d` used to wrap to 0 for large `d`) and builds a single Laplace point
  for any number of random effects.

## 0.1.2

- `QuasiNewtonEBE` is limited-memory BFGS. `QuasiNewtonEBE(; memory = 20)`
  keeps the last `memory` curvature pairs and applies the prior covariance one
  family block at a time on a `RaggedNLMELayout`, so a subject with `n` random
  effects stores `O(memory · n)` numbers. The dense BFGS it replaces kept an
  `n × n` inverse Hessian per subject and updated it at `O(n³)` cost per
  iteration: 2.9 s and 3.9 GB of allocation for one subject of 1,001 effects.
  The initial matrix is now rescaled at every update, not only the first, and
  nested subjects of 126 to 4,001 effects took fewer iterations than with the
  dense update. Modes agree with the earlier ones to the convergence tolerance
  but are not identical, and iteration counts change.

## 0.1.1

- `saem` runs the `DualAveragingMALA` and `WarmupHMCNUTS` kernels on a
  `RaggedNLMELayout`, not only `StandardSAEMMH`. Each subject adapts its own
  MALA step size and runs its own NUTS warmup over its own coordinates. The
  prior's precision and Cholesky factor are applied one family block at a
  time, so a subject with thousands of nested effects never forms a dense
  covariance.
- `RaggedSAEMResult` has `SAEMResult`'s kernel fields: `nuts_divergences`,
  `mala_stepsizes`, `mala_acceptance` and `mala_adaptation_steps`. For the
  gradient kernels `acceptance` and `rw_scale` are `missing`. Its type
  parameters are now `{T, A, R, P}`.
- Ragged SAEM keeps the running second moments of each random-effect block
  only, which is all its M-step reads, instead of a dense `n_k × n_k` matrix
  per subject. That memory grew with the square of a subject's number of nested
  effects (12.5 GB for twelve subjects of 2,254–25,871 effects each) and now
  grows linearly. Fits are unchanged bit for bit.
