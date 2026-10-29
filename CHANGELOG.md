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
