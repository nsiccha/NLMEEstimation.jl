# Estimators

| Method | Population parameters | Subject effects | Main output |
| --- | --- | --- | --- |
| `empirical_bayes` / `posthoc` | Fixed input θ, σ, Ω | Conditional posterior modes (EBEs) | `EBEResult.modes` |
| `focei` with `interaction = false` | Fitted by a derivative-free outer search | EBEs at the fit | `FOCEResult.theta`, `.sigma`, `.omega_matrix`, `.ofv` |
| `focei` with `interaction = true` (default) | Fitted by a derivative-free outer search | EBEs at the fit | Same fields, with the variance-Jacobian interaction term |
| `saem` with `StandardSAEMMH()` (default) | Fitted by stochastic approximation | One evolving η chain per subject | `SAEMResult.theta`, `.sigma`, `.omega_matrix`, `.acceptance` |
| `saem` with `DualAveragingMALA()` (experimental) | Fitted by stochastic approximation | Batched proposals with a separate continuously adapted step size per subject | `SAEMResult.theta`, `.sigma`, `.omega_matrix`, `.mala_stepsizes`, `.mala_acceptance` |
| `saem` with `WarmupHMCNUTS()` | Fitted by stochastic approximation | Per-subject WarmupHMC NUTS transitions | `SAEMResult.theta`, `.sigma`, `.omega_matrix`, `.nuts_divergences` |

Empirical Bayes is a conditional calculation at supplied population parameters. FOCE(I) and SAEM estimate population parameters. Their objectives and approximations differ, so compare estimates and predictions on the same data rather than treating objective values as interchangeable. Full joint MCMC can provide a posterior comparison but is not an NLMEEstimation estimator.

SAEM updates the population parameters every iteration by default. Its Gaussian random-effect mean and covariance use running sufficient statistics. For other coordinates, `SAEMOptions(numeric_mstep=:auto)` chooses a bounded-memory online optimizer update when all fitted structural `θ` coordinates have a declared `MuReferencing` update; otherwise it preserves the historical running-mixture objective. `SAEMResult.numeric_mstep` reports the selected mode. Explicit `:online` for undeclared structural `θ` is a model-specific approximation; the [BRM benchmark](saem-benchmarks.md) measures that case separately.

## Covariance step and standard errors

`covariance_step(model, fit)` estimates the sampling covariance of a fit's estimated population coordinates and returns a `CovarianceResult` with `covariance`, `standard_errors`, `names`, and the matrices it was built from. Its coordinates are the fitted ones: the model's unconstrained θ and σ, and the package's log-Cholesky ω.

Two matrices are combined. `R` is minus the Hessian of the fitted objective. `S = Σᵢ gᵢ gᵢᵀ` sums the outer products of the subjects' marginal log-likelihood gradients `gᵢ`.

| `method` | Covariance | Notes |
| --- | --- | --- |
| `:rs` (default) | `R⁻¹ S R⁻¹` | Sandwich estimator. It stays consistent when the random-effect or residual distribution is misspecified. |
| `:r` | `R⁻¹` | Inverse observed information. With a population prior, `R` includes it, and `R⁻¹` is the Laplace approximation of the population posterior covariance at the MAP point. |
| `:s` | `S⁻¹` | Inverse score cross-product. Needs at least as many informative subjects as estimated coordinates. |

Both matrices come from central finite differences of the per-subject marginal log-likelihoods, so the model supplies values only and no population derivatives. Coordinate `j` moves by `step * max(1, |x_j|)` with `step = 1e-3`. For `p` estimated coordinates, `:s` takes `2p + 1` evaluations and `:r` and `:rs` take `2p² + 1`. Every evaluation warm-starts the subject modes from the same solution at the estimate, so the terms are a smooth function of the coordinates. When `R` (or, for `:s`, `S`) is not positive definite, `CovarianceStepError` reports the smallest eigenvalue and the coordinate its eigenvector loads on most.

- **FOCE(I) fits.** `FOCEApproximation` gives the per-subject FOCE(I) terms, with the fit's interaction setting and its estimated coordinates (`fit.free`). The subject modes use Newton when the model provides η Hessian blocks, and otherwise BFGS at `gtol = 1e-10`.
- **SAEM fits.** All `(θ, σ, ω)` are estimated. With `FOCEApproximation` the result is the linearized covariance at the SAEM estimate, so it requires `GaussianObservations()` and an `NLMELayout`. Ragged layouts and non-Gaussian observations have no SAEM covariance step yet.
- **Natural units.** `delta_method(result, f)` propagates the covariance to `f(x)`, where `x = [θ; σ; ω]`. For example, `x -> exp.(x[1:nθ])` gives log-normal typical values, and `x -> omega_elements(layout, x[nθ+nσ+1:end])` gives the Ω variances and covariances, ordered like ω.

**Relation to nlmixr2.** nlmixr2's covariance methods `"r,s"`, `"r"` and `"s"` (with `covFull = TRUE`) are built from the same two matrices. nlmixr2's SAEM documentation states them on the `−2 log L` scale, where `R` is twice this package's and `S` four times, and gives the covariances as `R⁻¹ S R⁻¹`, `2 R⁻¹` and `4 S⁻¹`; on the log-likelihood scale these are the definitions above. Its SAEM accepts the same three methods at the SAEM estimate. Its SAEM default, `"sa"`, is a different estimator: Louis's observed information, accumulated by stochastic approximation in a separate simulation phase at the estimate. The suite compares the two packages on R's theophylline data (12 subjects, one-compartment oral model, additive error, diagonal Ω) at nlmixr2 7.0.1's estimate (`test/covariance.jl`):

- `ofv` minus `N_obs log(2π)` equals nlmixr2's `OBJF` to within 1.4e-4. nlmixr2 omits that constant; `ofv` keeps it.
- `:s` standard errors agree within 0.2%, and `:rs` within 2.4%.
- `:r` standard errors agree within 1.6% for the typical values and the residual SD. For the Ω variances, nlmixr2's are 5.5–14% smaller. On this package's log-likelihood scale, nlmixr2's implied `R` exceeds this package's by 5–7 (about `N/2 = 6`) on the diagonal entries of the three typical values and the three Ω coordinates, and by about 1 for the residual SD. This package's `R` matches the exact Hessian of a linear-Gaussian model to 1e-6. On theophylline, steps of `3e-3`, `1e-3` and `3e-4` give the same `R` to within 7e-5, and Newton and BFGS subject modes give the same `R` to within 3e-7. The cause of nlmixr2's difference has not been identified.

See the [PK comparison](pk-comparison.md) for one reproducible model, dataset, seeds, and all method calls.
The [SAEM kernel benchmarks](saem-benchmarks.md) compare MH, NUTS, and an experimental batched MALA trial, including their implementation and measurement scripts.
