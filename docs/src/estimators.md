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
- **SAEM fits.** All `(θ, σ, ω)` are estimated. A model with `GaussianObservations()` and an `NLMELayout` defaults to `FOCEApproximation`, the linearized covariance at the SAEM estimate. Every other SAEM fit, including ragged layouts and non-Gaussian likelihoods, defaults to `ImportanceSampling` (next section). Its draws, from the `rng` keyword, are held fixed while the coordinates move, so the estimated marginal is smooth in them. `AdaptiveGaussHermite` is the deterministic alternative when subjects have few random effects. With these two approximations, moving ω or a mu-referenced θ coordinate reuses the conditional values computed at the estimate, so only the remaining θ, σ and their pairs evaluate the model.
- **Fixed covariance families.** When the model declares `fixed_omega`, the fixed parts carry no uncertainty and are not differentiated. The estimated coordinates (`result.coordinates`) then omit a fully fixed family, and a partially fixed family contributes its estimated log standard deviations, `log sd(…)`, and, when its correlation is estimated, its correlation coordinates `m(…)` (`m_jl = L_jl / L_jj` of its Cholesky factor). `result.point` stays the full `[θ; σ; ω]`.
- **Natural units.** `delta_method(result, f)` propagates the covariance to `f(x)`, where `x = [θ; σ; ω]`. For example, `x -> exp.(x[1:nθ])` gives log-normal typical values, and `x -> omega_elements(layout, x[nθ+nσ+1:end])` gives the Ω variances and covariances, ordered like ω.

**Relation to nlmixr2.** nlmixr2's covariance methods `"r,s"`, `"r"` and `"s"` (with `covFull = TRUE`) are built from the same two matrices. nlmixr2's SAEM documentation states them on the `−2 log L` scale, where `R` is twice this package's and `S` four times, and gives the covariances as `R⁻¹ S R⁻¹`, `2 R⁻¹` and `4 S⁻¹`; on the log-likelihood scale these are the definitions above. Its SAEM accepts the same three methods at the SAEM estimate. Its SAEM default, `"sa"`, is a different estimator: Louis's observed information, accumulated by stochastic approximation in a separate simulation phase at the estimate. The suite compares the two packages on R's theophylline data (12 subjects, one-compartment oral model, additive error, diagonal Ω) at nlmixr2 7.0.1's estimate (`test/covariance.jl`):

- `ofv` minus `N_obs log(2π)` equals nlmixr2's `OBJF` to within 1.4e-4. nlmixr2 omits that constant; `ofv` keeps it.
- `:s` standard errors agree within 0.2%, and `:rs` within 2.4%.
- `:r` standard errors agree within 1.6% for the typical values and the residual SD. For the Ω variances, nlmixr2's are 5.5–14% smaller. On this package's log-likelihood scale, nlmixr2's implied `R` exceeds this package's by 5–7 (about `N/2 = 6`) on the diagonal entries of the three typical values and the three Ω coordinates, and by about 1 for the residual SD. This package's `R` matches the exact Hessian of a linear-Gaussian model to 1e-6. On theophylline, steps of `3e-3`, `1e-3` and `3e-4` give the same `R` to within 7e-5, and Newton and BFGS subject modes give the same `R` to within 3e-7. The cause of nlmixr2's difference has not been identified.

## Marginal likelihood, OFV and information criteria

FOCE(I) maximizes an approximate likelihood and reports it. SAEM maximizes the exact marginal likelihood without evaluating it. `marginal_loglikelihood(model, fit)` evaluates it afterwards, at any fit or at given `(θ, σ, ω)`:

```math
\log p(y \mid \theta, \sigma, \Omega) = \sum_i \log \int p(y_i \mid \theta, \sigma, \eta)\, N(\eta; 0, \Omega)\, d\eta .
```

| `approximation` | Method | Cost per evaluation |
| --- | --- | --- |
| `ImportanceSampling(; draws = 2000, df = 4)` (default) | Student-t proposal centred at each subject's conditional mode, scaled by the inverse curvature there. Reports a Monte Carlo standard error and per-subject effective sample sizes. | `draws` batched conditional evaluations |
| `AdaptiveGaussHermite(; nodes = 5)` | Tensor-product Gauss–Hermite rule centred and scaled the same way. `nodes = 1` is the Laplace approximation. Deterministic. | `nodes^d` batched evaluations for `d` random effects per subject |
| `FOCEApproximation()` | The FOCE(I) objective. Requires `GaussianObservations()` and an `NLMELayout`. | One subject-mode search |

The first two need only conditional log-likelihood values, plus η gradients for the subject modes, so they apply to any likelihood and to ragged layouts. The curvature comes from the model's η Hessian blocks when available, and otherwise from central differences of batched η gradients.

`MarginalLikelihoodResult` reports:
- `loglikelihood` and `subject_loglikelihoods`;
- `ofv`, minus twice the fitted objective: the log-likelihood, plus a model `population_logprior` when one exists (`objective` `:MAP`);
- `aic = -2 log L + 2k` and `bic = -2 log L + k log(N_obs)`. Here `k` counts the estimated coordinates: θ, σ and the Ω coordinates that `fixed_omega` leaves free, or `fit.free` for FOCE(I). `N_obs` counts observations for a model with an observation model; otherwise pass `nobs`.

Every constant is kept. nlmixr2's `OBJF` omits the Gaussian constant, so it equals `ofv - N_obs log(2π)`.

**Relation to saemix.** saemix computes the same two estimators after a fit (`ll.is` and `ll.gq`); its importance-sampling proposal is likewise a Student t with 4 degrees of freedom by default. At saemix 3.5's estimate on R's theophylline data (12 subjects, 132 observations; `test/marginal.jl`):

- Adaptive Gauss–Hermite here converges to −179.965 (9 and 12 nodes agree to 5e-4), and importance sampling with 20,000 draws gives −179.973 ± 0.010.
- saemix reports `ll.gq` = −179.972 (12 nodes over ±4 SD) and `ll.is` = −179.954 (50,000 draws).
- At the same point, the Laplace approximation (`nodes = 1`) gives −180.168 and FOCEI −179.326.
- saemix counts the same 8 parameters for its AIC, but its BIC penalizes log(number of subjects); `bic` here penalizes log(N_obs).

See the [PK comparison](pk-comparison.md) for one reproducible model, dataset, seeds, and all method calls.
The [SAEM kernel benchmarks](saem-benchmarks.md) compare MH, NUTS, and an experimental batched MALA trial, including their implementation and measurement scripts.
