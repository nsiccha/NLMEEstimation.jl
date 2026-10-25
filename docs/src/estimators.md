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

SAEM updates the population parameters every iteration by default. Its Gaussian random-effect mean and covariance use running sufficient statistics; with a population prior they move on from that closed-form update to the maximizer with the prior, without evaluating the model. For other coordinates, `SAEMOptions(numeric_mstep=:auto)` chooses a bounded-memory online optimizer update when all fitted structural `θ` coordinates have a declared `MuReferencing` update; otherwise it preserves the historical running-mixture objective. `SAEMResult.numeric_mstep` reports the selected mode. The numerical update holds each chain's η fixed rather than its φ, so it moves the mu-referenced `θ` jointly with the remaining `θ` and `σ`. A population scalar that trades off against a random effect, such as a rate against every subject's amplitude, can therefore move along that ridge in one update. Holding φ fixed, it could move only as fast as the chains followed. Its quasi-Newton search starts from a difference estimate of the objective's curvature and carries that metric from one iteration to the next, because an intercept pinned by thousands of observations and a weakly identified rate differ in curvature by many orders of magnitude. After the closed-form part of the update, every chain's η is shifted so that the subject's individual parameters φ = Dθ + η stay where they were (saemix likewise carries φ). Each subject's likelihood is unchanged, and its kernel continues where the chain had settled.

Each `StandardSAEMMH` round makes one prior-independence proposal, one joint random-walk proposal scaled by Ω, and one random-walk proposal per η coordinate, with each coordinate's scale adapted separately (saemix's componentwise kernel). A single joint scale has to follow a subject's narrowest posterior direction, so a coordinate that the data pin down less firmly hardly moves, and the estimated Ω shrinks. On a 38-subject washout model with three rate scalars in `σ` (snag `saem-population-8975f340`), these changes moved SAEM's estimate from 54 log units below a reference HMC fit's posterior means, in marginal log posterior, onto them. Seeded SAEM fits therefore differ from those before this change. Explicit `:online` for undeclared structural `θ` is a model-specific approximation; the [BRM benchmark](saem-benchmarks.md) measures that case separately.

## Minibatch SAEM

`SAEMOptions(minibatch_size = b)` runs minibatch SAEM (Kuhn, Matias & Rebafka, 2020) for data where one kernel sweep over every subject per iteration costs too much. Each SA iteration draws `b` of the fitted subjects uniformly at random without replacement. Only their chains move, with any kernel the layout supports, and only they are evaluated:

- **Statistics.** The stochastic-approximation statistics keep every subject's last sampled latent effects. The closed-form update of the mu-referenced θ and Ω (and its MAP form) therefore still uses all subjects, without evaluating the model.
- **Numerical update.** The update of σ and the remaining θ evaluates the batch alone, its conditional log density scaled by `nsubjects / b` so that it estimates the full objective. While γ = 1 it searches for that objective's optimum, as the online update does. Once γ < 1 it takes one Robbins–Monro step `γ M ĝ` along the batch's scaled score `ĝ`, with the quasi-Newton metric `M` of the searches held fixed. The step's mean is the full score, so the estimates converge to the same fixed point as full-batch SAEM. A search to each batch's optimum would instead average optimizers whose mean is off by order `1/b`. This holds for every θ and σ coordinate, so minibatch fits need no `:mixture` mode: `numeric_mstep` must be `:auto` or `:online`, and the result reports `:online`.
- **MCMC burn-in.** The `mcmc_burnin` sweeps at the starting estimate still move every chain, so that every chain starts the stochastic approximation from its conditional posterior. A minibatch burn-in left most chains near their start, and the first γ = 1 updates then shrank Ω, and with it every proposal scaled by Ω. Prior-metric MALA never recovered: on a 40-subject linear-growth model with a quarter of the subjects per iteration, all ten seeded fits stopped 11 to 13 log-likelihood units below the maximum.
- **Domain.** The numerical update screens only the batch's chains. A chain drawn later can therefore lie outside the model's domain at the parameters of its next iteration. It is then replaced by a feasible draw from the random-effect prior, an independence Metropolis–Hastings move that a zero-density state accepts.

Each chain moves in a fraction `b / nsubjects` of the iterations. On 40-subject linear-Gaussian models, whose marginal likelihood is exact, scaling `iterations`, `burnin` and `adapt_iterations` by `nsubjects / b` gave minibatch fits as close to the maximum likelihood as full-batch fits, for every kernel, under a population prior, with a fixed correlation and on a ragged layout. Each chain then makes as many transitions as in a full-batch fit. Without that scaling, MH and NUTS fits with a quarter of the subjects ended three to five times further from the maximum, in mean log-likelihood gap over eight seeds, and MALA fits did not converge within 300 iterations. `minibatch_size` equal to the number of fitted subjects is the full-batch algorithm.

Each iteration still does arithmetic for every subject, on its statistics and in the closed-form update, but evaluates the model for the batch alone.

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

## Residual diagnostics

`residual_diagnostics(model, fit; replicates = 1000, rng)` returns per-observation diagnostics for a model with `GaussianObservations()`, on equal-size or ragged layouts. It works at a FOCE(I) or SAEM estimate, or at given `(θ, σ, ω)`. Every field of `ResidualDiagnostics` holds one vector per subject:

| Field | Definition |
| --- | --- |
| `pred`, `ipred` | Observation means at `η = 0` and at the subject's EBE `η̂` |
| `iwres` | `(y - ipred) / sqrt(v(η̂))` |
| `cpred`, `cwres` | Hooker, Staatz and Karlsson (2007): linearize the means around `η̂`, `y ≈ μ(η̂) + J (η - η̂)`, giving mean `cpred = μ(η̂) - J η̂` and covariance `C = J Ω Jᵀ + diag(v(η̂))`; `cwres = U (y - cpred)` with `U` the upper Cholesky factor of `C⁻¹` (`UᵀU = C⁻¹`), which decorrelates within the subject |
| `cwres_per_observation` | `(y - cpred) ./ sqrt.(diag(C))`, each observation standardized without decorrelation |
| `epred`, `npd`, `pde`, `npde` | Brendel et al. (2006) and Comets, Brendel and Mentré (2008): simulate `replicates` datasets at the population parameters, decorrelate observed and simulated vectors with the Cholesky factor of the inverse of each subject's simulated covariance (as for `cwres`), and take the normal quantile of the fraction of simulated values below the observed one (`npd` skips the decorrelation) |

CWRES uses the residual variance at the EBE, as this package's FOCE(I) objective does. Each replicate costs one batched `observation_moments` call. `replicates` must exceed every subject's number of observations, and `replicates = 0` skips the simulation. Both residuals are approximately independent standard normal values under a correct model. For a linear Gaussian model the linearization is exact, and `npde` converges to `Φ⁻¹(Φ(cwres)) = cwres` as `replicates` grows; the tests use this as an oracle. Non-Gaussian likelihoods have no residual diagnostics: the protocol gives the package no way to simulate their observations.

**Relation to nlmixr2.** At nlmixr2 7.0.1's FOCEi estimate on R's theophylline data (`test/diagnostics.jl`), the two packages agree as follows:

- `pred` matches nlmixr2's PRED to 1e-13. `ipred`, `iwres` and `cpred` match IPRED, IWRES and CPRED to 3e-4, nlmixr2's subject-mode precision.
- nlmixr2's `CWRES` is the per-observation standardization: it equals `cwres_per_observation` to 2e-4. The decorrelated `cwres` of Hooker et al. is a different vector.
- nlmixr2's NPDE uses the same decorrelation as `npde`. Its 300 simulations against 3000 here correlate at 0.83, and NPD (no decorrelation) against 2000 here at 0.995. Decorrelating with the lower Cholesky factor of the covariance instead, an equally valid whitening, correlates at only 0.23. The decorrelation order therefore matters when comparing NPDE across software.

## Progress and early stopping

Every function above takes `progress = nothing`. Pass a [Treebars](https://github.com/nsiccha/Treebars.jl) node and the function builds its own labelled node under it, which a web page or terminal renders while the fit runs. A backend symbol such as `:term` starts a new root instead (`:term` draws terminal bars once `Term` is loaded). With `nothing`, no node is created and the results are identical.

```julia
using Treebars
root = Treebars.initialize_progress!(:state; description = "saem mh")
fit = saem(model, θ0, σ0, ω0; rng, options, progress = root)
Treebars.finalize_progress!(root)
println(Treebars.render_text(root))
```

For a 16-subject simulated one-compartment model and 12 SA iterations (a first call, so the times include compilation), this printed:

```
✓ saem mh [26.2s]
└─ ✓ SAEM [20.7s]
   ├─ ✓ MCMC burn-in (3/3) — acceptance 9.4% prior, 35% random walk [1ms]
   └─ ✓ Stochastic approximation (12/12) — γ = 0.211 · Σ log p(y|η) = -154.264 · acceptance 4.2% prior, 22% random walk [20.7s]
```

| Function | Its node |
| --- | --- |
| `saem` | `SAEM`, with the phases `NUTS warmup` (one transient WarmupHMC tree per subject), `MCMC burn-in` and `Stochastic approximation`, each counting iterations. The message shows the stage (`M-step` while it runs), the SA step size γ, the chains' summed conditional log-density at the current population estimate (MH and NUTS), and the kernel's acceptance, divergences or mean step size. |
| `empirical_bayes` / `posthoc` | `Empirical Bayes (Newton)` or `(BFGS)`, with the iteration and the number of converged subjects. |
| `focei` | `FOCE(I)`, counting objective calls against `maxeval`, with the best OFV so far. Each call's mode search is a transient child. |
| `focei_evaluate` | `FOCE(I) objective`, with its mode search. |
| `covariance_step` | `Covariance step`: the subject modes, then `Marginal likelihoods` counting the `2p² + 1` (or `2p + 1`) evaluations. |
| `marginal_loglikelihood` | `Marginal likelihood`: the subject modes, then a transient counter over the sampling or quadrature points. |
| `residual_diagnostics` | `Residual diagnostics`: the subject modes, then `Simulated replicates`. |

`Treebars.request_interrupt!(node)` asks a run to stop, for example from a Stop button. `saem` stops at the next iteration boundary and returns the estimate of the last completed iteration, with `interrupted = true` and `iterations` counting the completed SA iterations. This is the same estimate a run with that many iterations would return. `focei` stops at the next objective call and returns its best point so far, with `converged = false` and `optimizer_info.status == :FORCED_STOP`. The other functions have no partial result, so they raise `Treebars.ProgressInterrupted`, as `focei` does when the interrupt comes before its search starts.

See the [PK comparison](pk-comparison.md) for one reproducible model, dataset, seeds, and all method calls.
The [SAEM kernel benchmarks](saem-benchmarks.md) compare MH, NUTS, and an experimental batched MALA trial, including their implementation and measurement scripts.
