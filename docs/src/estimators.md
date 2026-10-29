# Estimators

| Method | Population parameters | Subject effects | Main output |
| --- | --- | --- | --- |
| `empirical_bayes` / `posthoc` | Fixed input θ, σ, Ω | Conditional posterior modes (EBEs) | `EBEResult.modes` |
| `focei` with `interaction = false` | Fitted by a derivative-free outer search | EBEs at the fit | `FOCEResult.theta`, `.sigma`, `.omega_matrix`, `.ofv` |
| `focei` with `interaction = true` (default) | Fitted by a derivative-free outer search | EBEs at the fit | Same fields, with the variance-Jacobian interaction term |
| `saem` with `StandardSAEMMH()` (default) | Fitted by stochastic approximation | One evolving η chain per subject | `SAEMResult.theta`, `.sigma`, `.omega_matrix`, `.acceptance` |
| `saem` with `DualAveragingMALA()` (experimental) | Fitted by stochastic approximation | Batched proposals with a separate continuously adapted step size per subject | `SAEMResult.theta`, `.sigma`, `.omega_matrix`, `.mala_stepsizes`, `.mala_acceptance` |
| `saem` with `WarmupHMCNUTS()` | Fitted by stochastic approximation | Per-subject WarmupHMC NUTS transitions | `SAEMResult.theta`, `.sigma`, `.omega_matrix`, `.nuts_divergences` |

On a ragged layout (`RaggedNLMELayout`) every SAEM kernel returns a `RaggedSAEMResult` with the same fields, `eta` and `omega_matrix` holding one vector per subject and one matrix per random-effect family.

Empirical Bayes is a conditional calculation at supplied population parameters. FOCE(I) and SAEM estimate population parameters. Their objectives and approximations differ, so compare estimates and predictions on the same data rather than treating objective values as interchangeable. Full joint MCMC can provide a posterior comparison but is not an NLMEEstimation estimator.

SAEM updates the population parameters every iteration by default. Its Gaussian random-effect mean and covariance use running sufficient statistics; with a population prior they move on from that closed-form update to the maximizer with the prior, without evaluating the model. The subjects' mu-referencing designs need not identify every mu-referenced `θ`: a combination that no subject's design carries (more coefficients on a random effect than subjects, or collinear covariates) does not change the likelihood. The closed-form update keeps it, and a population prior then sets it; without a prior, `saem` throws an `ArgumentError` naming those coefficients before it samples, because the maximum-likelihood estimate is not unique. For other coordinates, `SAEMOptions(numeric_mstep=:auto)` chooses a bounded-memory online optimizer update when all fitted structural `θ` coordinates have a declared `MuReferencing` update; otherwise it preserves the historical running-mixture objective. `SAEMResult.numeric_mstep` reports the selected mode. The numerical update holds each chain's η fixed rather than its φ, so it moves the mu-referenced `θ` jointly with the remaining `θ` and `σ`. A population scalar that trades off against a random effect, such as a rate against every subject's amplitude, can therefore move along that ridge in one update. Holding φ fixed, it could move only as fast as the chains followed. Its quasi-Newton search starts from a difference estimate of the objective's curvature and carries that metric from one iteration to the next, because an intercept pinned by thousands of observations and a weakly identified rate differ in curvature by many orders of magnitude. After the closed-form part of the update, every chain's η is shifted so that the subject's individual parameters φ = Dθ + η stay where they were (saemix likewise carries φ). Each subject's likelihood is unchanged, and its kernel continues where the chain had settled.

While γ = 1, SAEM anneals Ω (`SAEMOptions(omega_annealing = 0.95)`), as Monolix and saemix do. An M-step may shrink no covariance family by more than that factor in any direction: its new Ω is at least 0.95 times the previous one in the matrix order, and fixed parts are re-applied. Each of these M-steps sets Ω from one posterior draw per subject, so Ω wanders; where it wanders towards a singular correlation, every kernel whose proposals follow Ω stops moving across it, and the fit stays there. On the `fixed Ω` test fixture (one weakly informed variance and a covariance block with a fixed standard deviation), 3 of 20 seeded 300-iteration fits ended more than one log-likelihood unit below the maximum (the worst 3.5), with the block's correlation pushed towards one. Five times longer runs did not cure it: some still ended at a correlation of 0.985–0.998 against the maximum likelihood's 0.67. With annealing no 300-iteration fit ended more than 0.46 units below, and the largest Ω error halved. On a linear growth model, MALA with the prior metric ended more than 2 units below the maximum in 3 of 20 fits before and none after; MH and NUTS were unaffected; on Theoph the variances moved towards saemix's. `omega_annealing = 0` turns annealing off and reproduces earlier fits.

Each `StandardSAEMMH` round makes one prior-independence proposal, one joint random-walk proposal scaled by Ω, and one random-walk proposal per η coordinate, with each coordinate's scale adapted separately (saemix's componentwise kernel). A single joint scale has to follow a subject's narrowest posterior direction, so a coordinate that the data pin down less firmly hardly moves, and the estimated Ω shrinks. On a 38-subject washout model with three rate scalars in `σ` (snag `saem-population-8975f340`), these changes moved SAEM's estimate from 54 log units below a reference HMC fit's posterior means, in marginal log posterior, onto them. Seeded SAEM fits therefore differ from those before this change. Explicit `:online` for undeclared structural `θ` is a model-specific approximation; the [BRM benchmark](saem-benchmarks.md) measures that case separately.

On a ragged layout every kernel moves each subject's chain on that subject's own coordinates, with its own MALA step size or NUTS warmup. A subject with hundreds or thousands of nested random effects, such as a region with its districts and municipalities, needs `WarmupHMCNUTS`. MH's random walks move that many coordinates too little; the M-steps at γ = 1 then shrink the nested family's covariance from chains still near their start, and the fit settles with that variance collapsed. Prior-metric MALA, whose proposals follow Ω, collapses the same way. On six subjects of 250 to 8,000 nested units (15,750 unit effects) with an exact marginal likelihood, 300 iterations (150 at γ = 1) of MH ended 10,700–12,700 log-likelihood units below the maximum with the unit SD at 0.06–0.22, and MALA 12,900–13,400 below with it at 0.016. NUTS ended within 0.01 units of the maximum, at its unit SD of 0.40, with every subject's unit chains correlating 0.91–0.95 with their exact posterior means, as posterior draws do (three seeds; [benchmark](saem-benchmarks.md)). It took about twice as long. The kernels and SAEM's running statistics work one random-effect block at a time, so their memory grows with the number of nested effects, not its square; NUTS's `metric = :dense`, and MALA's `:hessian` metric when the model provides η Hessians, still form one dense matrix per subject.

## Minibatch SAEM

`SAEMOptions(minibatch_size = b)` runs minibatch SAEM (Kuhn, Matias & Rebafka, 2020) for data where one kernel sweep over every subject per iteration costs too much. Each SA iteration draws `b` of the `n` fitted subjects uniformly at random without replacement. Only their chains move, with any kernel the layout supports, and the model is evaluated for them alone:

- **Statistics.** The stochastic-approximation statistics keep every subject's last sampled latent effects, so the closed-form update of the mu-referenced θ and Ω (and its MAP form) still uses all subjects without evaluating the model.
- **Numerical update.** σ and the remaining θ take one Robbins–Monro step per M-step, `min(γ, 4b/n) M ĝ`, along the batch's score `ĝ` with its conditional log density scaled by `n / b`. The step's mean is the full score, so the estimates converge to the same fixed point as full-batch SAEM, for every θ and σ coordinate; minibatch fits therefore need no `:mixture` mode. `numeric_mstep` must be `:auto`, and the result reports `:minibatch`.
  - The cap `4b/n` lets one pass over the subjects shrink the distance to the optimum by about e⁻⁴, as a full-batch search nearly solves each M-step. Without it, a step at γ = 1 moves by the batch's whole sampling error: searching each batch's objective to its optimum put a weakly identified additive error SD (10 of 400 PK subjects per iteration) where its score vanishes, and 3 of 9 fits ended 300 to 1,650 standard errors off. A cap of `b/n` left σ lagging the chains, and in 1 of 8 NUTS fits of a 40-subject growth model Ω then collapsed onto a singular, absorbing correlation.
  - `M` is the inverse curvature of the full objective, estimated from every subject's chain at M-steps 1, 2, 4, 8, …, a logarithmic number of passes over all subjects. It is held fixed between estimates.
- **MCMC burn-in.** The `mcmc_burnin` sweeps still move every chain, and so does the first iteration when `mcmc_burnin = 0`. Without that, the first γ = 1 updates estimated Ω from chains still at their start and shrank it, and with it every proposal scaled by Ω.
- **Domain.** The numerical update screens only the batch's chains. A chain drawn later can therefore lie outside the model's domain at the parameters of its next iteration. It is then replaced by a feasible draw from the random-effect prior, an independence Metropolis–Hastings move that a zero-density state accepts.

Each chain moves in a fraction `b / n` of the iterations. Scaling `iterations`, `burnin` and `adapt_iterations` by `n / b` gives each chain as many transitions as a full-batch fit. On 40-subject linear-Gaussian models with an exact marginal likelihood, such minibatch fits with a quarter of the subjects ended as close to the maximum as full-batch fits (mean gap over eight seeds 0.10 against 0.08 log-likelihood units for MH, 0.05 against 0.04 for NUTS, 0.16 against 0.22 on a ragged layout), also without an MCMC burn-in, under a population prior and with a fixed correlation. Unscaled, MH and NUTS fits ended four to seven times further from the maximum, in mean gap. Use MH or NUTS for minibatch fits. MALA scales its proposals by a metric and adapts one step size per subject, which can still collapse a variance component under a minibatch, annealing notwithstanding:

- With its default prior metric, the proposals follow Ω. In 20 seeded fits of the growth model none ended more than 2 units below the maximum with full batches or with batches of a half, but 4 did with batches of a quarter (the worst 8.2).
- `DualAveragingMALA(metric = :hessian)` follows the posterior only when the model provides η Hessians. The growth model does, and all 20 fits stayed within 0.30 units in every mode. Without η Hessians the metric is the starting Ω, held fixed: as a variance's posterior narrows, the step size shrinks with it until the chains stop moving. On the `fixed Ω` test fixture that drove a weakly informed variance (maximum likelihood 0.081) to zero in 11 of 12 full-batch fits without annealing; annealed, the median fit is 0.062, the lowest 0.014.

What minibatches save, measured on 400 simulated oral one-compartment subjects ([`bench/saem_minibatch.jl`](https://github.com/nsiccha/NLMEEstimation.jl/blob/master/bench/saem_minibatch.jl), results in `bench/saem_minibatch_results.txt`; error is the largest population-coordinate distance from a 2000-iteration full-batch fit, in its standard errors, over three seeds):

| batch | passes over the subjects | subject evaluations | mean error (SE) |
| --- | --- | --- | --- |
| all 400 | 20 | 173,000 | 0.63 |
| all 400 | 50 | 391,000 | 0.53 |
| all 400 | 100 | 755,000 | 0.26 |
| 100 | 50 | 209,000 | 0.74 |
| 100 | 100 | 374,000 | 0.34 |
| 40 | 100 | 378,000 | 0.43 |
| 10 | 100 | 388,000 | 0.55 |

- Per pass over the subjects, minibatch fits were less accurate than full-batch fits, the more so the smaller the batch (0.34 to 0.55 against 0.26 standard errors after 100 passes); minibatches do not need fewer passes. Matching the full-data estimate to within its own standard error takes every subject's information, so no subsampling scheme gets below that.
- A minibatch pass evaluates each subject about half as often: one score step replaces the full-batch update's search. Per subject evaluation, batches of a quarter to a tenth of the subjects were more accurate than full batches (0.34 and 0.43 against 0.53 standard errors at about 380,000 evaluations), and batches of a fortieth about as accurate (0.55).
- What a minibatch bounds is each model call: `b` subjects per evaluation, so the time and memory of one iteration, and the population estimate moves `n / b` times per pass.
- Each iteration still does arithmetic for every subject (its statistics and the closed-form update) without evaluating the model: 1.3 to 4.6 ms per iteration for these 400 subjects (host load varied), against about 0.1 ms of model time with batches of 10. For a model this cheap, full batches finish first. Minibatches pay off when each subject's evaluation is expensive, as in a large batched program. Batches of a quarter to a tenth of the subjects, with iteration counts scaled by `n / b`, matched full-batch accuracy here.

`minibatch_size` equal to `n` is the full-batch algorithm.

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
- **SAEM fits.** All `(θ, σ, ω)` are estimated. A model with `GaussianObservations()` and an `NLMELayout` defaults to `FOCEApproximation`, the linearized covariance at the SAEM estimate. Every other SAEM fit, including ragged layouts and non-Gaussian likelihoods, defaults to `ImportanceSampling` (next section). Its draws, from the `rng` keyword, are held fixed while the coordinates move, so the estimated marginal is smooth in them, and moving ω or a mu-referenced θ coordinate reuses the conditional values computed at the estimate: only the remaining θ, σ and their pairs evaluate the model. `AdaptiveGaussHermite` is the deterministic alternative. It adapts to every point it is evaluated at: each evaluation finds the subject modes and curvature there again, warm-started from the estimate's modes with `gtol = 1e-10`, as `FOCEApproximation` does. Points held at the estimate's modes would not adapt, and with one node (Laplace) `R` would then be the complete-data information at those modes: too large, and without its cross terms. With one node it is the Laplace covariance, which subjects with many random effects need (next section).
- **Louis's identity.** `approximation = LouisInformation(; draws = 1000, burnin = 200)` builds `R` and `S` without a marginal likelihood or η curvature, from NUTS draws of every subject's η at the point: per subject, `R` adds the expected complete-data information minus the variance of the complete-data score, and `S` uses the mean scores (Louis 1982; the stochastic-approximation Fisher information of SAEM, as in Monolix and nlmixr2's `"sa"`). The complete-data density is taken in the mu-referenced coordinates, so the mu-referenced θ and Ω enter only through the Gaussian prior, in closed form; the other θ and σ use the model's population gradient, and their curvature its central differences. It works for any likelihood and layout and keeps no `n × n` matrix per subject. Its Monte Carlo error is largest for weakly informed variance components, where the two terms nearly cancel: on a twelve-subject linear-growth model, 10,000 draws gave the θ and σ standard errors within 1.2% and the Ω ones within 13% of the exact values; on eight nested subjects of 10–60 occasions, 2,000 draws gave all of them within 1%, and on six of 250–8,000 occasions (`bench/ebe_ragged_scale.jl`) the default 1,000 draws gave them within 1.3%. Compare two `rng` seeds before relying on its Ω standard errors.
- **Fixed covariance families.** When the model declares `fixed_omega`, the fixed parts carry no uncertainty and are not differentiated. The estimated coordinates (`result.coordinates`) then omit a fully fixed family, and a partially fixed family contributes its estimated log standard deviations, `log sd(…)`, and, when its correlation is estimated, its correlation coordinates `m(…)` (`m_jl = L_jl / L_jj` of its Cholesky factor). `result.point` stays the full `[θ; σ; ω]`.
- **Natural units.** `delta_method(result, f)` propagates the covariance to `f(x)`, where `x = [θ; σ; ω]`. For example, `x -> exp.(x[1:nθ])` gives log-normal typical values, and `x -> omega_elements(layout, x[nθ+nσ+1:end])` gives the Ω variances and covariances, ordered like ω.

**Relation to nlmixr2.** nlmixr2's covariance methods `"r,s"`, `"r"` and `"s"` (with `covFull = TRUE`) are built from the same two matrices. nlmixr2's SAEM documentation states them on the `−2 log L` scale, where `R` is twice this package's and `S` four times, and gives the covariances as `R⁻¹ S R⁻¹`, `2 R⁻¹` and `4 S⁻¹`; on the log-likelihood scale these are the definitions above. Its SAEM accepts the same three methods at the SAEM estimate. Its SAEM default, `"sa"`, is a different estimator: Louis's observed information, accumulated by stochastic approximation in a separate simulation phase at the estimate; `LouisInformation` is this package's form of it. The suite compares the two packages on R's theophylline data (12 subjects, one-compartment oral model, additive error, diagonal Ω) at nlmixr2 7.0.1's estimate (`test/covariance.jl`):

- `ofv` minus `N_obs log(2π)` equals nlmixr2's `OBJF` to within 1.4e-4. nlmixr2 omits that constant; `ofv` keeps it.
- `:s` standard errors agree within 0.2%, and `:rs` within 2.4%.
- `:r` standard errors agree within 1.6% for the typical values and the residual SD. For the Ω variances, nlmixr2's are 5.5–14% smaller. On this package's log-likelihood scale, nlmixr2's implied `R` exceeds this package's by 5–7 (about `N/2 = 6`) on the diagonal entries of the three typical values and the three Ω coordinates, and by about 1 for the residual SD. This package's `R` matches the exact Hessian of a linear-Gaussian model to 1e-6. On theophylline, steps of `3e-3`, `1e-3` and `3e-4` give the same `R` to within 7e-5, and Newton and BFGS subject modes give the same `R` to within 3e-7. The cause of nlmixr2's difference has not been identified.

`LouisInformation` and nlmixr2's SAEM `"sa"` are compared at nlmixr2 7.0.1's SAEM estimate on the same data and model (`test/louis.jl`). `LouisInformation`'s standard errors agree with those of `:r` with `AdaptiveGaussHermite(nodes = 3)` within 1% at 5,000 draws and within 3% at 1,000; importance sampling's agree too. nlmixr2's are 2–20% smaller (2% for ka's typical value, 20% for the residual SD), and the same with 500 and 5,000 iterations of its covariance phase, so the difference is not its Monte Carlo error. The typical values are mu-referenced, so their complete-data information is exactly `N Ω⁻¹`, and what the unobserved η take from it is the variance of the score: here 2.2, 24.8 and 187 for ka, CL and V, and in nlmixr2's information about two-thirds of that (1.5, 15.9 and 125). The cause has not been identified.

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

The first two need only conditional log-likelihood values, plus η gradients for the subject modes, so they apply to any likelihood and to ragged layouts. The curvature comes from the model's η Hessian blocks when available, from sparse Hessians when the model declares `eta_hessian_sparsity`, and otherwise from central differences of batched η gradients.

**Subjects with many random effects.** A subject with thousands of nested effects needs the model to declare `eta_hessian_sparsity` (see [PPL and AD requirements](requirements.md)); its curvature is then a sparse matrix, and the proposal points go through its sparse Cholesky factor. Use `AdaptiveGaussHermite(nodes = 1)`, the Laplace approximation, there. A tensor rule with more nodes costs `nodes^d` points. Importance sampling degenerates as `d` grows: with the default t₄ proposal on a Gaussian posterior, the effective sample size of 500 draws fell from 324 to 141 to 46 at 11, 101 and 1,001 random effects. A Gaussian proposal (`df = Inf`) stays exact only where the posterior itself is Gaussian. `bench/ebe_ragged_scale.jl` measures the Laplace marginal and covariance step at up to 25,872 random effects per subject.

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
