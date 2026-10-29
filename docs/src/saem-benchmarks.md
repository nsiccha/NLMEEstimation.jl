# SAEM subject-kernel benchmarks

This page compares tuned **per-subject** Metropolis–Hastings (MH) and WarmupHMC NUTS kernels. Six subjects each have a three-dimensional random effect `ηᵢ`. MH batches six independent proposals per model query; NUTS advances six separate three-dimensional chains. Neither proposes a joint 18-dimensional state.

With in-memory NUTS transitions, the selected NUTS setting delivered **15,081 median ESS/s**, versus **12,376 median ESS/s** for the selected MH setting on untouched fixed-target seeds. A separate 40-iteration SAEM fit took 0.012675 s with NUTS and 0.003113 s with MH. ESS/s describes a stationary target; the short fit measures changing-target cost and does not establish estimator accuracy.

Those short fits used the historical `numeric_mstep=:mixture` and an M-step every five iterations. They compare kernel overhead under that recorded setup; they are not a benchmark of the current default every-iteration M-step. The default chooses the online numerical update only when all structural θ coordinates have a declared sufficient-statistic update.

"Best" means the best observed ESS/s in the tested grid, subject to zero observed NUTS divergences in fixed-target and changing-target screens. It is not a global optimum or a guarantee of zero future divergences.

## Model and environment

The [Theoph fixture](https://github.com/nsiccha/NLMEEstimation.jl/blob/7502db02681b184cc153be41f99fb548cf166f02/test/common.jl) uses public theophylline concentrations, an oral one-compartment model, additive observation error, and Enzyme derivatives. The first six subjects start at `θ = [log(1.57), log(2.75), log(31.5), 0]`, `σ = [log(0.7)]`, and `Ω = diag(0.6², 0.26², 0.14²)`. At fixed population parameters, each subject has target

```math
p(\eta_i \mid y_i, \theta, \sigma, \Omega) \propto
p(y_i \mid \theta, \sigma, \eta_i)\,\mathcal N(\eta_i; 0, \Omega).
```

Measurements ran on a shared Linux x86-64 AMD Ryzen 9 9950X host (16 cores, 32 threads), Julia 1.10.12, one BLAS thread, Enzyme 0.13.215, [NLMEEstimation `f82ea8d`](https://github.com/nsiccha/NLMEEstimation.jl/commit/f82ea8d0115e4f5a9fc32f3136708c0b57d8964e), WarmupHMC [`d3d55e0`](https://github.com/nsiccha/WarmupHMC.jl/commit/d3d55e048aeafe4516daba0dc5f9e8550c8f8e9e) (version 0.2.1), and Treebars [`7426b9c`](https://github.com/nsiccha/Treebars.jl/commit/7426b9c8dc914d767c28ff22f1f75631f3afc41d). Timings can vary on a shared host; the two final-script runs below illustrate this.

## What executes

The [MH transition](https://github.com/nsiccha/NLMEEstimation.jl/blob/6c0010311a30b694d83cbda1852cdcd3d18a9aaa/src/saem.jl) uses a prior proposal and then a covariance-scaled random walk. Prior proposal density cancels from its acceptance ratio; the random-walk ratio includes the Gaussian prior. Likelihood queries are batched, with a separate acceptance decision per subject. These are its essential operations:

```julia
proposal = Matrix(prior.cholesky_factor * randn(rng, m, n))
proposed_ℓ = conditional_loglikelihoods(model, subjects, θ, σ, proposal)
for k in 1:n
    if log(rand(rng)) < proposed_ℓ[k] - ℓ[k]
        H[:, k] = proposal[:, k]
        ℓ[k] = proposed_ℓ[k]
    end
end
proposal = H + scale * (prior.cholesky_factor * randn(rng, m, n))
proposed_ℓ = conditional_loglikelihoods(model, subjects, θ, σ, proposal)
for k in 1:n
    logratio = proposed_ℓ[k] - ℓ[k] +
        re_logprior(prior, view(proposal, :, k)) -
        re_logprior(prior, view(H, :, k))
    if log(rand(rng)) < logratio
        H[:, k] = proposal[:, k]
        ℓ[k] = proposed_ℓ[k]
    end
end
```

The [NUTS integration](https://github.com/nsiccha/NLMEEstimation.jl/blob/6c0010311a30b694d83cbda1852cdcd3d18a9aaa/src/saem.jl) adds the analytic Gaussian-prior gradient to the model's conditional η gradient. Per subject, it inverts a regularized local negative log-posterior Hessian for the initial covariance, adapts once, and loads a fixed-kernel state. The selected setting uses the full three-by-three covariance; the public default uses its diagonal.

```julia
function LDP.logdensity_and_gradient(target::_SAEMSubjectTarget, η)
    ℓ, g = conditional_loglikelihood_and_gradient(target.model, target.subject,
        target.θ, target.σ, η)
    return ℓ + re_logprior(target.prior, η), g - target.prior.precision * η
end

target = _saem_target(model, i, θ, σ, prior)
η = copy(view(H, :, k))
Σ = _saem_hessian_metric(model, i, θ, σ, prior, η;
    curvature_floor=kernel.curvature_floor,
    diagonal=kernel.metric == :diagonal)
warm_dir = joinpath(root, "warm_$(k)")
WarmupHMC.adaptive_warmup_mcmc(Xoshiro(rand(rng, UInt64)), target;
    init=(position=η, squared_scale=Σ), checkpoint_dir=warm_dir,
    n_draws=kernel.warmup_draws,
    n_evaluations=kernel.warmup_evaluations,
    recording_target=kernel.warmup_draws,
    stepsize_adaptation_limit=kernel.stepsize_adaptation_limit,
    max_tree_depth=kernel.max_tree_depth,
    target_acceptance_rate=kernel.target_acceptance_rate,
    nonlinear_adapt=false, monitor_ess=false)
state = WarmupHMC.FixedKernelNUTSState(joinpath(warm_dir, "cp_latest.jls");
    max_tree_depth=kernel.max_tree_depth)

# Once per later SAEM round, with current population parameters:
target = _saem_target(model, i, θ, σ, prior)
step = WarmupHMC.fixed_kernel_nuts_step(state, target)
state = step.state
H[:, k] = state.position
divergences += step.divergent
```

Warmup writes one checkpoint per subject. Subsequent transitions, including those under changed population parameters, happen in memory. Each transition reevaluates its saved position under the current target; there is no file resume per draw. The [fixed-target benchmark driver](https://github.com/nsiccha/NLMEEstimation.jl/blob/6c0010311a30b694d83cbda1852cdcd3d18a9aaa/bench/saem_kernels.jl) uses the same step API and collects draws for ESS.

The public interaction for the selected settings is:

```julia
using NLMEEstimation, Random
include("test/common.jl")
using .TestCommon
model = theoph_model()
ω = omega_parameters(nlme_layout(model), Ω_THEOPH)
mh = StandardSAEMMH()
nuts = WarmupHMCNUTS(metric=:dense, warmup_draws=20,
    warmup_evaluations=50, stepsize_adaptation_limit=20,
    max_tree_depth=8, target_acceptance_rate=0.8)
base = (; iterations=40, burnin=15, mcmc_burnin=5,
    mh_rounds=1, numeric_maxiter=2, mstep_interval=5,
    numeric_mstep=:mixture)
mh_options = SAEMOptions(; base..., rw_scale=0.45, adapt_iterations=0)
nuts_options = SAEMOptions(; base..., adapt_iterations=0)
mh_fit = saem(model, θ_THEOPH, σ_THEOPH, ω;
    rng=MersenneTwister(2110), subjects=collect(1:6),
    options=mh_options, kernel=mh)
nuts_fit = saem(model, θ_THEOPH, σ_THEOPH, ω;
    rng=MersenneTwister(2110), subjects=collect(1:6),
    options=nuts_options, kernel=nuts)
```

## Tuning and measurement

Every fixed-target candidate retains **1,600 draws per subject**. MH runs 200 burn-in sweeps and tries six fixed proposal scales (0.25, 0.35, 0.45, 0.6, 0.8, 1.2), plus adaptation to acceptance targets 0.2, 0.3, or 0.4. The broad NUTS grid tries three initial metrics (diagonal Hessian, dense Hessian, prior covariance), warmup draw floors 8 or 20, initial gradient-evaluation budgets 50 or 100, and maximum tree depths 4 or 8, with acceptance target 0.8. Twelve follow-up runs vary acceptance target over 0.6–0.9 for top fixed-target NUTS settings. The [complete sweep and crosscheck table](https://github.com/nsiccha/NLMEEstimation.jl/blob/f82ea8d0115e4f5a9fc32f3136708c0b57d8964e/bench/saem_kernel_sweep.tsv) records all 49 measured rows.

The broad sweep uses seeds 2026–2030; seeds 2050–2054 check provisional choices. Four MH and six NUTS finalists were measured on seeds 2060–2069. NUTS finalists also ran 40-iteration changing-target SAEM fits on seeds **2080–2099**. A NUTS finalist qualified only with zero fixed-target and changing-target divergences and finite fit estimates. Final measurements use untouched seeds **2110–2119**.

The first five-seed screen (2080–2084) let an eight-draw NUTS candidate through, but its next ten fits (2090–2099) had 28 divergences. We preserved this [diagnostic output](https://github.com/nsiccha/NLMEEstimation.jl/blob/f82ea8d0115e4f5a9fc32f3136708c0b57d8964e/bench/saem_kernel_gate_diagnostic.txt), included those seeds in qualification, and reserved new final seeds.

ESS uses the initial-positive-pair estimate in the benchmark driver. Each seed's ESS is the median of 18 subject-by-coordinate traces. Reported ESS and wall time are medians across seeds; ESS/s is the median of each seed's ESS divided by its own sampling wall time. This estimator caps ESS at 1,600. Wall time includes MH burn-in or NUTS warmup/checkpoint and retained draws, but excludes Julia compilation, model construction, and ESS calculation. Fit time includes the complete `saem` call, five MCMC burn-in sweeps, and the same M-step settings.

### Divergence qualification

All six NUTS finalists used a dense Hessian metric. Fixed-target results use ten seeds; changing-target divergences sum over twenty fits. Warmup is draw floor / initial gradient evaluations. Only the bold row passed both zero-divergence screens.

| Warmup | Tree depth | Acceptance target | Fixed-target ESS/s | Fixed divergences | Changing-target divergences |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 / 100 | 8 | 0.7 | 20,344 | 24 | 29 |
| 8 / 50 | 8 | 0.7 | 20,531 | 0 | 60 |
| 8 / 50 | 8 | 0.8 | 21,523 | 0 | 26 |
| 8 / 100 | 8 | 0.8 | 19,905 | 0 | 68 |
| 20 / 50 | 4 | 0.8 | 14,671 | 0 | 1 |
| **20 / 50** | **8** | **0.8** | **15,732** | **0** | **0** |

MH finalists scored 11,547, **13,161**, 12,228, and 12,957 ESS/s for fixed scale 0.35, **fixed scale 0.45**, adaptation target 0.2, and adaptation target 0.3. A repeat scored 12,955 for fixed 0.45 and 13,049 for adaptive 0.3. Their order is sensitive to shared-host timing noise; they are a practical tie.

### Fresh-seed comparison

The [primary run](https://github.com/nsiccha/NLMEEstimation.jl/blob/f82ea8d0115e4f5a9fc32f3136708c0b57d8964e/bench/saem_kernel_final_results.txt) and its [repeat](https://github.com/nsiccha/NLMEEstimation.jl/blob/f82ea8d0115e4f5a9fc32f3136708c0b57d8964e/bench/saem_kernel_final_repeat.txt) contain every selection, qualification, and final measurement. In both, all ten short fits returned finite population estimates. The repeat selected adaptive MH target 0.3 in the near-tied tuning stage, so its MH row is a sensitivity check, not a replicate of fixed-scale MH.

| Run and kernel | Fixed seconds | Median ESS/coordinate | Median ESS/s | RW acceptance | Fixed divergences | Fit seconds | Maximum fit divergences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Primary: MH, fixed scale 0.45 | **0.010001** | 125.7 | 12,376 | 26.9% | — | **0.003113** | — |
| Primary: NUTS, dense Hessian, 20/50, target 0.8 | 0.085022 | **1,294.5** | **15,081** | — | 0 | 0.012675 | 0 |
| Repeat: MH, adaptive target 0.3 | **0.009939** | 123.5 | 11,827 | 30.1% | — | **0.003106** | — |
| Repeat: NUTS, same setting | 0.085151 | **1,294.5** | **15,372** | — | 0 | 0.011577 | 0 |

NUTS yielded about 10.3 times the ESS per retained draw and 22% more ESS per second than fixed-scale MH in the primary stationary experiment. In the short changing-target fit, MH finished about four times faster. The fit is a cost check: 40 SAEM iterations do not establish parameter recovery. NUTS still needs more work per transition, even without per-draw disk resume.

These ESS values apply **only to stationary targets**. SAEM changes its target across outer iterations, so positions from different targets cannot be pooled for one ESS. The result is specific to six subjects and three effects; posterior geometry, dimension, model cost, and fit length can change the preferred kernel. Some warmups logged a TSVD failure with a successful power-iteration fallback; divergence counts above come from sampler results.

## Reproduce

The scripts are [`bench/saem_kernel_tuning.jl`](https://github.com/nsiccha/NLMEEstimation.jl/blob/f82ea8d0115e4f5a9fc32f3136708c0b57d8964e/bench/saem_kernel_tuning.jl), [`bench/saem_kernel_final.jl`](https://github.com/nsiccha/NLMEEstimation.jl/blob/f82ea8d0115e4f5a9fc32f3136708c0b57d8964e/bench/saem_kernel_final.jl), and the driver linked above. From a clone, develop this package and both unregistered dependencies in one resolver call, then run:

```sh
git clone --branch dev https://github.com/nsiccha/WarmupHMC.jl.git ../WarmupHMC.jl
git clone --branch main https://github.com/nsiccha/Treebars.jl.git ../Treebars.jl
julia --project=test -e 'using Pkg; Pkg.develop([PackageSpec(path=pwd()), PackageSpec(path="../WarmupHMC.jl"), PackageSpec(path="../Treebars.jl")]); Pkg.instantiate()'
julia --project=test bench/saem_kernel_tuning.jl
julia --project=test bench/saem_kernel_final.jl
```

For an exact source replay, check out the three SHAs above. Runtime scores vary with host load. The [earlier file-backed comparison](https://github.com/nsiccha/NLMEEstimation.jl/blob/d733d33b9f34e094efa2e669ada4e0c2baa2d7f9/docs/src/saem-benchmarks.md) is historical context: its NUTS fit reopened a checkpointed stream once per subject per SAEM round and therefore measured disk resume overhead. It should not be used as an estimate of in-memory NUTS efficiency.

## Experimental batched MALA with ongoing adaptation

This later trial tests `DualAveragingMALA()` with **one proposal per subject per SAEM iteration**, a separate step size for every subject, dual averaging after every proposal, and `mcmc_burnin=0`. The earlier MH/NUTS ESS results above used different settings and remain a separate experiment. MALA is available as an experimental kernel; the default is still `StandardSAEMMH()`.

For subject `i`, let `fᵢ(η)` be its complete-data log density and `gᵢ(η) = ∇fᵢ(η)`. With proposal covariance `Mᵢ` and step size `εᵢ`, the proposal and correction are

```math
q_i(z\mid\eta)=\mathcal N\!\left(z;\eta+\frac{\varepsilon_i^2}{2}M_i g_i(\eta),\varepsilon_i^2 M_i\right),\qquad
\log r_i=f_i(z)-f_i(\eta)+\log q_i(\eta\mid z)-\log q_i(z\mid\eta).
```

The Gaussian prior covariance supplies `Mᵢ` for `metric=:prior`. `metric=:hessian` takes a regularized inverse local posterior Hessian **once at initialization** for each subject; it does not update that matrix as population parameters move. The model protocol evaluates all current positions or all proposals together. The Metropolis accept/reject and dual-averaging step-size update remain per subject.

```julia
kernel = DualAveragingMALA(metric=:prior, initial_stepsize=0.05,
    target_acceptance_rate=0.574)
options = SAEMOptions(iterations=120, burnin=40, mcmc_burnin=0,
    mh_rounds=1, mstep_interval=5, numeric_maxiter=3,
    numeric_mstep=:mixture)
fit = saem(model, θ_THEOPH, σ_THEOPH, ω;
    rng=MersenneTwister(601), options, kernel)
fit.mala_stepsizes      # final step size for each subject
fit.mala_acceptance     # each subject's acceptance fraction
```

Within one unchanged target, an accepted proposal's log density and gradient become the next source values; rejection retains the old values. Thus, after the first sweep, the usual cost is **one new batched value-and-gradient call** per sweep. A θ or σ M-step invalidates the cache and requires one extra source batch on the next sweep. An Ω-only M-step updates the Gaussian-prior part of the cached value and gradient algebraically. A rare invalid proposal triggers a value screening batch and a repaired gradient batch so it can be rejected safely. The [implementation](https://github.com/nsiccha/NLMEEstimation.jl/blob/774edac644329cb1a5ec64429066103935d4f4d6/src/saem.jl) and [tests](https://github.com/nsiccha/NLMEEstimation.jl/blob/774edac644329cb1a5ec64429066103935d4f4d6/test/saem_mala.jl) show these paths.

### Methods and measured results

The [reproducer](https://github.com/nsiccha/NLMEEstimation.jl/blob/e3c323aa2eb53b0b53d3dd528db743306749e684/bench/saem_mala_trial.jl) and [complete 49-line output](https://github.com/nsiccha/NLMEEstimation.jl/blob/ad08a560397f59ce1652aea8c205e8b0f0b8de04/bench/saem_mala_trial_results.tsv) contain every tested configuration and fit row. Measurements used the AMD Ryzen 9 9950X host, Julia 1.10.12, one Julia and one BLAS thread, Enzyme 0.13.215, NLMEEstimation source `774edac` with benchmark timing fix `e3c323a`, WarmupHMC `d3d55e0`, and Treebars `7426b9c`. Timers exclude compilation and fixture construction; they include MALA's initial Hessian setup when requested and MH's initial likelihood evaluation. Shared-host millisecond timings are noisy.

The fixed-target fixture has six independent **two-dimensional correlated Gaussian** subject posteriors with different curvatures and known posterior means. No draws were discarded. Four MH scales (0.25, 0.5, 0.8, 1.2) and 18 MALA combinations (prior or Hessian metric, initial step size 0.05/0.1/0.2, acceptance target 0.45/0.574/0.7) were screened on seeds 301–308. The lowest observed `RMSE² × median time` selected MH scale 0.8 and Hessian MALA with initial step size 0.1, target 0.574. Held-out seeds 401–416 used 400 transitions; seeds 701–716 used 4,000 transitions with the same selected settings. RMSE compares each run's all-draw subject-by-coordinate posterior mean with the analytic mean; it is **not ESS**.

| Held-out transitions per subject | Kernel | Mean RMSE | Median wall time | Acceptance |
| ---: | --- | ---: | ---: | ---: |
| 400 | MH, scale 0.8 | 0.04648 | 0.001021 s | 0.300 random walk |
| 400 | MALA, Hessian | **0.03425** | 0.002265 s | 0.574 |
| 4,000 | MH, scale 0.8 | 0.01428 | 0.011188 s | 0.295 random walk |
| 4,000 | MALA, Hessian | **0.01077** | 0.023757 s | 0.574 |

MALA lowered the mean error by roughly 25% at both horizons but took about 2.2 times as long. The observed `RMSE² × time` was about 21% higher for MALA at both horizons, so the result does not show a compute-efficiency win for this low-dimensional fixture. The selected MH scale changed between repeated millisecond-scale timing runs; 0.5 and 0.8 are close contenders in the complete table.

For the nonlinear Theoph model, a separate run used six subjects, 60 SAEM iterations, `burnin=20`, one kernel round and an M-step every five iterations on seeds 501–506. All runs used `mcmc_burnin=0`. The longer check used all 12 subjects, 120 iterations, `burnin=40` and seeds 601–603. Fit time includes kernel transitions and M-steps. The values below are medians across seeds; `diag(Ω)` is in random-effect variance units.

| Fit | Kernel | Median time | Median acceptance | Median `σ[1]` | Median `diag(Ω)` |
| --- | --- | ---: | ---: | ---: | --- |
| 6 × 60 | MH, scale 0.8 | 0.007193 s | 0.310 random walk | −0.0999 | [0.0287, 0.0624, 0.0135] |
| 6 × 60 | MALA, prior, initial 0.05, target 0.574 | 0.007733 s | 0.571 | −0.1167 | [0.0285, 0.0691, 0.0099] |
| 6 × 60 | MALA, Hessian, initial 0.1, target 0.574 | 0.007489 s | 0.554 | 0.2707 | [0.0114, 0.0052, 0.0003] |
| 12 × 120 | MH, scale 0.8 | 0.066519 s | 0.143 random walk | −0.3572 | [0.4239, 0.0710, 0.0166] |
| 12 × 120 | MALA, prior, initial 0.05, target 0.574 | 0.068583 s | 0.579 | −0.3625 | [0.4444, 0.0774, 0.0139] |
| 12 × 120 | MALA, Hessian, initial 0.1, target 0.574 | 0.067049 s | 0.579 | −0.2467 | [0.5071, 0.1194, 0.0367] |

The Gaussian-selected Hessian MALA setting produced near-zero variance estimates in the shorter six-subject fit despite acceptance near its target; it did not transfer reliably to that changing-target setting. The prior-metric 0.05 setting was close to MH in time and fit medians in the longer run. These short fits do not establish estimator accuracy or convergence. The same no-MCMC-burn-in NUTS configuration had 49 divergent transitions across the six short fits, so this trial does not qualify a new NUTS comparison; the earlier qualified NUTS benchmark used a different setup.

The Theoph reference model uses per-subject AD behind the batch protocol, whereas the Gaussian fixture supplies a fused batch value-and-gradient method. The tests also exercise default batched Enzyme AD. Throughput on a production joint-model backend still needs its own measurement. Continuous raw dual averaging under a changing target has no convergence guarantee from a Metropolis correction at each fixed target. The measured result supports keeping MH as the default while MALA remains available for model-specific trials.

### A nonlinear model with four or eight random effects

The [nonlinear benchmark driver](https://github.com/nsiccha/NLMEEstimation.jl/blob/8c137ac/bench/saem_mala_nonlinear.jl) and [complete 52-line output](https://github.com/nsiccha/NLMEEstimation.jl/blob/f2890d1/bench/saem_mala_nonlinear_results.tsv) replace the Gaussian dimension stress test as the evidence for this question. Six simulated subjects have repeated Hill Emax dose responses at 12 doses from 0.08 to 20. One endpoint has four random effects: baseline, log maximum response, log half-maximal dose, and log Hill slope. Two endpoints give eight effects per subject. Endpoint responses share a Gaussian random-effect prior with pairwise correlation 0.2; its standard deviations repeat `[0.15, 0.25, 0.25, 0.18]`. Observation error is independent Gaussian with SD 0.12. Data generation uses `MersenneTwister(1107)`. This is a synthetic nonlinear mixed-effects model, not a clinical data analysis.

Each fixed-target subject chain starts at zero. The population parameters are held fixed during these measurements, so the reference quantity is each subject's conditional posterior mean. Two independent 100,000-draw importance calculations per subject use a full-dimensional Laplace proposal with covariance inflated by 2.25. The smallest importance effective sample size was 47,413 at four effects and 22,768 at eight. The largest coordinate difference between the two reference means was 0.00098 and 0.00257, respectively, much smaller than the kernel errors below. The analytic likelihood gradient and Hessian were checked against ForwardDiff to absolute errors below `6e-14` at a representative eight-effect point.

For each dimension, eight selection seeds (`301:308`) screen five MH random-walk scales proportional to `sqrt(2/d)` and twelve MALA settings: prior or initial-Hessian metric, initial step size 0.05/0.1/0.2, and acceptance target 0.574/0.7. The lowest observed `RMSE² × median wall time` chooses one setting per kernel. Separate seeds `401:416` evaluate 400 transitions, and `701:708` evaluate 4,000. Every draw is retained, with no initial MCMC burn-in. RMSE is the across-seed root mean squared error of all-draw subject-by-coordinate posterior means relative to the importance reference. Timers include MH's initial likelihood evaluation and MALA's initial Hessian setup when selected, but exclude compilation, data generation, and reference integration. The host was the shared AMD Ryzen 9 9950X Linux x86-64 server, with Julia 1.10.12, one Julia thread, and one BLAS thread. The benchmark uses analytic batched likelihood gradients; production AD could change the timing ratio.

| Effects | Transitions per subject | Tuned MH: RMSE / time | Tuned MALA: RMSE / time | MALA/MH `RMSE² × time` |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 400 | 0.04080 / 1.808 ms | 0.05914 / 3.106 ms | 3.61× |
| 4 | 4,000 | 0.01343 / 17.976 ms | 0.02038 / 31.217 ms | 4.00× |
| 8 | 400 | 0.07656 / 3.023 ms | 0.06984 / 4.108 ms | 1.13× |
| 8 | 4,000 | 0.02190 / 30.323 ms | 0.02260 / 41.729 ms | 1.47× |

**Lower `RMSE² × time` is better.** At four effects, the selected MH scale was 0.849; MALA used the prior metric, initial step 0.05, and target 0.574. At eight effects, the selected MH scale was 0.4; MALA used the prior metric, initial step 0.2, and target 0.574. The prior-independence half of MH accepted only 3.7% and 1.1% of proposals at 400 transitions, respectively. The tuned random-walk half still made MH more efficient on this fixture, though the eight-effect 400-transition difference is small and shared-host millisecond timing varies. A repeated run chose a different, near-tied MH scale at four effects without changing this direction.

This experiment answers a narrower question than SAEM convergence. It compares **fixed conditional targets** in a plausible four-to-eight-effect nonlinear model; it does not include population M-steps, continuous target changes, multimodal posteriors, or a costly joint-model AD backend. The two-endpoint eight-effect target has genuine cross-endpoint prior coupling, but it is only one geometry. The result does not support replacing the MH default with MALA on dimension alone. MALA remains an experimental option for models where a model-specific benchmark shows an advantage.

## Every-iteration SAEM M-step on BRM + RKPPL

The preceding short-fit comparisons used a numerical M-step every third or fifth SA iteration. The [80-subject BRM reproducer](https://github.com/nsiccha/NLMEEstimation.jl/blob/master/bench/saem_streaming_brm.jl) instead runs the M-step **at every iteration**, with one batched prior proposal and one batched random-walk MH proposal per subject. It uses the [same public IV-bolus model](pk-comparison.md), with 80 independently simulated subjects and five observations each. The [complete recorded output](https://github.com/nsiccha/NLMEEstimation.jl/blob/master/bench/saem_streaming_brm_results.txt) includes estimates, acceptance, allocations, and cold-start costs.

The Gaussian random-effect M-step already keeps per-subject running first and second moments of `ϕᵢ = Xᵢβ + ηᵢ`; it does not need a draw history. For a non-exponential residual likelihood such as this model's `σ + 0.03 × prediction`, the online mode finds the conditional-likelihood optimizer `z*` for the current draw and updates the remaining model coordinates as `z ← z + γₖ(z* − z)`. The implementation works in NLMEEstimation's **unconstrained** parameter coordinates. This follows the practical optimizer-averaging pattern in the [saemix methodology guide, §2.1.1](https://saemixdevelopment.github.io/saemix_bookdown/pdfs/userguide.pdf), whose displayed residual-parameter updates are written on the natural scale. It is therefore an analogous coordinate-specific update, not an algebraically identical natural-scale update.

`SAEMOptions(numeric_mstep=:auto)` is the default. It selects bounded-memory `:online` only when every fitted structural `θ` coordinate has a declared `MuReferencing` update; otherwise it selects the historical `:mixture` objective. At the older BRM revision `a1ac62a` used for the first table, only the two intercepts are declared, so its default resolves to `:mixture`. That benchmark explicitly requests `:online` to measure an approximate weight-slope update. The later BRM revision `1e7fbaa` adds the subject-specific weight slope to `MuReferencing`; the default-mode check below uses that revision. `SAEMResult.numeric_mstep` reports the mode actually used.

The two measurements used Julia 1.10.12, Enzyme 0.13.210, one Julia and one BLAS thread, a shared AMD Ryzen 9 9950X Linux host, BRM `a1ac62ad4b2e0023a3d6c1dd6e8ce274c7619058`, ReactiveKernels/RKPPL `a2268e47b3a800df356ae6961c2caf863922a102`, WarmupHMC `65354994484a9b84075fd82049df7b3df922b3a4`, and NLMEEstimation implementation `304a349eb3fefc6e7606bdfff2c73b054d9d748d`. The first measurement used the same implementation just before that commit. Synthetic data used `Xoshiro(20261011)`; every fit used `Xoshiro(5)`. The options were `mcmc_burnin=15`, `mh_rounds=1`, `numeric_maxiter=12`, `mstep_interval=1`, and burn-in plus MH adaptation each set to half of the requested 100 or 200 SA iterations. After four-iteration compile calls, `GC.gc()` preceded each timed fit. Timings include the complete `saem` call and exclude model construction, compilation, protocol checking, and printing. The repeated driver also confirmed that `:auto` resolves to `:mixture` for this model.

| Numerical M-step | SA iterations | Warm time, run 1 | Warm time, run 2 | Allocation per run | Random-walk acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Online, including undeclared slope | 100 | 0.322 s | 0.186 s | 0.927 GB | 29.2% |
| Online, including undeclared slope | 200 | 0.475 s | 0.369 s | 1.882 GB | 29.2% |
| Historical running mixture | 100 | 1.908 s | 1.706 s | 6.460 GB | 29.2% |
| Historical running mixture | 200 | 7.203 s | 6.069 s | 23.346 GB | 29.1% |

The online mode cut warm fit time by 5.9× and 9.2× at 100 iterations, and by 15.2× and 16.4× at 200 iterations across the two runs. Its cumulative allocations approximately doubled with the iteration count; the mixture's allocations rose by 3.6×. The 100- and 200-iteration settings have different burn-in and adaptation lengths, so the comparison shows work scaling for those schedules, not a convergence rate. Fitted coordinates were finite and close across modes in this synthetic fixture; this is not evidence that their estimators have the same bias or long-run behavior.

Cold costs were separate: constructing the 80-subject BRM model took 53.5 and 56.9 s with 7.48 GB cumulative allocation, and the first four-iteration online fit took 118.0 and 117.1 s with compilation. Those costs matter for one-off fits and are not included in the warm table. The older BRM revision could not select the bounded-memory mode by default; the updated trait and default-mode measurements appear below.

To reproduce from this repository root, use a fresh dependency directory. The existing example setup supplies the nested RK packages; update the two source revisions that differ from its pinned comparison and develop this checkout in one resolver call:

```sh
export BENCH_DEPS=/path/to/empty/benchmark-deps
julia --project=docs/examples docs/examples/setup_env.jl "$BENCH_DEPS"
git -C "$BENCH_DEPS/BayesianRegressionModels" fetch origin a1ac62ad4b2e0023a3d6c1dd6e8ce274c7619058
git -C "$BENCH_DEPS/BayesianRegressionModels" checkout --detach a1ac62ad4b2e0023a3d6c1dd6e8ce274c7619058
git -C "$BENCH_DEPS/WarmupHMC" fetch origin 65354994484a9b84075fd82049df7b3df922b3a4
git -C "$BENCH_DEPS/WarmupHMC" checkout --detach 65354994484a9b84075fd82049df7b3df922b3a4
julia --project=docs/examples -e 'using Pkg; root=ENV["BENCH_DEPS"]; Pkg.develop(Pkg.PackageSpec[Pkg.PackageSpec(path=pwd()), Pkg.PackageSpec(path=joinpath(root,"BayesianRegressionModels")), Pkg.PackageSpec(path=joinpath(root,"WarmupHMC"))]); Pkg.instantiate()'
julia --project=docs/examples bench/saem_streaming_brm.jl
```

Use NLMEEstimation implementation commit `304a349eb3fefc6e7606bdfff2c73b054d9d748d` and the linked driver for an exact source replay. Timings vary with host load and compiler cache state.

## Default SAEM after BRM weight-slope MuReferencing

BRM revision [`1e7fbaa`](https://github.com/nsiccha/BayesianRegressionModels.jl/commit/1e7fbaa0d3e6f462e5ae4b2cc74056a7542f4c08) certifies that the weight slope is constant within each subject and can be absorbed through that subject's clearance intercept random effect. Its `MuReferencing` design contains the actual subject weight. On this same 80-subject IV-bolus fixture, all three fitted structural `θ` coordinates then use the Gaussian running-statistics M-step. The remaining residual `σ` uses the online current-draw optimizer. The benchmark asserts that `numeric_mstep=:auto` resolves to `:online`; this is the bounded-memory default path for this model.

The [new driver](https://github.com/nsiccha/NLMEEstimation.jl/blob/master/bench/saem_streaming_brm_mu.jl) and [complete output from both runs](https://github.com/nsiccha/NLMEEstimation.jl/blob/master/bench/saem_streaming_brm_mu_results.txt) use NLMEEstimation implementation `fbca2224af8c0f1df5d9f0b6468fee7dc16dda58`, BRM implementation tip `444659b2d79cd0c2f753bfa9a692e6c897589255` in the first run, and final reviewed BRM tip `1e7fbaa0d3e6f462e5ae4b2cc74056a7542f4c08` in the second. The latter adds only a test; their `src/` and `ext/` trees are identical. ReactiveKernels/RKPPL remained at `a2268e47b3a800df356ae6961c2caf863922a102`, WarmupHMC at `65354994484a9b84075fd82049df7b3df922b3a4`, and Enzyme at 0.13.210. Julia 1.10.12 ran with one Julia and one BLAS thread on the shared AMD Ryzen 9 9950X Linux host. Data, fit seeds, MCMC proposals, iteration schedules, and timer boundaries are identical to the preceding experiment. The added four-iteration default check produced exactly the explicit online estimate in both runs.

| Numerical M-step | SA iterations | Warm time, first/final BRM tip | Allocation per run | Random-walk acceptance |
| --- | ---: | ---: | ---: | ---: |
| Default `:auto → :online` | 100 | 0.117 / 0.121 s | 0.848 GB | 29.2% |
| Default `:auto → :online` | 200 | 0.215 / 0.219 s | 1.616 GB | 29.8% |
| Historical `:mixture` | 100 | 0.903 / 0.893 s | 3.906 GB | 29.2% |
| Historical `:mixture` | 200 | 3.475 / 3.414 s | 14.206 GB | 29.8% |

At the final BRM tip, the default mode was 7.4× faster at 100 iterations and 15.6× faster at 200 iterations than the historical mixture. Its cumulative allocations grew roughly in proportion to iterations, whereas the retained-draw mixture grew from 3.906 to 14.206 GB. The fitted `θ` and `diag(Ω)` were identical between modes in these seeded fits; `σ` differed by about 0.001 in its unconstrained coordinate. Those close results are a sanity check, not an equivalence or convergence proof.

At that recorded revision, cold work was material: construction took 55.6/55.4 s and the first four-iteration online fit took 117.0/115.5 s with compilation. BRM traced the latter mainly to the first batched Enzyme population-gradient compilation. These cold costs are excluded from the warm table, so a one-off 100-iteration fit was not a 0.12 s whole workflow. The next section measures the subsequent subject-cell gradient route.

To reproduce the default-path comparison after the dependency setup above, switch its BRM checkout to the final reviewed revision and run the new driver:

```sh
git -C "$BENCH_DEPS/BayesianRegressionModels" fetch origin 1e7fbaa0d3e6f462e5ae4b2cc74056a7542f4c08
git -C "$BENCH_DEPS/BayesianRegressionModels" checkout --detach 1e7fbaa0d3e6f462e5ae4b2cc74056a7542f4c08
julia --project=docs/examples bench/saem_streaming_brm_mu.jl
```

## Subject-cell gradients and first-use compilation

The preceding table records BRM `1e7fbaa` and remains a useful baseline. With
BRM [`627dbdd`](https://github.com/nsiccha/BayesianRegressionModels.jl/commit/627dbddbadc0d806f95b0460a86ac98b627dd977), the online residual-parameter
M-step obtains the conditional population gradient by summing compact
subject-cell evaluations. The retained-draw `:mixture` M-step still uses the
batched gradient. NLMEEstimation [`50f8534`](https://github.com/nsiccha/NLMEEstimation.jl/commit/50f8534b70b2e3feb5b2b7a9b5434111a00be260) selects a mode-specific
M-step callable once per fit, so Julia compiles the selected gradient path
when it is first used. These changes leave the MCMC sweep, running Gaussian
sufficient statistics, and the online residual update schedule intact.

The model and the fitted call in the benchmark have this form. The complete
synthetic [data generator and model source](https://github.com/nsiccha/NLMEEstimation.jl/blob/master/docs/examples/brm_rkppl_pk_fixture.jl)
and [benchmark driver](https://github.com/nsiccha/NLMEEstimation.jl/blob/master/bench/saem_streaming_brm_mu.jl)
provide the executable version.

```julia
using NLMEEstimation, BayesianRegressionModels, ReactiveKernelsPPL
using Distributions, Enzyme, Random
using DifferentiationInterface: AutoEnzyme
include("docs/examples/brm_rkppl_pk_fixture.jl")

data = pk_example_data(80; seed=20261011)
brmi = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | pk | subject)
    log_V ~ 1 + (1 | pk | subject)
    effect(log_CL, Intercept) ~ Normal(0.2, 1)
    effect(log_CL, weight) ~ Normal(0, 0.5)
    effect(log_V, Intercept) ~ Normal(1.6, 1)
    sd(:, pk) ~ Exponential(1)
    @plate for i in eachindex(log_CL)
        CL = exp(log_CL[i])
        V = exp(log_V[i])
        pred[i] = dose[i] / V .* exp.(-(CL / V) .* t[i])
        dv[i] ~ normal(pred[i], sigma .+ 0.03 .* pred[i])
    end
end
model = brm_nlme_model(RKBRMI(brmi);
    ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))

@assert saem_population_gradient_style(model, Val(:online)) isa PerSubjectEvaluation
@assert saem_population_gradient_style(model, Val(:mixture)) isa BatchedEvaluation

θ0 = [Dict("log_V_Intercept" => 1.6,
    "pop_log_CL.beta_pop.1" => 0.2,
    "pop_log_CL.beta_pop.2" => 0.25)[name]
    for name in parameter_names(model).theta]
σ0 = [log(0.6)]
ω0 = omega_parameters(nlme_layout(model), [0.3^2 0.0; 0.0 0.15^2])
options = SAEMOptions(; iterations=100, burnin=50, mcmc_burnin=15,
    mh_rounds=1, adapt_iterations=50, numeric_maxiter=12,
    mstep_interval=1, numeric_mstep=:auto)
fit = saem(model, θ0, σ0, ω0; rng=Xoshiro(5), options,
    kernel=StandardSAEMMH())
@assert fit.numeric_mstep == :online
```

The first run used BRM `b2101d2`; the second used `627dbdd`, which additionally
declares the online style explicitly. Both exact-tip runs used this 80-subject,
five-observation IV-bolus fixture,
data seed `Xoshiro(20261011)`, fit seed `Xoshiro(5)`, Julia 1.10.12, Enzyme
0.13.210, one Julia and one BLAS thread, and the shared AMD Ryzen 9 9950X
Linux host. ReactiveKernels and RKPPL were at `a2268e47`; WarmupHMC was at
`6535499`. One MCMC sweep and one M-step ran per SA iteration. Before timing
each fit, the driver called `GC.gc()`. The first explicit four-iteration fit
of each mode includes its compilation. The later 100/200-iteration fits are
warm. Fit timings include the 15 initial MH sweeps but exclude model
construction, protocol checking, and printing. Allocations below are
cumulative bytes allocated per fit, not retained memory. The
[complete output for both runs](https://github.com/nsiccha/NLMEEstimation.jl/blob/master/bench/saem_streaming_brm_cell_results.txt)
contains every estimate and acceptance measurement.

| Measurement | BRM `b2101d2` | BRM `627dbdd` |
| --- | ---: | ---: |
| Model construction | 54.904 s, 7.529 GB | 55.081 s, 7.531 GB |
| First explicit online fit, 4 SA iterations | 4.359 s, 0.627 GB | 4.343 s, 0.627 GB |
| First explicit mixture fit, 4 SA iterations | 114.256 s, 0.890 GB | 115.603 s, 0.890 GB |
| Warm default online, 100 iterations | 0.568 s, 5.912 GB | 0.629 s, 5.912 GB |
| Warm default online, 200 iterations | 1.117 s, 11.594 GB | 1.236 s, 11.594 GB |
| Warm mixture, 100 iterations | 0.880 s, 3.910 GB | 0.946 s, 3.910 GB |
| Warm mixture, 200 iterations | 3.397 s, 14.213 GB | 3.573 s, 14.213 GB |

The old BRM `1e7fbaa` first online fit took 115.5 s, versus 4.34 s at the
final subject-cell tip. Its warm online 100/200-iteration fits were faster,
0.121/0.219 s with 0.848/1.616 GB allocated; the new cell route takes
0.629/1.236 s and allocates 5.912/11.594 GB. The new route therefore favors
the first fit, while the earlier batched route had lower steady online cost
after its compilation. The mixture path's first compilation is now paid when
mixture is selected; its warm time and allocation stay close to the older
batched measurements. In both runs, seeded fits returned finite estimates;
the online population coordinates agreed with the older route to the shown
precision. These timings compare implementation costs, not statistical
convergence or accuracy across different models.

## Ragged subjects with thousands of nested effects

`bench/saem_ragged_kernels.jl` fits the test suite's nested-occasions model
(`occasions_model` in `test/common.jl`) on a ragged layout with six subjects of
250, 500, 1,000, 2,000, 4,000 and 8,000 nested units. Each subject has one
subject effect (SD 0.6) and one effect per unit (SD 0.4), and each unit two to
four observations with error SD 0.25: 15,750 unit effects in all. The model's
marginal likelihood and every unit effect's posterior have closed forms, so
each fit is compared with the exact maximum likelihood and each subject's unit
chains with their exact posterior there. All fits start at the generating
values with every η at zero and use `SAEMOptions(iterations = 300, burnin =
150)` and each kernel's defaults. Julia 1.10.12, one thread, NLMEEstimation
`9277cb0`; raw output in `bench/saem_ragged_kernels_results.txt`.

| Kernel | Log-likelihood gap to the maximum | Unit SD (maximum likelihood 0.40) | Unit chains' correlation with their posterior means | Seconds |
| --- | ---: | ---: | ---: | ---: |
| `StandardSAEMMH()` | 10,748–12,651 | 0.06–0.22 | 0.01–0.28 | 29–32 |
| `DualAveragingMALA()` | 12,859–13,413 | 0.016–0.017 | 0.03–0.30 | 29–33 |
| `WarmupHMCNUTS()` | 0.00–0.01 | 0.3990–0.4014, the maximum to four digits | 0.91–0.95 | 63–72 |

Ranges are over three seeds. A chain that samples its subject's posterior
correlates about 0.93 with the posterior means here. The MH and MALA chains
stay close to their start: the γ = 1 M-steps then estimate the unit
covariance from them and shrink it, and MALA's proposals, which follow that
covariance, shrink with it. The test suite checks the NUTS case at 150–600
units per subject (`test/saem_ragged_kernels.jl`).

Reproduce with:

```
julia --project=test bench/saem_ragged_kernels.jl 3
```
