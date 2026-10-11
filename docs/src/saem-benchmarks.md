# SAEM subject-kernel benchmarks

This page compares tuned **per-subject** Metropolis–Hastings (MH) and
WarmupHMC NUTS kernels in `NLMEEstimation.jl`. Each of six subjects has a
three-dimensional random effect `ηᵢ`. Neither kernel proposes an
18-dimensional state for all subjects: MH evaluates six independent proposals
in a batch, and NUTS advances six separate chains.

The final comparison uses only configurations selected by the procedure below.
“Best” means the best observed median effective sample size (ESS) per elapsed
second in the tested grid, subject to zero observed NUTS divergences in both
fixed-target and changing-target qualification runs. It is not a global
optimum or an estimator-accuracy claim.

## Model, target, and environment

The [Theoph fixture](https://github.com/nsiccha/NLMEEstimation.jl/blob/767effa54dfeb45bf636a5182d51f4648d3d3066/test/common.jl)
uses the public theophylline concentration data, an oral one-compartment
model, additive observation error, and Enzyme derivatives. The first six
subjects are sampled at fixed starting population parameters
`θ = [log(1.57), log(2.75), log(31.5), 0]`,
`σ = [log(0.7)]`, and
`Ω = diag(0.6², 0.26², 0.14²)`. Each subject's MCMC target is

```math
p(\eta_i \mid y_i, \theta, \sigma, \Omega) \propto
p(y_i \mid \theta, \sigma, \eta_i)\,\mathcal N(\eta_i; 0, \Omega).
```

Measurements ran on a shared Linux x86-64 host with an AMD Ryzen 9 9950X
(16 cores, 32 threads), Julia 1.10.12, one BLAS thread, Enzyme 0.13.215,
[NLMEEstimation benchmark revision `767effa`](https://github.com/nsiccha/NLMEEstimation.jl/commit/767effa54dfeb45bf636a5182d51f4648d3d3066),
WarmupHMC `6cd665de7591dc3024893106f64e72f9e90ad49c`
(version 0.2.1), and Treebars
`7426b9c8dc914d767c28ff22f1f75631f3afc41d`.
The host is shared, so short wall-clock timings can vary with other work.

## What the two kernels do

The real [MH transition](https://github.com/nsiccha/NLMEEstimation.jl/blob/767effa54dfeb45bf636a5182d51f4648d3d3066/src/saem.jl)
first proposes from each subject's Gaussian prior. The prior proposal density
cancels from its acceptance ratio. It then proposes a covariance-scaled
random walk and uses the complete-data log density. The package makes each
likelihood query for all requested subjects in one batch; acceptance remains
subject by subject. These are the proposal and acceptance lines from the
transition; the full source also validates model outputs:

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

The real [NUTS target and transition](https://github.com/nsiccha/NLMEEstimation.jl/blob/767effa54dfeb45bf636a5182d51f4648d3d3066/src/saem.jl)
combine the model's conditional η gradient with the analytic Gaussian-prior
gradient. Each subject warms once from an initial covariance obtained by
inverting its local negative log-posterior Hessian. The selected setting uses
the full three-by-three covariance; the default public kernel uses its
diagonal. A checkpoint-seeded `stream_mcmc` call resumes the subject under
the *current* SAEM target:

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
    init=(position=η, squared_scale=Σ),
    checkpoint_dir=warm_dir,
    n_draws=kernel.warmup_draws,
    n_evaluations=kernel.warmup_evaluations,
    recording_target=kernel.warmup_draws,
    stepsize_adaptation_limit=kernel.stepsize_adaptation_limit,
    max_tree_depth=kernel.max_tree_depth,
    target_acceptance_rate=kernel.target_acceptance_rate,
    nonlinear_adapt=false, monitor_ess=false)
push!(checkpoints, joinpath(warm_dir, "cp_latest.jls"))
push!(streams, joinpath(root, "subject_$(k).stream"))

# At each later SAEM round, construct target from current θ, σ, and Ω.
target = _saem_target(model, i, θ, σ, prior)
result = WarmupHMC.stream_mcmc(state.checkpoints[k], target;
    path=state.streams[k], n_draws=state.drawn[k] + 1)
state.drawn[k] = result.n_drawn
H[:, k] = result.position
divergences += result.n_divergent
```

The public interaction for the selected configurations is:

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
    mh_rounds=1, numeric_maxiter=2, mstep_interval=5)
mh_options = SAEMOptions(; base..., rw_scale=0.45, adapt_iterations=0)
nuts_options = SAEMOptions(; base..., adapt_iterations=0)
mh_fit = saem(model, θ_THEOPH, σ_THEOPH, ω;
    rng=MersenneTwister(2090), subjects=collect(1:6),
    options=mh_options, kernel=mh)
nuts_fit = saem(model, θ_THEOPH, σ_THEOPH, ω;
    rng=MersenneTwister(2090), subjects=collect(1:6),
    options=nuts_options, kernel=nuts)
```

The fixed-target scripts use the same kernel operations but collect every
subject draw for ESS. The SAEM fit consumes one transition per subject per
outer iteration; each NUTS transition reopens its stream file.

## Tuning and measurement

All fixed-target candidates retain **1,600 draws per subject**. MH first
runs 200 burn-in sweeps and tries six fixed proposal scales
(0.25, 0.35, 0.45, 0.6, 0.8, 1.2), plus adaptation to acceptance targets
0.2, 0.3, or 0.4. The broad NUTS grid tries three initial metrics
(diagonal Hessian, dense Hessian, prior covariance), warmup draw floors
8 or 20, initial gradient-evaluation budgets 50 or 100, and maximum tree
depths 4 or 8, with acceptance target 0.8. Twelve follow-up runs vary the
acceptance target over 0.6–0.9 for the top fixed-target NUTS settings.
The [complete 45-row sweep](https://github.com/nsiccha/NLMEEstimation.jl/blob/767effa54dfeb45bf636a5182d51f4648d3d3066/bench/saem_kernel_sweep.tsv)
records every setting and measured score.

The broad sweep uses seeds 2026–2030; seeds 2050–2054 check its provisional
choices. An aggressive 0.7 NUTS acceptance target initially scored well,
but yielded divergences on later seeds and was not selected. Four MH
finalists and six NUTS finalists were then measured on seeds 2060–2069.
NUTS finalists also ran a 40-iteration, changing-target SAEM fit on seeds
2080–2084. A NUTS finalist qualified only with **zero fixed-target and
changing-target divergences** and finite fit estimates. The highest-scoring
qualified NUTS finalist and highest-scoring MH finalist were compared on
untouched seeds **2090–2099**.

ESS is the initial-positive-pair estimate in
[`bench/saem_kernels.jl`](https://github.com/nsiccha/NLMEEstimation.jl/blob/767effa54dfeb45bf636a5182d51f4648d3d3066/bench/saem_kernels.jl).
Each seed's ESS is the median over the 18 subject-by-coordinate traces.
The reported ESS and wall time are medians over seeds; ESS/second is the
median of each seed's ESS divided by its sampling wall time. ESS is capped
at the 1,600 retained draws by this estimator. Wall time includes MH burn-in
or NUTS warmup and all retained draws, but excludes Julia compilation,
model construction, and ESS calculation. NUTS takes the 1,600 retained
draws in one `stream_mcmc` call per subject for this fixed-target test.

### Divergence qualification

The NUTS fixed-target scores below use ten selection seeds; fit divergences
sum over five changing-target SAEM seeds. All fixed-target runs in this
table had zero divergences. The selected safe configuration is bold.

| Initial metric | Draw floor | Initial gradient evaluations | Acceptance target | Fixed-target ESS/s | Changing-target divergences |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense Hessian | 8 | 50 | 0.8 | 13,556 | 14 |
| Dense Hessian | 8 | 100 | 0.8 | 13,553 | 55 |
| **Dense Hessian** | **20** | **50** | **0.8** | **11,213** | **0** |
| Dense Hessian | 20 | 100 | 0.8 | 10,852 | 0 |
| Dense Hessian | 8 | 50 | 0.9 | 7,417 | 1 |
| Diagonal Hessian | 20 | 100 | 0.8 | 6,961 | 0 |

MH's four stability finalists scored 11,405, **12,959**, 12,259, and
12,857 ESS/s respectively for fixed scale 0.35, **fixed scale 0.45**,
adaptation target 0.2, and adaptation target 0.3. The fixed 0.45 and
adaptive 0.3 settings are close at this resolution; fixed 0.45 won the
declared selection rule.

### Fresh-seed comparison of the selected kernels

Both rows use seeds 2090–2099 and 1,600 retained fixed-target draws per
subject. “Fit seconds” measures a separate 40-iteration SAEM run with
five MCMC burn-in sweeps, one kernel round per iteration, and the same
M-step settings. All ten fits returned finite estimates.

| Tuned kernel | Fixed-target seconds | Median ESS per coordinate | Median ESS/s | Random-walk acceptance | Fixed-target divergences | Fit seconds | Maximum fit divergences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MH, fixed scale 0.45 | 0.010342 | 126.3 | **12,225** | 26.9% | — | **0.003332** | — |
| NUTS, dense Hessian, 20/50 warmup, target 0.8 | 0.114005 | **1,282.9** | 11,014 | — | 0 | 0.718908 | 0 |

NUTS produced about 10 times the ESS per draw, while MH's observed
ESS per second was about 11% higher in the fixed-target experiment.
The short SAEM fit was about 216 times faster with MH. NUTS does more work
per transition, and this implementation reopens a stream file for every
subject at every outer iteration.
The fit is a cost and integration check, not an accuracy comparison:
40 SAEM iterations do not establish parameter recovery.

These ESS values apply **only to stationary, fixed-population targets**.
SAEM changes its target across outer iterations, so pooling those
positions into one chain and reporting an ESS would be invalid. Also,
the fixed-target NUTS script collects draws in one stream call, whereas
the actual SAEM fit pays the repeated one-transition resume cost. The
comparison is specific to this six-subject, three-effect Theoph model;
different dimensions, posterior geometry, models, and longer fits may
favor different settings. Some WarmupHMC candidate runs logged a TSVD
failure with a successful power-iteration fallback during warmup; the
reported divergence counts come from sampler results, not those logs.

## Reproduce

The source is
[`bench/saem_kernel_tuning.jl`](https://github.com/nsiccha/NLMEEstimation.jl/blob/d733d33b9f34e094efa2e669ada4e0c2baa2d7f9/bench/saem_kernel_tuning.jl)
and
[`bench/saem_kernel_final.jl`](https://github.com/nsiccha/NLMEEstimation.jl/blob/767effa54dfeb45bf636a5182d51f4648d3d3066/bench/saem_kernel_final.jl).
From a clone of NLMEEstimation.jl, develop the package and the two
unregistered dependencies in one resolver call, then run the scripts:

```sh
git clone --branch dev https://github.com/nsiccha/WarmupHMC.jl.git ../WarmupHMC.jl
git clone --branch main https://github.com/nsiccha/Treebars.jl.git ../Treebars.jl
julia --project=test -e 'using Pkg; Pkg.develop([PackageSpec(path=pwd()), PackageSpec(path="../WarmupHMC.jl"), PackageSpec(path="../Treebars.jl")]); Pkg.instantiate()'
julia --project=test bench/saem_kernel_tuning.jl
julia --project=test bench/saem_kernel_final.jl
```

For an exact replay, check out the package and dependency SHAs listed
above. The scripts print individual candidate scores, the selected
settings, final fixed-target results, fit timing, and divergences.

The [initial 800-draw exploratory run](https://github.com/nsiccha/NLMEEstimation.jl/blob/800ad51cccc43ebce74e854c33fa81c1a904a823/bench/RESULTS.md)
is retained as historical context. Its untuned settings and three-seed
sample do not replace the tuned-kernel comparison above.
