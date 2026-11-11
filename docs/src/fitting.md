# Estimation

Choose an estimator according to the quantity you need and the information your model supplies. Subject modes, population point estimates and marginal-likelihood calculations answer different questions.

## Estimator choice

| Task | Entry point | Result |
|:--|:--|:--|
| Subject effects at fixed population parameters | [`empirical_bayes`](@ref), alias [`posthoc`](@ref) | Modes and per-subject convergence information |
| Population estimation with Gaussian observation moments | [`focei`](@ref) | FOCE(I) approximation and population point estimate |
| Population estimation through stochastic approximation | [`saem`](@ref) | Population point estimate and kernel diagnostics |
| Likelihood at a fitted or specified population point | [`marginal_loglikelihood`](@ref) | Likelihood, OFV and information criteria |

A SAEM chain samples subject effects as part of optimization. It does not sample a posterior over population parameters. A declared population prior changes the population target from maximum likelihood to a MAP objective. The likelihood remains available separately in result fields.

## Subject modes

`empirical_bayes` combines the conditional likelihood with the package's random-effect prior. `NewtonEBE` uses Hessian blocks. `QuasiNewtonEBE` uses limited-memory BFGS and checks curvature before accepting stationary points. Where a model declares no Hessian or sparsity pattern, its curvature check uses gradient differences and may be limited in large effect spaces.

```@example fitting
using NLMEEstimation
method = QuasiNewtonEBE(memory=10)
options = EBEOptions(gtol=1e-7)
println((method=typeof(method), history_length=method.memory,
    standardized_gradient_tolerance=options.gtol))
```

The stopping gradient is measured in prior-standard-deviation coordinates. Inspect `converged`, `gradient_norm` and `nonsmooth` by subject. A nonsmooth mode can be a valid mode while operations requiring local curvature remain undefined. [`NonsmoothModeError`](@ref) distinguishes that situation from an ordinary failure to converge.

## Population search

FOCE minimizes its approximate OFV with a derivative-free outer search. Its `free` argument selects positions in the search vector; with an unconstrained covariance layout that vector is `[θ; σ; ω]`. Coordinates omitted from `free` stay at their starting values. A `fixed_omega` model declaration changes which covariance coordinates can be searched; consult [`FOCEResult`](@ref) before indexing a constrained fit.

```@example fitting
options = FOCEOptions(maxeval=800, interaction=true)
println((maximum_objective_calls=options.maxeval,
    interaction=options.interaction, terminal_radius=options.rhoend))
```

FOCE's approximation uses first derivatives of Gaussian observation means and variances. The inner mode search can also inspect Hessians supplied by the model. Turning interaction off removes the variance-Jacobian contribution; it does not switch to an exact marginal likelihood.

SAEM alternates transitions for subject effects with population updates. [`SAEMOptions`](@ref) controls stochastic-approximation iterations, burn-in, gain decay and numerical M-step behavior. Its `omega_floor` also applies to the start: an estimated part of `ω0` below the floor is raised to it before the first sweep, with a warning naming those families, while a start that holds the floor is used unchanged. [`MuReferencing`](@ref) permits closed-form updates for compatible population coordinates. Otherwise, the selected numerical update determines the required population derivatives.

| SAEM kernel | Local information | Diagnostic fields |
|:--|:--|:--|
| [`StandardSAEMMH`](@ref) | Likelihood values | Proposal acceptance |
| [`DualAveragingMALA`](@ref) | Effect gradients | Adapted step sizes and acceptance |
| [`WarmupHMCNUTS`](@ref) | Effect gradients | Divergences and kernel diagnostics |

`WarmupHMCNUTS` initializes with Pathfinder and adapts continuously as population parameters move. Its current constructor accepts `target_acceptance_rate`, `max_tree_depth` and `threads`. Threaded subject evaluation requires a model whose evaluation is thread-safe; `check_protocol(...; threads=true)` exercises that property.

## Stopping information

A terminated optimizer is not evidence that the model is identifiable. Check the optimizer status, subject convergence, covariance boundaries and sensitivity to starting values. Compare likelihood values only when their approximations and fitted targets are compatible. A MAP objective contains a prior term; calling its OFV a pure likelihood criterion would change its meaning.

Every estimator accepts `progress=nothing`. A Treebars backend or node enables progress reporting. Interrupting FOCE or SAEM returns the estimate reached so far, with its corresponding interruption status; other calculations raise `Treebars.ProgressInterrupted`.

The [oral PK example](oral-pk.md) shows a complete FOCE fit, its stopping information and observation-level diagnostics.
