# PPL and AD requirements

This page lists what a probabilistic programming language (PPL) and its automatic-differentiation (AD) backend must provide for each estimator. **Every implemented method runs on first-order derivatives.** Second derivatives are optional: when available, they are used only as per-subject `K × K` η Hessian blocks, where `K` is the number of random effects per subject. No estimator uses a joint Hessian, third derivatives, or finite differences. The covariance step takes finite differences of objective values and adds no derivative requirement.

## What the PPL must expose

A model gives NLMEEstimation the conditional log-likelihood `ℓᵢ(θ, σ, ηᵢ) = log p(yᵢ | θ, σ, ηᵢ)` for each subject `i`. NLMEEstimation supplies the random-effect prior `ηᵢ ~ N(0, Ω)` itself, so the PPL program must satisfy these conditions:

- **η is an input.** The program accepts each subject's `ηᵢ` on the scale of `N(0, Ω)` and returns the likelihood *without* a prior on η. A PPL that samples its own scaled or non-centered random effects must expose a mode in which those effects are fixed inputs. BRM does this by setting its own random-effect scales to 1 and its correlations to the identity.
- **Subjects are conditionally independent.** Given `(θ, σ)`, the log-likelihood is a sum of subject terms, each depending only on its own `ηᵢ`. The batched derivatives below rely on this property, and `check_protocol` tests it. A population-level prior goes in `population_logprior`, which is evaluated once per fit.
- **Population coordinates are unconstrained.** `θ` and `σ` are the model's own unconstrained coordinates. The package represents Ω by blockwise log-Cholesky coordinates `ω`.
- **Either evaluation style works.** A PPL that can evaluate one subject cheaply uses `PerSubjectEvaluation`. A PPL whose program always computes every subject in one pass uses `BatchedEvaluation`: it evaluates a list of subjects, one `η` column each, and subjects that are not listed do not affect the result.

Some estimators need more structure:

| Structure | Needed by | Without it |
| --- | --- | --- |
| Observed values and Gaussian observation means and variances, `GaussianObservations()` | FOCE and FOCEI | FOCE(I) is unavailable. Posthoc and SAEM work with any likelihood. |
| Mu-referencing, `ϕᵢ = Xᵢ θ_μ + ηᵢ` | SAEM's closed-form update of those θ | SAEM fits those θ numerically together with σ. |
| `population_logprior` | MAP instead of ML in FOCE(I) and SAEM | Maximum likelihood. |

## Derivatives by method

“Values” means conditional log-likelihoods, “∇η” means gradients in each subject's η, and “J” means the Jacobian of the observation means and variances in η. “∇θσ” means the gradient of the summed log-likelihood in the population coordinates `(θ, σ)` at fixed η. “Hessian blocks” means each subject's `K × K` η Hessian.

| Method | Values | ∇η | J | ∇θσ | Hessian blocks |
| --- | --- | --- | --- | --- | --- |
| `empirical_bayes` with `QuasiNewtonEBE()` | ✓ | ✓ | — | — | Optional: stored in the result for `ebe_covariances`, and used for its curvature check |
| `empirical_bayes` with `NewtonEBE()` | ✓ | ✓ | — | — | Required |
| `focei` (FOCE and FOCEI) | ✓ | ✓ | ✓ | — | Never used |
| `saem` with `StandardSAEMMH()` | ✓ | — | — | ✓ | Never used |
| `saem` with `DualAveragingMALA()` | ✓ | ✓ | — | ✓ | Optional: initial metric with `metric=:hessian` |
| `saem` with `WarmupHMCNUTS()` | ✓ | ✓ | — | ✓ | Optional: initial NUTS metric |
| `covariance_step` with `FOCEApproximation()` | ✓ | ✓ | ✓ | — | Optional: Newton subject modes |
| `marginal_loglikelihood` / `covariance_step` with `ImportanceSampling()` or `AdaptiveGaussHermite()` | ✓ | ✓ | — | — | Optional: proposal curvature |
| `residual_diagnostics` (PRED, IPRED, IWRES, CWRES, NPDE) | ✓ | ✓ | ✓ | — | Optional: Newton subject modes |

All of these are first-order quantities, except the Hessian blocks:

- `empirical_bayes` picks `NewtonEBE()` by default only when the model provides Hessian blocks (`default_ebe_method`); otherwise it uses `QuasiNewtonEBE()`, a lockstep BFGS method. `ebe_covariances` returns the per-subject Laplace covariance `(-∇²f)⁻¹` and therefore needs the Hessian blocks.
- `QuasiNewtonEBE()` checks each subject's curvature once no subject is still iterating, and wherever BFGS meets upward curvature. It does not report a saddle point as a mode; it leaves it with a trust-region step. The check uses the Hessian blocks when the model provides them, which costs no extra calls because the result reports those blocks. Otherwise it uses forward differences of batched ∇η, at `K` extra batched gradient calls per check.
- `focei` always computes its subject modes with `QuasiNewtonEBE()`. Its outer search over `(θ, σ, ω)` is NLopt's derivative-free BOBYQA, so FOCE(I) takes no population derivatives.
- SAEM's MH transitions use only values. MALA and NUTS need ∇η. A Hessian only seeds their initial metric. Without one, they use Ω.
- Without a population prior, SAEM updates the mu-referenced θ and Ω in closed form from sufficient statistics, without derivatives. The remaining θ and every σ coordinate use a numerical M-step that needs ∇θσ. In practice that means almost every model, because residual-error parameters live in σ.
- `covariance_step` evaluates the per-subject FOCE(I) marginal log-likelihoods at central-difference points in `(θ, σ, ω)`. Each evaluation needs what `focei` needs. With Hessian blocks it finds the subject modes by Newton, which converges to roundoff; otherwise it uses BFGS with a tighter tolerance. It takes no population derivatives ([source](https://github.com/nsiccha/NLMEEstimation.jl/blob/abea84d04bf65c2b90859d6b975032f29707c69c/src/covariance.jl#L213-L231)).
- `marginal_loglikelihood` with importance sampling or adaptive Gauss–Hermite quadrature needs conditional values at its points and ∇η for each subject's mode. The curvature at the mode comes from the Hessian blocks when provided, and otherwise from central differences of batched ∇η. These approximations need no Gaussian observation model and support ragged layouts.
- With a population prior, SAEM keeps the same two blocks. The mu-referenced θ and Ω move from their closed-form update to the maximizer with the prior, which needs only the prior's gradient; the remaining θ and σ take the numerical step with the prior added, which needs ∇θσ as before. A prior therefore adds no model evaluations to the M-step. `population_logprior_and_gradient` derives the prior's gradient with `ad_backend(model)` by default.

Ragged nested random effects (`RaggedNLMELayout`) have the same requirements. FOCE(I) does not support them yet, and SAEM supports only the MH kernel.

## What the AD backend must do

A model can implement any derivative method directly. Missing methods are derived through [DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl) with `ad_backend(model)`; Hessian blocks use `hessian_ad_backend(model)`, which defaults to the same backend. Reverse-mode Enzyme, `AutoEnzyme()`, is the tested configuration. Other ADTypes backends are passed through untested.

**Required: gradients of a scalar with constant context.** Reverse mode is the efficient and tested choice. For a batched model, every subject's η gradient comes from one reverse pass over the summed batch log-likelihood ([objective](https://github.com/nsiccha/NLMEEstimation.jl/blob/a3d9bfbfe34a5f08e1ba534ff7f4d2c2c32b835b/src/protocol.jl#L380), [gradient](https://github.com/nsiccha/NLMEEstimation.jl/blob/a3d9bfbfe34a5f08e1ba534ff7f4d2c2c32b835b/src/protocol.jl#L306-L311)):

```julia
_batch_objective(H, model, subjects, θ, σ) = sum(conditional_loglikelihoods(model, subjects, θ, σ, H))

ℓ = conditional_loglikelihoods(model, subjects, θ, σ, H)
G = DI.gradient(_batch_objective, backend, H, _batch_contexts(model, subjects, θ, σ)...)
```

Conditional independence makes column `k` of `G` subject `k`'s gradient. For a batched model, SAEM's ∇θσ is a reverse pass of the same summed objective, taken with respect to `vcat(θ, σ)` at fixed `H`. A per-subject model computes either gradient with one `DI.value_and_gradient` call per subject. In both cases the model data, subject indices, and the coordinates not being differentiated are `DI.Constant` contexts. The AD system must therefore differentiate the PPL program with respect to one input while treating the rest as constant.

**Required for FOCE(I): first-order Jacobians of a vector output.** FOCE(I) needs `Jμ` and `Jv`, each `nᵢ × K`, for every subject's observation means and variances. For a per-subject model, the default derives them from `observation_moments` with one reverse pullback per row of `Jμ` and of `Jv`, `2nᵢ` pullbacks per subject. A batched model implements `observation_moments_and_jacobians` itself. Forward mode suits it well: a tangent that sets η component `c` to 1 for every subject yields column `c` of every subject's Jacobians in one pushforward, so `K` pushforwards cover the batch. BRM's RKPPL adapter does this.

**Optional: Hessian-vector products.** A batched model's Hessian blocks come from `K` Hessian-vector products of the summed batch log-likelihood, normally forward-over-reverse. The same coloring argument applies: tangent `c` sets η component `c` of every subject to 1, and its product holds column `c` of every subject's block ([source](https://github.com/nsiccha/NLMEEstimation.jl/blob/a3d9bfbfe34a5f08e1ba534ff7f4d2c2c32b835b/src/protocol.jl#L337-L349)):

```julia
tangents = ntuple(m) do c
    V = zero(H)
    V[c, :] .= 1
    V
end
hvps = DI.hvp(_batch_objective, backend, H, tangents, _batch_contexts(model, subjects, θ, σ)...)
return [reduce(hcat, [hvps[c][:, k] for c in 1:m]) for k in eachindex(subjects)]
```

The `(N K) × (N K)` joint Hessian is never formed. A gradient-only model declares `hessian_ad_backend(model) = nothing`; every estimator then takes its gradient-only path.

A PPL and AD system that provide only first-order reverse mode, such as a PPL whose ODE solver supports only reverse-mode first derivatives, can therefore run posthoc, FOCE, FOCEI, and SAEM with MH, MALA, or NUTS. Vector Jacobians for FOCE(I) can come from reverse-mode pullbacks.

## Why FOCE(I) needs only first derivatives

For subject `i`, let `fᵢ(η) = ℓᵢ(θ, σ, η) + log N(η; 0, Ω)` be the complete-data log density, and let `η̂ᵢ` be its mode. FOCE(I) approximates the marginal log-likelihood as

```math
\log p(y \mid \theta, \sigma, \Omega) \approx
\sum_i \left[ f_i(\hat\eta_i) + \frac{K}{2}\log 2\pi - \frac12 \log\det C_i \right],
\qquad
C_i = \Omega^{-1} + J_{\mu,i}^\top V_i^{-1} J_{\mu,i}
      + \tfrac12 J_{v,i}^\top V_i^{-2} J_{v,i},
```

where `Vᵢ` is the diagonal matrix of predicted variances. The last term is the FOCEI interaction term; FOCE drops it. `Cᵢ` is the Fisher information of the Gaussian observations plus the prior precision. It uses only first derivatives of the predicted means and variances; this is what distinguishes FOCE(I) from the exact Laplace approximation. The implementation is direct ([source](https://github.com/nsiccha/NLMEEstimation.jl/blob/abea84d04bf65c2b90859d6b975032f29707c69c/src/focei.jl#L114-L119)):

```julia
C = prior.precision + Jμ' * (Jμ ./ v)
interaction && (C += Jv' * (Jv ./ (2 .* v.^2)))
factor = cholesky(Symmetric(C))
push!(curvatures, Matrix{T}(C))
push!(terms, ebes.complete_data_loglikelihood[k] + m * log(2π) / 2 -
    sum(log, diag(factor.L)))
```

## Where higher orders would come from

Higher derivatives appear only for an objective or optimizer that this package does not implement:

| Variant | Objective | Exact gradient in `(θ, σ, Ω)` |
| --- | --- | --- |
| FOCE(I), as implemented, with derivative-free BOBYQA | First order | Not needed |
| FOCE(I) with a gradient-based outer optimizer | First order | Second order: derivatives of `Jμ` and `Jv`, and `dη̂ᵢ/dθ`, which needs second derivatives of `fᵢ` |
| Exact Laplace with the true Hessian `Hᵢ = ∇²fᵢ(η̂ᵢ)` | Second order | Third order |

Here θ stands for any population coordinate. The third-order case comes from the log-determinant. At the mode, `∇η fᵢ = 0`, so the derivative of `fᵢ(η̂ᵢ)` in θ needs only first derivatives. The derivative of the log-determinant term, however, is

```math
\frac{d}{d\theta}\log\det(-H_i)
= \operatorname{tr}\!\left[H_i^{-1}\left(\partial_\theta H_i
  + \partial_\eta H_i\,\frac{d\hat\eta_i}{d\theta}\right)\right],
\qquad
\frac{d\hat\eta_i}{d\theta} = -H_i^{-1}\,\partial_\theta \nabla_\eta f_i ,
```

and `∂θ Hᵢ` and `∂η Hᵢ` are third derivatives of `fᵢ`. Third-order AD was mentioned in early planning for this reason. Both the gradient-based outer search and exact Laplace remain optional future work.

The covariance step needs the Hessian of the objective in `(θ, σ, Ω)`, one order above the gradient column: third order for FOCE(I). `covariance_step` therefore takes second differences of objective values instead.

## Checklist for a new PPL integration

1. Implement `nsubjects`, `nlme_layout`, and either `conditional_loglikelihood` or `conditional_loglikelihoods`, with η as an input and no η prior.
2. Provide ∇η: declare a reverse-mode `ad_backend`, or implement `conditional_loglikelihoods_and_gradients`. This enables posthoc and the gradient-based SAEM kernels.
3. Provide ∇θσ, either from the same backend or with `conditional_loglikelihood_and_population_gradient`. This enables SAEM's numerical M-step.
4. For FOCE(I), declare `GaussianObservations()` and provide `observations`, `observation_moments`, and their η Jacobians.
5. Optionally declare `mu_referencing` for closed-form SAEM updates, and Hessian blocks for Newton posthoc steps, `ebe_covariances`, and curvature-based MALA/NUTS metrics.
6. Run `check_protocol(model, θ, σ, ω)`. It checks the evaluation styles against each other, subject separability, the derivatives the model provides, and the declared traits.

See the [model protocol](protocol.md) for the full interface and the [PK comparison](pk-comparison.md) for a complete BRM+RKPPL model.
