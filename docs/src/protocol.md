# Model protocol

Implement a Julia likelihood that NLMEEstimation can evaluate by subject, differentiate and check before estimation. The interface separates the observation model from the Gaussian random-effect prior. A model may be an ordinary struct; it does not need to inherit from a package type.

## Conditional likelihood

This model has a population mean, a log residual standard deviation and one subject-specific offset. Its likelihood contains the observations only:

```@example protocol
using NLMEEstimation
struct LocationModel
    y::Vector{Vector{Float64}}
end
NLMEEstimation.nsubjects(m::LocationModel) = length(m.y)
NLMEEstimation.nlme_layout(::LocationModel) = NLMELayout(1, 1, [1])
function NLMEEstimation.conditional_loglikelihood(m::LocationModel, i, θ, σ, η)
    sum(y -> -(log(2π) + 2σ[1] + (y - θ[1] - η[1])^2 * exp(-2σ[1])) / 2, m.y[i])
end
function NLMEEstimation.conditional_loglikelihood_and_gradient(m::LocationModel, i, θ, σ, η)
    value = conditional_loglikelihood(m, i, θ, σ, η)
    gradient = [sum(y -> y - θ[1] - η[1], m.y[i]) * exp(-2σ[1])]
    return value, gradient
end
m = LocationModel([[1.0, 1.5], [2.0, 2.5]])
θ, σ, ω = [1.5], [log(0.5)], [log(0.3)]
println((subjects=nsubjects(m), eta_dimension=neta(nlme_layout(m))))
println(check_protocol(m, θ, σ, ω))
```

`check_protocol` checks consistency between the declared layout, scalar and batched evaluations, and derivatives. A pass at these coordinates is a local check. Exercise your model at representative parameters, including difficult regions, before starting a long fit.

The model's `θ` and `σ` coordinates must be unconstrained: every real vector of the declared length represents a valid parameter value. Apply logarithmic or other transformations inside the likelihood where needed. The package owns `ω`, the log-Cholesky representation of the random-effect covariance. See [random effects](random-effects.md) for that representation.

## Derivatives

The example supplies an analytical gradient in `η`. Alternatively, declare `ad_backend(model)` and load the package implementing that AD backend. The built-in [PK model](oral-pk.md) uses Enzyme. The package's documented AD path is reverse mode with Enzyme; a different backend should be verified against the operations in your likelihood.

| Operation | Model information |
|:--|:--|
| Subject modes with `QuasiNewtonEBE` | Conditional likelihood and `η` gradients |
| Subject modes with `NewtonEBE` | Conditional likelihood, `η` gradients and Hessian blocks |
| FOCE approximation | Gaussian observation means, variances and their `η` Jacobians |
| SAEM Metropolis transitions | Conditional likelihood |
| SAEM MALA or NUTS transitions | Conditional likelihood and `η` gradients |
| Numerical SAEM population update | Population gradients where the selected update uses them |

These are requirements of the named operation, not a claim that an entire fit never requests other information. FOCE first solves subject modes with `QuasiNewtonEBE`; that solver can inspect available Hessians to reject saddles. Its first-order approximation itself uses observation Jacobians. NUTS uses Pathfinder initialization and continuous adaptation; it does not require an `η` Hessian metric from the model.

`hessian_ad_backend` defaults to `ad_backend`. Return `nothing` when only first derivatives are supported. A model implementing Hessian methods directly must declare `provides_eta_hessian`. Hessians are subject blocks, never a joint matrix across all subjects. A truthful sparsity declaration through `eta_hessian_sparsity` can reduce the cost of recovering those blocks.

## Observation moments

FOCE and residual diagnostics need a Gaussian observation interface in addition to the density. Extend the location model with moments and their Jacobians:

```@example protocol
NLMEEstimation.observation_model(::LocationModel) = GaussianObservations()
NLMEEstimation.observations(m::LocationModel, i::Integer) = m.y[i]
function NLMEEstimation.observation_moments(m::LocationModel, i::Integer, θ, σ, η)
    n = length(m.y[i])
    return fill(θ[1] + η[1], n), fill(exp(2σ[1]), n)
end
function NLMEEstimation.observation_moments_and_jacobians(m::LocationModel, i::Integer, θ, σ, η)
    μ, v = observation_moments(m, i, θ, σ, η)
    return μ, v, ones(length(μ), 1), zeros(length(v), 1)
end
println(check_protocol(m, θ, σ, ω))
```

The variance is the conditional observation variance, not the residual standard deviation. Jacobian rows follow observation order; columns follow the subject's `η` coordinates. `check_protocol` checks that the moments reproduce the declared likelihood.

## Additional contracts

Use [`BatchedEvaluation`](@ref) when the model natively evaluates subjects together. Its likelihood output has one entry per requested subject, in request order. The fallback supplies batched calls for a per-subject model. A batched implementation must preserve separability: changing one subject's effects must not change another subject's density.

[`population_logprior`](@ref) declares a population prior. It is added once per population objective evaluation, after summing subjects. It changes the fitted target from ML to MAP and does not change subject EBEs at fixed population parameters. Return `nothing` when no population prior is intended.

[`MuReferencing`](@ref) declares how population coordinates shift the Gaussian subject effects and enables closed-form updates. It must describe an invariance of the likelihood, not an approximate relationship. The protocol check verifies that invariance locally. More specialized integrated-family and Laplace-gradient hooks are grouped in the [protocol reference](reference/protocol.md).

The [oral PK example](oral-pk.md) uses a model that implements the likelihood, derivatives, Gaussian moments and mu-referencing contracts together.
