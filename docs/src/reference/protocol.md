# Model protocol

## Likelihood interface

```@docs
nsubjects
nlme_layout
EvaluationStyle
PerSubjectEvaluation
BatchedEvaluation
evaluation_style
conditional_loglikelihood
conditional_loglikelihoods
parameter_names
```

## Derivatives

```@docs
conditional_loglikelihood_and_gradient
conditional_loglikelihoods_and_gradients
conditional_loglikelihood_gradient_and_hessian
conditional_loglikelihood_hessian_blocks
conditional_loglikelihood_and_population_gradient
conditional_loglikelihood_sum_and_population_gradient
ad_backend
hessian_ad_backend
provides_eta_hessian
eta_hessian_sparsity
```

## Population priors

```@docs
population_logprior
population_logprior_and_gradient
population_prior_provenance
```

## Observation moments

```@docs
mu_referencing
MuReferencing
observation_model
GaussianObservations
observations
observation_moments
observation_moments_and_jacobians
```

## Integrated effects

```@docs
provides_laplace_gradient
laplace_marginals_and_gradient
integrated_families
laplace_conditionals
laplace_conditionals_and_gradients
laplace_conditional_moments
laplace_conditionals_and_population_gradient
```

## Conformance

```@docs
check_protocol
ProtocolViolation
```
