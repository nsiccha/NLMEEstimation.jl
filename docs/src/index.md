```@raw html
---
layout: home

hero:
  name: "NLMEEstimation.jl"
  text: "Estimators for nonlinear mixed-effects models"
  tagline: "Bring a conditional model; estimate random effects and population parameters."
  actions:
    - theme: brand
      text: Model protocol
      link: /protocol
    - theme: alt
      text: PK comparison
      link: /pk-comparison

features:
  - title: Model-agnostic protocol
    details: Use a per-subject likelihood or a batched program with automatic differentiation.
  - title: Empirical Bayes and FOCE(I)
    details: Estimate subject effects at fixed population parameters, or fit the population by FOCE or FOCEI.
  - title: SAEM with MH or NUTS
    details: Fit population parameters using stochastic approximation with per-subject MCMC transitions.
---
```

NLMEEstimation.jl defines estimation methods around a conditional likelihood supplied by a model. The package supplies the Gaussian random-effect prior and its covariance parameterization. Start with the [model protocol](protocol.md), then compare the [available estimators](estimators.md) in the [PK example](pk-comparison.md).
