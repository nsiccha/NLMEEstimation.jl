# NLMEEstimation.jl

NLMEEstimation estimates population parameters and subject effects for nonlinear mixed-effects models defined by Julia likelihoods.

These shell commands install the example environment with Julia 1.11 or later:

```sh
git clone https://github.com/nsiccha/NLMEEstimation.jl
cd NLMEEstimation.jl && julia --project=docs docs/install.jl
```

Fit absorption, clearance, volume and residual error in a simulated oral PK study:

```@eval
Main.DocsSupport.run("pk_fit")
```

This output is from the [cached fit and receipt](oral-pk.md#Reproduce). The fit holds the weight effect and random-effect covariance fixed; `free` selects the estimated coordinates.

## Reading paths

Use the [oral PK example](oral-pk.md) to inspect the model, its predictions and residuals. For your own likelihood, begin with the [model protocol](protocol.md), then choose an [estimator](fitting.md). The [reference](reference/estimation.md) documents the available arguments and result fields.

`NLMEEstimation` owns the Gaussian random-effect prior and covariance coordinates. Your model supplies the conditional likelihood, parameter layout and derivative backend. A SAEM fit is a population point estimate; its internal chains do not constitute a population posterior sample.

## Subject predictions

The cached fit also distinguishes the typical population curve from the first subject's fitted curve:

```@example startfigure
using Main.DocsSupport # hide
DocsSupport.figure(:predictions) # hide
```
