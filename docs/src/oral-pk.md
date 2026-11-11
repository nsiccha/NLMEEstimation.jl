# Oral pharmacokinetics

Fit population pharmacokinetic parameters, inspect subject-specific predictions, and distinguish numerical convergence from model adequacy. The model is an analytical one-compartment system after an oral dose. No ODE solver or probabilistic-programming compiler is needed for this example; the same estimator interface also accepts likelihoods implemented with those tools.

## Data

The data are simulated concentrations after a single oral dose, with body weights drawn by the package simulator. Sampling times are measured in hours, doses in milligrams, weights in kilograms, volumes in litres and concentrations in milligrams per litre. Every subject has the same observation schedule. The simulation seed, generating parameters and residual model appear in the complete run below. Simulation makes the inputs reproducible and avoids implying that these observations represent a clinical study. The weight coefficient is fixed at zero, so the sampled body weights do not affect concentrations in this fit.

```@setup pk
using Main.DocsSupport
using NLMEEstimation, NLMEEstimation.ReferenceModels
model = DocsSupport.model()
summary = DocsSupport.cache()["fit"]
```

The model exposes the sample size and observation schedule directly:

```@example pk
first_subject = first(model.subjects)
println((subjects=nsubjects(model), dose_mg=first_subject.dose,
    observations_per_subject=length(first_subject.times)))
println((sampling_times_h=first_subject.times,))
```

The residual variation here is additive. Its standard deviation is constant on the concentration scale; it is not a percentage of the predicted concentration. Consequently, low concentrations can be negative under the observation model. That is a property of the stated Gaussian model, not a censoring rule. An assay with a lower quantification limit needs a likelihood that describes censoring explicitly.

## Model

For subject ``i``, absorption, clearance and volume are positive because the model uses logarithmic coordinates:

```math
\begin{aligned}
k_{a,i} &= \exp(\theta_1 + \eta_{i,1}),\\
CL_i &= \exp(\theta_2 + \theta_4\log(W_i/W_{\rm ref}) + \eta_{i,2}),\\
V_i &= \exp(\theta_3 + \eta_{i,3}),\qquad k_{e,i}=CL_i/V_i,\\
C_i(t) &= \frac{D_i k_{a,i}}{V_i(k_{a,i}-k_{e,i})}
          \left(e^{-k_{e,i}t}-e^{-k_{a,i}t}\right),\\
y_{ij} &\sim \operatorname{Normal}(C_i(t_{ij}),\exp(\sigma_1)^2),\qquad
\eta_i \sim \operatorname{Normal}(0,\Omega).
\end{aligned}
```

The package implementation evaluates the concentration expression with a stable limiting form when absorption and elimination rates coincide. The model owns the observation density. NLMEEstimation adds the random-effect prior, so that prior must not be included again inside the conditional likelihood.

The generated views identify the coordinates and covariance blocks used by this model:

```@example pk
println(parameter_names(model))
layout = nlme_layout(model)
println((population_coordinates=layout.ntheta,
    residual_coordinates=layout.nsigma, eta_blocks=layout.eta_blocks))
```

Each random effect has its own covariance block. This imposes independence between the absorption, clearance and volume effects. A single block would instead allow their correlations to be estimated. That choice changes the statistical model, not merely storage. The [random-effects guide](random-effects.md) explains the covariance coordinates and the difference between shared covariance families and subject-specific effects.

## Estimation

The complete run constructs the data and model, converts the covariance matrix into the package's unconstrained coordinates, and calls `focei`. The printed rates and volume are transformed back to physical units:

```@eval
Main.DocsSupport.run("pk_fit")
```

The starting values equal the simulation parameters. They do not encode the answer for a finite sample: the estimated parameters maximize an approximate likelihood for the observations actually drawn. The `free` vector selects the three logarithmic PK parameters and the logarithmic residual standard deviation. The weight exponent and covariance coordinates remain fixed. Treating those quantities as known is a deliberate restriction of this teaching example; a real analysis should justify it or estimate them.

FOCE uses a first-order approximation to the conditional observation moments around each subject's empirical Bayes estimate. With the additive residual model, the residual variance has no subject-effect dependence. The interaction option is therefore not a comparison being made by this example. The [fitting guide](fitting.md) describes when to use FOCE, SAEM or a subject-mode calculation at fixed population parameters.

This fit runs only when its content-addressed cache is generated. Documentation builds verify the digest and render the recorded output. They do not launch a population search. The receipt below identifies the code, environment and execution that produced the values; changing the model, settings or package sources invalidates that receipt.

## Predictions

Population predictions set the subject effects to zero. Individual predictions use the fitted subject mode and therefore incorporate that subject's observations. The following figure compares both curves with the first subject's concentrations. It is an in-sample comparison, not a predictive validation on held-out subjects.

```@example pk
DocsSupport.figure(:predictions) # hide
```

The line segments join model predictions at the observation times. They are not an additional interpolation model or uncertainty band. Differences between the blue and orange lines show how the subject effect changes the typical concentration profile. Agreement of the individual curve with the data is expected to improve after using those data to estimate the effect; it is insufficient evidence for the covariance structure or the population model.

## Diagnostics

Check both the outer optimizer and the subject-mode solves. A successful population stopping code alone does not certify a well-identified model or a unique optimum:

```@eval
Main.DocsSupport.run("pk_diagnostics")
```

The reported gradient is scaled by the random-effect prior, so its components are comparable in prior-standard-deviation coordinates. The convergence indicators describe numerical work performed for this run. They do not measure the accuracy of the FOCE approximation. Independent starting values, sensitivity to fixed covariance assumptions and an alternative likelihood approximation address different questions and belong in a substantive analysis.

Individual weighted residuals divide the observation-minus-individual-prediction difference by the conditional residual standard deviation. Their time pattern is visible across all subjects:

```@example pk
DocsSupport.figure(:residuals) # hide
```

A trend suggests a systematic mismatch in the time course; changing spread suggests a residual-error model that does not describe the concentration scale adequately. These are diagnostic interpretations, not hypothesis-test conclusions from this figure. The displayed residuals share fitted population parameters and subject effects. The [diagnostics guide](diagnostics.md) distinguishes IWRES from the conditional decorrelated residuals and simulation-based NPDE.

The example requests `replicates=0`, so the package does not simulate an NPDE reference distribution. It also does not calculate population standard errors. Those omissions keep the example focused on point estimation and observation-level checks. A covariance step would quantify local population-parameter uncertainty under its chosen approximation, while prediction uncertainty requires a separate propagation calculation.

## Reproduce

```@raw html
<details><summary>Commands and fit receipt</summary>
```

These shell commands explicitly generate a missing cached fit and rebuild the site; they are not executed by the documentation build:

```sh
julia --project=docs docs/install.jl
julia --project=docs research/docs/generate.jl
npm install --prefix docs
julia --project=docs docs/make.jl
```

```@eval
Main.DocsSupport.receipt()
```

The [generator and cache policy](https://github.com/nsiccha/NLMEEstimation.jl/tree/master/research/docs) describe the saved summaries. Earlier estimator comparisons and oracle provenance are in [bench](https://github.com/nsiccha/NLMEEstimation.jl/tree/master/bench) and [test/data](https://github.com/nsiccha/NLMEEstimation.jl/tree/master/test/data). They are research and validation records, not additional steps needed to use this model.

```@raw html
</details>
```
