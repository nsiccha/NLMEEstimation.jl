# Random effects

Choose covariance blocks and translate between covariance matrices and the unconstrained coordinates consumed by the estimators. The layout describes the statistical structure; the subject data remain in the model.

## Covariance coordinates

A block is a set of correlated random effects. Separate blocks are independent. The package stores each covariance block as ``\Omega_b=L_bL_b^\mathsf{T}``, with a logarithm on each diagonal element of ``L_b`` and untransformed entries below it.

```@example layout
using NLMEEstimation, LinearAlgebra
layout = NLMELayout(2, 1, [2])
Ω = [0.09 0.03; 0.03 0.16]
ω = omega_parameters(layout, Ω)
println((eta_dimension=neta(layout), covariance_coordinates=nomega(layout)))
println(isapprox(omega_matrix(layout, ω), Ω))
```

A single-effect block stores the logarithm of its standard deviation. A multivariate block stores Cholesky entries, so its off-diagonal coordinates are neither correlations nor covariances. Convert with `omega_parameters` and `omega_matrix` instead of constructing those coordinates by inspection.

## Subject-specific dimensions

A ragged layout shares covariance parameters across named families while allowing subjects to have different sets of effects. Each entry of `subject_families` lists the family of each successive block in that subject's effect vector. Repeating a family creates another independent block with the same covariance.

```@example layout
ragged = RaggedNLMELayout(2, 1, [2, 1], [[1], [1, 2], [1, 2, 2]])
println((subject_dimensions=[neta(ragged, i) for i in 1:3],
    shared_covariance_coordinates=nomega(ragged)))
```

The covariance dimension stays tied to the family definitions. It does not grow with every subject or repeated block. The model must use the same block order as the layout; absent effects are omitted, not padded with zeros. Ragged EBE and SAEM results store vectors for each subject because a rectangular matrix cannot describe those dimensions without padding.

## Fixed covariance components

Declare known covariance components with [`fixed_omega`](@ref). `FixedOmega(Ω)` fixes an entire family. The keyword form can instead fix selected standard deviations or a correlation matrix while leaving other components free:

```@example layout
known_sd = FixedOmega(; sd=[0.3, nothing])
println((fixed_standard_deviations=known_sd.sd,
    correlation_fixed=known_sd.correlation !== nothing))
```

The model returns a family-to-`FixedOmega` mapping. Its starting `ω` must already contain the fixed values; [`apply_fixed_omega`](@ref) constructs a compatible vector. FOCE and SAEM search only the remaining coordinates. This differs from selecting a few free coordinates for one FOCE call: the model declaration persists across estimator calls and is checked by the protocol validator.

## Prior and integration

NLMEEstimation adds the Gaussian random-effect prior to each conditional likelihood. Include observation terms in your model, and leave the random-effect normalizing constant and quadratic term to the package. [`re_prior`](@ref) and the [random-effect reference](reference/random-effects.md) expose the corresponding density calculations.

Integrated-family hooks permit a model to integrate selected effects by a Laplace approximation while other effects are sampled. They require consistent conditional values, gradients and moments; they are not a flag for ignoring an effect. Start with the unintegrated model and protocol checks before introducing that specialized interface.

The [oral PK example](oral-pk.md) uses independent covariance blocks and keeps their values fixed during its population search.
