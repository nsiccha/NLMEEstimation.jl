# NLMEEstimation.jl

NLMEEstimation estimates population parameters and subject effects for nonlinear
mixed-effects models defined by Julia conditional likelihoods.

[Start and installation](https://nsiccha.github.io/NLMEEstimation.jl/dev/) ·
[Oral PK example](https://nsiccha.github.io/NLMEEstimation.jl/dev/oral-pk) ·
[Model protocol](https://nsiccha.github.io/NLMEEstimation.jl/dev/protocol) ·
[API reference](https://nsiccha.github.io/NLMEEstimation.jl/dev/reference/estimation)

The model supplies the observation density, layout and derivatives.
NLMEEstimation supplies the Gaussian random-effect prior, subject-mode search,
FOCE and SAEM population estimation, and diagnostic calculations. SAEM returns
a population point estimate; its internal subject-effect chains are not a
population posterior sample.

## Installation

From a shell with Julia 1.11 or later, install the pinned example and docs
environment. These commands change the environment and are not run by the docs
build:

```sh
git clone https://github.com/nsiccha/NLMEEstimation.jl
cd NLMEEstimation.jl
julia --project=docs docs/install.jl
```

The package itself supports Julia 1.10; this installer uses Julia 1.11's
`[sources]` support to resolve the unregistered dependencies. The example script
is [`research/docs/pk_fit.jl`](research/docs/pk_fit.jl). The published example
shows its cached output and the receipt for the run.

## Development

[`docs/README.md`](docs/README.md) lists the documentation pages and build gates.
[`research/docs/README.md`](research/docs/README.md) describes explicit fit
regeneration. The docs build never runs a fit in CI and generates figures in its
ignored build directory.

Historical estimator experiments are under [`bench`](bench/README.md) and
[`research`](research/pk-comparison/README.md). External numerical-oracle fixtures
and their provenance are retained under [`test/data`](test/data/README.md).
