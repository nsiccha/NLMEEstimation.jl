# Documentation fit

`pk_fit.jl` is the complete script displayed on Start and the oral PK example.
`pk_diagnostics.jl` continues the same run. Both scripts execute when the cache
is generated; their stdout is saved verbatim. Pages read those exact scripts
and outputs, not separately maintained copies.

From a checkout with the docs dependencies installed:

```sh
julia --project=docs research/docs/generate.jl
```

The generator refuses CI. An existing content key is a cache hit and causes no
new fit. A changed package source, project, script or generator gives a new key.
The digest is computed over paths and file bytes, including separators. The
receipt records that key, source base revision, dirty state, Julia version,
resolved dependency versions, UTC generation time and elapsed seconds. Review
and remove the previous `docs/results/pk-*.toml` after generating a replacement.

The small TOML file contains population estimates, subject modes, observation
predictions, residuals and convergence diagnostics. It contains no sampler draws,
serialized model or generated figures. The build refuses a missing or stale
receipt rather than estimating in CI. Figures are generated from the summary
with AlgebraOfVega and embedded as same-origin HTML in the build directory.

The current example fixes the covariance and weight coefficient. Its simulated
data are a teaching fixture, not a benchmark or evidence for estimator accuracy.
The cache has one scientific run; elapsed time is provenance, not a performance
comparison. A future DocsKit/DocsFits adapter should replace the local cache
transport while retaining this explicit regeneration boundary.
