# Documentation

The navigation has four groups. These are the complete reader-facing pages;
research records and generators stay outside `docs/src`.

| Group | Page | Purpose |
|---|---|---|
| Start | `index.md` | Install the package and fit a complete simulated PK model. |
| Guides | `protocol.md` | Implement a model, its derivatives and observation moments. |
| Guides | `random-effects.md` | Choose covariance blocks, ragged families and fixed components. |
| Guides | `fitting.md` | Choose an estimator and interpret its stopping information. |
| Guides | `diagnostics.md` | Separate subject predictions, residuals and population uncertainty. |
| Examples | `oral-pk.md` | Fit a one-compartment oral PK model and inspect subject predictions. |
| Reference | `reference/protocol.md` | Model, derivative, prior and observation hooks. |
| Reference | `reference/random-effects.md` | Layouts, covariance coordinates and random-effect priors. |
| Reference | `reference/estimation.md` | Estimators, options, results and errors. |
| Reference | `reference/diagnostics.md` | Likelihood evaluation, uncertainty and residual APIs. |
| Reference | `reference/models.md` | The included PK model and simulation functions. |

## Build

From a checkout, use Julia 1.11 or later and Node.js 20 or later:

```sh
julia --project=docs docs/install.jl
npm install --prefix docs
julia --project=docs docs/make.jl
python3 docs/check_links.py docs/build/1
```

On a managed worktree, `kb-docs-build` resolves the environment and runs the build.
The build fails on missing export docs,
execution errors, stale fit receipts and local lint errors. It never estimates
population parameters or writes a fit cache.

`research/docs/README.md` describes the explicit cache-generation command.
Only small numerical summaries and receipts are committed. AlgebraOfVega renders
figures during the build; generated assets belong to the ignored build tree.
The build uses the default VitePress theme, retaining local colour tokens until
DocsKit supplies the shared configuration. No theme or widget code is copied
from another package.
