# Estimators

| Method | Population parameters | Subject effects | Main output |
| --- | --- | --- | --- |
| `empirical_bayes` / `posthoc` | Fixed input θ, σ, Ω | Conditional posterior modes (EBEs) | `EBEResult.modes` |
| `focei` with `interaction = false` | Fitted by a derivative-free outer search | EBEs at the fit | `FOCEResult.theta`, `.sigma`, `.omega_matrix`, `.ofv` |
| `focei` with `interaction = true` (default) | Fitted by a derivative-free outer search | EBEs at the fit | Same fields, with the variance-Jacobian interaction term |
| `saem` with `StandardSAEMMH()` (default) | Fitted by stochastic approximation | One evolving η chain per subject | `SAEMResult.theta`, `.sigma`, `.omega_matrix`, `.acceptance` |
| `saem` with `WarmupHMCNUTS()` | Fitted by stochastic approximation | Per-subject WarmupHMC NUTS transitions | `SAEMResult.theta`, `.sigma`, `.omega_matrix`, `.nuts_divergences` |

Empirical Bayes is a conditional calculation at supplied population parameters. FOCE(I) and SAEM estimate population parameters. Their objectives and approximations differ, so compare estimates and predictions on the same data rather than treating objective values as interchangeable. Full joint MCMC can provide a posterior comparison but is not an NLMEEstimation estimator.

See the [PK comparison](pk-comparison.md) for one reproducible model, dataset, seeds, and all method calls.
