# Historical emitted-model comparison

These scripts preserve the earlier BayesianRegressionModels / ReactiveKernels
comparison against the exact commits listed in `setup_env.jl`. They are a
historical reproduction and do not describe the current constructor contracts.
Use their pinned environment only; do not run them against the current package
as a smoke test. The current user example is in `research/docs` and the oral PK
documentation page.

The generated `pk_emitted_rkppl.jl` has been removed. The comparison script emits
that program from its pinned model artifact. Earlier outputs and the original
research narrative remain in Git history at package revision
`02d2c6e069b8404c2680cd94ae3601cf784ecaf8` (`docs/src/pk-comparison.md`).
