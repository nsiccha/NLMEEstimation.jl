# SAEM kernel benchmark, 2026-10-11

Run on gordito with Julia 1.10.12, one BLAS thread, the public Theoph model,
six subjects, three sampler seeds (2026–2028), and WarmupHMC `dev` at
`2b7b1549c5c91d99da42903110f15632db0b54ec`. These are local timing
measurements on a shared host; the scripts below reproduce them.

## Fixed population parameters

Command: `julia --startup-file=no --project=test bench/saem_kernels.jl`.
Each subject contributes 800 retained draws. MH uses 200 scale-adaptation
sweeps. NUTS warms each subject once (20 draw floor, 100 initial gradient
evaluations), then streams 800 draws at the same target. Seconds include
warmup and the 800 draws, exclude Julia compilation and a separate probe of
20 additional one-transition stream resumptions per subject. ESS is the
median initial-positive-sequence estimate over subject and η coordinates,
then over the three seeds.

| Kernel | Median seconds | Median ESS | ESS/second | Divergences in retained NUTS draws | Stream resume ms/transition |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prior + random-walk MH | 0.00593 | 73.9 | 12,446 | — | — |
| NUTS, Hessian diagonal initialization | 0.0760 | 407 | 5,351 | 0 | 2.63 |
| NUTS, prior covariance initialization | 0.0796 | 337 | 4,234 | 0 | 2.69 |

MH's median prior-proposal acceptance was 4.6% and random-walk acceptance
27.5%. Hessian initialization improved NUTS ESS on this model, while MH had
higher ESS per second. The 2.6 ms stream-resume cost matters for SAEM because
its target changes and `stream_mcmc` is called for one transition per subject
per iteration.

## End-to-end short SAEM fit

Command: `julia --startup-file=no --project=test bench/saem_fit_kernels.jl`.
The same six Theoph subjects ran 40 SAEM iterations plus five MCMC burn-in
sweeps, one kernel round per iteration. The NUTS path used an eight-draw warmup
floor and 50 initial gradient evaluations per subject. Results are medians of
three seeds after compilation.

| Kernel | Median fit seconds | Maximum NUTS divergences | Finite estimates |
| --- | ---: | ---: | --- |
| Prior + random-walk MH | 0.00319 | — | yes |
| WarmupHMC NUTS | 1.38 | 0 | yes |

The short fit measures execution cost and catches integration failures; it is
too short to establish parameter recovery or compare estimator accuracy. The
fixed-target ESS comparison and end-to-end timing represent this three-effect
Theoph model, not a general ranking of the samplers.
