# Estimator research

This directory holds generators for estimator experiments. They are not part of
the documentation build and are not a supported quick-start interface. Many
record a particular historical kernel contract; inspect the script and the
corresponding Git revision before rerunning an old experiment.

| Scripts | Question |
|:--|:--|
| `ebe_*` | Mode-search scaling, gradients and ragged layouts |
| `saem_kernels.jl`, `saem_fit_kernels.jl`, `saem_kernel_*` | Kernel comparisons and tuning |
| `saem_nuts_*` | Continuous adaptation, batching, threading and ragged NUTS |
| `saem_mala_*` | MALA behavior in nonlinear models |
| `saem_numeric_mstep.jl`, `saem_minibatch.jl` | Numerical population updates |
| `saem_integrated.jl`, `saem_ragged_kernels.jl` | Integrated effects and varying subject dimensions |
| `saem_streaming_brm*`, `saem_threads_vs_lockstep_brm.jl` | Emitted-model and execution experiments |
| `saem_map_block.jl` | Population-prior behavior |
| `saem.jl` | Basic SAEM experiment |

Regenerable `.txt` and `.tsv` dumps are no longer committed. Capture new runs
outside the repository and retain the source revision, dependency versions,
command, random seeds, host and exit receipt alongside any numerical summary.
Timed measurements need their host-load conditions and a clearly stated
comparison; a run on a shared host is not a portable speed claim.

The removed benchmark tables and their recorded environments remain in Git
history at `02d2c6e069b8404c2680cd94ae3601cf784ecaf8`, in
`docs/src/saem-benchmarks.md` and `bench/RESULTS.md`. Small external oracle data
under `test/data` remain because the tests consume them and their provenance
is documented there. Do not remove those fixtures as benchmark output.

The former machine-specific thread-sweep launcher and dependency-bootstrap script
were removed. Run the retained Julia experiment in an explicitly prepared
public package environment. Logs must label the machine generically rather
than recording a private hostname.
