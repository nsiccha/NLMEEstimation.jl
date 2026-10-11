# Textbook-frequency SAEM M-step with BRM's slope MuReferencing trait.
# Run from a project that develops this NLMEEstimation checkout, BRM 1e7fbaa,
# and the matching ReactiveKernels/RKPPL sources.
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using NLMEEstimation, Distributions, Random, LinearAlgebra, Printf
using DifferentiationInterface: AutoEnzyme
import Enzyme

include(joinpath(@__DIR__, "..", "docs", "examples", "brm_rkppl_pk_fixture.jl"))

BLAS.set_num_threads(1)
const AD = AutoEnzyme(; mode=Enzyme.Reverse)
data = pk_example_data(80)
construction = @timed brm_nlme_model(RKBRMI(pk_brmi(data)); ad_backend=AD)
model = construction.value
@printf("construction\tseconds=%.6f\tbytes=%d\n", construction.time, construction.bytes)
flush(stdout)

θ0 = [Dict("log_V_Intercept" => 1.6, "pop_log_CL.beta_pop.1" => 0.2,
    "pop_log_CL.beta_pop.2" => 0.25)[name] for name in parameter_names(model).theta]
σ0 = [log(0.6)]
ω0 = omega_parameters(nlme_layout(model), [0.3^2 0.0; 0.0 0.15^2])
@assert check_protocol(model, θ0, σ0, ω0; η=zeros(2, 80))

function fit(mode, iterations; warmup=false)
    options = SAEMOptions(; iterations, burnin=iterations ÷ 2,
        mcmc_burnin=15, mh_rounds=1, adapt_iterations=iterations ÷ 2,
        numeric_maxiter=12, mstep_interval=1, numeric_mstep=mode)
    GC.gc()
    measured = @timed saem(model, θ0, σ0, ω0; rng=Xoshiro(5),
        options, kernel=StandardSAEMMH())
    result = measured.value
    all(isfinite, (result.theta..., result.sigma..., result.omega...)) ||
        error("non-finite SAEM estimate")
    @printf("fit\tmode=%s\tresolved=%s\titerations=%d\twarmup=%s\tseconds=%.6f\tbytes=%d\taccept_rw=%.4f\ttheta=%s\tsigma=%s\tomega_diag=%s\n",
        mode, result.numeric_mstep, iterations, warmup, measured.time, measured.bytes,
        result.acceptance.random_walk, repr(result.theta), repr(result.sigma),
        repr(diag(result.omega_matrix)))
    flush(stdout)
    return result
end

fit(:online, 4; warmup=true)
fit(:mixture, 4; warmup=true)
@assert fit(:auto, 4; warmup=true).numeric_mstep == :online
for mode in (:auto, :mixture), iterations in (100, 200)
    fit(mode, iterations)
end
