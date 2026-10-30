"""
    NLMEEstimation

Model-agnostic estimators for nonlinear mixed-effects (NLME) models.

A model is any Julia value for which the protocol functions are implemented
(see [`nsubjects`](@ref), [`nlme_layout`](@ref), and per [`evaluation_style`](@ref)
either [`conditional_loglikelihood`](@ref) or [`conditional_loglikelihoods`](@ref)).
The package owns the random-effect prior `η_i ~ N(0, Ω)` and its parameterization
(see [`NLMELayout`](@ref)); the model owns everything else.

Derivatives come from automatic differentiation: a model either implements the
derivative methods itself or declares an AD backend through [`ad_backend`](@ref).
The package promises and tests reverse-mode AD with Enzyme. Second order is
optional and only ever per-subject blocks ([`provides_eta_hessian`](@ref)).

Estimators:

- [`empirical_bayes`](@ref) (alias [`posthoc`](@ref)): per-subject posterior modes
  (EBEs), by Newton on Hessian blocks or by gradient-only BFGS, at fixed population
  parameters.
- [`focei`](@ref): population estimation using first-order conditional curvature
  and a derivative-free outer optimizer.
- [`saem`](@ref): population estimation by MCMC stochastic approximation.

[`covariance_step`](@ref) gives standard errors of a fit's population coordinates, and
[`marginal_loglikelihood`](@ref) its likelihood, OFV, AIC and BIC (for SAEM, by
importance sampling or adaptive Gauss–Hermite quadrature).
[`residual_diagnostics`](@ref) gives PRED, IPRED, IWRES, CWRES and NPDE.

Each of these functions takes `progress = nothing`: a Treebars node under which it
builds its own labelled progress node, or a backend symbol such as `:term` for a new
root. With `nothing` no node is created. `Treebars.request_interrupt!` on that node
stops [`saem`](@ref) and [`focei`](@ref) early with their estimate so far; the other
functions raise `Treebars.ProgressInterrupted`.
"""
module NLMEEstimation

using LinearAlgebra
using Random
import SparseArrays
using SparseArrays: SparseMatrixCSC, sparse
import SparseMatrixColorings as SMC
import ADTypes
import DifferentiationInterface as DI
import NLopt
import LogDensityProblems as LDP
import WarmupHMC
using Treebars: Fraction, ProgressInterrupted, finalize_progress!, interrupt_requested,
    is_running, prepare_progress!, short_string, start_progress!, throw_if_interrupted,
    update_progress!, with_progress

export NLMELayout, RaggedNLMELayout, neta, nomega, omega_cholesky, omega_matrix, omega_parameters,
    FixedOmega, fixed_omega, apply_fixed_omega
export nsubjects, nlme_layout, EvaluationStyle, PerSubjectEvaluation, BatchedEvaluation,
    evaluation_style, conditional_loglikelihood, conditional_loglikelihoods,
    conditional_loglikelihood_and_gradient, conditional_loglikelihoods_and_gradients,
    conditional_loglikelihood_gradient_and_hessian, conditional_loglikelihood_hessian_blocks,
    conditional_loglikelihood_and_population_gradient,
    ad_backend, hessian_ad_backend, provides_eta_hessian, eta_hessian_sparsity, parameter_names,
    population_logprior, population_logprior_and_gradient, population_prior_provenance
export mu_referencing, MuReferencing, observation_model, GaussianObservations,
    observations, observation_moments, observation_moments_and_jacobians
export NormalREPrior, RaggedNormalREPrior, re_prior, re_logprior, re_logprior_gradient_and_hessian,
    complete_data_loglikelihood, complete_data_loglikelihood_gradient_and_hessian
export check_protocol, ProtocolViolation
export empirical_bayes, posthoc, EBEMethod, NewtonEBE, QuasiNewtonEBE, default_ebe_method,
    EBEOptions, EBEResult, RaggedEBEResult, EBEConvergenceError, ebe_covariances
export focei, focei_evaluate, FOCEOptions, FOCEResult, FOCEEvaluation
export covariance_step, CovarianceResult, CovarianceStepError, FOCEApproximation,
    delta_method, omega_elements
export marginal_loglikelihood, MarginalLikelihoodResult, ImportanceSampling,
    AdaptiveGaussHermite, LouisInformation
export residual_diagnostics, ResidualDiagnostics
export saem, saem_population_gradient_style, SAEMOptions, SAEMResult, RaggedSAEMResult,
    StandardSAEMMH, WarmupHMCNUTS, DualAveragingMALA

include("layout.jl")
include("protocol.jl")
include("re_prior.jl")
include("sparse_curvature.jl")
include("fixed_omega.jl")
include("conformance.jl")
include("empirical_bayes.jl")
include("focei.jl")
include("saem.jl")
include("ragged.jl")
include("covariance.jl")
include("marginal.jl")
include("louis.jl")
include("diagnostics.jl")
include("ReferenceModels.jl")

end
