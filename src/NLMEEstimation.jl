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

[`covariance_step`](@ref) gives standard errors of a fit's population coordinates.
"""
module NLMEEstimation

using LinearAlgebra
using Random
import ADTypes
import DifferentiationInterface as DI
import NLopt
import LogDensityProblems as LDP
import WarmupHMC

export NLMELayout, RaggedNLMELayout, neta, nomega, omega_cholesky, omega_matrix, omega_parameters,
    FixedOmega, fixed_omega, apply_fixed_omega
export nsubjects, nlme_layout, EvaluationStyle, PerSubjectEvaluation, BatchedEvaluation,
    evaluation_style, conditional_loglikelihood, conditional_loglikelihoods,
    conditional_loglikelihood_and_gradient, conditional_loglikelihoods_and_gradients,
    conditional_loglikelihood_gradient_and_hessian, conditional_loglikelihood_hessian_blocks,
    conditional_loglikelihood_and_population_gradient,
    ad_backend, hessian_ad_backend, provides_eta_hessian, parameter_names,
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
export saem, saem_population_gradient_style, SAEMOptions, SAEMResult, RaggedSAEMResult,
    StandardSAEMMH, WarmupHMCNUTS, DualAveragingMALA

include("layout.jl")
include("protocol.jl")
include("re_prior.jl")
include("fixed_omega.jl")
include("conformance.jl")
include("empirical_bayes.jl")
include("focei.jl")
include("saem.jl")
include("ragged.jl")
include("covariance.jl")
include("ReferenceModels.jl")

end
