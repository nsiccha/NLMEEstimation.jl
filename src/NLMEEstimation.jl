"""
    NLMEEstimation

Model-agnostic estimators for nonlinear mixed-effects (NLME) models.

A model is any Julia value for which the protocol functions are implemented
(see [`nsubjects`](@ref), [`nlme_layout`](@ref), [`conditional_loglikelihood`](@ref)).
The package owns the random-effect prior `η_i ~ N(0, Ω)` and its parameterization
(see [`NLMELayout`](@ref)); the model owns everything else.

Derivatives always come from automatic differentiation: a model either implements
the derivative methods itself or declares an AD backend through [`ad_backend`](@ref).
Finite-difference backends are refused.

Estimators:

- [`empirical_bayes`](@ref) (alias [`posthoc`](@ref)): per-subject posterior modes
  (EBEs) and Hessians at fixed population parameters.
"""
module NLMEEstimation

using LinearAlgebra
using Random
import ADTypes
import DifferentiationInterface as DI

export NLMELayout, neta, nomega, omega_cholesky, omega_matrix, omega_parameters
export nsubjects, nlme_layout, conditional_loglikelihood,
    conditional_loglikelihood_and_gradient, conditional_loglikelihood_gradient_and_hessian,
    conditional_loglikelihood_and_population_gradient,
    ad_backend, hessian_ad_backend, parameter_names
export mu_referencing, MuReferencing, observation_model, GaussianObservations,
    observations, observation_moments
export NormalREPrior, re_prior, re_logprior, re_logprior_gradient_and_hessian,
    complete_data_loglikelihood, complete_data_loglikelihood_gradient_and_hessian
export check_protocol, ProtocolViolation
export empirical_bayes, posthoc, EBEOptions, EBEResult, EBEConvergenceError,
    ebe_covariances

include("layout.jl")
include("protocol.jl")
include("re_prior.jl")
include("conformance.jl")
include("empirical_bayes.jl")
include("ReferenceModels.jl")

end
