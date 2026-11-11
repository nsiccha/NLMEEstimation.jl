using NLMEEstimation, NLMEEstimation.ReferenceModels
using ADTypes, Enzyme, LinearAlgebra, Random
θ = [log(1.2), log(4.0), log(35.0), 0.0]
σ = [log(0.25)]
Ω = Diagonal(fill(0.3^2, 3))
subjects, _ = simulate_one_compartment_oral(MersenneTwister(42), θ, σ, Ω;
    error_model=AdditiveError(), nsubjects=20)
model = OneCompartmentOral(subjects; backend=AutoEnzyme(),
    error_model=AdditiveError())
ω = omega_parameters(nlme_layout(model), Ω)
fit = focei(model, θ, σ, ω; free=[1, 2, 3, 5], options=FOCEOptions(maxeval=800))
println("Converged: ", fit.converged, "; subjects: ", nsubjects(model))
println("ka (1/h), CL (L/h), V (L): ", round.(exp.(fit.theta[1:3]); digits=3))
