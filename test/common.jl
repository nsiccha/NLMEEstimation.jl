# Shared fixtures: the reference model on simulated data and on R's Theoph data,
# and the brute-force joint-optimization reference for the EBEs.
module TestCommon

using NLMEEstimation
using NLMEEstimation.ReferenceModels
using DifferentiationInterface: AutoForwardDiff
using DelimitedFiles
using LinearAlgebra
using Random
import ForwardDiff
import Optim

export θ_SIM, σ_SIM, Ω_SIM, simulated_model, θ_THEOPH, σ_THEOPH, Ω_THEOPH, theoph_model,
    joint_reference, stacked

const θ_SIM = [log(1.5), log(2.8), log(32.0), 0.75]
const σ_SIM = [log(0.3), log(0.1)]
const Ω_SIM = Matrix(Diagonal([0.4, 0.3, 0.2] .^ 2))

function simulated_model(; seed = 11, nsubjects = 24, Ω = Ω_SIM, eta_blocks = [1, 1, 1])
    subjects, η = simulate_one_compartment_oral(Xoshiro(seed), θ_SIM, σ_SIM, Ω; nsubjects)
    return OneCompartmentOral(subjects; backend = AutoForwardDiff(), eta_blocks), η
end

# Typical-value parameters of the standard one-compartment Theoph fit (no weight
# effect), additive error.
const θ_THEOPH = [log(1.57), log(2.75), log(31.5), 0.0]
const σ_THEOPH = [log(0.7)]
const Ω_THEOPH = Matrix(Diagonal([0.6, 0.26, 0.14] .^ 2))

function theoph_model()
    data, header = readdlm(joinpath(@__DIR__, "data", "theoph.csv"), ','; header = true)
    col(name) = data[:, findfirst(==(name), vec(header))]
    id, wt, dose, time, conc = col("Subject"), col("Wt"), col("Dose"), col("Time"), col("conc")
    subjects = map(sort(unique(id))) do s
        rows = findall(==(s), id)
        PKSubject(dose[rows[1]] * wt[rows[1]], wt[rows[1]], time[rows], conc[rows])
    end
    return OneCompartmentOral(subjects; backend = AutoForwardDiff(), error_model = AdditiveError())
end

stacked(modes) = reduce(vcat, modes)

# The EBEs as one joint optimization: maximize the sum of all subjects' complete-data
# log-densities over all η at once with L-BFGS and an AD gradient. It shares no code
# with the per-subject Newton solver beyond the log-density itself.
function joint_reference(model, θ, σ, ω)
    n = nsubjects(model)
    m = neta(nlme_layout(model))
    prior = re_prior(model, ω)
    block(x, i) = view(x, ((i - 1) * m + 1):(i * m))
    joint(x) = sum(i -> complete_data_loglikelihood(model, i, θ, σ, prior, block(x, i)), 1:n)
    objective(x) = -joint(x)
    gradient!(G, x) = ForwardDiff.gradient!(G, objective, x)
    result = Optim.optimize(objective, gradient!, zeros(n * m), Optim.LBFGS(),
        Optim.Options(g_tol = 1.0e-11, iterations = 100_000))
    Optim.converged(result) || error("joint reference did not converge: $result")
    x = Optim.minimizer(result)
    return [collect(block(x, i)) for i in 1:n], joint
end

end
