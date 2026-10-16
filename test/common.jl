# Shared fixtures: the reference model on simulated data and on R's Theoph data, a
# batched joint-program view of it, and the brute-force joint-optimization reference
# for the EBEs. The large joint oracle uses ForwardDiff; production paths use Enzyme.
module TestCommon

using NLMEEstimation
using NLMEEstimation.ReferenceModels
import NLMEEstimation as NE
import DifferentiationInterface as DI
using DifferentiationInterface: AutoEnzyme, AutoForwardDiff
import Enzyme
import ForwardDiff
using DelimitedFiles
using LinearAlgebra
using Random
import Optim

export θ_SIM, σ_SIM, Ω_SIM, simulated_model, θ_THEOPH, σ_THEOPH, Ω_THEOPH, theoph_model,
    JointView, joint_reference, joint_logdensity, joint_conditional, REVERSE, RUNTIME_REVERSE, RUNTIME_HESSIAN, JOINT_HESSIAN,
    DoubleWell

# The promised configuration: Enzyme reverse mode with static activity analysis (DI
# selects forward-over-reverse for Hessians of `AutoEnzyme()`). The package's own
# code and the reference model are tested with it.
const REVERSE = AutoEnzyme()
# Test fixtures written as plain Julia that Enzyme's static activity analysis cannot
# prove (closures and comprehensions mixing constant model data with active η:
# `JointView`, the coupled negative control, the joint reference) declare Enzyme's
# documented runtime-activity mode. The same backends serve as an independently
# configured reference for the promised path.
const RUNTIME_REVERSE = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
const RUNTIME_HESSIAN = DI.SecondOrder(AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Forward)), RUNTIME_REVERSE)
# An independent joint-Hessian oracle uses forward-mode AD over all subjects.
# Enzyme's forward-over-reverse compiler on Julia prereleases can fail on this
# large reference expression even when the package's per-subject Hessians work.
const JOINT_HESSIAN = AutoForwardDiff()

const θ_SIM = [log(1.5), log(2.8), log(32.0), 0.75]
const σ_SIM = [log(0.3), log(0.1)]
const Ω_SIM = Matrix(Diagonal([0.4, 0.3, 0.2] .^ 2))

function simulated_model(; seed = 11, nsubjects = 24, Ω = Ω_SIM, eta_blocks = [1, 1, 1])
    subjects, η = simulate_one_compartment_oral(Xoshiro(seed), θ_SIM, σ_SIM, Ω; nsubjects)
    return OneCompartmentOral(subjects; backend = REVERSE, eta_blocks), η
end

# One η per subject, ℓ_i(η) = -(η² - a)²/4 + b_i η, with closed-form derivatives, for
# either layout. With a unit prior variance and a > 1 the complete-data density is
# convex near η = 0 and concave at its maximum, so Newton's damping must rise past
# trials where -H + λI is not positive definite.
struct DoubleWell{L}
    layout::L
    a::Float64
    b::Vector{Float64}
end
NE.nsubjects(m::DoubleWell) = length(m.b)
NE.nlme_layout(m::DoubleWell) = m.layout
NE.conditional_loglikelihood(m::DoubleWell, i, θ, σ, η) = -(η[1]^2 - m.a)^2 / 4 + m.b[i] * η[1]
NE.conditional_loglikelihood_and_gradient(m::DoubleWell, i, θ, σ, η) =
    (NE.conditional_loglikelihood(m, i, θ, σ, η), [-(η[1]^2 - m.a) * η[1] + m.b[i]])
NE.conditional_loglikelihood_gradient_and_hessian(m::DoubleWell, i, θ, σ, η) =
    (NE.conditional_loglikelihood_and_gradient(m, i, θ, σ, η)..., fill(m.a - 3 * η[1]^2, 1, 1))
NE.provides_eta_hessian(::DoubleWell) = true

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
    return OneCompartmentOral(subjects; backend = REVERSE, error_model = AdditiveError())
end

# A joint-program view of a per-subject model, shaped like a BRM/RK backend: it
# declares BatchedEvaluation and implements only the batched log-density, by placing
# the requested subjects' η into a full N-subject matrix (other subjects at zero) and
# evaluating every subject. `hessian_backend = nothing` makes it gradient-only.
# `calls` counts batched evaluations.
struct JointView{M, B, HB}
    inner::M
    backend::B
    hessian_backend::HB
    calls::Base.RefValue{Int}
end
JointView(inner; backend = RUNTIME_REVERSE, hessian_backend = RUNTIME_HESSIAN) =
    JointView(inner, backend, hessian_backend, Ref(0))
NE.evaluation_style(::JointView) = BatchedEvaluation()
NE.nsubjects(v::JointView) = nsubjects(v.inner)
NE.nlme_layout(v::JointView) = nlme_layout(v.inner)
NE.ad_backend(v::JointView) = v.backend
NE.hessian_ad_backend(v::JointView) = v.hessian_backend
function NE.conditional_loglikelihoods(v::JointView, subjects, θ, σ, H)
    v.calls[] += 1
    n = nsubjects(v.inner)
    # Scalar scatter avoids Julia 1.13 Enzyme's unsupported indexed slice write.
    full = zeros(eltype(H), size(H, 1), n)
    for (k, i) in enumerate(subjects), r in axes(H, 1)
        full[r, i] = H[r, k]
    end
    all = [conditional_loglikelihood(v.inner, i, θ, σ, view(full, :, i)) for i in 1:n]
    return all[subjects]
end

# The joint complete-data log-density over every subject's η, stacked in x.
function joint_logdensity(x, model, θ, σ, prior, m)
    n = nsubjects(model)
    return sum(i -> complete_data_loglikelihood(model, i, θ, σ, prior, x[((i - 1) * m + 1):(i * m)]), 1:n)
end
_negative_joint(x, model, θ, σ, prior, m) = -joint_logdensity(x, model, θ, σ, prior, m)

# The joint conditional log-density (no η prior) over every subject's η, stacked in x.
# Its Hessian plus the prior's closed-form block-diagonal -Ω⁻¹ is the joint Hessian;
# AD never differentiates the prior's triangular solve in forward mode (Enzyme has no
# forward-mode rule for LAPACK `trtrs`).
function joint_conditional(x, model, θ, σ, m)
    n = nsubjects(model)
    return sum(i -> conditional_loglikelihood(model, i, θ, σ, x[((i - 1) * m + 1):(i * m)]), 1:n)
end

# The EBEs as one joint optimization: maximize the sum of all subjects' complete-data
# log-densities over all η at once with L-BFGS and an Enzyme gradient (runtime
# activity, see `RUNTIME_REVERSE`). It shares no
# code with the per-subject solvers beyond the log-density itself.
function joint_reference(model, θ, σ, ω)
    n = nsubjects(model)
    m = neta(nlme_layout(model))
    prior = re_prior(model, ω)
    contexts = (DI.Constant(model), DI.Constant(θ), DI.Constant(σ), DI.Constant(prior), DI.Constant(m))
    objective(x) = _negative_joint(x, model, θ, σ, prior, m)
    gradient!(G, x) = DI.gradient!(_negative_joint, G, RUNTIME_REVERSE, x, contexts...)
    result = Optim.optimize(objective, gradient!, zeros(n * m), Optim.LBFGS(),
        Optim.Options(g_tol = 1.0e-11, iterations = 100_000))
    Optim.converged(result) || error("joint reference did not converge: $result")
    return reshape(Optim.minimizer(result), m, n), prior
end

end
