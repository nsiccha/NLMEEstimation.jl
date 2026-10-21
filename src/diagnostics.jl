"""
    ResidualDiagnostics

Per-observation residual diagnostics from [`residual_diagnostics`](@ref). Every
field below except `subjects`, `modes` and `replicates` holds one vector per
subject in `subjects`, in observation order:

- `observations`: `y`.
- `pred`: population predictions, the observation means at `η = 0`.
- `ipred`: individual predictions, the means at the subject's EBE `η̂` (`modes`).
- `iwres`: `(y - ipred) / sqrt(v(η̂))`, with `v` the observation variance.
- `cpred`: the population mean of the first-order linearization around `η̂`,
  `μ(η̂) - J η̂` with `J = ∂μ/∂η` at `η̂`.
- `cwres`: conditional weighted residuals: `U (y - cpred)`. Here `U` is the
  upper Cholesky factor of the inverse of the linearized covariance
  `C = J Ω Jᵀ + diag(v(η̂))`, so `UᵀU = C⁻¹` and `cwres` is decorrelated within
  the subject.
- `cwres_per_observation`: `(y - cpred) ./ sqrt.(diag(C))`, each observation
  standardized without decorrelation; this is what nlmixr2 reports as `CWRES`.
- `epred`, `npd`, `npde`: simulation-based diagnostics over `replicates`
  datasets simulated at the population parameters. `epred` is the simulated
  mean, and `npd` the normal quantile of the prediction discrepancy, the
  fraction of simulated values below `y`. `npde` is the same after
  decorrelation by the Cholesky factor of the inverse simulated covariance, as
  for `cwres`.
  `pde` is the decorrelated discrepancy before the quantile. A discrepancy of 0
  or 1 is replaced by `1/(2K)` or `1 - 1/(2K)` for `K` replicates. These fields
  are `nothing` when `replicates = 0`.

Under a correct model, `cwres` and `npde` are approximately independent standard
normal values.
"""
struct ResidualDiagnostics{T, E, S}
    subjects::Vector{Int}
    modes::E
    observations::Vector{Vector{T}}
    pred::Vector{Vector{T}}
    ipred::Vector{Vector{T}}
    iwres::Vector{Vector{T}}
    cpred::Vector{Vector{T}}
    cwres::Vector{Vector{T}}
    cwres_per_observation::Vector{Vector{T}}
    epred::S
    npd::S
    pde::S
    npde::S
    replicates::Int
end

"""
    residual_diagnostics(model, fit; replicates = 1000, rng = Random.default_rng(),
                         progress = nothing)
    residual_diagnostics(model, θ, σ, ω; subjects = 1:nsubjects(model), η0 = nothing,
                         replicates = 1000, rng = Random.default_rng(),
                         ebe_options = EBEOptions(), progress = nothing) -> ResidualDiagnostics

PRED, IPRED, IWRES, CWRES and NPDE for a model with
[`GaussianObservations`](@ref), at a fit's estimate or at given population
coordinates. Subject modes (EBEs) warm-start from the fit's EBEs or final chain
states, or from `η0`. Both [`NLMELayout`](@ref) and [`RaggedNLMELayout`](@ref)
are supported.

CWRES follows Hooker, Staatz and Karlsson (2007): the observations of subject
`i` are linearized around its EBE, giving mean `μ(η̂) - J η̂` and covariance
`J Ω Jᵀ + diag(v(η̂))`, and the residual is decorrelated by the Cholesky factor
of that covariance's inverse. This is the approximation the FOCE(I) objective
of [`focei_evaluate`](@ref) uses, with the residual variance at the EBE.

NPDE follows Brendel, Comets, Laffont, Laveille and Mentré (2006) and Comets,
Brendel and Mentré (2008). `replicates` datasets are simulated at the
population parameters (`η ~ N(0, Ω)`, then `y ~ N(μ(η), v(η))`), each costing
one batched [`observation_moments`](@ref) call. The observed and simulated
vectors are decorrelated with the Cholesky factor of the inverse of each
subject's simulated covariance. `replicates` must exceed every subject's number of
observations; `replicates = 0` skips the simulation.

`progress` (a Treebars node or backend symbol) gets a `Residual diagnostics` node
with the subject modes and a `Simulated replicates` counter; an interrupt raises
`Treebars.ProgressInterrupted`.
"""
function residual_diagnostics(model, fit::FOCEResult; kwargs...)
    return residual_diagnostics(model, fit.theta, fit.sigma, fit.omega;
        subjects = fit.ebes.subjects, η0 = fit.ebes, kwargs...)
end

function residual_diagnostics(model, fit::Union{SAEMResult, RaggedSAEMResult}; kwargs...)
    return residual_diagnostics(model, fit.theta, fit.sigma, fit.omega;
        subjects = fit.subjects, η0 = fit.eta, kwargs...)
end

function residual_diagnostics(model, θ::AbstractVector, σ::AbstractVector, ω::AbstractVector;
        subjects = nothing, η0 = nothing, replicates::Integer = 1000,
        rng::AbstractRNG = Random.default_rng(), ebe_options::EBEOptions = EBEOptions(),
        progress = nothing)
    return with_progress(progress; description = "Residual diagnostics") do node
        _residual_diagnostics(model, θ, σ, ω, subjects, η0, replicates, rng, ebe_options, node)
    end
end

function _residual_diagnostics(model, θ, σ, ω, subjects, η0, replicates, rng, ebe_options,
        progress)
    observation_model(model) isa GaussianObservations || throw(ArgumentError(
        "residual_diagnostics requires observation_model(model) = GaussianObservations()"))
    replicates >= 0 || throw(ArgumentError("replicates must be nonnegative"))
    layout = nlme_layout(model)
    _check_length("θ", θ, layout.ntheta)
    _check_length("σ", σ, layout.nsigma)
    _check_length("ω", ω, nomega(layout))
    ebe = _run_empirical_bayes(model, θ, σ, ω, subjects, η0, default_ebe_method(model),
        ebe_options, true, progress, false)
    ids = collect(Int, ebe.subjects)
    modes = [Vector{Float64}(_eta_column(ebe.modes, k)) for k in eachindex(ids)]
    ys = [Vector{Float64}(y) for y in observations(model, ids)]
    population = observation_moments(model, ids, θ, σ,
        _eta_batch(layout, [zero(η) for η in modes]))
    individual = observation_moments_and_jacobians(model, ids, θ, σ, ebe.modes)
    length(population) == length(individual) == length(ys) == length(ids) ||
        throw(DimensionMismatch("the observation protocol returned a different number of subjects"))
    prior = re_prior(model, ω)
    pred, ipred, iwres, cpred, cwres, per_observation = (Vector{Float64}[] for _ in 1:6)
    for (k, i) in enumerate(ids)
        y = ys[k]
        μ0, _ = population[k]
        μ, v, Jμ, _ = individual[k]
        n = length(y)
        length(μ0) == length(μ) == length(v) == n || throw(DimensionMismatch(
            "subject $i has $n observations but $(length(μ)) predicted means"))
        all(x -> isfinite(x) && x > 0, v) || throw(DomainError(v,
            "subject $i has non-positive or non-finite observation variance at its EBE"))
        linear = μ - Jμ * modes[k]
        C = Symmetric(Jμ * _subject_omega(prior, i) * Jμ' + Diagonal(v))
        push!(pred, μ0)
        push!(ipred, μ)
        push!(iwres, (y - μ) ./ sqrt.(v))
        push!(cpred, linear)
        push!(cwres, _whiten(C, y - linear))
        push!(per_observation, (y - linear) ./ sqrt.(diag(C)))
    end
    simulated = replicates == 0 ? (nothing, nothing, nothing, nothing) :
        _prediction_discrepancies(model, layout, ids, θ, σ, prior, ys, Int(replicates), rng,
            progress)
    return ResidualDiagnostics(ids, ebe.modes, ys, pred, ipred, iwres, cpred, cwres,
        per_observation, simulated..., Int(replicates))
end

# `U x` for the upper Cholesky factor U of V⁻¹ (UᵀU = V⁻¹), the decorrelation of
# the NPDE literature. With P the order reversal and P V P = L̃ L̃ᵀ, U = P L̃⁻¹ P,
# which avoids forming V⁻¹.
function _whiten(V, x)
    n = size(V, 1)
    reversal = n:-1:1
    L = cholesky(Symmetric(Matrix(V)[reversal, reversal])).L
    z = L \ (x isa AbstractVector ? x[reversal] : x[reversal, :])
    return x isa AbstractVector ? z[reversal] : z[reversal, :]
end

_subject_omega(prior::NormalREPrior, i) = Matrix(prior.cholesky_factor * prior.cholesky_factor')

function _subject_omega(prior::RaggedNormalREPrior, i)
    n = neta(prior.layout, i)
    Ω = zeros(n, n)
    for (family, r) in _ragged_subject_blocks(prior.layout, i)
        L = prior.cholesky_factors[family]
        Ω[r, r] = L * L'
    end
    return Ω
end

_draw_eta(rng, prior::NormalREPrior, i) = prior.cholesky_factor * randn(rng, neta(prior))

function _draw_eta(rng, prior::RaggedNormalREPrior, i)
    η = zeros(neta(prior.layout, i))
    for (family, r) in _ragged_subject_blocks(prior.layout, i)
        η[r] = prior.cholesky_factors[family] * randn(rng, length(r))
    end
    return η
end

# Simulated prediction discrepancies: EPRED, NPD, PDE and NPDE per subject.
function _prediction_discrepancies(model, layout, ids, θ, σ, prior, ys, K, rng,
        progress = nothing)
    K > maximum(length, ys) || throw(ArgumentError(
        "replicates = $K must exceed the largest number of observations of a subject ($(maximum(length, ys))) for the simulated covariance to be invertible"))
    simulations = [zeros(length(y), K) for y in ys]
    with_progress(progress, K; description = "Simulated replicates") do counter
        for r in 1:K
            throw_if_interrupted(counter)
            H = _eta_batch(layout, [_draw_eta(rng, prior, i) for i in ids])
            moments = observation_moments(model, ids, θ, σ, H)
            for (k, (μ, v)) in enumerate(moments)
                simulations[k][:, r] = μ .+ sqrt.(v) .* randn(rng, length(μ))
            end
            counter === nothing || update_progress!(counter, r)
        end
    end
    epred, npd, pde, npde = (Vector{Float64}[] for _ in 1:4)
    bounded(p) = clamp(p, 1 / (2K), 1 - 1 / (2K))
    for (k, i) in enumerate(ids)
        Y = simulations[k]
        y = ys[k]
        mean_y = vec(sum(Y; dims = 2)) ./ K
        D = Y .- mean_y
        V = Symmetric(D * D' ./ (K - 1))
        isposdef(V) || throw(DomainError(K,
            "subject $i: the simulated covariance of its observations is singular"))
        observed = _whiten(V, y - mean_y)
        decorrelated = _whiten(V, D)
        push!(epred, mean_y)
        push!(npd, [_normal_quantile(bounded(count(<(y[j]), view(Y, j, :)) / K))
            for j in eachindex(y)])
        p = [bounded(count(<(observed[j]), view(decorrelated, j, :)) / K) for j in eachindex(y)]
        push!(pde, p)
        push!(npde, _normal_quantile.(p))
    end
    return epred, npd, pde, npde
end

# The standard normal quantile, Wichura's Algorithm AS 241 (Applied Statistics
# 37, 1988), accurate to about 1e-16.
function _normal_quantile(p::Real)
    0 < p < 1 || throw(DomainError(p, "the normal quantile needs 0 < p < 1"))
    q = p - 0.5
    if abs(q) <= 0.425
        r = 0.180625 - q * q
        return q * (((((((2509.0809287301226727 * r + 33430.575583588128105) * r +
            67265.770927008700853) * r + 45921.953931549871457) * r +
            13731.693765509461125) * r + 1971.5909503065514427) * r +
            133.14166789178437745) * r + 3.387132872796366608) /
            (((((((5226.495278852545925 * r + 28729.085735721942674) * r +
            39307.89580009271061) * r + 21213.794301586595867) * r +
            5394.1960214247511077) * r + 687.1870074920579083) * r +
            42.313330701600911252) * r + 1.0)
    end
    r = sqrt(-log(q < 0 ? p : 1 - p))
    if r <= 5
        r -= 1.6
        value = (((((((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r +
            0.24178072517745061177) * r + 1.27045825245236838258) * r +
            3.64784832476320460504) * r + 5.7694972214606914055) * r +
            4.6303378461565452959) * r + 1.42343711074968357734) /
            (((((((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r +
            0.0151986665636164571966) * r + 0.14810397642748007459) * r +
            0.68976733498510000455) * r + 1.6763848301838038494) * r +
            2.05319162663775882187) * r + 1.0)
    else
        r -= 5
        value = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r +
            0.0012426609473880784386) * r + 0.026532189526576123093) * r +
            0.29656057182850489123) * r + 1.7848265399172913358) * r +
            5.4637849111641143699) * r + 6.6579046435011037772) /
            (((((((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r +
            1.8463183175100546818e-5) * r + 7.868691311456132591e-4) * r +
            0.0148753612908506148525) * r + 0.13692988092273580531) * r +
            0.59983220655588793769) * r + 1.0)
    end
    return q < 0 ? -value : value
end
