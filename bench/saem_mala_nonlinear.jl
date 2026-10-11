# Fixed-target MH versus adaptive MALA on a nonlinear repeated-dose NLME model.
# From repository root: julia --project=test bench/saem_mala_nonlinear.jl
# Each endpoint has four random effects: baseline, log Emax, log EC50, log Hill.
include(joinpath(@__DIR__, "saem_mala_trial.jl"))
using Optim
using ForwardDiff

struct HillBatch
    doses::Vector{Float64}
    y::Array{Float64,3}             # dose, endpoint, subject
    base::Matrix{Float64}           # parameter, endpoint
    sd::Vector{Float64}
    omega::Matrix{Float64}
    noise::Float64
end
NE.nsubjects(m::HillBatch) = size(m.y, 3)
NE.nlme_layout(m::HillBatch) = NLMELayout(0, 0, [length(m.sd)])
NE.evaluation_style(::HillBatch) = BatchedEvaluation()
NE.provides_eta_hessian(::HillBatch) = true

function hill_subject(m::HillBatch, i, η; derivatives=false, hessian=false)
    # Avoid overflow in stress proposals. This truncates only beyond 40 prior
    # standard deviations, where posterior mass is negligible here.
    if any(x -> !isfinite(x) || abs(x) > 10, η)
        return -Inf, derivatives ? zeros(length(η)) : nothing,
            hessian ? zeros(length(η), length(η)) : nothing
    end
    ll = 0.0
    g = derivatives ? zeros(length(η)) : nothing
    H = hessian ? zeros(length(η), length(η)) : nothing
    invvar = inv(m.noise^2)
    for e in axes(m.y, 2)
        k = 4*e - 3
        baseline = m.base[1, e] + η[k]
        emax = exp(m.base[2, e] + η[k+1])
        logec50 = m.base[3, e] + η[k+2]
        hill = exp(m.base[4, e] + η[k+3])
        for (r, dose) in enumerate(m.doses)
            u = hill * (log(dose) - logec50)
            s = inv(1 + exp(-u))
            a = s * (1 - s)
            pred = baseline + emax * s
            residual = m.y[r, e, i] - pred
            ll -= 0.5 * residual^2 * invvar
            derivatives || continue
            J = (1.0, emax*s, -emax*a*hill, emax*a*u)
            for p in 1:4
                g[k+p-1] += residual * J[p] * invvar
            end
            hessian || continue
            b = a * (1 - 2s)
            T = zeros(4, 4)
            T[2, 2] = emax*s
            T[2, 3] = T[3, 2] = -emax*a*hill
            T[2, 4] = T[4, 2] = emax*a*u
            T[3, 3] = emax*b*hill^2
            T[3, 4] = T[4, 3] = -emax*(b*hill*u + a*hill)
            T[4, 4] = emax*(b*u^2 + a*u)
            for p in 1:4, q in 1:4
                H[k+p-1, k+q-1] += (residual*T[p, q] - J[p]*J[q])*invvar
            end
        end
    end
    return ll, g, H
end

NE.conditional_loglikelihoods(m::HillBatch, ids, θ, σ, H) =
    [first(hill_subject(m, i, view(H, :, j))) for (j, i) in enumerate(ids)]
function NE.conditional_loglikelihoods_and_gradients(m::HillBatch, ids, θ, σ, H)
    values = zeros(length(ids))
    G = similar(H)
    for (j, i) in enumerate(ids)
        values[j], g, _ = hill_subject(m, i, view(H, :, j); derivatives=true)
        G[:, j] = g
    end
    return values, G
end
NE.conditional_loglikelihood_gradient_and_hessian(m::HillBatch, i, θ, σ, η) =
    hill_subject(m, i, η; derivatives=true, hessian=true)

function nonlinear_model(endpoints; noise=0.12)
    endpoints in (1, 2) || throw(ArgumentError("one or two endpoints"))
    rng = MersenneTwister(1107)
    doses = collect(exp.(range(log(0.08), log(20.0); length=12)))
    base = [0.12 -0.10; log(1.1) log(0.8); log(1.0) log(2.8);
        log(1.25) log(0.85)][:, 1:endpoints]
    sd = repeat([0.15, 0.25, 0.25, 0.18], endpoints)
    correlation = fill(0.2, length(sd), length(sd))
    correlation[diagind(correlation)] .= 1.0
    omega = Matrix(Diagonal(sd) * correlation * Diagonal(sd))
    y = zeros(length(doses), endpoints, 6)
    m = HillBatch(doses, y, base, sd, omega, noise)
    for i in 1:6
        η = cholesky(Symmetric(omega)).L * randn(rng, length(sd))
        for e in 1:endpoints
            k = 4*e - 3
            b = base[1, e] + η[k]
            E = exp(base[2, e] + η[k+1])
            c = base[3, e] + η[k+2]
            h = exp(base[4, e] + η[k+3])
            for (r, x) in enumerate(doses)
                y[r, e, i] = b + E / (1 + exp(-h*(log(x)-c))) +
                    noise*randn(rng)
            end
        end
    end
    return m
end

# Independent importance references use a full subject-level Laplace proposal,
# widened by 1.5. A second estimate checks reference Monte Carlo error.
function subject_reference(m::HillBatch, i, seed; draws=100_000)
    d = length(m.sd)
    precision = inv(m.omega)
    function posterior(x; deriv=false)
        ℓ, g, H = hill_subject(m, i, x; derivatives=deriv, hessian=deriv)
        ℓ -= 0.5dot(x, precision*x)
        deriv || return ℓ
        return ℓ, g - precision*x, H - precision
    end
    objective(x) = -posterior(x)
    gradient!(G, x) = (G[:] = -posterior(x; deriv=true)[2])
    opt = Optim.optimize(objective, gradient!, zeros(d), Optim.LBFGS(),
        Optim.Options(g_tol=1e-8, iterations=2000))
    Optim.converged(opt) || error("reference mode search: $opt")
    mode = Optim.minimizer(opt)
    _, gmode, H = posterior(mode; deriv=true)
    maximum(abs, gmode) < 1e-4 || error("reference mode gradient: $gmode")
    curvature = eigen(Symmetric(-H))
    Σ = curvature.vectors * Diagonal(1 ./ max.(curvature.values, 1e-3)) * curvature.vectors'
    L = cholesky(Symmetric(Σ)).L * 1.5
    rng = MersenneTwister(seed)
    xs = Matrix{Float64}(undef, d, draws)
    lw = Vector{Float64}(undef, draws)
    for j in 1:draws
        z = randn(rng, d)
        x = mode + L*z
        xs[:, j] = x
        lw[j] = posterior(x) + 0.5sum(abs2, z)
    end
    w = exp.(lw .- maximum(lw))
    w ./= sum(w)
    return xs*w, inv(sum(abs2, w))
end

function nonlinear_fixture(endpoints; reference_draws=100_000)
    m = nonlinear_model(endpoints)
    prior = re_prior(m, omega_parameters(nlme_layout(m), m.omega))
    truth = zeros(length(m.sd), 6)
    checks = Float64[]
    esses = Float64[]
    for i in 1:6
        a, ea = subject_reference(m, i, 9000+100i; draws=reference_draws)
        b, eb = subject_reference(m, i, 10000+100i; draws=reference_draws)
        truth[:, i] = (a+b)/2
        append!(checks, abs.(a-b))
        append!(esses, (ea, eb))
    end
    return (m, prior, truth), (maximum(checks), minimum(esses))
end

function configs_nonlinear(d)
    mh = [(scale=s*sqrt(2/d),) for s in (0.3, 0.5, 0.8, 1.2, 1.8)]
    mala = [(metric=metric, eps=eps, target=target)
        for metric in (:prior, :hessian), eps in (0.05, 0.1, 0.2),
            target in (0.574, 0.7)]
    return mh, vec(mala)
end

function print_nonlinear(phase, d, method, cfg, s)
    println(join((phase, d, method, cfg, s.rmse, s.seconds,
        s.rmse^2*s.seconds, s.acceptance, s.independence_acceptance,
        s.min_step, s.max_step), '\t'))
end

function check_differentials()
    m = nonlinear_model(2)
    η = 0.1 .* collect(1:8)
    _, g, H = hill_subject(m, 1, η; derivatives=true, hessian=true)
    f(x) = first(hill_subject(m, 1, x))
    maximum(abs, g - ForwardDiff.gradient(f, η)) < 1e-8 ||
        error("analytic nonlinear gradient disagrees with ForwardDiff")
    maximum(abs, H - ForwardDiff.hessian(f, η)) < 1e-8 ||
        error("analytic nonlinear Hessian disagrees with ForwardDiff")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    BLAS.set_num_threads(1)
    check_differentials()
    println("Julia=$(VERSION) threads=$(Threads.nthreads()) BLAS_threads=$(BLAS.get_num_threads())")
    println("repo_sha=$(readchomp(`git rev-parse HEAD`)) host=$(readchomp(`hostname`))")
    println("fixture=6 subjects; 12 doses 0.08..20 per endpoint; 4 random effects per Hill Emax endpoint; 1 or 2 endpoints; observation SD 0.12; Gaussian prior with pairwise correlation 0.2; all draws retained")
    println("reference=two independent 100000-draw full-subject Laplace-proposal importance estimates; proposal covariance inflated 1.5^2")
    println("selection_seeds=301:308 heldout_seeds=401:416 longheldout_seeds=701:708")
    println("phase\tdimension\tmethod\tconfig\trmse\tmedian_seconds\trmse2_times_seconds\trw_or_mala_acceptance\tprior_independence_acceptance\tmin_final_step\tmax_final_step")
    for endpoints in (1, 2)
        payload, check = nonlinear_fixture(endpoints)
        d = 4endpoints
        check[1] < 0.005 || error("importance references disagree at dimension $d")
        check[2] > 10_000 || error("importance reference ESS is too low at dimension $d")
        println("reference_dimension=$d max_pair_difference=$(check[1]) min_importance_ess=$(check[2])")
        fixture = () -> payload
        mh_cfgs, mala_cfgs = configs_nonlinear(d)
        fixed_trial(:MH, first(mh_cfgs), 1; steps=2, fixture)
        fixed_trial(:MALA, first(mala_cfgs), 1; steps=2, fixture)
        mh_tune = [(cfg=c, s=summarize_fixed(:MH, c, 301:308; fixture)) for c in mh_cfgs]
        mala_tune = [(cfg=c, s=summarize_fixed(:MALA, c, 301:308; fixture)) for c in mala_cfgs]
        for (method, trials) in ((:MH, mh_tune), (:MALA, mala_tune)), x in trials
            print_nonlinear("selection", d, method, x.cfg, x.s)
        end
        mh_best = argmin(x -> x.s.rmse^2*x.s.seconds, mh_tune)
        mala_best = argmin(x -> x.s.rmse^2*x.s.seconds, mala_tune)
        println("selected_dimension=$d MH=$(mh_best.cfg) MALA=$(mala_best.cfg)")
        for (method, cfg) in ((:MH, mh_best.cfg), (:MALA, mala_best.cfg))
            print_nonlinear("heldout400", d, method, cfg,
                summarize_fixed(method, cfg, 401:416; fixture))
            print_nonlinear("heldout4000", d, method, cfg,
                summarize_fixed(method, cfg, 701:708; steps=4000, fixture))
        end
    end
end
