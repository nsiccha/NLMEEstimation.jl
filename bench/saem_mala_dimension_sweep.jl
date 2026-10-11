# Reproduce a dimension sweep for batched per-subject MH and adaptive MALA.
# From repository root: julia --project=test bench/saem_mala_dimension_sweep.jl
# All draws are retained; selection and held-out seeds are disjoint.
include(joinpath(@__DIR__, "saem_mala_trial.jl"))

function dimension_fixture(d::Int)
    d >= 2 || throw(ArgumentError("dimension must be at least 2"))
    rng = MersenneTwister(1700 + d)
    Ω = Matrix{Float64}(I, d, d)
    y = [0.7sin(0.17k + 0.39i) + 0.3cos(0.51k - 0.23i)
         for k in 1:d, i in 1:6]
    λ = exp.(range(log(0.5), log(20.0); length=d))
    Q = [begin
        R = Matrix(qr(randn(rng, d, d)).Q)
        Matrix(Symmetric(R * Diagonal(λ .* (0.85 + 0.05i)) * R'))
    end for i in 1:6]
    model = CorrelatedGaussianBatch(y, Q)
    prior = re_prior(model, omega_parameters(nlme_layout(model), Ω))
    truth = hcat([((Q[i] + I) \ (Q[i] * y[:, i])) for i in 1:6]...)
    return model, prior, truth
end

function configs(d)
    mh = [(scale=s * sqrt(2 / d),) for s in (0.3, 0.5, 0.8, 1.2, 1.8)]
    mala = vec([(metric=:hessian, eps=e, target=a)
        for e in (0.05, 0.1), a in (0.45, 0.574, 0.7)])
    append!(mala, [(metric=:prior, eps=0.05, target=0.574),
        (metric=:prior, eps=0.1, target=0.574)])
    return mh, mala
end

function summarize_dimension(method, cfg, seeds; steps=400, fixture)
    return summarize_fixed(method, cfg, seeds; steps, fixture)
end

function print_row(phase, d, method, cfg, s)
    println(join((phase, d, method, cfg, s.rmse, s.seconds,
        s.rmse^2 * s.seconds, s.acceptance,
        s.independence_acceptance, s.min_step, s.max_step), '\t'))
end

if abspath(PROGRAM_FILE) == @__FILE__
    BLAS.set_num_threads(1)
    println("Julia=$(VERSION) threads=$(Threads.nthreads()) BLAS_threads=$(BLAS.get_num_threads())")
    println("repo_sha=$(readchomp(`git rev-parse HEAD`)) host=$(readchomp(`hostname`))")
    println("fixture=6 independent d-dimensional Gaussian subjects; prior N(0,I); likelihood eigenvalues 0.5..20 with deterministic subject rotations; H initialized at zero; all draws retained")
    println("dimensions=2,4,8,16,32,64 selection_seeds=301:308 heldout_seeds=401:416 longheldout_seeds=701:708")
    println("phase\tdimension\tmethod\tconfig\trmse\tmedian_seconds\trmse2_times_seconds\trw_or_mala_acceptance\tprior_independence_acceptance\tmin_final_step\tmax_final_step")
    for d in (2, 4, 8, 16, 32, 64)
        payload = dimension_fixture(d)
        fixture = () -> payload
        mh_cfgs, mala_cfgs = configs(d)
        fixed_trial(:MH, first(mh_cfgs), 1; steps=2, fixture)
        fixed_trial(:MALA, first(mala_cfgs), 1; steps=2, fixture)
        mh_tune = [(cfg=c, s=summarize_dimension(:MH, c, 301:308; fixture)) for c in mh_cfgs]
        mala_tune = [(cfg=c, s=summarize_dimension(:MALA, c, 301:308; fixture)) for c in mala_cfgs]
        for (method, trials) in ((:MH, mh_tune), (:MALA, mala_tune)), x in trials
            print_row("selection", d, method, x.cfg, x.s)
        end
        mh_best = argmin(x -> x.s.rmse^2 * x.s.seconds, mh_tune)
        mala_best = argmin(x -> x.s.rmse^2 * x.s.seconds, mala_tune)
        mala_prior_best = argmin(x -> x.s.rmse^2 * x.s.seconds,
            filter(x -> x.cfg.metric == :prior, mala_tune))
        println("selected_dimension=$d MH=$(mh_best.cfg) MALA=$(mala_best.cfg) MALA_prior=$(mala_prior_best.cfg)")
        for (method, cfg) in ((:MH, mh_best.cfg), (:MALA, mala_best.cfg),
                (:MALA, mala_prior_best.cfg))
            print_row("heldout400", d, method, cfg,
                summarize_dimension(method, cfg, 401:416; fixture))
            print_row("heldout4000", d, method, cfg,
                summarize_dimension(method, cfg, 701:708; steps=4000, fixture))
        end
    end
end
