using NLMEEstimation
using LinearAlgebra
using Random
using Test
import ForwardDiff
import NLMEEstimation as NE
import Optim
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon: occasions_model, occasions_sparse, poisson_model, poisson_sparse,
    poisson_exact, occasions_exact, occasions_exact_subject, LaplaceHooked

# The exact maximum of a Poisson nested model's marginal, and minus its Hessian
# there, through the fixed-node quadrature (re-centred at each outer step).
function poisson_maximum(m, x)
    for _ in 1:4
        f = poisson_exact(m, x)
        x = Optim.minimizer(Optim.optimize(y -> -f(y), y -> -ForwardDiff.gradient(f, y), x,
            Optim.LBFGS(); inplace = false))
    end
    return x, -ForwardDiff.hessian(poisson_exact(m, x), x)
end

@testset "Laplace standard errors on non-Gaussian nested posteriors" begin
    # Four subjects of 10–80 units with two to four counts each, at mean counts
    # near 1.6 (28% zeros) and 0.4. The Laplace marginal is not exact here, but
    # its curvature is close to the exact one: SEs measured within 1.5% and
    # 2.6%. Louis's identity needs no curvature and no population gradient (θ is
    # mu-referenced and there is no σ), but its unit-variance element nearly
    # cancels: the curvature and score variance of each weakly informed unit are
    # close, and their Monte Carlo errors add up. At θ = -1.5 that element of R
    # was off by up to a factor 4 and indefinite for 2 of 4 seeds at 1,000 draws,
    # with the 0.1.x and the 0.2.0 NUTS kernel alike; at 3,000 draws it stayed
    # positive definite for every seed tried with the 0.2.0 kernel, and θ's SE,
    # the one checked here, was within 3%.
    for θ in (0.0, -1.5)
        m = poisson_model(; θ)
        x, exact_R = poisson_maximum(m, [θ, log(0.5), log(0.5)])
        exact_se = sqrt.(diag(inv(exact_R)))
        for model in (m, poisson_sparse(m))
            laplace = covariance_step(model, x[1:1], Float64[], x[2:3]; method = :r,
                approximation = AdaptiveGaussHermite(nodes = 1))
            @test laplace.standard_errors ≈ exact_se rtol = 0.04
        end
        louis = covariance_step(m, x[1:1], Float64[], x[2:3]; method = :r,
            approximation = LouisInformation(draws = 3000), rng = Xoshiro(1))
        @test louis.standard_errors[1] ≈ exact_se[1] rtol = 0.05
    end
end

@testset "ragged SAEM fits: the default covariance approximation" begin
    # Decision 0hwegve: Laplace where the model reports η Hessians (a declared
    # pattern or its own blocks, as for default_ebe_method), Louis's identity
    # where it does not. Importance sampling, the earlier default, loses
    # effective draws as subjects grow.
    occasions = occasions_model()
    fit = saem(occasions, [1.9], [log(0.3)], [log(0.5), log(0.4)];
        rng = Xoshiro(5), options = SAEMOptions(iterations = 120, burnin = 60))
    @test covariance_step(occasions, fit; method = :s, rng = Xoshiro(1)).approximation isa
        LouisInformation
    @test covariance_step(occasions_sparse(occasions), fit; method = :s).approximation ==
        AdaptiveGaussHermite(nodes = 1)
    # A model without η Hessians that supplies its Laplace gradient takes Laplace
    # through it, starting each mode search from the fit's final chain states.
    hooked = LaplaceHooked(occasions, occasions_exact_subject)
    @test covariance_step(hooked, fit; method = :s).approximation ==
        AdaptiveGaussHermite(nodes = 1)
    @test length(hooked.starts) == 2 * 4 + 1
    @test all(==(fit.eta), hooked.starts)
    # A model with only the η gradient, fitted end to end: its default needs no
    # population gradient.
    poisson = poisson_model()
    fit = saem(poisson, [0.0], Float64[], [log(0.5), log(0.5)];
        rng = Xoshiro(2), options = SAEMOptions(iterations = 120, burnin = 60))
    default = covariance_step(poisson, fit; rng = Xoshiro(3))
    @test default.approximation isa LouisInformation
    @test all(isfinite, default.standard_errors)
end

@testset "ragged covariance step: the model's own Laplace gradient" begin
    # Gaussian nested intercepts, where Laplace is exact: the hook's closed-form
    # marginals equal the package's Laplace, and R its exact curvature.
    occasions = occasions_model()
    model = LaplaceHooked(occasions, occasions_exact_subject)
    x = [2.0, log(0.25), log(0.6), log(0.4)]
    @test check_protocol(model, x[1:1], x[2:2], x[3:4])
    @test_throws ProtocolViolation check_protocol(
        LaplaceHooked(occasions, occasions_exact_subject; scale = 1.001), x[1:1], x[2:2], x[3:4])
    empty!(model.starts)
    result = covariance_step(model, x[1:1], x[2:2], x[3:4];
        approximation = AdaptiveGaussHermite(nodes = 1), method = :r)
    @test result.evaluations == 2 * 4 + 1
    @test all(==([zeros(1 + length(o)) for o in occasions.y]), model.starts)
    # The O(h²) error of differencing the gradient, measured 1.2e-6, is largest in σ.
    @test result.R ≈ -ForwardDiff.hessian(y -> occasions_exact(occasions, y), x) rtol = 1e-5
end

@testset "ragged marginal likelihood: the default, and Laplace's bias" begin
    # Decision 1jf8d4p: on a ragged layout whose model reports η Hessians the
    # marginal defaults to Laplace, as the covariance step does; elsewhere to
    # importance sampling.
    occasions = occasions_model()
    x = [2.0, log(0.25), log(0.6), log(0.4)]
    @test marginal_loglikelihood(occasions, x[1:1], x[2:2], x[3:4];
        rng = Xoshiro(1)).approximation isa ImportanceSampling
    @test marginal_loglikelihood(occasions_sparse(occasions), x[1:1], x[2:2], x[3:4]).approximation ==
        AdaptiveGaussHermite(nodes = 1)
    # On non-Gaussian counts Laplace's marginal is biased, by an amount that adds
    # up over the units (measured −0.22 here, −2.19 at 1,260 units), while its
    # curvature stays accurate (the first testset).
    m = poisson_model()
    x, _ = poisson_maximum(m, [0.0, log(0.5), log(0.5)])
    laplace = marginal_loglikelihood(poisson_sparse(m), x[1:1], Float64[], x[2:3])
    @test laplace.approximation == AdaptiveGaussHermite(nodes = 1)
    @test -0.4 < laplace.loglikelihood - poisson_exact(m, x)(x) < -0.1
end
