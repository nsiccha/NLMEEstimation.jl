using NLMEEstimation
using LinearAlgebra
using Test
import NLMEEstimation as NE
isdefined(@__MODULE__, :TestCommon) || include("common.jl")
using .TestCommon: occasions_model, occasions_unit_posterior

# The exact mode of an occasions subject: for this linear-Gaussian model, the
# posterior means of its subject effect and of its occasion effects.
function occasions_exact_mode(m, i, x)
    θ, s2, a2, b2 = x[1], exp(2x[2]), exp(2x[3]), exp(2x[4])
    counts = length.(m.y[i])
    means = [sum(o) / length(o) for o in m.y[i]]
    w = counts ./ (s2 .+ counts .* b2)
    va = inv(inv(a2) + sum(w))
    return vcat(va * sum(w .* (means .- θ)), first(occasions_unit_posterior(m, i, x)))
end

@testset "limited-memory BFGS modes of subjects with hundreds of nested effects" begin
    # Subjects of 150, 300 and 600 occasions (151 to 601 η). The subject effect's
    # curvature is hundreds of times the occasions' in prior standard deviations,
    # and the BFGS approximation keeps only `memory` pairs (at most 20 by default,
    # 3 here as well), yet every subject reaches its exact mode.
    model = occasions_model(seed = 3, units = [150, 300, 600])
    x = [2.0, log(0.25), log(0.6), log(0.4)]
    for method in (QuasiNewtonEBE(), QuasiNewtonEBE(memory = 3))
        r = empirical_bayes(model, x[1:1], x[2:2], x[3:4]; method)
        @test r.method === method
        @test all(r.converged)
        @test all(<=(EBEOptions().gtol), r.gradient_norm)
        for (k, i) in enumerate(r.subjects)
            @test r.modes[k] ≈ occasions_exact_mode(model, i, x) atol = 1e-8
        end
    end
    # refused: limited-memory BFGS stores at least one curvature pair
    @test_throws ArgumentError QuasiNewtonEBE(memory = 0)
end
