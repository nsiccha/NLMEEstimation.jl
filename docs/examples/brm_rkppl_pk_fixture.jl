# Synthetic public IV-bolus fixture shared by the comparison and SAEM benchmark.
function pk_example_data(n::Integer=8; seed::Integer=20261011)
    rng = Xoshiro(seed)
    weight = collect(range(-0.8, 0.8; length=n))
    dose = fill(100.0, n)
    t = [Float64[0.5, 1, 2, 4, 8] for _ in 1:n]
    eta_cl_true = 0.3 .* randn(rng, n)
    eta_v_true = 0.15 .* randn(rng, n)
    dv = map(1:n) do i
        cl = exp(0.2 + 0.25weight[i] + eta_cl_true[i])
        v = exp(1.6 + eta_v_true[i])
        pred = dose[i] / v .* exp.(-(cl / v) .* t[i])
        pred .+ (0.6 .+ 0.03 .* pred) .* randn(rng, length(t[i]))
    end
    return (; subject=["s$i" for i in 1:n], weight, dose, t, dv)
end

function pk_brmi(data)
    @brm data begin
        sigma ~ Exponential(1)
        log_CL ~ 1 + weight + (1 | pk | subject)
        log_V ~ 1 + (1 | pk | subject)
        effect(log_CL, Intercept) ~ Normal(0.2, 1)
        effect(log_CL, weight) ~ Normal(0, 0.5)
        effect(log_V, Intercept) ~ Normal(1.6, 1)
        sd(:, pk) ~ Exponential(1)
        @plate for i in eachindex(log_CL)
            CL = exp(log_CL[i])
            V = exp(log_V[i])
            pred[i] = dose[i] / V .* exp.(-(CL / V) .* t[i])
            dv[i] ~ normal(pred[i], sigma .+ 0.03 .* pred[i])
        end
    end
end
