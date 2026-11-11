# Explicit local generation only; the docs build imports cache.jl, never this file.
get(ENV, "CI", "false") == "true" && error("Documentation fits must not run in CI")
include("cache.jl")
using .FitCache, TOML, Dates, Pkg
if isfile(FitCache.path())
    FitCache.load()
    println("Cached fit already exists: ", FitCache.path())
    exit()
end
run = Module(:PKRun)
outputs = Dict{String,Any}()
elapsed = @elapsed for name in ("pk_fit", "pk_diagnostics")
    tmp, io = mktemp()
    try
        redirect_stdout(io) do
            Base.include(run, joinpath(@__DIR__, "$name.jl"))
        end
        close(io)
        outputs[name] = read(tmp, String)
    finally
        close(io)
        rm(tmp; force=true)
    end
end
fit, model, diag = run.fit, run.model, run.diagnostics
@assert fit.converged "Population optimization did not converge"
@assert all(fit.ebes.converged) "Some subject modes did not converge"
summary = Dict("theta" => fit.theta, "sigma" => fit.sigma, "omega" => fit.omega,
    "converged" => fit.converged, "evaluations" => fit.evaluations,
    "ofv" => fit.ofv, "loglikelihood" => fit.loglikelihood,
    "mode_gradient_max" => maximum(fit.ebes.gradient_norm),
    "times" => [s.times for s in model.subjects],
    "doses" => [s.dose for s in model.subjects], "weights" => [s.weight for s in model.subjects],
    "observed" => diag.observations, "pred" => diag.pred, "ipred" => diag.ipred,
    "iwres" => diag.iwres, "cwres" => diag.cwres,
    "modes" => [collect(fit.ebes.modes[:,i]) for i in axes(fit.ebes.modes,2)])
deps = Dict(info.name => string(info.version) for info in values(Pkg.dependencies()) if info.version !== nothing)
receipt = Dict("key" => FitCache.key(), "julia" => string(VERSION),
    "created_utc" => string(now(UTC)), "elapsed_seconds" => elapsed,
    "source_revision" => strip(read(`git -C $(FitCache.ROOT) rev-parse HEAD`, String)),
    "source_dirty" => !isempty(read(`git -C $(FitCache.ROOT) status --porcelain`, String)),
    "inputs" => FitCache.inputs(), "dependencies" => deps)
mkpath(FitCache.RESULT_DIR)
open(FitCache.path(), "w") do io
    TOML.print(io, Dict("receipt" => receipt, "outputs" => outputs, "fit" => summary); sorted=true)
end
println(outputs["pk_fit"], outputs["pk_diagnostics"])
println("Fit receipt: ", FitCache.path())
