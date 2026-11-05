# The environment of `saem_threads_vs_lockstep_brm.jl`: BRM's own test pins (its
# `test/setup_env.jl`) with BRM and this checkout developed beside them.
using Pkg
const repo, env, bootstrap, nlme = ENV["BRM_REPO"], ENV["BRM_ENV"], ENV["BRM_BOOTSTRAP_ROOT"], ENV["NLME_REPO"]
include(joinpath(repo, "test/dependency_floors.jl"))
mkpath(env)
cp(joinpath(repo, "test/Project.toml"), joinpath(env, "Project.toml"); force = true)
const mirrors = joinpath(homedir(), ".local/state/kb-agents/git-mirrors")
pins = Tuple{String, String, String}[]
for line in eachline(joinpath(repo, "test/setup_env.jl"))
    m = match(r"^\s*\(\"([A-Za-z]+)\",\s*\"([^\"]+)\",\s*\"([0-9a-f]{40})\"\)", line)
    m === nothing || push!(pins, (m[1], m[2], m[3]))
end
@assert length(pins) >= 8 pins
println("PINS ", join(("$n=$(r[1:8])" for (n, _, r) in pins), " "))
paths = Dict{String, String}("BayesianRegressionModels" => repo, "NLMEEstimation" => nlme)
for (name, url, rev) in pins
    mirror = joinpath(mirrors, name * ".jl.git")
    paths[name] = resolve_git_revision_checkout(name, rev; cache_root = bootstrap,
        mirror = isdir(mirror) ? mirror : "", origin = url)
end
paths["EnzymeCore"] = joinpath(paths["Enzyme"], "lib/EnzymeCore")
for n in ("ReactiveKernelsPPL", "ReactiveKernelsPPLExamples", "ReactiveKernelsDistributionKernels")
    paths[n] = joinpath(paths["ReactiveKernels"], "packages", n)
end
Pkg.activate(env)
Pkg.develop([PackageSpec(path = p) for (_, p) in sort!(collect(paths))])
Pkg.instantiate(; allow_autoprecomp = false)
lib = joinpath(homedir(), "github/nsiccha/Claude/lib-resolve.sh")
step = read(Cmd(["bash", "-c", "source \"\$1\" >/dev/null && dogfood_libenzyme_program resolve", "_", lib]), String)
include_string(Main, step, "dogfood_libenzyme")
Pkg.precompile(["BayesianRegressionModels", "ReactiveKernelsPPL", "NLMEEstimation", "Enzyme"])
for name in ("BayesianRegressionModels", "ReactiveKernels", "NLMEEstimation", "Enzyme", "WarmupHMC")
    for (_, info) in Pkg.dependencies()
        info.name == name && println("ENV_PKG ", name, " version=", info.version, " source=", info.source)
    end
end
println("ENV_READY")
