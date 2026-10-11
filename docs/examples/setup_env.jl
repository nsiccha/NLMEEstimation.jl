# Run from the repository root with:
# julia --project=docs/examples docs/examples/setup_env.jl /path/to/empty/deps-dir
# Julia 1.10 ignores Project.toml [sources], so develop all source packages together.
using Pkg

length(ARGS) == 1 || error("Pass one directory for the pinned dependency clones")
root = abspath(ARGS[1])
mkpath(root)

sources = [
    ("BayesianRegressionModels", "https://github.com/nsiccha/BayesianRegressionModels.jl.git", "35c2258fc34d5e32f90347ef669dad451f3b3862"),
    ("NLMEEstimation", "https://github.com/nsiccha/NLMEEstimation.jl.git", "527a593c0992ba2fb48d2727799bdb4fc39d247c"),
    ("ReactiveKernels", "https://github.com/nsiccha/ReactiveKernels.jl.git", "a2268e47b3a800df356ae6961c2caf863922a102"),
    ("WarmupHMC", "https://github.com/nsiccha/WarmupHMC.jl.git", "6cd665de7591dc3024893106f64e72f9e90ad49c"),
    ("StanBlocks", "https://github.com/nsiccha/StanBlocks.jl.git", "75d9feb41f6f9148799c4c1f3236aacd5bdfdf7c"),
    ("Treebars", "https://github.com/nsiccha/Treebars.jl.git", "7426b9c8dc914d767c28ff22f1f75631f3afc41d"),
]

for (name, url, sha) in sources
    dir = joinpath(root, name)
    if !ispath(dir)
        run(`git clone --quiet --no-checkout $url $dir`)
        run(`git -C $dir fetch --quiet origin $sha`)
        run(`git -C $dir checkout --quiet --detach $sha`)
    else
        isdir(joinpath(dir, ".git")) || error("$dir exists but is not a git checkout")
        got = readchomp(`git -C $dir rev-parse HEAD`)
        got == sha || error("$dir is at $got; expected $sha. Choose another dependency directory.")
        isempty(readchomp(`git -C $dir status --porcelain`)) || error("$dir has local changes")
    end
end

rk = joinpath(root, "ReactiveKernels")
paths = [
    joinpath(root, "BayesianRegressionModels"),
    joinpath(root, "NLMEEstimation"),
    rk,
    joinpath(rk, "packages", "ReactiveKernelsDistributionKernels"),
    joinpath(rk, "packages", "ReactiveKernelsPPL"),
    joinpath(root, "StanBlocks"),
    joinpath(root, "Treebars"),
    joinpath(root, "WarmupHMC"),
]

Pkg.activate(@__DIR__)
Pkg.develop(Pkg.PackageSpec[Pkg.PackageSpec(path=path) for path in paths])
Pkg.instantiate()
