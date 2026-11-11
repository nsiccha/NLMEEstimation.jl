module FitCache
using SHA, TOML
const ROOT = normpath(joinpath(@__DIR__, "../.."))
const RESULT_DIR = joinpath(ROOT, "docs/results")
function inputs()
    paths = ["Project.toml", "docs/Project.toml", "research/docs/pk_fit.jl", "research/docs/pk_diagnostics.jl",
        "research/docs/generate.jl", "research/docs/cache.jl"]
    append!(paths, sort([joinpath("src", f) for f in readdir(joinpath(ROOT, "src")) if endswith(f, ".jl")]))
    return paths
end
function key()
    io = IOBuffer()
    for path in inputs()
        write(io, path, UInt8(0), read(joinpath(ROOT, path)), UInt8(0))
    end
    return bytes2hex(sha256(take!(io)))
end
path() = joinpath(RESULT_DIR, "pk-$(key()).toml")
function load()
    file = path()
    isfile(file) || error("The PK fit receipt is missing or stale. Run julia --project=docs research/docs/generate.jl locally; docs builds never run fits.")
    cache = TOML.parsefile(file)
    cache["receipt"]["key"] == key() || error("Fit receipt key mismatch")
    return cache
end
end
