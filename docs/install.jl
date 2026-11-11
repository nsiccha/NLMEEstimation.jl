# Run from a checkout with Julia 1.11 or later. Sources are declared in Project.toml.
VERSION >= v"1.11" || error("This installer requires Julia 1.11+ for [sources] support. The package itself supports Julia 1.10.")
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()
