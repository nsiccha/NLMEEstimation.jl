using Documenter, DocumenterVitepress, NLMEEstimation
include("support.jl")
include("lint.jl")

pages = [
    "Start" => ["Introduction" => "index.md"],
    "Guides" => ["Model protocol" => "protocol.md", "Random effects" => "random-effects.md",
        "Estimation" => "fitting.md", "Diagnostics" => "diagnostics.md"],
    "Examples" => ["Oral pharmacokinetics" => "oral-pk.md"],
    "Reference" => ["Protocol" => "reference/protocol.md", "Random effects" => "reference/random-effects.md",
        "Estimation" => "reference/estimation.md", "Diagnostics" => "reference/diagnostics.md",
        "Models" => "reference/models.md"],
]
lint_docs(pages)
DocsSupport.cache() # Refuse a missing/stale receipt; never run a fit from the build.
makedocs(
    sitename = "NLMEEstimation.jl",
    modules = [NLMEEstimation, NLMEEstimation.ReferenceModels],
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/nsiccha/NLMEEstimation.jl", devurl = "dev", devbranch = "master"),
    pages = pages,
    checkdocs = :exports,
    warnonly = false,
)
let redirect = joinpath(@__DIR__, "build", "index.html")
    isfile(redirect) || write(redirect, """
    <!DOCTYPE html><html><head><meta http-equiv="refresh" content="0; url=dev/"></head>
    <body><a href="dev/">Documentation</a></body></html>
    """)
end
DocumenterVitepress.deploydocs(repo = "github.com/nsiccha/NLMEEstimation.jl",
    devbranch = "master", push_preview = true)
