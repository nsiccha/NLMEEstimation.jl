using Documenter, DocumenterVitepress, NLMEEstimation
include("support.jl")
include("lint.jl")

pages = include("pages.jl")
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
