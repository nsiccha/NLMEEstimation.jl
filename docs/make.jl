using Documenter, DocumenterVitepress, NLMEEstimation

makedocs(
    sitename = "NLMEEstimation.jl",
    modules = [NLMEEstimation, NLMEEstimation.ReferenceModels],
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/nsiccha/NLMEEstimation.jl",
        devurl = "dev",
        devbranch = "master",
    ),
    pages = [
        "Home" => "index.md",
        "Model protocol" => "protocol.md",
        "PPL and AD requirements" => "requirements.md",
        "Estimators" => "estimators.md",
        "PK comparison" => "pk-comparison.md",
        "SAEM benchmarks" => "saem-benchmarks.md",
        "API" => "api.md",
    ],
    checkdocs = :none,
    warnonly = true,
)

let redirect = joinpath(@__DIR__, "build", "index.html")
    isfile(redirect) || write(redirect, """
    <!DOCTYPE html>
    <html><head><meta http-equiv="refresh" content="0; url=dev/"></head>
    <body>Redirecting to <a href="dev/">dev</a>...</body></html>
    """)
end

DocumenterVitepress.deploydocs(
    repo = "github.com/nsiccha/NLMEEstimation.jl",
    devbranch = "master",
    push_preview = true,
)
