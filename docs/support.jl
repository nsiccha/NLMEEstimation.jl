module DocsSupport
using Markdown, NLMEEstimation, NLMEEstimation.ReferenceModels, ADTypes, Enzyme
using AlgebraOfVega
include(joinpath(@__DIR__, "../research/docs/cache.jl"))
cache() = FitCache.load()
function run(name)
    result = cache()
    source = read(joinpath(FitCache.ROOT, "research/docs/$name.jl"), String)
    return Markdown.MD([Markdown.Code("julia", strip(source)),
        Markdown.Code("text", strip(result["outputs"][name]))])
end
function model()
    f = cache()["fit"]
    subjects = [PKSubject(f["doses"][i], f["weights"][i], t, f["observed"][i])
        for (i,t) in enumerate(f["times"])]
    return OneCompartmentOral(subjects; backend=AutoEnzyme(), error_model=AdditiveError())
end
function receipt()
    r = cache()["receipt"]
    return Markdown.parse("""
    | Receipt | Value |
    |:--|:--|
    | Content SHA-256 | `$(r["key"])` |
    | Package source revision | `$(r["source_revision"])` |
    | Julia | $(r["julia"]) |
    | Generated (UTC) | $(r["created_utc"]) |
    | Fit and diagnostics (s) | $(round(r["elapsed_seconds"]; digits=2)) |

    The content digest covers the package sources, project, generator and both
    displayed scripts. The receipt contains every resolved dependency version.
    The source revision identifies the base commit; `source_dirty = $(r["source_dirty"])`
    records whether this run included edits. The digest identifies those exact inputs.
    """)
end
function figure(kind)
    f = cache()["fit"]
    if kind == :predictions
        rows = (; time=f["times"][1], observed=f["observed"][1],
            population=f["pred"][1], individual=f["ipred"][1])
        spec = data(rows) * (
            mapping(:time => "Time after dose (h)", :observed => "Concentration (mg/L)") * visual(Scatter; color=:ink) +
            mapping(:time => "Time after dose (h)", :population => "Concentration (mg/L)") * visual(Lines; color="#2a78d6") +
            mapping(:time => "Time after dose (h)", :individual => "Concentration (mg/L)") * visual(Lines; color="#eb6834"))
        caption = "Subject 1: observations (points), population prediction (blue), and individual prediction (orange). The individual curve uses the fitted subject mode."
    elseif kind == :residuals
        rows = (; time=reduce(vcat,f["times"]), residual=reduce(vcat,f["iwres"]))
        spec = data(rows) * mapping(:time => "Time after dose (h)", :residual => "IWRES (unitless)") * visual(Scatter; color="#2a78d6", alpha=0.65)
        caption = "Individual weighted residuals for all subjects. Look for a time-dependent trend or changing spread; this simulated example does not establish adequacy for clinical data."
    else
        error("Unknown figure: $kind")
    end
    spec *= config(height=260)
    head = join(repr(MIME"text/html"(), node) for node in vega_head(; source=:inline))
    plot = repr(MIME"text/html"(), vdraw(spec))
    file = "pk-$(kind).html"
    mkpath("public")
    write(joinpath("public", file), """<!doctype html><html lang="en"><head><meta charset="utf-8">
    <style>html{color:#0b0b0b;background:var(--docs-background,#fff);font:14px system-ui,sans-serif}html.dark{color:#e5e7eb}body{margin:8px}</style>
    $head</head><body>$plot</body></html>""")
    return Base.HTML("""<figure><iframe class="docs-figure" data-docs-figure src="$file" title="$caption" loading="eager"></iframe><figcaption>$caption</figcaption></figure>""")
end
end
