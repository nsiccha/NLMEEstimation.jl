using NLMEEstimation
using Markdown
using Test
import NLMEEstimation as NE

# Every `[`name`](@ref)` in the package's docstrings must parse as a Markdown link
# and name a documented binding. Julia's Markdown parser can leave a link as raw
# text (a stray `_` earlier in the paragraph, as in "η̂_i", pairs with the link's
# underscores as emphasis); Documenter then never sees the reference, and the
# Docs build fails on a dead `./@ref` link in api.md with no Documenter warning
# (Docs run 40862419235 on e466614). A detached docstring breaks the build the
# same way (run 38228011497).
function _doc_links!(links, raw, x, owner)
    x isa Markdown.Link && push!(links, (owner, x))
    x isa AbstractString && occursin("](@ref)", x) && push!(raw, (owner, x))
    for field in (:content, :text, :items)
        hasproperty(x, field) || continue
        value = getproperty(x, field)
        value isa AbstractVector ? foreach(c -> _doc_links!(links, raw, c, owner), value) :
            _doc_links!(links, raw, value, owner)
    end
    x isa AbstractVector && foreach(c -> _doc_links!(links, raw, c, owner), x)
    return nothing
end

@testset "docstring cross-references parse and resolve" begin
    modules = (NE, NE.ReferenceModels)
    links, raw = Tuple{Symbol, Markdown.Link}[], Tuple{Symbol, String}[]
    for mod in modules, (binding, multi) in Docs.meta(mod), (_, doc) in multi.docs
        _doc_links!(links, raw, Markdown.parse(join(string.(doc.text))), binding.var)
    end
    documented = Set(binding.var for mod in modules for binding in keys(Docs.meta(mod)))
    refs = [(owner, link) for (owner, link) in links if link.url == "@ref"]
    @test length(refs) > 100
    @test isempty(raw)
    unresolved = [(owner, Markdown.plain(Markdown.Paragraph(link.text))) for (owner, link) in refs
        if Symbol(strip(replace(Markdown.plain(Markdown.Paragraph(link.text)), "`" => ""))) ∉ documented]
    @test isempty(unresolved)
end
