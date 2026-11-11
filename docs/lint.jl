# Local documentation checks until DocsKit provides the shared lint.
function lint_docs(pages)
    first.(pages) == ["Start", "Guides", "Examples", "Reference"] || error("Docs navigation must have four ordered groups")
    listed = String[]
    for (_, entries) in pages, (_, file) in entries
        push!(listed, file)
    end
    actual = String[]
    for (root, _, files) in walkdir(joinpath(@__DIR__, "src")), file in files
        endswith(file, ".md") && push!(actual, relpath(joinpath(root,file), joinpath(@__DIR__, "src")))
    end
    sort(listed) == sort(actual) || error("Every Markdown page must have exactly one navigation home")
    banned = r"\b(seamless|robust|powerful|leverage|delve|crucial|comprehensive|unlock|simply)\b|it's worth noting"i
    for (group, entries) in pages, (_, file) in entries
        content = read(joinpath(@__DIR__, "src", file), String)
        occursin(banned, content) && error("Banned prose in $file")
        occursin(r"(?m)^```(?:julia|python|r|javascript|js|toml|text)\s*$", content) && error("Static runnable/output fence in $file")
        words = length(split(replace(content, r"```[\s\S]*?```" => "")))
        group == "Guides" && words > 1500 && error("Guide exceeds 1500 words: $file ($words)")
        if group == "Examples"
            800 <= words <= 1500 || error("Example outside 800–1500 words: $file ($words)")
            occursin("```@example", content) && occursin("DocsSupport.figure", content) || error("Example needs executed code and figure: $file")
        end
        println("DOCS-LINT\t$group\t$file\t$words prose words")
    end
    tracked = split(read(`git -C $(@__DIR__) ls-files`, String), '\n')
    any(f -> occursin(r"\.(png|svg|pdf|webm|mp4)$"i, f), tracked) && error("Committed generated image/video in docs")
    return nothing
end
