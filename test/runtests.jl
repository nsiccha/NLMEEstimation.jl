using Test

@testset "NLMEEstimation" begin
    include("layout.jl")
    include("protocol.jl")
    include("empirical_bayes.jl")
end
