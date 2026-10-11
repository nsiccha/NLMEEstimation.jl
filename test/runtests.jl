using Test

@testset "NLMEEstimation" begin
    include("layout.jl")
    include("protocol.jl")
    include("empirical_bayes.jl")
include("focei.jl")
include("saem.jl")
end
