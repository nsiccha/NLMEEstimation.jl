using Test

@testset "NLMEEstimation" begin
    include("layout.jl")
    include("protocol.jl")
    include("empirical_bayes.jl")
include("focei.jl")
include("saem.jl")
include("saem_nuts.jl")
include("saem_nuts_reference.jl")
end
