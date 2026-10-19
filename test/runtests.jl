using Test

@testset "NLMEEstimation" begin
    include("layout.jl")
    include("protocol.jl")
    include("empirical_bayes.jl")
    include("ragged.jl")
include("focei.jl")
include("covariance.jl")
include("marginal.jl")
include("diagnostics.jl")
include("saem.jl")
include("saem_nuts.jl")
include("saem_mala.jl")
include("saem_nuts_reference.jl")
include("population_prior.jl")
include("fixed_omega.jl")
include("infeasible.jl")
end
