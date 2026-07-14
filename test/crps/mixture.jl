@testitem "mixture of normals scores match R scoringRules" tags=[:crps] setup=[References] begin
    using Distributions

    atol = 1e-9
    rtol = 1e-6

    @testset "MixtureModel Normal" begin
        c, n = References.load("mixnorm")
        for i in 1:n
            y = c["y"][i]
            # Determine whether this row has a 3rd component (m3 not NaN)
            if isnan(c["m3"][i])
                means   = [c["m1"][i], c["m2"][i]]
                sds     = [c["s1"][i], c["s2"][i]]
                weights = [c["w1"][i], c["w2"][i]]
            else
                means   = [c["m1"][i], c["m2"][i], c["m3"][i]]
                sds     = [c["s1"][i], c["s2"][i], c["s3"][i]]
                weights = [c["w1"][i], c["w2"][i], c["w3"][i]]
            end
            d = MixtureModel(Normal.(means, sds), weights)
            @test crps(d, y) ≈ c["crps"][i] atol=atol rtol=rtol
            @test logs(d, y) ≈ c["logs"][i] atol=atol rtol=rtol
            @test dss(d, y) ≈ c["dss"][i] atol=atol rtol=rtol
        end
    end
end
