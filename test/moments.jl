@testitem "moment-based scores match R scoringRules" setup = [References] begin
    using ScoringRules

    atol = 1.0e-9
    rtol = 1.0e-7

    @testset "ess_moments" begin
        c, n = References.load("ess")
        for i in 1:n
            score = ess_moments(c["y"][i], c["mean"][i], c["var"][i], c["skew"][i])
            @test score ≈ c["ess"][i] atol = atol rtol = rtol
        end
    end

    @testset "dss_moments matches dss on a Normal forecast" begin
        using Distributions
        for (m, v, y) in [(0.0, 1.0, 0.5), (-1.0, 4.0, 2.0), (2.5, 0.25, 3.0)]
            @test dss_moments(y, m, v) ≈ dss(Normal(m, sqrt(v)), y)
        end
    end
end
