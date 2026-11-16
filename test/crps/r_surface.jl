@testitem "scores take only the families R scoringRules scores" begin
    using ScoringRules
    using Distributions

    # R has crps_*, logs_* and dss_* functions for these.
    @test isfinite(crps(Normal(0, 1), 0.3))
    @test isfinite(logs(truncated(Normal(0, 1), 0, 2), 1.0))
    @test isfinite(logs(truncated(1.0 + 2.0 * TDist(4), 0.0, nothing), 1.0))
    @test isfinite(logs(Hypergeometric(5, 6, 4), 2))
    @test isfinite(dss(MixtureModel([Normal(0, 1), Normal(2, 1)]), 1.0))
    @test isfinite(dss(NegativeBinomial(3.0, 0.4), 2))

    # R has no score for these.
    @test_throws MethodError crps(Cauchy(), 0.3)
    @test_throws MethodError crps(truncated(Gamma(2.0, 1.0), 0.5, 3.0), 1.0)
    @test_throws MethodError crps(Geometric(0.3), 2)
    @test_throws MethodError logs(Cauchy(), 0.3)
    @test_throws MethodError logs(censored(Normal(0, 1), 0, 1), 0.5)
    @test_throws MethodError dss(Binomial(5, 0.3), 1)
    @test_throws MethodError dss(truncated(Normal(0, 1), 0, 2), 1.0)
    @test_throws MethodError dss(TwoPieceNormal(0.0, 1.0, 2.0), 0.5)
end
