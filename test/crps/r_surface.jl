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

    # R's beta on [lower, upper] and shifted exponential; values from R
    # scoringRules 1.1.3 (crps_beta, logs_beta, dss_beta, crps_expM, logs_exp2).
    bi = 1 + 2 * Beta(2, 3)
    @test crps(bi, 2.0) ≈ 0.146428571428571 rtol = 1.0e-12
    @test logs(bi, 2.0) ≈ 0.287682072451781 rtol = 1.0e-12
    @test dss(bi, 2.0) ≈ -1.58258146374831 rtol = 1.0e-12
    @test crps(3 + (-2) * Beta(3, 2), 2.0) ≈ crps(bi, 2.0) rtol = 1.0e-12
    ex = 1 + Exponential(2.0)
    @test crps(ex, 2.0) ≈ 0.426122638850534 rtol = 1.0e-12
    @test logs(ex, 2.0) ≈ 1.19314718055995 rtol = 1.0e-12
    @test_throws ArgumentError crps(1 + (-1) * Exponential(2.0), 0.0)
    @test_throws ArgumentError logs(1 + (-1) * Exponential(2.0), 0.0)

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
