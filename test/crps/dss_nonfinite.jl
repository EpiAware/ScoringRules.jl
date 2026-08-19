# Generic `dss` NaN convention where the forecast variance does not exist.
# R scoringRules 1.1.3 returns NaN from every dss_* function in these cases.

@testitem "dss returns NaN where the forecast variance does not exist" begin
    using ScoringRules
    using Distributions

    rtol = 1e-10

    # Finite reference values from R scoringRules 1.1.3 (dss_t, dss_gev,
    # dss_gpd, dss_llapl, dss_llogis).

    # Student-t: variance exists for df > 2.
    @test isnan(dss(TDist(1.5), 0.3))
    @test isnan(dss(TDist(2.0), 0.3))
    @test dss(TDist(2.1), 0.3)≈3.04880815200914 rtol=rtol

    # GEV / GPD: variance exists for shape < 1/2.
    @test dss(GeneralizedExtremeValue(0, 1, 0.45), 0.3)≈3.56231579954632 rtol=rtol
    @test isnan(dss(GeneralizedExtremeValue(0, 1, 0.5), 0.3))
    @test isnan(dss(GeneralizedExtremeValue(0, 1, 0.7), 0.3))
    @test dss(GeneralizedPareto(0, 1, 0.45), 0.3)≈3.56798159450529 rtol=rtol
    @test isnan(dss(GeneralizedPareto(0, 1, 0.5), 0.3))
    @test isnan(dss(GeneralizedPareto(0, 1, 0.7), 0.3))

    # Log-Laplace: variance exists for σ < 1/2.
    @test dss(LogLaplace(0.1, 0.4), 1.5)≈0.528329206994286 rtol=rtol
    @test isnan(dss(LogLaplace(0.1, 0.5), 1.5))
    @test isnan(dss(LogLaplace(0.1, 0.6), 1.5))

    # Log-logistic: variance exists for shape β > 2. The R comparison uses
    # locationlog = 0, where dss_llogis is unaffected by its precedence bug
    # (see the "differences from R" docs page).
    @test isnan(dss(LogLogistic(1.0, 1.8), 1.5))
    @test isnan(dss(LogLogistic(1.0, 2.0), 1.5))
    @test dss(LogLogistic(1.0, 2.5), 1.5)≈0.940835230375825 rtol=rtol
end
