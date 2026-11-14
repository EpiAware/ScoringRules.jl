@testitem "extreme value family scores match R scoringRules" tags = [:crps] setup = [References] begin
    using ScoringRules
    using Distributions

    atol = 1.0e-9
    # GEV Gumbel limit (shape≈0) uses quadrature in R so allow slightly looser tolerance
    rtol_gev = 1.0e-6
    rtol = 1.0e-8

    @testset "GEV" begin
        c, n = References.load("gev")
        for i in 1:n
            shape = c["shape"][i]
            loc = c["location"][i]
            sc = c["scale"][i]
            y = c["y"][i]
            ref_crps = c["crps"][i]
            ref_logs = c["logs"][i]
            ref_dss = c["dss"][i]
            # Skip rows where R produced NaN (y outside support for bounded GEV)
            isnan(ref_crps) && continue
            d = GeneralizedExtremeValue(loc, sc, shape)
            @test crps(d, y) ≈ ref_crps atol = atol rtol = rtol_gev
            (isnan(ref_logs) || isinf(ref_logs)) && continue
            @test logs(d, y) ≈ ref_logs atol = atol rtol = rtol
            (isnan(ref_dss) || isinf(ref_dss)) && continue
            @test dss(d, y) ≈ ref_dss atol = atol rtol = rtol
        end
    end

    @testset "GPD" begin
        c, n = References.load("gpd")
        for i in 1:n
            shape = c["shape"][i]
            loc = c["location"][i]
            sc = c["scale"][i]
            y = c["y"][i]
            ref_crps = c["crps"][i]
            ref_logs = c["logs"][i]
            ref_dss = c["dss"][i]
            isnan(ref_crps) && continue
            d = GeneralizedPareto(loc, sc, shape)
            @test crps(d, y) ≈ ref_crps atol = atol rtol = rtol
            (isnan(ref_logs) || isinf(ref_logs)) && continue
            @test logs(d, y) ≈ ref_logs atol = atol rtol = rtol
            (isnan(ref_dss) || isinf(ref_dss)) && continue
            @test dss(d, y) ≈ ref_dss atol = atol rtol = rtol
        end
    end
end

@testitem "an infinite observation scores Inf under every extreme-value shape" tags = [:crps] begin
    using ScoringRules
    using Distributions

    # Wherever these scores are defined (shape below 1) the forecast has a
    # finite mean, so an observation at infinity is infinitely far from it.
    # The shape-zero limits used to form `0 * Inf` or `-Inf + Inf` there.
    for ξ in (0.0, 1.0e-12, 1.0e-9, 0.2, -0.3), y in (Inf, -Inf)
        @test crps(GeneralizedExtremeValue(0.5, 2.0, ξ), y) == Inf
        @test crps(GeneralizedPareto(0.5, 2.0, ξ), y) == Inf
    end
end
