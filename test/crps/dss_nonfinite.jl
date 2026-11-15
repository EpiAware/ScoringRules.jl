# Generic `dss` NaN convention where the forecast variance does not exist.
# R scoringRules 1.1.3 returns NaN from every dss_* function in these cases.

@testitem "dss returns NaN where the forecast variance does not exist" begin
    using ScoringRules
    using Distributions

    rtol = 1.0e-10

    # Finite reference values from R scoringRules 1.1.3 (dss_t, dss_gev,
    # dss_gpd, dss_llapl, dss_llogis).

    # Student-t: variance exists for df > 2.
    @test isnan(dss(TDist(1.5), 0.3))
    @test isnan(dss(TDist(2.0), 0.3))
    @test dss(TDist(2.1), 0.3) ≈ 3.04880815200914 rtol = rtol

    # GEV / GPD: variance exists for shape < 1/2.
    @test dss(GeneralizedExtremeValue(0, 1, 0.45), 0.3) ≈ 3.56231579954632 rtol = rtol
    @test isnan(dss(GeneralizedExtremeValue(0, 1, 0.5), 0.3))
    @test isnan(dss(GeneralizedExtremeValue(0, 1, 0.7), 0.3))
    @test dss(GeneralizedPareto(0, 1, 0.45), 0.3) ≈ 3.56798159450529 rtol = rtol
    @test isnan(dss(GeneralizedPareto(0, 1, 0.5), 0.3))
    @test isnan(dss(GeneralizedPareto(0, 1, 0.7), 0.3))

    # Log-Laplace: variance exists for σ < 1/2.
    @test dss(LogLaplace(0.1, 0.4), 1.5) ≈ 0.528329206994286 rtol = rtol
    @test isnan(dss(LogLaplace(0.1, 0.5), 1.5))
    @test isnan(dss(LogLaplace(0.1, 0.6), 1.5))

    # Log-logistic: variance exists for shape β > 2. The R comparison uses
    # locationlog = 0, where dss_llogis is unaffected by its precedence bug
    # (see the "differences from R" docs page).
    @test isnan(dss(LogLogistic(1.0, 1.8), 1.5))
    @test isnan(dss(LogLogistic(1.0, 2.0), 1.5))
    @test dss(LogLogistic(1.0, 2.5), 1.5) ≈ 0.940835230375825 rtol = rtol
end

@testitem "dss of a wrapped forecast is NaN where a kept tail has no variance" begin
    using ScoringRules
    using Distributions

    # A wrapper that keeps an infinite tail has a variance only where its base
    # does, whichever wrapper it is; with both bounds finite it always has one.
    # The quadrature cannot tell a divergent integral from a large one, so a
    # missing guard shows up as a finite score.
    for base in (Cauchy(), TDist(1.5), TDist(2.0), GeneralizedPareto(0.0, 1.0, 0.6))
        lo = quantile(base, 0.3)
        hi = quantile(base, 0.9)
        for wrap in (
                (l, u) -> truncated(base; lower = l, upper = u),
                (l, u) -> censored(base; lower = l, upper = u),
                (l, u) -> BoundaryMass(
                    base; lower = something(l, minimum(base)),
                    upper = something(u, maximum(base)),
                    lmass = l === nothing ? 0.0 : 0.1,
                    umass = u === nothing ? 0.0 : 0.1
                ),
            )
            @test isnan(dss(wrap(lo, nothing), hi))
            isfinite(minimum(base)) || @test isnan(dss(wrap(nothing, hi), lo))
            @test isfinite(dss(wrap(lo, hi), (lo + hi) / 2))
        end
    end
end

@testitem "dss of a wrapped forecast is NaN where a kept tail is too heavy to integrate" begin
    using ScoringRules
    using ScoringRules: _forecast_moments
    using Distributions
    using ScoringRules: gamma

    # Mean of a Student-t restricted to its positive half.
    halfmean(ν) = 2 * sqrt(ν) * gamma((ν + 1) / 2) / (sqrt(pi) * gamma(ν / 2) * (ν - 1))

    # Where the variance only just exists the quadrature cannot resolve the
    # tail, and the variance it would report is up to 16% low.
    for ν in (2.05, 2.1)
        @test isnan(dss(truncated(TDist(ν); lower = 0), 1.0))
        @test isnan(dss(censored(TDist(ν); lower = 0), 1.0))
    end
    @test isnan(dss(truncated(Pareto(2.1, 1.0); lower = 2), 3.0))
    @test isnan(dss(truncated(GeneralizedPareto(0.0, 1.0, 0.48); lower = 1), 3.0))

    # A tail it can resolve keeps its variance.
    for ν in (2.5, 3.0)
        m, v = _forecast_moments(truncated(TDist(ν); lower = 0))
        @test m ≈ halfmean(ν) rtol = 1.0e-9
        @test v ≈ ν / (ν - 2) - halfmean(ν)^2 rtol = 1.0e-7
        m, v = _forecast_moments(censored(TDist(ν); lower = 0))
        @test m ≈ halfmean(ν) / 2 rtol = 1.0e-9
        @test v ≈ ν / (ν - 2) / 2 - halfmean(ν)^2 / 4 rtol = 1.0e-7
    end
    @test _forecast_moments(truncated(Pareto(3.0, 1.0); lower = 2))[2] ≈
        var(Pareto(3.0, 2.0)) rtol = 1.0e-7
end
