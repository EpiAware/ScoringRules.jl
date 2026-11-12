# Scores at arguments far outside the reference grids, where a closed form can
# overflow, cancel or divide by a probability that has underflowed to zero.
# Every expected value here is either an R scoringRules 1.1.3 result or an
# exact limit; the comment on each line says which.

@testitem "discrete CRPS holds up at large parameters" tags = [:crps] begin
    using ScoringRules
    using Distributions

    rtol = 1.0e-9

    # The hypergeometric factor of the negative binomial CRPS. The Gauss series
    # loses digits from size ≈ 40 and returns NaN from size ≈ 100.
    # Reference: crps_nbinom(y, size, prob) in R scoringRules 1.1.3.
    @test crps(NegativeBinomial(40.0, 0.3), 5.0) ≈ 78.4150319255607 rtol = rtol
    @test crps(NegativeBinomial(100.0, 0.3), 5.0) ≈ 212.61976265306 rtol = rtol
    @test crps(NegativeBinomial(200.0, 0.3), 5.0) ≈ 439.429528293562 rtol = rtol
    @test crps(NegativeBinomial(1000.0, 0.3), 5.0) ≈ 2278.5831081912 rtol = rtol

    # Non-integer sizes, where R returns -Inf for some combinations
    # (size = 2.5 or 5.5 at prob = 0.3, size = 10.5 at prob = 0.7) and the right
    # answer for others. Reference: brute-force Σ_k (F(k) − 1{k ≥ y})² over the
    # pmf, which the closed form must reproduce either way.
    @test crps(NegativeBinomial(40.5, 0.3), 5.0) ≈ 79.5194913104852 rtol = rtol
    @test crps(NegativeBinomial(5.5, 0.3), 5.0) ≈ 4.49843409683683 rtol = rtol
    @test crps(NegativeBinomial(2.5, 0.3), 5.0) ≈ 0.950281678491355 rtol = rtol
    @test crps(NegativeBinomial(10.5, 0.7), 5.0) ≈ 0.670705084814377 rtol = rtol

    # Poisson: `besseli(0, 2λ)` overflows the Amos routine from λ ≈ 351, so the
    # exponentially scaled form has to be used. Reference: crps_pois.
    @test crps(Poisson(350.0), 340.0) ≈ 6.39185500767645 rtol = rtol
    @test crps(Poisson(400.0), 390.0) ≈ 6.56785042111822 rtol = rtol
    @test crps(Poisson(5000.0), 5010.0) ≈ 17.1061600317601 rtol = rtol

    # An observation past `typemax(Int)` must not throw on the `dpois` term.
    @test isfinite(crps(Poisson(3.0), 1.0e30))
end

@testitem "the generic discrete CRPS integrates its whole step CDF" tags = [:crps] begin
    using ScoringRules
    using Distributions

    rtol = 1.0e-9

    # Outside the support the score has to keep growing with the distance: the
    # squared difference is 1 at every lattice point between `y` and the
    # support. Reference: Σ_k (F(k) − 1{k ≥ y})², which for Bernoulli(0.3) at
    # y = −3 is 3 + 0.09.
    @test crps(Bernoulli(0.3), -3.0) ≈ 3.09 rtol = rtol
    @test crps(Bernoulli(0.3), -10.0) ≈ 10.09 rtol = rtol
    @test crps(Bernoulli(0.3), 4.0) ≈ 3.49 rtol = rtol

    # The same forecast written as a Binomial takes the closed form, so the two
    # routes must agree.
    for y in (-3.0, -0.5, 0.0, 0.4, 1.0, 4.0)
        @test crps(Bernoulli(0.3), y) ≈ crps(Binomial(1, 0.3), y) rtol = rtol
    end

    # Families with neither a `quantile` method (Skellam) nor a representable
    # `support` (an upper-unbounded truncated Poisson) must still score.
    # Reference: brute-force summation over the pmf.
    @test crps(Skellam(1.0, 2.0), 2.0) ≈ 2.08717915795548 rtol = rtol
    @test crps(truncated(Poisson(3.0), 1, nothing), 2.0) ≈ 0.578038899674316 rtol = rtol

    # A support that is not the unit lattice: the step CDF is integrated
    # exactly, not summed over integers.
    d = DiscreteNonParametric([0.5, 2.5], [0.4, 0.6])
    # ∫(F − 1{x ≥ y})² dx with y = 1.0: (2.5−0.5) split at 1.0 gives
    # 0.4²·0.5 + 0.6²·1.5, plus nothing outside the support.
    @test crps(d, 1.0) ≈ 0.4^2 * 0.5 + 0.6^2 * 1.5 rtol = rtol

    # And it agrees with the closed-form Poisson it is not used for.
    for λ in (0.5, 3.0, 25.0), y in (-2.0, 0.0, 1.0, 4.0, 30.0)
        ref = crps(Poisson(λ), y)
        got = ScoringRules._crps_step(
            Poisson(λ), y, ScoringRules._crps_discrete_nodes(Poisson(λ))
        )
        @test got ≈ ref atol = 1.0e-9 rtol = 1.0e-9
    end
end

@testitem "logistic and normal CRPS survive a far tail and a zero scale" tags = [:crps] begin
    using ScoringRules
    using Distributions

    rtol = 1.0e-10

    # `log(logistic(z))` underflows to -Inf for z ≲ -745, taking the score with
    # it. The exact value is |z| − 1 up to a term below eps.
    @test crps(Logistic(0.0, 1.0), -1000.0) ≈ 999.0 rtol = rtol
    @test crps(Logistic(0.0, 1.0), -1.0e6) ≈ 999999.0 rtol = rtol
    # The same thing reached through a small scale. Reference: crps_logis.
    @test crps(Logistic(0.0, 1.0e-3), -1.0) ≈ 0.999 rtol = rtol
    @test crps(Logistic(2.0, 1.0e-8), -1.0) ≈ 3.0 - 1.0e-8 rtol = rtol
    # Symmetry: the upper tail was always finite, so it pins the lower one.
    @test crps(Logistic(0.0, 1.0), -1000.0) ≈ crps(Logistic(0.0, 1.0), 1000.0) rtol = rtol

    # A zero scale is the point forecast at the location; its CRPS is the
    # absolute error. The general expression gives σ·Inf = NaN.
    @test crps(Normal(1.0, 0.0), 2.5) ≈ 1.5
    @test crps(Normal(1.0, 0.0), 1.0) ≈ 0.0
    @test crps(Normal(-2.0, 0.0), -5.0) ≈ 3.0
    @test isnan(crps(Normal(0.0, -1.0; check_args = false), 0.5))
end

@testitem "truncated Student-t CRPS holds up at a far bound" tags = [:crps] begin
    using ScoringRules
    using Distributions

    rtol = 1.0e-7

    # The normaliser is F(upper) − F(lower). Built as `½ + ½·sign(z)·I_w(½, ν/2)`
    # it underflows to exactly 0 far into the tail and the score divides by
    # zero. Reference: crps_tt(y, df, lower, upper) in R scoringRules 1.1.3.
    @test crps(truncated(TDist(10.0), nothing, -100.0), -101.0) ≈ 4.36410789456736 rtol = rtol
    @test crps(truncated(TDist(10.0), nothing, -200.0), -201.0) ≈ 9.57769901941353 rtol = rtol
    @test crps(truncated(TDist(10.0), nothing, -1000.0), -1001.0) ≈ 51.6420017809987 rtol = rtol
    @test crps(truncated(TDist(20.0), nothing, -50.0), -51.0) ≈ 0.638939464884277 rtol = rtol
    @test crps(truncated(TDist(3.0), nothing, -200.0), -201.0) ≈ 39.0169568466966 rtol = rtol

    # Reflected: the formula swaps signs when the lower bound exceeds 3.
    @test crps(truncated(TDist(10.0), 200.0, nothing), 201.0) ≈ 9.57769901941353 rtol = rtol

    # Every score here is a CRPS and so non-negative.
    for df in (2.5, 3.0, 5.0, 10.0, 20.0), u in (-20.0, -100.0, -1.0e4, -1.0e5)
        @test crps(truncated(TDist(df), nothing, u), u - 1.0) > 0
    end

    # The same bound in `logs`: here R loses the tail instead, normalising by
    # `1 - pt(lower)` rather than the upper-tail probability, so these are
    # pinned to exact quadrature at 300-bit precision. R is 6e-5 out.
    for (loc, ref) in (
            (-1.0, 2.42407848440008),
            (0.0, 2.4152658733786),
            (2.0, 2.3974093830396),
        )
        d = truncated(loc + 2.0 * TDist(10.0), 100.0, nothing)
        @test logs(d, 101.0) ≈ ref rtol = 1.0e-12
    end
end

@testitem "extreme-value CRPS is accurate through the Gumbel limit" tags = [:crps] begin
    using ScoringRules
    using Distributions

    # GEV at shape → 0. Reference: the Gumbel CRPS, which the general closed
    # form must approach; the error either side of the switch stays below 1e-7.
    gumbel_ref = Dict(
        0.0 => 0.322836353132628,
        1.5 => 0.652188420933944,
        -1.0 => 0.921533423488204,
    )
    for (y, ref) in gumbel_ref
        @test crps(GeneralizedExtremeValue(0.0, 1.0, 0.0), y) ≈ ref rtol = 1.0e-12
        for ξ in (1.0e-12, 1.0e-10, 1.0e-8, 1.0e-7, -1.0e-10, -1.0e-8)
            @test crps(GeneralizedExtremeValue(0.0, 1.0, ξ), y) ≈ ref rtol = 1.0e-7
        end
    end

    # GPD at shape → 0 is the exponential CRPS, and `log1p` makes the general
    # form exact there rather than 2.5e-4 out (which is where R sits). The
    # tolerance tracks the genuine O(shape) gap between the two, so a numerical
    # error larger than that gap still fails.
    for y in (0.5, 3.0)
        ref = crps(Exponential(1.0), y)
        @test crps(GeneralizedPareto(0.0, 1.0, 0.0), y) ≈ ref rtol = 1.0e-14
        for ξ in (1.0e-12, 1.0e-10, 1.0e-8, -1.0e-12, -1.0e-8)
            @test crps(GeneralizedPareto(0.0, 1.0, ξ), y) ≈ ref rtol =
                max(1.0e-13, 10 * abs(ξ))
        end
    end
end

@testitem "the quadrature fallback resolves a narrow or discontinuous CDF" tags = [:crps] begin
    using ScoringRules
    using Distributions

    # `quadgk` over a half-infinite interval chooses its first nodes from the
    # interval alone. A boundary-mass exponential whose continuous part has
    # scale 0.057 at location 1.985 was missed entirely and the fallback
    # returned 0.15899015 against the exact 0.15898723, reporting convergence
    # whatever `rtol` was asked for.
    loc = 1.98509270790964
    sc = 0.0571883754423546
    mass = 0.248775732819922
    y = 1.84224226559283
    # Exact: the point mass sits above `y`, so the score is
    # (loc − y) + (1 − mass)²·scale/2.
    exact = (loc - y) + (1 - mass)^2 * sc / 2
    @test crps(BoundaryMass(loc + Exponential(sc); lmass = mass), y) ≈ exact rtol = 1.0e-12

    # The fallback must still reproduce every closed form it is not used for.
    fallback(d, z) = invoke(crps, Tuple{ContinuousUnivariateDistribution, Real}, d, z)
    for (d, z) in (
            (Normal(0.0, 1.0), 1.2),
            (Normal(0.0, 1.0), 40.0),
            (Normal(5.0, 1.0e-4), 5.3),
            (Gamma(2.0, 1.5), 3.0),
            (Gamma(2.0, 1.0e-3), 0.01),
            (Beta(2.0, 3.0), 0.4),
            (Exponential(1.0e-3), 0.02),
            (Logistic(0.0, 1.0), 0.7),
            (Uniform(0.0, 2.0), 0.7),
        )
        @test fallback(d, z) ≈ crps(d, z) atol = 1.0e-12 rtol = 1.0e-9
    end
end

@testitem "dss moments stay accurate and defined on wrapped forecasts" tags = [:crps] begin
    using ScoringRules
    using Distributions

    # GEV variance cancels to nothing as shape → 0, and `dss` with it: at
    # shape = 1e-8 the stock expression is out by a third. Reference: the
    # Gumbel limit, mean μ + σγ and variance σ²π²/6.
    γ = 0.577215664901533
    for (μ, σ) in ((0.0, 1.0), (0.5, 2.0))
        v = σ^2 * π^2 / 6
        ref = (1.3 - (μ + σ * γ))^2 / v + log(v)
        @test dss(GeneralizedExtremeValue(μ, σ, 0.0), 1.3) ≈ ref rtol = 1.0e-12
        for ξ in (1.0e-12, 1.0e-9, 1.0e-8, 1.0e-7, -1.0e-9)
            @test dss(GeneralizedExtremeValue(μ, σ, ξ), 1.3) ≈ ref rtol = 1.0e-6
        end
    end
    # Reference: dss_gev in R scoringRules 1.1.3, away from the limit.
    @test dss(GeneralizedExtremeValue(0.0, 1.0, 0.2), 0.3) ≈ 1.28839631495693 rtol = 1.0e-8
    @test isnan(dss(GeneralizedExtremeValue(0.0, 1.0, 0.5), 0.3))

    # `Distributions.var` has no method for most truncated or censored bases,
    # so `dss` threw a MethodError where it documents a numeric score.
    # Reference: moments by quadrature, checked against a 10^7-draw sample in
    # development; here the invariant is that each is finite and matches the
    # moment-based entry point.
    for d in (
            truncated(Exponential(1.0), 0.0, 2.0),
            truncated(Gamma(2.0, 1.5), 1.0, 4.0),
            truncated(Laplace(0.0, 1.0), -1.0, 3.0),
            Distributions.censored(Exponential(1.0), 0.0, 2.0),
            Distributions.censored(Gamma(2.0, 1.5), 1.0, 4.0),
            BoundaryMass(Exponential(1.0); lower = 0.0, upper = 2.0, lmass = 0.1, umass = 0.2),
            BoundaryMass(Gamma(2.0, 1.5); lower = 1.0, upper = 4.0, lmass = 0.05),
        )
        s = dss(d, 1.0)
        @test isfinite(s)
        m, v = ScoringRules._forecast_moments(d)
        @test s ≈ dss_moments(1.0, m, v) rtol = 1.0e-10
    end

    # Where Distributions.jl does supply analytic moments they are still used,
    # so nothing that worked before moves.
    @test dss(truncated(Normal(0.0, 1.0), 0.0, 2.0), 1.0) ≈
        dss_moments(
        1.0, mean(truncated(Normal(0.0, 1.0), 0.0, 2.0)),
        var(truncated(Normal(0.0, 1.0), 0.0, 2.0))
    ) rtol = 1.0e-14
end
