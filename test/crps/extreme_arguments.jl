# Scores at arguments far outside the reference grids, where a closed form can
# overflow, cancel or divide by a probability that has underflowed to zero.
# Every expected value here is either an R scoringRules 1.1.3 result or an
# exact limit; the comment on each line says which.

@testitem "discrete CRPS holds up at large parameters" tags = [:crps] begin
    using ScoringRules
    using Distributions

    rtol = 1.0e-9

    # The hypergeometric factor of the negative binomial CRPS. The Gauss series
    # that `HypergeometricFunctions` sums returns NaN from size ≈ 100 here.
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

    # A peak the quadrature has to be told about. The integrand of the
    # hypergeometric factor is a spike at the origin of half-width 1/√(aZ);
    # sampling the whole interval blind, every node underflows to zero, the
    # error estimate is zero too, and the factor comes back as 0 having
    # "converged". That dropped a term the size of the score itself:
    # `NegativeBinomial(724, 0.01)` gave 2135.92 against the true 625.71, one
    # unit of `size` away from a case that was right.
    # Reference: brute-force Σ_k (F(k) − 1{k ≥ y})² over the pmf, which R's
    # crps_nbinom also agrees with to the digits it prints.
    @test crps(NegativeBinomial(723.0, 0.01), 71580.0) ≈ 625.270617245071 rtol = rtol
    @test crps(NegativeBinomial(724.0, 0.01), 71680.0) ≈ 625.713745875651 rtol = rtol
    @test crps(NegativeBinomial(200.0, 0.002), 1.0e5) ≈ 1656.97968453583 rtol = rtol
    @test crps(NegativeBinomial(2.1e6, 0.5), 2.1e6) ≈ 478.932173935184 rtol = rtol

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

    # An affine transform of a lattice forecast, in both directions. A
    # decreasing map is still a forecast, and its lattice is the integers when
    # the location and scale are: guarding the equivariance route on `σ > 0`
    # alone answered `NaN` where the step-CDF sum had been right.
    # Reference: brute-force Σ_k (F(k) − 1{k ≥ y})² over the lattice.
    @test crps(-1 * Poisson(2.0), -2.0) ≈ 0.31117674444005744 rtol = rtol
    for base in (Poisson(2.0), Binomial(6, 0.4), Geometric(0.3))
        for y in (-3.0, -0.5, 0.0, 1.0, 2.5, 7.0)
            # CRPS is invariant under reflecting both forecast and observation.
            @test crps(-1 * base, -y) ≈ crps(base, y) rtol = rtol
            @test crps((-2) * base + 3, 3 - 2 * y) ≈ 2 * crps(base, y) rtol = rtol
            @test crps(2 * base + 3, 3 + 2 * y) ≈ 2 * crps(base, y) rtol = rtol
        end
    end

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

@testitem "a degenerate truncation window is a point forecast" tags = [:crps] begin
    using ScoringRules
    using Distributions

    # Coincident bounds leave a point mass at the bound, so the CRPS is the
    # absolute error. The logistic paths returned 0 instead, where the normal
    # and Student-t ones, and R, all return |y − l|.
    for base in (Logistic(0.0, 1.0), Normal(0.0, 1.0), TDist(5.0), 1.0 + 2.0 * TDist(5.0))
        for y in (3.0, 1.0, -2.5)
            @test crps(Distributions.censored(base, 1.0, 1.0), y) ≈ abs(y - 1.0)
            @test crps(truncated(base, 1.0, 1.0), y) ≈ abs(y - 1.0)
        end
    end
    # The generalised form is not reachable with coincident bounds through
    # `BoundaryMass`, which requires `lower < upper`, so it is checked
    # directly. R returns 2 for all three of these.
    @test ScoringRules._crps_gtclogis(3.0, 0.0, 1.0, 1.0, 1.0, 0.2, 0.3) ≈ 2.0
    @test ScoringRules._crps_gtcnorm(3.0, 0.0, 1.0, 1.0, 1.0, 0.2, 0.3) ≈ 2.0
    @test ScoringRules._crps_gtct(3.0, 5.0, 0.0, 1.0, 1.0, 1.0, 0.2, 0.3) ≈ 2.0
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

@testitem "scores stay defined where a wrapper's own CDF has cancelled" tags = [:crps] begin
    using ScoringRules
    using Distributions
    using ForwardDiff: derivative

    # A truncated forecast's CDF is a difference of base CDFs over its
    # normalising mass. Deep in a tail all three round to the same thing, and
    # `cdf` and `ccdf` of this object are both exactly 0 at 45, which made the
    # generic CRPS 0.0 for an observation four units from the forecast mean.
    # Reference: by memorylessness this forecast is 40 + Exponential(1).
    d = truncated(Exponential(1.0), 40.0, nothing)
    @test cdf(d, 45.0) == 0.0 && ccdf(d, 45.0) == 0.0
    @test crps(d, 45.0) ≈ crps(Exponential(1.0), 5.0) rtol = 1.0e-10
    @test crps(d, 0.5) ≈ 40.0 rtol = 1.0e-12
    @test dss(d, 45.0) ≈ dss_moments(45.0, 41.0, 1.0) rtol = 1.0e-9

    # The same cancellation in the moments: the mean of this one came out
    # below its own lower bound.
    let m = first(ScoringRules._forecast_moments(truncated(Gamma(2.0, 1.0), 40.0, nothing)))
        @test m > 40.0
    end

    # A heavy tail puts the deepest quantile around 1e16, and a segment
    # boundary that far out next to an infinite endpoint made QuadGK produce a
    # NaN and raise. Reference for the Cauchy: 300-bit quadrature of the exact
    # truncated CDF, 0.8745799901742567.
    @test crps(truncated(Cauchy(0.0, 1.0), 0.5, nothing), 0.7) ≈
        0.8745799901742567 rtol = 1.0e-10
    for dd in (
            truncated(Cauchy(0.0, 1.0), 0.5, nothing),
            truncated(LogLogistic(1.0, 1.0), 0.5, nothing),
            truncated(Gamma(0.5, 4.0), 0.5, nothing),
        )
        @test isfinite(crps(dd, 0.7))
    end
    # A truncated Student-t reaches its own closed form, which returns NaN for
    # one degree of freedom or fewer, as R's `crps_tt` does.
    @test isnan(crps(truncated(TDist(1.0), 0.5, nothing), 0.7))

    # `dss` reports NaN where the forecast has no variance, including a
    # truncation that keeps an infinite tail of a base that has none. The
    # quadrature would otherwise stop at the deepest quantile it can represent
    # and call the result a score: a truncated Cauchy scored 37.4.
    for dd in (
            truncated(Cauchy(0.0, 1.0), 0.5, nothing),
            truncated(TDist(1.0), 0.5, nothing),
            truncated(LogNormal(0.0, 30.0), 0.5, nothing),
        )
        @test isnan(dss(dd, 0.7))
    end
    # With both bounds finite the moments always exist, so these must not be
    # swept up by that check.
    @test isfinite(dss(truncated(Cauchy(0.0, 1.0), -2.0, 3.0), 0.7))
    @test isfinite(dss(truncated(TDist(1.0), -2.0, 3.0), 0.7))

    # The variance guard has to run before the mean is asked for: a
    # log-logistic with β ≤ 1 has no mean and throws rather than returning
    # one, which took `dss` down where it had returned NaN.
    @test isnan(dss(LogLogistic(1.0, 1.0), 0.5))
    @test isnan(dss(LogLogistic(1.0, 0.5), 0.5))
    @test isnan(dss(LogLogistic(1.0, 2.0), 0.5))
    @test isfinite(dss(LogLogistic(1.0, 2.5), 0.5))

    # The generalised Pareto survival function is `exp(-log1p(ξz)/ξ)`, which is
    # 0/0 at ξ = 0. Branching on `ξ == 0` is not enough: equality is false for
    # a dual with a zero value and a nonzero partial, so under automatic
    # differentiation the branch was skipped and both the value and the
    # gradient came out NaN.
    @test crps(GeneralizedPareto(0.0, 1.0, 0.0), 0.5) ≈
        crps(Exponential(1.0), 0.5) rtol = 1.0e-14
    let g0 = derivative(x -> crps(GeneralizedPareto(0.0, 1.0, x), 0.5), 0.0)
        @test isfinite(g0)
        @test g0 ≈ derivative(x -> crps(GeneralizedPareto(0.0, 1.0, x), 0.5), 1.0e-12) rtol = 1.0e-9
    end
    for y in (0.3, 2.0, 7.0)
        @test isfinite(derivative(x -> crps(GeneralizedPareto(0.0, 1.0, x), y), 0.0))
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
        @test isfinite(dss(d, 1.0))
    end

    # Cases with a closed form, so the moments are checked against something
    # other than themselves. For `Exponential(1)` truncated to [0, 2],
    # `E[X] = (1 − 3e⁻²)/Z` and `E[X²] = (2 − 10e⁻²)/Z` with `Z = 1 − e⁻²`;
    # censoring at 2 instead moves the upper tail onto an atom there, giving
    # `E[Y] = 1 − e⁻²` and `E[Y²] = 2 − 6e⁻²`.
    let em2 = exp(-2.0), z = 1 - exp(-2.0)
        mt = (1 - 3em2) / z
        vt = (2 - 10em2) / z - mt^2
        m, v = ScoringRules._forecast_moments(truncated(Exponential(1.0), 0.0, 2.0))
        @test m ≈ mt rtol = 1.0e-12
        @test v ≈ vt rtol = 1.0e-12
        @test dss(truncated(Exponential(1.0), 0.0, 2.0), 1.0) ≈
            dss_moments(1.0, mt, vt) rtol = 1.0e-12

        mc = 1 - em2
        vc = (2 - 6em2) - mc^2
        m, v = ScoringRules._forecast_moments(
            Distributions.censored(Exponential(1.0), 0.0, 2.0)
        )
        @test m ≈ mc rtol = 1.0e-12
        @test v ≈ vc rtol = 1.0e-12
    end

    # A bound deep in the upper tail: the retained mass is a difference of CDFs
    # that cancels to exactly zero there, which made the mean of
    # `truncated(Gamma(2, 1), 40, Inf)` come out below its own lower bound.
    # Reference: the upper incomplete gammas, `Γ(3,40)/Γ(2,40) = 1682/41`.
    let d = truncated(Gamma(2.0, 1.0), 40.0, nothing)
        m, v = ScoringRules._forecast_moments(d)
        @test m ≈ 1682 / 41 rtol = 1.0e-12
        @test v ≈ 6 * (1 + 40 + 800 + 40.0^3 / 6) / 41 - (1682 / 41)^2 rtol = 1.0e-10
        @test m > 40
    end

    # A forecast whose whole support is narrow and far from the midpoint of a
    # half-infinite interval: both moments came back as zero.
    let d = truncated(Exponential(1.0e-6), 0.0, nothing)
        m, v = ScoringRules._forecast_moments(d)
        @test m ≈ 1.0e-6 rtol = 1.0e-9
        @test v ≈ 1.0e-12 rtol = 1.0e-8
    end

    # The mirror image: a bound so far out that it retains the whole base, with
    # the mass nowhere near it. Measuring the first moment from the bound puts
    # the cancellation straight back, and `dss` on a truncated `Laplace(0, 1)`
    # at `lower = -1e6` came out 1.224 against `log 2`.
    for (d, mref, vref) in (
            (truncated(TDist(5.0); lower = -1.0e6), 0.0, 5 / 3),
            (truncated(TDist(5.0); lower = -1.0e8), 0.0, 5 / 3),
            (truncated(Laplace(0.0, 1.0); lower = -1.0e6), 0.0, 2.0),
            (truncated(Laplace(0.0, 1.0); lower = -1.0e8), 0.0, 2.0),
            (Distributions.censored(Laplace(0.0, 1.0); lower = -1.0e6), 0.0, 2.0),
        )
        m, v = ScoringRules._forecast_moments(d)
        @test m ≈ mref atol = 1.0e-6
        @test v ≈ vref rtol = 1.0e-6
        @test dss(d, 0.0) ≈ dss_moments(0.0, mref, vref) rtol = 1.0e-6
    end

    # The variance has to be taken about the mean. Built as `E[X²] − E[X]²` it
    # cancels away for an interval far from the origin: a `BoundaryMass` on
    # `Normal(1e8, 1)` gave a variance of exactly 0, and the truncated
    # `Logistic(1e8, 1)` a negative one, where the spread is O(1).
    for (base, shifted) in (
            (Normal(1.0e8, 1.0), Normal(0.0, 1.0)),
            (Normal(1.0e6, 1.0), Normal(0.0, 1.0)),
            (Logistic(1.0e8, 1.0), Logistic(0.0, 1.0)),
            (Logistic(-1.0e9, 2.0), Logistic(0.0, 2.0)),
        )
        c = mean(base)
        for (far, near) in (
                (
                    BoundaryMass(base; lower = c - 1, upper = c + 1, lmass = 0.1, umass = 0.1),
                    BoundaryMass(shifted; lower = -1.0, upper = 1.0, lmass = 0.1, umass = 0.1),
                ),
                (truncated(base, c - 1, c + 1), truncated(shifted, -1.0, 1.0)),
                (
                    Distributions.censored(base, c - 1, c + 1),
                    Distributions.censored(shifted, -1.0, 1.0),
                ),
            )
            _, vfar = ScoringRules._forecast_moments(far)
            _, vnear = ScoringRules._forecast_moments(near)
            @test vfar > 0
            # `x − m` inside the quadrature is known only to `eps(c)`, so the
            # variance of a unit-wide window at `c` cannot beat that relative
            # to its own size. The tolerance is that floor, not a guess.
            @test vfar ≈ vnear rtol = max(1.0e-9, 100 * eps(c))
            @test isfinite(dss(far, c))
        end
    end

    # Where Distributions.jl does supply analytic moments they are still used,
    # so nothing that worked before moves.
    @test dss(truncated(Normal(0.0, 1.0), 0.0, 2.0), 1.0) ≈
        dss_moments(
        1.0, mean(truncated(Normal(0.0, 1.0), 0.0, 2.0)),
        var(truncated(Normal(0.0, 1.0), 0.0, 2.0))
    ) rtol = 1.0e-14
end

@testitem "the Poisson CRPS holds at means past where besselix stops" tags = [:crps] setup = [DefinitionOracle] begin
    using ScoringRules
    using Distributions
    using ScoringRules: _pois_dens, loggamma

    # The score uses the probability mass at the observation, and the textbook
    # `exp(k log λ - λ - log k!)` subtracts numbers of size `k log λ` to get
    # one of size one: 2.5e-7 out at a mean of 1e8. It has to hold at any
    # mean, near the mode and well into both tails.
    setprecision(BigFloat, 300) do
        exact(λ, k) = exp(big(k) * log(big(λ)) - big(λ) - loggamma(big(k) + 1))
        for λ in (0.5, 3.0, 20.0, 350.0, 1.0e4, 1.0e6, 1.0e8, 1.0e9, 1.0e10, 5.0e18, 9.0e18)
            for z in (-6.0, -2.0, 0.0, 0.5, 3.0, 6.0)
                k = max(0, round(Int, λ + z * sqrt(λ)))
                @test _pois_dens(λ, k) ≈ Float64(exact(λ, k)) rtol = 1.0e-13
            end
        end
    end
    @test _pois_dens(0.0, 0) == 1
    @test _pois_dens(0.0, 3) == 0
    @test _pois_dens(2.0, 0) ≈ exp(-2.0) rtol = 1.0e-15

    # `besselix` raises an error once its argument passes about 1e9, which is
    # a Poisson mean of 5e8. The score itself is unremarkable there.
    for λ in (1.0e8, 1.0e9, 1.0e10), z in (-2.0, 0.0, 0.7, 3.0)
        y = round(λ + z * sqrt(λ))
        @test crps(Poisson(λ), y) ≈ oracle_crps(Poisson(λ), y) rtol = 1.0e-8
    end
    # No step where the large-argument form takes over.
    for λ in (4.99e7, 5.0e7, 5.01e7)
        @test crps(Poisson(λ), λ) ≈ oracle_crps(Poisson(λ), λ) rtol = 1.0e-9
    end
end

@testitem "the negative binomial CRPS returns where its hypergeometric argument overflows" tags = [:crps] begin
    using ScoringRules
    using Distributions

    # As `prob` goes to zero, `prob * X` tends to a Gamma(size, 1) variable,
    # so the score tends to the Gamma one rescaled, with a relative error of
    # order `prob`. Right up to where `(1 - prob) / prob^2` overflows, the
    # closed form must agree with it.
    for (r, p, y) in ((5.0, 1.0e-150, 3.0), (5.0, 1.0e-153, 3.0), (2.5, 1.0e-140, 1.0e140))
        @test crps(NegativeBinomial(r, p), y) ≈ crps(Gamma(r, 1.0), y * p) / p rtol = 1.0e-8
    end
    # Past it, and at a size where the closed form's two terms cancel beyond
    # what double precision holds, there is no score to give. The quadrature's
    # panel loop used to go round for ever on the first, and the second came
    # out negative.
    @test isnan(crps(NegativeBinomial(5.0, 1.0e-160), 3.0))
    @test isnan(crps(NegativeBinomial(3.0e307, 0.5), 3.0e307))
    @test isnan(crps(NegativeBinomial(1.0e300, 0.3), 1.0e300 * 0.7 / 0.3))
    # An infinite observation is still an infinite score.
    @test crps(NegativeBinomial(5.0, 0.3), Inf) == Inf
end

@testitem "the negative binomial CRPS holds for a nearly degenerate forecast" tags = [:crps] setup = [DefinitionOracle] begin
    using ScoringRules
    using Distributions

    # A size near zero or a `prob` near one puts almost all the mass at zero,
    # and the score at `y = 0` is tiny. The closed form reaches it by
    # cancelling two terms of order one, which costs about `log10(1/(1 - prob))`
    # digits unless the hypergeometric factor's distance from one is taken
    # without forming the factor itself.
    for (r, p) in ((1.0e-8, 0.5), (1.0e-12, 0.5), (1.0, 1 - 1.0e-9), (5.0, 1 - 1.0e-9), (1.0, 1 - 1.0e-12), (0.5, 1 - 1.0e-6))
        d = NegativeBinomial(r, p)
        for y in (0.0, 1.0, 3.0)
            @test crps(d, y) ≈ oracle_crps(d, y) rtol = 1.0e-10
        end
    end
    # A tiny size with a small `prob` has a tail that shrinks by only
    # `1 - prob` a step. Summing it would take of order `20/prob` terms, about
    # a second here, and the closed form is already within 1e-8.
    d = NegativeBinomial(1.0e-6, 2.0e-5)
    @test crps(d, 0.0) ≈ oracle_crps(d, 0.0) rtol = 1.0e-8
    @test (@elapsed crps(d, 0.0)) < 0.05
end
