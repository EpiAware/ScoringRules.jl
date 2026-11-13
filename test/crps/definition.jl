# The generic numeric route - the CRPS of a wrapped distribution, and the
# moments behind `dss` - is the one part of this package with no closed form
# and no R counterpart to check against, so a wrong answer there is a
# plausible number rather than a failure. These items check it against the
# definition instead, at 300-bit precision where the family allows it. See
# `test/definition_oracle.jl`.

@testitem "the CRPS of a wrapped forecast matches its definition" tags = [:crps] setup = [DefinitionOracle] begin
    using Distributions
    using ScoringRules
    using ScoringRules: BoundaryMass, TwoPieceNormal

    # One forecast per shape of support, rather than one per family: what is
    # under test is the generic route through the wrappers, and every closed
    # form behind it already has its own reference values from R. Each family
    # here costs about twenty seconds, nearly all of it compiling the
    # quadrature for its types, so the list is kept to the shapes that reach
    # different branches - unbounded both ways, a heavy tail, bounded below,
    # bounded both ways, bounded by a shape parameter, asymmetric, and a
    # lattice.
    bases = [
        "Normal(0,1)" => Normal(0, 1),
        "Normal(5,3)" => Normal(5, 3),
        "Logistic(0,1)" => Logistic(0, 1),
        "TDist(3)" => TDist(3.0),
        "Exponential(1)" => Exponential(1.0),
        "Uniform(-1,2)" => Uniform(-1, 2),
        "GEV(0,1,0.2)" => GeneralizedExtremeValue(0.0, 1.0, 0.2),
        "2pNormal(0,1,2)" => TwoPieceNormal(0.0, 1.0, 2.0),
        "Poisson(4)" => Poisson(4.0),
        "NegativeBinomial(5,0.3)" => NegativeBinomial(5.0, 0.3),
        "Binomial(20,0.3)" => Binomial(20, 0.3),
    ]

    # The bounds that matter are the ones a branch is written for: inside the
    # body, one-sided either way, and far enough out that the retained mass
    # has to be read from the tail rather than from a difference of two
    # nearly equal probabilities.
    # Bounds in the body of the distribution and one-sided either way. A
    # bound far out in a tail belongs to the high-precision item below: there
    # the retained mass is 1e-8, and a double-precision CDF of the truncation
    # is cancellation noise, which cannot referee anything.
    cuts(d) = [
        (quantile(d, 0.2), quantile(d, 0.8)),
        (quantile(d, 0.3), Inf),
        (-Inf, quantile(d, 0.7)),
        (quantile(d, 0.05), quantile(d, 0.95)),
    ]

    function wrappers(d)
        out = Any[]
        d isa DiscreteUnivariateDistribution && return out
        for (lo, hi) in cuts(d)
            lo = max(lo, float(minimum(d)))
            hi = min(hi, float(maximum(d)))
            (isnan(lo) || isnan(hi) || lo >= hi) && continue
            l = isfinite(lo) ? lo : nothing
            u = isfinite(hi) ? hi : nothing
            for f in (
                    () -> truncated(d; lower = l, upper = u),
                    () -> censored(d; lower = l, upper = u),
                    () -> BoundaryMass(
                        d; lower = isfinite(lo) ? lo : minimum(d),
                        upper = isfinite(hi) ? hi : maximum(d),
                        lmass = 0.15, umass = 0.25
                    ),
                )
                try
                    push!(out, f())
                catch
                end
            end
        end
        return out
    end

    function points(d)
        ys = Float64[]
        for p in (1.0e-3, 0.25, 0.5, 0.75, 1 - 1.0e-3)
            q = try
                float(quantile(d, p))
            catch
                continue
            end
            isfinite(q) && push!(ys, q)
        end
        lo, hi = float(minimum(d)), float(maximum(d))
        # A point on a bound, and one outside it, reach branches nothing else
        # does.
        isfinite(lo) && append!(ys, (lo, lo - 0.5))
        isfinite(hi) && append!(ys, (hi, hi + 0.5))
        sort!(ys)
        unique!(ys)
        return ys
    end

    ncheck = Ref(0)
    for (name, base) in bases
        for d in vcat(Any[base], wrappers(base))
            for y in points(d)
                ncheck[] += 1
                # The gap this leaves is seventy times the largest the
                # arithmetic produces anywhere in the grid, and four orders
                # tighter than the smallest error worth calling a defect.
                @test crps(d, y) ≈ oracle_crps(d, y) atol = 1.0e-10 rtol = 1.0e-6
            end
        end
    end
    @test ncheck[] > 300
end

@testitem "the CRPS of a far-tail truncation matches its definition" tags = [:crps] setup = [DefinitionOracle] begin
    using Distributions
    using ScoringRules

    # Truncating at the 1e-8 quantile leaves a tail of that mass, and the
    # generic `(F(x) - F(lo)) / (F(hi) - F(lo))` loses all of it to
    # cancellation in double precision. Reading the mass from the other tail
    # is what the package does; at 300 bits the naive expression needs no such
    # care, so it can referee the result. Only the families whose CDF has a
    # generic-real method can be checked this way.
    setprecision(BigFloat, 300)

    n = Ref(0)
    for base in (Normal(0, 1), Normal(5, 3), Logistic(0, 1), Logistic(2, 0.5))
        for (lo, hi) in (
                (quantile(base, 1 - 1.0e-8), nothing),
                (nothing, quantile(base, 1.0e-8)),
            )
            d = truncated(base; lower = lo, upper = hi)
            tw = bigtwin(d)
            @test tw !== nothing
            for p in (0.1, 0.5, 0.9)
                y = quantile(d, p)
                n[] += 1
                want = Float64(oracle_crps(tw, BigFloat(y); rtol = 1.0e-20, ref = d))
                @test crps(d, y) ≈ want atol = 1.0e-12 rtol = 1.0e-6
            end
        end
    end
    @test n[] == 24
end

@testitem "the moments behind dss match their definition" tags = [:crps] setup = [DefinitionOracle] begin
    using Distributions
    using ScoringRules
    using ScoringRules: BoundaryMass, TwoPieceNormal, _forecast_moments

    bases = [
        Normal(0, 1), Normal(5, 3), Logistic(0, 1), TDist(3.0), Exponential(1.0),
        Gamma(2.0, 1.5), Laplace(0, 1), Uniform(-1, 2), LogNormal(0, 1),
        GeneralizedExtremeValue(0.0, 1.0, 0.2), GeneralizedPareto(0.0, 1.0, 0.2),
        TwoPieceNormal(0.0, 1.0, 2.0), Poisson(4.0), NegativeBinomial(5.0, 0.3),
        Binomial(20, 0.3),
    ]

    function wrapped(d)
        out = Any[d]
        d isa DiscreteUnivariateDistribution && return out
        for (lo, hi) in (
                (quantile(d, 0.2), quantile(d, 0.8)),
                (quantile(d, 0.3), Inf),
                (-Inf, quantile(d, 0.7)),
                (-Inf, quantile(d, 0.02)),
            )
            lo = max(lo, float(minimum(d)))
            hi = min(hi, float(maximum(d)))
            (isnan(lo) || isnan(hi) || lo >= hi) && continue
            l = isfinite(lo) ? lo : nothing
            u = isfinite(hi) ? hi : nothing
            for f in (
                    () -> truncated(d; lower = l, upper = u),
                    () -> censored(d; lower = l, upper = u),
                    () -> BoundaryMass(
                        d; lower = isfinite(lo) ? lo : minimum(d),
                        upper = isfinite(hi) ? hi : maximum(d),
                        lmass = 0.15, umass = 0.25
                    ),
                )
                try
                    push!(out, f())
                catch
                end
            end
        end
        return out
    end

    n = Ref(0)
    for base in bases
        for d in wrapped(base)
            pm, pv = _forecast_moments(d)
            om, ov = oracle_moments(d)
            (isfinite(om) && isfinite(ov)) || continue
            n[] += 1
            # The mean is judged against the spread, since a mean of 1e-16
            # where the standard deviation is 1 is zero.
            sd = sqrt(max(ov, 0.0))
            @test abs(pm - om) <= 1.0e-8 * max(abs(om), sd, 1.0e-300)
            @test pv ≈ ov atol = 1.0e-12 rtol = 1.0e-6
        end
    end
    @test n[] > 60
end
