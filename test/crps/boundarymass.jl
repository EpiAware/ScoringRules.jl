@testitem "BoundaryMass scores match R scoringRules" setup=[References] begin
    using ScoringRules
    using Distributions

    atol = 1e-9
    rtol = 1e-8

    # -Inf/Inf columns in the reference CSVs are valid `BoundaryMass` bounds
    # directly, so no conversion helper is needed.

    @testset "generalised truncated/censored Normal" begin
        c, n = References.load("gtcnorm")
        for i in 1:n
            d = BoundaryMass(Normal(c["location"][i], c["scale"][i]);
                lower = c["lower"][i], upper = c["upper"][i],
                lmass = c["lmass"][i], umass = c["umass"][i])
            @test crps(d, c["y"][i])≈c["crps"][i] atol=atol rtol=rtol
        end
    end

    @testset "generalised truncated/censored Logistic" begin
        c, n = References.load("gtclogis")
        for i in 1:n
            d = BoundaryMass(Logistic(c["location"][i], c["scale"][i]);
                lower = c["lower"][i], upper = c["upper"][i],
                lmass = c["lmass"][i], umass = c["umass"][i])
            @test crps(d, c["y"][i])≈c["crps"][i] atol=atol rtol=rtol
        end
    end

    @testset "generalised truncated/censored Student-t" begin
        c, n = References.load("gtct")
        for i in 1:n
            loc, sc = c["location"][i], c["scale"][i]
            base = (loc == 0 && sc == 1) ? TDist(c["df"][i]) :
                   loc + sc * TDist(c["df"][i])
            d = BoundaryMass(base;
                lower = c["lower"][i], upper = c["upper"][i],
                lmass = c["lmass"][i], umass = c["umass"][i])
            @test crps(d, c["y"][i])≈c["crps"][i] atol=atol rtol=rtol
        end
    end

    @testset "Exponential with point mass" begin
        c, n = References.load("exp_mass")
        for i in 1:n
            d = BoundaryMass(Exponential(c["scale"][i]);
                lower = c["location"][i], lmass = c["mass"][i])
            @test crps(d, c["y"][i])≈c["crps"][i] atol=atol rtol=rtol
        end
    end

    @testset "Uniform with boundary masses" begin
        c, n = References.load("unif_mass")
        for i in 1:n
            d = BoundaryMass(Uniform(c["min"][i], c["max"][i]);
                lmass = c["lmass"][i], umass = c["umass"][i])
            @test crps(d, c["y"][i])≈c["crps"][i] atol=atol rtol=rtol
        end
    end

    @testset "GPD with point mass" begin
        c, n = References.load("gpd_mass")
        for i in 1:n
            d = BoundaryMass(
                GeneralizedPareto(c["location"][i], c["scale"][i], c["shape"][i]);
                lmass = c["mass"][i])
            @test crps(d, c["y"][i])≈c["crps"][i] atol=atol rtol=rtol
        end
    end
end

@testitem "BoundaryMass consistency and interface" begin
    using ScoringRules
    using Distributions

    ys = [-2.0, -0.5, 0.0, 0.7, 1.5, 3.0]

    @testset "zero masses reduce to truncation" begin
        # `Truncated` CRPS is closed-form for the first three bases and generic
        # quadrature for the last two, hence the per-base tolerance.
        for (base, l, u, rt) in [(Normal(0.5, 1.2), -1.0, 2.0, 1e-12),
            (Logistic(0.0, 0.8), -1.5, 1.5, 1e-12),
            (TDist(5), -1.0, 2.0, 1e-12),
            (Uniform(-1.0, 2.0), -0.5, 1.5, 1e-8),
            (Exponential(2.0), 0.5, Inf, 1e-8)]
            w = BoundaryMass(base; lower = l, upper = u)
            t = truncated(base; lower = l, upper = isfinite(u) ? u : nothing)
            for y in ys
                @test crps(w, y)≈crps(t, y) rtol=rt atol=1e-9
            end
        end
    end

    @testset "tail-probability masses reduce to censoring" begin
        for (base, l, u) in [(Normal(0.5, 1.2), -1.0, 2.0),
            (Logistic(0.0, 0.8), -1.5, 1.5),
            (TDist(5), -1.0, 2.0)]
            w = BoundaryMass(base; lower = l, upper = u,
                lmass = cdf(base, l), umass = ccdf(base, u))
            cen = censored(base; lower = l, upper = u)
            for y in ys
                @test crps(w, y)≈crps(cen, y) rtol=1e-10
            end
        end
    end

    @testset "closed forms agree with the quadrature fallback" begin
        w = BoundaryMass(Normal(0, 1); lower = -1, upper = 2,
            lmass = 0.1, umass = 0.2)
        for y in ys
            quad = invoke(crps, Tuple{ContinuousUnivariateDistribution, Real}, w, y)
            @test crps(w, y)≈quad rtol=1e-6
        end
    end

    @testset "distribution interface" begin
        w = BoundaryMass(Normal(0, 1); lower = -1, upper = 2,
            lmass = 0.1, umass = 0.2)
        int = truncated(Normal(0, 1), -1, 2)

        @test minimum(w) == -1
        @test maximum(w) == 2
        @test cdf(w, -1.5) == 0
        @test cdf(w, -1)≈0.1 atol=1e-12
        @test cdf(w, 2) == 1
        @test cdf(w, 0.5)≈0.1 + 0.7 * cdf(int, 0.5) atol=1e-12
        @test pdf(w, -1) == 0.1        # atom reported as its mass (as Censored)
        @test pdf(w, 2) == 0.2
        @test pdf(w, 0.5)≈0.7 * pdf(int, 0.5) atol=1e-12
        @test logpdf(w, -1)≈log(0.1) atol=1e-12
        @test logpdf(w, 3) == -Inf
        @test quantile(w, 0.05) == -1
        @test quantile(w, 0.95) == 2
        @test quantile(w, 0.5)≈quantile(int, (0.5 - 0.1) / 0.7) atol=1e-12
        @test mean(w)≈-1 * 0.1 + 2 * 0.2 + 0.7 * mean(int) atol=1e-12
        m = mean(w)
        ex2 = 0.1 * 1 + 0.2 * 4 + 0.7 * (var(int) + mean(int)^2)
        @test var(w)≈ex2 - m^2 atol=1e-12

        # logs/dss flow through the generic density and moment methods
        @test logs(w, 0.5)≈-logpdf(w, 0.5)
        @test logs(w, -1)≈-log(0.1)
        @test dss(w, 0.5)≈(0.5 - mean(w))^2 / var(w) + log(var(w))

        draws = rand(w, 2000)
        @test all(-1 .<= draws .<= 2)
        @test any(==(-1.0), draws)
        @test any(==(2.0), draws)
    end

    @testset "constructor validation" begin
        @test_throws ArgumentError BoundaryMass(Normal(); lower = 1, upper = 1)
        @test_throws ArgumentError BoundaryMass(Normal(); lmass = 0.1)
        @test_throws ArgumentError BoundaryMass(Normal(); umass = 0.1)
        @test_throws DomainError BoundaryMass(Normal(); lower = 0, lmass = -0.1)
        @test_throws DomainError BoundaryMass(Normal(); lower = 0, upper = 1,
            lmass = 0.6, umass = 0.6)
    end

    @testset "unrouted combinations fall back to quadrature" begin
        # No closed form exists for these; the generic CDF quadrature applies.
        wg = BoundaryMass(Gamma(2.0, 1.0); lower = 0.5, upper = 3.0,
            lmass = 0.1, umass = 0.05)
        @test isfinite(crps(wg, 1.0)) && crps(wg, 1.0) > 0
        # Upper-censored exponential: the fallback quadrature should agree with
        # the closed-form censored-exponential CRPS via Distributions.censored.
        we = BoundaryMass(Exponential(1.0); upper = 2.0,
            umass = ccdf(Exponential(1.0), 2.0))
        ce = censored(Exponential(1.0); upper = 2.0)
        @test crps(we, 1.0)≈crps(ce, 1.0) rtol=1e-6
    end
end
