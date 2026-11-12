@testitem "extra distribution scores match R scoringRules" setup = [References] begin
    using ScoringRules
    using Distributions

    atol = 1.0e-9
    rtol = 1.0e-7

    # ---------- LogLogistic ----------
    # R parameterisation: crps_llogis(y, locationlog, scalelog)
    # Julia: LogLogistic(α, β) with α = exp(locationlog), β = 1/scalelog
    # (so that log(X) ~ Logistic(log α, 1/β) = Logistic(locationlog, scalelog)).
    # R's dss_llogis always returns NaN (known upstream issue); dss is not checked here.

    @testset "LogLogistic crps" begin
        c, n = References.load("llogis")
        for i in 1:n
            locationlog = c["locationlog"][i]
            scalelog = c["scalelog"][i]
            y = c["y"][i]
            ref = c["crps"][i]
            d = LogLogistic(exp(locationlog), 1 / scalelog)
            @test crps(d, y) ≈ ref atol = atol rtol = rtol
        end
    end

    @testset "LogLogistic logs" begin
        c, n = References.load("llogis")
        for i in 1:n
            locationlog = c["locationlog"][i]
            scalelog = c["scalelog"][i]
            y = c["y"][i]
            ref = c["logs"][i]
            d = LogLogistic(exp(locationlog), 1 / scalelog)
            @test logs(d, y) ≈ ref atol = atol rtol = rtol
        end
    end

    # ---------- LogLaplace ----------
    # R: crps_llapl(y, locationlog, scalelog) / logs_llapl / dss_llapl
    # Julia: LogLaplace(μ, σ) with μ = locationlog, σ = scalelog.
    # dss is only valid when σ < 0.5 (finite variance); the CSV marks other rows NA.

    @testset "LogLaplace crps" begin
        c, n = References.load("llapl")
        for i in 1:n
            μ = c["locationlog"][i]
            σ = c["scalelog"][i]
            y = c["y"][i]
            ref = c["crps"][i]
            d = LogLaplace(μ, σ)
            @test crps(d, y) ≈ ref atol = atol rtol = rtol
        end
    end

    @testset "LogLaplace logs" begin
        c, n = References.load("llapl")
        for i in 1:n
            μ = c["locationlog"][i]
            σ = c["scalelog"][i]
            y = c["y"][i]
            ref = c["logs"][i]
            d = LogLaplace(μ, σ)
            @test logs(d, y) ≈ ref atol = atol rtol = rtol
        end
    end

    @testset "LogLaplace dss (σ < 0.5 only)" begin
        c, n = References.load("llapl")
        for i in 1:n
            isnan(c["dss"][i]) && continue   # NA rows skipped
            μ = c["locationlog"][i]
            σ = c["scalelog"][i]
            y = c["y"][i]
            ref = c["dss"][i]
            d = LogLaplace(μ, σ)
            @test dss(d, y) ≈ ref atol = atol rtol = rtol
        end
    end

    # ---------- TwoPieceNormal ----------
    # R: crps_2pnorm(y, scale1, scale2, location) / logs_2pnorm
    # Julia: TwoPieceNormal(location, scale1, scale2)

    @testset "TwoPieceNormal crps" begin
        c, n = References.load("twopiecenorm")
        for i in 1:n
            loc = c["location"][i]
            scale1 = c["scale1"][i]
            scale2 = c["scale2"][i]
            y = c["y"][i]
            ref = c["crps"][i]
            d = TwoPieceNormal(loc, scale1, scale2)
            @test crps(d, y) ≈ ref atol = atol rtol = rtol
        end
    end

    @testset "TwoPieceNormal logs" begin
        c, n = References.load("twopiecenorm")
        for i in 1:n
            loc = c["location"][i]
            scale1 = c["scale1"][i]
            scale2 = c["scale2"][i]
            y = c["y"][i]
            ref = c["logs"][i]
            d = TwoPieceNormal(loc, scale1, scale2)
            @test logs(d, y) ≈ ref atol = atol rtol = rtol
        end
    end

    # ---------- TwoPieceExponential ----------
    # R: crps_2pexp(y, scale1, scale2, location) / logs_2pexp
    # Julia: TwoPieceExponential(location, scale1, scale2)

    @testset "TwoPieceExponential crps" begin
        c, n = References.load("twopieceexp")
        for i in 1:n
            loc = c["location"][i]
            scale1 = c["scale1"][i]
            scale2 = c["scale2"][i]
            y = c["y"][i]
            ref = c["crps"][i]
            d = TwoPieceExponential(loc, scale1, scale2)
            @test crps(d, y) ≈ ref atol = atol rtol = rtol
        end
    end

    @testset "TwoPieceExponential logs" begin
        c, n = References.load("twopieceexp")
        for i in 1:n
            loc = c["location"][i]
            scale1 = c["scale1"][i]
            scale2 = c["scale2"][i]
            y = c["y"][i]
            ref = c["logs"][i]
            d = TwoPieceExponential(loc, scale1, scale2)
            @test logs(d, y) ≈ ref atol = atol rtol = rtol
        end
    end
end

@testitem "TwoPiece quantile and rand invert their own CDF" tags = [:crps] begin
    using ScoringRules
    using Distributions
    using Random: MersenneTwister
    using Statistics: mean, var

    # `quantile` and `rand` rest on the standard normal quantile function;
    # neither had a test, so this pins them to `cdf` and to the closed-form
    # moments.
    for d in (
            TwoPieceNormal(0.0, 1.0, 1.0),
            TwoPieceNormal(-1.5, 0.5, 3.0),
            TwoPieceNormal(2.0, 4.0, 0.25),
            TwoPieceExponential(0.0, 1.0, 1.0),
            TwoPieceExponential(1.0, 0.3, 2.5),
        )
        for p in (1.0e-10, 0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999, 1 - 1.0e-10)
            q = quantile(d, p)
            @test cdf(d, q) ≈ p atol = 1.0e-9 rtol = 1.0e-7
        end
        # The median splits the mass in half whichever arm it falls on.
        @test cdf(d, quantile(d, 0.5)) ≈ 0.5 rtol = 1.0e-10

        # The sample mean and variance against the closed-form ones, to a
        # tolerance set by the standard error rather than a relative one: the
        # symmetric cases have mean zero, where `rtol` means nothing.
        n = 400_000
        draws = rand(MersenneTwister(4711), d, n)
        @test mean(draws) ≈ mean(d) atol = 6 * sqrt(var(d) / n)
        @test var(draws) ≈ var(d) rtol = 0.05
        @test all(isfinite, draws)
    end
end
