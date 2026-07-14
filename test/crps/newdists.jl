@testitem "extra distribution scores match R scoringRules" setup=[References] begin
    using ScoringRules
    using Distributions

    atol = 1e-9
    rtol = 1e-7

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
            @test crps(d, y)≈ref atol=atol rtol=rtol
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
            @test logs(d, y)≈ref atol=atol rtol=rtol
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
            @test crps(d, y)≈ref atol=atol rtol=rtol
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
            @test logs(d, y)≈ref atol=atol rtol=rtol
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
            @test dss(d, y)≈ref atol=atol rtol=rtol
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
            @test crps(d, y)≈ref atol=atol rtol=rtol
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
            @test logs(d, y)≈ref atol=atol rtol=rtol
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
            @test crps(d, y)≈ref atol=atol rtol=rtol
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
            @test logs(d, y)≈ref atol=atol rtol=rtol
        end
    end
end
