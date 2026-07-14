@testitem "continuous family scores match R scoringRules" tags=[:crps] setup=[References] begin
    using ScoringRules
    using Distributions

    atol=1e-9
    rtol=1e-8

    @testset "Laplace" begin
        c, n = References.load("laplace")
        for i in 1:n
            d = Laplace(c["location"][i], c["scale"][i])
            y = c["y"][i]
            @test crps(d, y)≈c["crps"][i] atol=atol rtol=rtol
            @test logs(d, y)≈c["logs"][i] atol=atol rtol=rtol
            @test dss(d, y)≈c["dss"][i] atol=atol rtol=rtol
        end
    end

    @testset "Exponential" begin
        c, n = References.load("exponential")
        for i in 1:n
            # Distributions.Exponential uses scale = 1/rate
            d = Exponential(1 / c["rate"][i])
            y = c["y"][i]
            @test crps(d, y)≈c["crps"][i] atol=atol rtol=rtol
            @test logs(d, y)≈c["logs"][i] atol=atol rtol=rtol
            @test dss(d, y)≈c["dss"][i] atol=atol rtol=rtol
        end
    end

    @testset "Gamma" begin
        c, n = References.load("gamma")
        for i in 1:n
            # Distributions.Gamma(shape, scale)
            d = Gamma(c["shape"][i], c["scale"][i])
            y = c["y"][i]
            @test crps(d, y)≈c["crps"][i] atol=atol rtol=rtol
            @test logs(d, y)≈c["logs"][i] atol=atol rtol=rtol
            @test dss(d, y)≈c["dss"][i] atol=atol rtol=rtol
        end
    end

    @testset "Beta" begin
        c, n = References.load("beta")
        for i in 1:n
            d = Beta(c["shape1"][i], c["shape2"][i])
            y = c["y"][i]
            @test crps(d, y)≈c["crps"][i] atol=atol rtol=rtol
            @test logs(d, y)≈c["logs"][i] atol=atol rtol=rtol
            @test dss(d, y)≈c["dss"][i] atol=atol rtol=rtol
        end
    end

    @testset "Uniform" begin
        c, n = References.load("unif")
        for i in 1:n
            d = Uniform(c["min"][i], c["max"][i])
            y = c["y"][i]
            ref_crps = c["crps"][i]
            ref_logs = c["logs"][i]
            ref_dss = c["dss"][i]
            @test crps(d, y)≈ref_crps atol=atol rtol=rtol
            # logs and dss are Inf when y is outside [min, max]
            (isnan(ref_logs) || isinf(ref_logs)) && continue
            @test logs(d, y)≈ref_logs atol=atol rtol=rtol
            (isnan(ref_dss) || isinf(ref_dss)) && continue
            @test dss(d, y)≈ref_dss atol=atol rtol=rtol
        end
    end

    @testset "LogNormal" begin
        c, n = References.load("lnorm")
        for i in 1:n
            d = LogNormal(c["meanlog"][i], c["sdlog"][i])
            y = c["y"][i]
            @test crps(d, y)≈c["crps"][i] atol=atol rtol=rtol
            @test logs(d, y)≈c["logs"][i] atol=atol rtol=rtol
            @test dss(d, y)≈c["dss"][i] atol=atol rtol=rtol
        end
    end
end
