@testitem "logistic family scores match R scoringRules" tags = [:crps] setup = [References] begin
    using ScoringRules
    using Distributions

    atol = 1.0e-9
    rtol = 1.0e-8

    _bound(x) = isfinite(x) ? x : nothing
    function trunc_logis(loc, scale, l, u)
        (isinf(l)&&isinf(u))&&return Logistic(loc, scale)
        return truncated(Logistic(loc, scale); lower = _bound(l), upper = _bound(u))
    end
    function cens_logis(loc, scale, l, u)
        (isinf(l)&&isinf(u))&&return Logistic(loc, scale)
        return censored(Logistic(loc, scale); lower = _bound(l), upper = _bound(u))
    end

    @testset "Logistic" begin
        c, n = References.load("logis")
        for i in 1:n
            d = Logistic(c["location"][i], c["scale"][i])
            y = c["y"][i]
            @test crps(d, y) ≈ c["crps"][i] atol = atol rtol = rtol
            @test logs(d, y) ≈ c["logs"][i] atol = atol rtol = rtol
            @test dss(d, y) ≈ c["dss"][i] atol = atol rtol = rtol
        end
    end

    @testset "truncated Logistic" begin
        c, n = References.load("tlogis")
        for i in 1:n
            d = trunc_logis(c["location"][i], c["scale"][i], c["lower"][i], c["upper"][i])
            y = c["y"][i]
            ref_crps = c["crps"][i]
            ref_logs = c["logs"][i]
            isnan(ref_crps) && continue
            @test crps(d, y) ≈ ref_crps atol = atol rtol = rtol
            (isnan(ref_logs) || isinf(ref_logs)) && continue
            @test logs(d, y) ≈ ref_logs atol = atol rtol = rtol
        end
    end

    @testset "censored Logistic" begin
        c, n = References.load("clogis")
        for i in 1:n
            d = cens_logis(c["location"][i], c["scale"][i], c["lower"][i], c["upper"][i])
            y = c["y"][i]
            ref_crps = c["crps"][i]
            isnan(ref_crps) && continue
            @test crps(d, y) ≈ ref_crps atol = atol rtol = rtol
        end
    end

    @testset "coincident bounds leave a point mass" begin
        # R scoringRules: crps_clogis(0.5, 0, 1, lower = 1, upper = 1) and
        # crps_tlogis with the same arguments both return 0.5.
        @test crps(censored(Logistic(0.0, 1.0), 1.0, 1.0), 0.5) ≈ 0.5
        @test crps(truncated(Logistic(0.0, 1.0), 1.0, 1.0), 0.5) ≈ 0.5
    end

    @testset "zero scale is a point forecast" begin
        # R scoringRules: crps_logis(y, location, 0) is |y - location|, e.g.
        # crps_logis(c(-3, 0, 5), -2, 0) returns 1 2 7.
        for μ in (-2.0, 0.0, 1.0), y in (-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0)
            d = Logistic(μ, 0.0; check_args = false)
            @test crps(d, y) == abs(y - μ)
        end
    end
end
