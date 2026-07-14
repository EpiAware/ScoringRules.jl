@testitem "logistic family scores match R scoringRules" tags=[:crps] setup=[References] begin
    using Distributions

    atol = 1e-9
    rtol = 1e-8

    _bound(x) = isfinite(x) ? x : nothing
    function trunc_logis(loc, scale, l, u)
        (isinf(l) && isinf(u)) && return Logistic(loc, scale)
        return truncated(Logistic(loc, scale); lower = _bound(l), upper = _bound(u))
    end
    function cens_logis(loc, scale, l, u)
        (isinf(l) && isinf(u)) && return Logistic(loc, scale)
        return censored(Logistic(loc, scale); lower = _bound(l), upper = _bound(u))
    end

    @testset "Logistic" begin
        c, n = References.load("logis")
        for i in 1:n
            d = Logistic(c["location"][i], c["scale"][i])
            y = c["y"][i]
            @test crps(d, y) ≈ c["crps"][i] atol=atol rtol=rtol
            @test logs(d, y) ≈ c["logs"][i] atol=atol rtol=rtol
            @test dss(d, y) ≈ c["dss"][i] atol=atol rtol=rtol
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
            @test crps(d, y) ≈ ref_crps atol=atol rtol=rtol
            isnan(ref_logs) || isinf(ref_logs) && continue
            @test logs(d, y) ≈ ref_logs atol=atol rtol=rtol
        end
    end

    @testset "censored Logistic" begin
        c, n = References.load("clogis")
        for i in 1:n
            d = cens_logis(c["location"][i], c["scale"][i], c["lower"][i], c["upper"][i])
            y = c["y"][i]
            ref_crps = c["crps"][i]
            isnan(ref_crps) && continue
            @test crps(d, y) ≈ ref_crps atol=atol rtol=rtol
        end
    end
end
