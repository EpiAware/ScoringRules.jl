@testitem "normal family scores match R scoringRules" setup=[References] begin
    using Distributions
    using ScoringRules: _crps_gtcnorm

    atol = 1e-9
    rtol = 1e-8

    # helpers to build (possibly one-sided) truncated / censored normals
    _bound(x) = isfinite(x) ? x : nothing
    function trunc_normal(loc, scale, l, u)
        (isinf(l) && isinf(u)) && return Normal(loc, scale)
        return truncated(Normal(loc, scale); lower = _bound(l), upper = _bound(u))
    end
    function cens_normal(loc, scale, l, u)
        (isinf(l) && isinf(u)) && return Normal(loc, scale)
        return censored(Normal(loc, scale); lower = _bound(l), upper = _bound(u))
    end

    @testset "Normal" begin
        c, n = References.load("norm")
        for i in 1:n
            d = Normal(c["mean"][i], c["sd"][i])
            y = c["y"][i]
            @test crps(d, y) ≈ c["crps"][i] atol=atol rtol=rtol
            @test logs(d, y) ≈ c["logs"][i] atol=atol rtol=rtol
            @test dss(d, y) ≈ c["dss"][i] atol=atol rtol=rtol
        end
    end

    @testset "truncated Normal" begin
        c, n = References.load("tnorm")
        for i in 1:n
            d = trunc_normal(c["location"][i], c["scale"][i], c["lower"][i], c["upper"][i])
            @test crps(d, c["y"][i]) ≈ c["crps"][i] atol=atol rtol=rtol
        end
    end

    @testset "censored Normal" begin
        c, n = References.load("cnorm")
        for i in 1:n
            d = cens_normal(c["location"][i], c["scale"][i], c["lower"][i], c["upper"][i])
            @test crps(d, c["y"][i]) ≈ c["crps"][i] atol=atol rtol=rtol
        end
    end

    @testset "generalised truncated/censored Normal" begin
        c, n = References.load("gtcnorm")
        for i in 1:n
            got = _crps_gtcnorm(c["y"][i], c["location"][i], c["scale"][i],
                c["lower"][i], c["upper"][i], c["lmass"][i], c["umass"][i])
            @test got ≈ c["crps"][i] atol=atol rtol=rtol
        end
    end
end
