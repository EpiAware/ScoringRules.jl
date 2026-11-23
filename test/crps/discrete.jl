@testitem "discrete family scores match R scoringRules" tags = [:crps] setup = [References] begin
    using ScoringRules
    using Distributions

    atol = 1.0e-9
    rtol = 1.0e-8

    @testset "Poisson" begin
        c, n = References.load("pois")
        for i in 1:n
            d = Poisson(c["lambda"][i])
            y = c["y"][i]
            @test crps(d, y) ≈ c["crps"][i] atol = atol rtol = rtol
            @test logs(d, y) ≈ c["logs"][i] atol = atol rtol = rtol
            @test dss(d, y) ≈ c["dss"][i] atol = atol rtol = rtol
        end
    end

    @testset "NegativeBinomial" begin
        c, n = References.load("nbinom")
        for i in 1:n
            # Distributions.NegativeBinomial(r, p): r = size (successes), p = prob
            d = NegativeBinomial(c["size"][i], c["prob"][i])
            y = c["y"][i]
            @test crps(d, y) ≈ c["crps"][i] atol = atol rtol = rtol
            @test logs(d, y) ≈ c["logs"][i] atol = atol rtol = rtol
            @test dss(d, y) ≈ c["dss"][i] atol = atol rtol = rtol
        end
    end

    @testset "Binomial" begin
        c, n = References.load("binom")
        for i in 1:n
            d = Binomial(Int(c["size"][i]), c["prob"][i])
            y = c["y"][i]
            @test crps(d, y) ≈ c["crps"][i] atol = atol rtol = rtol
            @test logs(d, y) ≈ c["logs"][i] atol = atol rtol = rtol
        end
    end

    @testset "Hypergeometric" begin
        c, n = References.load("hyper")
        for i in 1:n
            # R: phyper(y, m, n, k) — m=white, n=black, k=draws
            # Distributions: Hypergeometric(s, f, n) — s=successes(white), f=failures(black), n=draws
            m = Int(c["m"][i])
            nblack = Int(c["n"][i])
            k = Int(c["k"][i])
            d = Hypergeometric(m, nblack, k)
            y = c["y"][i]
            @test crps(d, y) ≈ c["crps"][i] atol = atol rtol = rtol
            @test logs(d, y) ≈ c["logs"][i] atol = atol rtol = rtol
        end
    end
end

@testitem "the Poisson CRPS takes a NaN, infinite or huge observation as R does" tags = [:crps] begin
    using ScoringRules
    using Distributions

    # R: crps_pois(y, lambda). Past the range of `Int` the score is
    # `|y - λ|` less half the forecast's mean absolute difference, which R
    # returns to full precision.
    for λ in (0.5, 3.0, 50.0)
        @test isnan(crps(Poisson(λ), NaN))
        @test crps(Poisson(λ), Inf) == Inf
        @test crps(Poisson(λ), -Inf) == Inf
        for y in (1.0e30, -1.0e30, 1.0e19, -1.0e19, 9.1e18, 1.0e18)
            @test crps(Poisson(λ), y) == abs(y)
        end
    end
    @test crps(Poisson(3.0), 2.5) ≈ 0.46493408896051547 rtol = 1.0e-14
    @test crps(Poisson(50.0), -1.0) ≈ 47.015573383788656 rtol = 1.0e-14
end
