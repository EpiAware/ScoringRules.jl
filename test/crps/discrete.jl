@testitem "discrete family scores match R scoringRules" tags=[:crps] setup=[References] begin
    using Distributions

    atol=1e-9
    rtol=1e-8

    @testset "Poisson" begin
        c, n = References.load("pois")
        for i in 1:n
            d = Poisson(c["lambda"][i])
            y = c["y"][i]
            @test crps(d, y)≈c["crps"][i] atol=atol rtol=rtol
            @test logs(d, y)≈c["logs"][i] atol=atol rtol=rtol
            @test dss(d, y)≈c["dss"][i] atol=atol rtol=rtol
        end
    end

    @testset "NegativeBinomial" begin
        c, n = References.load("nbinom")
        for i in 1:n
            # Distributions.NegativeBinomial(r, p): r = size (successes), p = prob
            d = NegativeBinomial(c["size"][i], c["prob"][i])
            y = c["y"][i]
            @test crps(d, y)≈c["crps"][i] atol=atol rtol=rtol
            @test logs(d, y)≈c["logs"][i] atol=atol rtol=rtol
            @test dss(d, y)≈c["dss"][i] atol=atol rtol=rtol
        end
    end

    @testset "Binomial" begin
        c, n = References.load("binom")
        for i in 1:n
            d = Binomial(Int(c["size"][i]), c["prob"][i])
            y = c["y"][i]
            @test crps(d, y)≈c["crps"][i] atol=atol rtol=rtol
            @test logs(d, y)≈c["logs"][i] atol=atol rtol=rtol
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
            @test crps(d, y)≈c["crps"][i] atol=atol rtol=rtol
            @test logs(d, y)≈c["logs"][i] atol=atol rtol=rtol
        end
    end
end
