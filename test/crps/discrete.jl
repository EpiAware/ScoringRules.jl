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

@testitem "a count forecast censored between the integers keeps its point masses" tags = [:crps] begin
    using ScoringRules
    using Distributions

    # Censoring at 1.5 piles the mass below it onto 1.5 itself, but
    # `support` lists only the integers 2:6. The step CDF jumps at the bound
    # as well as at the integers, and the moments include the mass there.
    function stepcrps(d, y, pts)
        xs = sort(pts)
        s = max(first(xs) - y, 0.0)
        for i in 1:(length(xs) - 1)
            a, b, F = xs[i], xs[i + 1], cdf(d, xs[i])
            s += y <= a ? (1 - F)^2 * (b - a) : y >= b ? F^2 * (b - a) :
                F^2 * (y - a) + (1 - F)^2 * (b - y)
        end
        return s + max(y - last(xs), 0.0)
    end
    for (lo, hi, pts) in ((1.5, 6.5, [1.5; 2:6; 6.5]), (1.5, nothing, [1.5; 2:60]), (nothing, 4.5, [0:4; 4.5]))
        d = censored(Poisson(3.0); lower = lo, upper = hi)
        for y in (0.0, 1.7, 3.0, 5.2, 9.0)
            @test crps(d, y) ≈ stepcrps(d, y, pts) rtol = 1.0e-10
        end
        p = [pdf(d, x) for x in pts]
        m = sum(p .* pts)
        v = sum(p .* (pts .- m) .^ 2)
        @test dss(d, 2.0) ≈ (2.0 - m)^2 / v + log(v) rtol = 1.0e-10
    end
end
