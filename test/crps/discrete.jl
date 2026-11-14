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

@testitem "a wrapped discrete forecast off the integers is scored on its own support" tags = [:crps] begin
    using ScoringRules
    using Distributions

    # `support` of a truncated or censored discrete forecast assumes the
    # integers, `round(minimum):round(maximum)`, which for a forecast on
    # non-integer points lists none of them. Truncated, an equally weighted
    # forecast is the ensemble of the members it keeps.
    x = [-1.0, -0.3, 0.1, 0.4, 0.9, 1.5, 2.0]
    d = DiscreteNonParametric(x, fill(1 / 7, 7))
    t = truncated(d; lower = -0.5, upper = 1.2)
    for y in (-2.0, 0.3, 1.0, 3.0)
        @test crps(t, y) ≈ crps([-0.3, 0.1, 0.4, 0.9], y) rtol = 1.0e-12
    end
    @test dss(t, 0.3) ≈ dss([-0.3, 0.1, 0.4, 0.9], 0.3) rtol = 1.0e-12
end

@testitem "an empirical discrete forecast scores in time linear in its support" tags = [:crps] begin
    using ScoringRules
    using Distributions
    using Random: Xoshiro

    # `cdf` of a `DiscreteNonParametric` scans its support, so calling it at
    # every node made the score quadratic: 3.7 s at a hundred thousand
    # points. An equally weighted one is an ensemble, which is the reference.
    x = sort(randn(Xoshiro(1), 100_000))
    d = DiscreteNonParametric(x, fill(1 / length(x), length(x)))
    crps(DiscreteNonParametric(x[1:10], fill(0.1, 10)), 0.1)
    @test crps(d, 0.1) ≈ crps(x, 0.1) rtol = 1.0e-10
    @test (@elapsed crps(d, 0.1)) < 0.5
    t = truncated(d; lower = -0.5, upper = 1.2)
    crps(t, 0.3)
    @test crps(t, 0.3) ≈ crps(filter(v -> -0.5 <= v <= 1.2, x), 0.3) rtol = 1.0e-9
    @test (@elapsed crps(t, 0.3)) < 0.5
end

@testitem "the generic discrete CRPS refuses a forecast with mass between its nodes" tags = [:crps] begin
    using ScoringRules
    using Distributions

    # The step CDF can only jump at the nodes it is summed on, so a forecast
    # with mass between them would be scored as if that mass were not there.
    # A mixture with a shifted component, or a shifted forecast inside a
    # truncation or censoring, is placed on the integers it does not take.
    P = Poisson(2.0)
    for d in (MixtureModel([P, 3.5 + P]), MixtureModel([Binomial(6, 0.3), 0.5 + Binomial(6, 0.3)]))
        @test_throws ArgumentError crps(d, 4.0)
    end
    # A wrapped one reaches the same nodes for its moments; a mixture takes
    # its moments from Distributions.jl, exactly, and its `dss` stands.
    for d in (truncated(3.5 + P, 4, Inf), censored(3.5 + P; upper = 7))
        @test_throws ArgumentError crps(d, 4.0)
        @test_throws ArgumentError dss(d, 4.0)
    end
    # A forecast held in Float32 sums its masses with Float32 rounding, which
    # is no evidence of mass between the nodes.
    for (d32, d64) in (
            (truncated(Poisson(2.0f0), 0, 20), truncated(Poisson(2.0), 0, 20)),
            (truncated(Binomial(30, 0.3f0), 2, 25), truncated(Binomial(30, 0.3), 2, 25)),
            (truncated(Poisson(2.0f0), 1, Inf), truncated(Poisson(2.0), 1, Inf)),
        )
        @test crps(d32, 4.0) ≈ crps(d64, 4.0) rtol = 1.0e-5
        @test dss(d32, 4.0) ≈ dss(d64, 4.0) rtol = 1.0e-5
    end
    # A mixture whose components both sit on the integers is scored, and
    # matches E|X - y| - E|X - X'|/2 over its enumerated support.
    k = 0:60
    xs = vcat(collect(Float64, k), 2.0 .+ k)
    ps = vcat(0.5 .* pdf.(P, k), 0.5 .* pdf.(P, k))
    y = 4.0
    exact = sum(ps .* abs.(xs .- y)) -
        0.5 * sum(ps[i] * ps[j] * abs(xs[i] - xs[j]) for i in eachindex(xs), j in eachindex(xs))
    @test crps(MixtureModel([P, 2 + P]), y) ≈ exact rtol = 1.0e-10
end
