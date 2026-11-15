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

@testitem "the generic discrete CRPS sums a forecast at the points it takes" tags = [:crps] begin
    using ScoringRules
    using Distributions

    # A forecast off the integers, whether a mixture with a shifted component
    # or a shifted forecast inside a truncation or censoring, has its mass
    # where its own pieces put it. The reference integrates the step CDF over
    # the true points and masses, written out by hand.
    function stepcrps(xs, ps, y)
        o = sortperm(xs)
        xs, F = xs[o], cumsum(ps[o])
        s = max(first(xs) - y, 0.0)
        for i in 1:(length(xs) - 1)
            a, b = xs[i], xs[i + 1]
            s += y <= a ? (1 - F[i])^2 * (b - a) : y >= b ? F[i]^2 * (b - a) :
                F[i]^2 * (y - a) + (1 - F[i])^2 * (b - y)
        end
        return s + max(y - last(xs), 0.0)
    end
    P = Poisson(2.0)
    k = 0:80
    pk = pdf.(P, k)
    @test crps(MixtureModel([P, 3.5 + P]), 4.0) ≈
        stepcrps(vcat(collect(Float64, k), 3.5 .+ k), vcat(0.5 .* pk, 0.5 .* pk), 4.0) rtol = 1.0e-10
    xt, pt = 3.5 .+ k[2:end], pk[2:end] ./ sum(pk[2:end])
    @test crps(truncated(3.5 + P, 4, Inf), 6.0) ≈ stepcrps(xt, pt, 6.0) rtol = 1.0e-10
    xc, pc = [3.5, 4.5, 5.5, 6.5, 7.0], [pk[1:4]; 1 - sum(pk[1:4])]
    c = censored(3.5 + P; upper = 7)
    @test crps(c, 6.0) ≈ stepcrps(xc, pc, 6.0) rtol = 1.0e-10
    m = sum(pc .* xc)
    v = sum(pc .* (xc .- m) .^ 2)
    @test dss(c, 6.0) ≈ (6.0 - m)^2 / v + log(v) rtol = 1.0e-10
    # However small its weight, a component off the integers is counted.
    b = Binomial(60, 0.3)
    pb = pdf.(b, 0:60)
    for w in (2.0e-5, 9.0e-4, 0.005)
        d = MixtureModel([b, DiscreteNonParametric([30.5], [1.0])], [1 - w, w])
        @test crps(d, 18.0) ≈ stepcrps([collect(0.0:60.0); 30.5], [(1 - w) .* pb; w], 18.0) rtol = 1.0e-10
    end
    # Forecasts held in Float32 score as their Float64 counterparts, at any
    # support size.
    for (d32, d64, y) in (
            (truncated(Poisson(2.0f0), 0, 20), truncated(Poisson(2.0), 0, 20), 4.0),
            (truncated(Binomial(30, 0.3f0), 2, 25), truncated(Binomial(30, 0.3), 2, 25), 4.0),
            (truncated(Poisson(2.0f0), 1, Inf), truncated(Poisson(2.0), 1, Inf), 4.0),
            (
                MixtureModel([Binomial(20000, 0.5f0), Binomial(20000, 0.45f0)], Float32[0.5, 0.5]),
                MixtureModel([Binomial(20000, 0.5), Binomial(20000, 0.45)], [0.5, 0.5]), 9500.0,
            ),
        )
        @test crps(d32, y) ≈ crps(d64, y) rtol = 1.0e-5
    end
    @test isfinite(crps(MixtureModel([Binomial(300000, 0.5f0)], Float32[1]), 150000.0))
    # A scale that does not round-trip, so the CDF is read between nodes
    # rather than at them.
    L = 0.1 * P
    xl = 0.1 .* k
    tl = findall(x -> 0 <= x <= 2, xl)
    @test crps(truncated(L, 0, 2), 0.25) ≈ stepcrps(xl[tl], pk[tl] ./ sum(pk[tl]), 0.25) rtol = 1.0e-10
    @test crps(MixtureModel([P, L]), 1.0) ≈
        stepcrps(vcat(collect(Float64, k), xl), vcat(0.5 .* pk, 0.5 .* pk), 1.0) rtol = 1.0e-10
    pl = pk[tl] ./ sum(pk[tl])
    ml = sum(pl .* xl[tl])
    vl = sum(pl .* (xl[tl] .- ml) .^ 2)
    @test dss(truncated(L, 0, 2), 0.25) ≈ (0.25 - ml)^2 / vl + log(vl) rtol = 1.0e-10
    # Too many points to list is refused, not guessed at.
    @test_throws ArgumentError crps(MixtureModel([P, 2.0^-30 * P]), 1.0)
    # A mixture whose components both sit on the integers.
    @test crps(MixtureModel([P, 2 + P]), 4.0) ≈
        stepcrps(vcat(collect(Float64, k), 2.0 .+ k), vcat(0.5 .* pk, 0.5 .* pk), 4.0) rtol = 1.0e-10
end

@testitem "wrapped discrete forecasts keep their speed and their window" tags = [:crps] begin
    using ScoringRules
    using Distributions

    # The moments of a wrapped empirical forecast take its masses from `pdf`,
    # a binary search, not from `cdf`, which scans the whole support.
    n = 100_000
    d = truncated(DiscreteNonParametric(collect(1.0:n), fill(1 / n, n)), 0, n + 1)
    dss(d, 5.0)
    @test (@elapsed dss(d, 5.0)) < 0.5
    @test dss(d, 5.0) ≈ dss(collect(1.0:n), 5.0) rtol = 1.0e-9
    # A finite bound far from the mass is cut at the tail like an infinite
    # one: a zero-truncated Poisson with a large mean has nothing near zero.
    @test crps(truncated(Poisson(2.0e7); lower = 1), 2.0e7) ≈ crps(Poisson(2.0e7), 2.0e7) rtol = 1.0e-9
    # Mass beyond the reach of the tail search keeps the forecast's own bounds.
    for s in (1, -1)
        x = s * 3.0e18
        e = DiscreteNonParametric(sort([x, x + 1024]), [0.5, 0.5])
        @test crps(e, x) == 256.0
        @test crps(Dirac(x), x) == 0.0
        @test dss(truncated(e, -4.0e18, 4.0e18), x) ≈ log(512.0^2) + 1
    end
    # The cap on the window applies to a window of integers, not to a few
    # listed points far apart.
    @test crps(DiscreteNonParametric([0.0, 3.0e9], [0.5, 0.5]), 1.0e9) ≈ crps([0.0, 3.0e9], 1.0e9)
    @test crps(MixtureModel([Dirac(0.0), Dirac(1.0e10)]), 0.0) ≈ crps([0.0, 1.0e10], 0.0)
    @test_throws ArgumentError crps(Geometric(1.0e-9), 1.0)
end
