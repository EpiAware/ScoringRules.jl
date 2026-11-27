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
    using ForwardDiff

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
    # The early returns keep the dual numbers attached to `y` and `λ`.
    @test isnan(ForwardDiff.derivative(λ -> crps(Poisson(λ), NaN), 3.0))
    @test isnan(ForwardDiff.derivative(y -> crps(Poisson(3.0), y), NaN))
    @test ForwardDiff.derivative(λ -> crps(Poisson(λ), Inf), 3.0) == -1
    @test ForwardDiff.derivative(y -> crps(Poisson(3.0), y), -Inf) == -1
    @test crps(Poisson(3.0), 2.5) ≈ 0.46493408896051547 rtol = 1.0e-14
    @test crps(Poisson(50.0), -1.0) ≈ 47.015573383788656 rtol = 1.0e-14
end

@testitem "the negative binomial CRPS holds up at large sizes" tags = [:crps] begin
    using ScoringRules
    using Distributions

    rtol = 1.0e-9

    # The Gauss series for the hypergeometric factor loses digits from a size
    # of about 15 and returns NaN from about 100. Reference: crps_nbinom(y,
    # size, prob) in R scoringRules, which brute-force summation agrees with
    # to 1e-12.
    @test crps(NegativeBinomial(40.0, 0.3), 5.0) ≈ 78.4150319255607 rtol = rtol
    @test crps(NegativeBinomial(100.0, 0.3), 5.0) ≈ 212.61976265306 rtol = rtol
    @test crps(NegativeBinomial(100.0, 0.5), 5.0) ≈ 87.0336256579815 rtol = rtol
    @test crps(NegativeBinomial(200.0, 0.3), 5.0) ≈ 439.429528293562 rtol = rtol
    @test crps(NegativeBinomial(200.0, 0.3), 400.0) ≈ 45.6067195156157 rtol = rtol
    @test crps(NegativeBinomial(1000.0, 0.3), 5.0) ≈ 2278.5831081912 rtol = rtol
    @test crps(NegativeBinomial(1.0e4, 0.5), 1.0e4) ≈ 33.0492968621174 rtol = rtol
    @test crps(NegativeBinomial(723.0, 0.01), 71580.0) ≈ 625.270617245274 rtol = rtol
    @test crps(NegativeBinomial(724.0, 0.01), 71680.0) ≈ 625.713745875955 rtol = rtol
    @test crps(NegativeBinomial(200.0, 0.002), 1.0e5) ≈ 1656.9796845365 rtol = rtol
    @test crps(NegativeBinomial(80.0, 0.732), 25.0) ≈ 2.46701965460339 rtol = rtol

    # Half-integer sizes, where the series is right at small sizes and R
    # returns -Inf for some (2.5 and 5.5 at prob = 0.3, 10.5 at prob = 0.7).
    # Reference: brute-force Σ_k (F(k) − 1{k ≥ y})² in 160-bit arithmetic.
    @test crps(NegativeBinomial(40.5, 0.3), 5.0) ≈ 79.51949131048526 rtol = rtol
    @test crps(NegativeBinomial(5.5, 0.3), 5.0) ≈ 4.498434096836827 rtol = rtol
    @test crps(NegativeBinomial(2.5, 0.3), 5.0) ≈ 0.9502816784913554 rtol = rtol
    @test crps(NegativeBinomial(10.5, 0.7), 5.0) ≈ 0.6707050848143765 rtol = rtol

    # From a size of about 1e6 R's own hypergeometric factor drifts, by 9e-9
    # here and 7e-3 at a size of 1e12, so these are brute-force sums as above.
    # The closed form keeps about 1e-9 at 1e12, where it subtracts terms of
    # order 1e12 to leave one of order 1e5.
    @test crps(NegativeBinomial(2.1e6, 0.5), 2.1e6) ≈ 478.93217393518677 rtol = rtol
    @test crps(NegativeBinomial(1.0e12, 0.5), 1.0e12) ≈ 330494.60629263084 rtol = 1.0e-8

    # A near-Poisson forecast written with a huge size tends to the Poisson of
    # its mean, within about mean/size.
    for (r, mu, y) in ((1.0e11, 5.0, 3.0), (1.0e12, 0.01, 0.0), (2.0e10, 1.0, 1.0))
        d = NegativeBinomial(r, r / (r + mu))
        @test crps(d, y) ≈ crps(Poisson(mean(d)), y) rtol = 1.0e-9
    end
end

@testitem "the negative binomial CRPS keeps the Gauss series only where it is accurate" tags = [:crps] begin
    using ScoringRules
    using Distributions
    using ScoringRules: _f21_nbinom, _₂F₁

    # Below `a = 16` or `a·Z = 6` the series is within about 1e-13 and is
    # used as it is.
    for (a, Z) in (
            (2.0, 1.0e6), (10.5, 31.1), (14.5, 0.5), (15.99, 1.0e3),
            (1001.0, 5.0e-3), (1.0e6, 5.0e-6),
        )
        @test _f21_nbinom(a, Z) === _₂F₁(a, 0.5, 2.0, -Z)
    end
    # Elsewhere, against `₂F₁(a, ½; 2; −Z)` from its Euler integral in 200-bit
    # arithmetic. The first eight take the Pfaff series, the rest the Laguerre
    # rule; the comment gives the Gauss series' relative error where it is
    # above 1e-14.
    for (a, Z, ref) in (
            (16.0, 1.0, 0.2837910598522961),
            (19.0, 0.56, 0.3434993119452377),    # 1.1e-12
            (16.0, 8.4, 0.09948463718372556),
            (201.0, 0.1, 0.2489317796263936),    # 6.0e-10
            (51.0, 0.5, 0.22280127792966553),    # 9.4e-6
            (101.0, 0.25, 0.2230921986790706),   # 8.0e-7
            (81.0, 0.5, 0.17700245218904695),    # 2.4e4
            (1001.0, 0.01, 0.34746415664814895), # 1.6e-14
            (16.5, 8.0, 0.10030755659410587),
            (201.0, 0.5, 0.11248370833730106),   # 1.4e40
            (101.0, 31.1, 0.02020677039417542),  # NaN
            (16.0, 1.0e6, 0.0002889288912072031),
        )
        @test _f21_nbinom(a, Z) ≈ ref rtol = 2.0e-14
    end

    # The score against Σ_k (F(k) − 1{k ≥ y})² across sizes and probabilities
    # on both sides of the switch, the observation at zero and in the body.
    function lattice_crps(d, y)
        lo = max(0, floor(Int, quantile(d, 1.0e-18)) - 2)
        hi = ceil(Int, cquantile(d, 1.0e-18)) + 2
        # Below the window `F(k)` is under 1e-18: each term with `k ≥ y` is one.
        s = float(max(0, lo - max(0, ceil(Int, y))))
        for k in lo:hi
            s += k < y ? cdf(d, k)^2 : ccdf(d, k)^2
        end
        return s
    end
    for r in (10.0, 14.0, 15.0, 40.0, 100.0, 1000.0, 1.0e4, 1.0e5)
        for p in (1.0e-3, 0.05, 0.3, 0.7, 0.95, 0.999, 1 - 1.0e-6)
            d = NegativeBinomial(r, p)
            std(d) > 1.0e4 && continue
            m, sd = mean(d), std(d)
            for y in unique(max.(0.0, round.((0.0, m - 2sd, m, m + 3sd))))
                @test crps(d, y) ≈ lattice_crps(d, y) rtol = 1.0e-10
            end
        end
    end
    # Sizes 15 to 50 with a `prob` near 0.9, where the series has already lost
    # digits but not yet failed: 4e-11 at size 15 and `a·Z = 11`, 4e-7 at size
    # 28.5 and `a·Z = 18`. The Laguerre rule loses digits below `a·log1p(Z) =
    # 36` in the same way, 6e-11 at size 50 and 20.5. Here the score is within
    # 3e-14 of the lattice sum.
    for (r, p) in (
            (15.0, 0.87), (20.0, 0.9), (25.0, 0.78), (25.0, 0.88),
            (28.5, 0.88), (29.5, 0.88), (50.0, 0.88), (50.0, 0.9),
        )
        d = NegativeBinomial(r, p)
        m, sd = mean(d), std(d)
        for y in unique(max.(0.0, round.((0.0, m - 2sd, m, m + 3sd))))
            @test crps(d, y) ≈ lattice_crps(d, y) rtol = 1.0e-12
        end
    end
end
