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

@testitem "Poisson CRPS at large means" tags = [:crps] begin
    using ScoringRules
    using Distributions

    # CRPS = ∫ (F(x) - 1{x ≥ y})² dx, with F constant on each [k, k + 1).
    function crps_by_summation(λ, y)
        d = Poisson(λ)
        σ = sqrt(λ)
        lo = max(0, floor(Int, λ - 14σ))
        hi = ceil(Int, λ + 14σ)
        return sum(k -> (cdf(d, k) - (k >= y ? 1.0 : 0.0))^2, lo:hi)
    end

    @testset "agrees with direct summation across the switch to the expansion" begin
        for (λ, y) in (
                (4.9e5, 4.9e5), (5.1e5, 5.08e5), (2.0e6, 2.0e6),
                (1.0e7, 1.0003e7), (1.0e8, 1.0e8),
            )
            @test crps(Poisson(λ), y) ≈ crps_by_summation(λ, y) rtol = 1.0e-8
        end
    end

    @testset "approaches the normal limit past the Amos range" begin
        for λ in (6.0e8, 1.0e9, 1.0e12, 1.0e15)
            σ = sqrt(λ)
            @test crps(Poisson(λ), λ) ≈ σ * (sqrt(2) - 1) / sqrt(π) rtol = 1.0e-4
        end
    end

    @testset "is differentiable in the mean" begin
        using ForwardDiff
        λ = 1.0e7
        g = ForwardDiff.derivative(m -> crps(Poisson(m), λ), λ)
        h = 30.0
        fd = (crps(Poisson(λ + h), λ) - crps(Poisson(λ - h), λ)) / 2h
        @test g ≈ fd rtol = 1.0e-3
    end
end
