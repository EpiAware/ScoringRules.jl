@testitem "RPS matches R scoringRules rps_probs" setup = [References] begin
    using ScoringRules
    atol = 1.0e-9
    rtol = 1.0e-7

    c, n = References.load("rps_scores")

    @testset "rps" begin
        for i in 1:n
            K = Int(c["K"][i])
            y = Int(c["y"][i])
            # Reconstruct probability vector from p1..p4 columns.
            p_vals = [c["p$j"][i] for j in 1:K]
            ref = c["rps"][i]
            @test rps(p_vals, y) ≈ ref atol = atol rtol = rtol
        end
    end
end

@testitem "rps rejects anything that is not a probability vector" tags = [:sample] begin
    using ScoringRules

    # Handed counts, the score comes out as 106.0 without the check. R's
    # `rps_probs` errors on a vector that is negative or does not sum to one.
    @test_throws ArgumentError rps([3.0, 2.0, 5.0], 2)
    @test_throws ArgumentError rps([-0.1, 0.6, 0.5], 2)
    @test_throws ArgumentError rps([0.2, 0.2, 0.2], 2)
    @test_throws ArgumentError rps([0.5, Inf], 1)
    @test_throws ArgumentError rps([1.0], 1)      # K ≥ 2, as in R
    # Rounding a user cannot avoid still passes.
    p = fill(1 / 7, 7)
    @test rps(p ./ sum(p), 3) ≈ rps(p, 3)
end

@testitem "rps accepts a probability vector normalised in lower precision" tags = [:sample] begin
    using ScoringRules
    using Random: Xoshiro

    # A Float32 vector divided by its own sum lands within a few Float32 eps
    # of one, far more than 1e-10; it is still a probability vector.
    rng = Xoshiro(1)
    for k in (3, 20, 200)
        p = rand(rng, Float32, k)
        p ./= sum(p)
        @test rps(p, 2) ≈ rps(Float64.(p) ./ sum(Float64.(p)), 2) rtol = 1.0e-5
    end
    # A Float32 vector that is plainly not normalised is still refused.
    @test_throws ArgumentError rps(Float32[0.2, 0.2, 0.5], 2)
end
