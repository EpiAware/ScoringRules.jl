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
