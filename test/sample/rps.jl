@testitem "RPS matches R scoringRules rps_probs" setup=[References] begin

    atol = 1e-9
    rtol = 1e-7

    c, n = References.load("rps_scores")

    @testset "rps" begin
        for i in 1:n
            K  = Int(c["K"][i])
            y  = Int(c["y"][i])
            # Reconstruct probability vector from p1..p4 columns.
            p_vals = [c["p$j"][i] for j in 1:K]
            ref = c["rps"][i]
            @test rps(p_vals, y) ≈ ref atol=atol rtol=rtol
        end
    end
end
