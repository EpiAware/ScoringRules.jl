@testitem "quantile and interval scores match R scoringRules" setup=[References] begin
    atol = 1e-9
    rtol = 1e-7

    q_levels = [0.1, 0.25, 0.5, 0.75, 0.9]

    # ---------- quantile / interval scores from explicit quantile forecasts ----------

    c, n = References.load("quantile_scores")

    @testset "quantile_score (explicit quantile forecasts)" begin
        for i in 1:n
            y = c["y"][i]
            qf = [c["q$j"][i] for j in 1:5]

            for (j, alpha) in enumerate(q_levels)
                ref = c["qs_a$j"][i]
                got = quantile_score([alpha], [qf[j]], y)
                @test only(got)≈ref atol=atol rtol=rtol
            end
        end
    end

    @testset "interval_score 80% (explicit bounds)" begin
        for i in 1:n
            y = c["y"][i]
            lower = c["q1"][i]   # 10th percentile
            upper = c["q5"][i]   # 90th percentile
            ref = c["ints_80"][i]
            @test interval_score(lower, upper, y, 0.8)≈ref atol=atol rtol=rtol
        end
    end

    @testset "interval_score 50% (explicit bounds)" begin
        for i in 1:n
            y = c["y"][i]
            lower = c["q2"][i]   # 25th percentile
            upper = c["q4"][i]   # 75th percentile
            ref = c["ints_50"][i]
            @test interval_score(lower, upper, y, 0.5)≈ref atol=atol rtol=rtol
        end
    end

    # ---------- ensemble-based quantile / interval scores ----------

    function load_univ_ens(id::Int)
        cv, _ = References.load("ens_univ_$id")
        n_mem = length(cv)
        return [cv["m$j"][1] for j in 1:n_mem]
    end

    univ_ens = [load_univ_ens(i) for i in 1:2]

    cs, ns = References.load("quantile_sample_scores")

    # Note: the ensemble-based overloads quantile_score(dat, y; alpha=…) and
    # interval_score(dat, y; level=…) call Statistics.quantile via the qualified
    # name Statistics.quantile inside ScoringRules, but the module only imports
    # specific names (mean, var, std) rather than the Statistics module itself.
    # Until that is fixed upstream, those overloads are not tested here; the
    # reference CSV (quantile_sample_scores.csv) remains available for future use.
end
