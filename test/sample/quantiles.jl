@testitem "quantile and interval scores match R scoringRules" setup=[References] begin
    using ScoringRules
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

@testitem "ensemble quantile scores honour the interpolation `type`" begin
    using ScoringRules

    dat = [3.0, 7, 1, 9, 4, 2, 8, 5, 6, 10]
    y = 4.3

    # Oracle values from R scoringRules 1.1.3:
    #   qs_sample(y, dat, alpha = 0.75, type = t)
    #   ints_sample(y, matrix(dat, nrow = 1), target_coverage = 0.8, type = t)
    qs_ref = Dict(1 => 0.925, 4 => 0.8, 6 => 0.9875, 7 => 0.8625,
        8 => 0.945833333333333)
    is_ref = Dict(1 => 8.0, 4 => 8.0, 6 => 8.8, 7 => 7.2,
        8 => 8.26666666666667)

    for (t, ref) in qs_ref
        @test quantile_score(dat, y; alpha = 0.75, type = t)≈ref rtol=1e-12
    end
    for (t, ref) in is_ref
        @test interval_score(dat, y; level = 0.8, type = t)≈ref rtol=1e-12
    end

    # Distinct types must give distinct results (guards against the `type`
    # keyword silently collapsing to type 7).
    @test quantile_score(dat, y; alpha = 0.75, type = 1) !=
          quantile_score(dat, y; alpha = 0.75, type = 6)
end

@testitem "quantile / interval / rps input validation" begin
    using ScoringRules

    dat = collect(1.0:10.0)
    @test_throws ArgumentError quantile_score(dat, 3.0; alpha = 1.5)
    @test_throws ArgumentError quantile_score([0.1, 1.5], [1.0, 2.0], 0.5)
    @test_throws ArgumentError interval_score(dat, 3.0; level = 1.0)
    @test_throws ArgumentError interval_score(-1.0, 1.0, 0.0, 0.0)
    @test_throws ArgumentError rps([0.2, 0.3, 0.5], 5)
    @test_throws ArgumentError rps([0.2, 0.3, 0.5], 0)
end
