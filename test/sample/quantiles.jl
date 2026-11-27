@testitem "quantile and interval scores match R scoringRules" setup = [References] begin
    using ScoringRules
    atol = 1.0e-9
    rtol = 1.0e-7

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
                @test got ≈ ref atol = atol rtol = rtol
            end
        end
    end

    @testset "interval_score 80% (explicit bounds)" begin
        for i in 1:n
            y = c["y"][i]
            lower = c["q1"][i]   # 10th percentile
            upper = c["q5"][i]   # 90th percentile
            ref = c["ints_80"][i]
            @test interval_score(lower, upper, y, 0.8) ≈ ref atol = atol rtol = rtol
        end
    end

    @testset "interval_score 50% (explicit bounds)" begin
        for i in 1:n
            y = c["y"][i]
            lower = c["q2"][i]   # 25th percentile
            upper = c["q4"][i]   # 75th percentile
            ref = c["ints_50"][i]
            @test interval_score(lower, upper, y, 0.5) ≈ ref atol = atol rtol = rtol
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

    @testset "quantile_score / interval_score (ensemble)" begin
        for i in 1:ns
            dat = univ_ens[Int(cs["ens_id"][i])]
            y = cs["y"][i]
            got_qs = quantile_score(dat, y; alpha = cs["alpha"][i])
            @test got_qs ≈ cs["qs"][i] atol = atol rtol = rtol
            got_is = interval_score(dat, y; level = 0.8)
            @test got_is ≈ cs["ints_80"][i] atol = atol rtol = rtol
        end
    end

    # ---------- near-integer n*p boundary (quantile index fuzz) ----------
    # At coverage 0.9 the lower level 0.5 * (1 - 0.9) puts n*p just below an
    # integer (24.99… at n = 500) and the upper level just above one, so the
    # scores are only correct if the quantile helper mirrors R's index fuzz.
    # The ensembles are 1:n, matching generate_references.R exactly.

    cf, nf = References.load("quantile_fuzz_scores")

    @testset "ensemble scores at near-integer n*p" begin
        alpha_lo = 0.5 * (1 - 0.9)
        alpha_hi = 1 - alpha_lo
        for i in 1:nf
            dat = collect(1.0:cf["n"][i])
            t = Int(cf["type"][i])
            y = cf["y"][i]
            @test quantile_score(dat, y; alpha = alpha_lo, type = t) ≈ cf["qs_lo"][i] atol = atol rtol = rtol
            @test quantile_score(dat, y; alpha = alpha_hi, type = t) ≈ cf["qs_hi"][i] atol = atol rtol = rtol
            @test interval_score(dat, y; level = 0.9, type = t) ≈ cf["ints_90"][i] atol = atol rtol = rtol
        end
    end
end

@testitem "ensemble quantile scores honour the interpolation `type`" begin
    using ScoringRules

    dat = [3.0, 7, 1, 9, 4, 2, 8, 5, 6, 10]
    y = 4.3

    # Oracle values from R scoringRules 1.1.3:
    #   qs_sample(y, dat, alpha = 0.75, type = t)
    #   ints_sample(y, matrix(dat, nrow = 1), target_coverage = 0.8, type = t)
    qs_ref = Dict(
        1 => 0.925, 4 => 0.8, 6 => 0.9875, 7 => 0.8625,
        8 => 0.945833333333333
    )
    is_ref = Dict(
        1 => 8.0, 4 => 8.0, 6 => 8.8, 7 => 7.2,
        8 => 8.26666666666667
    )

    for (t, ref) in qs_ref
        @test quantile_score(dat, y; alpha = 0.75, type = t) ≈ ref rtol = 1.0e-12
    end
    for (t, ref) in is_ref
        @test interval_score(dat, y; level = 0.8, type = t) ≈ ref rtol = 1.0e-12
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

@testitem "quantile_score reduces the per-level scores" begin
    using ScoringRules

    levels = [0.1, 0.5, 0.9]
    q = [-1.28, 0.0, 1.28]
    y = 1.0

    per_level = quantile_score(levels, q, y; reduce = identity)
    @test per_level isa Vector{Float64}
    @test length(per_level) == length(levels)

    # The headline call returns a scalar, as every other score here does.
    got = quantile_score(levels, q, y)
    @test got isa Real
    @test got ≈ sum(per_level) / length(per_level)

    @test quantile_score(levels, q, y; reduce = sum) ≈ sum(per_level)

    # A single level reduces to that level's pinball loss either way.
    for (i, alpha) in enumerate(levels)
        @test quantile_score([alpha], [q[i]], y) ≈ per_level[i]
    end
end

@testitem "quantile and interval scores do not wrap integer inputs" tags = [:sample] begin
    using ScoringRules

    # An unsigned or narrow integer difference would wrap in its own type.
    # R gives 22 for ints_quantiles(5, 1, 3, 0.8).
    @test interval_score(UInt32(1), UInt32(3), UInt32(5), 0.8) ≈ 22 rtol = 1.0e-12

    # Inputs near the limits of each type, scored against the same values as
    # Float64 (whose scores are checked against R above).
    for T in (UInt8, UInt32, UInt64, Int8, Int32)
        dat = T[typemin(T) + 2, typemax(T) - 10, typemax(T) - 3, typemax(T) ÷ 2, 1, 0, 7]
        y = typemax(T) - T(5)
        datf, yf = Float64.(dat), Float64(y)
        levels = [0.2, 0.5, 0.9]
        @testset "$T" begin
            @test quantile_score(dat, y; alpha = 0.3) ≈ quantile_score(datf, yf; alpha = 0.3) rtol = 1.0e-12
            @test interval_score(dat, y; level = 0.8) ≈ interval_score(datf, yf; level = 0.8) rtol = 1.0e-12
            @test quantile_score(levels, dat[[5, 4, 3]], y; reduce = identity) ≈
                quantile_score(levels, datf[[5, 4, 3]], yf; reduce = identity) rtol = 1.0e-12
            # The observation above, inside and below the interval in turn.
            for obs in (dat[2], dat[4] - T(1), dat[6])
                @test interval_score(dat[5], dat[4], obs, 0.8) ≈
                    interval_score(datf[5], datf[4], Float64(obs), 0.8) rtol = 1.0e-12
            end
            lo = min(dat[1], dat[6])
            @test interval_score(lo, dat[5], dat[3], 0.8) ≈
                interval_score(Float64(lo), datf[5], datf[3], 0.8) rtol = 1.0e-12
        end
    end
end
