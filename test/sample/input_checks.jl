@testitem "sample scores reject the inputs R scoringRules stops on" begin
    using ScoringRules

    # R's rps_probs stops on a negative probability, a sum more than `tol`
    # (by default one eps) from one, and fewer than two categories.
    @test_throws ArgumentError rps([1.2, -0.2], 1)
    @test_throws ArgumentError rps([0.5, 0.2], 1)
    @test_throws ArgumentError rps([0.5, 0.5 + 1.0e-12], 1)
    @test rps([0.5, 0.5 + 1.0e-12], 1; tol = 1.0e-10) ≈ 0.25
    @test_throws ArgumentError rps([1.0], 1)
    @test_throws ArgumentError rps([0.5, 0.5], 1; tol = -1.0)
    # A normalised vector passes the default `tol`, as in R, even where its
    # plain Float64 sum is a few eps from one.
    @test rps(fill(1 / 14, 14), 1) ≈ 4.178571428571429  # R: rps_probs(1, rep(1/14, 14))
    @test all(K -> isfinite(rps(fill(1 / K, K), 1)), 2:60)
    # The default tolerance is one eps of the forecast's own precision.
    @test rps(Float32[0.3, 0.3, 0.4], 2) ≈ rps([0.3, 0.3, 0.4], 2) rtol = 1.0e-6
    @test_throws ArgumentError rps(Float32[0.5, 0.5 + 2 * eps(Float32)], 1)
    # A wider type keeps R's tolerance.
    @test rps(big.([0.1, 0.2, 0.7]), 2) ≈ rps([0.1, 0.2, 0.7], 2)
    @test all(1:2000) do i
        p = Float32[mod(i * k * 0.6180339887498949, 1) + 0.01 for k in 1:(2 + i % 49)]
        isfinite(rps(p ./ Float32(sum(big, p)), 1))
    end
    # Vectors of 2 to 50 categories from a golden-ratio sequence, normalised as
    # R's `p / sum(p)` does, with the sum taken to extended precision.
    @test all(1:2000) do i
        p = [mod(i * k * 0.6180339887498949, 1) + 0.01 for k in 1:(2 + i % 49)]
        isfinite(rps(p ./ Float64(sum(big, p)), 1))
    end

    # crps_sample, logs_sample, dss_sample, clogs_sample, twcrps_sample,
    # owcrps_sample, qs_sample and ints_sample stop on an empty ensemble.
    empty = Float64[]
    @test_throws ArgumentError crps(empty, 1.0)
    @test_throws ArgumentError crps(empty, 1.0; method = :kde)
    @test_throws ArgumentError logs(empty, 1.0)
    @test_throws ArgumentError dss(empty, 1.0)
    @test_throws ArgumentError clogs(empty, 1.0)
    @test_throws ArgumentError twcrps(empty, 1.0)
    @test_throws ArgumentError owcrps(empty, 1.0)
    @test_throws ArgumentError quantile_score(empty, 1.0; alpha = 0.5)
    @test_throws ArgumentError interval_score(empty, 1.0; level = 0.8)
    @test_throws ArgumentError quantile_score(Float64[], Float64[], 1.0)

    # crps_sample (kde), logs_sample and clogs_sample stop on a negative
    # bandwidth; a bandwidth of zero is accepted.
    d = [0.1, 0.5, 0.9]
    @test_throws ArgumentError crps(d, 0.3; method = :kde, bw = -0.5)
    @test_throws ArgumentError logs(d, 0.3; bw = -0.5)
    @test_throws ArgumentError clogs(d, 0.3; bw = -0.5)
    @test crps(d, 0.3; method = :kde, bw = 0) ≈ 0.1555556 rtol = 1.0e-6  # R: crps_sample(0.3, d, "kde", bw = 0)

    # At bw = 0 the log scores see point masses at the members.
    # R: logs_sample(y, c(0, 1, 1, 3), bw = 0).
    d0 = [0.0, 1.0, 1.0, 3.0]
    @test logs(d0, 1.0; bw = 0) == -Inf
    @test logs(d0, 0.5; bw = 0) == Inf
    @test logs(d0, 5.0; bw = 0) == Inf
    # R: clogs_sample(y, c(0, 1, 1, 3), a, b, bw = 0, cens), y inside (a, b).
    for cens in (true, false)
        @test clogs(d0, 1.0; bw = 0, cens) == -Inf
        @test clogs(d0, 1.0; a = 0.2, b = 2.0, bw = 0, cens) == -Inf
        @test clogs(d0, 1.0; a = -1.0, b = 2.0, bw = 0, cens) == -Inf
        @test clogs(d0, 0.5; bw = 0, cens) == Inf
        @test clogs(d0, 0.5; a = 0.0, b = 1.0, bw = 0, cens) == Inf
        @test clogs(d0, 0.5; a = 0.2, b = 2.0, bw = 0, cens) == Inf
        @test clogs(d0, 5.0; bw = 0, cens) == Inf
    end
    # R gives NaN with y outside the window, and for the censored score in a
    # window holding no member.
    for cens in (true, false)
        @test isnan(clogs(d0, 5.0; a = 0.2, b = 2.0, bw = 0, cens))
        @test isnan(clogs(d0, 1.0; a = 0.0, b = 1.0, bw = 0, cens))
        @test isnan(clogs(d0, 0.5; a = 4.0, b = 6.0, bw = 0, cens))
        @test isnan(clogs(d0, 5.0; a = 4.0, b = 6.0, bw = 0, cens))
    end
    @test isnan(logs(d0, NaN; bw = 0))

    # ints_quantiles stops when the lower end is above the upper one.
    @test_throws ArgumentError interval_score(2.0, 1.0, 1.5, 0.8)
    @test interval_score(1.0, 1.0, 1.5, 0.8) ≈ 5.0  # R: ints_quantiles(1.5, 1, 1, 0.8)

    # vs_sample, twvs_sample and owvs_sample stop on a negative order and
    # return 0 at an order of zero.
    X = [0.1 0.4 -0.3; 1.2 0.8 1.0]
    y = [0.2, 1.1]
    @test_throws ArgumentError vs(X, y; p = -1)
    @test_throws ArgumentError twvs(X, y; p = -1)
    @test_throws ArgumentError owvs(X, y; p = -1)
    @test vs(X, y; p = 0) == 0
end
