@testitem "student-t family scores match R scoringRules" tags = [:crps] setup = [References] begin
    using ScoringRules
    using Distributions

    atol = 1.0e-9
    rtol = 1.0e-8

    _bound(x) = isfinite(x) ? x : nothing
    function trunc_t(df, loc, scale, l, u)
        d_inner = loc + scale * TDist(df)
        (isinf(l)&&isinf(u))&&return d_inner
        return truncated(d_inner; lower = _bound(l), upper = _bound(u))
    end
    function cens_t(df, loc, scale, l, u)
        d_inner = loc + scale * TDist(df)
        (isinf(l)&&isinf(u))&&return d_inner
        return censored(d_inner; lower = _bound(l), upper = _bound(u))
    end

    @testset "Student-t" begin
        c, n = References.load("student_t")
        for i in 1:n
            d = c["location"][i] + c["scale"][i] * TDist(c["df"][i])
            y = c["y"][i]
            @test crps(d, y) ≈ c["crps"][i] atol = atol rtol = rtol
            @test logs(d, y) ≈ c["logs"][i] atol = atol rtol = rtol
            # R's dss_t returns NaN for df ≤ 2 (infinite variance); the t
            # variance is genuinely infinite there, so skip those rows.
            isnan(c["dss"][i]) && continue
            @test dss(d, y) ≈ c["dss"][i] atol = atol rtol = rtol
        end
    end

    @testset "truncated Student-t" begin
        c, n = References.load("tt")
        for i in 1:n
            d = trunc_t(
                c["df"][i], c["location"][i], c["scale"][i],
                c["lower"][i], c["upper"][i]
            )
            y = c["y"][i]
            ref_crps = c["crps"][i]
            ref_logs = c["logs"][i]
            isnan(ref_crps) && continue
            @test crps(d, y) ≈ ref_crps atol = atol rtol = rtol
            (isnan(ref_logs) || isinf(ref_logs)) && continue
            # R's `ft` normalises a lower-bounded truncation by
            # `1 - pt(lower)`, which keeps no relative accuracy once the
            # retained tail is small (at df = 10 and a standardised bound of
            # 50 its normaliser is 1.5e-4 out). Those rows are checked against
            # exact quadrature in test/crps/extreme_arguments.jl instead.
            if cdf(TDist(c["df"][i]), (c["lower"][i] - c["location"][i]) / c["scale"][i]) >
                    1 - 1.0e-8
                continue
            end
            @test logs(d, y) ≈ ref_logs atol = atol rtol = rtol
        end
    end

    @testset "censored Student-t" begin
        c, n = References.load("ct")
        for i in 1:n
            d = cens_t(
                c["df"][i], c["location"][i], c["scale"][i],
                c["lower"][i], c["upper"][i]
            )
            y = c["y"][i]
            ref_crps = c["crps"][i]
            isnan(ref_crps) && continue
            @test crps(d, y) ≈ ref_crps atol = atol rtol = rtol
        end
    end
end
