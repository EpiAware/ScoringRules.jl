@testitem "sample univariate scores match R scoringRules" setup=[References] begin
    using ScoringRules
    using ScoringRules: _bw_nrd

    atol = 1e-9
    rtol = 1e-7

    # Load a fixed ensemble or weight vector (stored as a 1-row CSV of m columns).
    function load_ens_file(name::AbstractString)
        c, _ = References.load(name)
        # Columns are named m1, m2, ... ; collect values in order.
        n_mem = length(c)
        dat = [c["m$j"][1] for j in 1:n_mem]
        return dat
    end

    ensembles = [load_ens_file("ens_univ_$i") for i in 1:4]

    c, n = References.load("sample_univ_scores")

    @testset "crps EDF" begin
        for i in 1:n
            eid = Int(c["ens_id"][i])
            dat = ensembles[eid]
            y = c["y"][i]
            ref = c["crps"][i]
            @test crps(dat, y; method = :edf)≈ref atol=atol rtol=rtol
        end
    end

    @testset "crps KDE" begin
        for i in 1:n
            eid = Int(c["ens_id"][i])
            dat = ensembles[eid]
            y = c["y"][i]
            ref = c["crps_kde"][i]
            # KDE bandwidth matching R's bw.nrd may introduce rounding differences.
            @test crps(dat, y; method = :kde)≈ref atol=atol rtol=1e-6
        end
    end

    @testset "logs (KDE)" begin
        for i in 1:n
            eid = Int(c["ens_id"][i])
            dat = ensembles[eid]
            y = c["y"][i]
            ref = c["logs"][i]
            @test logs(dat, y)≈ref atol=atol rtol=1e-6
        end
    end

    @testset "dss" begin
        for i in 1:n
            eid = Int(c["ens_id"][i])
            dat = ensembles[eid]
            y = c["y"][i]
            ref = c["dss"][i]
            @test dss(dat, y)≈ref atol=atol rtol=rtol
        end
    end

    # Member-weighted scores. crps_w and dss_w come from R's crps_sample(w=)
    # and dss_sample(w=); logs_w is the weighted-KDE density computed from
    # dnorm() in the generator script (R's logs_sample has no weights).
    weights = [load_ens_file("ens_univ_w_$i") for i in 1:4]
    cw, nw = References.load("sample_univ_weighted_members")

    @testset "member weights" begin
        for i in 1:nw
            eid = Int(cw["ens_id"][i])
            dat = ensembles[eid]
            w = weights[eid]
            y = cw["y"][i]
            @test crps(dat, y; w = w)≈cw["crps_w"][i] atol=atol rtol=rtol
            @test dss(dat, y; w = w)≈cw["dss_w"][i] atol=atol rtol=rtol
            @test logs(dat, y; w = w)≈cw["logs_w"][i] atol=atol rtol=1e-6
        end
    end

    @testset "member weight edge cases" begin
        dat = ensembles[1]
        y = 0.3
        uniform = fill(2.5, length(dat))
        # Uniform weights reproduce the unweighted scores (any common scale).
        @test dss(dat, y; w = uniform) ≈ dss(dat, y)
        @test logs(dat, y; w = uniform) ≈ logs(dat, y)
        # Rescaling weights leaves the scores unchanged.
        w = weights[1]
        @test dss(dat, y; w = 10 .* w) ≈ dss(dat, y; w = w)
        @test logs(dat, y; w = 10 .* w) ≈ logs(dat, y; w = w)
        # Invalid weights throw.
        @test_throws DimensionMismatch dss(dat, y; w = w[1:3])
        @test_throws DimensionMismatch logs(dat, y; w = w[1:3])
        @test_throws ArgumentError dss(dat, y; w = -w)
        @test_throws ArgumentError logs(dat, y; w = -w)
        # Non-finite weights and an all-zero weight vector throw too, so a
        # degenerate w cannot leak NaN into an aggregated score.
        for bad in ([NaN; w[2:end]], [Inf; w[2:end]], zero(w))
            @test_throws ArgumentError dss(dat, y; w = bad)
            @test_throws ArgumentError logs(dat, y; w = bad)
            @test_throws ArgumentError crps(dat, y; w = bad)
        end
    end
end
