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
    # and dss_sample(w=).
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
        end
    end

    @testset "member weight edge cases" begin
        dat = ensembles[1]
        y = 0.3
        uniform = fill(2.5, length(dat))
        # Uniform weights reproduce the unweighted scores (any common scale).
        @test dss(dat, y; w = uniform) ≈ dss(dat, y)
        # Rescaling weights leaves the scores unchanged.
        w = weights[1]
        @test dss(dat, y; w = 10 .* w) ≈ dss(dat, y; w = w)
        # Invalid weights throw.
        @test_throws DimensionMismatch dss(dat, y; w = w[1:3])
        @test_throws ArgumentError dss(dat, y; w = -w)
        # Non-finite weights and an all-zero weight vector throw too, so a
        # degenerate w cannot leak NaN into an aggregated score.
        for bad in ([NaN; w[2:end]], [Inf; w[2:end]], zero(w))
            @test_throws ArgumentError dss(dat, y; w = bad)
            @test_throws ArgumentError crps(dat, y; w = bad)
        end
    end
end

@testitem "sample censored/conditional likelihood score matches R scoringRules" setup=[References] begin
    using ScoringRules

    # Load the fixed ensembles (each stored as a 1-row CSV of m columns).
    function load_ens(m::Int)
        c, _ = References.load("ens_clogs_$m")
        return [c["m$j"][1] for j in 1:length(c)]
    end

    ensembles = Dict(m => load_ens(m) for m in (5, 30, 500))

    c, n = References.load("sample_clogs")
    for i in 1:n
        dat = ensembles[Int(c["n"][i])]
        y = c["y"][i]
        a = c["a"][i]
        b = c["b"][i]
        # NaN in the bw column marks R's default (bw.nrd) bandwidth.
        bw = isnan(c["bw"][i]) ? nothing : c["bw"][i]
        @test clogs(dat, y; a, b, bw, cens = true)≈c["clogs_cens"][i] atol=1e-9 rtol=1e-6
        @test clogs(dat, y; a, b, bw, cens = false)≈c["clogs_cond"][i] atol=1e-9 rtol=1e-6
    end
end

@testitem "clogs edge cases" begin
    using ScoringRules

    dat = [-1.2, -0.3, 0.1, 0.8, 1.5]

    # Unbounded window: both variants reduce to the plain KDE log score.
    @test clogs(dat, 0.4)≈logs(dat, 0.4) atol=1e-12
    @test clogs(dat, 0.4; cens = false)≈logs(dat, 0.4) atol=1e-12

    # Observation outside the window: conditional score is exactly zero,
    # censored score is the log probability of falling outside.
    @test clogs(dat, 3.0; a = -1.0, b = 1.0, cens = false) == 0.0
    @test clogs(dat, 3.0; a = -1.0, b = 1.0, cens = true) > 0.0

    # Bounds use strict inequalities: y on a bound counts as outside.
    @test clogs(dat, 1.0; a = -1.0, b = 1.0, cens = false) == 0.0

    # Degenerate window is rejected.
    @test_throws ArgumentError clogs(dat, 0.0; a = 1.0, b = 1.0)
    @test_throws ArgumentError clogs(dat, 0.0; a = 2.0, b = 1.0)
end
