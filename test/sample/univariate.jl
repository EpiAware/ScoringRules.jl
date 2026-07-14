@testitem "sample univariate scores match R scoringRules" setup=[References] begin
    using ScoringRules
    using ScoringRules: _bw_nrd

    atol = 1e-9
    rtol = 1e-7

    # Load the four fixed ensembles (each stored as a 1-row CSV of m columns).
    function load_ens(id::Int)
        c, _ = References.load("ens_univ_$id")
        # Columns are named m1, m2, ... ; collect values in order.
        n_mem = length(c)
        dat = [c["m$j"][1] for j in 1:n_mem]
        return dat
    end

    ensembles = [load_ens(i) for i in 1:4]

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
end
