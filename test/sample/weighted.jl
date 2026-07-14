@testitem "weighted ensemble scores match R scoringRules" setup=[References] begin
    atol = 1e-9
    rtol = 1e-7

    # ---------- helpers: load ensembles ----------

    function load_univ_ens(id::Int)
        c, _ = References.load("ens_univ_$id")
        n_mem = length(c)
        return [c["m$j"][1] for j in 1:n_mem]
    end

    function load_mv_ens(id::Int)
        c, nrows = References.load("ens_mv_$id")
        m = length(c)
        d = nrows
        X = Matrix{Float64}(undef, d, m)
        for j in 1:m
            X[:, j] = c["m$j"]
        end
        return X
    end

    univ_ens = [load_univ_ens(i) for i in 1:2]
    mv_ens = [load_mv_ens(i) for i in 1:2]

    ys_mv = [
        [0.0, 0.0, 0.0],
        [1.0, -1.0, 0.5],
        [-2.0, 2.0, -1.0]
    ]

    # ---------- univariate weighted scores ----------

    c, n = References.load("sample_weighted_univ")

    @testset "twcrps" begin
        for i in 1:n
            eid = Int(c["ens_id"][i])
            dat = univ_ens[eid]
            y = c["y"][i]
            a = c["a"][i]
            b = c["b"][i]
            ref = c["twcrps"][i]
            @test twcrps(dat, y; a = a, b = b)≈ref atol=atol rtol=rtol
        end
    end

    @testset "owcrps" begin
        for i in 1:n
            eid = Int(c["ens_id"][i])
            dat = univ_ens[eid]
            y = c["y"][i]
            a = c["a"][i]
            b = c["b"][i]
            ref = c["owcrps"][i]
            got = owcrps(dat, y; a = a, b = b)
            # Both Julia and R return 0 when y is outside (a,b); NaN when all
            # member weights are zero.
            if isnan(ref)
                @test isnan(got)
            else
                @test got≈ref atol=atol rtol=rtol
            end
        end
    end

    # ---------- multivariate weighted scores ----------

    cw, nw = References.load("sample_weighted_mv")

    @testset "twes" begin
        for i in 1:nw
            eid = Int(cw["ens_id"][i])
            yid = Int(cw["y_id"][i])
            X = mv_ens[eid]
            y = ys_mv[yid]
            a = cw["a"][i]
            b = cw["b"][i]
            ref = cw["twes"][i]
            @test twes(X, y; a = a, b = b)≈ref atol=atol rtol=rtol
        end
    end

    @testset "owes" begin
        for i in 1:nw
            eid = Int(cw["ens_id"][i])
            yid = Int(cw["y_id"][i])
            X = mv_ens[eid]
            y = ys_mv[yid]
            a = cw["a"][i]
            b = cw["b"][i]
            ref = cw["owes"][i]
            got = owes(X, y; a = a, b = b)
            if isnan(ref)
                @test isnan(got)
            else
                @test got≈ref atol=atol rtol=rtol
            end
        end
    end

    @testset "twvs" begin
        for i in 1:nw
            eid = Int(cw["ens_id"][i])
            yid = Int(cw["y_id"][i])
            X = mv_ens[eid]
            y = ys_mv[yid]
            a = cw["a"][i]
            b = cw["b"][i]
            p = cw["p_vs"][i]
            ref = cw["twvs"][i]
            @test twvs(X, y; p = p, a = a, b = b)≈ref atol=atol rtol=rtol
        end
    end

    @testset "owvs" begin
        for i in 1:nw
            eid = Int(cw["ens_id"][i])
            yid = Int(cw["y_id"][i])
            X = mv_ens[eid]
            y = ys_mv[yid]
            a = cw["a"][i]
            b = cw["b"][i]
            p = cw["p_vs"][i]
            ref = cw["owvs"][i]
            got = owvs(X, y; p = p, a = a, b = b)
            if isnan(ref)
                @test isnan(got)
            else
                @test got≈ref atol=atol rtol=rtol
            end
        end
    end

    @testset "twmmds" begin
        for i in 1:nw
            eid = Int(cw["ens_id"][i])
            yid = Int(cw["y_id"][i])
            X = mv_ens[eid]
            y = ys_mv[yid]
            a = cw["a"][i]
            b = cw["b"][i]
            ref = cw["twmmds"][i]
            @test twmmds(X, y; a = a, b = b)≈ref atol=atol rtol=rtol
        end
    end

    @testset "owmmds" begin
        for i in 1:nw
            eid = Int(cw["ens_id"][i])
            yid = Int(cw["y_id"][i])
            X = mv_ens[eid]
            y = ys_mv[yid]
            a = cw["a"][i]
            b = cw["b"][i]
            ref = cw["owmmds"][i]
            got = owmmds(X, y; a = a, b = b)
            if isnan(ref)
                @test isnan(got)
            else
                @test got≈ref atol=atol rtol=rtol
            end
        end
    end
end
