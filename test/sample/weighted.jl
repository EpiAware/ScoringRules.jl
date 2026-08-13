@testitem "weighted ensemble scores match R scoringRules" setup=[References] begin
    using ScoringRules
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

@testitem "member-weighted tw/ow scores match R scoringRules" setup=[References] begin
    using ScoringRules
    atol = 1e-9
    rtol = 1e-7

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

    # Deterministic member-weight vectors, mirroring generate_references.R.
    function member_w(m::Int, w_id::Int)
        w_id == 1 && return fill(2.0, m)
        w_id == 2 && return collect(1.0:m)
        return vcat(zeros(5), collect(1.0:(m - 5)))
    end

    # Compare a possibly-NaN reference (all combined weights zero) with the
    # computed score.
    function test_ref(got, ref)
        if isnan(ref)
            @test isnan(got)
        else
            @test got≈ref atol=atol rtol=rtol
        end
    end

    c, n = References.load("member_w_weighted_univ")

    @testset "twcrps / owcrps with member weights" begin
        for i in 1:n
            dat = univ_ens[Int(c["ens_id"][i])]
            y = c["y"][i]
            a = c["a"][i]
            b = c["b"][i]
            wv = member_w(length(dat), Int(c["w_id"][i]))
            @test twcrps(dat, y; a = a, b = b, w = wv)≈c["twcrps"][i] atol=atol rtol=rtol
            test_ref(owcrps(dat, y; a = a, b = b, w = wv), c["owcrps"][i])
        end
    end

    cw, nw = References.load("member_w_weighted_mv")

    @testset "tw/ow multivariate scores with member weights" begin
        for i in 1:nw
            X = mv_ens[Int(cw["ens_id"][i])]
            y = ys_mv[Int(cw["y_id"][i])]
            a = cw["a"][i]
            b = cw["b"][i]
            p = cw["p_vs"][i]
            wv = member_w(size(X, 2), Int(cw["w_id"][i]))
            @test twes(X, y; a = a, b = b, w = wv)≈cw["twes"][i] atol=atol rtol=rtol
            @test twvs(X, y; p = p, a = a, b = b, w = wv)≈cw["twvs"][i] atol=atol rtol=rtol
            @test twmmds(X, y; a = a, b = b, w = wv)≈cw["twmmds"][i] atol=atol rtol=rtol
            test_ref(owes(X, y; a = a, b = b, w = wv), cw["owes"][i])
            test_ref(owvs(X, y; p = p, a = a, b = b, w = wv), cw["owvs"][i])
            test_ref(owmmds(X, y; a = a, b = b, w = wv), cw["owmmds"][i])
        end
    end

    @testset "argument validation" begin
        X = mv_ens[1]
        y = ys_mv[1]
        m = size(X, 2)
        dat = univ_ens[1]
        @test_throws DimensionMismatch twes(X, y; w = ones(m + 1))
        @test_throws ArgumentError owes(X, y; w = fill(-1.0, m))
        @test_throws DimensionMismatch owcrps(dat, 0.0; w = ones(length(dat) + 1))
        @test_throws ArgumentError twcrps(dat, 0.0; w = fill(-1.0, length(dat)))
    end

    @testset "all-zero member weights give NaN for outcome-weighted scores" begin
        # No member has positive weight, so the outcome-weighted scores are
        # undefined: R's ow*_sample return NaN, and so must we, consistently
        # across the univariate and multivariate scores.
        X = mv_ens[1]
        y = ys_mv[1]
        m = size(X, 2)
        dat = univ_ens[1]
        @test isnan(owcrps(dat, 0.0; w = zeros(length(dat))))
        @test isnan(owes(X, y; w = zeros(m)))
        @test isnan(owvs(X, y; w = zeros(m)))
        @test isnan(owmmds(X, y; w = zeros(m)))
    end
end
