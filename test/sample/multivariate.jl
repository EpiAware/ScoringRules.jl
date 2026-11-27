@testitem "sample multivariate scores match R scoringRules" setup = [References] begin
    using ScoringRules
    atol = 1.0e-9
    rtol = 1.0e-7

    # Load a d×m ensemble from a CSV where each row is a dimension and each
    # column is a member (R stores the matrix row-major, d rows × m cols).
    function load_mv_ens(id::Int)
        c, nrows = References.load("ens_mv_$id")
        # Columns are m1, m2, ...; rows correspond to dimensions.
        m = length(c)
        d = nrows
        X = Matrix{Float64}(undef, d, m)
        for j in 1:m
            X[:, j] = c["m$j"]
        end
        return X
    end

    ensembles = [load_mv_ens(i) for i in 1:2]

    # Observations used in the R script.
    ys_mv = [
        [0.0, 0.0, 0.0],
        [1.0, -1.0, 0.5],
        [-2.0, 2.0, -1.0],
    ]

    c, n = References.load("sample_mv_scores")

    @testset "energy score" begin
        for i in 1:n
            eid = Int(c["ens_id"][i])
            yid = Int(c["y_id"][i])
            X = ensembles[eid]
            y = ys_mv[yid]
            ref = c["es"][i]
            # es does not depend on p_vs; only check once per (ens, y) combo.
            c["p_vs"][i] == 0.5 || continue
            @test es(X, y) ≈ ref atol = atol rtol = rtol
        end
    end

    @testset "variogram score" begin
        for i in 1:n
            eid = Int(c["ens_id"][i])
            yid = Int(c["y_id"][i])
            X = ensembles[eid]
            y = ys_mv[yid]
            p = c["p_vs"][i]
            ref = c["vs"][i]
            @test vs(X, y; p = p) ≈ ref atol = atol rtol = rtol
        end
    end

    @testset "MMD score" begin
        for i in 1:n
            eid = Int(c["ens_id"][i])
            yid = Int(c["y_id"][i])
            X = ensembles[eid]
            y = ys_mv[yid]
            ref = c["mmds"][i]
            # mmds does not depend on p_vs; check once per (ens, y) combo.
            c["p_vs"][i] == 0.5 || continue
            @test mmds(X, y) ≈ ref atol = atol rtol = rtol
        end
    end
end

@testitem "member-weighted multivariate scores match R scoringRules" setup = [References] begin
    using ScoringRules
    atol = 1.0e-9
    rtol = 1.0e-7

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

    ensembles = [load_mv_ens(i) for i in 1:2]

    ys_mv = [
        [0.0, 0.0, 0.0],
        [1.0, -1.0, 0.5],
        [-2.0, 2.0, -1.0],
    ]

    # Deterministic member-weight vectors, mirroring generate_references.R.
    function member_w(m::Int, w_id::Int)
        w_id == 1 && return fill(2.0, m)
        w_id == 2 && return collect(1.0:m)
        return vcat(zeros(5), collect(1.0:(m - 5)))
    end

    # Pairwise weight matrix for the variogram score (matches wvs_mv in R).
    d = 3
    wvs = [1 / (1 + abs(k - l)) for k in 1:d, l in 1:d]

    c, n = References.load("member_w_mv")

    @testset "es / vs / mmds with member weights" begin
        for i in 1:n
            X = ensembles[Int(c["ens_id"][i])]
            y = ys_mv[Int(c["y_id"][i])]
            wv = member_w(size(X, 2), Int(c["w_id"][i]))
            p = c["p_vs"][i]
            @test es(X, y; w = wv) ≈ c["es"][i] atol = atol rtol = rtol
            @test vs(X, y; p = p, w = wv) ≈ c["vs"][i] atol = atol rtol = rtol
            @test mmds(X, y; w = wv) ≈ c["mmds"][i] atol = atol rtol = rtol
        end
    end

    @testset "vs with pairwise weight matrix" begin
        for i in 1:n
            X = ensembles[Int(c["ens_id"][i])]
            y = ys_mv[Int(c["y_id"][i])]
            wv = member_w(size(X, 2), Int(c["w_id"][i]))
            p = c["p_vs"][i]
            @test vs(X, y; p = p, w_vs = wvs) ≈ c["vs_wvs"][i] atol = atol rtol = rtol
            @test vs(X, y; p = p, w = wv, w_vs = wvs) ≈ c["vs_w_wvs"][i] atol = atol rtol = rtol
        end
    end

    @testset "argument validation" begin
        X = ensembles[1]
        y = ys_mv[1]
        m = size(X, 2)
        # The pairwise matrix now travels under w_vs; a matrix passed as the
        # member-weight argument w is rejected.
        @test_throws DimensionMismatch vs(X, y; w = ones(3, 3))
        @test_throws ArgumentError vs(X, y; w_vs = [1.0 2.0 3.0; 0.0 1.0 0.0; 0.0 0.0 1.0])
        @test_throws DimensionMismatch vs(X, y; w_vs = ones(2, 2))
        @test_throws ArgumentError vs(X, y; w = fill(-1.0, m))
        @test_throws DimensionMismatch mmds(X, y; w = ones(m + 1))
        @test_throws ArgumentError mmds(X, y; w = fill(-1.0, m))
    end
end

@testitem "multivariate scores reject offset axes" tags = [:sample] begin
    using ScoringRules
    using OffsetArrays: OffsetArray

    # The score loops index `X`, `y`, `w` and `w_vs` over one-based ranges
    # under `@inbounds`, so offset axes read past the end of the array:
    # `es(OffsetArray([1.0 2; 3 4], 0:1, 0:1), OffsetArray([0.0, 1], 0:1))`
    # returned 1.8028 against the true 2.5673.
    P = [0.3 -1.2 0.8 0.1; -0.4 0.9 -0.2 1.3; 1.1 0.2 -1.5 0.6]
    y = [0.2, -0.3, 0.7]
    w = [1.0, 2.0, 0.5, 1.5]
    wvs = [1 / (1 + abs(k - l)) for k in 1:3, l in 1:3]
    bounds = (a = -1.25, b = 1.25)
    scores = (es, vs, mmds, twes, owes, twvs, owvs, twmmds, owmmds)
    kw(f) = f in (es, vs, mmds) ? (;) : bounds

    offset_X = (
        OffsetArray(P, 0:2, 1:4), OffsetArray(P, 1:3, 0:3),
        OffsetArray(P, 101:103, 101:104),
    )
    offset_y = (OffsetArray(y, 0:2), OffsetArray(y, 4:6))
    offset_w = (OffsetArray(w, 0:3), OffsetArray(w, 5:8))
    offset_wvs = (OffsetArray(wvs, 0:2, 0:2), OffsetArray(wvs, 1:3, 2:4))

    for f in scores
        for X in offset_X
            @test_throws ArgumentError f(X, y; kw(f)...)
        end
        for yo in offset_y
            @test_throws ArgumentError f(P, yo; kw(f)...)
        end
        for wo in offset_w
            @test_throws ArgumentError f(P, y; w = wo, kw(f)...)
        end
    end
    for f in (vs, twvs, owvs), wvo in offset_wvs
        @test_throws ArgumentError f(P, y; w_vs = wvo, kw(f)...)
        @test_throws ArgumentError f(P, y; w = w, w_vs = wvo, kw(f)...)
    end

    # An OffsetArray whose axes start at one is an ordinary array and scores
    # as one.
    X1 = OffsetArray(P, 1:3, 1:4)
    y1 = OffsetArray(y, 1:3)
    w1 = OffsetArray(w, 1:4)
    wvs1 = OffsetArray(wvs, 1:3, 1:3)
    for f in scores
        @test f(X1, y1; w = w1, kw(f)...) == f(P, y; w = w, kw(f)...)
    end
    for f in (vs, twvs, owvs)
        @test f(X1, y1; w = w1, w_vs = wvs1, kw(f)...) ==
            f(P, y; w = w, w_vs = wvs, kw(f)...)
    end
end
