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

@testitem "vs on an integer ensemble does not wrap" begin
    using ScoringRules

    # The score written out over all ordered pairs in BigFloat, where no
    # difference can wrap or overflow.
    function vs_ref(X, y, p, w, w_vs)
        Xb, yb = big.(X), big.(y)
        d, m = size(Xb)
        wn = big.(w) ./ sum(big.(w))
        return sum(
            w_vs[k, l] * (
                abs(yb[k] - yb[l])^p -
                    sum(wn[i] * abs(Xb[k, i] - Xb[l, i])^p for i in 1:m)
            )^2
                for k in 1:d, l in 1:d
        )
    end

    @test vs(Int32[2000000000 0; -2000000000 0], Int32[0, 0]) ≈ 2.0e9 rtol = 1.0e-12

    for T in (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64)
        lo, hi = typemin(T), typemax(T)
        # Members and observation at both ends of the type's range, so every
        # difference between dimensions leaves the type.
        X = T[hi lo hi; lo hi zero(T); hi zero(T) lo]
        y = T[lo, hi, zero(T)]
        w = [1.0, 2.0, 0.5]
        w_vs = [1.0 0.5 2.0; 0.5 1.0 0.0; 2.0 0.0 1.0]
        for p in (0.5, 1.0, 2.0)
            unif = ones(3)
            ref = vs_ref(X, y, p, unif, ones(3, 3))
            ref_w = vs_ref(X, y, p, w, ones(3, 3))
            ref_wvs = vs_ref(X, y, p, unif, w_vs)
            ref_both = vs_ref(X, y, p, w, w_vs)
            @test vs(X, y; p) ≈ ref rtol = 1.0e-10
            @test vs(X, y; p, w) ≈ ref_w rtol = 1.0e-10
            @test vs(X, y; p, w_vs) ≈ ref_wvs rtol = 1.0e-10
            @test vs(X, y; p, w, w_vs) ≈ ref_both rtol = 1.0e-10
            # Identity chaining and a constant outcome weight leave the
            # ensemble as it is, so the weighted forms reduce to `vs`.
            @test twvs(X, y; p, chain_func = identity, w, w_vs) ≈ ref_both rtol = 1.0e-10
            @test owvs(X, y; p, weight_func = z -> 1.0, w, w_vs) ≈ ref_both rtol = 1.0e-10
        end
    end

    # A missing value in a member or a dimension still reaches the score
    # when its member or pair weight is zero, as in R.
    X = [1.0 NaN 3.0; 2.0 0.0 1.0; 0.5 1.0 2.0]
    y = [0.0, 1.0, 2.0]
    @test isnan(vs(X, y; w = [1.0, 0.0, 1.0]))
    @test isnan(vs(X, y; w_vs = [0.0 0.0 0.0; 0.0 1.0 1.0; 0.0 1.0 1.0]))
    @test isnan(owvs(X, y; weight_func = z -> 1.0, w = [1.0, 0.0, 1.0]))
end
