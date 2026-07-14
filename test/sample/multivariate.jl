@testitem "sample multivariate scores match R scoringRules" setup=[References] begin

    atol = 1e-9
    rtol = 1e-7

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
            X   = ensembles[eid]
            y   = ys_mv[yid]
            ref = c["es"][i]
            # es does not depend on p_vs; only check once per (ens, y) combo.
            c["p_vs"][i] == 0.5 || continue
            @test es(X, y) ≈ ref atol=atol rtol=rtol
        end
    end

    @testset "variogram score" begin
        for i in 1:n
            eid = Int(c["ens_id"][i])
            yid = Int(c["y_id"][i])
            X   = ensembles[eid]
            y   = ys_mv[yid]
            p   = c["p_vs"][i]
            ref = c["vs"][i]
            @test vs(X, y; p=p) ≈ ref atol=atol rtol=rtol
        end
    end

    @testset "MMD score" begin
        for i in 1:n
            eid = Int(c["ens_id"][i])
            yid = Int(c["y_id"][i])
            X   = ensembles[eid]
            y   = ys_mv[yid]
            ref = c["mmds"][i]
            # mmds does not depend on p_vs; check once per (ens, y) combo.
            c["p_vs"][i] == 0.5 || continue
            @test mmds(X, y) ≈ ref atol=atol rtol=rtol
        end
    end
end
