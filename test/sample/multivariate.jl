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

@testitem "the weighted variogram score matches its kernel form" tags = [:sample] begin
    using ScoringRules
    using Random: MersenneTwister

    # `vs` with member weights used to evaluate the kernel score
    #   Σ_i w_i κ(X_i, y) − ½ Σ_{i,j} w_i w_j κ(X_i, X_j),
    # which is m² kernel evaluations for the same number the single pass over
    # the ensemble gives. This pins the two together.
    function kernel_form(y, X, w_vs, w, p)
        d = length(y)
        m = size(X, 2)
        κ = (x1, x2) -> sum(
            2.0 * w_vs[k, l] *
                (abs(x1[k] - x1[l])^p - abs(x2[k] - x2[l])^p)^2
                for k in 1:d, l in 1:d if l >= k
        )
        s1 = sum(w[i] * κ(view(X, :, i), y) for i in 1:m)
        s2 = sum(w[i] * w[j] * κ(view(X, :, i), view(X, :, j)) for i in 1:m, j in 1:m)
        return s1 - s2 / 2
    end

    rng = MersenneTwister(20260218)
    for (d, m, p) in ((3, 20, 0.5), (5, 40, 1.0), (4, 13, 2.0), (6, 7, 0.25))
        X = randn(rng, d, m)
        y = randn(rng, d)
        wraw = rand(rng, m) .+ 0.1
        w = wraw ./ sum(wraw)
        A = [1 / (1 + abs(k - l)) for k in 1:d, l in 1:d]
        for wv in (ones(d, d), A)
            @test vs(X, y; p = p, w = wraw, w_vs = wv) ≈
                kernel_form(y, X, wv, w, p) rtol = 1.0e-12
        end
        # And the unweighted path is the uniform-weight case of the same form.
        @test vs(X, y; p = p) ≈
            kernel_form(y, X, ones(d, d), fill(1 / m, m), p) rtol = 1.0e-12
    end
end

@testitem "multivariate scores reject an empty ensemble" tags = [:sample] begin
    using ScoringRules

    X = Matrix{Float64}(undef, 3, 0)
    y = [0.1, -0.2, 0.5]
    @test_throws ArgumentError es(X, y)
    @test_throws ArgumentError vs(X, y)
    @test_throws ArgumentError mmds(X, y)
end
