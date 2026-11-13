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

@testitem "multivariate scores reject an ensemble with offset axes" tags = [:sample] begin
    using ScoringRules

    # Every loop in the multivariate file pairs `axes(X, ...)` with a weight
    # vector built one-based, and `@inbounds` turns the mismatch into a read of
    # whatever sits next in memory rather than an error. With a row offset
    # `es` returned 0.9276553962740587 for an ensemble whose score is
    # 0.70419137025989309, having read `y` out of bounds.
    struct Offset{T} <: AbstractMatrix{T}
        parent::Matrix{T}
        roff::Int
        coff::Int
    end
    Base.size(A::Offset) = size(A.parent)
    Base.axes(A::Offset) = (A.roff .+ axes(A.parent, 1), A.coff .+ axes(A.parent, 2))
    Base.getindex(A::Offset, i::Int, j::Int) = A.parent[i - A.roff, j - A.coff]

    P = [0.3 -1.2 0.8 0.1; -0.4 0.9 -0.2 1.3; 1.1 0.2 -1.5 0.6]
    y = [0.2, -0.3, 0.7]
    ref = (es(P, y), vs(P, y), mmds(P, y))
    @test all(isfinite, ref)

    for (roff, coff) in ((-1, 0), (0, -1), (100, 100), (3, 0))
        X = Offset(P, roff, coff)
        @test_throws ArgumentError es(X, y)
        @test_throws ArgumentError vs(X, y)
        @test_throws ArgumentError mmds(X, y)
        @test_throws ArgumentError twes(X, y; a = -1.0)
        @test_throws ArgumentError owvs(X, y; a = -1.0)
    end
    # The weights are indexed over one-based ranges too, and `_member_weights`
    # keeps whatever axes it was given, so an offset weight vector was the same
    # silent out-of-bounds read reached through another argument.
    struct OffsetVec{T} <: AbstractVector{T}
        parent::Vector{T}
        off::Int
    end
    Base.size(a::OffsetVec) = size(a.parent)
    Base.axes(a::OffsetVec) = (a.off .+ axes(a.parent, 1),)
    Base.getindex(a::OffsetVec, i::Int) = a.parent[i - a.off]

    wv = collect(1:4) ./ 10
    wvs = [1.0 0.5 0.2; 0.5 1.0 0.3; 0.2 0.3 1.0]
    @test isfinite(es(P, y; w = wv))
    for off in (-1, 1, 10)
        @test_throws ArgumentError es(P, y; w = OffsetVec(wv, off))
        @test_throws ArgumentError vs(P, y; w = OffsetVec(wv, off))
        @test_throws ArgumentError mmds(P, y; w = OffsetVec(wv, off))
        @test_throws ArgumentError vs(P, y; w_vs = Offset(wvs, off, off))
    end
    # A one-based wrapper of the same data still scores, and to the same value.
    @test es(Offset(P, 0, 0), y) ≈ ref[1] rtol = 1.0e-14
    @test vs(Offset(P, 0, 0), y) ≈ ref[2] rtol = 1.0e-14
    @test mmds(Offset(P, 0, 0), y) ≈ ref[3] rtol = 1.0e-14
end

@testitem "multivariate scores promote an integer ensemble" tags = [:sample] begin
    using ScoringRules

    # The squared-distance accumulator used to be seeded at the matrix element
    # type, so an integer ensemble accumulated in integer arithmetic and
    # overflowed: `es(Int32[2147483647 0; 0 0], Int32[0, 0])` returned 0.25
    # against the true 536870911.75, and the Int64 version threw a DomainError
    # out of `sqrt`.
    for T in (Int32, Int64)
        big_val = T === Int32 ? T(2147483647) : T(4_000_000_000)
        X = T[big_val 0; 0 0]
        y = T[0, 0]
        Xf = Float64.(X)
        yf = Float64.(y)
        @test es(X, y) ≈ es(Xf, yf) rtol = 1.0e-12
        @test mmds(X, y) ≈ mmds(Xf, yf) rtol = 1.0e-12
        @test vs(X, y) ≈ vs(Xf, yf) rtol = 1.0e-12
    end
    # Small integer ensembles already worked and must keep their value.
    Xs = [1 4 2; 3 0 5]
    ys = [2, 1]
    @test es(Xs, ys) ≈ es(Float64.(Xs), Float64.(ys)) rtol = 1.0e-14
    @test mmds(Xs, ys) ≈ mmds(Float64.(Xs), Float64.(ys)) rtol = 1.0e-14

    # The variogram score raises its differences to a power rather than going
    # through the squared-distance helper, so it is a separate path: it
    # returned 5.899e8 for an ensemble whose score is 8.0e9, and threw out of
    # `^0.5` when a wrapped difference landed on `typemin`.
    for (X, y) in (
            (Int32[2000000000 2000000000; -2000000000 -2000000000], Int32[0, 0]),
            (Int32[2147483647 2147483647; -1 -1], Int32[0, 0]),
            (Int64[4000000000 0; 0 4000000000], Int64[0, 0]),
        )
        @test vs(X, y) ≈ vs(Float64.(X), Float64.(y)) rtol = 1.0e-12
        for p in (0.5, 1.0, 2.0)
            @test vs(X, y; p = p) ≈ vs(Float64.(X), Float64.(y); p = p) rtol = 1.0e-12
        end
        @test twvs(X, y; a = -Inf) ≈
            twvs(Float64.(X), Float64.(y); a = -Inf) rtol = 1.0e-12
    end
end
