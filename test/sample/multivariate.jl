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

@testitem "energy score gradients stay finite where members coincide" tags = [:sample] begin
    using ScoringRules
    using ForwardDiff: derivative, gradient

    # Ties are ordinary: an integer ensemble has them, and the default clamp
    # in `twes` sends every member past the bound to the same point. The
    # distance between two tied members has a kink there, and the subgradient
    # is zero; the score must not turn it into NaN.
    cd(f, x; h = 1.0e-6) = (f(x + h) - f(x - h)) / (2h)
    y = [0.5, 1.0]
    tied = [1.0 1.0 3.0; 2.0 2.0 0.0]
    at_obs = [0.5 1.0 3.0; 1.0 2.0 0.0]
    clamped = [0.2 5.0 6.0 -0.3; 0.1 7.0 8.0 0.4]
    wf = weight_function(:norm_cdf; mu = [0.0, 0.0], sigma = [2.0, 2.0])
    # A central difference averages the slopes either side of a kink, which
    # is the zero subgradient, so it is an independent reference both where
    # tied members move together and where a member crosses the observation.
    cases = [
        s -> es(tied .+ s, y),
        s -> es(tied .* (1 + s), y),
        s -> es(at_obs .+ s, y),
        s -> es(tied .+ s, y; w = [0.2, 0.5, 0.3]),
        s -> twes(clamped .+ s, y; b = 1.0),
        s -> owes(tied .+ s, y; weight_func = wf),
    ]
    for f in cases
        g = derivative(f, 0.0)
        @test isfinite(g)
        @test g ≈ cd(f, 0.0) atol = 1.0e-6
    end

    # Two members at x = (1, 2) and y = 0: the tied pair contributes its zero
    # subgradient, leaving the derivative of mean_i ‖x_i‖, x_i / (2‖x_i‖).
    g = gradient(x -> es(reshape(x, 2, 2), [0.0, 0.0]), [1.0, 2.0, 1.0, 2.0])
    @test g ≈ [1.0, 2.0, 1.0, 2.0] ./ (2 * sqrt(5.0))

    # The MMD kernel is smooth at a tie and needs no special case.
    gm = gradient(x -> mmds(reshape(x, 2, 2), [0.0, 0.0]), [1.0, 2.0, 1.0, 2.0])
    @test all(isfinite, gm)
    @test gm ≈ [1.0, 2.0, 1.0, 2.0] .* exp(-2.5) ./ 2
end

@testitem "energy and MMD scores propagate a missing value" tags = [:sample] begin
    using ScoringRules

    # R's es_sample and mmds_sample return NA for an NA member or observation.
    Xm = Union{Missing, Float64}[1.0 missing; 2.0 3.0]
    X = [1.0 0.5; 2.0 3.0]
    ym = Union{Missing, Float64}[0.0, missing]
    for f in (es, mmds)
        @test ismissing(f(Xm, [0.0, 0.0]))
        @test ismissing(f(X, ym))
    end
    @test ismissing(twes(Xm, [0.0, 0.0]; a = -Inf, b = Inf))
    # A Union element type with no missing value scores as the plain matrix.
    @test es(Union{Missing, Float64}[1.0 0.5; 2.0 3.0], [0.0, 0.0]) ≈ es(X, [0.0, 0.0])
end
