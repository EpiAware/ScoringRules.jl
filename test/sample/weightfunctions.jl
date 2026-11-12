@testitem "smooth weight and chaining functions match R get_weight_func" tags = [:sample] setup = [References] begin
    using ScoringRules
    using DelimitedFiles: readdlm

    atol = 1.0e-10
    rtol = 1.0e-9

    # `References.load` drops the string `name` column, so the shapes are read
    # back straight from the CSV here.
    function load_names(file)
        raw, header = readdlm(
            joinpath(References.DATA_DIR, file), ',', Any; header = true
        )
        cols = Dict(strip(string(h), '"') => j for (j, h) in enumerate(vec(header)))
        return raw, cols
    end

    @testset "univariate" begin
        dat_cols, _ = References.load("weight_func_univ_dat")
        dat = dat_cols["dat"]
        y = dat_cols["y"][1]
        raw, cols = load_names("weight_func_univ.csv")
        for i in axes(raw, 1)
            name = Symbol(strip(string(raw[i, cols["name"]]), '"'))
            mu = Float64(raw[i, cols["mu"]])
            sigma = Float64(raw[i, cols["sigma"]])
            wf = weight_function(name; mu = mu, sigma = sigma)
            cf = chaining_function(name; mu = mu, sigma = sigma)
            @test owcrps(dat, y; weight_func = wf) ≈
                Float64(raw[i, cols["ow"]]) atol = atol rtol = rtol
            @test twcrps(dat, y; chain_func = cf) ≈
                Float64(raw[i, cols["tw"]]) atol = atol rtol = rtol
        end
    end

    @testset "multivariate" begin
        Xc, m = References.load("weight_func_mv_dat")
        # The CSV holds one member per row, so the d × m matrix is its transpose.
        X = permutedims(reduce(hcat, [Xc["V$k"] for k in 1:4]))
        yc, _ = References.load("weight_func_mv_obs")
        yv = yc["y"]
        raw, cols = load_names("weight_func_mv.csv")
        for i in axes(raw, 1)
            name = Symbol(strip(string(raw[i, cols["name"]]), '"'))
            mu = fill(Float64(raw[i, cols["mu"]]), 4)
            sigma = fill(Float64(raw[i, cols["sigma"]]), 4)
            wf = weight_function(name; mu = mu, sigma = sigma)
            cf = chaining_function(name; mu = mu, sigma = sigma)
            @test owes(X, yv; weight_func = wf) ≈
                Float64(raw[i, cols["ow"]]) atol = atol rtol = rtol
            @test twes(X, yv; chain_func = cf) ≈
                Float64(raw[i, cols["tw"]]) atol = atol rtol = rtol
        end
    end
end

@testitem "each chaining function is the antiderivative of its weight" tags = [:sample] begin
    using ScoringRules
    using ForwardDiff: derivative

    # This is what makes the threshold-weighted score proper for that weight,
    # so it is the property worth pinning rather than the formulae.
    for name in (:norm_cdf, :norm_surv, :norm_pdf, :logis_cdf, :logis_surv, :logis_pdf)
        for mu in (-0.5, 0.0, 1.2), sigma in (0.4, 1.0, 2.5)
            v = chaining_function(name; mu = mu, sigma = sigma)
            w = weight_function(name; mu = mu, sigma = sigma)
            for z in (-3.0, -0.2, 0.0, 0.7, 4.0)
                @test derivative(v, z) ≈ w(z) atol = 1.0e-12 rtol = 1.0e-9
            end
        end
    end
end

@testitem "weight and chaining function arguments are validated" tags = [:sample] begin
    using ScoringRules

    @test_throws ArgumentError weight_function(:norm)
    @test_throws ArgumentError chaining_function(:gumbel_cdf)
    @test_throws DomainError weight_function(:norm_cdf; sigma = 0.0)
    @test_throws DomainError weight_function(:norm_cdf; sigma = -1.0)
    @test_throws ArgumentError weight_function(:norm_cdf; mu = Inf)
    @test_throws DimensionMismatch weight_function(:norm_cdf; mu = [0.0, 1.0], sigma = [1.0, 2.0, 3.0])
    # As in R, the logistic shapes have no multivariate form.
    @test_throws ArgumentError weight_function(:logis_cdf; mu = [0.0, 1.0], sigma = [1.0, 1.0])
    # A length-d parameter applied to a forecast of another dimension.
    w = weight_function(:norm_cdf; mu = [0.0, 1.0], sigma = 1.0)
    @test_throws DimensionMismatch w([0.1, 0.2, 0.3])
    @test w([0.1, 0.2]) > 0

    # Scalars broadcast across dimensions, and the product form is the
    # per-dimension weights multiplied together.
    ws = weight_function(:norm_cdf; mu = 0.5, sigma = 2.0)
    @test ws([0.1, -0.4]) ≈ ws(0.1) * ws(-0.4) rtol = 1.0e-14
    cs = chaining_function(:norm_surv; mu = 0.5, sigma = 2.0)
    @test cs([0.1, -0.4]) ≈ [cs(0.1), cs(-0.4)] rtol = 1.0e-14
end
