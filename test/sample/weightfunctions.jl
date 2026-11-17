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
            # The generator varies both across the four dimensions.
            mu = Float64(raw[i, cols["mu"]]) .+ [0.0, 0.4, -0.3, 0.8]
            sigma = Float64(raw[i, cols["sigma"]]) .* [1.0, 0.7, 1.3, 0.9]
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
    @test_throws DimensionMismatch weight_function(:norm_cdf; mu = [0.0, 1.0], sigma = [1.0, 2.0, 3.0])
    # As in R, the logistic shapes have no multivariate form. R applies the
    # same check whichever half of the pair it is asked for, and a chaining
    # function that is the antiderivative of no product weight would leave the
    # threshold-weighted score improper for the weight the caller chose.
    @test_throws ArgumentError weight_function(:logis_cdf; mu = [0.0, 1.0], sigma = [1.0, 1.0])
    @test_throws ArgumentError chaining_function(:logis_cdf; mu = [0.0, 1.0], sigma = [1.0, 1.0])
    @test_throws ArgumentError chaining_function(:logis_surv; mu = zeros(3), sigma = ones(3))
    @test_throws ArgumentError chaining_function(:logis_pdf; mu = zeros(3), sigma = ones(3))
    # An empty parameter builds a callable that can only throw.
    @test_throws DimensionMismatch weight_function(:norm_cdf; mu = Float64[], sigma = Float64[])
    @test_throws DimensionMismatch chaining_function(:norm_cdf; mu = Float64[], sigma = Float64[])
    @test_throws DimensionMismatch weight_function(:logis_cdf; mu = Float64[], sigma = Float64[])
    # A length-d parameter applied to a forecast of another dimension.
    w = weight_function(:norm_cdf; mu = [0.0, 1.0], sigma = [1.0, 1.0])
    @test_throws DimensionMismatch w([0.1, 0.2, 0.3])
    # R's get_weight_func stops on mu and sigma of different lengths.
    @test_throws DimensionMismatch weight_function(:norm_cdf; mu = [0.0, 1.0], sigma = 1.0)
    @test_throws DimensionMismatch chaining_function(:norm_cdf; mu = 0.0, sigma = [1.0, 2.0])
    @test w([0.1, 0.2]) > 0

    # A length-one vector is a scalar: its weight is univariate and its chaining
    # function applies in every dimension.
    for (mu, sigma) in (([0.5], [2.0]), (0.5, [2.0]), ([0.5], 2.0))
        wv = weight_function(:norm_cdf; mu = mu, sigma = sigma)
        cv = chaining_function(:norm_cdf; mu = mu, sigma = sigma)
        ws = weight_function(:norm_cdf; mu = 0.5, sigma = 2.0)
        cs = chaining_function(:norm_cdf; mu = 0.5, sigma = 2.0)
        @test wv(0.3) ≈ ws(0.3) rtol = 1.0e-14
        @test_throws ArgumentError wv([0.3, -0.2, 1.1])
        @test cv([0.3, -0.2]) ≈ [cs(0.3), cs(-0.2)] rtol = 1.0e-14
    end

    # A multivariate weight handed a scalar observation used to drop every
    # dimension but the first and return a plausible wrong number.
    wmv = weight_function(:norm_cdf; mu = [0.0, 10.0], sigma = [1.0, 1.0])
    @test_throws DimensionMismatch wmv(1.0)
    @test_throws DimensionMismatch chaining_function(:norm_cdf; mu = [0.0, 10.0], sigma = [1.0, 1.0])(1.0)
    @test_throws DimensionMismatch owcrps(randn(50), 0.4; weight_func = wmv)

    # As in R, a scalar-parameter weight does not score a multivariate forecast,
    # for any shape, while a scalar-parameter chaining function applies in
    # every dimension; a one-dimensional forecast takes either.
    X = [0.1 0.4 -0.3; 1.2 0.8 1.0]
    yv = [0.2, 1.1]
    for name in (:norm_cdf, :logis_cdf)
        ws = weight_function(name; mu = 0.5, sigma = 2.0)
        @test_throws ArgumentError ws([0.1, -0.4])
        @test_throws ArgumentError owes(X, yv; weight_func = ws)
        @test ws([0.1]) ≈ ws(0.1) rtol = 1.0e-14
        cs = chaining_function(name; mu = 0.5, sigma = 2.0)
        @test cs([0.1, -0.4]) ≈ [cs(0.1), cs(-0.4)] rtol = 1.0e-14
        @test isfinite(twes(X, yv; chain_func = cs))
    end
end

@testitem "an infinite mu gives R's constant weights and chaining functions" tags = [:sample] begin
    using ScoringRules

    # R: get_weight_func(name, mu, 1) and get_weight_func(name, mu, 1,
    # weight = FALSE) evaluated at x = c(-1, 0, 2).
    x = [-1.0, 0.0, 2.0]
    ref = Dict(
        (:norm_cdf, Inf) => (0.0, NaN), (:norm_cdf, -Inf) => (1.0, Inf),
        (:norm_surv, Inf) => (1.0, NaN), (:norm_surv, -Inf) => (0.0, -Inf),
        (:norm_pdf, Inf) => (0.0, 0.0), (:norm_pdf, -Inf) => (0.0, 1.0),
        (:logis_cdf, Inf) => (0.0, 0.0), (:logis_cdf, -Inf) => (1.0, Inf),
        (:logis_surv, Inf) => (1.0, nothing), (:logis_surv, -Inf) => (0.0, -Inf),
        (:logis_pdf, Inf) => (0.0, 0.0), (:logis_pdf, -Inf) => (0.0, 1.0),
    )
    for ((name, mu), (wr, cr)) in ref
        w = weight_function(name; mu = mu)
        c = chaining_function(name; mu = mu)
        @test all(==(wr), w.(x))
        # R's logis_surv chaining function at mu = Inf is the identity.
        @test isequal(c.(x), cr === nothing ? x : fill(cr, 3))
        # Refused, as most of R's weighted scores stop on the NaN weights a
        # NaN mu gives (R's twes_sample and twvs_sample return NaN instead).
        @test_throws ArgumentError weight_function(name; mu = NaN)
        @test_throws ArgumentError chaining_function(name; mu = NaN)
    end
end

@testitem "an infinite sigma gives R's flat weights and chaining functions" tags = [:sample] begin
    using ScoringRules

    # R: get_weight_func(name, 0, Inf) and get_weight_func(name, 0, Inf,
    # weight = FALSE) evaluated at x = c(-1, 0, 2).
    x = [-1.0, 0.0, 2.0]
    ref = Dict(
        :norm_cdf => (0.5, NaN), :norm_surv => (0.5, NaN), :norm_pdf => (0.0, 0.5),
        :logis_cdf => (0.5, Inf), :logis_surv => (0.5, -Inf), :logis_pdf => (0.0, 0.5),
    )
    for (name, (wr, cr)) in ref
        @test all(==(wr), weight_function(name; sigma = Inf).(x))
        @test isequal(chaining_function(name; sigma = Inf).(x), fill(cr, 3))
    end
    # R: owcrps_sample(0.4, c(0.1, 0.5, 0.9),
    #   weight_func = get_weight_func("norm_cdf", 0, Inf)).
    w = weight_function(:norm_cdf; sigma = Inf)
    @test owcrps([0.1, 0.5, 0.9], 0.4; weight_func = w) ≈ 0.06111111 rtol = 1.0e-6
end

@testitem "infinite mu and sigma together give R's normal limits" tags = [:sample] begin
    using ScoringRules

    # R: get_weight_func(name, mu, Inf) and get_weight_func(name, mu, Inf,
    # weight = FALSE) evaluated at x = c(-1, 0.1, 2), where pnorm and dnorm
    # return their limits.
    x = [-1.0, 0.1, 2.0]
    ref = Dict(
        (:norm_cdf, -Inf) => (1.0, NaN), (:norm_cdf, Inf) => (0.0, NaN),
        (:norm_surv, -Inf) => (0.0, NaN), (:norm_surv, Inf) => (1.0, NaN),
        (:norm_pdf, -Inf) => (0.0, 1.0), (:norm_pdf, Inf) => (0.0, 0.0),
    )
    for ((name, mu), (wr, cr)) in ref
        @test all(==(wr), weight_function(name; mu = mu, sigma = Inf).(x))
        @test isequal(chaining_function(name; mu = mu, sigma = Inf).(x), fill(cr, 3))
    end
    # R: owvs_sample(c(0.4, 0.3), X, weight_func =
    #   get_weight_func("norm_cdf", c(-Inf, 0), c(Inf, 1))).
    X = [0.1 0.9 0.2 0.4; 0.5 1.3 0.3 0.8]
    w = weight_function(:norm_cdf; mu = [-Inf, 0.0], sigma = [Inf, 1.0])
    @test owvs(X, [0.4, 0.3]; weight_func = w) ≈ 0.07792619 rtol = 1.0e-6
end
