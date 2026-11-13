# The CRPS written out from its definition, and the forecast moments from the
# quantile function, both at 300-bit precision where the distribution supports
# it.
#
# Every other test in this suite checks a score against a number: a value from
# R, a brute-force sum, a closed form. That works for the families that have a
# closed form, and leaves the generic numeric route for a wrapped distribution
# checked against nothing, where a wrong answer is a plausible number rather
# than a failure. This module is the reference that route lacks. It reads only
# `cdf`, `ccdf`, `quantile` and the support bounds, so it shares no reasoning
# with the code it is used to check.
@testmodule DefinitionOracle begin
    using Distributions
    using QuadGK
    using ScoringRules: BoundaryMass

    export oracle_crps, oracle_moments, bigtwin

    # The mass of `(F(x) - 1{x >= y})^2` sits in a region the integrator
    # cannot find on its own, so the quantile function places a breakpoint at
    # each of these levels. Their accuracy does not matter: a breakpoint only
    # has to be roughly right for the panels either side of it to converge.
    const LEVELS = [
        1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3,
        1.0e-2, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95,
        1 - 1.0e-2, 1 - 1.0e-3, 1 - 1.0e-4, 1 - 1.0e-5, 1 - 1.0e-6,
        1 - 1.0e-8, 1 - 1.0e-10, 1 - 1.0e-12, 1 - 1.0e-14,
    ]

    function _breaks(::Type{T}, ref, y) where {T}
        pts = T[]
        for p in LEVELS
            q = try
                p <= 0.5 ? quantile(ref, p) : cquantile(ref, 1 - p)
            catch
                continue
            end
            (q isa Real && isfinite(q)) && push!(pts, T(q))
        end
        for b in (minimum(ref), maximum(ref))
            bf = T(b)
            isfinite(bf) && push!(pts, bf)
        end
        isfinite(y) && push!(pts, T(y))
        # A polynomial tail is not spent at the 1e-14 quantile: for a t with
        # three degrees of freedom the variance integrand still decays only as
        # x^-2 there, so the panels have to reach further than any quantile
        # the distribution will name.
        if !isempty(pts)
            lo, hi = first(sort(pts)), last(sort(pts))
            if !isfinite(T(minimum(ref)))
                for k in 1:8
                    push!(pts, lo - abs(lo) * T(10)^k - T(10)^k)
                end
            end
            if !isfinite(T(maximum(ref)))
                for k in 1:8
                    push!(pts, hi + abs(hi) * T(10)^k + T(10)^k)
                end
            end
        end
        sort!(pts)
        unique!(pts)
        return pts
    end

    # `quadgk` specialises on the type of its integrand, so a closure built
    # per distribution recompiles the whole quadrature for every family and
    # every wrapper around it: seconds each, and by far the largest cost of
    # these items. Boxing the function behind one concrete type trades a
    # dynamic call per evaluation, which is nothing beside the CDF itself, for
    # a single compilation.
    struct Squared{T}
        f::Any
    end
    (g::Squared{T})(x) where {T} = (v = T(g.f(x)); v * v)

    struct Plain{T}
        f::Any
    end
    (g::Plain{T})(x) where {T} = T(g.f(x))

    function _crps_continuous(::Type{T}, d, ref, y, rtol) where {T}
        yT = T(y)
        pts = _breaks(T, ref, yT)
        isempty(pts) && return T(NaN)
        # Reading the upper half through `ccdf` keeps the relative accuracy of
        # a far upper tail, where `1 - cdf(d, x)` has cancelled to nothing.
        lower = Squared{T}(x -> cdf(d, x))
        upper = Squared{T}(x -> ccdf(d, x))
        # A bound on the work, so an integrand the precision cannot resolve
        # returns something wrong and cheap rather than running away.
        a, _ = quadgk(
            lower, T(-Inf), filter(<(yT), pts)..., yT;
            rtol, order = 15, maxevals = 10^5
        )
        b, _ = quadgk(
            upper, yT, filter(>(yT), pts)..., T(Inf);
            rtol, order = 15, maxevals = 10^5
        )
        return a + b
    end

    function _window(d)
        lo, hi = float(minimum(d)), float(maximum(d))
        left = isfinite(lo) ? Int(ceil(lo)) : Int(floor(quantile(d, 1.0e-17))) - 2
        right = isfinite(hi) ? Int(floor(hi)) : Int(ceil(cquantile(d, 1.0e-17))) + 2
        return left, right
    end

    # On a unit lattice `F` is a step function, so the integral is a sum with
    # no quadrature in it at all.
    function _crps_lattice(d, y)
        isfinite(y) || return NaN
        left, right = _window(d)
        left = min(left, Int(floor(y)) - 1)
        right = max(right, Int(ceil(y)) + 1)
        (right - left) > 50_000_000 && return NaN
        total = 0.0
        for k in left:(right - 1)
            xlo, xhi = float(k), float(k + 1)
            F, S = cdf(d, k), ccdf(d, k)
            if y <= xlo
                total += S * S * (xhi - xlo)
            elseif y >= xhi
                total += F * F * (xhi - xlo)
            else
                total += F * F * (y - xlo) + S * S * (xhi - y)
            end
        end
        return total
    end

    "The CRPS of `d` at `y`, from the definition."
    function oracle_crps(d, y; rtol = 1.0e-13, ref = d)
        d isa DiscreteUnivariateDistribution && return _crps_lattice(d, float(y))
        T = typeof(float(cdf(d, median(d))))
        return _crps_continuous(T, d, ref, y, rtol)
    end

    # The moments are taken over the density. The obvious alternative, an
    # integral of the tail probability, cannot be used here: `ccdf` of a
    # truncated distribution is formed as `1 - cdf`, which is noise once the
    # tail is below the rounding of one, and the moment integrals multiply
    # that noise by x. A density has no such cancellation.
    struct Weighted{T}
        f::Any
        m::T
        k::Int
    end
    (g::Weighted{T})(x) where {T} = (w = T(x) - g.m; w^g.k * T(g.f(x)))

    # A forecast with a point mass on a bound is a density plus atoms, and the
    # atoms are not in the density: a censored distribution puts each
    # discarded tail on its bound, and a `BoundaryMass` puts a stated mass
    # there. `_parts` splits any of them into the atoms, the distribution the
    # interior density belongs to, and the weight that interior carries.
    _parts(d) = (Tuple{Float64, Float64}[], d, 1.0)

    function _parts(d::Distributions.Censored)
        b, at = d.uncensored, Tuple{Float64, Float64}[]
        d.lower === nothing || push!(at, (float(d.lower), float(cdf(b, d.lower))))
        d.upper === nothing || push!(at, (float(d.upper), float(ccdf(b, d.upper))))
        return (filter(t -> t[2] > 0, at), b, 1.0)
    end

    function _parts(d::BoundaryMass)
        at = Tuple{Float64, Float64}[]
        d.lmass > 0 && push!(at, (float(d.lower), float(d.lmass)))
        d.umass > 0 && push!(at, (float(d.upper), float(d.umass)))
        inner = truncated(d.dist; lower = d.lower, upper = d.upper)
        return (at, inner, float(1 - d.lmass - d.umass))
    end

    "The mean and variance of `d`, from its density."
    function oracle_moments(d; rtol = 1.0e-12)
        if d isa DiscreteUnivariateDistribution
            left, right = _window(d)
            (right - left) > 50_000_000 && return (NaN, NaN)
            m = 0.0
            for k in left:right
                m += k * pdf(d, k)
            end
            # Centred, because a lattice far from the origin loses the whole
            # variance to cancellation in `E[X^2] - E[X]^2`.
            v = 0.0
            for k in left:right
                v += (k - m)^2 * pdf(d, k)
            end
            return (m, v)
        end
        T = typeof(float(cdf(d, median(d))))
        atoms, base, wt = _parts(d)
        lo = max(T(minimum(d)), T(minimum(base)))
        hi = min(T(maximum(d)), T(maximum(base)))
        pts = filter(p -> lo < p < hi, _breaks(T, base, (lo + hi) / 2))
        dens = Plain{T}(x -> pdf(base, x))
        mi, _ = quadgk(
            Weighted{T}(dens.f, zero(T), 1), lo, pts..., hi;
            rtol, order = 15, maxevals = 10^5
        )
        m = T(wt) * mi
        for (x, w) in atoms
            m += T(x) * T(w)
        end
        vi, _ = quadgk(
            Weighted{T}(dens.f, m, 2), lo, pts..., hi;
            rtol, order = 15, maxevals = 10^5
        )
        v = T(wt) * vi
        for (x, w) in atoms
            v += (T(x) - m)^2 * T(w)
        end
        return (m, v)
    end

    # A truncation far out in a tail is where the generic
    # `(F(x) - F(lo)) / (F(hi) - F(lo))` cancels away in double precision, and
    # that cancellation is what the package works around. At 300 bits the same
    # naive expression is exact, so it can referee the workaround without
    # sharing its reasoning. Families whose CDF has no generic-real method get
    # no twin, and are checked in double precision instead.
    _bigparams(d) = map(p -> p isa Real ? BigFloat(p) : p, params(d))

    function bigtwin(d::Truncated)
        b = bigtwin(d.untruncated)
        b === nothing && return nothing
        return truncated(
            b; lower = d.lower === nothing ? nothing : BigFloat(d.lower),
            upper = d.upper === nothing ? nothing : BigFloat(d.upper)
        )
    end

    function bigtwin(d::Distributions.Censored)
        b = bigtwin(d.uncensored)
        b === nothing && return nothing
        return censored(
            b; lower = d.lower === nothing ? nothing : BigFloat(d.lower),
            upper = d.upper === nothing ? nothing : BigFloat(d.upper)
        )
    end

    function bigtwin(d::Distributions.AffineDistribution)
        b = bigtwin(d.ρ)
        return b === nothing ? nothing : BigFloat(d.μ) + BigFloat(d.σ) * b
    end

    function bigtwin(d::BoundaryMass)
        b = bigtwin(d.dist)
        b === nothing && return nothing
        return try
            BoundaryMass(
                b; lower = BigFloat(d.lower), upper = BigFloat(d.upper),
                lmass = BigFloat(d.lmass), umass = BigFloat(d.umass)
            )
        catch
            nothing
        end
    end

    function bigtwin(d::UnivariateDistribution)
        d isa DiscreteUnivariateDistribution && return nothing
        return try
            b = Base.typename(typeof(d)).wrapper(_bigparams(d)...)
            # A family with no generic-real CDF promotes its argument and
            # hands back a Float64, which would be an oracle no sharper than
            # the thing it is judging.
            cdf(b, BigFloat(median(d))) isa BigFloat ? b : nothing
        catch
            nothing
        end
    end
end
