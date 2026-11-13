# CRPS for the generalised extreme value (GEV) and generalised Pareto (GPD)
# distributions. Ported from R scoringRules `scores_gev.R` and `scores_gpd.R`
# (Jordan, Krüger, Lerch, Allen; GPL-2.0-or-later).
#
# Parameter conventions follow Distributions.jl:
#   GeneralizedExtremeValue(μ, σ, ξ)  — location μ, scale σ, shape ξ
#   GeneralizedPareto(μ, σ, ξ)        — location μ, scale σ, shape ξ
#
# Dependencies (in the module): SpecialFunctions gamma, digamma, expinti;
#   EpiAwareADTools cdf_ad_safe (the Gamma CDF, dual-safe in shape and x).

# ---------------------------------------------------------------------------
# GEV
# ---------------------------------------------------------------------------

"""
CRPS of a GEV(location, scale, shape) forecast at observation `y`,
in closed form (Friederichs & Thorarinsdottir 2012). Shape must be < 1.

The Gumbel limit (|shape| below `_SHAPE_ATOL`) is handled analytically via
the exponential integral Ei.
"""
function _crps_gev(y::Real, shape::Real, location::Real = 0.0, scale::Real = 1.0)
    scale < 0 && return oftype(float(y), NaN)
    shape >= 1 && return oftype(float(y), NaN)

    y_std = (y - location) / scale   # standardised observation

    out = if abs(shape) < _SHAPE_ATOL
        # Gumbel limit: use exponential integral Ei
        # out = -y_std + γ_E - log 2 - 2·Ei(-exp(-y_std))
        # where γ_E = -digamma(1)
        -y_std - digamma(one(float(y_std))) - log(oftype(float(y_std), 2)) -
            2 * expinti(-exp(-y_std))
    else
        # General case
        x_inner = 1 + shape * y_std
        # When x_inner ≤ 0 the R code sets it to 0 before computing the
        # fractional power; Julia's 0^p replicates this correctly:
        #   shape > 0 → exponent -1/shape < 0 → 0^(neg) = Inf
        #   shape < 0 → exponent -1/shape > 0 → 0^(pos) = 0
        x = (x_inner <= 0 ? zero(float(y_std)) : x_inner)^(-1 / shape)
        c1 = 2 * exp(-x) - 1
        g = gamma(1 - shape)
        # pgamma(x, 1-shape) = P(1-shape, x) = cdf(Gamma(1-shape, 1), x), via
        # cdf_ad_safe: a direct gamma_inc call is not Dual-safe in its shape
        # argument (1-shape), which depends on `shape` here (#11).
        p = cdf_ad_safe(Gamma(1 - shape, 1.0), x)
        (y_std + 1 / shape) * c1 + g / shape * (2 * p - 2^shape)
    end
    return scale * out
end

crps(d::GeneralizedExtremeValue, y::Real) = _crps_gev(y, d.ξ, d.μ, d.σ)

# ---------------------------------------------------------------------------
# GPD
# ---------------------------------------------------------------------------

"""
CRPS of a GPD(location, scale, shape) forecast at observation `y`,
in closed form (Friederichs & Thorarinsdottir 2012). Shape must be < 1.

The exponential limit is exact at `shape == 0`, and `log1p` makes the general
expression accurate to the last bit or two for every `|shape| ≥ 1e-308`, so no
threshold is needed between them. Subnormal shapes, where `shape * z`
underflows, and shapes within about `1e-9` of 1, where `2/(1-ξ)·(1-x^{1-ξ})`
cancels, are both less accurate; R does the same at the upper end.

`mass` is an optional point mass at the location (lower boundary); it is
retained for internal use but not exposed through the Distributions dispatch.
"""
function _crps_gpd(
        y::Real, shape::Real,
        location::Real = 0.0, scale::Real = 1.0,
        mass::Real = 0.0
    )
    scale < 0 && return oftype(float(y), NaN)
    shape >= 1 && return oftype(float(y), NaN)
    (mass < 0 || mass > 1) && return oftype(float(y), NaN)

    z = (y - location) / scale   # standardised observation

    # Survival function of the standardised GPD, (1 + ξz)^(−1/ξ).
    x = if shape == 0
        exp(-z)               # exponential limit
    elseif 1 + shape * z <= 0
        # Match R, which clips 1 + shape·z at 0 before the fractional power:
        # that gives 0^(-1/shape) = Inf below the support for shape > 0 (and
        # the clip from above then makes it 1) and 0 above it for shape < 0.
        shape > 0 ? one(float(z)) : zero(float(z))
    else
        # `log1p` rather than `log(1 + shape*z)`: the latter loses the whole
        # of `shape*z` to rounding for a small shape, and dividing by `shape`
        # then multiplies that error by 1/shape (relative error 2·10⁻⁴ at
        # shape = 10⁻¹²). Through `log1p` the result is accurate to the last
        # bit or two for every shape.
        exp(-log1p(shape * z) / shape)
    end
    # x represents the survival function value; clip from above at 1
    x = min(x, one(x))

    a = 1 - mass
    b = 1 - shape

    return abs(y - location) - scale * a * (2 / b * (1 - x^b) - a / (2 - shape))
end

crps(d::GeneralizedPareto, y::Real) = _crps_gpd(y, d.ξ, d.μ, d.σ)
