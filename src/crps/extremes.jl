# CRPS for the generalised extreme value (GEV) and generalised Pareto (GPD)
# distributions. Ported from R scoringRules `scores_gev.R` and `scores_gpd.R`
# (Jordan, Krüger, Lerch, Allen; GPL-2.0-or-later).
#
# Parameter conventions follow Distributions.jl:
#   GeneralizedExtremeValue(μ, σ, ξ)  — location μ, scale σ, shape ξ
#   GeneralizedPareto(μ, σ, ξ)        — location μ, scale σ, shape ξ
#
# Dependencies (in the module): SpecialFunctions gamma, loggamma, digamma,
#   expinti; gamma_inc from SpecialFunctions (returns (P,Q) regularised).

# Below this |shape| the general closed forms are numerically unstable, so the
# analytic limit (Gumbel for GEV, exponential for GPD) is used instead.
const _SHAPE_ATOL = 1e-12

# ---------------------------------------------------------------------------
# GEV
# ---------------------------------------------------------------------------

"""
CRPS of a GEV(location, scale, shape) forecast at observation `y`,
in closed form (Friederichs & Thorarinsdottir 2012). Shape must be < 1.

The Gumbel limit (|shape| < 1e-12) is handled analytically via the
exponential integral Ei.
"""
function _crps_gev(y::Real, shape::Real, location::Real = 0.0, scale::Real = 1.0)
    scale < 0 && return oftype(float(y), NaN)
    shape >= 1 && return oftype(float(y), NaN)

    y_std = (y - location) / scale   # standardised observation

    out = if abs(shape) < _SHAPE_ATOL
        # Gumbel limit: use exponential integral Ei
        # out = -y_std - γ_E - log 2 - 2·Ei(-exp(-y_std))
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
        # pgamma(x, 1-shape) is the lower regularised incomplete gamma P(1-shape, x)
        p = gamma_inc(1 - shape, x)[1]
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

`mass` is an optional point mass at the location (lower boundary); it is
retained for internal use but not exposed through the Distributions dispatch.
"""
function _crps_gpd(y::Real, shape::Real,
        location::Real = 0.0, scale::Real = 1.0,
        mass::Real = 0.0)
    scale < 0 && return oftype(float(y), NaN)
    shape >= 1 && return oftype(float(y), NaN)
    (mass < 0 || mass > 1) && return oftype(float(y), NaN)

    z = (y - location) / scale   # standardised observation

    x = if abs(shape) < _SHAPE_ATOL
        exp(-z)               # exponential limit
    else
        # Match R: clip 1 + shape·z at 0, then raise to -1/shape. Below the
        # support this gives 0^(-1/shape) = Inf for shape > 0 (clipped to 1
        # below) and 0 for shape < 0, exactly as scores_gpd.R does.
        max(1 + shape * z, zero(float(z)))^(-1 / shape)
    end
    # x represents the survival function value; clip from above at 1
    x = min(x, one(x))

    a = 1 - mass
    b = 1 - shape

    return abs(y - location) - scale * a * (2 / b * (1 - x^b) - a / (2 - shape))
end

crps(d::GeneralizedPareto, y::Real) = _crps_gpd(y, d.ξ, d.μ, d.σ)
