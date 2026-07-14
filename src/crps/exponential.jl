# CRPS for the exponential distribution and its location-scale-mass extension.
# Ported from R scoringRules `scores_exp.R` (Jordan, Krüger, Lerch, Allen).

# Exponential CDF: P(X ≤ y) for X ~ Exp(rate). Returns 0 for y ≤ 0.
@inline _cdf_exp(y::Real, rate::Real) = y > 0 ? -expm1(-rate * y) : zero(float(y))

"""
    _crps_exp(y, rate)

CRPS of an Exp(rate) forecast in closed form.

Formula: |y| - (2·F(y) - 0.5) / rate  where F is the Exp(rate) CDF.
"""
function _crps_exp(y::Real, rate::Real)
    return abs(y) - (2 * _cdf_exp(y, rate) - oftype(float(y), 0.5)) / rate
end

crps(d::Exponential, y::Real) = _crps_exp(y, 1 / scale(d))

"""
    _crps_expM(y, location, scale, mass)

CRPS of a mixed distribution that places point mass `mass` at `location` and
distributes the remaining probability (1 - mass) as Exp(1/scale) shifted by
`location`. Internal helper; reused by the two-piece exponential family.

Formula: |y - location| - scale·a·(2·F(y - location, 1/scale) - 0.5·a)
where a = 1 - mass and F is the Exp(1/scale) CDF.
"""
function _crps_expM(y::Real, location::Real, scale::Real, mass::Real)
    z = y - location
    a = 1 - mass
    return abs(z) - scale * a * (2 * _cdf_exp(z, 1 / scale) - oftype(float(y), 0.5) * a)
end
