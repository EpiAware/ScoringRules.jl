# CRPS for the beta distribution. Ported from R scoringRules `scores_beta.R`
# (Jordan, Krüger, Lerch, Allen).

"""
    _crps_beta(y, shape1, shape2, lower, upper)

CRPS of a Beta(shape1, shape2) distribution scaled to [lower, upper],
in closed form.  For [0, 1] bounds the formula is

    c1 + (shape1/(shape1+shape2)) · (c3 − c4)

where
    c1 = y · (2·I_y(shape1, shape2) − 1)
    c3 = 1 − 2·I_y(shape1+1, shape2)
    c4 = (2/shape1) · B(2·shape1, 2·shape2) / B(shape1, shape2)²

and I_x(a,b) is the regularised incomplete beta (= `beta_inc(a,b,x)[1]`).
A Stirling approximation is substituted for c4 when the beta-function ratio
overflows.  Non-unit intervals are handled by linear rescaling.
"""
function _crps_beta(y::Real, shape1::Real, shape2::Real, lower::Real, upper::Real)
    if lower == 0 && upper == 1
        z = clamp(y, zero(y), one(y))
        c1 = y * (2 * beta_inc(shape1, shape2, z)[1] - 1)
        c2 = shape1 / (shape1 + shape2)
        c3 = 1 - 2 * beta_inc(shape1 + 1, shape2, z)[1]
        lb = logbeta(shape1, shape2)
        c4_log = log(2) - log(shape1) + logbeta(2 * shape1, 2 * shape2) - 2 * lb
        c4 = isfinite(c4_log) ? exp(c4_log) : sqrt(shape2 / (oftype(float(shape1), π) * shape1 * (shape1 + shape2)))
        return c1 + c2 * (c3 - c4)
    else
        !isfinite(lower) && return oftype(float(y), NaN)
        !isfinite(upper) && return oftype(float(y), NaN)
        sc = upper - lower
        sc < 0 && return oftype(float(y), NaN)
        sc == 0 && return abs(y - lower)
        return sc * _crps_beta((y - lower) / sc, shape1, shape2, zero(lower), one(upper))
    end
end

crps(d::Beta, y::Real) = _crps_beta(y, d.α, d.β, zero(float(y)), one(float(y)))
