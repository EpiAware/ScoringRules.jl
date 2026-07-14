# CRPS for the log-Laplace distribution.
# Ported from R scoringRules `scores_llapl.R` (Jordan, Krüger, Lerch, Allen).
#
# Formula (R source):
#   z  = (log(max(y,0)) - locationlog) / scalelog
#   p  = 0.5 + 0.5*sign(z)*pexp(|z|)   = CDF of Laplace at z
#   c1 = y * (2p - 1)
#   c2 = if z < 0: (1 - (2p)^(1+σ)) / (1+σ)
#        else:    -(1 - (2(1-p))^(1-σ)) / (1-σ)
#   c3 = σ / (4 - σ²)
#   CRPS = c1 + exp(locationlog) * (c2 + c3)
#
# Requires scalelog ∈ (0, 1).

"""
    _crps_llapl(y, locationlog, scalelog)

CRPS of a log-Laplace forecast with log-scale parameters `locationlog` and
`scalelog` (where `scalelog` ∈ (0,1)), evaluated at observation `y`.
"""
function _crps_llapl(y::Real, locationlog::Real, scalelog::Real)
    (scalelog <= 0 || scalelog >= 1) && return oftype(float(y), NaN)
    y1 = max(y, zero(y))
    z = (log(y1) - locationlog) / scalelog
    # Laplace CDF: p = 0.5 + 0.5*sign(z)*(1 - exp(-|z|)) = 0.5*(1 + sign(z)*(-expm1(-|z|)))
    az = abs(z)
    p = z < 0 ? exp(z) / 2 : 1 - exp(-z) / 2
    c1 = y * (2 * p - 1)
    if z < 0
        c2 = (1 - (2 * p)^(1 + scalelog)) / (1 + scalelog)
    else
        c2 = -(1 - (2 * (1 - p))^(1 - scalelog)) / (1 - scalelog)
    end
    c3 = scalelog / (4 - scalelog^2)
    return c1 + exp(locationlog) * (c2 + c3)
end

crps(d::LogLaplace, y::Real) = _crps_llapl(y, d.μ, d.σ)
