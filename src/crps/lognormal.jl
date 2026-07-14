# CRPS for the log-normal distribution. Ported from R scoringRules
# `scores_lnorm.R` (Jordan, Krüger, Lerch, Allen).

"""
    _crps_lnorm(y, meanlog, sdlog)

CRPS of a LogNormal(meanlog, sdlog) forecast in closed form.

Formula:
    c1 − c2·c3
where
    c1 = y · (2·Φ_LN(y) − 1)
    c2 = 2 · exp(meanlog + sdlog²/2)
    c3 = Φ_LN*(y) + Φ(sdlog/√2) − 1

and Φ_LN  is the LogNormal(meanlog, sdlog) CDF,
    Φ_LN* is the LogNormal(meanlog + sdlog², sdlog) CDF,
    Φ     is the standard-normal CDF.
"""
function _crps_lnorm(y::Real, meanlog::Real, sdlog::Real)
    c1 = y * (2 * cdf(LogNormal(meanlog, sdlog), y) - 1)
    c2 = 2 * exp(meanlog + oftype(float(sdlog), 0.5) * sdlog^2)
    c3 = cdf(LogNormal(meanlog + sdlog^2, sdlog), y) +
         _norm_cdf(sdlog / sqrt(oftype(float(sdlog), 2))) - 1
    return c1 - c2 * c3
end

crps(d::LogNormal, y::Real) = _crps_lnorm(y, d.μ, d.σ)
