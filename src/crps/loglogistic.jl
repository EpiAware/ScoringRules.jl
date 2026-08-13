# CRPS for the log-logistic (Fisk) distribution.
# Ported from R scoringRules `scores_llogis.R` (Jordan, Krüger, Lerch, Allen).
#
# Formula (R source):
#   p  = plogis(log(max(y,0)); locationlog, scalelog)
#   c1 = y * (2p - 1)
#   c2 = 2 * exp(locationlog) * beta(1+scalelog, 1-scalelog)
#   c3 = (1-scalelog)/2 - pbeta(p, 1+scalelog, 1-scalelog)
#   CRPS = c1 + c2 * c3
#
# Requires scalelog ∈ (0, 1) for the CRPS to exist (beta function needs
# both arguments positive).

"""
    _crps_llogis(y, locationlog, scalelog)

CRPS of a log-logistic forecast with log-scale parameters `locationlog` and
`scalelog` (where `scalelog` ∈ (0,1)), evaluated at observation `y`.
"""
function _crps_llogis(y::Real, locationlog::Real, scalelog::Real)
    (scalelog <= 0 || scalelog >= 1) && return oftype(float(y), NaN)
    y1 = max(y, zero(y))
    # At y ≤ 0 the CDF is 0 (left support boundary) for every parameter value.
    # Compute it as a constant there rather than pushing log(0) = -Inf through
    # an AD dual: the primal `logistic(-Inf)` is a clean 0, but its `scalelog`
    # partial is NaN, which would poison the whole gradient (#6 review).
    p = y1 > 0 ? logistic((log(y1) - locationlog) / scalelog) :
        oftype(float(locationlog) * float(scalelog) * one(y), 0)
    c1 = y * (2 * p - 1)
    # beta(a,b) = Γ(a)Γ(b)/Γ(a+b); use logbeta for numerical stability
    c2 = 2 * exp(locationlog) * exp(logbeta(1 + scalelog, 1 - scalelog))
    # pbeta(p, a, b) = regularised incomplete beta function I_p(a,b), routed
    # through cdf_ad_safe rather than beta_inc directly: beta_inc cannot take
    # a Dual in either shape argument, breaking `scalelog` differentiation
    # (#6). The incomplete-beta term itself needs no p==0/p==1 guard: the beta
    # density's x-derivative only diverges at the boundary whose shape argument
    # is < 1 (the second, 1 - scalelog, at p == 1 — unreachable for finite y),
    # while at p == 0 the first argument 1 + scalelog > 1 keeps the density
    # finite. The only NaN risk was `p`'s log(0) above, now guarded.
    Ip = cdf_ad_safe(Beta(1 + scalelog, 1 - scalelog), p)
    c3 = (1 - scalelog) / 2 - Ip
    return c1 + c2 * c3
end

# Distributions.jl provides `LogLogistic(α, β)` (Fisk), the standard scale/shape
# parameterisation. It maps to the log-scale (location, scale) form used above by
# `locationlog = log(α)` and `scalelog = 1/β` (since `log(X) ~ Logistic(log α, 1/β)`).
crps(d::LogLogistic, y::Real) = _crps_llogis(y, log(d.α), 1 / d.β)
