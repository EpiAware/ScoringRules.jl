# CRPS for the gamma distribution. Ported from R scoringRules `scores_gamma.R`
# (Jordan, Krüger, Lerch, Allen).

"""
    _crps_gamma(y, shape, scale)

CRPS of a Gamma(shape, scale) forecast in closed form.

Formula:
    y·(2·F₁ − 1) − scale·(shape·(2·F₂ − 1) + 1/B(0.5, shape))
where F₁ = CDF of Gamma(shape, scale) at y,
      F₂ = CDF of Gamma(shape+1, scale) at y,
      B(a,b) = exp(logbeta(a, b)).

F₁/F₂ go through `cdf_ad_safe` rather than `cdf` directly: the stock
`cdf(::Gamma)` routes through `SpecialFunctions.gamma_inc`, whose
`ChainRule` leaves the shape-parameter partial unimplemented, breaking
`shape` differentiation on every AD backend (#11).
"""
function _crps_gamma(y::Real, shape::Real, scale::Real)
    p1 = cdf_ad_safe(Gamma(shape, scale), y)
    p2 = cdf_ad_safe(Gamma(shape + 1, scale), y)
    return y * (2 * p1 - 1) -
           scale * (shape * (2 * p2 - 1) + exp(-logbeta(oftype(float(shape), 0.5), shape)))
end

crps(d::Gamma, y::Real) = _crps_gamma(y, shape(d), scale(d))
