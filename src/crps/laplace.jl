# CRPS for the Laplace distribution. Ported from R scoringRules `scores_lapl.R`
# (Jordan, Krüger, Lerch, Allen).

"""
    _crps_lapl(y, location, scale)

CRPS of a Laplace(location, scale) forecast in closed form.

Formula (unit-scale, location-shifted):
    CRPS = scale * (|z| + exp(-|z|) - 3/4)
where z = (y - location) / scale.
"""
function _crps_lapl(y::Real, location::Real, scale::Real)
    z = (y - location) / scale
    az = abs(z)
    return scale * (az + exp(-az) - oftype(float(y), 0.75))
end

crps(d::Laplace, y::Real) = _crps_lapl(y, d.μ, d.θ)
