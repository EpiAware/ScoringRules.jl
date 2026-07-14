# CRPS for the uniform distribution (with optional point masses at the
# endpoints). Ported from R scoringRules `scores_unif.R`
# (Jordan, Krüger, Lerch, Allen).

"""
    _crps_unif(y, min, max, lmass, umass)

CRPS of a generalised uniform distribution on [min, max] with optional point
masses `lmass` at `min` and `umass` at `max`.  Setting both masses to zero
gives the plain Uniform(min, max) CRPS.

For the unit interval [0, 1] the formula is

    |y − z| + z²·a − z·(1 − 2·lmass) + a²/3 + (1 − lmass)·umass

where z = clamp(y, 0, 1) and a = 1 − (lmass + umass).
Non-unit intervals are handled by linear rescaling.
"""
function _crps_unif(y::Real, mn::Real, mx::Real, lmass::Real, umass::Real)
    if mn == 0 && mx == 1
        z = clamp(y, zero(y), one(y))
        a = 1 - (lmass + umass)
        a < 0 && return oftype(float(y), NaN)
        return abs(y - z) + z^2 * a - z * (1 - 2 * lmass) + a^2 / 3 + (1 - lmass) * umass
    else
        !isfinite(mn) && return oftype(float(y), NaN)
        !isfinite(mx) && return oftype(float(y), NaN)
        sc = mx - mn
        sc < 0 && return oftype(float(y), NaN)
        sc == 0 && return abs(y - mn)
        return sc * _crps_unif((y - mn) / sc, zero(mn), one(mx), lmass, umass)
    end
end

crps(d::Uniform, y::Real) = _crps_unif(y, d.a, d.b, zero(float(y)), zero(float(y)))
