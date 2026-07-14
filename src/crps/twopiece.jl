# CRPS for the two-piece-normal and two-piece-exponential distributions.
# Ported from R scoringRules `scores_2pnorm.R` and `scores_2pexp.R`
# (Jordan, Krüger, Lerch, Allen).
#
# Both implementations decompose the CRPS into two half-line contributions
# by reusing the generalised truncated/censored helpers already defined in
# crps/normal.jl (_crps_gtcnorm) and crps/exponential.jl (_crps_expM).

# ---------------------------------------------------------------------------
# Two-piece normal
# ---------------------------------------------------------------------------
# R:
#   y_neg = min(y - location, 0)
#   y_pos = max(y - location, 0)
#   s = scale1 + scale2
#   crps_gtcnorm(y_neg, scale=scale1, upper=0, umass=scale2/s) +
#     crps_gtcnorm(y_pos, scale=scale2, lower=0, lmass=scale1/s)
#
# Julia _crps_gtcnorm signature: (y, μ, σ, l, u, lmass, umass)

"""
    _crps_2pnorm(y, scale1, scale2, location)

CRPS of a two-piece-normal forecast in closed form.
Calls `_crps_gtcnorm` (generalised truncated/censored normal) for each arm.
"""
function _crps_2pnorm(y::Real, scale1::Real, scale2::Real, location::Real)
    yc = y - location                      # centred observation
    y_neg = min(yc, zero(yc))             # ≤ 0: left arm
    y_pos = max(yc, zero(yc))             # ≥ 0: right arm
    s = scale1 + scale2
    if s == 0
        return abs(yc)
    end
    lhs = _crps_gtcnorm(
        y_neg, zero(yc), scale1, oftype(yc, -Inf), zero(yc), zero(yc), scale2 / s)
    rhs = _crps_gtcnorm(
        y_pos, zero(yc), scale2, zero(yc), oftype(yc, Inf), scale1 / s, zero(yc))
    return lhs + rhs
end

crps(d::TwoPieceNormal, y::Real) = _crps_2pnorm(y, d.scale1, d.scale2, d.location)

# ---------------------------------------------------------------------------
# Two-piece exponential
# ---------------------------------------------------------------------------
# R:
#   y_neg = min(y - location, 0)
#   y_pos = max(y - location, 0)
#   s = scale1 + scale2
#   crps_expM(-y_neg, scale=scale1, mass=scale2/s) +
#     crps_expM( y_pos, scale=scale2, mass=scale1/s)
#
# Julia _crps_expM signature: (y, location, scale, mass)
# with location=0 here (shifts already applied in y_neg/y_pos).

"""
    _crps_2pexp(y, scale1, scale2, location)

CRPS of a two-piece-exponential forecast in closed form.
Calls `_crps_expM` (exponential with point mass) for each arm.
"""
function _crps_2pexp(y::Real, scale1::Real, scale2::Real, location::Real)
    yc = y - location
    y_neg = min(yc, zero(yc))   # ≤ 0
    y_pos = max(yc, zero(yc))   # ≥ 0
    s = scale1 + scale2
    if s == 0
        return abs(yc)
    end
    lhs = _crps_expM(-y_neg, zero(yc), scale1, scale2 / s)
    rhs = _crps_expM(y_pos, zero(yc), scale2, scale1 / s)
    return lhs + rhs
end

crps(d::TwoPieceExponential, y::Real) = _crps_2pexp(y, d.scale1, d.scale2, d.location)
