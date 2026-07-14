# CRPS for the logistic family: Logistic, truncated logistic, censored logistic
# and the generalised truncated/censored logistic (point masses `lmass`/`umass`
# at the truncation points). Ported from R scoringRules `scores_logis.R`
# (Jordan, Krüger, Lerch, Allen).
#
# Key identity used throughout (standard logistic, σ = 1):
#   G(z) := z · F(z) + log F(-z)        (antiderivative appearing in CRPS)
# Taylor expansion for z → -∞ to avoid underflow:
#   F(z) + log(1 - F(z))  ≈  -F(z)²/2 - F(z)³/3   when F(z) < 1e-8

# --- logistic helpers ---------------------------------------------------------

# Numerically stable   F(z) + log F(-z)  =  F(z) + log(1 - F(z)).
# When F(z) is tiny, log(1-F(z)) ≈ -F(z) - F(z)²/2 - F(z)³/3 collapses to
# the Taylor expansion used in the R source.
@inline function _logis_Fz_plus_logFmz(z::Real)
    p = logistic(z)
    return p > 1e-8 ? p + log1p(-p) : -p^2 / 2 - p^3 / 3
end

# Normalising constant appearing in the truncated/gtc formulas:
#   out(z) = [F(z) + log F(-z)] - F(z)·[z·F(z) + 2·log F(-z)]
# Equivalent to the R expressions `ifelse(p>1e-8, p+lp_m, Taylor) - p*(z*p + 2*lp_m)`.
@inline function _logis_out(z::Real)
    p = logistic(z)
    lpm = log1p(-p)             # log F(-z)
    fz = _logis_Fz_plus_logFmz(z)   # F(z) + log F(-z), Taylor-safe
    return fz - p * (z * p + 2 * lpm)
end

# --- standard logistic --------------------------------------------------------
"""
CRPS of a `Logistic(μ, σ)` forecast, in closed form.

Unit formula (σ = 1, μ = 0):  CRPS = y - 2·log F(y) - 1
"""
function _crps_logis(y::Real, location::Real, scale::Real)
    scale < 0 && return oftype(float(y), NaN)
    z = (y - location) / scale
    return scale * (z - 2 * log(logistic(z)) - 1)
end

crps(d::Logistic, y::Real) = _crps_logis(y, d.μ, d.θ)

# --- censored logistic --------------------------------------------------------
# Unit-scale (σ = 1) core.
function _crps_clogis_unit(y::Real, l::Real, u::Real)
    out_l = 0.0
    out_u = 0.0
    z = y
    if isfinite(l)
        # out_l = F(l) + log F(-l)
        out_l = _logis_Fz_plus_logFmz(l)
        z = max(l, z)
    end
    if isfinite(u)
        # out_u = F(-u) + log F(u)
        p_mu = logistic(-u)           # 1 - F(u)
        out_u = p_mu + log(logistic(u))
        z = min(u, z)
    end
    l > u && return NaN
    l == u && return zero(float(y))
    out_z = z - 2 * log(logistic(z)) - 1
    return out_z + out_l + out_u + abs(y - z)
end

function _crps_clogis(y::Real, location::Real, scale::Real, l::Real, u::Real)
    scale < 0 && return oftype(float(y), NaN)
    y0 = y - location
    l0 = isfinite(l) ? l - location : l
    u0 = isfinite(u) ? u - location : u
    if scale == 0
        l0 <= u0 || return oftype(float(y), NaN)
        return abs(y0 - max(l0, zero(l0)) - min(u0, zero(u0)))
    end
    return scale * _crps_clogis_unit(y0 / scale,
        isfinite(l0) ? l0 / scale : l0,
        isfinite(u0) ? u0 / scale : u0)
end

function crps(d::Censored{<:Logistic}, y::Real)
    μ, θ = params(d.uncensored)
    return _crps_clogis(y, μ, θ, _lo(d.lower), _hi(d.upper))
end

# --- truncated logistic -------------------------------------------------------
# Unit-scale core with the `lower > 3` sign-swap for numerical stability far
# in the upper tail (mirrors the R source and the normal.jl pattern).
function _crps_tlogis_unit(y::Real, l::Real, u::Real)
    # Reflect when lower > 3: logistic is symmetric about 0
    if l > 3
        y, l, u = -y, -u, -l
    end

    out_l = 0.0;
    p_l = 0.0
    out_u = 1.0;
    p_u = 1.0
    z = y
    if isfinite(l)
        p_l = logistic(l)
        out_l = _logis_out(l)
        z = max(l, z)
    end
    if isfinite(u)
        p_u = logistic(u)
        out_u = _logis_out(u)
        z = min(u, z)
    end
    l > u && return NaN
    l == u && return zero(float(y))

    a = p_u - p_l
    b = out_u - out_l
    b == 0 && return NaN

    lp_mz = log1p(-logistic(z))   # log F(-z)
    out_z = z * (-p_l - p_u) - 2 * lp_mz

    out = (out_z - b / a) / a
    return out + abs(y - z)
end

function _crps_tlogis(y::Real, location::Real, scale::Real, l::Real, u::Real)
    scale < 0 && return oftype(float(y), NaN)
    y0 = y - location
    l0 = isfinite(l) ? l - location : l
    u0 = isfinite(u) ? u - location : u
    if scale == 0
        return (l0 < 0 && u0 > 0) ? abs(y0) : oftype(float(y), NaN)
    end
    return scale * _crps_tlogis_unit(y0 / scale,
        isfinite(l0) ? l0 / scale : l0,
        isfinite(u0) ? u0 / scale : u0)
end

function crps(d::Truncated{<:Logistic}, y::Real)
    μ, θ = params(d.untruncated)
    return _crps_tlogis(y, μ, θ, _lo(d.lower), _hi(d.upper))
end

# --- generalised truncated/censored logistic ----------------------------------
# Point masses `lmass`, `umass` at the (finite) lower/upper truncation points.
# Internal only — no Distributions.jl type captures this directly.
function _crps_gtclogis(y::Real, location::Real, scale::Real,
        l::Real, u::Real, lmass::Real, umass::Real)
    scale < 0 && return oftype(float(y), NaN)
    y0 = y - location
    l0 = isfinite(l) ? l - location : l
    u0 = isfinite(u) ? u - location : u
    if scale == 0
        if l0 < 0 && u0 > 0
            return (min(y0, zero(y0)) - l0) * lmass^2 -
                   min(y0, zero(y0)) * (1 - lmass)^2 +
                   (u0 - max(y0, zero(y0))) * umass^2 +
                   max(y0, zero(y0)) * (1 - umass)^2
        end
        return oftype(float(y), NaN)
    end
    return scale * _crps_gtclogis_unit(y0 / scale,
        isfinite(l0) ? l0 / scale : l0,
        isfinite(u0) ? u0 / scale : u0,
        lmass, umass)
end

function _crps_gtclogis_unit(y::Real, l::Real, u::Real, lmass::Real, umass::Real)
    # Reflect when lower > 3 (tail stability)
    if l > 3
        y, l, u = -y, -u, -l
        lmass, umass = umass, lmass
    end

    out_l1 = out_l2 = out_l3 = 0.0
    out_u1 = out_u2 = 0.0
    p_l = 0.0
    p_u = 1.0;
    out_u3 = 1.0
    z = y

    if isfinite(l) || lmass != 0
        (lmass < 0 || lmass > 1) && return NaN
        p_l = logistic(l)
        lp_ml = log1p(-p_l)       # log F(-l)
        out_l1 = lmass == 0 ? 0.0 : l * lmass^2
        out_l2 = isfinite(l) ? 2 * (l * p_l + lp_ml) * lmass : 0.0
        out_l3 = isfinite(l) ? _logis_out(l) : 0.0
        z = max(l, z)
    end
    if isfinite(u) || umass != 0
        (umass < 0 || umass > 1) && return NaN
        p_u = logistic(u)
        lp_mu = log1p(-p_u)       # log F(-u)
        out_u1 = umass == 0 ? 0.0 : u * umass^2
        out_u2 = isfinite(u) ? 2 * (u * p_u + lp_mu) * umass : 0.0
        out_u3 = isfinite(u) ? _logis_out(u) : 1.0
        z = min(u, z)
    end

    l > u && return NaN
    l == u && return zero(float(y))

    a1 = p_u - p_l
    a2 = 1 - (umass + lmass)
    (a2 < 0 || a2 > 1) && return NaN
    b = out_u3 - out_l3
    b == 0 && return NaN

    lp_mz = log1p(-logistic(z))   # log F(-z)

    out = out_u1 - out_l1 -
          (z * ((1 - 2 * lmass) * p_u + (1 - 2 * umass) * p_l) +
           (2 * lp_mz - out_u2 - out_l2 + a2 * b / a1) * a2) / a1

    return out + abs(y - z)
end
