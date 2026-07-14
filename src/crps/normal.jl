# CRPS for the normal family: Normal, truncated normal, censored normal and the
# generalised truncated/censored normal (point masses `lmass`/`umass` at the
# truncation points). Ported from R scoringRules `scores_norm.R`
# (Jordan, Krüger, Lerch, Allen).

# --- helpers on the standard normal ---------------------------------------
# `Φ(x*√2)` appears throughout (it is `pnorm(x, sd = 1/√2)` in the R source).
@inline _Phi_root2(x::Real) = _norm_cdf(x * sqrt(oftype(float(x), 2)))
const _INV_SQRTPI = 1 / sqrt(π)

# --- standard normal, N(μ, σ) ---------------------------------------------
"""
CRPS of a `Normal(μ, σ)` forecast, in closed form (Gneiting et al. 2005).
"""
function _crps_norm(y::Real, μ::Real, σ::Real)
    z = (y - μ) / σ
    return σ * (z * (2 * _norm_cdf(z) - 1) + 2 * _norm_pdf(z) - _INV_SQRTPI)
end

crps(d::Normal, y::Real) = _crps_norm(y, d.μ, d.σ)

# --- lower/upper bound extraction from Truncated / Censored ----------------
_lo(x) = x === nothing ? -Inf : float(x)
_hi(x) = x === nothing ? Inf : float(x)

# --- truncated normal ------------------------------------------------------
# Unit-scale (σ = 1) core with the `lower > 3` sign-swap used in the R source
# for numerical stability far in the tail.
function _crps_tnorm_unit(y::Real, l::Real, u::Real)
    if l > 3
        y, l, u = -y, -u, -l
    end
    p_l = 0.0;
    out_l = 0.0
    p_u = 1.0;
    out_u = 1.0
    z = y
    if isfinite(l)
        p_l = _norm_cdf(l);
        out_l = _Phi_root2(l);
        z = max(l, z)
    end
    if isfinite(u)
        p_u = _norm_cdf(u);
        out_u = _Phi_root2(u);
        z = min(u, z)
    end
    l > u && return NaN
    l == u && return abs(y - z)
    a = p_u - p_l
    b = out_u - out_l
    out_z = z * (2 * _norm_cdf(z) - p_l - p_u) + 2 * _norm_pdf(z)
    out = (out_z - b / a * _INV_SQRTPI) / a
    return out + abs(y - z)
end

function _crps_tnorm(y::Real, μ::Real, σ::Real, l::Real, u::Real)
    σ < 0 && return oftype(float(y), NaN)
    y -= μ
    isfinite(l) && (l -= μ)
    isfinite(u) && (u -= μ)
    if σ == 0
        return (l < 0 && u > 0) ? abs(y) : σ * _crps_tnorm_unit(y, l, u)
    end
    return σ * _crps_tnorm_unit(y / σ, isfinite(l) ? l / σ : l, isfinite(u) ? u / σ : u)
end

function crps(d::Truncated{<:Normal}, y::Real)
    μ, σ = params(d.untruncated)
    return _crps_tnorm(y, μ, σ, _lo(d.lower), _hi(d.upper))
end

# --- censored normal -------------------------------------------------------
function _crps_cnorm_unit(y::Real, l::Real, u::Real)
    out_l1 = 0.0;
    out_u1 = 0.0;
    out_l2 = 0.0;
    out_u2 = 1.0
    z = y
    if isfinite(l)
        p_l = _norm_cdf(l)
        out_l1 = -l * p_l^2 - 2 * _norm_pdf(l) * p_l
        out_l2 = _Phi_root2(l)
        z = max(l, z)
    end
    if isfinite(u)
        p_u = _norm_ccdf(u)
        out_u1 = u * p_u^2 - 2 * _norm_pdf(u) * p_u
        out_u2 = _Phi_root2(u)
        z = min(u, z)
    end
    l > u && return NaN
    l == u && return abs(y - z)
    b = out_u2 - out_l2
    out_z = z * (2 * _norm_cdf(z) - 1) + 2 * _norm_pdf(z)
    out = out_z + out_l1 + out_u1 - b * _INV_SQRTPI
    return out + abs(y - z)
end

@inline _norm_ccdf(x::Real) = erfc(x / sqrt(oftype(float(x), 2))) / 2

function _crps_cnorm(y::Real, μ::Real, σ::Real, l::Real, u::Real)
    σ < 0 && return oftype(float(y), NaN)
    y -= μ
    isfinite(l) && (l -= μ)
    isfinite(u) && (u -= μ)
    if σ == 0
        return l <= u ? abs(y - max(l, 0) - min(u, 0)) : oftype(float(y), NaN)
    end
    return σ * _crps_cnorm_unit(y / σ, isfinite(l) ? l / σ : l, isfinite(u) ? u / σ : u)
end

function crps(d::Censored{<:Normal}, y::Real)
    μ, σ = params(d.uncensored)
    return _crps_cnorm(y, μ, σ, _lo(d.lower), _hi(d.upper))
end

# --- generalised truncated/censored normal --------------------------------
# Point masses `lmass`, `umass` at the (finite) lower/upper truncation points.
function _crps_gtcnorm_unit(y::Real, l::Real, u::Real, lmass::Real, umass::Real)
    if l > 3
        y, l, u = -y, -u, -l
        lmass, umass = umass, lmass
    end
    out_l1 = 0.0;
    out_l2 = 0.0;
    out_l3 = 0.0
    out_u1 = 0.0;
    out_u2 = 0.0
    p_l = 0.0;
    p_u = 1.0;
    out_u3 = 1.0
    z = y
    if isfinite(l) || lmass != 0
        (lmass < 0 || lmass > 1) && return NaN
        p_l = _norm_cdf(l)
        out_l1 = lmass == 0 ? 0.0 : l * lmass^2
        out_l2 = 2 * _norm_pdf(l) * lmass
        out_l3 = _Phi_root2(l)
        z = max(l, z)
    end
    if isfinite(u) || umass != 0
        (umass < 0 || umass > 1) && return NaN
        p_u = _norm_cdf(u)
        out_u1 = umass == 0 ? 0.0 : u * umass^2
        out_u2 = 2 * _norm_pdf(u) * umass
        out_u3 = _Phi_root2(u)
        z = min(u, z)
    end
    l > u && return NaN
    l == u && return abs(y - z)
    a1 = p_u - p_l
    a2 = 1 - (umass + lmass)
    (a2 < 0 || a2 > 1) && return NaN
    b = out_u3 - out_l3
    out = out_u1 - out_l1 +
          (z * (2 * a2 * _norm_cdf(z) - (1 - 2 * lmass) * p_u - (1 - 2 * umass) * p_l) +
           (2 * _norm_pdf(z) - out_u2 - out_l2 - a2 * b / a1 * _INV_SQRTPI) * a2) / a1
    return out + abs(y - z)
end

function _crps_gtcnorm(
        y::Real, μ::Real, σ::Real, l::Real, u::Real, lmass::Real, umass::Real)
    σ < 0 && return oftype(float(y), NaN)
    y -= μ
    isfinite(l) && (l -= μ)
    isfinite(u) && (u -= μ)
    if σ == 0
        if l < 0 && u > 0
            return (min(y, 0) - l) * lmass^2 - min(y, 0) * (1 - lmass)^2 +
                   (u - max(y, 0)) * umass^2 + max(y, 0) * (1 - umass)^2
        end
        return σ * _crps_gtcnorm_unit(y, l, u, lmass, umass)
    end
    return σ * _crps_gtcnorm_unit(y / σ, isfinite(l) ? l / σ : l,
        isfinite(u) ? u / σ : u, lmass, umass)
end
