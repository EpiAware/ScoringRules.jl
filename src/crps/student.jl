# CRPS for the Student-t family: TDist, location–scale t, truncated t, censored t
# and the generalised truncated/censored t (point masses `lmass`/`umass` at the
# truncation points). Ported from R scoringRules `scores_t.R`
# (Jordan, Krüger, Lerch, Allen).

# --- helpers for the standard t distribution ----------------------------------

# G(z, df) = −(df + z²)/(df−1) · f_t(z; df), used throughout as the
# antiderivative of the t CDF up to a sign; named G in the R source.
@inline _t_G(z::Real, df::Real) = -(df + z^2) / (df - 1) * pdf(TDist(df), z)

# AD-safe Student-t CDF, differentiable in `df` (#6). `cdf(::TDist)` routes
# through StatsFuns' tdistcdf -> fdistccdf -> betaccdf chain, each step
# sharing one type parameter across all its arguments the same way
# StatsFuns.gammaccdf does for Poisson (see crps/discrete.jl): promoting `df`
# to a Dual forces the untouched constant `1` (the numerator "ν1" argument)
# to the same Dual type, and beta_inc has no method at all for a Dual shape
# argument. Rebuilt here from the standard relation
#   F(z; df) = 1/2 + 1/2 * sign(z) * I_w(1/2, df/2),  w = z²/(z²+df),
# routed through cdf_ad_safe so the two shape arguments (1/2 fixed, df/2
# differentiated) and the evaluation point w (also df-dependent) each keep
# their own type instead of a shared one.
#
# Guarded at z == 0: there, w ≡ 0 for every df (so the true ∂F/∂df is 0), but
# ∂I_w/∂w diverges as w → 0 (the beta density blows up there since the first
# shape argument is 1/2 < 1) while ∂w/∂df → 0 with an explicit z² factor --
# the product is the correct limit 0, but IEEE Inf * 0.0 evaluates to NaN
# rather than taking that limit. z is a plain observation-derived constant in
# every case this package differentiates (never itself part of the
# differentiated parameter vector), so short-circuiting the whole expression
# at z == 0 costs nothing and sidesteps the singularity entirely. Also
# guarded at infinite z (reachable from `_crps_gtct_unit`'s point-mass
# bookkeeping, where a truncation bound can be unbounded while its mass is
# still nonzero): `w` would otherwise divide `Inf` by `Inf`.
@inline function _t_cdf(df::Real, z::Real)
    isinf(z) && return oftype(float(df) * one(z), z > 0 ? 1.0 : 0.0)
    z == 0 && return oftype(float(df) * one(z), 0.5)
    w = z^2 / (z^2 + df)
    return 0.5 + 0.5 * sign(z) * cdf_ad_safe(Beta(0.5, df / 2), w)
end

# `_Phi_t2(x, df)` is the analogue of `_Phi_root2` from normal.jl for the t
# family. The R source computes it as:
#   p  = pt(x, df)
#   pb = pbeta(df / (df + x²), df − 0.5, 0.5)
#   0.5 * (p ≤ 0.5 ? pb : 2 − pb)
# In Julia, R's `pbeta(x, a, b)` = `beta_inc(a, b, x)[1]`, replaced here by
# `cdf_ad_safe(Beta(a, b), x)` for the same reason as `_t_cdf` above: the
# shape argument `df - 0.5` is differentiated in `df`, and `beta_inc` cannot
# take a Dual there. Guarded the same way at `x == 0` (where the argument
# `df / (df + x²)` is identically 1 regardless of `df`, and the beta density
# there diverges since the *second* shape argument, fixed at 0.5, is < 1).
@inline function _Phi_t2(x::Real, df::Real)
    p = _t_cdf(df, x)
    pb = if x == 0
        oftype(float(df) * one(x), 1.0)
    else
        cdf_ad_safe(Beta(df - 0.5, 0.5), df / (df + x^2))
    end
    return 0.5 * (p <= 0.5 ? pb : 2 - pb)
end

# The constant `bfrac(df)` = 2√df/(df−1) · B(½, df−½) / B(½, df/2)²,
# shared by all four CRPS formulae once standardised.
@inline _t_bfrac(df::Real) = 2 * sqrt(df) / (df - 1) *
                             exp(logbeta(0.5, df - 0.5) - 2 * logbeta(0.5, 0.5 * df))

# --- standard t: _crps_t(y, df, location, scale) ------------------------------

"""
    _crps_t(y, df, location, scale)

CRPS for a location–scale Student-t forecast with `df` degrees of freedom,
`location` μ, and `scale` σ > 0. Returns `NaN` when `df ≤ 1`.
"""
function _crps_t(y::Real, df::Real, location::Real, scale::Real)
    scale < 0 && return oftype(float(y), NaN)
    scale == 0 && return abs(y - location)
    df <= 1 && return oftype(float(y), NaN)
    z = (y - location) / scale
    G_z = _t_G(z, df)
    out_z = z * (2 * _t_cdf(df, z) - 1) - 2 * G_z
    return scale * (out_z - _t_bfrac(df))
end

crps(d::TDist, y::Real) = _crps_t(y, dof(d), 0, 1)

# Location–scale t: `loc + scale*TDist(df)` is represented in Distributions.jl
# as `LocationScale{T,Continuous,TDist{T}}`.  The public alias is
# `Distributions.AffineDistribution` (defined as a `Union` type alias in
# Distributions ≥ 0.25; at runtime the concrete type is `LocationScale`).
function crps(d::Distributions.LocationScale{<:Real, Continuous, <:TDist}, y::Real)
    return _crps_t(y, dof(d.ρ), d.μ, d.σ)
end

# --- truncated t: _crps_tt(y, df, location, scale, lower, upper) -------------

function _crps_tt_unit(y::Real, df::Real, l::Real, u::Real)
    # Numerical stability: swap sign when lower > 3 (mirrors the R source).
    if l > 3
        y, l, u = -y, -u, -l
    end
    p_l = 0.0
    out_l = 0.0
    p_u = 1.0
    out_u = 1.0
    z = y
    if isfinite(l)
        p_l = _t_cdf(df, l)
        out_l = _Phi_t2(l, df)
        z = max(l, z)
    end
    if isfinite(u)
        p_u = _t_cdf(df, u)
        out_u = _Phi_t2(u, df)
        z = min(u, z)
    end
    l > u && return oftype(float(y), NaN)
    l == u && return abs(y - z)
    a = p_u - p_l
    b = out_u - out_l
    b == 0 && return oftype(float(y), NaN)
    G_z = _t_G(z, df)
    out_z = z * (2 * _t_cdf(df, z) - p_l - p_u) - 2 * G_z
    out = (out_z - b / a * _t_bfrac(df)) / a
    return out + abs(y - z)
end

function _crps_tt(y::Real, df::Real, location::Real, scale::Real, l::Real, u::Real)
    scale < 0 && return oftype(float(y), NaN)
    df <= 1 && return oftype(float(y), NaN)
    ys = y - location
    ls = isfinite(l) ? (l - location) / scale : l
    us = isfinite(u) ? (u - location) / scale : u
    if scale == 0
        return (ls < 0 && us > 0) ? abs(ys) : oftype(float(y), NaN)
    end
    return scale * _crps_tt_unit(ys / scale, df, ls, us)
end

function crps(d::Truncated{<:TDist}, y::Real)
    df = dof(d.untruncated)
    return _crps_tt(y, df, 0, 1, _lo(d.lower), _hi(d.upper))
end

# Location–scale truncated t: `truncated(loc + scale*TDist(df), lower, upper)`
# produces `Truncated{LocationScale{…,TDist{…}},…}`. There is no public
# named type for this combination; dispatch matches the concrete parametric
# type below.
function crps(d::Truncated{<:Distributions.LocationScale{<:Real, Continuous, <:TDist}}, y::Real)
    inner = d.untruncated
    return _crps_tt(y, dof(inner.ρ), inner.μ, inner.σ,
        _lo(d.lower), _hi(d.upper))
end

# --- censored t: _crps_ct(y, df, location, scale, lower, upper) --------------

function _crps_ct_unit(y::Real, df::Real, l::Real, u::Real)
    out_l1 = 0.0
    out_l2 = 0.0
    out_u1 = 0.0
    out_u2 = 1.0
    z = y
    if isfinite(l)
        p_l = _t_cdf(df, l)
        G_l = _t_G(l, df)
        out_l1 = -l * p_l^2 + 2 * G_l * p_l      # sign: G_l is negative
        out_l2 = _Phi_t2(l, df)
        z = max(l, z)
    end
    if isfinite(u)
        p_u = 1 - _t_cdf(df, u)
        G_u = _t_G(u, df)
        out_u1 = u * p_u^2 + 2 * G_u * p_u        # G_u is negative, so this subtracts
        out_u2 = _Phi_t2(u, df)
        z = min(u, z)
    end
    l > u && return oftype(float(y), NaN)
    l == u && return abs(y - z)
    b = out_u2 - out_l2
    G_z = _t_G(z, df)
    out_z = z * (2 * _t_cdf(df, z) - 1) - 2 * G_z
    out = out_z + out_l1 + out_u1 - b * _t_bfrac(df)
    return out + abs(y - z)
end

function _crps_ct(y::Real, df::Real, location::Real, scale::Real, l::Real, u::Real)
    scale < 0 && return oftype(float(y), NaN)
    df <= 1 && return oftype(float(y), NaN)
    ys = y - location
    ls = isfinite(l) ? (l - location) / scale : l
    us = isfinite(u) ? (u - location) / scale : u
    if scale == 0
        return (l <= u) ? abs(ys - max(ls, 0) - min(us, 0)) : oftype(float(y), NaN)
    end
    return scale * _crps_ct_unit(ys / scale, df, ls, us)
end

function crps(d::Distributions.Censored{<:TDist}, y::Real)
    df = dof(d.uncensored)
    return _crps_ct(y, df, 0, 1, _lo(d.lower), _hi(d.upper))
end

# Location–scale censored t: `censored(loc + scale*TDist(df), lower, upper)`
# gives `Censored{LocationScale{…,TDist{…}},…}` — again no public named type.
function crps(
        d::Distributions.Censored{<:Distributions.LocationScale{
            <:Real, Continuous, <:TDist}}, y::Real)
    inner = d.uncensored
    return _crps_ct(y, dof(inner.ρ), inner.μ, inner.σ,
        _lo(d.lower), _hi(d.upper))
end

# --- generalised truncated/censored t: _crps_gtct(..., lmass, umass) ---------

function _crps_gtct_unit(y::Real, df::Real, l::Real, u::Real,
        lmass::Real, umass::Real)
    # Sign-swap for numerical stability when lower > 3.
    if l > 3
        y, l, u = -y, -u, -l
        lmass, umass = umass, lmass
    end
    out_l1 = 0.0
    out_l2 = 0.0
    out_l3 = 0.0
    out_u1 = 0.0
    out_u2 = 0.0
    p_l = 0.0
    p_u = 1.0
    out_u3 = 1.0
    z = y
    if isfinite(l) || lmass != 0
        (lmass < 0 || lmass > 1) && return oftype(float(y), NaN)
        p_l = _t_cdf(df, l)
        G_l = isfinite(l) ? _t_G(l, df) : 0.0
        out_l1 = lmass == 0 ? 0.0 : l * lmass^2
        out_l2 = 2 * G_l * lmass
        out_l3 = _Phi_t2(l, df)
        z = max(l, z)
    end
    if isfinite(u) || umass != 0
        (umass < 0 || umass > 1) && return oftype(float(y), NaN)
        p_u = _t_cdf(df, u)
        G_u = isfinite(u) ? _t_G(u, df) : 0.0
        out_u1 = umass == 0 ? 0.0 : u * umass^2
        out_u2 = 2 * G_u * umass
        out_u3 = _Phi_t2(u, df)
        z = min(u, z)
    end
    l > u && return oftype(float(y), NaN)
    l == u && return abs(y - z)
    a1 = p_u - p_l
    a2 = 1 - (umass + lmass)
    (a2 < 0 || a2 > 1) && return oftype(float(y), NaN)
    b = out_u3 - out_l3
    b == 0 && return oftype(float(y), NaN)
    G_z = _t_G(z, df)
    out = out_u1 - out_l1 +
          (z * (2 * a2 * cdf(d, z) -
                (1 - 2 * lmass) * p_u -
                (1 - 2 * umass) * p_l) -
           (2 * G_z - out_u2 - out_l2 +
            a2 * b / a1 * _t_bfrac(df)) * a2) / a1
    return out + abs(y - z)
end

function _crps_gtct(y::Real, df::Real, location::Real, scale::Real,
        l::Real, u::Real, lmass::Real, umass::Real)
    scale < 0 && return oftype(float(y), NaN)
    df <= 1 && return oftype(float(y), NaN)
    ys = y - location
    ls = isfinite(l) ? (l - location) / scale : l
    us = isfinite(u) ? (u - location) / scale : u
    if scale == 0
        if ls < 0 && us > 0
            return (min(ys, 0) - ls) * lmass^2 - min(ys, 0) * (1 - lmass)^2 +
                   (us - max(ys, 0)) * umass^2 + max(ys, 0) * (1 - umass)^2
        end
        return oftype(float(y), NaN)
    end
    return scale * _crps_gtct_unit(ys / scale, df, ls, us, lmass, umass)
end
