# Univariate ensemble (sample-based) scoring rules: CRPS, LogS, DSS.
# Ported from R scoringRules v1.1.3 (scores_sample_univ.R; mixn.cpp;
# Jordan, Krüger, Lerch, Allen) under GPL-2.0-or-later.
#
# Three scoring rules are exposed via the existing generics `crps`, `logs` and
# `dss` (all exported from the parent module). No new exports are needed here.
#
# References
# ----------
# Krueger, Lerch, Thorarinsdottir & Gneiting (2021): Predictive inference
#   based on Markov chain Monte Carlo output. International Statistical Review
#   89, 274–301. doi:10.1111/insr.12405
# Laio & Tamea (2007): Verification tools for probabilistic forecasts of
#   continuous hydrological variables. Hydrology and Earth System Sciences 11,
#   1267–1277.

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# `_auxcrps` and `_crps_mixnorm` (the Gaussian-mixture CRPS, from mixn.cpp) are
# defined in crps/mixture.jl, which loads before this file. The kernel-density
# CRPS below reuses `_crps_mixnorm`: a Gaussian KDE is an equally-weighted normal
# mixture with common bandwidth as the standard deviation.

# Validate a member-weight vector against its ensemble. R errors on missing,
# infinite and negative weights too; the positive-sum requirement is stricter
# than R, which returns NaN for an all-zero weight vector.
function _check_member_weights(dat::AbstractVector, w::AbstractVector)
    axes(dat) == axes(w) || throw(DimensionMismatch(
        "dat and w must have the same axes"))
    all(x -> isfinite(x) && x >= 0, w) ||
        throw(ArgumentError("weights w must be finite and non-negative"))
    sum(w) > 0 || throw(ArgumentError("weights w must have a positive sum"))
    return nothing
end

# Silverman's rule-of-thumb bandwidth, matching R's `bw.nrd`.
# bw.nrd(x) = 1.06 * min(sd(x), IQR(x)/1.34) * n^(-1/5)
# R's var() uses the (n − 1) denominator and quantile() uses type 7 (linear
# interpolation), which is also StatsBase's default — so `std` and `quantile`
# from Statistics/StatsBase match R directly.
function _bw_nrd(x::AbstractVector)
    n = length(x)
    n < 2 && error("need at least 2 data points for bandwidth estimation")
    s = std(x)                                   # n-1 denominator
    q25, q75 = quantile(x, 0.25), quantile(x, 0.75)
    h = (q75 - q25) / 1.34
    return 1.06 * min(s, h) * n^(-0.2)
end

# ---------------------------------------------------------------------------
# crps — EDF method (weighted empirical distribution)
# Translates `crps_edf` from scores_sample_univ.R.
# ---------------------------------------------------------------------------

# Unweighted path.
function _crps_edf_unweighted(y::Real, dat::AbstractVector)
    n = length(dat)
    c1n = 1.0 / n
    x = sort(dat)
    s = 0.0
    @inbounds for i in eachindex(x)
        a = (i - 0.5) * c1n       # (i − 0.5) / n, i.e. seq(0.5/n, 1−0.5/n)
        s += ((y < x[i]) - a) * (x[i] - y)
    end
    return 2 * c1n * s
end

# Weighted path.
# Weights are normalised so that Σw = 1 (or equivalently divided by P = Σw).
function _crps_edf_weighted(y::Real, dat::AbstractVector, w::AbstractVector)
    _check_member_weights(dat, w)

    ord = sortperm(dat)
    x = dat[ord]
    ww = float.(w[ord])
    p = cumsum(ww)
    P = p[end]

    s = 0.0
    @inbounds for i in eachindex(x)
        a = (p[i] - 0.5 * ww[i]) / P   # mid-point cumulative weight
        s += ww[i] * ((y < x[i]) - a) * (x[i] - y)
    end
    return 2 / P * s
end

# ---------------------------------------------------------------------------
# crps — KDE method (Gaussian kernel)
# Translates `crps_kdens` / `crpsmixnC` from scores_sample_univ.R and mixn.cpp.
# ---------------------------------------------------------------------------

function _crps_kde(y::Real, dat::AbstractVector, bw::Union{Real, Nothing})
    bw_val = bw === nothing ? _bw_nrd(dat) : Float64(bw)
    n = length(dat)
    ww = fill(1.0 / n, n)
    sw = fill(bw_val, n)
    return _crps_mixnorm(y, dat, sw, ww)
end

# ---------------------------------------------------------------------------
# Public methods on AbstractVector (ensemble forecasts)
# ---------------------------------------------------------------------------

"""
    crps(dat::AbstractVector{<:Real}, y::Real; method=:edf, w=nothing, bw=nothing)

CRPS of an ensemble forecast `dat` (a vector of `m` simulation draws) at
observation `y`.

Two approximation methods are available via `method`:

  - `:edf` (default) — empirical distribution function approximation using the
    quantile decomposition of Laio & Tamea (2007).  Optional finite, non-negative weights
    `w` (length `m`) are normalised to sum to one internally.

  - `:kde` — Gaussian kernel density estimate with bandwidth `bw`.  If `bw` is
    `nothing`, Silverman's rule-of-thumb is applied (matching R's `bw.nrd`).
    The `w` argument is ignored for this method.

Lower is better.

# Arguments

  - `dat`: ensemble of `m` simulation draws.
  - `y`: scalar observation.

# Keyword Arguments

  - `method`: `:edf` (default) or `:kde`.
  - `w`: optional finite, non-negative weight vector (length `m`) with a positive sum; only used for `:edf`.
  - `bw`: optional bandwidth; only used for `:kde`.

# Provenance

Ported from `crps_sample` / `crps_edf` / `crps_kdens` in R scoringRules
(scores_sample_univ.R; mixn.cpp; Jordan, Krüger, Lerch, Allen).

# Example

```@example
using ScoringRules
dat = randn(100)
crps(dat, 0.5)
```
"""
function crps(dat::AbstractVector{<:Real}, y::Real;
        method::Symbol = :edf, w = nothing, bw = nothing)
    if method === :edf
        if w === nothing
            return _crps_edf_unweighted(y, dat)
        else
            return _crps_edf_weighted(y, dat, w)
        end
    elseif method === :kde
        w === nothing ||
            @warn "the `w` argument is ignored for method = :kde" maxlog=1
        return _crps_kde(y, dat, bw)
    else
        throw(ArgumentError("method must be :edf or :kde, got :$method"))
    end
end

"""
    logs(dat::AbstractVector{<:Real}, y::Real; bw=nothing, w=nothing)

Logarithmic score of an ensemble forecast `dat` (a vector of `m` simulation
draws) at observation `y` using Gaussian kernel density estimation.

If `bw` is `nothing`, Silverman's rule-of-thumb bandwidth is used (matching
R's `bw.nrd`).  Optional finite, non-negative member weights `w` (length `m`) are
normalised to sum to one internally, so the KDE density becomes
``\\sum_i w_i \\varphi_{bw}(y - dat_i) / \\sum_i w_i``.  The rule-of-thumb
bandwidth is computed from `dat` alone and does not use `w`.  Lower is better.

# Arguments

  - `dat`: ensemble of `m` simulation draws.
  - `y`: scalar observation.

# Keyword Arguments

  - `bw`: optional bandwidth; defaults to Silverman's rule-of-thumb.
  - `w`: optional finite, non-negative member weight vector (length `m`) with a positive sum.

# Provenance

Ported from `logs_sample` / `lsmixnC` in R scoringRules (scores_sample_univ.R;
mixn.cpp; Jordan, Krüger, Lerch, Allen).  The `w` keyword extends the R
interface, which has no member weights for `logs_sample`.

# Example

```@example
using ScoringRules
dat = randn(100)
logs(dat, 0.5)
```
"""
function logs(dat::AbstractVector{<:Real}, y::Real; bw = nothing, w = nothing)
    w === nothing || _check_member_weights(dat, w)
    bw_val = bw === nothing ? _bw_nrd(dat) : Float64(bw)
    # KDE density at y: Σ_i wᵢ φ_{bw}(y − datᵢ) / Σ_i wᵢ
    # log score = −log density
    den = 0.0
    wsum = 0.0
    @inbounds for i in eachindex(dat)
        wi = w === nothing ? 1.0 : w[i]
        wsum += wi
        den += wi * _norm_pdf((y - dat[i]) / bw_val) / bw_val
    end
    return -log(den / wsum)
end

"""
    dss(dat::AbstractVector{<:Real}, y::Real; w=nothing)

Dawid–Sebastiani score of an ensemble forecast `dat` (a vector of `m`
simulation draws) at observation `y`:

```math
\\mathrm{DSS} = \\frac{(y - \\bar{x})^2}{s^2} + \\log s^2
```

where ``\\bar{x}`` is the sample mean and ``s^2 = \\tfrac{1}{n}\\sum_i(x_i - \\bar{x})^2``
is the **population** variance (R uses `mean(dat^2) - mean(dat)^2`, i.e. the
biased estimator). Optional finite, non-negative member weights `w` (length `m`) are
normalised to sum to one internally and replace the mean and variance with
their weighted versions. Lower is better.

# Arguments

  - `dat`: ensemble of `m` simulation draws.
  - `y`: scalar observation.

# Keyword Arguments

  - `w`: optional finite, non-negative member weight vector (length `m`) with a positive sum.

# Provenance

Ported from `dss_sample` / `dss_edf` in R scoringRules (scores_sample_univ.R;
Jordan, Krüger, Lerch, Allen), including the member-weight handling of
`dss_edf`.

# Example

```@example
using ScoringRules
dat = randn(100)
dss(dat, 0.5)
```
"""
function dss(dat::AbstractVector{<:Real}, y::Real; w = nothing)
    if w === nothing
        m = mean(dat)
        # Population variance: mean(dat.^2) - mean(dat).^2  (matches R dss_edf)
        v = mean(x^2 for x in dat) - m^2
    else
        _check_member_weights(dat, w)
        W = swx = swx2 = 0.0
        @inbounds for i in eachindex(dat)
            W += w[i]
            swx += w[i] * dat[i]
            swx2 += w[i] * dat[i]^2
        end
        m = swx / W
        v = swx2 / W - m^2
    end
    return (y - m)^2 / v + log(v)
end
