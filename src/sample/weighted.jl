# Weighted ensemble scoring rules: threshold-weighted and outcome-weighted
# versions of CRPS, energy score, variogram score, and MMD score.
#
# Ported from R scoringRules v1.1.3 (scores_sample_univ_weighted.R and
# scores_sample_multiv_weighted.R; Jordan, Krüger, Lerch, Allen) under GPL-2.0-or-later.
#
# Reference
# ---------
# Allen, S. (2024): "Weighted scoringRules: Emphasising Particular Outcomes when
#   Evaluating Probabilistic Forecasts." Journal of Statistical Software 110(8).
#   doi:10.18637/jss.v110.i08

export twcrps, owcrps, twes, owes, twvs, owvs, twmmds, owmmds

# ---------------------------------------------------------------------------
# Chaining / weight-function convention for [a, b] intervals
# ---------------------------------------------------------------------------
#
# Univariate case
# ---------------
# Given an interval [a, b] ⊆ [-∞, ∞], the default functions are:
#
#   Chain function (for twcrps):  v(z) = min(max(z, a), b)   (= clamp(z, a, b))
#   Weight function (for owcrps): w(z) = 1{ a < z < b }       (strict inequalities)
#
# These match R's `twcrps_sample` (`pmin(pmax(x, a), b)`) and
# `owcrps_sample` (`as.numeric(x > a & x < b)`).
#
# Multivariate case
# -----------------
# Given vectors a, b ∈ ℝ^d (scalars are broadcast to length d):
#
#   Chain function: v(z) = clamp.(z, a, b)   (element-wise clamp)
#   Weight function: w(z) = 1{ ∀k: a[k] < z[k] < b[k] }   (all-dimensions)
#
# These match R's `pmin(pmax(x, a), b)` and `all(x > a & x < b)`.

# ---------------------------------------------------------------------------
# Univariate threshold-weighted CRPS  (twcrps_sample in R)
# ---------------------------------------------------------------------------

"""
    twcrps(dat, y; a=-Inf, b=Inf, chain_func=nothing)

Threshold-weighted CRPS of the ensemble `dat` (a vector of m simulation draws)
at observation `y`, emphasising outcomes in the interval (a, b).

The chaining function `v(z) = clamp(z, a, b)` is applied to both `y` and
every ensemble member; the standard EDF-based CRPS of the transformed forecast
is returned.  Alternatively, supply a custom vectorised `chain_func` (takes a
`Real`, returns a `Real`); supplying `chain_func` ignores `a` and `b`.

Lower is better.

# Arguments

  - `dat`: ensemble of simulation draws.
  - `y`: scalar observation.

# Keyword Arguments

  - `a`: lower threshold (default `-Inf`).
  - `b`: upper threshold (default `Inf`).
  - `chain_func`: custom chaining function; overrides `a` and `b` when supplied.

# Provenance

Ported from `twcrps_sample` in R scoringRules (scores_sample_univ_weighted.R;
Allen 2024, JSS 110(8)).

# Example

```@example
using ScoringRules
dat = randn(100)
twcrps(dat, 0.5; a = 0.0, b = 1.0)
```
"""
function twcrps(dat::AbstractVector, y::Real;
        a::Real = -Inf, b::Real = Inf,
        chain_func = nothing)
    if chain_func === nothing
        a < b || throw(ArgumentError("a must be strictly less than b, got a=$a, b=$b"))
        v = z -> clamp(z, a, b)
    else
        v = chain_func
    end
    v_y = v(y)
    v_dat = v.(dat)
    return _crps_edf_unweighted(v_y, v_dat)
end

# ---------------------------------------------------------------------------
# Univariate outcome-weighted CRPS  (owcrps_sample in R)
# ---------------------------------------------------------------------------

"""
    owcrps(dat, y; a=-Inf, b=Inf, weight_func=nothing)

Outcome-weighted CRPS of the ensemble `dat` at observation `y`, emphasising
outcomes in the interval (a, b).

Each ensemble member xᵢ receives weight w(xᵢ) = 1{a < xᵢ < b}; the
observation gets weight w(y) = 1{a < y < b}.  The weighted EDF-CRPS
(normalised by the sum of member weights) is then multiplied by w(y).

Alternatively, supply a custom vectorised `weight_func` (takes a `Real`,
returns a non-negative `Real`); supplying `weight_func` ignores `a` and `b`.

Returns `NaN` when all ensemble weights are zero (no member in the region).
Lower is better.

# Arguments

  - `dat`: ensemble of simulation draws.
  - `y`: scalar observation.

# Keyword Arguments

  - `a`: lower threshold (default `-Inf`).
  - `b`: upper threshold (default `Inf`).
  - `weight_func`: custom weight function; overrides `a` and `b` when supplied.

# Provenance

Ported from `owcrps_sample` in R scoringRules (scores_sample_univ_weighted.R;
Allen 2024, JSS 110(8)).

# Example

```@example
using ScoringRules
dat = randn(100)
owcrps(dat, 0.5; a = 0.0, b = 1.0)
```
"""
function owcrps(dat::AbstractVector, y::Real;
        a::Real = -Inf, b::Real = Inf,
        weight_func = nothing)
    if weight_func === nothing
        a < b || throw(ArgumentError("a must be strictly less than b, got a=$a, b=$b"))
        w_func = z -> Float64(a < z < b)
    else
        w_func = weight_func
    end
    w_y = w_func(y)
    w_dat = w_func.(dat)
    # _crps_edf_weighted normalises internally so only relative weights matter
    return _crps_edf_weighted(y, dat, w_dat) * w_y
end

# ---------------------------------------------------------------------------
# Internal helper: apply a multivariate chaining / weight function
# ---------------------------------------------------------------------------

# Clamp each component of a vector z to the corresponding [a_i, b_i] interval.
function _mv_chain_default(z::AbstractVector, a::AbstractVector, b::AbstractVector)
    return clamp.(z, a, b)
end

# Returns 1.0 if every component of z is strictly inside (a_i, b_i), else 0.0.
function _mv_weight_default(z::AbstractVector, a::AbstractVector, b::AbstractVector)
    return Float64(all(a .< z .< b))
end

# Broadcast a scalar bound to length d, or validate an existing vector.
function _broadcast_bound(v, d::Int, name::String)
    if v isa Real
        return fill(Float64(v), d)
    else
        length(v) == d || throw(DimensionMismatch(
            "bound $name has length $(length(v)) but y has length $d"))
        return Float64.(v)
    end
end

# ---------------------------------------------------------------------------
# Multivariate threshold-weighted energy score  (twes_sample in R)
# ---------------------------------------------------------------------------

"""
    twes(X, y; a=-Inf, b=Inf, chain_func=nothing)

Threshold-weighted energy score of the ensemble `X` (a `d × m` matrix, each
column one member) at the `d`-dimensional observation `y`.

The chaining function `v(z) = clamp.(z, a, b)` (element-wise) is applied to
`y` and every column of `X`; the standard energy score of the transformed
forecast is returned.  `a` and `b` may be scalars (broadcast to all dimensions)
or length-`d` vectors.  A custom `chain_func` (takes a length-`d` vector,
returns a length-`d` vector) overrides `a` and `b`.

Lower is better.

# Provenance

Ported from `twes_sample` in R scoringRules (scores_sample_multiv_weighted.R;
Allen 2024, JSS 110(8)).

# Example

```@example
using ScoringRules
X = randn(2, 50)
y = [0.0, 0.0]
twes(X, y; a = -1.0, b = 1.0)
```
"""
function twes(X::AbstractMatrix, y::AbstractVector;
        a = -Inf, b = Inf, chain_func = nothing)
    _check_multiv(X, y)
    d = length(y)
    if chain_func === nothing
        av = _broadcast_bound(a, d, "a")
        bv = _broadcast_bound(b, d, "b")
        all(av .<= bv) && any(av .< bv) ||
            throw(ArgumentError("each a[i] must be ≤ b[i] with at least one strict"))
        v = z -> _mv_chain_default(z, av, bv)
    else
        v = chain_func
    end
    v_y = v(y)
    v_dat = stack(v, eachcol(X))
    m = size(v_dat, 2)
    wv = fill(1.0 / m, m)
    return _esC_xy(v_y, v_dat, wv) - 0.5 * _esC_xx(v_dat, wv)
end

# ---------------------------------------------------------------------------
# Multivariate outcome-weighted energy score  (owes_sample in R)
# ---------------------------------------------------------------------------

"""
    owes(X, y; a=-Inf, b=Inf, weight_func=nothing)

Outcome-weighted energy score of the ensemble `X` (a `d × m` matrix) at the
`d`-dimensional observation `y`.

Each column Xᵢ of `X` receives weight w(Xᵢ) where the default weight function
is w(z) = 1{∀k: a[k] < z[k] < b[k]}.  The observation weight is w(y).  The
energy score is computed with the normalised member weights and then multiplied
by w(y).

Returns `NaN` when all member weights are zero.  A custom `weight_func` (takes
a length-`d` vector, returns a non-negative scalar) overrides `a` and `b`.
Lower is better.

# Provenance

Ported from `owes_sample` in R scoringRules (scores_sample_multiv_weighted.R;
Allen 2024, JSS 110(8)).

# Example

```@example
using ScoringRules
X = randn(2, 50)
y = [0.0, 0.0]
owes(X, y; a = -1.0, b = 1.0)
```
"""
function owes(X::AbstractMatrix, y::AbstractVector;
        a = -Inf, b = Inf, weight_func = nothing)
    _check_multiv(X, y)
    d = length(y)
    if weight_func === nothing
        av = _broadcast_bound(a, d, "a")
        bv = _broadcast_bound(b, d, "b")
        all(av .<= bv) && any(av .< bv) ||
            throw(ArgumentError("each a[i] must be ≤ b[i] with at least one strict"))
        wf = z -> _mv_weight_default(z, av, bv)
    else
        wf = weight_func
    end
    w_y = wf(y)
    w_dat = [wf(col) for col in eachcol(X)]
    sw = sum(w_dat)
    if sw == 0
        return NaN
    end
    wv = w_dat ./ sw
    return (_esC_xy(y, X, wv) - 0.5 * _esC_xx(X, wv)) * w_y
end

# ---------------------------------------------------------------------------
# Multivariate threshold-weighted variogram score  (twvs_sample in R)
# ---------------------------------------------------------------------------

"""
    twvs(X, y; p=0.5, a=-Inf, b=Inf, chain_func=nothing)

Threshold-weighted variogram score of order `p` of the ensemble `X` (a
`d × m` matrix) at the `d`-dimensional observation `y`.

The chaining function is applied to `y` and each column of `X`; the standard
variogram score is then evaluated on the transformed forecast.  See `twes` for
the conventions on `a`, `b`, and `chain_func`.  Lower is better.

# Provenance

Ported from `twvs_sample` in R scoringRules (scores_sample_multiv_weighted.R;
Allen 2024, JSS 110(8)).

# Example

```@example
using ScoringRules
X = randn(2, 50)
y = [0.0, 0.0]
twvs(X, y; a = -1.0, b = 1.0)
```
"""
function twvs(X::AbstractMatrix, y::AbstractVector;
        p::Real = 0.5, a = -Inf, b = Inf, chain_func = nothing)
    _check_multiv(X, y)
    d = length(y)
    if chain_func === nothing
        av = _broadcast_bound(a, d, "a")
        bv = _broadcast_bound(b, d, "b")
        all(av .<= bv) && any(av .< bv) ||
            throw(ArgumentError("each a[i] must be ≤ b[i] with at least one strict"))
        v = z -> _mv_chain_default(z, av, bv)
    else
        v = chain_func
    end
    v_y = v(y)
    v_dat = stack(v, eachcol(X))
    return _vsC(v_y, v_dat, p)
end

# ---------------------------------------------------------------------------
# Multivariate outcome-weighted variogram score  (owvs_sample in R)
# ---------------------------------------------------------------------------

"""
    owvs(X, y; p=0.5, a=-Inf, b=Inf, weight_func=nothing)

Outcome-weighted variogram score of order `p` of the ensemble `X` (a `d × m`
matrix) at the `d`-dimensional observation `y`.

The variogram score is computed using the normalised per-member weights w(Xᵢ),
then multiplied by w(y).  Returns `NaN` when all member weights are zero.  See
`owes` for conventions on `a`, `b`, and `weight_func`.  Lower is better.

# Provenance

Ported from `owvs_sample` in R scoringRules (scores_sample_multiv_weighted.R;
Allen 2024, JSS 110(8)).

# Example

```@example
using ScoringRules
X = randn(2, 50)
y = [0.0, 0.0]
owvs(X, y; a = -1.0, b = 1.0)
```
"""
function owvs(X::AbstractMatrix, y::AbstractVector;
        p::Real = 0.5, a = -Inf, b = Inf, weight_func = nothing)
    _check_multiv(X, y)
    d = length(y)
    if weight_func === nothing
        av = _broadcast_bound(a, d, "a")
        bv = _broadcast_bound(b, d, "b")
        all(av .<= bv) && any(av .< bv) ||
            throw(ArgumentError("each a[i] must be ≤ b[i] with at least one strict"))
        wf = z -> _mv_weight_default(z, av, bv)
    else
        wf = weight_func
    end
    w_y = wf(y)
    w_dat = [wf(col) for col in eachcol(X)]
    sw = sum(w_dat)
    if sw == 0
        return NaN
    end
    wv = w_dat ./ sw
    return _vsC_w(y, X, ones(d, d), wv, p) * w_y
end

# ---------------------------------------------------------------------------
# Multivariate threshold-weighted MMD score  (twmmds_sample in R)
# ---------------------------------------------------------------------------

"""
    twmmds(X, y; a=-Inf, b=Inf, chain_func=nothing)

Threshold-weighted MMD score (Gaussian kernel, σ = 1) of the ensemble `X` (a
`d × m` matrix) at the `d`-dimensional observation `y`.

The chaining function is applied to `y` and each column of `X`; the standard
MMD score is evaluated on the transformed forecast.  See `twes` for conventions
on `a`, `b`, and `chain_func`.  Lower is better.

# Provenance

Ported from `twmmds_sample` in R scoringRules (scores_sample_multiv_weighted.R;
Allen 2024, JSS 110(8)).

# Example

```@example
using ScoringRules
X = randn(2, 50)
y = [0.0, 0.0]
twmmds(X, y; a = -1.0, b = 1.0)
```
"""
function twmmds(X::AbstractMatrix, y::AbstractVector;
        a = -Inf, b = Inf, chain_func = nothing)
    _check_multiv(X, y)
    d = length(y)
    if chain_func === nothing
        av = _broadcast_bound(a, d, "a")
        bv = _broadcast_bound(b, d, "b")
        all(av .<= bv) && any(av .< bv) ||
            throw(ArgumentError("each a[i] must be ≤ b[i] with at least one strict"))
        v = z -> _mv_chain_default(z, av, bv)
    else
        v = chain_func
    end
    v_y = v(y)
    v_dat = stack(v, eachcol(X))
    m = size(v_dat, 2)
    wv = fill(1.0 / m, m)
    return 0.5 * _mmdsC_xx(v_dat, wv) - _mmdsC_xy(v_y, v_dat, wv)
end

# ---------------------------------------------------------------------------
# Multivariate outcome-weighted MMD score  (owmmds_sample in R)
# ---------------------------------------------------------------------------

"""
    owmmds(X, y; a=-Inf, b=Inf, weight_func=nothing)

Outcome-weighted MMD score (Gaussian kernel, σ = 1) of the ensemble `X` (a
`d × m` matrix) at the `d`-dimensional observation `y`.

The MMD score is computed using the normalised per-member weights w(Xᵢ), then
multiplied by w(y).  Returns `NaN` when all member weights are zero.  See
`owes` for conventions on `a`, `b`, and `weight_func`.  Lower is better.

# Provenance

Ported from `owmmds_sample` in R scoringRules (scores_sample_multiv_weighted.R;
Allen 2024, JSS 110(8)).

# Example

```@example
using ScoringRules
X = randn(2, 50)
y = [0.0, 0.0]
owmmds(X, y; a = -1.0, b = 1.0)
```
"""
function owmmds(X::AbstractMatrix, y::AbstractVector;
        a = -Inf, b = Inf, weight_func = nothing)
    _check_multiv(X, y)
    d = length(y)
    if weight_func === nothing
        av = _broadcast_bound(a, d, "a")
        bv = _broadcast_bound(b, d, "b")
        all(av .<= bv) && any(av .< bv) ||
            throw(ArgumentError("each a[i] must be ≤ b[i] with at least one strict"))
        wf = z -> _mv_weight_default(z, av, bv)
    else
        wf = weight_func
    end
    w_y = wf(y)
    w_dat = [wf(col) for col in eachcol(X)]
    sw = sum(w_dat)
    if sw == 0
        return NaN
    end
    wv = w_dat ./ sw
    return (0.5 * _mmdsC_xx(X, wv) - _mmdsC_xy(y, X, wv)) * w_y
end
