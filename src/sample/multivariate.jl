# Multivariate ensemble scoring rules: energy score, variogram score, and
# maximum-mean-discrepancy score. Ported from R scoringRules v1.1.3
# (scores_sample_multiv.R; Jordan, Krüger, Lerch, Allen) with the inner kernels
# translated from the C++ source (procs_es.cpp).
#
# Convention: a d-dimensional ensemble of m members is represented as a d × m
# matrix X, where each COLUMN is one ensemble member — matching R's `dat` layout
# in `es_sample(y, dat)` where `dim(dat) == c(d, m)`.

export es, vs, mmds

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

"""
    _check_multiv(X, y)

Verify that `X` is a `d × m` matrix and `y` is a length-`d` vector.
"""
function _check_multiv(X::AbstractMatrix, y::AbstractVector)
    d = length(y)
    return size(X, 1) == d || throw(
        DimensionMismatch("rows of X ($(size(X, 1))) must equal length of y ($d)")
    )
end

"""
    _member_weights(m, w)

Return a normalised member-weight vector of length `m`. If `w` is `nothing`,
uniform weights `1/m` are used. Otherwise `w` is validated (length `m`,
non-negative) and rescaled so that its entries sum to one (analogous to
`w.helper.multiv` in R).
"""
function _member_weights(m::Integer, w)
    if w === nothing
        return fill(1.0 / m, m)
    end
    length(w) == m || throw(
        DimensionMismatch(
            "length of w ($(length(w))) must equal the number of ensemble members ($m)"
        )
    )
    any(<(0), w) && throw(ArgumentError("member weights w must be non-negative"))
    sw = sum(w)
    return w ./ sw
end

"""
    _w_helper(X, w)

Normalised member weights for the columns of the `d × m` ensemble matrix `X`
(see [`_member_weights`](@ref)).
"""
_w_helper(X::AbstractMatrix, w) = _member_weights(size(X, 2), w)

"""
    _check_w_vs(w_vs, d)

Validate a `d × d` pairwise weight matrix for the variogram score: square of
the right size, non-negative and symmetric (the checks R's `vs_sample` applies
to `w_vs`).
"""
function _check_w_vs(w_vs::AbstractMatrix, d::Integer)
    size(w_vs) == (d, d) || throw(
        DimensionMismatch(
            "w_vs must be a $d × $d matrix, got $(size(w_vs))"
        )
    )
    any(<(0), w_vs) && throw(ArgumentError("weight matrix w_vs must be non-negative"))
    return isapprox(w_vs, w_vs'; atol = 1.0e-12) || throw(
        ArgumentError(
            "weight matrix w_vs must be symmetric"
        )
    )
end

# ---------------------------------------------------------------------------
# Energy score
# ---------------------------------------------------------------------------

# "XY" part: Σ_i w_i ‖X_i − y‖
function _esC_xy(y::AbstractVector, X::AbstractMatrix, w::AbstractVector)
    out = 0.0
    @inbounds for (wi, xi) in zip(w, eachcol(X))
        out += wi * norm(xi .- y)
    end
    return out
end

# "XX" part: Σ_{i≤j} 2 w_i w_j ‖X_i − X_j‖
function _esC_xx(X::AbstractMatrix, w::AbstractVector)
    out = 0.0
    m = size(X, 2)
    @inbounds for i in 1:m
        @inbounds for j in (i + 1):m
            out += 2.0 * w[i] * w[j] * norm(view(X, :, i) .- view(X, :, j))
        end
    end
    return out
end

"""
    es(X, y; w=nothing)

Energy score of the ensemble forecast `X` (a `d × m` matrix, each column one
member) at the `d`-dimensional observation `y`:

```math
\\mathrm{ES} = \\sum_i w_i \\|X_i - y\\| - \\tfrac{1}{2} \\sum_{i,j} w_i w_j \\|X_i - X_j\\|
```

Optional per-member weights `w` (length `m`); they are normalised to sum to one
internally. If `w` is `nothing`, uniform weights are used. Lower is better.

# Provenance

Ported from `es_sample` in R scoringRules (scores_sample_multiv.R; procs_es.cpp;
Jordan, Krüger, Lerch, Allen). Gneiting et al. (2008), TEST 17, 211–235.

# Example

```@example
using ScoringRules
X = randn(2, 50)
y = [0.0, 0.0]
es(X, y)
```
"""
function es(X::AbstractMatrix, y::AbstractVector; w = nothing)
    _check_multiv(X, y)
    wv = _w_helper(X, w)
    return _esC_xy(y, X, wv) - 0.5 * _esC_xx(X, wv)
end

# ---------------------------------------------------------------------------
# Variogram score of order p
# ---------------------------------------------------------------------------

# No-weight path: Σ_{k≤l} 2 (|y_k − y_l|^p − mean_i |X_{k,i} − X_{l,i}|^p)^2
function _vsC(y::AbstractVector, X::AbstractMatrix, p::Real)
    out = 0.0
    d = length(y)
    @inbounds for k in 1:d
        @inbounds for l in k:d
            vdat = mean(abs(X[k, i] - X[l, i])^p for i in 1:size(X, 2))
            vy = abs(y[k] - y[l])^p
            out += 2.0 * (vy - vdat)^2
        end
    end
    return out
end

# w_vs-weighted path: Σ_{k≤l} 2 w_vs[k,l] (|y_k − y_l|^p − mean_i |X_{k,i} − X_{l,i}|^p)^2
function _vsC_w_vs(y::AbstractVector, X::AbstractMatrix, w_vs::AbstractMatrix, p::Real)
    out = 0.0
    d = length(y)
    @inbounds for k in 1:d
        @inbounds for l in k:d
            vdat = mean(abs(X[k, i] - X[l, i])^p for i in 1:size(X, 2))
            vy = abs(y[k] - y[l])^p
            out += 2.0 * w_vs[k, l] * (vy - vdat)^2
        end
    end
    return out
end

# Variogram kernel: Σ_{k≤l} 2 w_vs[k,l] (|x1_k − x1_l|^p − |x2_k − x2_l|^p)^2
function _vskernel(x1::AbstractVector, x2::AbstractVector, w_vs::AbstractMatrix, p::Real)
    out = 0.0
    d = length(x1)
    @inbounds for k in 1:d
        @inbounds for l in k:d
            vx1 = abs(x1[k] - x1[l])^p
            vx2 = abs(x2[k] - x2[l])^p
            out += 2.0 * w_vs[k, l] * (vx1 - vx2)^2
        end
    end
    return out
end

# Member-weight path (w per column, default w_vs = ones)
function _vsC_w(
        y::AbstractVector, X::AbstractMatrix, w_vs::AbstractMatrix,
        w::AbstractVector, p::Real
    )
    m = size(X, 2)
    s1 = 0.0
    @inbounds for i in 1:m
        s1 += w[i] * _vskernel(view(X, :, i), y, w_vs, p)
    end
    s2 = 0.0
    @inbounds for i in 1:m
        @inbounds for j in 1:m
            s2 += w[i] * w[j] * _vskernel(view(X, :, i), view(X, :, j), w_vs, p)
        end
    end
    return s1 - s2 / 2.0
end

"""
    vs(X, y; p=0.5, w=nothing, w_vs=nothing)

Variogram score of order `p` of the ensemble forecast `X` (a `d × m` matrix)
at the `d`-dimensional observation `y`:

```math
\\mathrm{VS}_p = \\sum_{k,l} w_{kl}
  \\bigl(|y_k - y_l|^p - \\overline{|X_{k,\\cdot} - X_{l,\\cdot}|^p}\\bigr)^2
```

`w` is an optional per-member weight vector (length `m`, as in `es`); it is
normalised to sum to one internally. `w_vs` is an optional `d × d` non-negative
symmetric pairwise weight matrix (R's `w_vs`; defaults to all ones). When `w`
is supplied the score uses the weighted-ensemble form of the variogram kernel.
R's `vs_sample` silently ignores `w_vs` when `w` is given; here both are
honoured together. Lower is better.

# Provenance

Ported from `vs_sample` in R scoringRules (scores_sample_multiv.R; procs_es.cpp;
Jordan, Krüger, Lerch, Allen). Scheuerer & Hamill (2015), MWR 143, 1321–1334.

# Example

```@example
using ScoringRules
X = randn(2, 50)
y = [0.0, 0.0]
vs(X, y)
```
"""
function vs(
        X::AbstractMatrix, y::AbstractVector;
        p::Real = 0.5, w = nothing, w_vs = nothing
    )
    _check_multiv(X, y)
    d = length(y)
    w_vs === nothing || _check_w_vs(w_vs, d)
    if w === nothing
        return w_vs === nothing ? _vsC(y, X, p) : _vsC_w_vs(y, X, w_vs, p)
    end
    wv = _w_helper(X, w)
    wvs = w_vs === nothing ? ones(d, d) : w_vs
    return _vsC_w(y, X, wvs, wv, p)
end

# ---------------------------------------------------------------------------
# MMD score (Gaussian kernel, σ = 1)
# ---------------------------------------------------------------------------

# "XY" part: Σ_i w_i exp(−½ ‖X_i − y‖²)
function _mmdsC_xy(y::AbstractVector, X::AbstractMatrix, w::AbstractVector)
    out = 0.0
    @inbounds for (wi, xi) in zip(w, eachcol(X))
        d2 = sum(abs2, xi .- y)
        out += wi * exp(-0.5 * d2)
    end
    return out
end

# "XX" part: Σ_i w_i² + Σ_{i<j} 2 w_i w_j exp(−½ ‖X_i − X_j‖²)
function _mmdsC_xx(X::AbstractMatrix, w::AbstractVector)
    out = 0.0
    m = size(X, 2)
    @inbounds for i in 1:m
        out += w[i]^2
        @inbounds for j in (i + 1):m
            d2 = sum(abs2, view(X, :, i) .- view(X, :, j))
            out += 2.0 * w[i] * w[j] * exp(-0.5 * d2)
        end
    end
    return out
end

"""
    mmds(X, y; w=nothing)

Maximum-mean-discrepancy score (Gaussian kernel with σ = 1) of the ensemble
forecast `X` (a `d × m` matrix) at the `d`-dimensional observation `y`:

```math
\\mathrm{MMDS} = \\tfrac{1}{2} \\sum_{i,j} w_i w_j k(X_i, X_j) - \\sum_i w_i k(X_i, y),
\\quad k(x,z) = \\exp(-\\tfrac{1}{2}\\|x-z\\|^2)
```

Optional per-member weights `w` (length `m`); they are normalised to sum to one
internally. If `w` is `nothing`, uniform weights are used. Lower is better.

# Provenance

Ported from `mmds_sample` in R scoringRules (scores_sample_multiv.R; procs_es.cpp;
Jordan, Krüger, Lerch, Allen). Gneiting & Raftery (2007), JASA 102, 359–378;
Gretton et al. (2012), JMLR 13, 723–773.

# Example

```@example
using ScoringRules
X = randn(2, 50)
y = [0.0, 0.0]
mmds(X, y)
```
"""
function mmds(X::AbstractMatrix, y::AbstractVector; w = nothing)
    _check_multiv(X, y)
    wv = _w_helper(X, w)
    return 0.5 * _mmdsC_xx(X, wv) - _mmdsC_xy(y, X, wv)
end
