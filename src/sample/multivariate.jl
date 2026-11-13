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

Verify that `X` is a `d × m` matrix with at least one member and `y` is a
length-`d` vector.
"""
function _check_multiv(X::AbstractMatrix, y::AbstractVector)
    d = length(y)
    size(X, 1) == d || throw(
        DimensionMismatch("rows of X ($(size(X, 1))) must equal length of y ($d)")
    )
    size(X, 2) >= 1 ||
        throw(ArgumentError("the ensemble X must have at least one member"))
    # Every loop below indexes `X`, `y`, the member weights and `w_vs` over
    # one-based ranges, so offset axes on any of them read past the end of
    # another. `@inbounds` makes that a silent read of whatever is next in
    # memory rather than an error, which is why the wrong answer it produced
    # was different on each call. `w` and `w_vs` are checked where they are
    # validated, since they do not reach here.
    # `Base.require_one_based_indexing` would say this in one call but is not
    # public API, and this package has just finished removing its reliance on
    # an unexported one.
    (_one_based(X) && _one_based(y)) || throw(
        ArgumentError(
            "X and y must have one-based axes, got $(axes(X)) and $(axes(y))"
        )
    )
    return nothing
end

_one_based(A) = all(ax -> ax == 1:length(ax), axes(A))

# Euclidean distance between two columns of the ensemble, and between a column
# and the observation, accumulated in place. Writing these out keeps the
# `m(m−1)/2` inner products of `es` and `mmds` free of the temporary that
# `norm(view(X, :, i) .- view(X, :, j))` allocates on every pair: at d = 50,
# m = 3000 that temporary accounted for 2061 MiB per call, and removing it cuts
# the call from 594 ms to 127 ms.
# Each element is converted before the subtraction, not just the accumulator:
# an integer difference squares to an integer and overflows there, which
# `norm` (the call this replaced) never did.
# `es(Int32[2147483647 0; 0 0], Int32[0, 0])` came out as 0.25 against the
# true 536870911.75, and at `Int64[4000000000 0; 0 0]` the wrapped negative
# square threw out of `sqrt`.
@inline function _col_sqdist(X::AbstractMatrix, i::Integer, j::Integer)
    T = _float_type(view(X, :, i), view(X, :, j))
    s = zero(T)
    @inbounds @simd for k in axes(X, 1)
        s += abs2(T(X[k, i]) - T(X[k, j]))
    end
    return s
end

@inline function _col_sqdist(X::AbstractMatrix, i::Integer, y::AbstractVector)
    T = _float_type(view(X, :, i), y)
    s = zero(T)
    @inbounds @simd for k in axes(X, 1)
        s += abs2(T(X[k, i]) - T(y[k]))
    end
    return s
end

# The Euclidean distance from its square. Two coincident columns put the
# square at exactly zero, where the derivative of `sqrt` is `0 * Inf`, so
# automatic differentiation would return NaN for any score with a tied pair.
# The distance has a kink there and zero is its subgradient, which is also
# what `norm` gave. Only an exact zero is special: a NaN square, from a
# missing member or observation, has to stay NaN.
@inline _sqrt_dist(s::Real) = iszero(s) ? zero(s) : sqrt(s)

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
    # `w ./ sw` keeps `w`'s axes, and the score loops index it over `1:m`.
    _one_based(w) || throw(
        ArgumentError("member weights w must have one-based axes, got $(axes(w))")
    )
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
    _one_based(w_vs) || throw(
        ArgumentError("w_vs must have one-based axes, got $(axes(w_vs))")
    )
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
    @inbounds for i in axes(X, 2)
        out += w[i] * _sqrt_dist(_col_sqdist(X, i, y))
    end
    return out
end

# "XX" part: Σ_{i≤j} 2 w_i w_j ‖X_i − X_j‖
function _esC_xx(X::AbstractMatrix, w::AbstractVector)
    out = 0.0
    m = size(X, 2)
    @inbounds for i in 1:m
        wi = w[i]
        acc = 0.0
        for j in (i + 1):m
            acc += w[j] * _sqrt_dist(_col_sqdist(X, i, j))
        end
        out += 2.0 * wi * acc
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

# Variogram score, one pass over the ensemble.
#
#   VS_p = Σ_{k≤l} 2 w_kl (|y_k − y_l|^p − Σ_i w_i |X_{k,i} − X_{l,i}|^p)²
#
# `w_vs` and `w` may each be `nothing`, meaning all-ones pairs and uniform
# members. The member-weighted case used to be written as the kernel score
#
#   Σ_i w_i κ(X_i, y) − ½ Σ_{i,j} w_i w_j κ(X_i, X_j),
#
# which is the same number: expanding the double sum gives
# `2(E_w[V²] − E_w[V]²)` for each pair (k, l), and that cancels the variance
# term out of the first sum. It costs m² kernel evaluations instead of m, so at
# d = 10, m = 2000 the kernel form took 14.7 s against 3.3 ms here.
function _vs_core(
        y::AbstractVector, X::AbstractMatrix, w_vs, w, p::Real
    )
    out = 0.0
    d = length(y)
    m = size(X, 2)
    # Floating point before the subtraction, for the reason `_col_sqdist`
    # gives: an integer difference raised to `p` wraps, and `abs` of `typemin`
    # stays negative, so `vs` returned 5.9e8 for an ensemble scoring 8.0e9 and
    # threw out of `^0.5` on the wrapped case.
    T = _float_type(X, y)
    @inbounds for k in 1:d
        for l in k:d
            wkl = w_vs === nothing ? 1.0 : w_vs[k, l]
            wkl == 0 && continue
            vdat = 0.0
            if w === nothing
                for i in 1:m
                    vdat += abs(T(X[k, i]) - T(X[l, i]))^p
                end
                vdat /= m
            else
                for i in 1:m
                    vdat += w[i] * abs(T(X[k, i]) - T(X[l, i]))^p
                end
            end
            vy = abs(T(y[k]) - T(y[l]))^p
            out += 2.0 * wkl * (vy - vdat)^2
        end
    end
    return out
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
    w_vs === nothing || _check_w_vs(w_vs, length(y))
    wv = w === nothing ? nothing : _w_helper(X, w)
    return _vs_core(y, X, w_vs, wv, p)
end

# ---------------------------------------------------------------------------
# MMD score (Gaussian kernel, σ = 1)
# ---------------------------------------------------------------------------

# "XY" part: Σ_i w_i exp(−½ ‖X_i − y‖²)
function _mmdsC_xy(y::AbstractVector, X::AbstractMatrix, w::AbstractVector)
    out = 0.0
    @inbounds for i in axes(X, 2)
        out += w[i] * exp(-0.5 * _col_sqdist(X, i, y))
    end
    return out
end

# "XX" part: Σ_i w_i² + Σ_{i<j} 2 w_i w_j exp(−½ ‖X_i − X_j‖²)
function _mmdsC_xx(X::AbstractMatrix, w::AbstractVector)
    out = 0.0
    m = size(X, 2)
    @inbounds for i in 1:m
        out += w[i]^2
        acc = 0.0
        for j in (i + 1):m
            acc += w[j] * exp(-0.5 * _col_sqdist(X, i, j))
        end
        out += 2.0 * w[i] * acc
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
