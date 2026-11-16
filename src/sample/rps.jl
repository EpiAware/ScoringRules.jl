# Ranked Probability Score (RPS) for categorical / ordinal probability
# forecasts. Ported from R scoringRules `rps.R`
# (Jordan, Krüger, Lerch, Allen). GPL-2.0-or-later.
#
# References
# ----------
# Epstein, E. S. (1969): A scoring system for probability forecasts of ranked
#   categories, Journal of Applied Meteorology and Climatology 8, 985–987.
#
# Krueger, F. and Pavlova, L. (2024): Quantifying subjective uncertainty in
#   survey expectations, International Journal of Forecasting 40, 796–810.
#   https://doi.org/10.1016/j.ijforecast.2023.06.001

export rps

# ---------------------------------------------------------------------------
# Internal core
# ---------------------------------------------------------------------------

"""
    _rps0(p, y)

Core RPS computation for a single observation `y` and probability vector `p`
over K categories, where `y ∈ {1, …, K}`.  Mirrors `rps0` in `rps.R`.

The RPS is the sum of squared differences between cumulative forecast
probabilities and cumulative empirical (indicator) probabilities:

    RPS = Σ_{k=1}^{K} (P_k - 1{y ≤ k})²

where `P_k = Σ_{j=1}^{k} p_j` is the cumulative forecast probability through
category k.  This is equivalent to a sum of K Brier scores, one per category
boundary.
"""
function _rps0(p::AbstractVector, y::Integer)
    K = length(p)
    Pk = zero(eltype(p))       # running cumulative forecast probability
    s = zero(Float64)
    for k in 1:K
        Pk += p[k]
        Iy = y <= k ? 1.0 : 0.0   # cumulative indicator: 1{y ≤ k}
        s += (Pk - Iy)^2
    end
    return s
end

# The sum of `p` for the check against one. R accumulates its sum in extended
# precision, and a normalised vector then sums to one within `eps`; a plain
# `Float64` sum can land a few `eps` away. Neumaier's compensated sum is at
# least as accurate as R's, and the two decide alike except for vectors of
# thousands of categories whose exact sum lies within half an `eps` of `tol`.
function _accurate_sum(p::AbstractVector)
    s = float(zero(first(p)))
    c = zero(s)
    for x in p
        t = s + x
        c += abs(s) >= abs(x) ? (s - t) + x : (x - t) + s
        s = t
    end
    return s + c
end

# R's default `tol` is one `eps` of its double precision. A `Float32` vector
# cannot sum to one within that, and a narrower type gets its own `eps`; a
# wider type keeps R's, which values widened from `Float64` meet.
_rps_default_tol(::AbstractVector{T}) where {T <: AbstractFloat} = max(eps(T), eps(Float64))
_rps_default_tol(::AbstractVector) = eps(Float64)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    rps(p, y; tol = max(eps(eltype(p)), eps(Float64)))

Ranked Probability Score (RPS; Epstein 1969) for a categorical / ordinal
forecast.

# Arguments
- `p` — `AbstractVector` of forecast probabilities over K ≥ 2 ordered outcome
        categories.  Must be non-negative and sum to 1.
- `y` — observed outcome category as an `Integer` in `{1, …, K}`, where
        `y = 1` is the smallest and `y = K` is the largest category.
- `tol` — how far the sum of `p` may lie from 1, as in R's `rps_probs`;
        by default one `eps` of `Float64`, as in R, or the element type's own
        `eps` where that is larger, as for `Float32`.
        Negative probabilities, a sum further from 1 and fewer than two
        categories throw an `ArgumentError`, as R stops on them.

# Returns
A scalar `Float64` score.  Lower values indicate better forecasts.

The RPS is defined as

    RPS = Σ_{k=1}^{K} (P_k - 1{y ≤ k})²

where `P_k = Σ_{j=1}^{k} p_j` is the cumulative forecast probability through
category k.  This sums K Brier scores across all category boundaries,
reflecting the ordinal structure of the outcome.

The convention matches `rps_probs` in `rps.R` (Jordan, Krüger, Lerch, Allen):
outcome `y` is 1-indexed and the sum runs over all K categories (the last term
is always zero when probabilities sum to 1, but is retained for consistency
with the R source).

# Example

```@example
using ScoringRules
p = [0.3, 0.2, 0.5]
rps(p, 2)
```
"""
function rps(p::AbstractVector, y::Integer; tol::Real = _rps_default_tol(p))
    length(p) >= 2 ||
        throw(ArgumentError("need at least 2 outcome categories, got $(length(p))"))
    tol >= 0 || throw(ArgumentError("tol must be non-negative, got $tol"))
    any(<(0), p) && throw(ArgumentError("probabilities in p must be non-negative"))
    total = _accurate_sum(p)
    abs(total - 1) <= tol || throw(
        ArgumentError("probabilities in p must sum to one within tol = $tol, got $total")
    )
    1 <= y <= length(p) ||
        throw(ArgumentError("observed category y must satisfy 1 ≤ y ≤ length(p) = $(length(p)), got $y"))
    return _rps0(p, y)
end
