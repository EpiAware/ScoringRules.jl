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
    K  = length(p)
    Pk = zero(eltype(p))       # running cumulative forecast probability
    s  = zero(Float64)
    for k in 1:K
        Pk += p[k]
        Iy  = y <= k ? 1.0 : 0.0   # cumulative indicator: 1{y ≤ k}
        s  += (Pk - Iy)^2
    end
    return s
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    rps(p, y)

Ranked Probability Score (RPS; Epstein 1969) for a categorical / ordinal
forecast.

# Arguments
- `p` — `AbstractVector` of forecast probabilities over K ≥ 2 ordered outcome
        categories.  Must be non-negative and sum to 1.
- `y` — observed outcome category as an `Integer` in `{1, …, K}`, where
        `y = 1` is the smallest and `y = K` is the largest category.

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

# Examples
```julia
p = [0.3, 0.2, 0.5]
rps(p, 2)   # ≈ 0.34
rps(p, 1)   # ≈ 0.74
```
"""
function rps(p::AbstractVector, y::Integer)
    return _rps0(p, y)
end
