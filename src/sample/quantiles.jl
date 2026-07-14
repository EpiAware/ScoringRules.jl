# Quantile score and interval score for ensemble and quantile forecasts.
# Ported from R scoringRules `scores_quantiles.R`
# (Jordan, Krüger, Lerch, Allen). GPL-2.0-or-later.
#
# References
# ----------
# Koenker, R. and Bassett, G. (1978): Regression quantiles, Econometrica 46,
#   33–50. https://doi.org/10.2307/1913643
#
# Gneiting, T. and Raftery, A. E. (2007): Strictly proper scoring rules,
#   prediction and estimation, Journal of the American Statistical Association
#   102, 359–378. https://doi.org/10.1198/016214506000001437

export quantile_score, interval_score

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

"""
    _qs_quantile(y, x, alpha)

Pinball loss at a single quantile level `alpha` with quantile forecast `x` and
observation `y`.  Directly mirrors `qs_quantiles` in `scores_quantiles.R`:

    score = ((y < x) - alpha) * (x - y)

Equivalently:
  - if y < x:  (1 - alpha) * (x - y)   [over-forecast]
  - if y >= x: alpha * (y - x)          [under-forecast]

This is the negatively-oriented (lower = better) formulation of the check
function / pinball loss.
"""
@inline function _qs_quantile(y::Real, x::Real, alpha::Real)
    return (Float64(y < x) - alpha) * (x - y)
end

"""
    _ints_quantile(y, x_lower, x_upper, target_coverage)

Interval score at a single lower/upper quantile pair and nominal coverage
`target_coverage` (e.g. 0.8 for an 80 % prediction interval).  Directly
mirrors `ints_quantiles` in `scores_quantiles.R`:

    alpha1 = 0.5 * (1 - target_coverage)
    alpha2 = 0.5 * (1 + target_coverage)
    score  = (2 / (1 - target_coverage)) * (qs(y, x_lower, alpha1) +
                                             qs(y, x_upper, alpha2))

Expanded this gives:
    (x_upper - x_lower)
      + (2/(1-level)) * (x_lower - y) * 1{y < x_lower}
      + (2/(1-level)) * (y - x_upper) * 1{y > x_upper}

The convention follows *nominal coverage* (not alpha = 1 - coverage).
"""
@inline function _ints_quantile(y::Real, x_lower::Real, x_upper::Real,
                                target_coverage::Real)
    alpha1 = 0.5 * (1 - target_coverage)
    alpha2 = 0.5 * (1 + target_coverage)
    scale  = 2 / (1 - target_coverage)
    return scale * (_qs_quantile(y, x_lower, alpha1) +
                    _qs_quantile(y, x_upper, alpha2))
end

# ---------------------------------------------------------------------------
# Public API — quantile-based entry points
# ---------------------------------------------------------------------------

"""
    quantile_score(q_levels, q_forecasts, y)

Quantile score (pinball loss) for a vector of quantile levels and the
corresponding quantile forecasts, evaluated at observation `y`.

# Arguments
- `q_levels`    — `AbstractVector` of quantile levels α ∈ (0, 1).
- `q_forecasts` — `AbstractVector` of the corresponding quantile forecast
                  values; must have the same length as `q_levels`.
- `y`           — scalar observation.

# Returns
A `Vector{Float64}` of per-level scores, one entry per element of `q_levels`.
To obtain a single summary value use `mean(quantile_score(...))`.

The score at level α with forecast q and observation y is

    score_α = ((y < q) - α) * (q - y)

which equals `(1 - α)(q - y)` when `y < q` and `α(y - q)` when `y ≥ q`
(Koenker & Bassett 1978).  Lower values indicate better forecasts.

# Examples
```julia
levels = [0.1, 0.5, 0.9]
q      = [-1.28, 0.0, 1.28]
quantile_score(levels, q, 1.0)
```
"""
function quantile_score(q_levels::AbstractVector, q_forecasts::AbstractVector,
                        y::Real)
    length(q_levels) == length(q_forecasts) ||
        throw(DimensionMismatch(
            "q_levels and q_forecasts must have the same length"))
    return [_qs_quantile(y, q_forecasts[i], q_levels[i])
            for i in eachindex(q_levels)]
end

# ---------------------------------------------------------------------------
# Public API — ensemble (sample) entry points
# ---------------------------------------------------------------------------

"""
    quantile_score(dat, y; alpha, type=7)

Quantile score for a single quantile level `alpha`, where the quantile is
estimated from an ensemble `dat`.  Mirrors `qs_sample` in
`scores_quantiles.R`.

# Arguments
- `dat`  — `AbstractVector` of ensemble draws.
- `y`    — scalar observation.
- `alpha` — quantile level α ∈ (0, 1).
- `type`  — quantile interpolation type (1–9), passed to `Statistics.quantile`;
            default 7 matches R's default.

# Returns
A scalar `Float64` score.

The empirical α-quantile `q̂` is computed from `dat`, and the score is

    score = ((y < q̂) - α) * (q̂ - y)
"""
function quantile_score(dat::AbstractVector, y::Real;
                        alpha::Real, type::Int=7)
    q_hat = Statistics.quantile(dat, alpha)   # type-7 by default in Julia
    return _qs_quantile(y, q_hat, alpha)
end

# ---------------------------------------------------------------------------
# Public API — interval score, quantile-based entry point
# ---------------------------------------------------------------------------

"""
    interval_score(lower, upper, y, level)

Interval score (Gneiting & Raftery 2007) for a central prediction interval at
nominal coverage `level` (e.g. `level = 0.8` for an 80 % interval), with
lower endpoint `lower`, upper endpoint `upper`, and observation `y`.

# Arguments
- `lower` — lower quantile forecast (the α₁ = (1-level)/2 quantile).
- `upper` — upper quantile forecast (the α₂ = 1 - (1-level)/2 quantile).
- `y`     — scalar observation.
- `level` — nominal coverage, `0 < level < 1`.

# Returns
A scalar `Float64` score.

The score is

    IS = (upper - lower)
           + (2 / (1-level)) * (lower - y) * 1{y < lower}
           + (2 / (1-level)) * (y - upper) * 1{y > upper}

This convention uses *coverage* (not alpha = 1 - coverage), matching
`ints_quantiles` in `scores_quantiles.R`.  Lower values indicate better
forecasts.

# Examples
```julia
interval_score(-1.64, 1.64, 0.5, 0.9)
```
"""
function interval_score(lower::Real, upper::Real, y::Real, level::Real)
    return _ints_quantile(y, lower, upper, level)
end

"""
    interval_score(dat, y; level, type=7)

Interval score estimated from an ensemble `dat`, at nominal coverage `level`.
The α₁ and α₂ quantiles are computed from `dat` with interpolation type `type`
(default 7, matching R), then `interval_score(lower, upper, y, level)` is
called.  Mirrors `ints_sample` in `scores_quantiles.R`.

# Arguments
- `dat`   — `AbstractVector` of ensemble draws.
- `y`     — scalar observation.
- `level` — nominal coverage, `0 < level < 1`.
- `type`  — quantile interpolation type (1–9); default 7.

# Returns
A scalar `Float64` score.
"""
function interval_score(dat::AbstractVector, y::Real;
                        level::Real, type::Int=7)
    alpha1 = 0.5 * (1 - level)
    alpha2 = 1 - alpha1
    qs     = Statistics.quantile(dat, [alpha1, alpha2])
    return _ints_quantile(y, qs[1], qs[2], level)
end
