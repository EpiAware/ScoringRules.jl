# Generic definitions of the three univariate scores.
#
# `logs` and `dss` have closed forms that depend only on the log-density and on
# the first two moments, so a single generic method covers every
# `Distributions.jl` type. `crps` needs a distribution-specific closed form;
# the generic methods here provide a numerical fallback (quadrature for
# continuous distributions, summation for discrete ones) that the
# per-distribution methods in `crps/` override.
#
# All three follow the negative orientation used throughout the package: a
# *lower* score indicates a better forecast. Broadcast over vectors of
# forecasts and observations with the usual dot syntax, e.g. `crps.(ds, ys)`.

"""
    logs(d::UnivariateDistribution, y)

Logarithmic score of the forecast distribution `d` at the observation `y`,
equal to the negative log-likelihood `-logpdf(d, y)` (or `-logpmf` for discrete
`d`). Lower is better.

# Arguments

  - `d`: forecast distribution (any `UnivariateDistribution`).
  - `y`: scalar observation.

# Example

```@example
using Distributions, ScoringRules
logs(Normal(0, 1), 0.5)
```
"""
logs(d::UnivariateDistribution, y::Real) = -logpdf(d, y)

"""
    dss(d::UnivariateDistribution, y)

Dawid–Sebastiani score of the forecast distribution `d` at the observation `y`,

```math
\\mathrm{DSS}(F, y) = \\frac{(y - \\mu_F)^2}{\\sigma_F^2} + \\log \\sigma_F^2,
```

where ``\\mu_F`` and ``\\sigma_F^2`` are the mean and variance of `d`. Only the
first two moments of the forecast enter. Lower is better.

Returns `NaN` when the forecast variance is not finite and positive (for
example a Student-t with `df ≤ 2`, a GEV/GPD with `shape ≥ 1/2`, a
[`LogLaplace`](@ref) with `σ ≥ 1/2` or a log-logistic with `β ≤ 2`), matching
the `dss_*` functions in R scoringRules.

# Arguments

  - `d`: forecast distribution (any `UnivariateDistribution`).
  - `y`: scalar observation.

# Example

```@example
using Distributions, ScoringRules
dss(Normal(0, 1), 0.5)
```
"""
function dss(d::UnivariateDistribution, y::Real)
    v = _forecast_var(d)
    # A nonexistent variance means the DSS is undefined; return NaN, as the
    # dss_* functions in R scoringRules do.
    (isfinite(v) && v > 0) || return NaN
    m = mean(d)
    return (y - m)^2 / v + log(v)
end

# Forecast variance for `dss`, evaluating to `Inf`/`NaN` where the variance
# does not exist. `Distributions.var` already follows this convention for most
# families; `LogLogistic` instead throws, so its heavy-tailed case (infinite
# second moment for shape β ≤ 2) is guarded here.
_forecast_var(d::UnivariateDistribution) = var(d)
_forecast_var(d::LogLogistic) = d.β > 2 ? var(d) : oftype(float(d.β), Inf)

"""
    dss_moments(y, mean, var)

Dawid–Sebastiani score from a moment forecast given directly as its `mean` and
`var`iance, without constructing a distribution. Since the DSS depends on the
forecast only through its first two moments, this equals `dss(d, y)` for any `d`
with that mean and variance. This is the moment-based forecast input mode. Lower
is better.

# Arguments

  - `y`: scalar observation.
  - `mean`: forecast mean.
  - `var`: forecast variance.

# Example

```@example
using ScoringRules
dss_moments(0.5, 0.0, 1.0)
```
"""
dss_moments(y::Real, mean::Real, var::Real) = (y - mean)^2 / var + log(var)

"""
    ess_moments(y, mean, var, skew)

Error-spread score (Christensen, Moroz and Palmer 2015) from a moment forecast
given directly as its `mean`, `var`iance and `skew`ness,

```math
\\mathrm{ESS}(y) = \\bigl(\\sigma^2 - (\\mu - y)^2 - (\\mu - y)\\,\\sigma\\,\\gamma\\bigr)^2,
```

where ``\\mu``, ``\\sigma^2`` and ``\\gamma`` are the forecast mean, variance
and skewness. The score assesses whether the spread and skewness of an ensemble
forecast are consistent with its error. This is the moment-based forecast input
mode. Lower is better.

# Arguments

  - `y`: scalar observation.
  - `mean`: forecast mean.
  - `var`: forecast variance; must be non-negative (a negative value throws a
    `DomainError`, where R returns `NaN` with a warning).
  - `skew`: forecast skewness.

# Example

```@example
using ScoringRules
ess_moments(0.5, 0.0, 1.0, 0.5)
```
"""
function ess_moments(y::Real, mean::Real, var::Real, skew::Real)
    e = mean - y
    return (var - e^2 - e * sqrt(var) * skew)^2
end

"""
    crps(d::UnivariateDistribution, y)

Continuous ranked probability score of the forecast distribution `d` at the
observation `y`,

```math
\\mathrm{CRPS}(F, y) = \\int_{-\\infty}^{\\infty} \\bigl(F(x) - \\mathbf{1}\\{x \\ge y\\}\\bigr)^2 \\, dx .
```

Distribution-specific methods provide closed forms; this generic method is a
numerical fallback (adaptive quadrature for continuous `d`). Lower is better.

# Arguments

  - `d`: forecast distribution (any `UnivariateDistribution`).
  - `y`: scalar observation.

# Example

```@example
using Distributions, ScoringRules
crps(Normal(0, 1), 0.5)
```
"""
function crps(d::ContinuousUnivariateDistribution, y::Real)
    lo = minimum(d)
    hi = maximum(d)
    left, _ = quadgk(x -> cdf(d, x)^2, lo, y)
    right, _ = quadgk(x -> ccdf(d, x)^2, y, hi)
    return left + right
end

# Generic discrete fallback: CRPS = Σ_k (F(k) - 1{k ≥ y})^2 over the integer
# support, truncating an unbounded upper tail once the survival probability is
# negligible.
function crps(d::DiscreteUnivariateDistribution, y::Real)
    lo = minimum(d)
    hi = maximum(d)
    if !isfinite(hi)
        hi = ceil(Int, quantile(d, 1 - 1.0e-12)) + 1
    end
    lo = isfinite(lo) ? ceil(Int, lo) : floor(Int, quantile(d, 1.0e-12)) - 1
    s = 0.0
    for k in lo:hi
        Fk = cdf(d, k)
        s += (Fk - (k >= y ? 1.0 : 0.0))^2
    end
    return s
end
