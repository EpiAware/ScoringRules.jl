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
"""
function dss(d::UnivariateDistribution, y::Real)
    m = mean(d)
    v = var(d)
    return (y - m)^2 / v + log(v)
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
        hi = ceil(Int, quantile(d, 1 - 1e-12)) + 1
    end
    lo = isfinite(lo) ? ceil(Int, lo) : floor(Int, quantile(d, 1e-12)) - 1
    s = 0.0
    for k in lo:hi
        Fk = cdf(d, k)
        s += (Fk - (k >= y ? 1.0 : 0.0))^2
    end
    return s
end
