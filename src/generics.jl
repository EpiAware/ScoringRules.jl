# Generic definitions of the three univariate scores.
#
# `logs` needs only the log-density, so the single generic method here covers
# every `Distributions.jl` type. `dss` needs the first two moments, which some
# wrapper types (truncated, censored, `BoundaryMass`) do not supply
# analytically for every base; `_forecast_moments` falls back to quadrature
# there. `crps` needs a distribution-specific closed form; the generic methods
# here provide a numerical fallback (quadrature for continuous distributions,
# exact integration of the step CDF for discrete ones) that the
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
    m, v = _forecast_moments(d)
    # A nonexistent variance means the DSS is undefined; return NaN, as the
    # dss_* functions in R scoringRules do.
    (isfinite(v) && v > 0) || return NaN
    return (y - m)^2 / v + log(v)
end

# Mean and variance of the forecast, as `dss` needs them. The variance
# evaluates to `Inf`/`NaN` where it does not exist; `Distributions.var` already
# follows that convention for most families, and the exceptions are handled
# below.
_forecast_moments(d::UnivariateDistribution) = (mean(d), _forecast_var(d))

_forecast_var(d::UnivariateDistribution) = var(d)

# `LogLogistic` throws rather than returning `Inf` once its second moment stops
# existing (shape β ≤ 2).
_forecast_var(d::LogLogistic) = d.β > 2 ? var(d) : oftype(float(d.β), Inf)

# --- GEV moments near the Gumbel limit -------------------------------------
#
# `mean` and `var` of a GEV are built from `Γ(1−ξ)` and `Γ(1−2ξ)`, and both
# combinations cancel as ξ → 0: `var` evaluates `(Γ(1−2ξ) − Γ(1−ξ)²)/ξ²`,
# whose numerator is O(ξ²). At ξ = 10⁻⁸ the variance that
# `Distributions.var` returns is out by a third, and `dss` with it (18.5
# against 0.81). Replacing `lgamma` differences for the cancelling parts does
# not help either, because `lgamma(1−ξ)` is itself O(ξ) and carries an
# absolute, not relative, error.
#
# Both are instead taken from the Maclaurin series of `log Γ(1−ξ)`,
#
#   log Γ(1−ξ) = γξ + Σ_{k≥2} ζ(k) ξ^k / k,
#
# which converges for |ξ| < 1 and whose coefficients are exact. Factoring the
# leading ξ (for the mean) or ξ² (for the variance) out of the series before
# dividing removes the cancellation completely: the result is accurate to the
# last bit or two for every |ξ| ≤ `_GEV_SERIES_ATOL`, the Gumbel limit falls
# out of the ξ = 0 case without a special branch, and above the cutoff the
# plain gamma expressions have nothing left to cancel.
const _ZETA_SERIES = ntuple(k -> zeta(k + 1), 30)   # ζ(2) … ζ(31)
const _EULER_GAMMA = -digamma(1.0)
const _GEV_SERIES_ATOL = 0.1

# `expm1(q)/q`, continued to its limit 1 at q = 0.
@inline _expm1_over(q::Real) = iszero(q) ? one(float(q)) : expm1(q) / q

# `log Γ(1−ξ) / ξ`  =  γ + Σ_{k≥2} ζ(k) ξ^{k−1} / k.
function _lgamma1m_over(ξ::Real)
    acc = oftype(float(ξ), _EULER_GAMMA)
    pw = float(ξ)
    for k in 2:(length(_ZETA_SERIES) + 1)
        t = _ZETA_SERIES[k - 1] / k * pw
        acc += t
        abs(t) <= eps(abs(acc)) && break
        pw *= ξ
    end
    return acc
end

# `(log Γ(1−2ξ) − 2 log Γ(1−ξ)) / ξ²`  =  Σ_{k≥2} ζ(k)(2^k − 2) ξ^{k−2} / k.
# The γξ terms cancel exactly, which is why the sum starts at k = 2.
function _lgamma_gap_over(ξ::Real)
    acc = zero(float(ξ))
    pw = one(float(ξ))
    for k in 2:(length(_ZETA_SERIES) + 1)
        t = _ZETA_SERIES[k - 1] * (exp2(k) - 2) / k * pw
        acc += t
        abs(t) <= eps(abs(acc)) && break
        pw *= ξ
    end
    return acc
end

function _forecast_moments(d::GeneralizedExtremeValue)
    ξ = d.ξ
    T = typeof(float(ξ * d.σ * d.μ))
    abs(ξ) > _GEV_SERIES_ATOL && return (T(mean(d)), T(_forecast_var(d)))
    m = _lgamma1m_over(ξ)
    v = _lgamma_gap_over(ξ)
    g1 = gamma(1 - ξ)
    return (
        T(d.μ + d.σ * m * _expm1_over(ξ * m)),
        T(d.σ^2 * g1^2 * v * _expm1_over(ξ^2 * v)),
    )
end

function _forecast_var(d::GeneralizedExtremeValue)
    ξ = d.ξ
    T = typeof(float(ξ * d.σ))
    ξ >= 1 // 2 && return T(Inf)
    abs(ξ) <= _GEV_SERIES_ATOL && return last(_forecast_moments(d))
    return T(var(d))
end

# --- moments of the wrapper types ------------------------------------------
#
# `Distributions.jl` gives truncated and censored distributions analytic
# moments only for a handful of bases (the normal and the uniform among them);
# for the rest `var` has no applicable method and throws. Since `dss` is
# documented to work for any univariate forecast, the moments are then taken
# numerically. The analytic route is still preferred where it exists, so no
# result that works today moves.

# Raw first and second moments of a continuous density over `[lo, hi]`, by
# adaptive quadrature. `quadgk` handles the infinite endpoints directly.
function _raw_moments(f, lo::Real, hi::Real)
    m1, _ = quadgk(x -> x * f(x), lo, hi)
    m2, _ = quadgk(x -> x^2 * f(x), lo, hi)
    return m1, m2
end

function _numeric_moments(d::Truncated{<:ContinuousUnivariateDistribution})
    base = d.untruncated
    lo = d.lower === nothing ? minimum(base) : float(d.lower)
    hi = d.upper === nothing ? maximum(base) : float(d.upper)
    z = cdf(base, hi) - cdf(base, lo)
    z > 0 || return (oftype(float(z), NaN), oftype(float(z), NaN))
    m1, m2 = _raw_moments(x -> pdf(base, x), lo, hi)
    m = m1 / z
    return (m, m2 / z - m^2)
end

function _numeric_moments(d::Distributions.Censored{<:ContinuousUnivariateDistribution})
    base = d.uncensored
    lo = d.lower === nothing ? minimum(base) : float(d.lower)
    hi = d.upper === nothing ? maximum(base) : float(d.upper)
    pl = d.lower === nothing ? zero(float(lo)) : cdf(base, lo)
    pu = d.upper === nothing ? zero(float(hi)) : ccdf(base, hi)
    m1, m2 = _raw_moments(x -> pdf(base, x), lo, hi)
    pl > 0 && (m1 += lo * pl; m2 += lo^2 * pl)
    pu > 0 && (m1 += hi * pu; m2 += hi^2 * pu)
    return (m1, m2 - m1^2)
end

# Discrete truncated/censored wrappers: the same two moments by summation over
# the support window the CRPS fallback already knows how to find.
function _numeric_moments(d::DiscreteUnivariateDistribution)
    nodes = _crps_discrete_nodes(d)
    m1 = 0.0
    m2 = 0.0
    for k in nodes
        p = pdf(d, k)
        m1 += p * k
        m2 += p * k^2
    end
    return (m1, m2 - m1^2)
end

# Prefer the analytic moments; fall back to the numerical ones only where
# `Distributions.jl` has no method for this base.
function _wrapper_moments(d::UnivariateDistribution)
    return try
        (mean(d), var(d))
    catch e
        e isa MethodError || rethrow()
        _numeric_moments(d)
    end
end

_forecast_moments(d::Truncated) = _wrapper_moments(d)
_forecast_moments(d::Distributions.Censored) = _wrapper_moments(d)

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

Distribution-specific methods provide closed forms. Where there is none, a
continuous forecast falls back to adaptive quadrature and a discrete one to
exact integration of its step CDF, which needs only `cdf` and so also covers
families with no `quantile` method. Lower is better.

# Arguments

  - `d`: forecast distribution (any `UnivariateDistribution`).
  - `y`: scalar observation.

# Example

```@example
using Distributions, ScoringRules
crps(Normal(0, 1), 0.5)
```
"""
# Probabilities whose quantiles become quadrature segment boundaries. A bare
# `quadgk(f, lo, hi)` places its first nodes from the interval alone, so a
# forecast whose mass sits in a narrow band far from the midpoint of a
# half-infinite interval can be missed entirely, and the error estimate then
# reports convergence: a boundary-mass exponential of scale 0.057 at location
# 1.985 came out 1.8e-5 relative low, whatever `rtol` was asked for. Splitting
# at the forecast's own quantiles puts a node in every region that holds mass.
const _CRPS_QUAD_PROBS = (
    1.0e-8, 1.0e-4, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1 - 1.0e-4, 1 - 1.0e-8,
)

# Interior segment boundaries strictly between `a` and `b`, as `Float64`.
# Anything that does not convert is skipped, which is what keeps the fallback
# usable under automatic differentiation: a dual-valued quantile is dropped and
# the call degenerates to the plain two-point one.
function _crps_quad_breaks(d::ContinuousUnivariateDistribution, a::Real, b::Real)
    pts = Float64[]
    lo, hi = a < b ? (a, b) : (b, a)
    for p in _CRPS_QUAD_PROBS
        q = try
            Float64(quantile(d, p))
        catch
            continue
        end
        isfinite(q) && lo < q < hi && push!(pts, q)
    end
    for edge in (minimum(d), maximum(d))
        e = try
            Float64(edge)
        catch
            continue
        end
        isfinite(e) && lo < e < hi && push!(pts, e)
    end
    sort!(pts)
    unique!(pts)
    return a < b ? pts : reverse!(pts)
end

function crps(d::ContinuousUnivariateDistribution, y::Real)
    lo = minimum(d)
    hi = maximum(d)
    left, _ = quadgk(x -> cdf(d, x)^2, lo, _crps_quad_breaks(d, lo, y)..., y)
    right, _ = quadgk(x -> ccdf(d, x)^2, y, _crps_quad_breaks(d, y, hi)..., hi)
    return left + right
end

# Tail probability below which the discrete fallback stops widening its
# summation window, and the largest number of support points it will walk.
const _CRPS_DISCRETE_TOL = 1.0e-12
const _CRPS_DISCRETE_MAXNODES = 1 << 24

"""
    _crps_step(d, y, nodes)

CRPS of a discrete forecast by exact integration of its step CDF.

`nodes` lists the support points in increasing order. Between consecutive
nodes the CDF is the constant `F(nodes[i])`, so

```math
\\int_{s_i}^{s_{i+1}} (F(s_i) - \\mathbf{1}\\{x \\ge y\\})^2 dx
```

is piecewise quadratic in `y` and integrates in closed form. The two
half-infinite pieces contribute `max(s_1 - y, 0)` below the support and
`max(y - s_n, 0)` above it, which is what makes the score keep growing as `y`
moves away from the support instead of flattening out.
"""
function _crps_step(
        d::DiscreteUnivariateDistribution, y::Real,
        nodes::AbstractVector
    )
    n = length(nodes)
    n == 0 && throw(ArgumentError("the support of $d is empty"))
    out = max(float(first(nodes)) - y, 0.0)
    @inbounds for i in 1:(n - 1)
        a = float(nodes[i])
        b = float(nodes[i + 1])
        Fa = cdf(d, nodes[i])
        if y <= a
            out += (Fa - 1)^2 * (b - a)
        elseif y >= b
            out += Fa^2 * (b - a)
        else
            out += Fa^2 * (y - a) + (Fa - 1)^2 * (b - y)
        end
    end
    return out + max(y - float(last(nodes)), 0.0)
end

# Smallest integer `k` with `pred(k)`, for a predicate that is false below the
# transition and true above it. Exponential search outward from zero, then
# bisection, so neither an unbounded support nor a distribution whose mass sits
# far from zero (a `Poisson(10^9)`) needs a `quantile` method or a wide window.
function _first_true(pred, limit::Int = typemax(Int) ÷ 4)
    if pred(0)
        step = 1
        while step <= limit && pred(-step)
            step *= 2
        end
        step > limit && return -limit
        lo, hi = -step, -(step ÷ 2)
    else
        step = 1
        while step <= limit && !pred(step)
            step *= 2
        end
        step > limit && return limit
        lo, hi = step ÷ 2, step
    end
    while hi - lo > 1
        mid = lo + (hi - lo) ÷ 2
        pred(mid) ? (hi = mid) : (lo = mid)
    end
    return hi
end

"""
    _crps_discrete_nodes(d)

Support points of the discrete forecast `d` that the CRPS fallback sums over.

A support small enough to walk is used whole. Otherwise the window is the
integer range where the forecast keeps at least `_CRPS_DISCRETE_TOL` of
probability in each tail, located with `_first_true` so that only `cdf` is
needed: neither `quantile` nor `support` has a method for every discrete type
(`Skellam` has neither, and `support` of an unbounded one throws).
"""
function _crps_discrete_nodes(d::DiscreteUnivariateDistribution)
    lo, hi = minimum(d), maximum(d)
    if isfinite(lo) && isfinite(hi)
        s = support(d)
        length(s) <= _CRPS_DISCRETE_MAXNODES && return s
    end
    tol = _CRPS_DISCRETE_TOL
    klo = _first_true(k -> cdf(d, k) > tol)
    khi = _first_true(k -> ccdf(d, k) <= tol)
    isfinite(lo) && (klo = max(klo, ceil(Int, lo)))
    isfinite(hi) && (khi = min(khi, floor(Int, hi)))
    return klo:max(khi, klo)
end

# Generic discrete fallback. The named discrete families in `crps/discrete.jl`
# have closed forms and never reach this.
crps(d::DiscreteUnivariateDistribution, y::Real) =
    _crps_step(d, y, _crps_discrete_nodes(d))
