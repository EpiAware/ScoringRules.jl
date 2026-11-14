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
function _forecast_moments(d::UnivariateDistribution)
    v = _forecast_var(d)
    # The mean is asked for only once the variance is known to be usable. A
    # variance that is not finite and positive makes the score `NaN` whatever
    # the mean is, and a family with no mean throws rather than returning one:
    # `LogLogistic` with β ≤ 1 took `dss` down with it where it had returned
    # `NaN`, which is what this package documents and what R does.
    (isfinite(v) && v > 0) || return (oftype(float(v) * one(v), NaN), v)
    return (mean(d), v)
end

_forecast_var(d::UnivariateDistribution) = var(d)

# `LogLogistic` throws rather than returning `Inf` once its second moment stops
# existing (shape β ≤ 2).
_forecast_var(d::LogLogistic) = d.β > 2 ? var(d) : oftype(float(d.β), Inf)

# --- GEV moments near the Gumbel limit -------------------------------------
#
# `mean` and `var` of a GEV are built from `Γ(1−ξ)` and `Γ(1−2ξ)`, and both
# combinations cancel as ξ → 0: `var` evaluates `(Γ(1−2ξ) − Γ(1−ξ)²)/ξ²`,
# whose numerator is O(ξ²). At ξ = 10⁻⁸ the variance that `Distributions.var`
# returns is 2.22 against the true 1.645, and the `dss` built on it 0.80
# against 0.50. Replacing the cancelling parts with `lgamma` differences does
# not help either, because `lgamma(1−ξ)` is itself O(ξ) and its error is
# absolute rather than relative.
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

# `expm1(q)/q`, continued to its limit 1 at q = 0. Below the cutoff the
# Maclaurin series `Σ qⁿ/(n+1)!` is used rather than a branch on `q == 0`: the
# division is 0/0 there, and under automatic differentiation the branch is not
# even taken, because a dual with a zero value and a nonzero partial is not
# `iszero`. The series is right in the value and the derivative alike, and its
# truncation error at the cutoff is below 10⁻²¹.
@inline function _expm1_over(q::Real)
    abs(q) < 1.0e-5 && return 1 + q / 2 + q^2 / 6 + q^3 / 24
    return expm1(q) / q
end

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

# Every variance below is taken about the mean rather than as
# `E[X²] − E[X]²`. The raw second moment of a forecast whose support sits far
# from the origin is dominated by the mean, and the difference loses the spread
# entirely: a `Logistic(1e8, 1)` truncated to ±1 gave −4.4e5 where its variance
# is 0.312. On the `BoundaryMass` path, where the same form was used, a
# `Normal(1e8, 1)` on the same interval gave exactly 0.

# Probability that a truncation to `[lo, hi]` retains. As a difference of
# CDFs this cancels when both bounds sit in the same tail:
# `cdf(Exponential(1), Inf) - cdf(Exponential(1), 40)` is exactly 0 where the
# retained mass is 4.2e-18, which made the moments below, and the `dss` built
# on them, nonsense. Taking the difference on whichever side the interval lies
# keeps both terms small and the result accurate.
function _trunc_mass(base, lo::Real, hi::Real)
    return cdf(base, lo) > 0.5 ? ccdf(base, lo) - ccdf(base, hi) :
        cdf(base, hi) - cdf(base, lo)
end

# A point to measure the first moment from. `∫ x f(x) dx` over
# `[1e8 − 1, 1e8 + 1]` is dominated by `1e8` times the retained mass, so
# `quadgk`'s relative tolerance leaves an absolute error of order 1 in a window
# only 2 wide and the mean is useless. Integrating `x − c` removes that term.
#
# The median is the point that sits near the mass whichever way the interval is
# lopsided. A bound does not: for `truncated(Laplace(0, 1); lower = -1e8)` the
# whole forecast is at 0 and the bound is 1e8 away, and measuring from there
# put the mean out by 3.7e6 and `dss` 45 times over.
function _shift_point(d, lo::Real, hi::Real)
    q = try
        Float64(quantile(d, 0.5))
    catch
        NaN
    end
    isfinite(q) && return q
    isfinite(lo) && isfinite(hi) && return (float(lo) + float(hi)) / 2
    isfinite(lo) && return float(lo)
    isfinite(hi) && return float(hi)
    return zero(float(lo))
end

# First moment of a continuous density over `[lo, hi]` measured from `c`, and
# the second about `m`, by adaptive quadrature. Both are split at the quantiles
# of `d`, for the reason `_CRPS_QUAD_PROBS` gives: without them `quadgk` on a
# half-infinite interval misses a narrow band of mass and reports convergence,
# and `truncated(Exponential(1e-6), 0, Inf)` came back with both moments zero.
# Each segment integrated on its own and the results added, rather than handing
# `quadgk` the whole range in one call. Its convergence test is relative to the
# total, and the first moment of a near-symmetric forecast is close to zero, so
# one call spends its entire budget on a stretch that contributes nothing and
# stops at its evaluation limit: `truncated(Laplace(0, 1); lower = -1e8)` gave
# a mean of -0.0056 rather than 0. Per segment the budget is relative to that
# segment's own contribution, so the empty stretch costs nothing and the tail
# that holds the second moment is still resolved. Clipping the range instead
# would discard that tail: it holds a negligible share of the probability and a
# material share of the second moment, which cost the variance of a truncated
# `LogNormal(0, 3)` 1.4%.
function _quad_sum(g, pts::AbstractVector)
    acc = first(quadgk(g, pts[1], pts[2]; rtol = 1.0e-12))
    for i in 2:(length(pts) - 1)
        acc += first(quadgk(g, pts[i], pts[i + 1]; rtol = 1.0e-12))
    end
    return acc
end

_quad_pts(d, lo::Real, hi::Real) =
    [float(lo); _crps_quad_breaks(d, lo, hi, _MOMENT_QUAD_PROBS); float(hi)]

_quad_first(f, d, lo::Real, hi::Real, c::Real) =
    _quad_sum(x -> (x - c) * f(x), _quad_pts(d, lo, hi))

_quad_central(f, d, lo::Real, hi::Real, m::Real) =
    _quad_sum(x -> (x - m)^2 * f(x), _quad_pts(d, lo, hi))

# A truncation that keeps an infinite tail has moments only where the base
# does. Without this the quadrature stops at the deepest quantile it can
# represent and reports a finite number for an integral that diverges: a
# `Cauchy` truncated from below scored 37.4, where its variance does not exist
# and the same family untruncated correctly scores `NaN`.
function _tail_moments_exist(base, lo::Real, hi::Real)
    isfinite(lo) && isfinite(hi) && return true
    v = try
        _forecast_var(base)
    catch e
        e isa MethodError || rethrow()
        return false
    end
    return isfinite(v)
end

function _numeric_moments(d::Truncated{<:ContinuousUnivariateDistribution})
    base = d.untruncated
    lo = d.lower === nothing ? minimum(base) : float(d.lower)
    hi = d.upper === nothing ? maximum(base) : float(d.upper)
    _tail_moments_exist(base, lo, hi) || return (NaN, NaN)
    z = _trunc_mass(base, lo, hi)
    z > 0 || return (oftype(float(z), NaN), oftype(float(z), NaN))
    f = x -> pdf(base, x)
    c = _shift_point(d, lo, hi)
    m = c + _quad_first(f, d, lo, hi, c) / z
    return (m, _quad_central(f, d, lo, hi, m) / z)
end

function _numeric_moments(d::Distributions.Censored{<:ContinuousUnivariateDistribution})
    base = d.uncensored
    lo = d.lower === nothing ? minimum(base) : float(d.lower)
    hi = d.upper === nothing ? maximum(base) : float(d.upper)
    _tail_moments_exist(base, lo, hi) || return (NaN, NaN)
    pl = d.lower === nothing ? zero(float(lo)) : cdf(base, lo)
    pu = d.upper === nothing ? zero(float(hi)) : ccdf(base, hi)
    f = x -> pdf(base, x)
    c = _shift_point(d, lo, hi)
    m = c + _quad_first(f, d, lo, hi, c)
    pl > 0 && (m += (lo - c) * pl)
    pu > 0 && (m += (hi - c) * pu)
    v = _quad_central(f, d, lo, hi, m)
    pl > 0 && (v += (lo - m)^2 * pl)
    pu > 0 && (v += (hi - m)^2 * pu)
    return (m, v)
end

# Discrete truncated/censored wrappers: the same two moments by summation over
# the support window the CRPS fallback already knows how to find. The window
# can clip a negligible tail, so both are normalised by the mass it holds.
function _numeric_moments(d::DiscreteUnivariateDistribution)
    nodes = _crps_discrete_nodes(d)
    w = 0.0
    m = 0.0
    for k in nodes
        p = pdf(d, k)
        w += p
        m += p * k
    end
    w > 0 || return (NaN, NaN)
    m /= w
    v = 0.0
    for k in nodes
        v += pdf(d, k) * (k - m)^2
    end
    return (m, v / w)
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
# 1.985 came out 1.8e-5 relative high, whatever `rtol` was asked for. Splitting
# at the forecast's own quantiles puts a node in every region that holds mass.
const _CRPS_QUAD_PROBS = (
    1.0e-8, 1.0e-4, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1 - 1.0e-4, 1 - 1.0e-8,
)

# The moment integrands are weighted by `x` and `(x - m)^2`, so a tail holding
# only 1e-8 of the probability can still hold 1e-6 of the second moment:
# stopping at 1e-8 left the variance of `truncated(Exponential(1e-6), 0, Inf)`
# 3.4e-6 out, and reaching 1e-14 brings it to 1e-11. The CRPS integrand is
# bounded by 1 and needs no such depth, which is just as well: on a heavy tail
# the 1e-14 quantile is around 1e14, and a segment boundary that far out next
# to an infinite endpoint puts a Kronrod node on the endpoint itself, where
# QuadGK's substitution evaluates `0 * Inf` and raises.
const _MOMENT_QUAD_PROBS = (
    1.0e-14, 1.0e-10, 1.0e-8, 1.0e-4, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99,
    1 - 1.0e-4, 1 - 1.0e-8, 1 - 1.0e-10, 1 - 1.0e-14,
)

# Interior segment boundaries strictly between `a` and `b`, as `Float64`.
# Anything that does not convert is skipped, which is what keeps the fallback
# usable under automatic differentiation: a dual-valued quantile is dropped and
# the call degenerates to the plain two-point one.
# How far out a segment boundary may sit, measured in interquartile ranges
# from the median. On a heavy tail the 1e-8 quantile can be 1e16, and a
# boundary that far out next to an infinite endpoint makes QuadGK's
# substitution produce a NaN and raise, which took `crps` of a truncated
# `Pareto(0.5, 1)` down. The integrand is bounded by 1, so the quadrature
# resolves whatever lies beyond the last boundary on its own.
const _QUAD_BREAK_REACH = 1.0e6

function _crps_quad_breaks(
        d::ContinuousUnivariateDistribution, a::Real, b::Real,
        probs = _CRPS_QUAD_PROBS
    )
    pts = Float64[]
    lo, hi = a < b ? (a, b) : (b, a)
    reach = _break_reach(d)
    for p in probs
        q = try
            Float64(quantile(d, p))
        catch
            continue
        end
        isfinite(q) && lo < q < hi && abs(q) <= reach && push!(pts, q)
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

# The largest magnitude a segment boundary may take, from the forecast's own
# interquartile scale. Falls back to `Inf` (no bound) where the quartiles are
# not available.
function _break_reach(d::ContinuousUnivariateDistribution)
    q1, q3, md = try
        (
            Float64(quantile(d, 0.25)), Float64(quantile(d, 0.75)),
            Float64(quantile(d, 0.5)),
        )
    catch
        return Inf
    end
    (isfinite(q1) && isfinite(q3) && isfinite(md)) || return Inf
    scale = max(q3 - q1, abs(md), 1.0)
    return abs(md) + _QUAD_BREAK_REACH * scale
end

function crps(d::ContinuousUnivariateDistribution, y::Real)
    lo = minimum(d)
    hi = maximum(d)
    left, _ = quadgk(x -> cdf(d, x)^2, lo, _crps_quad_breaks(d, lo, y)..., y)
    right, _ = quadgk(x -> ccdf(d, x)^2, y, _crps_quad_breaks(d, y, hi)..., hi)
    return left + right
end

# A truncated forecast's own CDF is a difference of base CDFs over its
# normalising mass, and for a window deep in a tail all three round to the same
# thing: `cdf` and `ccdf` of `truncated(Exponential(1), 40, Inf)` are both
# exactly 0 at 45, which made the generic CRPS 0.0 for an observation four
# units from the forecast mean. Rebuilding both from the base on whichever side
# keeps its accuracy, and taking the segment boundaries the same way, puts the
# score back. The families with a closed form have their own more specific
# methods and never reach this.
function crps(d::Truncated{<:ContinuousUnivariateDistribution}, y::Real)
    base = d.untruncated
    lo = d.lower === nothing ? minimum(base) : float(d.lower)
    hi = d.upper === nothing ? maximum(base) : float(d.upper)
    z = _trunc_mass(base, lo, hi)
    z > 0 || return oftype(float(y), NaN)
    upper_side = cdf(base, lo) > 0.5
    F = function (x)
        x <= lo && return 0.0
        x >= hi && return 1.0
        v = upper_side ? (ccdf(base, lo) - ccdf(base, x)) / z :
            (cdf(base, x) - cdf(base, lo)) / z
        return clamp(v, 0.0, 1.0)
    end
    pts = _trunc_break_pts(base, lo, hi, z, upper_side)
    left, _ = quadgk(x -> F(x)^2, lo, filter(q -> lo < q < y, pts)..., y)
    right, _ = quadgk(x -> (1 - F(x))^2, y, filter(q -> y < q < hi, pts)..., hi)
    return left + right
end

# Quantiles of the truncated forecast, solved in the base's own tail so that a
# window deep in one does not collapse: `quantile(truncated(Exponential(1), 40,
# Inf), 0.5)` is `Inf`, where the answer is 40.69.
function _trunc_break_pts(base, lo::Real, hi::Real, z::Real, upper_side::Bool)
    pts = Float64[]
    reach = _break_reach(base)
    for p in _CRPS_QUAD_PROBS
        q = try
            upper_side ? Float64(cquantile(base, ccdf(base, lo) - p * z)) :
                Float64(quantile(base, cdf(base, lo) + p * z))
        catch
            continue
        end
        isfinite(q) && lo < q < hi && abs(q) <= reach && push!(pts, q)
    end
    sort!(pts)
    unique!(pts)
    return pts
end

# Tail probability below which the discrete fallback stops widening its
# summation window; the largest support it will enumerate directly rather than
# fall back to that window; and the widest window it will walk at all, beyond
# which a single score would take minutes (`Geometric(1e-9)` spreads over
# 2.8e10 lattice points) and an error is the better answer.
const _CRPS_DISCRETE_TOL = 1.0e-12
const _CRPS_DISCRETE_MAXNODES = 1 << 24
const _CRPS_DISCRETE_MAXWINDOW = 1 << 31

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
        nodes::AbstractVector; cumulative::Bool = false
    )
    n = length(nodes)
    n == 0 && throw(ArgumentError("the support of $d is empty"))
    out = max(float(first(nodes)) - y, 0.0)
    # Where the nodes are the whole support, the CDF at each is the running
    # sum of the masses up to it. `cdf` of a `DiscreteNonParametric` scans its
    # support on every call, which made the score quadratic in its size.
    Fa = 0.0
    @inbounds for i in 1:(n - 1)
        a = float(nodes[i])
        b = float(nodes[i + 1])
        Fa = cumulative ? Fa + pdf(d, nodes[i]) : cdf(d, nodes[i])
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
    nodes = _with_bounds(d, _crps_lattice_nodes(d))
    _check_mass_on_nodes(d, nodes)
    return nodes
end

# The step CDF can only jump at the nodes, so between the first node and the
# last the masses at the nodes must account for the whole rise of the CDF. A
# forecast placed on the integers it does not take, such as a mixture with a
# shifted component or a shifted forecast inside a truncation, fails this, and
# summing it would score it as if the mass between the nodes were not there.
function _check_mass_on_nodes(d, nodes)
    length(nodes) < 2 && return nothing
    # `pdf` of a shifted Poisson at an integer it does not take evaluates the
    # Poisson at a non-integer, where Distributions.jl throws a `DomainError`
    # rather than returning zero; that is the same finding.
    held = try
        sum(x -> pdf(d, x), @view nodes[2:end])
    catch e
        e isa DomainError || rethrow()
        _off_nodes(d)
    end
    rise = cdf(d, last(nodes)) - cdf(d, first(nodes))
    abs(held - rise) <= 1.0e-8 * max(rise, 1.0e-300) + 1.0e-12 || _off_nodes(d)
    return nothing
end

_off_nodes(d) = throw(
    ArgumentError(
        "$d has mass between the points the generic discrete CRPS sums on, " *
            "and cannot be scored by it"
    )
)

function _crps_lattice_nodes(d::DiscreteUnivariateDistribution)
    lo, hi = minimum(d), maximum(d)
    s = _finite_support(d)
    s === nothing || return s
    tol = _CRPS_DISCRETE_TOL
    klo = _first_true(k -> cdf(d, k) > tol)
    khi = _first_true(k -> ccdf(d, k) <= tol)
    isfinite(lo) && (klo = max(klo, ceil(Int, lo)))
    isfinite(hi) && (khi = min(khi, floor(Int, hi)))
    khi = max(khi, klo)
    (khi - klo) <= _CRPS_DISCRETE_MAXWINDOW || throw(
        ArgumentError(
            "the forecast $d spreads over $(khi - klo + 1) lattice points, " *
                "more than the generic discrete CRPS will sum"
        )
    )
    return klo:khi
end

# The support points of a finite discrete forecast, or `nothing` where the
# support is unbounded or too large to list.
function _finite_support(d::DiscreteUnivariateDistribution)
    (isfinite(minimum(d)) && isfinite(maximum(d))) || return nothing
    s = support(d)
    length(s) <= _CRPS_DISCRETE_MAXNODES || return nothing
    # `support` is an `AbstractVector` for every family in Distributions.jl
    # but `Dirac`, whose is a one-element tuple.
    return s isa AbstractVector ? s : collect(s)
end

# A truncated or censored discrete forecast inherits `support` from the
# generic method, which assumes the integers: for a forecast on non-integer
# points it is `round(minimum):round(maximum)` and lists none of them. Its
# points are the base's, inside its bounds.
const _WrappedDiscrete = Union{
    Truncated{<:DiscreteUnivariateDistribution},
    Distributions.Censored{<:DiscreteUnivariateDistribution},
}

_wrapped_base(d::Truncated) = d.untruncated
_wrapped_base(d::Distributions.Censored) = d.uncensored

function _finite_support(d::_WrappedDiscrete)
    s = _finite_support(_wrapped_base(d))
    s === nothing && return nothing
    lo, hi = minimum(d), maximum(d)
    return filter(x -> lo <= x <= hi, s)
end

# A forecast censored between the integers piles the mass beyond each bound
# onto the bound itself, yet `support` lists only the integers inside: for
# `censored(Poisson(3), 1.5, 6.5)` it is `2:6`, while `pdf` at 1.5 is 0.2. So
# the forecast's own finite bounds join the nodes. Truncation at the same
# bounds puts no mass there, and the extra node then changes nothing.
function _with_bounds(d, nodes)
    extra = filter(b -> isfinite(b) && !(b in nodes), [minimum(d), maximum(d)])
    isempty(extra) && return nodes
    return sort!(vcat(collect(nodes), extra))
end

# Generic discrete fallback. The named discrete families in `crps/discrete.jl`
# have closed forms and never reach this. Where the support is unbounded,
# `_crps_discrete_nodes` places the nodes on the integers, so the step CDF has
# to jump there; an affine transform of such a forecast is routed through the
# equivariance below rather than summed off its own lattice.
function crps(d::DiscreteUnivariateDistribution, y::Real)
    nodes = _crps_discrete_nodes(d)
    return _crps_step(d, y, nodes; cumulative = _finite_support(d) !== nothing)
end

# CRPS is equivariant under an affine map: for `d = μ + σX`,
# CRPS(d, y) = |σ|·CRPS(X, (y − μ)/σ) when σ > 0, and the reflection
# CRPS(−X, z) = CRPS(X, −z) carries the decreasing case. Without this,
# `3.5 + Poisson(2)` was summed at the integers while its CDF jumps at the
# half-integers, and the score came out 0.499 against the true 0.405.
function crps(
        d::Distributions.LocationScale{<:Real, Discrete, <:DiscreteUnivariateDistribution},
        y::Real
    )
    σ = d.σ
    σ == 0 && return abs(float(y - d.μ))
    z = (y - d.μ) / abs(σ)
    return abs(σ) * crps(d.ρ, σ > 0 ? z : -z)
end
