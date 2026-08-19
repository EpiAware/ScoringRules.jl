# Continuous distribution truncated to an interval, with free point masses at
# the bounds. This is the "generalised truncated/censored" construction of
# R scoringRules (crps_gtcnorm and friends): truncation corresponds to zero
# masses, censoring to masses equal to the tail probabilities, and anything in
# between is a forecast that hedges between the two.
#
# Distributions.jl has no type for "distribution plus boundary point masses",
# so this wrapper provides one and the crps/logs/dss generics dispatch on it
# (see src/crps/boundarymass.jl for the closed-form CRPS routes).

"""
    BoundaryMass(dist; lower, upper, lmass, umass)

A continuous forecast distribution `dist` truncated to the interval
`[lower, upper]`, with free point masses `lmass` at `lower` and `umass` at
`upper`; the interior carries the remaining probability `1 - lmass - umass`.
This is the "generalised truncated/censored" distribution of R `scoringRules`:
zero masses recover `truncated(dist; lower, upper)`, and
`lmass = cdf(dist, lower)`, `umass = ccdf(dist, upper)` recover
`censored(dist; lower, upper)`.

`lower` and `upper` default to the support bounds of `dist` (no truncation);
a bound must be finite to carry a nonzero mass. As for
`Distributions.Censored`, `pdf`/`logpdf` at a bound with nonzero mass return
the probability mass itself.

[`crps`](@ref) has closed forms for normal, logistic and (location–scale)
Student-t bases, and for the boundary-mass exponential, uniform and
generalised Pareto forms; other bases fall back to quadrature. [`logs`](@ref)
and [`dss`](@ref) work through the generic density and moment methods.

# Example

```@example
using Distributions, ScoringRules
d = BoundaryMass(Normal(0, 1); lower = 0, upper = 2, lmass = 0.1, umass = 0.2)
crps(d, 0.5)
```
"""
struct BoundaryMass{T <: Real, D <: ContinuousUnivariateDistribution} <:
       ContinuousUnivariateDistribution
    dist::D   # base (untruncated) continuous distribution
    lower::T  # lower truncation bound (may be -Inf when lmass == 0)
    upper::T  # upper truncation bound (may be Inf when umass == 0)
    lmass::T  # point mass at `lower` (∈ [0, 1])
    umass::T  # point mass at `upper` (∈ [0, 1], lmass + umass ≤ 1)
    function BoundaryMass{T, D}(dist::D, lower::T, upper::T, lmass::T,
            umass::T) where {T <: Real, D <: ContinuousUnivariateDistribution}
        lower < upper ||
            throw(ArgumentError("lower must be strictly below upper"))
        (lmass >= 0 && umass >= 0 && lmass + umass <= 1) ||
            throw(DomainError((lmass, umass),
                "point masses must be nonnegative with lmass + umass ≤ 1"))
        lmass == 0 || isfinite(lower) ||
            throw(ArgumentError("nonzero lmass requires a finite lower bound"))
        umass == 0 || isfinite(upper) ||
            throw(ArgumentError("nonzero umass requires a finite upper bound"))
        return new{T, D}(dist, lower, upper, lmass, umass)
    end
end

function BoundaryMass(dist::ContinuousUnivariateDistribution;
        lower::Real = minimum(dist), upper::Real = maximum(dist),
        lmass::Real = 0, umass::Real = 0)
    T = promote_type(typeof(lower), typeof(upper), typeof(lmass),
        typeof(umass), Float64)
    return BoundaryMass{T, typeof(dist)}(dist, T(lower), T(upper), T(lmass),
        T(umass))
end

Distributions.params(d::BoundaryMass) = (d.dist, d.lower, d.upper, d.lmass, d.umass)
function Distributions.minimum(d::BoundaryMass)
    return d.lmass > 0 ? d.lower : max(minimum(d.dist), d.lower)
end
function Distributions.maximum(d::BoundaryMass)
    return d.umass > 0 ? d.upper : min(maximum(d.dist), d.upper)
end
function Distributions.insupport(d::BoundaryMass, x::Real)
    return minimum(d) <= x <= maximum(d)
end

# Probability left for the continuous interior between the two point masses.
_interior_mass(d::BoundaryMass) = 1 - (d.lmass + d.umass)

# The renormalised continuous part on (lower, upper). Only meaningful when
# `_interior_mass(d) > 0`; callers guard on that.
function _interior(d::BoundaryMass)
    return truncated(d.dist;
        lower = isfinite(d.lower) ? d.lower : nothing,
        upper = isfinite(d.upper) ? d.upper : nothing)
end

function Distributions.pdf(d::BoundaryMass, x::Real)
    (x == d.lower && d.lmass > 0) && return oftype(float(x), d.lmass)
    (x == d.upper && d.umass > 0) && return oftype(float(x), d.umass)
    (x < d.lower || x > d.upper) && return zero(float(x))
    a = _interior_mass(d)
    a == 0 && return zero(float(x))
    return a * pdf(_interior(d), x)
end

function Distributions.logpdf(d::BoundaryMass, x::Real)
    (x == d.lower && d.lmass > 0) && return oftype(float(x), log(d.lmass))
    (x == d.upper && d.umass > 0) && return oftype(float(x), log(d.umass))
    (x < d.lower || x > d.upper) && return oftype(float(x), -Inf)
    a = _interior_mass(d)
    a == 0 && return oftype(float(x), -Inf)
    return log(a) + logpdf(_interior(d), x)
end

function Distributions.cdf(d::BoundaryMass, x::Real)
    x < d.lower && return zero(float(x))
    x >= d.upper && return one(float(x))
    a = _interior_mass(d)
    a == 0 && return oftype(float(x), d.lmass)
    return d.lmass + a * cdf(_interior(d), x)
end

function Distributions.quantile(d::BoundaryMass, p::Real)
    (d.lmass > 0 && p <= d.lmass) && return oftype(float(p), d.lower)
    (d.umass > 0 && p >= 1 - d.umass) && return oftype(float(p), d.upper)
    return quantile(_interior(d), (p - d.lmass) / _interior_mass(d))
end

function Distributions.mean(d::BoundaryMass)
    a = _interior_mass(d)
    # Guard the mass terms: 0 * Inf would poison an unbounded, mass-free side.
    m = a > 0 ? a * mean(_interior(d)) : zero(a)
    d.lmass > 0 && (m += d.lmass * d.lower)
    d.umass > 0 && (m += d.umass * d.upper)
    return m
end

function Distributions.var(d::BoundaryMass)
    a = _interior_mass(d)
    m = mean(d)
    v = zero(m)
    d.lmass > 0 && (v += d.lmass * (d.lower - m)^2)
    d.umass > 0 && (v += d.umass * (d.upper - m)^2)
    if a > 0
        int = _interior(d)
        v += a * (var(int) + (mean(int) - m)^2)
    end
    return v
end

function Distributions.rand(rng::AbstractRNG, d::BoundaryMass)
    u = rand(rng)
    u < d.lmass && return oftype(float(u), d.lower)
    u >= 1 - d.umass && return oftype(float(u), d.upper)
    return rand(rng, _interior(d))
end
