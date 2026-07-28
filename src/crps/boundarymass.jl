# Closed-form CRPS routes for `BoundaryMass` wrappers. Each method forwards to
# the validated internal ported from R scoringRules; combinations without a
# closed form fall back to the generic quadrature over the wrapper's CDF.

# Generic fallback: the quadrature method for continuous distributions handles
# the jump discontinuities of a mixed CDF through adaptive refinement.
function _crps_boundarymass_quad(d::BoundaryMass, y::Real)
    return invoke(crps, Tuple{ContinuousUnivariateDistribution, Real}, d, y)
end

# --- generalised truncated/censored normal, logistic, Student-t -------------
# The gtc internals are fully general in bounds and masses, so these routes
# are unconditional.
function crps(d::BoundaryMass{<:Real, <:Normal}, y::Real)
    μ, σ = params(d.dist)
    return _crps_gtcnorm(y, μ, σ, d.lower, d.upper, d.lmass, d.umass)
end

function crps(d::BoundaryMass{<:Real, <:Logistic}, y::Real)
    μ, θ = params(d.dist)
    return _crps_gtclogis(y, μ, θ, d.lower, d.upper, d.lmass, d.umass)
end

function crps(d::BoundaryMass{<:Real, <:TDist}, y::Real)
    return _crps_gtct(y, dof(d.dist), 0, 1, d.lower, d.upper, d.lmass, d.umass)
end

function crps(
        d::BoundaryMass{
            <:Real, <:Distributions.LocationScale{<:Real, Continuous, <:TDist}},
        y::Real)
    inner = d.dist
    return _crps_gtct(y, dof(inner.ρ), inner.μ, inner.σ,
        d.lower, d.upper, d.lmass, d.umass)
end

# --- uniform with boundary masses -------------------------------------------
# A truncated uniform is again uniform, so the closed form applies whenever
# each nonzero mass sits at the corresponding endpoint of the effective
# interval [max(a, lower), min(b, upper)].
function crps(d::BoundaryMass{<:Real, <:Uniform}, y::Real)
    a, b = params(d.dist)
    mn = max(a, d.lower)
    mx = min(b, d.upper)
    if mn < mx && (d.lmass == 0 || d.lower >= a) && (d.umass == 0 || d.upper <= b)
        return _crps_unif(y, mn, mx, d.lmass, d.umass)
    end
    return _crps_boundarymass_quad(d, y)
end

# --- exponential with a point mass at its lower end -------------------------
# By memorylessness an exponential truncated below at l ≥ 0 is the same
# exponential shifted to start at l, which is exactly the `expM` form with
# `location = l` and mass `lmass` there.
function crps(d::BoundaryMass{<:Real, <:Exponential}, y::Real)
    if d.umass == 0 && !isfinite(d.upper) && (d.lmass == 0 || d.lower >= 0)
        return _crps_expM(y, max(d.lower, zero(d.lower)), scale(d.dist), d.lmass)
    end
    return _crps_boundarymass_quad(d, y)
end

# --- generalised Pareto with a point mass at its location -------------------
# Truncating a GPD below at its own location μ is a no-op, so a mass there
# matches the `mass` form of R's crps_gpd.
function crps(d::BoundaryMass{<:Real, <:GeneralizedPareto}, y::Real)
    μ, σ, ξ = params(d.dist)
    if d.umass == 0 && d.upper >= maximum(d.dist) &&
       (d.lower == μ || (d.lmass == 0 && d.lower <= μ))
        return _crps_gpd(y, ξ, μ, σ, d.lmass)
    end
    return _crps_boundarymass_quad(d, y)
end
