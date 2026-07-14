# Log-Laplace distribution.
# X ~ LogLaplace(μ, σ)  iff  log(X) ~ Laplace(μ, σ).
# Parameters: μ (location on log scale), σ > 0 (scale on log scale).
# CRPS exists for σ < 1; mean/variance for σ < 1 and σ < 1/2 respectively.
#
# Ported from R scoringRules distributionFunctions.R (fllapl/flapl) and
# scores_llapl.R (Jordan, Krüger, Lerch, Allen).

"""
    LogLaplace(μ, σ)

The log-Laplace distribution: `X ~ LogLaplace(μ, σ)` iff `log(X) ~ Laplace(μ, σ)`,
with `μ` the location and `σ > 0` the scale, both on the log scale. Supported on
the positive reals. The mean exists for `σ < 1` and the variance for `σ < 1/2`.
Not part of Distributions.jl; provided here so it flows through
[`crps`](@ref)/[`logs`](@ref)/[`dss`](@ref) dispatch.
"""
struct LogLaplace{T<:Real} <: ContinuousUnivariateDistribution
    μ::T  # location on log scale
    σ::T  # scale on log scale (> 0)
    function LogLaplace{T}(μ::T, σ::T) where {T<:Real}
        σ > zero(T) || error("σ must be positive")
        return new{T}(μ, σ)
    end
end

function LogLaplace(μ::Real, σ::Real)
    T = promote_type(typeof(μ), typeof(σ), Float64)
    return LogLaplace{T}(T(μ), T(σ))
end

Distributions.params(d::LogLaplace) = (d.μ, d.σ)
Distributions.minimum(::LogLaplace) = 0.0
Distributions.maximum(::LogLaplace) = Inf
Distributions.insupport(::LogLaplace, x::Real) = x > 0

# flapl(x, location, scale) = dexp(|x - location|, 1/scale) / 2
#   = (1/(2*scale)) * exp(-|x - location| / scale)
# fllapl(x, μ, σ) = (1/x) * flapl(log(x), μ, σ)
function Distributions.pdf(d::LogLaplace, x::Real)
    x <= 0 && return zero(float(x))
    lx = log(x)
    return exp(-abs(lx - d.μ) / d.σ) / (2 * d.σ * x)
end

function Distributions.logpdf(d::LogLaplace, x::Real)
    x <= 0 && return oftype(float(x), -Inf)
    lx = log(x)
    return -abs(lx - d.μ) / d.σ - log(2 * d.σ * x)
end

# CDF of Laplace(μ,σ) at log(x):
#   x <= exp(μ): F(x) = (1/2)*exp((log(x)-μ)/σ)
#   x >  exp(μ): F(x) = 1 - (1/2)*exp(-(log(x)-μ)/σ)
function Distributions.cdf(d::LogLaplace, x::Real)
    x <= 0 && return zero(float(x))
    z = (log(x) - d.μ) / d.σ
    return z < 0 ? exp(z) / 2 : 1 - exp(-z) / 2
end

# quantile: Q(p) = exp(μ + σ*sign(p-0.5)*log(2*min(p,1-p)))
# equivalently: p <= 0.5 → exp(μ + σ*log(2p)); p > 0.5 → exp(μ - σ*log(2(1-p)))
function Distributions.quantile(d::LogLaplace, p::Real)
    if p <= 0.5
        return exp(d.μ + d.σ * log(2 * p))
    else
        return exp(d.μ - d.σ * log(2 * (1 - p)))
    end
end

# Mean: E[X] = exp(μ) / (1 - σ²)   (requires σ < 1)
function Distributions.mean(d::LogLaplace)
    return exp(d.μ) / (1 - d.σ^2)
end

# Var[X] = exp(2μ) * (1/(1 - 4σ²) - 1/(1 - σ²)²)   (requires σ < 1/2)
function Distributions.var(d::LogLaplace)
    sl2 = d.σ^2
    e2  = exp(2 * d.μ)
    return e2 * (1 / (1 - 4 * sl2) - 1 / (1 - sl2)^2)
end

function Distributions.rand(rng::AbstractRNG, d::LogLaplace)
    u = rand(rng) - 0.5
    return exp(d.μ - d.σ * sign(u) * log(1 - 2 * abs(u)))
end
