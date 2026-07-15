# Two-piece distributions (split normal and split exponential).
#
# Both distributions are centred at `location` with asymmetric arms scaled
# by `scale1` (left) and `scale2` (right).
#
# Densities follow f2pnorm / f2pexp in R scoringRules distributionFunctions.R
# (Jordan, Krüger, Lerch, Allen).

# ---------------------------------------------------------------------------
# TwoPieceNormal
# ---------------------------------------------------------------------------
# f(x; ℓ, σ₁, σ₂) = 2·s/(σ₁+σ₂) · φ((x-ℓ)/s)   where s=σ₁ if x<ℓ, s=σ₂ if x≥ℓ
# Mean  = ℓ + √(2/π)·(σ₂ - σ₁)
# Var   = (1 - 2/π)·(σ₂² + σ₁²) + (4/π - 1)·(σ₂ - σ₁)²/2  ... simplified:
#       = σ₁² + σ₂² - (4/π-1)·(σ₁ - σ₂)² ... derived below
#
# Derivation of moments (location=0):
#   E[X] = 2/(σ₁+σ₂) * [∫_{-∞}^0 x·φ(x/σ₁)dx + ∫_0^∞ x·φ(x/σ₂)dx]
#         = 2/(σ₁+σ₂) * [−σ₁²·φ(0) + σ₂²·φ(0)]
#         = 2(σ₂²−σ₁²)/((σ₁+σ₂)·√(2π))
#         = √(2/π)·(σ₂ − σ₁)
#
#   E[X²]= 2/(σ₁+σ₂) * [σ₁³·∫_{−∞}^0 z²·φ(z)dz + σ₂³·∫_0^∞ z²·φ(z)dz]
#         = 2/(σ₁+σ₂) * [σ₁³ + σ₂³]·(1/2)   (each half-normal second moment = 1/2)
#         = (σ₁³ + σ₂³)/(σ₁+σ₂)
#         = σ₁² − σ₁·σ₂ + σ₂²
#
#   Var   = E[X²] − (E[X])²
#         = σ₁² − σ₁σ₂ + σ₂² − (2/π)(σ₂ − σ₁)²

const _SQRT2DIVPI = sqrt(2 / π)
const _2DIVPI = 2 / π

"""
    TwoPieceNormal(location, scale1, scale2)

The two-piece (split) normal distribution: a normal density with scale `scale1`
below `location` and `scale2` above it, renormalised to a proper density. With
`scale1 == scale2` it reduces to `Normal(location, scale1)`. Not part of
Distributions.jl; provided here so it flows through
[`crps`](@ref)/[`logs`](@ref)/[`dss`](@ref) dispatch.
"""
struct TwoPieceNormal{T <: Real} <: ContinuousUnivariateDistribution
    location::T
    scale1::T  # left-arm scale  (> 0)
    scale2::T  # right-arm scale (> 0)
    function TwoPieceNormal{T}(location::T, scale1::T, scale2::T) where {T <: Real}
        scale1 > zero(T) || throw(DomainError(scale1, "scale1 must be positive"))
        scale2 > zero(T) || throw(DomainError(scale2, "scale2 must be positive"))
        return new{T}(location, scale1, scale2)
    end
end

function TwoPieceNormal(location::Real, scale1::Real, scale2::Real)
    T = promote_type(typeof(location), typeof(scale1), typeof(scale2), Float64)
    return TwoPieceNormal{T}(T(location), T(scale1), T(scale2))
end

Distributions.params(d::TwoPieceNormal) = (d.location, d.scale1, d.scale2)
Distributions.minimum(::TwoPieceNormal) = -Inf
Distributions.maximum(::TwoPieceNormal) = Inf
Distributions.insupport(::TwoPieceNormal, ::Real) = true

function Distributions.pdf(d::TwoPieceNormal, x::Real)
    z = x - d.location
    s = z < 0 ? d.scale1 : d.scale2
    S = d.scale1 + d.scale2
    return 2 * s / S * exp(-(z / s)^2 / 2) / (s * sqrt(2 * oftype(float(x), π)))
end

function Distributions.logpdf(d::TwoPieceNormal, x::Real)
    z = x - d.location
    s = z < 0 ? d.scale1 : d.scale2
    S = d.scale1 + d.scale2
    return log(2) - log(S) - (z / s)^2 / 2 - log(sqrt(2 * oftype(float(x), π)))
end

# CDF: F(x) = 2σ₁/(σ₁+σ₂) * Φ((x-ℓ)/σ₁)             for x < ℓ
#             2σ₁/(σ₁+σ₂) * Φ(0) + 2σ₂/(σ₁+σ₂) * Φ((x-ℓ)/σ₂) * ...
# More carefully:
#   F(x) = 2σ₁/(σ₁+σ₂) * Φ((x-ℓ)/σ₁)                           x ≤ ℓ
#   F(x) = σ₁/(σ₁+σ₂) + 2σ₂/(σ₁+σ₂) * (Φ((x-ℓ)/σ₂) - 1/2)    x > ℓ
function Distributions.cdf(d::TwoPieceNormal, x::Real)
    z = x - d.location
    S = d.scale1 + d.scale2
    if z <= 0
        return 2 * d.scale1 / S * _norm_cdf(z / d.scale1)
    else
        return d.scale1 / S +
               2 * d.scale2 / S * (_norm_cdf(z / d.scale2) - oftype(float(x), 0.5))
    end
end

# Quantile: invert CDF piecewise
function Distributions.quantile(d::TwoPieceNormal, p::Real)
    S = d.scale1 + d.scale2
    p_split = d.scale1 / S
    if p <= p_split
        # p = 2σ₁/S * Φ(q/σ₁) → q = σ₁ * Φ⁻¹(p*S/(2σ₁))
        return d.location + d.scale1 * Distributions.norminvcdf(p * S / (2 * d.scale1))
    else
        # p = σ₁/S + 2σ₂/S*(Φ(q/σ₂) - 1/2)
        # (p - σ₁/S)*S/(2σ₂) + 1/2 = Φ(q/σ₂)
        arg = (p - p_split) * S / (2 * d.scale2) + oftype(float(p), 0.5)
        return d.location + d.scale2 * Distributions.norminvcdf(arg)
    end
end

function Distributions.mean(d::TwoPieceNormal)
    return d.location + _SQRT2DIVPI * (d.scale2 - d.scale1)
end

function Distributions.var(d::TwoPieceNormal)
    s1, s2 = d.scale1, d.scale2
    # E[X²] (centred) = s1² - s1*s2 + s2²; subtract (E[X-loc])² = (2/π)*(s2-s1)²
    return s1^2 - s1 * s2 + s2^2 - _2DIVPI * (s2 - s1)^2
end

function Distributions.rand(rng::AbstractRNG, d::TwoPieceNormal)
    S = d.scale1 + d.scale2
    u = rand(rng)
    p_split = d.scale1 / S
    if u <= p_split
        return d.location + d.scale1 * Distributions.norminvcdf(u * S / (2 * d.scale1))
    else
        arg = (u - p_split) * S / (2 * d.scale2) + oftype(float(u), 0.5)
        return d.location + d.scale2 * Distributions.norminvcdf(arg)
    end
end

# ---------------------------------------------------------------------------
# TwoPieceExponential
# ---------------------------------------------------------------------------
# f(x; ℓ, σ₁, σ₂) = s/(σ₁+σ₂) · (1/s)·exp(-|x-ℓ|/s)   where s=σ₁ if x<ℓ, s=σ₂ if x≥ℓ
#                  = 1/(σ₁+σ₂) · exp(-|x-ℓ|/s)
#
# Mean  = ℓ + σ₂ - σ₁
# Var   = σ₁² + σ₂²
#
# Derivation of moments (location=0):
#   E[X] = 1/(σ₁+σ₂) * [∫_{-∞}^0 x·exp(x/σ₁)dx + ∫_0^∞ x·exp(-x/σ₂)dx]
#         = 1/(σ₁+σ₂) * [−σ₁² + σ₂²] = σ₂ − σ₁
#
#   E[X²]= 1/(σ₁+σ₂) * [∫_{-∞}^0 x²·exp(x/σ₁)dx + ∫_0^∞ x²·exp(-x/σ₂)dx]
#         = 1/(σ₁+σ₂) * [2σ₁³ + 2σ₂³]
#         = 2(σ₁³+σ₂³)/(σ₁+σ₂) = 2(σ₁²−σ₁σ₂+σ₂²)
#
#   Var   = E[X²] − (E[X])² = 2(σ₁²−σ₁σ₂+σ₂²) − (σ₂−σ₁)²
#         = 2σ₁² − 2σ₁σ₂ + 2σ₂² − σ₂² + 2σ₁σ₂ − σ₁²
#         = σ₁² + σ₂²

"""
    TwoPieceExponential(location, scale1, scale2)

The two-piece (double / asymmetric) exponential distribution: back-to-back
exponential tails with scale `scale1` below `location` and `scale2` above it.
Not part of Distributions.jl; provided here so it flows through
[`crps`](@ref)/[`logs`](@ref)/[`dss`](@ref) dispatch.
"""
struct TwoPieceExponential{T <: Real} <: ContinuousUnivariateDistribution
    location::T
    scale1::T  # left-arm scale  (> 0)
    scale2::T  # right-arm scale (> 0)
    function TwoPieceExponential{T}(location::T, scale1::T, scale2::T) where {T <: Real}
        scale1 > zero(T) || throw(DomainError(scale1, "scale1 must be positive"))
        scale2 > zero(T) || throw(DomainError(scale2, "scale2 must be positive"))
        return new{T}(location, scale1, scale2)
    end
end

function TwoPieceExponential(location::Real, scale1::Real, scale2::Real)
    T = promote_type(typeof(location), typeof(scale1), typeof(scale2), Float64)
    return TwoPieceExponential{T}(T(location), T(scale1), T(scale2))
end

Distributions.params(d::TwoPieceExponential) = (d.location, d.scale1, d.scale2)
Distributions.minimum(::TwoPieceExponential) = -Inf
Distributions.maximum(::TwoPieceExponential) = Inf
Distributions.insupport(::TwoPieceExponential, ::Real) = true

function Distributions.pdf(d::TwoPieceExponential, x::Real)
    z = x - d.location
    s = z < 0 ? d.scale1 : d.scale2
    S = d.scale1 + d.scale2
    return exp(-abs(z) / s) / S
end

function Distributions.logpdf(d::TwoPieceExponential, x::Real)
    z = x - d.location
    s = z < 0 ? d.scale1 : d.scale2
    S = d.scale1 + d.scale2
    return -abs(z) / s - log(S)
end

# CDF: F(x) = σ₁/(σ₁+σ₂) * exp((x-ℓ)/σ₁)                      x ≤ ℓ
#             1 - σ₂/(σ₁+σ₂) * exp(-(x-ℓ)/σ₂)                  x > ℓ
function Distributions.cdf(d::TwoPieceExponential, x::Real)
    z = x - d.location
    S = d.scale1 + d.scale2
    if z <= 0
        return d.scale1 / S * exp(z / d.scale1)
    else
        return 1 - d.scale2 / S * exp(-z / d.scale2)
    end
end

function Distributions.quantile(d::TwoPieceExponential, p::Real)
    S = d.scale1 + d.scale2
    p_split = d.scale1 / S
    if p <= p_split
        return d.location + d.scale1 * log(p * S / d.scale1)
    else
        return d.location - d.scale2 * log((1 - p) * S / d.scale2)
    end
end

function Distributions.mean(d::TwoPieceExponential)
    return d.location + d.scale2 - d.scale1
end

function Distributions.var(d::TwoPieceExponential)
    return d.scale1^2 + d.scale2^2
end

function Distributions.rand(rng::AbstractRNG, d::TwoPieceExponential)
    S = d.scale1 + d.scale2
    u = rand(rng)
    p_split = d.scale1 / S
    if u <= p_split
        return d.location + d.scale1 * log(u * S / d.scale1)
    else
        return d.location - d.scale2 * log((1 - u) * S / d.scale2)
    end
end
