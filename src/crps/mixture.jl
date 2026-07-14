# CRPS for finite mixtures of normal distributions, in closed form.
# Ported from R scoringRules `scores_mixnorm.R` and the C++ helper
# `crpsmixnC` / `auxcrpsC` in `src/mixn.cpp`
# (Jordan, Krüger, Lerch, Allen; GPL-2.0-or-later).
#
# The A(μ, σ) auxiliary function used below is
#   auxcrpsC(m, s) = 2s·φ(m/s) + m·(2Φ(m/s) − 1)
# which is the CRPS of N(0,s²) at m (the `crps_norm` formula at μ=0).

"""
CRPS of a finite Gaussian mixture forecast at observation `y`.

`means` and `sds` are vectors of component means and standard deviations;
`weights` is a vector of non-negative mixture weights (need not sum to 1 —
they are normalised internally, matching the R implementation).
"""
function _crps_mixnorm(y::Real,
                       means::AbstractVector{<:Real},
                       sds::AbstractVector{<:Real},
                       weights::AbstractVector{<:Real})
    N = length(means)
    W     = zero(float(y))
    crps1 = zero(float(y))
    crps2 = zero(float(y))

    for i in 1:N
        wi = weights[i]
        si = sds[i]
        W     += wi
        crps1 += wi * _auxcrps(y - means[i], si)

        # Diagonal (self) contribution: ½ wᵢ · A(0, √2 · sᵢ)
        crps3 = 0.5 * wi * _auxcrps(zero(float(y)), sqrt(oftype(float(y), 2)) * si)

        # Off-diagonal contributions j < i
        si2 = si^2
        for j in 1:(i - 1)
            crps3 += weights[j] * _auxcrps(means[i] - means[j],
                                            sqrt(si2 + sds[j]^2))
        end
        crps2 += wi * crps3
    end

    return (crps1 - crps2 / W) / W
end

# Auxiliary: CRPS of N(0, s²) at m, i.e. E|X − m| − ½E|X − X'| for X,X'~N(0,s²).
# Matches `auxcrpsC` in the R package C++ source.
@inline function _auxcrps(m::Real, s::Real)
    s < 0 && return oftype(float(m), NaN)
    s == 0 && return abs(m)
    ms = m / s
    return 2s * _norm_pdf(ms) + m * (2 * _norm_cdf(ms) - 1)
end

"""
CRPS of a `MixtureModel` of normal components at observation `y`.

Dispatches on `MixtureModel{Univariate,Continuous,<:Normal}`, extracting
component parameters via `components(d)` and `probs(d)`.
"""
function crps(d::MixtureModel{Univariate, Continuous, <:Normal}, y::Real)
    comps   = components(d)
    weights = probs(d)
    means   = [c.μ for c in comps]
    sds     = [c.σ for c in comps]
    return _crps_mixnorm(y, means, sds, weights)
end
