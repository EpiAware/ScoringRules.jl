# Small numerical helpers shared across scoring-rule implementations.

# Standard normal pdf and cdf, written out to avoid constructing a `Normal`
# object in hot inner loops.
@inline _norm_pdf(x::Real) = exp(-x^2 / 2) / sqrt(2 * oftype(float(x), pi))
@inline _norm_cdf(x::Real) = erfc(-x / sqrt(oftype(float(x), 2))) / 2
@inline _norm_logpdf(x::Real) = -(x^2 + log(2 * oftype(float(x), pi))) / 2

# Below this |shape| the general GEV closed forms, and the GEV variance, are
# less accurate than their Gumbel limits. The two error sources cross near
# √eps: the general CRPS cancels two terms of size 1/ξ, losing a relative
# `eps/ξ`, while the Gumbel limit is wrong by O(ξ). The worst relative error
# either side of the threshold is then around 3·10⁻⁸, against the 7·10⁻⁵ that
# a threshold of 10⁻¹² leaves (R scoringRules is out by 8·10⁻⁵ here).
const _SHAPE_ATOL = sqrt(eps(Float64))

# Standard normal quantile function. `Φ⁻¹(p) = −√2·erfc⁻¹(2p)`, which keeps
# full relative accuracy in both tails and uses only public SpecialFunctions
# API (`Distributions.norminvcdf` is neither exported nor public).
@inline _norm_invcdf(p::Real) = -sqrt(oftype(float(p), 2)) * erfcinv(2 * p)
