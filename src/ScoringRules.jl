"""
    ScoringRules

Proper scoring rules for probabilistic forecasts in Julia.

A port of the R package [`scoringRules`](https://github.com/FK83/scoringRules)
by Alexander I. Jordan, Fabian Krüger, Sebastian Lerch and Sam Allen. See
the package README for attribution and provenance. Distributed under
GPL-2.0-or-later.

Scores follow the *lower-is-better* convention (negative orientation) and are
broadcast-friendly. The three univariate entry points

  * [`crps`](@ref) — continuous ranked probability score
  * [`logs`](@ref) — logarithmic score
  * [`dss`](@ref)  — Dawid–Sebastiani score

dispatch on `Distributions.jl` types for parametric forecasts and on
`AbstractVector`s for simulated (ensemble) forecasts. Multivariate ensemble
forecasts are scored with [`es`](@ref) (energy score), [`vs`](@ref) (variogram
score) and [`mmds`](@ref) (maximum-mean-discrepancy score).

# Example

```@example
using Distributions, ScoringRules
crps(Normal(0, 1), 0.5)
```
"""
module ScoringRules

using Distributions: Distributions, Beta, Binomial, Censored, Continuous,
                     ContinuousUnivariateDistribution, DiscreteUnivariateDistribution,
                     Exponential, Gamma, GeneralizedExtremeValue, GeneralizedPareto,
                     Hypergeometric, Laplace, LogLogistic, LogNormal,
                     Logistic, MixtureModel, NegativeBinomial, Normal, Poisson, TDist,
                     Truncated, Uniform, Univariate, UnivariateDistribution,
                     ccdf, cdf, components, dof, location, logpdf, params, pdf, probs,
                     scale, shape
using SpecialFunctions: erfc, gamma, digamma, beta_inc, logbeta,
                        besseli, expinti, gamma_inc
using LogExpFunctions: logistic
using HypergeometricFunctions: _₂F₁
using QuadGK: quadgk
using Statistics: mean, var, std, quantile
using LinearAlgebra: norm
using Random: AbstractRNG
using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF, TYPEDFIELDS, DOCSTRING, @template

# Generic scoring-rule interface (parametric + ensemble via dispatch)
export crps, logs, dss, dss_moments
# Multivariate ensemble scores
export es, vs, mmds
# Weighted (threshold- and outcome-weighted) ensemble scores
export twcrps, owcrps, twes, owes, twvs, owvs, twmmds, owmmds
# Quantile / interval scores and the ranked probability score
export quantile_score, interval_score, rps
# Distribution types provided here (not in Distributions.jl)
export LogLaplace, TwoPieceNormal, TwoPieceExponential

include("docstrings.jl")
include("utils.jl")
include("generics.jl")

# Extra distribution types, so they flow through crps/logs/dss dispatch
# (log-logistic is Distributions.LogLogistic; only these are missing upstream)
include("distributions/loglaplace.jl")
include("distributions/twopiece.jl")

# Closed-form CRPS per distribution family
include("crps/normal.jl")
include("crps/logistic.jl")
include("crps/student.jl")
include("crps/laplace.jl")
include("crps/exponential.jl")
include("crps/gamma.jl")
include("crps/beta.jl")
include("crps/uniform.jl")
include("crps/lognormal.jl")
include("crps/extremes.jl")
include("crps/mixture.jl")
include("crps/discrete.jl")
include("crps/loglogistic.jl")
include("crps/loglaplace.jl")
include("crps/twopiece.jl")

# Simulated / ensemble forecasts and sample-based scores
include("sample/univariate.jl")
include("sample/multivariate.jl")
include("sample/weighted.jl")
include("sample/quantiles.jl")
include("sample/rps.jl")

end # module ScoringRules
