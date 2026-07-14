"""
    ScoringRules

Proper scoring rules for probabilistic forecasts in Julia.

A port of the R package [`scoringRules`](https://github.com/FK83/scoringRules)
by Alexander I. Jordan, Fabian Krüger, Sebastian Lerch and Sam Allen. See
`NOTICE.md` for attribution and provenance. Distributed under GPL-2.0-or-later.

Scores follow the *lower-is-better* convention (negative orientation) and are
broadcast-friendly. The three univariate entry points

  * [`crps`](@ref) — continuous ranked probability score
  * [`logs`](@ref) — logarithmic score
  * [`dss`](@ref)  — Dawid–Sebastiani score

dispatch on `Distributions.jl` types for parametric forecasts and on
`AbstractVector`s for simulated (ensemble) forecasts. Multivariate ensemble
forecasts are scored with [`es`](@ref) (energy score), [`vs`](@ref) (variogram
score) and [`mmds`](@ref) (maximum-mean-discrepancy score).
"""
module ScoringRules

using Distributions
using Distributions: UnivariateDistribution, ContinuousUnivariateDistribution,
                     DiscreteUnivariateDistribution, Truncated, Censored
using SpecialFunctions: erf, erfc, gamma, loggamma, digamma, beta_inc, logbeta,
                        besseli, expinti, gamma_inc
using LogExpFunctions: logistic
using HypergeometricFunctions: _₂F₁
using QuadGK: quadgk
using Statistics: mean, var, std
using LinearAlgebra: norm
using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF, TYPEDFIELDS, DOCSTRING, @template

# Generic scoring-rule interface (parametric + ensemble via dispatch)
export crps, logs, dss

include("docstrings.jl")
include("utils.jl")
include("generics.jl")

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

end # module ScoringRules
