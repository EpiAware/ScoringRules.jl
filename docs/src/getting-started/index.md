# [Getting started](@id getting-started)

ScoringRules.jl evaluates probabilistic forecasts using proper scoring rules.
This page walks through the most common workflow — scoring a parametric or
ensemble forecast against an observation — and points to the deeper guides.

## Installation

```julia
using Pkg
Pkg.add("ScoringRules")
```

## A first example

```@example gs
using ScoringRules, Distributions

# Score a Normal forecast against an observation
d = Normal(0.0, 1.0)
y = 0.8

crps(d, y)   # continuous ranked probability score
```

```@example gs
logs(d, y)   # logarithmic score (= −log-likelihood)
```

```@example gs
dss(d, y)    # Dawid–Sebastiani score
```

All three functions follow the **lower-is-better** (negative-orientation)
convention throughout: a smaller score indicates a better forecast.

## Ensemble forecasts

When a distributional forecast is not available but simulation draws are, pass
the sample vector directly:

```@example gs
draws = randn(500)         # 500 samples from the forecast
crps(draws, y)
```

```@example gs
dss(draws, y)
```

## Broadcasting over many forecasts

Use Julia's dot syntax to score a vector of forecasts against a matching vector
of observations:

```@example gs
ds = [Normal(0.0, 1.0), Normal(1.0, 2.0), Normal(-0.5, 0.5)]
ys = [0.8, 1.2, -0.3]

crps.(ds, ys)
```

```@example gs
mean(crps.(ds, ys))   # mean score over the evaluation set
```

## Count and ordinal forecasts

The same interface handles discrete distributions:

```@example gs
logs(Poisson(3.0), 2)
```

```@example gs
crps(NegativeBinomial(5, 0.4), 3)
```

## Learning more

- [Forecast input modes](@ref input-modes) — parametric, ensemble, moment-based, and quantile inputs.
- [Scoring rules reference](@ref scoring-rules-reference) — multivariate, weighted, quantile, and ordinal scores.
- [Supported distributions](@ref supported-distributions) — every family with a closed-form CRPS.
- [Differences from R](@ref differences-from-r) — known divergences from the R `scoringRules` package.
- [Public API](@ref public-api) — complete function reference.

## Attribution

ScoringRules.jl is a Julia port of the R package
[`scoringRules`](https://github.com/FK83/scoringRules) by Alexander I. Jordan,
Fabian Krüger, Sebastian Lerch and Sam Allen. The **initial port was generated
by a large language model (Claude) under human guidance**. See
[`NOTICE.md`](https://github.com/EpiAware/ScoringRules.jl/blob/main/NOTICE.md)
for full attribution and licence details.
