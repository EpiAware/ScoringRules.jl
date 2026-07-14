# [Forecast input modes](@id input-modes)

ScoringRules.jl accepts four kinds of forecast input. The same scoring
function names (`crps`, `logs`, `dss`) dispatch on the type of the first
argument, so you rarely need to think about which method you are calling.

## Parametric forecasts (Distributions.jl)

Pass any `UnivariateDistribution` from
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl) as the
first argument. The package uses a closed-form CRPS where one exists; it falls
back to adaptive quadrature otherwise.

```@example modes
using ScoringRules, Distributions

crps(Normal(1.0, 2.0), 0.5)
```

```@example modes
crps(Gamma(2.0, 1.5), 3.0)
```

```@example modes
logs(LogNormal(0.0, 0.5), 1.2)
```

```@example modes
dss(StudentT(5.0), 1.0)
```

Truncated and censored variants from Distributions.jl are also supported:

```@example modes
d_trunc = truncated(Normal(0.0, 1.0); lower=0.0)
crps(d_trunc, 0.7)
```

```@example modes
d_cens = Distributions.censored(Normal(0.0, 1.0); lower=0.0)
crps(d_cens, 0.7)
```

## Ensemble / sample forecasts

When a parametric form is not available, pass a vector of draws. The CRPS is
computed via the empirical distribution function (EDF) approximation by
default; a kernel density estimate is available via `method = :kde`.

```@example modes
draws = randn(1000)

crps(draws, 0.5)
```

```@example modes
crps(draws, 0.5; method = :kde)   # Gaussian KDE with Silverman bandwidth
```

```@example modes
logs(draws, 0.5)   # always uses KDE
```

```@example modes
dss(draws, 0.5)    # uses sample mean and population variance
```

### Weighted ensemble members

Importance weights can be passed to `crps` with the EDF method:

```@example modes
w = abs.(randn(1000)) .+ 0.01   # arbitrary positive weights (normalised internally)
crps(draws, 0.5; w = w)
```

## Moment-based forecasts

When you know only the mean and variance of the forecast — not its full
distribution — use `dss_moments`. The Dawid–Sebastiani score depends on the
forecast through its first two moments alone, so no distribution needs to be
specified.

```@example modes
dss_moments(0.5, 1.0, 4.0)   # observation y=0.5, mean=1.0, variance=4.0
```

This is useful when forecasts arrive as published summary statistics rather
than as distributional objects.

## Quantile forecasts

For quantile-format forecasts supply vectors of quantile levels and the
corresponding quantile values:

```@example modes
levels = [0.1, 0.25, 0.5, 0.75, 0.9]
qs     = [-1.28, -0.67, 0.0, 0.67, 1.28]   # standard-normal quantiles

scores = quantile_score(levels, qs, 0.5)
```

```@example modes
mean(scores)   # mean quantile score
```

For a single quantile level from an ensemble:

```@example modes
quantile_score(draws, 0.5; alpha = 0.9)   # 90th-percentile score
```

Prediction intervals can be scored with `interval_score`:

```@example modes
# 90% prediction interval for N(0,1): approximately (−1.645, 1.645)
interval_score(-1.645, 1.645, 0.5, 0.9)
```

```@example modes
interval_score(draws, 0.5; level = 0.9)   # same, estimated from the ensemble
```

## Multivariate ensembles

For a `d`-dimensional ensemble of `m` members, represent the forecast as a
`d × m` matrix (each column is one member) and the observation as a length-`d`
vector:

```@example modes
d, m = 3, 200
X = randn(d, m)   # 3-dimensional, 200-member ensemble
y = [0.1, -0.2, 0.5]

es(X, y)    # energy score
```

```@example modes
vs(X, y)    # variogram score (default order p = 0.5)
```

```@example modes
mmds(X, y)  # MMD score (Gaussian kernel, σ = 1)
```

See [Scoring rules reference](@ref scoring-rules-reference) for the weighted
variants of these scores.
