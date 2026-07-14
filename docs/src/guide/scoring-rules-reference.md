# [Scoring rules reference](@id scoring-rules-reference)

This page covers the multivariate ensemble scores, the weighted (threshold- and
outcome-weighted) extensions, the quantile and interval scores, and the ranked
probability score. All functions follow the **lower-is-better** convention.

For the three main univariate scores (`crps`, `logs`, `dss`) see
[Getting started](@ref getting-started) and
[Forecast input modes](@ref input-modes).

## Multivariate ensemble scores

These scores operate on a `d × m` matrix `X` where each column is one ensemble
member, and a length-`d` observation vector `y`.

### Energy score

```math
\mathrm{ES}(F, y) = \mathbb{E}\|X - y\| - \tfrac{1}{2}\mathbb{E}\|X - X'\|
```

where ``X, X'`` are independent draws from the forecast ``F`` and
``\|\cdot\|`` is the Euclidean norm. The sample approximation is

```math
\widehat{\mathrm{ES}} =
  \sum_i w_i \|X_i - y\| - \tfrac{1}{2}\sum_{i,j} w_i w_j \|X_i - X_j\|
```

```@example ref
using ScoringRules, Distributions

d, m = 4, 500
X = randn(d, m)
y = randn(d)

es(X, y)
```

```@example ref
# Per-member weights (normalised internally)
w = rand(m) .+ 0.1
es(X, y; w = w)
```

### Variogram score

The variogram score of order ``p`` penalises differences in the variogram
structure between the forecast ensemble and the observation:

```math
\mathrm{VS}_p(F, y) = \sum_{k,l} w_{kl}
  \bigl(|y_k - y_l|^p - \overline{|X_{k,\cdot} - X_{l,\cdot}|^p}\bigr)^2
```

The default order is ``p = 0.5``; a ``d \times d`` non-negative symmetric
weight matrix ``w_{kl}`` can be supplied.

```@example ref
vs(X, y)            # default order p = 0.5
```

```@example ref
vs(X, y; p = 1.0)   # order p = 1
```

### MMD score

The maximum-mean-discrepancy score uses a Gaussian kernel ``k(x, z) = \exp(-\tfrac{1}{2}\|x-z\|^2)``:

```math
\mathrm{MMDS}(F, y) = \tfrac{1}{2}\sum_{i,j} w_i w_j k(X_i, X_j) - \sum_i w_i k(X_i, y)
```

```@example ref
mmds(X, y)
```

## Weighted scores

Weighted scoring rules let you emphasise a particular region of the outcome
space. Two weighting mechanisms are available for each score.

**Threshold-weighted** (`tw*`) scores apply a *chaining function* ``v``
to the observations and ensemble members before computing the standard score.
The default chaining function is ``v(z) = \mathrm{clamp}(z, a, b)``, which
focuses attention on outcomes in the interval ``(a, b)``.

**Outcome-weighted** (`ow*`) scores instead assign each ensemble member a
*weight function* value ``w(x_i)``. The default weight function is the
indicator ``w(z) = \mathbf{1}\{a < z < b\}``.

### Threshold- and outcome-weighted CRPS

```@example ref
dat = randn(500)
y1d = 0.5

# Score outcomes above 1.0 more heavily
twcrps(dat, y1d; a = 1.0)
```

```@example ref
owcrps(dat, y1d; a = 1.0)
```

```@example ref
# Custom chaining function
twcrps(dat, y1d; chain_func = z -> max(z, 0.0))
```

### Threshold- and outcome-weighted energy score

```@example ref
# Focus on outcomes above 0 in all dimensions
twes(X, y; a = 0.0)
```

```@example ref
owes(X, y; a = 0.0)
```

### Threshold- and outcome-weighted variogram score

```@example ref
twvs(X, y; a = 0.0)
```

```@example ref
owvs(X, y; a = 0.0)
```

### Threshold- and outcome-weighted MMD score

```@example ref
twmmds(X, y; a = 0.0)
```

```@example ref
owmmds(X, y; a = 0.0)
```

For multivariate weighted scores, `a` and `b` can be scalars (broadcast to
all dimensions) or length-`d` vectors. A custom `chain_func` or `weight_func`
can be supplied when the default interval-based functions do not suit.

## Quantile and interval scores

### Quantile score (pinball loss)

The quantile score at level ``\alpha`` with forecast ``q`` and observation ``y`` is

```math
\mathrm{QS}_\alpha(q, y) = \bigl(\mathbf{1}\{y < q\} - \alpha\bigr)(q - y)
```

which equals ``(1-\alpha)(q-y)`` when ``y < q`` and ``\alpha(y-q)`` when
``y \ge q``.

```@example ref
levels = [0.1, 0.5, 0.9]
qs     = [-1.28, 0.0, 1.28]  # standard-normal quantiles
quantile_score(levels, qs, 1.0)
```

```@example ref
mean(quantile_score(levels, qs, 1.0))   # mean quantile score
```

### Interval score

The interval score for a ``(1-\alpha)`` central prediction interval
``[\ell, u]`` is

```math
\mathrm{IS}_\alpha(\ell, u, y) = (u - \ell)
  + \frac{2}{\alpha}(\ell - y)\,\mathbf{1}\{y < \ell\}
  + \frac{2}{\alpha}(y - u)\,\mathbf{1}\{y > u\}
```

In this package the `level` argument is the nominal *coverage* ``1 - \alpha``
(e.g. `level = 0.9` for a 90% interval):

```@example ref
# 90% prediction interval for N(0,1): roughly (−1.645, 1.645)
interval_score(-1.645, 1.645, 0.5, 0.9)
```

## Ranked probability score

The ranked probability score (Epstein 1969) is for categorical / ordinal
forecasts over ``K \ge 2`` ordered categories. Given a vector of forecast
probabilities ``p = (p_1, \ldots, p_K)`` and an observed category
``y \in \{1, \ldots, K\}``:

```math
\mathrm{RPS}(p, y) = \sum_{k=1}^{K} (P_k - \mathbf{1}\{y \le k\})^2,
\quad P_k = \sum_{j=1}^{k} p_j
```

```@example ref
p = [0.2, 0.5, 0.3]   # probabilities over 3 ordered categories
rps(p, 2)              # category 2 was observed
```

```@example ref
rps(p, 1)
```

## References

- Jordan, A., Krüger, F., & Lerch, S. (2019). Evaluating Probabilistic Forecasts with scoringRules. *Journal of Statistical Software*, 90(12), 1–37. [doi:10.18637/jss.v090.i12](https://doi.org/10.18637/jss.v090.i12)
- Allen, S. (2024). Weighted scoringRules: Emphasizing Particular Outcomes When Evaluating Probabilistic Forecasts. *Journal of Statistical Software*, 110(8), 1–26. [doi:10.18637/jss.v110.i08](https://doi.org/10.18637/jss.v110.i08)
- Gneiting, T. & Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *JASA*, 102, 359–378.
- Scheuerer, M. & Hamill, T. M. (2015). Variogram-based proper scoring rules for probabilistic forecasts of multivariate quantities. *Monthly Weather Review*, 143, 1321–1334.
- Epstein, E. S. (1969). A scoring system for probability forecasts of ranked categories. *Journal of Applied Meteorology and Climatology*, 8, 985–987.
