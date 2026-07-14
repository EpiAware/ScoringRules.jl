## Unreleased

First working version: a port of the R
[`scoringRules`](https://github.com/FK83/scoringRules) package (Jordan, Krüger,
Lerch and Allen). See `NOTICE.md` for attribution and provenance.

- Three univariate scores via multiple dispatch: `crps`, `logs`, `dss`. `logs`
  and `dss` work for any `Distributions.jl` univariate type; `crps` has a
  closed form for each supported family and a quadrature fallback otherwise.
- Closed-form CRPS for the normal (plus truncated, censored and generalised
  truncated/censored), logistic, Student-t, Laplace, exponential, gamma, beta,
  uniform, log-normal, log-logistic, log-Laplace, GEV, GPD, two-piece normal,
  two-piece exponential and mixture-of-normals families, and for the Poisson,
  negative binomial, binomial and hypergeometric discrete families.
- `LogLaplace`, `TwoPieceNormal` and `TwoPieceExponential` distribution types
  (the log-logistic uses `Distributions.LogLogistic`).
- Sample/ensemble forecasts: `crps` (empirical and kernel-density), `logs`,
  `dss` on vectors; the energy score `es`, variogram score `vs` and
  maximum-mean-discrepancy score `mmds` on multivariate ensembles.
- Threshold- and outcome-weighted scores (`twcrps`, `owcrps`, `twes`, `owes`,
  `twvs`, `owvs`, `twmmds`, `owmmds`).
- Quantile and interval scores (`quantile_score`, `interval_score`), the ranked
  probability score `rps`, and the moment-based `dss_moments`.
- Every scoring function is checked against R `scoringRules` 1.1.3 in the test
  suite.

This file tracks notes for major releases and significant milestones; GitHub
Releases (auto-generated from merged PRs) cover every release in between.
