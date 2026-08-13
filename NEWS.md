## Unreleased

First working version: a port of the R
[`scoringRules`](https://github.com/FK83/scoringRules) package (Jordan, Krüger,
Lerch and Allen). See the README for attribution and provenance.

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
  maximum-mean-discrepancy score `mmds` on multivariate ensembles. All three
  multivariate scores take optional member weights `w`, matching R. The
  pairwise `d × d` weight matrix of `vs` is named `w_vs` (as in R); it was
  briefly exposed as `w`, so any early code passing a matrix via `w` must
  switch to `w_vs`. Unlike R's `vs_sample`, which ignores `w_vs` when member
  weights are given, `vs` honours both together.
- Threshold- and outcome-weighted scores (`twcrps`, `owcrps`, `twes`, `owes`,
  `twvs`, `owvs`, `twmmds`, `owmmds`), all with optional member weights `w`:
  the tw\* scores weight the chained ensemble, and in the ow\* scores the
  outcome weights multiply the member weights, as in R.
- Censored and conditional likelihood scores (`clogs`) of Diks et al. (2011).
- Quantile and interval scores (`quantile_score`, `interval_score`), the ranked
  probability score `rps`, and the moment-based `dss_moments` and `ess_moments`
  (error-spread score of Christensen, Moroz and Palmer 2015).
- Every scoring function is checked against R `scoringRules` 1.1.3 in the test
  suite.
- The sample quantile helper behind `quantile_score(dat, y; ...)` and
  `interval_score(dat, y; ...)` mirrors the index fuzz of R's
  `stats::quantile`, so quantile levels whose `n * p` lands just below an
  integer in floating point (e.g. `0.5 * (1 - 0.9)` at `n = 500`) pick the
  same order statistic as R.

This file tracks notes for major releases and significant milestones; GitHub
Releases (auto-generated from merged PRs) cover every release in between.
