## 1.0.0

First working version: a port of the R
[`scoringRules`](https://github.com/FK83/scoringRules) package (Jordan, Krüger,
Lerch and Allen). See the README for attribution and provenance.

- Three univariate scores via multiple dispatch: `crps`, `logs`, `dss`. `logs`
  works for any `Distributions.jl` univariate type. So does `dss`: where
  `Distributions.jl` has no analytic variance for a truncated or censored base,
  the first two moments are taken by quadrature. `crps` has a closed form for
  each supported family; otherwise it falls back to quadrature for a continuous
  forecast and to exact integration of the step CDF for a discrete one, which
  needs only `cdf` and so reaches families with no `quantile` method.
- Closed-form CRPS for the normal (plus truncated, censored and generalised
  truncated/censored), logistic, Student-t, Laplace, exponential, gamma, beta,
  uniform, log-normal, log-logistic, log-Laplace, GEV, GPD, two-piece normal,
  two-piece exponential and mixture-of-normals families, and for the Poisson,
  negative binomial, binomial and hypergeometric discrete families.
- `LogLaplace`, `TwoPieceNormal` and `TwoPieceExponential` distribution types
  (the log-logistic uses `Distributions.LogLogistic`).
- `BoundaryMass(dist; lower, upper, lmass, umass)` distribution wrapper for
  forecasts with free point masses at the truncation bounds, with closed-form
  `crps` for normal, logistic and (location–scale) Student-t bases and for the
  point-mass exponential, uniform and generalised Pareto forms; `logs` and
  `dss` also work on the wrapper (R exposes no logs/dss counterparts).
- Sample/ensemble forecasts: `crps` (empirical and kernel-density), `logs`,
  `dss` on vectors; the energy score `es`, variogram score `vs` and
  maximum-mean-discrepancy score `mmds` on multivariate ensembles. The sample
  `crps` (`method = :edf`) and `dss` take optional member weights `w`, matching
  R's `crps_sample` and `dss_sample`. All three multivariate scores also take
  optional member weights `w`, matching R. The pairwise `d × d` weight matrix of
  `vs` is named `w_vs` (as in R); it was briefly exposed as `w`, so any early
  code passing a matrix via `w` must switch to `w_vs`. Unlike R's `vs_sample`,
  which ignores `w_vs` when member weights are given, `vs` honours both together.
- Threshold- and outcome-weighted scores (`twcrps`, `owcrps`, `twes`, `owes`,
  `twvs`, `owvs`, `twmmds`, `owmmds`), all with optional member weights `w`:
  the tw\* scores weight the chained ensemble, and in the ow\* scores the
  outcome weights multiply the member weights, as in R. `twvs` and `owvs` also
  take the pairwise `d × d` weight matrix `w_vs`; unlike R, which drops `w_vs`
  whenever member or outcome weights are active, both scores honour it
  alongside `w`.
- Censored and conditional likelihood scores (`clogs`) of Diks et al. (2011).
- Quantile and interval scores (`quantile_score`, `interval_score`), the ranked
  probability score `rps`, and the moment-based `dss_moments` and `ess_moments`
  (error-spread score of Christensen, Moroz and Palmer 2015).
- `quantile_score` over a vector of quantile levels returns the mean pinball loss. Pass `reduce = identity` for one score per level, or another reduction such as `sum`.
- Smooth weight and chaining functions for the weighted scores,
  `weight_function(name; mu, sigma)` and `chaining_function(name; mu, sigma)`,
  covering the six shapes of R's `get_weight_func`: `:norm_cdf`, `:norm_surv`,
  `:norm_pdf`, `:logis_cdf`, `:logis_surv` and `:logis_pdf`. Each chaining
  function is the antiderivative of the weight of the same name, which is what
  keeps the threshold-weighted score proper. Scalar `mu`/`sigma` give a
  univariate function; length-`d` vectors give the product weight and the
  element-wise chaining function of a `d`-dimensional forecast, and as in R the
  logistic shapes are univariate only.
- Every scoring function is checked against R `scoringRules` 1.1.3 in the test
  suite. The reference grids reach the parameters where closed forms
  misbehave: a Poisson mean past 350, a negative binomial size past 40, a
  normal or logistic scale of zero, an extreme-value shape within `1e-3` of the
  Gumbel limit, a Student-t truncation bound 1000 standard units into the tail,
  and observations outside the support.
- Scores that were wrong or threw outside those grids now hold up. The
  negative binomial CRPS takes its hypergeometric factor through an Euler
  integral rather than a Gauss series (which lost digits from `size ≈ 40` and
  returned `NaN` from 100); the Poisson CRPS scales its Bessel terms (the
  unscaled ones overflowed from `λ ≈ 351`); the generic discrete CRPS
  integrates the whole step CDF, including the half-infinite pieces outside the
  support, and needs only `cdf`, so it no longer flattens out below the support
  or throws on a `Skellam`; the logistic CRPS stays finite past `z ≈ -745`; the
  Student-t CDF is oriented to its tail, so a truncation bound far out no
  longer divides by an underflowed probability; `crps(Normal(μ, 0))` returns the
  absolute error; the GEV moments come from a series that holds through the
  Gumbel limit, where the stock variance was out by a third at `shape = 1e-8`;
  the extreme-value shape threshold sits at `√eps` and the GPD survival goes
  through `log1p`; and `dss` falls back to numerical moments for the truncated
  and censored bases where `Distributions.jl` has no analytic variance, which
  used to throw a `MethodError`. The continuous quadrature fallback now splits
  the integral at the forecast's own quantiles, so a narrow band of mass or a
  jump in the CDF cannot be missed; a boundary-mass exponential of scale 0.057
  at location 1.985 was 1.8e-5 relative low whatever tolerance was asked for.
- Input validation to match R's: `interval_score` rejects a reversed interval,
  `rps` rejects a vector that is negative or does not sum to one, and the
  ensemble scores reject an empty ensemble. Passing `bw` to `crps(dat, y;
  method = :edf)` warns rather than being dropped in silence.
- `es`, `mmds` and `vs` no longer allocate a temporary per member pair, which
  at `d = 50`, `m = 3000` takes `es` from 2061 MiB and 594 ms per call to
  0.02 MiB and 127 ms. `vs` with member weights now computes the same number in
  one pass over the ensemble instead of `m²` kernel evaluations, 14.7 s to
  3.3 ms at `d = 10`, `m = 2000`; `owvs` goes through the same path.
- `dss` returns `NaN` whenever the forecast variance is not finite and
  positive (for example t with `df ≤ 2`, GEV/GPD with `shape ≥ 1/2`,
  log-Laplace with `σ ≥ 1/2`, log-logistic with `β ≤ 2`), matching R.
- `owcrps` returns `NaN` when no ensemble member has positive weight, matching
  R and the multivariate outcome-weighted scores.
- The sample quantile helper behind `quantile_score(dat, y; ...)` and
  `interval_score(dat, y; ...)` mirrors the index fuzz of R's
  `stats::quantile`, so quantile levels whose `n * p` lands just below an
  integer in floating point (e.g. `0.5 * (1 - 0.9)` at `n = 500`) pick the
  same order statistic as R.
- Closed-form CRPS is automatic-differentiation-safe for the Student-t, beta,
  log-logistic, gamma, GEV and Poisson families, whose CDFs route through
  `EpiAwareADTools` AD-safe primitives so gradients propagate dual numbers
  instead of hitting the non-differentiable `beta_inc`/`gamma_inc`. The
  per-backend AD test matrix (ForwardDiff, ReverseDiff, Enzyme, Mooncake)
  exercises real score gradients over these families.

This file tracks notes for major releases and significant milestones; GitHub
Releases (auto-generated from merged PRs) cover every release in between.
