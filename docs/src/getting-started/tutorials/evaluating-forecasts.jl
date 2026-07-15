#src Package-owned light tutorial. Runs in-process as Documenter `@example`
#src blocks, so keep the data small and the code cheap. Registered in
#src docs/docs_config.jl under LIGHT_TUTORIALS and in docs/pages.jl.

md"""
# [Evaluating forecasts end-to-end](@id evaluating-forecasts)

This tutorial walks a full forecast-evaluation workflow: score a single
parametric forecast, score an ensemble (sample) forecast, compare two
competing forecasters, move to a multivariate outcome, and finish by
emphasising particular outcomes with weighted scores.

It reworks the spirit of the *scoringRules* articles by Jordan, Krüger &
Lerch (2019) and Allen (2024) as a self-contained Julia workflow (see
[How to cite](@ref)). The data here is synthetic and generated
in-code; the original articles used real macroeconomic and weather forecasts.
Everything runs from a fixed seed, so the page is reproducible.
"""

md"""
## Setup

We only need ScoringRules, Distributions for the parametric forecasts, and a
seeded RNG for the synthetic draws.
"""

using ScoringRules
using Distributions
using Statistics
using Random

rng = Random.MersenneTwister(1234)

md"""
## One forecast, three scores

Every univariate score is negatively oriented: **lower is better**. Pass a
`Distributions.jl` distribution as the forecast and a scalar observation.
"""

forecast = Normal(0.0, 1.0)
y = 0.7

(crps = crps(forecast, y), logs = logs(forecast, y), dss = dss(forecast, y))

md"""
The CRPS is on the scale of the observation, the log score is the negative
log-density, and the Dawid–Sebastiani score uses only the forecast's first two
moments. A forecast centred closer to the outcome scores lower:
"""

crps(Normal(0.5, 1.0), y), crps(Normal(2.0, 1.0), y)

md"""
## Ensemble (sample) forecasts

When the forecast is a set of simulation draws rather than a closed-form
distribution, pass the vector directly. The CRPS then uses the empirical
distribution function (`:edf`, the default) or a Gaussian kernel estimate
(`:kde`); the log and Dawid–Sebastiani scores follow.
"""

ensemble = rand(rng, Normal(0.0, 1.0), 500)

(edf = crps(ensemble, y),
    kde = crps(ensemble, y; method = :kde),
    logs = logs(ensemble, y),
    dss = dss(ensemble, y))

md"""
## Comparing two forecasters

The point of proper scores is ranking. We draw a batch of observations from a
"truth", then score two competing predictive distributions over the same
observations and compare their mean scores. A calibrated forecaster should win.
"""

truth = Normal(1.0, 2.0)
observations = rand(rng, truth, 200)

calibrated = Normal(1.0, 2.0)          # correct
overconfident = Normal(1.0, 1.0)       # right centre, too sharp

mean_crps_calibrated = mean(crps(calibrated, yi) for yi in observations)
mean_crps_overconfident = mean(crps(overconfident, yi) for yi in observations)

(calibrated = mean_crps_calibrated, overconfident = mean_crps_overconfident)

md"""
The calibrated forecaster earns the lower mean CRPS. To score a whole vector of
forecast/observation pairs at once, broadcast: `crps.(distributions, observations)`
composes with the usual Julia reductions.
"""

md"""
## Multivariate forecasts

For a `d`-dimensional outcome, represent the ensemble as a `d × m` matrix (each
column one member) and the observation as a length-`d` vector. Three multivariate
scores are available: the energy score, the variogram score, and the
maximum-mean-discrepancy score.
"""

d, m = 3, 300
X = rand(rng, Normal(0.0, 1.0), d, m)   # 3 × 300 ensemble, independent components
obs = [0.2, -0.1, 0.4]

(es = es(X, obs), vs = vs(X, obs), mmds = mmds(X, obs))

md"""
## Emphasising particular outcomes

Weighted scores let you concentrate the assessment on the outcomes you care
about, for example the upper tail when large values are what matter. The
threshold-weighted CRPS transforms forecast and observation through a chaining
function; the outcome-weighted CRPS reweights by a region indicator. Both reduce
to the ordinary CRPS when the region is the whole line.
"""

tail_ensemble = rand(rng, Normal(0.0, 1.0), 500)
y_tail = 1.5

(unweighted = crps(tail_ensemble, y_tail),
    threshold_weighted = twcrps(tail_ensemble, y_tail; a = 1.0),
    outcome_weighted = owcrps(tail_ensemble, y_tail; a = 1.0))

md"""
Here `a = 1.0` focuses the score on outcomes above 1. The same `a`/`b` interval
and custom-function conventions carry over to the multivariate weighted scores
(`twes`, `owes`, `twvs`, `owvs`, `twmmds`, `owmmds`).
"""

md"""
## Fitting forecasts by minimising a score

To *fit* a forecast model rather than just evaluate one (optimum-score
estimation, e.g. EMOS), minimise a mean score over training data. Because every
score here is an ordinary Julia function of the forecast parameters, you
differentiate it with any automatic-differentiation backend:

```julia
using ForwardDiff
ForwardDiff.gradient(p -> crps(Normal(p[1], p[2]), y), [0.0, 1.0])
```

This replaces R's hand-coded `gradcrps_*` / `hesscrps_*` family with AD that
works for every distribution and score. See the
[Automatic differentiation backends](@ref ad-backends) tutorial for which
backends are supported and how to configure them.
"""

md"""
## Where to go next

  * [Forecast input modes](@ref input-modes) — every way to pass a forecast.
  * [Scoring rules reference](@ref scoring-rules-reference) — the full list of
    scores and their arguments.
  * [Supported distributions](@ref supported-distributions) — the parametric
    families with closed-form CRPS.

If you use these scores in your work, please cite the original *scoringRules*
authors (see [How to cite](@ref)).
"""
