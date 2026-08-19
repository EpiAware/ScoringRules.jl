# [Supported distributions](@id supported-distributions)

ScoringRules.jl dispatches on
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl) types.
This page lists which families have a **closed-form CRPS** and which
distribution types are provided by this package itself.

## Logarithmic and Dawid–Sebastiani scores

`logs` and `dss` work for **any** `UnivariateDistribution` in Distributions.jl.
`logs` evaluates `-logpdf(d, y)` directly; `dss` needs only `mean(d)` and
`var(d)`. No closed-form CRPS is required for these two scores.

## Families with a closed-form CRPS

The table below lists every distribution family for which `crps` uses a
closed-form expression rather than numerical quadrature or summation. The
column "Restrictions" notes parameter constraints under which the formula is
valid; outside these constraints the function returns `NaN`.

### Continuous families from Distributions.jl

| Family | Distributions.jl constructor | Restrictions |
|:---|:---|:---|
| Normal | `Normal(μ, σ)` | — |
| Truncated Normal | `truncated(Normal(μ, σ); lower=l, upper=u)` | — |
| Censored Normal | `censored(Normal(μ, σ); lower=l, upper=u)` | — |
| Logistic | `Logistic(μ, θ)` | — |
| Truncated Logistic | `truncated(Logistic(μ, θ); lower=l, upper=u)` | — |
| Censored Logistic | `censored(Logistic(μ, θ); lower=l, upper=u)` | — |
| Student-t | `LocationScale(μ, σ, TDist(ν))` | — |
| Truncated Student-t | `truncated(LocationScale(μ, σ, TDist(ν)); ...)` | — |
| Censored Student-t | `censored(LocationScale(μ, σ, TDist(ν)); ...)` | — |
| Laplace | `Laplace(μ, θ)` | — |
| Exponential | `Exponential(θ)` | — |
| Gamma | `Gamma(α, θ)` | α > 0, θ > 0 |
| Beta | `Beta(α, β)` | — |
| Uniform | `Uniform(a, b)` | — |
| Log-Normal | `LogNormal(μ, σ)` | — |
| Log-Logistic (Fisk) | `LogLogistic(α, β)` | β > 1 (i.e. scale < 1) |
| Generalised Extreme Value | `GeneralizedExtremeValue(μ, σ, ξ)` | ξ < 1 |
| Generalised Pareto | `GeneralizedPareto(μ, σ, ξ)` | ξ < 1 |
| Normal mixture | `MixtureModel(Normal, [(μ₁,σ₁), ...], w)` | — |

### Discrete families from Distributions.jl

Discrete distributions use closed-form expressions or exact finite sums over
the support (Poisson and Negative Binomial use a closed-form special-function
representation; Binomial and Hypergeometric use a finite sum).

| Family | Constructor |
|:---|:---|
| Poisson | `Poisson(λ)` |
| Negative Binomial | `NegativeBinomial(r, p)` |
| Binomial | `Binomial(n, p)` |
| Hypergeometric | `Hypergeometric(s, f, n)` |

### Distribution types provided by ScoringRules.jl

These types are not in Distributions.jl and are exported by ScoringRules.jl
directly. They behave as standard `ContinuousUnivariateDistribution` subtypes
and support `pdf`, `logpdf`, `cdf`, `quantile`, `mean`, `var`, and `rand`.

| Type | Parameters | CRPS formula |
|:---|:---|:---|
| `LogLaplace(μ, σ)` | μ: log-scale location, σ: log-scale scale (σ ∈ (0,1)) | closed form, requires σ < 1 |
| `TwoPieceNormal(loc, σ₁, σ₂)` | location, left-arm scale, right-arm scale | via generalised truncated/censored Normal |
| `TwoPieceExponential(loc, σ₁, σ₂)` | location, left-arm scale, right-arm scale | via exponential CRPS |
| `BoundaryMass(dist; lower, upper, lmass, umass)` | base distribution, bounds, boundary point masses | see below |

`LogLogistic` is provided by Distributions.jl (as `LogLogistic(α, β)`, the
Fisk distribution); `crps` for it uses the closed form from R's
`scores_llogis.R`.

### Point masses at the truncation bounds

`BoundaryMass(dist; lower, upper, lmass, umass)` truncates a continuous
distribution `dist` to `[lower, upper]` and places free point masses `lmass`
and `umass` at the bounds, with the interior renormalised to carry the
remaining probability. This is the "generalised truncated/censored"
construction of R `scoringRules` (`crps_gtcnorm` and friends): zero masses
recover `truncated(dist; ...)` and tail-probability masses recover
`censored(dist; ...)`, with anything in between available too. The bounds
default to the support of `dist`, so e.g.
`BoundaryMass(Exponential(θ); lmass = 0.1)` places its mass at zero (R's
`crps_expM`).

Closed-form CRPS is used for these bases:

| Base | R counterpart | Conditions |
|:---|:---|:---|
| `Normal` | `crps_gtcnorm` | — |
| `Logistic` | `crps_gtclogis` | — |
| `TDist` / `μ + σ * TDist(ν)` | `crps_gtct` | ν > 1 |
| `Uniform` | `crps_unif` with `lmass`/`umass` | masses at the interval endpoints |
| `Exponential` | `crps_expM` | mass at the lower end, no upper bound |
| `GeneralizedPareto` | `crps_gpd` with `mass` | mass at the location, no upper bound, ξ < 1 |

Any other base (or an unrouted bound/mass combination) falls back to the
generic quadrature over the wrapper's CDF. `logs` and `dss` work through the
generic density and moment methods; R has no logs/dss counterparts for these
forms.

## Quadrature fallback

For any continuous distribution not in the table above, `crps` falls back to
adaptive Gauss–Kronrod quadrature (via
[QuadGK.jl](https://github.com/JuliaMath/QuadGK.jl)). The fallback is
correct but slower and less numerically precise for some distributions. If you
encounter a distribution family that should have a closed form but does not,
please open an issue.

## Example: using package-provided types

```@example dists
using ScoringRules, Distributions

d_ll  = LogLaplace(0.0, 0.5)
crps(d_ll, 1.5)
```

```@example dists
d_tp = TwoPieceNormal(0.0, 1.0, 2.0)
crps(d_tp, 0.5)
```

```@example dists
d_tpe = TwoPieceExponential(0.0, 1.0, 2.0)
crps(d_tpe, 1.0)
```

```@example dists
# Log-logistic from Distributions.jl
d_llog = LogLogistic(1.0, 2.0)
crps(d_llog, 1.5)
```
