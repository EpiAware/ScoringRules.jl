# [Differences from R](@id differences-from-r)

ScoringRules.jl is a port of the R package
[`scoringRules`](https://github.com/FK83/scoringRules) (version 1.1.3). Most
scores agree to floating-point round-off, as the
[Validation against R](@ref validation-against-r) page recomputes on every docs
build. This page lists every case where they do not, and says which side is
right.

## Where the two packages give different numbers

In each of the cases below the R result has been checked against an independent
reference: brute-force summation over the probability mass function, quadrature
at 300-bit precision, or the analytic limit. Where that reference agrees with
this package, the entry says so.

### Log-logistic DSS

R's `dss_llogis` computes the forecast variance as

```r
v <- ell^2 * 2 * b/sin(2 * b) - b^2/sb^2
```

which is missing a pair of parentheses: `ell^2` should multiply both terms. The
two packages disagree for every log-logistic forecast with a nonzero
`locationlog`, at `scalelog = 0.3`:

| `locationlog` | R variance | correct variance |
|:---|:---|:---|
| -1.0 | -1.0889 | 0.08456 |
| 0.0 | 0.6248 | 0.6248 |
| 1.0 | 13.2877 | 4.61677 |
| 2.0 | 106.8542 | 34.1136 |

Far enough below zero R's expression goes negative and `dss_llogis` returns
`NaN`, which at that `scalelog` starts around `locationlog = -0.19`. Between
there and zero, and everywhere above it, R returns a finite but wrong number.
This package evaluates the parenthesised formula and matches the correct column
at every point.

### Generalised Pareto CRPS near shape 0

R forms `(1 + ξz)^(-1/ξ)` directly. For a small shape the whole of `ξz` is lost
to rounding in the `1 + ξz`, and dividing by `ξ` in the exponent multiplies that
error by `1/ξ`: at `shape = 1e-12` and `y = 0.5`, `crps_gpd` gives 0.2130074
where the exponential limit is 0.2130613. This package takes the same expression
through `log1p`, which is accurate to the last bit or two for every shape.

### Negative binomial CRPS at a half-integer size

The CRPS has a Gauss hypergeometric factor in it. R sums the series for that factor through the `hypergeo` package, which returns `-Inf` for every half-integer size below a threshold that depends on `prob`: sizes 0.5 to 5.5 at `prob = 0.1`, 0.5 to 9.5 at `prob = 0.3`, and 0.5 to 26.5 at `prob = 0.7`. Only the half-integers are affected; quarter-integer and other fractional sizes come out finite and right. This package evaluates the same factor as an Euler integral and reproduces brute-force summation over the probability mass function to 15 digits at moderate sizes, and to 1e-10 at a size of 1e10. Everywhere R returns a finite value it agrees, to 1e-14 at sizes up to 1000.

The closed form subtracts terms that grow like `size/prob` to leave a score that grows like `√size/prob`, and loses digits as the size grows. A nearly degenerate forecast observed at zero loses them the same way, and is summed exactly over its lattice points instead, except at a size of 1e-10 or less with a `prob` of 1e-5 or less, where the tail decays too slowly to sum and the score is `NaN`. Past a size of 1e15 to 1e21, depending on `prob`, the closed form keeps fewer than six digits, and below a `prob` of about 1e-154 one of its terms overflows; at both, this package returns `NaN`. R stops with an error in `hypergeo` at the first and returns `Inf` at the second.

### Truncated-density log score with a far lower bound

R's `ft`, `fnorm` and `flogis` (behind `logs_tt`, `logs_tnorm` and
`logs_tlogis`) normalise a lower-bounded truncation by `1 - p(lower)`, which
keeps no relative accuracy once the retained tail is small, and reaches exactly
zero further out. At `df = 10` with a standardised lower bound of 50, `logs_tt`
is 1.6e-4 relative out, and 6.1e-5 at a bound of 60; `logs_tlogis` at
`location = 0, scale = 2, lower = 100` returns `-Inf` at `y = 101`, where the
truncated density is 0.303. This package reads the upper tail directly and
matches 300-bit quadrature to 1e-13.

### GEV CRPS: Gumbel limit (shape ≈ 0)

As the shape goes to zero the general closed form cancels two terms of size
`1/ξ`, so it loses a relative `eps/ξ`. R falls back to numerical integration
there (and warns that installing the `gsl` package would sharpen it). This
package evaluates the Gumbel limit analytically with the exponential integral
``\mathrm{Ei}`` from
[SpecialFunctions.jl](https://github.com/JuliaMath/SpecialFunctions.jl):

```math
\mathrm{CRPS}_\mathrm{Gumbel} = -z + \gamma_E - \log 2 - 2\,\mathrm{Ei}(-e^{-z})
```

where ``z = (y - \mu)/\sigma`` and ``\gamma_E = -\psi(1)`` is the
Euler–Mascheroni constant. The switch between the two expressions sits at
``|\xi| = \sqrt{\mathrm{eps}}``, where the error of each is around
``3 \times 10^{-8}``. R switches at ``10^{-12}`` instead, which leaves its
general expression to cancel for four more orders of magnitude: just above that
threshold it is out by up to ``5 \times 10^{-4}``. At ``\sqrt{\mathrm{eps}}``
R is on its general branch and accurate.

### GEV moments near the Gumbel limit

`mean` and `var` of a GEV are built from `Γ(1−ξ)` and `Γ(1−2ξ)`, and the
combination the variance needs, `(Γ(1−2ξ) − Γ(1−ξ)²)/ξ²`, cancels to nothing as
`ξ → 0`. Both packages lose accuracy there if the expression is taken at face
value: at `ξ = 1e-5` R's `dss_gev` is 6e-7 out, and at `ξ = 1e-8` the variance
`Distributions.jl` returns is out by a third. This package takes both moments
from the Maclaurin series of `log Γ(1−ξ)` instead, which is accurate to the last
bit or two for `|ξ| ≤ 0.1` and gives the Gumbel limit exactly at `ξ = 0`.

## Where the two packages take different routes to the same number

### Ensemble DSS: population versus sample variance

`dss(dat, y)` for an ensemble `dat` uses the **population** variance
``\hat{\sigma}^2 = \frac{1}{n}\sum_i(x_i - \bar{x})^2``, computed as
`mean(dat.^2) - mean(dat).^2`. This matches R's `dss_sample` / `dss_edf`, which
also uses the biased estimator, so the numbers agree. Julia's `var` uses the
``n-1`` denominator, so passing `var(dat)` by hand would not match R; the
implementation avoids `Statistics.var` deliberately.

## Where the two packages make different design choices

### Pairwise weights alongside member weights in the variogram score

R's `vs_sample` ignores the ``d \times d`` pairwise weight matrix `w_vs`
whenever member weights are given, and `twvs_sample` / `owvs_sample` drop it
whenever member or outcome weights are active. Here `vs`, `twvs` and `owvs`
honour `w_vs` and `w` together, so a call that passes both returns a different
number from R's. With only one of the two the results agree.

### Density at a boundary point mass

`pdf(::BoundaryMass, x)` at a bound that has a point mass returns that mass,
following `Distributions.Censored`. R's `f*` functions return the renormalised
interior density there and warn that the result is not a density: for
`fnorm(-1, location = 0, scale = 0.5, lower = -1, upper = 1.5, lmass = 0.2)` R
gives 0.0885 where `pdf` here gives 0.2. `logs` at such a bound differs
accordingly. Away from the bounds the two agree.

### CRPS gradients and Hessians

R exports closed-form CRPS derivatives with respect to location and scale
(`gradcrps_*`, `hesscrps_*`) for the normal, logistic and Student's ``t``
families and their truncated and censored variants. This package provides no
closed-form derivatives; gradients come from automatic differentiation of
`crps`, which reproduces R's analytic values for the untruncated normal and
logistic exactly (see the
[Validation against R](@ref validation-against-r) page). `gradcrps_tt` is worth
knowing about: it does not agree with central differences of R's own `crps_tt`
at any truncation bound tried (at `df = 3`, `lower = 0`, `upper = 1` it reports
1.509 for the location derivative where the finite difference is -0.014), so it
is not a useful reference to compare an AD gradient against. `gradcrps_ct`
agrees with its own `crps_ct` throughout.

### Ensemble member weights

The sample `crps` (`method = :edf`) and `dss` accept member weights `w`,
matching R's `crps_sample` and `dss_sample`. Member weights must be finite,
non-negative and sum to a positive value, otherwise an error is thrown. R also
errors on missing, infinite and negative weights but returns `NaN` for an
all-zero weight vector.

### Input validation

Three arguments are checked here that R checks too, with the same effect but a different message: a forecast vector in `rps` that is negative or does not sum to one, an empty univariate ensemble, and a prediction interval in `interval_score` whose lower end is above its upper one, which R's `ints_quantiles` rejects. Passing `bw` to `crps(dat, y; method = :edf)` warns, where R prints a message saying the same thing.

Two of the checks have no counterpart in R, or depart from it. `rps` accepts a probability vector whose sum is within `1e-10` of one, or within `K` of its own `eps` where that is larger, where R insists on a single `eps`. A forecast normalised over many categories, or held in Float32, carries rounding in its sum that it cannot avoid. `es` and `mmds` throw on an ensemble with no members, where R's `es_sample` and `mmds_sample` return 0; R's `vs_sample` errors, as `vs` does here.

## Parameter conventions

The following parameter conventions differ from R's function arguments.

| Quantity | R convention | Julia / Distributions.jl |
|:---|:---|:---|
| `Exponential` | `crps_exp(y, rate)` — a **rate** | `Exponential(θ)` — a **scale**, so `θ = 1/rate` |
| `Gamma` | `crps_gamma(y, shape, rate)` or `scale` | `Gamma(α, θ)` — shape and **scale**, so `θ = 1/rate` |
| Student-t | `crps_t(y, df, location, scale)` | `location + scale * TDist(df)` |
| `LogLogistic` | `locationlog, scalelog` | `LogLogistic(α, β)`: α = exp(locationlog), β = 1/scalelog |
| `LogLaplace` | `locationlog, scalelog` | `LogLaplace(locationlog, scalelog)` — same |
| `GeneralizedExtremeValue` | `crps_gev(y, shape, location, scale)` | `GeneralizedExtremeValue(μ, σ, ξ)` |
| `GeneralizedPareto` | `crps_gpd(y, shape, location, scale)` | `GeneralizedPareto(μ, σ, ξ)` |
| `NegativeBinomial` | `size, prob` (success prob) | `NegativeBinomial(r, p)` — same meaning |
| `Hypergeometric` | `m, n, k` (white, black, draws) | `Hypergeometric(s, f, n)` (successes, failures, draws) — s=m, f=n, n=k |
| `TwoPieceNormal` | `crps_2pnorm(y, scale1, scale2, location)` | `TwoPieceNormal(location, scale1, scale2)` |
| `TwoPieceExponential` | `crps_2pexp(y, scale1, scale2, location)` | `TwoPieceExponential(location, scale1, scale2)` |
| generalised truncated/censored | `lower, upper, lmass, umass` on each `crps_gtc*` | `BoundaryMass(dist; lower, upper, lmass, umass)` |
| smooth weights | `get_weight_func(name, mu, sigma, weight)` | [`weight_function`](@ref) and [`chaining_function`](@ref) |

## Reporting discrepancies

If you find a numerical result that differs from R and is not listed above,
please open an issue on the
[GitHub repository](https://github.com/EpiAware/ScoringRules.jl). The test
suite includes a set of reference values generated from R's `scoringRules`
package, so discrepancies can usually be identified precisely.
