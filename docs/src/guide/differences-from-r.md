# [Differences from R](@id differences-from-r)

ScoringRules.jl is a port of the R package
[`scoringRules`](https://github.com/FK83/scoringRules) (version 1.1.3). In
most cases the two packages produce identical numerical results, but there are
known divergences that users coming from R should be aware of.

## DSS for Log-Logistic

R's `dss_llogis` returns `NaN` in all tested configurations. The root cause is
that R's implementation drops a location-dependent factor when computing the
variance of the log-logistic distribution, making the formula incorrect and
producing numerically degenerate results.

ScoringRules.jl uses the correct variance:

```math
\mathrm{Var}[X] = \alpha^2 \left(\frac{2/\beta}{\sin(2\pi/\beta)} - \left(\frac{\pi/\beta}{\sin(\pi/\beta)}\right)^2\right)
```

via Distributions.jl's `var(LogLogistic(α, β))`, so `dss(LogLogistic(α, β), y)`
returns a finite result wherever the variance exists (requires ``\beta > 2``).
Numerical comparison with R is not possible because R's implementation is
broken for this family.

## GEV CRPS: Gumbel case (shape ≈ 0)

For a `GeneralizedExtremeValue(μ, σ, 0)` distribution (the Gumbel limit,
``\xi \to 0``), the general closed-form expression is numerically unstable.
R's implementation falls back to numerical integration in this limit.

ScoringRules.jl instead evaluates the Gumbel limit analytically using the
exponential integral ``\mathrm{Ei}`` from
[SpecialFunctions.jl](https://github.com/JuliaMath/SpecialFunctions.jl):

```math
\mathrm{CRPS}_\mathrm{Gumbel} = -z - \gamma_E - \log 2 - 2\,\mathrm{Ei}(-e^{-z})
```

where ``z = (y - \mu)/\sigma`` and ``\gamma_E = -\psi(1)`` is the
Euler–Mascheroni constant. This is the same formula as for the general GEV but
taken in the limit ``\xi \to 0``. Numerically it is more accurate than the
integrate fallback because `expinti` is implemented in arbitrary precision.
Results agree with R's numerical integration to within the integration tolerance
in all tested cases.

## Ensemble DSS: population vs sample variance

`dss(dat, y)` for an ensemble `dat` uses the **population** variance
``\hat{\sigma}^2 = \frac{1}{n}\sum_i(x_i - \bar{x})^2``, computed as
`mean(dat.^2) - mean(dat).^2`. This matches R's `dss_sample` / `dss_edf`,
which also uses the biased (population) estimator. Julia's `var` function uses
the ``n-1`` denominator, so passing `var(dat)` directly would not match R.
The implementation avoids `Statistics.var` deliberately.

## Parameter conventions

The following parameter conventions differ from R's function arguments:

| Quantity | R convention | Julia / Distributions.jl |
|:---|:---|:---|
| `LogLogistic` | `locationlog, scalelog` | `LogLogistic(α, β)`: α = exp(locationlog), β = 1/scalelog |
| `GeneralizedPareto` | `location, scale, shape` | `GeneralizedPareto(μ, σ, ξ)` — same order |
| `NegativeBinomial` | `size, prob` (success prob) | `NegativeBinomial(r, p)` — same meaning |
| `Hypergeometric` | `m, n, k` (white, black, draws) | `Hypergeometric(s, f, n)` (successes, failures, draws) — s=m, f=n, n=k |
| `TwoPieceNormal` | `location, scale1, scale2` | `TwoPieceNormal(location, scale1, scale2)` — same |

## Reporting discrepancies

If you find a numerical result that differs from R and is not listed above,
please open an issue on the
[GitHub repository](https://github.com/EpiAware/ScoringRules.jl). The test
suite includes a set of reference values generated from R's `scoringRules`
package, so discrepancies can usually be identified precisely.
