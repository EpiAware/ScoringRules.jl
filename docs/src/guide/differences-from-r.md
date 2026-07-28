# [Differences from R](@id differences-from-r)

ScoringRules.jl is a port of the R package
[`scoringRules`](https://github.com/FK83/scoringRules) (version 1.1.3). In
most cases the two packages produce identical numerical results, but there are
known divergences that users coming from R should be aware of.

## DSS for Log-Logistic

R's `dss_llogis` has an operator-precedence bug: in the variance computation
`v <- ell^2 * 2*b/sin(2*b) - b^2/sb^2`, the squared location factor `ell^2`
multiplies only the first term. The consequences depend on the location
parameter:

  * `locationlog = 0`: the factor equals one, the bug cancels, and R agrees
    with ScoringRules.jl to machine precision.
  * `locationlog > 0`: R returns wrong finite values (observed relative errors
    up to 166%).
  * `locationlog < 0`: the mis-scaled variance goes negative and R returns
    `NaN`.

ScoringRules.jl uses the correct variance:

```math
\mathrm{Var}[X] = \alpha^2 \left(\frac{2/\beta}{\sin(2\pi/\beta)} - \left(\frac{\pi/\beta}{\sin(\pi/\beta)}\right)^2\right)
```

via Distributions.jl's `var(LogLogistic(α, β))`, so `dss(LogLogistic(α, β), y)`
returns a finite result wherever the variance exists (requires ``\beta > 2``),
verified against a manual computation from the log-logistic moments.

## CRPS for the negative binomial: half-integer `size`

R's `crps_nbinom` returns `-Inf` whenever the `size` parameter is a
half-integer (0.5, 1.5, 2.5, … — all tested values). Its Gaussian
hypergeometric dependency evaluates a gamma function at a pole in exactly
those configurations. ScoringRules.jl's `crps(NegativeBinomial(r, p), y)`
returns the correct value there, matching a brute-force evaluation of
``\sum_k (F(k) - \mathbb{1}\{y \le k\})^2`` to about twelve significant
digits. For all other `size` values the two packages agree to machine
precision.

## CRPS gradients and Hessians

R exports closed-form CRPS derivatives with respect to location and scale
(`gradcrps_*`, `hesscrps_*`) for the normal, logistic and Student's ``t``
families and their truncated/censored variants. ScoringRules.jl provides no
closed-form derivatives; gradients come from automatic differentiation of
`crps`.

Validation against R found three errors in R's closed forms for the ``t``
families. Finite differences of R's *own* CRPS functions confirm each one,
and agree with automatic differentiation of the Julia implementation:

  * `gradcrps_tt` is wrong whenever the observation is not clipped clear of a
    finite truncation bound (sign flips and errors of up to two orders of
    magnitude).
  * `hesscrps_ct` and `hesscrps_tt` omit the ``1/\sigma`` factor in their
    location–scale branch, so every result with `scale ≠ 1` is off by exactly
    that factor.

`gradcrps_norm`, `gradcrps_logis`, `gradcrps_t` and the remaining censored and
truncated variants agree with automatic differentiation of the Julia
implementation to about ``10^{-9}`` or better.

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
