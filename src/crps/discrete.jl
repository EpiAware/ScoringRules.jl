# Closed-form CRPS for four discrete families: Poisson, NegativeBinomial,
# Binomial, and Hypergeometric.
#
# Ported from R scoringRules v1.1.3 (GPL-2):
#   scores_pois.R, scores_nbinom.R, scores_binom.R, scores_hyper.R
# by Alexander I. Jordan, Fabian Krüger, Sebastian Lerch and Sam Allen.
#
# Module-level imports required (in addition to those already in ScoringRules.jl):
#   using SpecialFunctions: besselix     (for Poisson)
#   using HypergeometricFunctions: _₂F₁ (for NegativeBinomial; already imported)

# ── Poisson ────────────────────────────────────────────────────────────────
# Formula from Jordan et al. (2019), translating crps_pois in scores_pois.R.
# R's `besselI(x, nu, expon.scaled = TRUE)` is `besselix(nu, x)`, the scaled
# form; the unscaled `besseli` overflows from x = 2λ ≈ 700. `besselix` itself
# throws an `AmosException` from x ≈ 1e9, so large arguments use the expansion
# below instead.

# Large-argument expansion exp(-x) I_ν(x) ≈ (2πx)^(-1/2) Σₖ cₖ with c₀ = 1 and
# cₖ = cₖ₋₁ ((2k-1)² - 4ν²) / (8kx). Ten terms leave a relative error below
# 1e-30 for x ≥ 1e6.
const _BESSEL_ASYMPTOTIC_MIN = 1.0e6

function _besselix_asymptotic(nu::Integer, x::Real)
    term = one(float(x))
    total = term
    for k in 1:10
        term *= ((2k - 1)^2 - 4nu^2) / (8k * x)
        total += term
    end
    return total / sqrt(2 * pi * x)
end

# `logpdf(Poisson(λ), k)` subtracts terms of size λ and loses about log10(λ) - 16
# digits of the mass at large λ. Loader's saddle-point form evaluates the
# deviance term `λ - k + k log(k/λ)` without that cancellation.
function _poispdf_large(k::Integer, lambda::Real)
    d = k - lambda
    if abs(d) < 0.1 * (k + lambda)
        v = d / (k + lambda)
        s = d * v
        ej = 2k * v
        v2 = v^2
        for j in 1:200
            ej *= v2
            s1 = s + ej / (2j + 1)
            s1 == s && break
            s = s1
        end
    else
        s = k * log(k / lambda) + lambda - k
    end
    # Stirling error of k!; accurate to 1e-18 for k ≥ 1000.
    stirlerr = 1 / (12k) - 1 / (360 * float(k)^3)
    return exp(-stirlerr - s) / sqrt(2 * pi * k)
end

_besselix01(nu::Integer, x::Real) =
    x < _BESSEL_ASYMPTOTIC_MIN ? besselix(nu, x) : _besselix_asymptotic(nu, x)

"""
CRPS of a `Poisson(λ)` forecast, in closed form.

The Poisson CDF is `F(y) = P(Y ≤ ⌊y⌋) = Q(⌊y⌋+1, λ) = ccdf(Gamma(⌊y⌋+1, 1), λ)`,
and `0` for `y < 0` (empty support — matching R's `crps_pois`). Routed through
`ccdf_ad_safe` so `λ` stays differentiable: the stock `cdf(::Poisson)` reaches
`gamma_inc`, which has no method for a Dual shape argument (#11). `ccdf_ad_safe`
returns `Q` directly rather than `1 - P`, keeping the far upper tail accurate.
"""
function _crps_pois(y::Real, lambda::Real)
    k = floor(y + 1)
    poiscdf = k <= 0 ? oftype(float(lambda), 0) : ccdf_ad_safe(Gamma(k, 1.0), lambda)
    c1 = (y - lambda) * (2 * poiscdf - 1)
    x = 2 * lambda
    n = floor(Int, y)
    pmf = x < _BESSEL_ASYMPTOTIC_MIN || n < 1000 ? pdf(Poisson(lambda), n) :
        _poispdf_large(n, lambda)
    c2 = 2 * pmf - (_besselix01(0, x) + _besselix01(1, x))
    return c1 + lambda * c2
end

crps(d::Poisson, y::Real) = _crps_pois(y, d.λ)

# ── Negative binomial ──────────────────────────────────────────────────────
# Formula from Jordan et al. (2019), translating crps_nbinom in scores_nbinom.R.
#
# Parameter mapping:
#   R `pnbinom(y, size, prob)` ↔ Distributions `NegativeBinomial(size, prob)`
#   where `prob` is the probability of SUCCESS on each Bernoulli trial.
#   Distributions.NegativeBinomial(r, p): r = size (number of successes),
#   p = prob (success probability); mean = r*(1-p)/p.
#
# The hypergeometric function call is `_₂F₁(size+1, 0.5, 2, -4*c2)`.
"""
CRPS of a `NegativeBinomial(size, prob)` forecast, in closed form.
"""
function _crps_nbinom(y::Real, size::Real, prob::Real)
    c1 = y * (2 * cdf(NegativeBinomial(size, prob), y) - 1)
    c2 = (1 - prob) / prob^2
    # pnbinom(y-1, size+1, prob) is the CDF of NB(size+1, prob) at y-1
    c3 = (
        prob * (2 * cdf(NegativeBinomial(size + 1, prob), y - 1) - 1)
            +
            _₂F₁(size + 1, 0.5, 2.0, -4 * c2)
    )
    return c1 - size * c2 * c3
end

crps(d::NegativeBinomial, y::Real) = _crps_nbinom(y, d.r, d.p)

# ── Binomial ───────────────────────────────────────────────────────────────
# Finite sum over the full support {0, …, n}, translating crps_binom in
# scores_binom.R.  The formula `2 * Σ w*(𝟙{y < x} - a)*(x - y)` with
# `a = F(x) - 0.5*w` is the energy-score representation for a finite
# discrete distribution.
"""
CRPS of a `Binomial(n, p)` forecast, via finite sum over the support.
"""
function _crps_binom(y::Real, size::Integer, prob::Real)
    s = 0.0
    for x in 0:size
        w = pdf(Binomial(size, prob), x)
        a = cdf(Binomial(size, prob), x) - 0.5 * w
        s += w * ((y < x ? 1.0 : 0.0) - a) * (x - y)
    end
    return 2 * s
end

crps(d::Binomial, y::Real) = _crps_binom(y, d.n, d.p)

# ── Hypergeometric ─────────────────────────────────────────────────────────
# Finite sum over the support {max(0, k-n), …, min(k, m)}, translating
# crps_hyper in scores_hyper.R.
#
# Parameter mapping:
#   R `phyper(y, m, n, k)`:       m = #white balls, n = #black balls, k = #draws
#   Distributions `Hypergeometric(s, f, n)`: s = successes (white),
#                                             f = failures  (black), n = draws
#   so: s ↔ m,  f ↔ n,  n(draws) ↔ k.
#
# Function signature preserves the R convention (m, n, k) for clarity.
"""
CRPS of a `Hypergeometric(m, n, k)` forecast, via finite sum over the support.
Here `m` = number of white balls, `n` = number of black balls, `k` = draws
(matching R's `phyper` / `crps_hyper` convention).
"""
function _crps_hyper(y::Real, m::Integer, n::Integer, k::Integer)
    # Distributions.Hypergeometric(s, f, n_draws): s=white, f=black, n_draws=k
    d = Hypergeometric(m, n, k)
    xlo = max(0, k - n)
    xhi = min(k, m)
    s = 0.0
    for x in xlo:xhi
        w = pdf(d, x)
        a = cdf(d, x) - 0.5 * w
        s += w * ((y < x ? 1.0 : 0.0) - a) * (x - y)
    end
    return 2 * s
end

# Distributions.Hypergeometric(s, f, n): s=successes(white), f=failures(black), n=draws
crps(d::Hypergeometric, y::Real) = _crps_hyper(y, d.ns, d.nf, d.n)
