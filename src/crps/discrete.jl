# Closed-form CRPS for four discrete families: Poisson, NegativeBinomial,
# Binomial, and Hypergeometric.
#
# Ported from R scoringRules v1.1.3 (GPL-2):
#   scores_pois.R, scores_nbinom.R, scores_binom.R, scores_hyper.R
# by Alexander I. Jordan, Fabian Krüger, Sebastian Lerch and Sam Allen.
#
# Module-level imports required (in addition to those already in ScoringRules.jl):
#   using SpecialFunctions: besselix     (for Poisson)
#   using QuadGK: quadgk                 (for the NegativeBinomial ₂F₁)

# ── Poisson ────────────────────────────────────────────────────────────────
# Formula from Jordan et al. (2019), translating crps_pois in scores_pois.R.
# R's `besselI(x, nu, expon.scaled = TRUE)` equals `exp(-x) * besseli(nu, x)`.
# The two calls share the same argument x = 2λ, so we factor out exp(-2λ).
"""
CRPS of a `Poisson(λ)` forecast, in closed form.

The Poisson CDF is `F(y) = P(Y ≤ ⌊y⌋) = Q(⌊y⌋+1, λ) = ccdf(Gamma(⌊y⌋+1, 1), λ)`,
and `0` for `y < 0` (empty support — matching R's `crps_pois`). Routed through
`ccdf_ad_safe` so `λ` stays differentiable: the stock `cdf(::Poisson)` reaches
`gamma_inc`, which has no method for a Dual shape argument (#11). `ccdf_ad_safe`
returns `Q` directly rather than `1 - P`, keeping the far upper tail accurate.
"""
function _crps_pois(y::Real, lambda::Real)
    isnan(y) && return oftype(float(lambda) * one(y), NaN)
    k = floor(y + 1)
    poiscdf = k <= 0 ? oftype(float(lambda), 0) : ccdf_ad_safe(Gamma(k, 1.0), lambda)
    c1 = (y - lambda) * (2 * poiscdf - 1)
    x = 2 * lambda
    # `besselix(ν, x) = exp(−x)·besseli(ν, x)` is R's `besselI(expon.scaled =
    # TRUE)`. Forming the unscaled `besseli(0, 2λ)` first overflows the Amos
    # routine from λ ≈ 351 even though the scaled product is O(10⁻²) there.
    c2 = 2 * _pois_dens(lambda, y) - (besselix(0, x) + besselix(1, x))
    return c1 + lambda * c2
end

# `dpois(⌊y⌋, λ)`, zero outside the support. The `Int` conversion is guarded so
# that an observation beyond `typemax(Int)` gives the correct zero density
# rather than an `InexactError`.
@inline function _pois_dens(lambda::Real, y::Real)
    (isfinite(y) && 0 <= y <= typemax(Int)) || return zero(float(lambda))
    return pdf(Poisson(lambda), floor(Int, y))
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
# The hypergeometric factor is `₂F₁(size+1, ½; 2; −4·c2)`, evaluated through
# `_f21_half_two` below.

# `₂F₁(a, ½; 2; −Z)` for `Z ≥ 0`, from the Euler integral
#
#   ₂F₁(a, b; c; z) = B(b, c−b)⁻¹ ∫₀¹ t^{b−1} (1−t)^{c−b−1} (1−zt)^{−a} dt,
#
# with `t = sin²θ` clearing the `t^{−1/2}` endpoint singularity:
#
#   ₂F₁(a, ½; 2; −Z) = (4/π) ∫₀^{π/2} cos²θ · (1 + Z sin²θ)^{−a} dθ.
#
# For `Z ≥ 0` the integrand is smooth and bounded by 1, so the quadrature is
# accurate for every `a > 0`. The Gauss series that `HypergeometricFunctions`
# sums instead loses digits from `a ≈ 40` at the `Z ≈ 31` of a `prob = 0.3`
# forecast and returns `NaN` from `a ≈ 100`; it also needs `a` to be a
# non-positive integer after a Pfaff transformation, which a half-integer
# `size` never is.
function _f21_half_two(a::Real, Z::Real)
    (Z < 0 || isnan(Z) || isnan(a)) && return oftype(float(a) * float(Z), NaN)
    integrand = θ -> cos(θ)^2 * exp(-a * log1p(Z * sin(θ)^2))
    val, _ = quadgk(integrand, 0.0, Float64(π) / 2; rtol = 1.0e-13)
    return 4 * val / π
end

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
            _f21_half_two(size + 1, 4 * c2)
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
