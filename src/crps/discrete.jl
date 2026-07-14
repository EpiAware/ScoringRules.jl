# Closed-form CRPS for four discrete families: Poisson, NegativeBinomial,
# Binomial, and Hypergeometric.
#
# Ported from R scoringRules v1.1.3 (GPL-2):
#   scores_pois.R, scores_nbinom.R, scores_binom.R, scores_hyper.R
# by Alexander I. Jordan, Fabian Krüger, Sebastian Lerch and Sam Allen.
#
# Module-level imports required (in addition to those already in ScoringRules.jl):
#   using SpecialFunctions: besseli      (for Poisson)
#   using HypergeometricFunctions: _₂F₁ (for NegativeBinomial; already imported)

# ── Poisson ────────────────────────────────────────────────────────────────
# Formula from Jordan et al. (2019), translating crps_pois in scores_pois.R.
# R's `besselI(x, nu, expon.scaled = TRUE)` equals `exp(-x) * besseli(nu, x)`.
# The two calls share the same argument x = 2λ, so we factor out exp(-2λ).
"""
CRPS of a `Poisson(λ)` forecast, in closed form.
"""
function _crps_pois(y::Real, lambda::Real)
    c1 = (y - lambda) * (2 * cdf(Poisson(lambda), y) - 1)
    x = 2 * lambda
    c2 = 2 * pdf(Poisson(lambda), floor(Int, y)) -
         exp(-x) * (besseli(0, x) + besseli(1, x))
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
    c3 = (prob * (2 * cdf(NegativeBinomial(size + 1, prob), y - 1) - 1)
          +
          _₂F₁(size + 1, 0.5, 2.0, -4 * c2))
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
