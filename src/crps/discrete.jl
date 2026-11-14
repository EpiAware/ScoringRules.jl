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
# R's `besselI(x, nu, expon.scaled = TRUE)` equals `exp(-x) * besseli(nu, x)`,
# which is `besselix` here; both terms use it and no `exp(-x)` is formed.
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
    c2 = 2 * _pois_dens(lambda, y) - _besselix01(x)
    return c1 + lambda * c2
end

# `besselix(0, x) + besselix(1, x)`. Past about `x = 1e9` the Amos routine
# refuses its argument outright, so above `1e8` the sum is taken from its
# large-argument expansion, `sqrt(2 / (π x)) · (1 − 1/(8x) − 3/(128x²) …)`,
# whose first omitted term is below 1e-17 relative there.
@inline function _besselix01(x::Real)
    x > 1.0e8 || return besselix(0, x) + besselix(1, x)
    return sqrt(2 / (oftype(float(x), π) * x)) * (1 - 1 / (8 * x))
end

# `dpois(⌊y⌋, λ)`, zero outside the support. The `Int` conversion is guarded so
# that an observation beyond `typemax(Int)` gives the correct zero density
# rather than an `InexactError`.
#
# The mass is taken in Loader's saddle-point form, as R's `dpois` takes it. The
# textbook `exp(k log λ − λ − log k!)`, which `pdf(::Poisson)` uses, subtracts
# terms of size `k log λ` to leave one of size one, and is 2.5e-7 out at a mean
# of 1e8. Here those terms cancel analytically inside `_bd0`.
@inline function _pois_dens(lambda::Real, y::Real)
    (isfinite(y) && 0 <= y <= typemax(Int)) || return zero(float(lambda))
    k = floor(Int, y)
    k == 0 && return exp(-float(lambda))
    iszero(lambda) && return zero(float(lambda))
    return exp(-_stirlerr(k) - _bd0(k, lambda)) / sqrt(2 * oftype(float(lambda), π) * k)
end

# The error of Stirling's formula, `log k! − ((k + 1/2) log k − k + log √(2π))`.
# Small `k` takes it directly, where the terms are small enough not to cancel;
# above that its asymptotic series, with as many terms as `k` needs for the
# first omitted one to fall below 1e-16 (the split points are R's).
function _stirlerr(k::Integer)
    k <= 15 && return loggamma(k + 1.0) - (k + 0.5) * log(Float64(k)) + k - log(sqrt(2π))
    n = Float64(k)
    nn = n * n
    S0, S1, S2, S3, S4 = 1 / 12, 1 / 360, 1 / 1260, 1 / 1680, 1 / 1188
    k > 500 && return (S0 - S1 / nn) / n
    k > 80 && return (S0 - (S1 - S2 / nn) / nn) / n
    k > 35 && return (S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / n
    return (S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n
end

# `k log(k/λ) + λ − k`, the deviance term of the saddle-point form. Near the
# mode the two halves nearly cancel, so there it is summed as a series in
# `v = (k − λ)/(k + λ)`, whose terms are all of one sign.
function _bd0(k::Integer, lambda::Real)
    d = k - lambda
    if abs(d) < (k + lambda) / 10
        v = d / (k + lambda)
        s = d * v
        # In floating point: `2k` overflows an `Int` past a mean of 4.6e18.
        ej = 2 * float(k) * v
        v2 = v * v
        for j in 1:1000
            ej *= v2
            s1 = s + ej / (2j + 1)
            s1 == s && return s1
            s = s1
        end
        return s
    end
    return k * log(k / lambda) + lambda - k
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
# sums instead returns `NaN` from `a ≈ 100` at the `Z ≈ 31` of a `prob = 0.3`
# forecast, which is what this replaces. R sums the same series through its
# `hypergeo` package and returns `-Inf` for some half-integer sizes; see the
# Differences from R page.
function _f21_half_two(a::Real, Z::Real)
    (Z < 0 || isnan(Z) || isnan(a)) && return oftype(float(a) * float(Z), NaN)
    hi = Float64(π) / 2
    integrand = θ -> cos(θ)^2 * exp(-a * log1p(Z * sin(θ)^2))
    # The integrand is a peak at θ = 0 whose half-width is about 1/√(aZ).
    # Left to choose its own first panel over the whole interval, `quadgk`
    # places its lowest node at θ ≈ 0.0067, and once `a·log1p(Z·sin²θ)` passes
    # 745 there the integrand underflows to zero at every node it looks at: the
    # estimate and its error are both zero, the tolerance test passes, and the
    # function returns 0 having subdivided nothing. That made
    # `crps(NegativeBinomial(724, 0.01), 71680)` 2136 instead of 626, with the
    # knife edge one unit of `size` away. Splitting at geometric multiples of
    # the half-width puts nodes where the mass is, whatever the parameters.
    #
    # The half-width is taken as `1/(√a·√Z)`: the product `a·Z` overflows for
    # a `prob` below about 1e-154, and a step of exactly zero never grows.
    # An infinite `Z` leaves nothing to integrate, the integrand being zero
    # for every θ > 0.
    isinf(Z) && return zero(float(a) * float(Z))
    segments = Float64[]
    step = 1 / (8 * max(sqrt(a) * sqrt(Z), 1.0))
    while step < hi
        push!(segments, step)
        step *= 4
    end
    val, _ = quadgk(integrand, 0.0, segments..., hi; rtol = 1.0e-13)
    return 4 * val / π
end

"""
CRPS of a `NegativeBinomial(size, prob)` forecast, in closed form.
"""
function _crps_nbinom(y::Real, size::Real, prob::Real)
    c1 = y * (2 * cdf(NegativeBinomial(size, prob), y) - 1)
    c2 = (1 - prob) / prob^2
    # pnbinom(y-1, size+1, prob) is the CDF of NB(size+1, prob) at y-1
    u = prob * (2 * cdf(NegativeBinomial(size + 1, prob), y - 1) - 1)
    h = _f21_half_two(size + 1, 4 * c2)
    out = c1 - size * c2 * (u + h)
    # Each `2F − 1` is a difference of two probabilities, good to about one
    # `eps` absolute however small it comes out, and the closed form multiplies
    # those errors by `y` and by `size * c2 ≈ size/prob`. Where the score is
    # small beside that, the result keeps few digits: at a huge size, where
    # the score grows only like `√size/prob`; at a nearly degenerate forecast
    # observed at zero, where the score is of order `size·(1 − prob)`; and
    # below `prob ≈ 1e-154`, where `c2` overflows outright. The bound below is
    # within a few times of the error measured against lattice sums.
    #
    # A forecast compact enough to sum is summed exactly wherever the closed
    # form falls short of 1e-10. One too wide to sum keeps the closed form to
    # 1e-6, and past that has no score to give: a NaN says so where the
    # subtraction would hand back an arbitrary, often negative, number. An
    # infinite observation keeps its infinite score.
    isfinite(y) || return out
    rel = 2 * eps(Float64) * (abs(y) + size * c2 * (prob + abs(h))) / abs(out)
    rel <= 1.0e-10 && return out
    usable = rel <= 1.0e-6 ? out : oftype(out, NaN)
    # How far the sum has to run: past `y`, past the bulk, and down a tail that
    # shrinks by `1 − prob` a step, which takes about `20/prob` steps to fall
    # below `eps` squared. Each step is a `cdf` and a `ccdf` call. A short sum,
    # a few milliseconds, is worth it for the last digits; a long one, up to
    # about a second, only where the closed form has no usable answer at all.
    mu = size * (1 - prob) / prob
    span = max(y, mu + 40 * sqrt(mu / prob), 20 / prob)
    worth = span <= 1.0e4 || (isnan(usable) && span <= 1.0e6)
    worth && return _crps_nbinom_sum(y, size, prob, usable)
    return usable
end

# The CRPS of a negative binomial forecast as the integral over its step CDF,
# for a forecast the closed form scores too coarsely. Below the support the
# integrand is one; on `[k, k + 1)` it is `F(k)²` below `y` and `S(k)²` above.
# The sum runs until the tail left is too small to change the total, which for
# a compact forecast is a handful of terms; past a million it gives up and
# returns `fallback`.
function _crps_nbinom_sum(y::Real, size::Real, prob::Real, fallback::Real)
    d = NegativeBinomial(size, prob)
    total = y < 0 ? -float(y) : zero(float(y))
    for k in 0:1_000_000
        F, S = cdf(d, k), ccdf(d, k)
        total += if y <= k
            S^2
        elseif y >= k + 1
            F^2
        else
            F^2 * (y - k) + S^2 * (k + 1 - y)
        end
        # Past `k` each step multiplies the mass by at most `ρ`, the largest
        # ratio `P(j + 1)/P(j)` from here on, so once `ρ < 1` what the tail
        # can still add after a term `S²` is at most `S²/(1 − ρ²)`.
        ρ = (1 - prob) * max(one(size), (k + size) / (k + 1))
        k >= y && ρ < 1 && S^2 / (1 - ρ^2) <= eps(Float64) * total && return total
    end
    return fallback
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
