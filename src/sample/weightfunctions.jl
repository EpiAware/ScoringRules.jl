# Smooth weight and chaining functions for the threshold- and outcome-weighted
# scores, matching `get_weight_func` in R scoringRules v1.1.3
# (scores_sample_univ_weighted.R; Allen 2024, JSS 110(8)) under GPL-2.0-or-later.
#
# The default weight of `owcrps` and friends is the indicator of an interval,
# and the default chaining function is the clamp onto it. Allen (2024) also
# uses smooth weights: a normal or logistic CDF to emphasise the upper tail, a
# survival function for the lower tail, a density for a region around a point.
# Each needs its matching chaining function, the antiderivative of the weight,
# which is what makes the threshold-weighted score proper.

export weight_function, chaining_function

const _WEIGHT_NAMES = (
    :norm_cdf, :norm_surv, :norm_pdf,
    :logis_cdf, :logis_surv, :logis_pdf,
)

const _LOGIS_NAMES = (:logis_cdf, :logis_surv, :logis_pdf)

# --- the six scalar kernels, as `(z, μ, σ) -> value` -----------------------
#
# Each chaining kernel is the antiderivative of the weight kernel of the same
# name, so `d/dz chain(z) == weight(z)`.

_wk_norm_cdf(z, μ, σ) = _norm_cdf((z - μ) / σ)
_wk_norm_surv(z, μ, σ) = _norm_cdf(-(z - μ) / σ)
_wk_norm_pdf(z, μ, σ) = _norm_pdf((z - μ) / σ) / σ
_wk_logis_cdf(z, μ, σ) = logistic((z - μ) / σ)
_wk_logis_surv(z, μ, σ) = logistic(-(z - μ) / σ)
function _wk_logis_pdf(z, μ, σ)
    u = (z - μ) / σ
    return logistic(u) * logistic(-u) / σ
end

_ck_norm_cdf(z, μ, σ) = (z - μ) * _norm_cdf((z - μ) / σ) + σ * _norm_pdf((z - μ) / σ)
_ck_norm_surv(z, μ, σ) = z - _ck_norm_cdf(z, μ, σ)
_ck_norm_pdf(z, μ, σ) = _norm_cdf((z - μ) / σ)
_ck_logis_cdf(z, μ, σ) = σ * log1pexp((z - μ) / σ)
_ck_logis_surv(z, μ, σ) = z - σ * log1pexp((z - μ) / σ)
_ck_logis_pdf(z, μ, σ) = logistic((z - μ) / σ)

function _weight_kernel(name::Symbol)
    name === :norm_cdf && return _wk_norm_cdf
    name === :norm_surv && return _wk_norm_surv
    name === :norm_pdf && return _wk_norm_pdf
    name === :logis_cdf && return _wk_logis_cdf
    name === :logis_surv && return _wk_logis_surv
    return _wk_logis_pdf
end

function _chain_kernel(name::Symbol)
    name === :norm_cdf && return _ck_norm_cdf
    name === :norm_surv && return _ck_norm_surv
    name === :norm_pdf && return _ck_norm_pdf
    name === :logis_cdf && return _ck_logis_cdf
    name === :logis_surv && return _ck_logis_surv
    return _ck_logis_pdf
end

# --- the callables ---------------------------------------------------------

_param_at(x::Real, ::Integer) = x
# A length-one vector is a scalar written as a vector, so it broadcasts across
# dimensions rather than being indexed past its own end.
_param_at(x::AbstractVector, i::Integer) = length(x) == 1 ? first(x) : x[i]

_param_len(::Real) = 1
_param_len(x::AbstractVector) = length(x)

"""
    SmoothWeight

Weight function built by [`weight_function`](@ref). Call it on a `Real` for a
univariate forecast, or on a length-`d` vector for a multivariate one, where it
returns the product of the per-dimension weights (as R's `get_weight_func`
does).
"""
struct SmoothWeight{K, M, S} <: Function
    kernel::K
    mu::M
    sigma::S
end

function (f::SmoothWeight)(z::Real)
    _check_scalar_params(f)
    return f.kernel(z, _param_at(f.mu, 1), _param_at(f.sigma, 1))
end

function (f::SmoothWeight)(z::AbstractVector)
    _check_param_dim(f, length(z))
    out = one(_float_type(z))
    @inbounds for i in eachindex(z)
        out *= f.kernel(z[i], _param_at(f.mu, i), _param_at(f.sigma, i))
    end
    return out
end

"""
    SmoothChain

Chaining function built by [`chaining_function`](@ref): the antiderivative of
the weight function of the same name. Call it on a `Real` for a univariate
forecast, or on a length-`d` vector for a multivariate one, where it applies
element-wise.
"""
struct SmoothChain{K, M, S} <: Function
    kernel::K
    mu::M
    sigma::S
end

function (f::SmoothChain)(z::Real)
    _check_scalar_params(f)
    return f.kernel(z, _param_at(f.mu, 1), _param_at(f.sigma, 1))
end

function (f::SmoothChain)(z::AbstractVector)
    _check_param_dim(f, length(z))
    return [
        f.kernel(z[i], _param_at(f.mu, i), _param_at(f.sigma, i))
            for i in eachindex(z)
    ]
end

# A multivariate weight applied to a scalar observation would use the first
# dimension's parameters and drop the rest, returning a plausible wrong score.
function _check_scalar_params(f)
    n = max(_param_len(f.mu), _param_len(f.sigma))
    n == 1 || throw(
        DimensionMismatch(
            "mu and sigma have length $n, so this function takes a length-$n " *
                "vector rather than a scalar"
        )
    )
    return nothing
end

function _check_param_dim(f, d::Integer)
    n = max(_param_len(f.mu), _param_len(f.sigma))
    n == 1 || n == d || throw(
        DimensionMismatch(
            "mu and sigma have length $n but the forecast has $d dimensions"
        )
    )
    return nothing
end

# --- validation ------------------------------------------------------------

function _check_weight_params(name::Symbol, mu, sigma)
    name in _WEIGHT_NAMES || throw(
        ArgumentError(
            "name must be one of $(join(_WEIGHT_NAMES, ", ")), got :$name"
        )
    )
    lm, ls = _param_len(mu), _param_len(sigma)
    # An empty parameter would build a callable that can only throw.
    (lm == 0 || ls == 0) && throw(
        DimensionMismatch("mu and sigma must not be empty")
    )
    (lm == ls || lm == 1 || ls == 1) || throw(
        DimensionMismatch("mu has length $lm and sigma has length $ls")
    )
    all(>(0), sigma) ||
        throw(DomainError(sigma, "sigma must be strictly positive"))
    all(isfinite, mu) && all(isfinite, sigma) ||
        throw(ArgumentError("mu and sigma must be finite"))
    # R rejects the logistic shapes for a multivariate forecast whichever half
    # of the pair it is asked for, and so must both constructors here: an
    # element-wise logistic chaining function is the antiderivative of no
    # product weight, which would leave the threshold-weighted score improper
    # for the weight the caller believes they chose.
    max(lm, ls) > 1 && name in _LOGIS_NAMES && throw(
        ArgumentError(
            "the logistic weight functions are univariate only; use " *
                ":norm_cdf, :norm_surv or :norm_pdf for a multivariate forecast"
        )
    )
    return nothing
end

# --- public constructors ---------------------------------------------------

"""
    weight_function(name; mu = 0.0, sigma = 1.0)

Build a smooth weight function for the outcome-weighted scores (`owcrps`,
`owes`, `owvs`, `owmmds`), the counterpart of `get_weight_func(..., weight =
TRUE)` in R scoringRules.

# Arguments

  - `name`: one of `:norm_cdf`, `:norm_surv`, `:norm_pdf`, `:logis_cdf`,
    `:logis_surv`, `:logis_pdf`.

# Keyword Arguments

  - `mu`: location of the weight, a scalar or a length-`d` vector.
  - `sigma`: scale of the weight, strictly positive, a scalar or a length-`d`
    vector.

`name` selects the shape, with `mu` and `sigma` its location and scale:

| `name`        | weight                                     |
|:--------------|:-------------------------------------------|
| `:norm_cdf`   | `Φ((z − μ)/σ)`, emphasising larger outcomes |
| `:norm_surv`  | `1 − Φ((z − μ)/σ)`, emphasising smaller ones |
| `:norm_pdf`   | `φ((z − μ)/σ)/σ`, emphasising outcomes near `μ` |
| `:logis_cdf`  | the logistic CDF                            |
| `:logis_surv` | the logistic survival function              |
| `:logis_pdf`  | the logistic density                        |

Pass length-`d` vectors for `mu` and `sigma` (or scalars, which broadcast) to
weight a `d`-dimensional forecast; the weight is then the product over
dimensions. As in R, the logistic shapes are univariate only, because the
matching chaining function has no product form.

The paired [`chaining_function`](@ref) gives the chaining function the
threshold-weighted scores need for the same weight.

# Example

```@example
using ScoringRules
dat = randn(500)
w = weight_function(:norm_cdf; mu = 1.0, sigma = 0.5)
owcrps(dat, 0.5; weight_func = w)
```
"""
function weight_function(name::Symbol; mu = 0.0, sigma = 1.0)
    _check_weight_params(name, mu, sigma)
    return SmoothWeight(_weight_kernel(name), mu, sigma)
end

"""
    chaining_function(name; mu = 0.0, sigma = 1.0)

Build a smooth chaining function for the threshold-weighted scores (`twcrps`,
`twes`, `twvs`, `twmmds`), the counterpart of `get_weight_func(..., weight =
FALSE)` in R scoringRules.

# Arguments

  - `name`: one of `:norm_cdf`, `:norm_surv`, `:norm_pdf`, `:logis_cdf`,
    `:logis_surv`, `:logis_pdf`.

# Keyword Arguments

  - `mu`: location of the weight, a scalar or a length-`d` vector.
  - `sigma`: scale of the weight, strictly positive, a scalar or a length-`d`
    vector.

`name`, `mu` and `sigma` are as in [`weight_function`](@ref), and the function
returned is the antiderivative of the weight of the same name:

| `name`        | chaining function                            |
|:--------------|:---------------------------------------------|
| `:norm_cdf`   | `(z − μ)Φ((z − μ)/σ) + σφ((z − μ)/σ)`         |
| `:norm_surv`  | `z` minus the `:norm_cdf` function           |
| `:norm_pdf`   | `Φ((z − μ)/σ)`                               |
| `:logis_cdf`  | `σ·log(1 + exp((z − μ)/σ))`                  |
| `:logis_surv` | `z − σ·log(1 + exp((z − μ)/σ))`              |
| `:logis_pdf`  | the logistic CDF                             |

Being the antiderivative is what makes the threshold-weighted score a proper
scoring rule for that weight. With length-`d` `mu` and `sigma` the function
applies element-wise to a `d`-dimensional forecast.

# Example

```@example
using ScoringRules
dat = randn(500)
v = chaining_function(:norm_cdf; mu = 1.0, sigma = 0.5)
twcrps(dat, 0.5; chain_func = v)
```
"""
function chaining_function(name::Symbol; mu = 0.0, sigma = 1.0)
    _check_weight_params(name, mu, sigma)
    return SmoothChain(_chain_kernel(name), mu, sigma)
end
