# [Validation against R](@id validation-against-r)

ScoringRules.jl is a port of the R `scoringRules` package, so its numbers should
match the original. This page recomputes that agreement **live on every docs
build**: it loads reference values generated from R `scoringRules` 1.1.3
(committed under `test/references/data/`, produced by
`test/references/generate_references.R`), computes the same quantities in Julia,
and reports the difference. Nothing here is hand-copied.

## Score values

For each parametric family, the CRPS computed in Julia versus R's reference
value, as the maximum absolute and relative difference over all reference rows:

```@example validation
using ScoringRules, Distributions, DelimitedFiles, Printf

dir = joinpath(pkgdir(ScoringRules), "test", "references", "data")
col(h, name) = findfirst(==(name), vec(h))

function crps_parity(name, f)
    d, h = readdlm(joinpath(dir, name * ".csv"), ',', header = true)
    ci = col(h, "crps")
    n = 0; ma = 0.0; mr = 0.0
    for i in 1:size(d, 1)
        ref = d[i, ci]
        (ref isa Number && isfinite(ref)) || continue
        jv = f(d[i, :], vec(h))
        isfinite(jv) || continue
        a = abs(jv - ref)
        ma = max(ma, a); mr = max(mr, a / max(abs(ref), 1e-8)); n += 1
    end
    @sprintf("%-13s %5d    %.1e    %.1e", name, n, ma, mr)
end

println("family         rows    max|Δ|    max rel")
for line in [
    crps_parity("norm", (r, h) -> crps(Normal(r[col(h, "mean")], r[col(h, "sd")]), r[col(h, "y")])),
    crps_parity("gamma", (r, h) -> crps(Gamma(r[col(h, "shape")], r[col(h, "scale")]), r[col(h, "y")])),
    crps_parity("lnorm", (r, h) -> crps(LogNormal(r[col(h, "meanlog")], r[col(h, "sdlog")]), r[col(h, "y")])),
    crps_parity("logis", (r, h) -> crps(Logistic(r[col(h, "location")], r[col(h, "scale")]), r[col(h, "y")])),
    crps_parity("laplace", (r, h) -> crps(Laplace(r[col(h, "location")], r[col(h, "scale")]), r[col(h, "y")])),
    crps_parity("exponential", (r, h) -> crps(Exponential(1 / r[col(h, "rate")]), r[col(h, "y")])),
    crps_parity("beta", (r, h) -> crps(Beta(r[col(h, "shape1")], r[col(h, "shape2")]), r[col(h, "y")])),
    crps_parity("student_t", (r, h) -> crps(r[col(h, "location")] + r[col(h, "scale")] * TDist(r[col(h, "df")]), r[col(h, "y")])),
    crps_parity("pois", (r, h) -> crps(Poisson(r[col(h, "lambda")]), r[col(h, "y")])),
    crps_parity("nbinom", (r, h) -> crps(NegativeBinomial(r[col(h, "size")], r[col(h, "prob")]), r[col(h, "y")])),
    crps_parity("binom", (r, h) -> crps(Binomial(Int(r[col(h, "size")]), r[col(h, "prob")]), r[col(h, "y")])),
]
    println(line)
end
```

The differences are at the level of floating-point round-off, so the two
implementations agree for practical purposes. The full test suite checks
`logs`/`dss`, the ensemble scores and the weighted/multivariate scores the same
way, at `atol = 1e-9, rtol = 1e-7`.

## Gradients via automatic differentiation

R ships analytic CRPS gradients (`gradcrps_*`) only for the Normal, Logistic and
Student-t families. Where our closed forms are AD-differentiable, ForwardDiff
reproduces those gradients exactly — so replacing R's hand-coded gradients with
AD loses no accuracy:

```@example validation
using ForwardDiff

function grad_parity(name, gradf)
    d, h = readdlm(joinpath(dir, "grad_" * name * ".csv"), ',', header = true)
    hh = vec(h); dl = col(hh, "dloc"); ds = col(hh, "dscale")
    ma = 0.0
    for i in 1:size(d, 1)
        g = gradf(d[i, :], hh)
        ma = max(ma, abs(g[1] - d[i, dl]), abs(g[2] - d[i, ds]))
    end
    @sprintf("%-8s %d rows   max|Δ∇crps vs R gradcrps| = %.1e", name, size(d, 1), ma)
end

println(grad_parity("norm",
    (r, h) -> ForwardDiff.gradient(p -> crps(Normal(p[1], p[2]), r[col(h, "y")]),
        [r[col(h, "location")], r[col(h, "scale")]])))
println(grad_parity("logis",
    (r, h) -> ForwardDiff.gradient(p -> crps(Logistic(p[1], p[2]), r[col(h, "y")]),
        [r[col(h, "location")], r[col(h, "scale")]])))
```

## Which families are AD-differentiable

Not every closed form is differentiable by ForwardDiff. Some route through
special functions — `beta_inc`, `gamma_inc`, `besseli`, `₂F₁` — that do not yet
propagate dual numbers. The live support map for `crps`, differentiating with
respect to the distribution parameters and with respect to the observation `y`:

```@example validation
cases = [
    ("Normal", θ -> Normal(θ[1], θ[2]), [0.5, 1.5], 0.7),
    ("Logistic", θ -> Logistic(θ[1], θ[2]), [0.5, 1.5], 0.7),
    ("Laplace", θ -> Laplace(θ[1], θ[2]), [0.5, 1.5], 0.7),
    ("Exponential", θ -> Exponential(θ[1]), [1.5], 0.7),
    ("Uniform", θ -> Uniform(θ[1], θ[2]), [0.0, 2.0], 0.7),
    ("LogNormal", θ -> LogNormal(θ[1], θ[2]), [0.0, 0.5], 1.2),
    ("GPD", θ -> GeneralizedPareto(θ[1], θ[2], 0.2), [0.0, 1.0], 0.7),
    ("Gamma", θ -> Gamma(θ[1], θ[2]), [2.0, 1.5], 1.2),
    ("Beta", θ -> Beta(θ[1], θ[2]), [2.0, 3.0], 0.4),
    ("Student-t", θ -> θ[1] + θ[2] * TDist(5.0), [0.5, 1.5], 0.7),
    ("LogLogistic", θ -> LogLogistic(θ[1], θ[2]), [1.0, 3.0], 1.2),
    ("Poisson", θ -> Poisson(θ[1]), [3.0], 2.0),
]
status(f) = try
    all(isfinite, f()) ? "yes" : "NaN"
catch
    "no"
end
println(rpad("family", 13), "  ", rpad("∇ wrt params", 14), "d/dy")
for (name, D, θ0, y0) in cases
    wp = status(() -> ForwardDiff.gradient(θ -> crps(D(θ), y0), θ0))
    wy = status(() -> [ForwardDiff.derivative(y -> crps(D(θ0), y), y0)])
    println(rpad(name, 13), "  ", rpad(wp, 14), wy)
end
```

The `beta_inc`/`gamma_inc` families (Student-t, Beta, LogLogistic, Gamma, GEV)
and the special-function discrete families are not yet AD-differentiable with
respect to all arguments. Closing that gap (an AD rule for `beta_inc` and a
dual-safe t-CDF) is tracked as future work; R itself only provides analytic
gradients for the Normal, Logistic and Student-t families.

## Intentional divergences

Three families deliberately differ from R (a corrected log-logistic DSS, the GEV
Gumbel limit, and the ensemble population variance). These are documented on the
[Differences from R](@ref differences-from-r) page.
