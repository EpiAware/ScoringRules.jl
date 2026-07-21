"""
    ADFixtures

Shared AD gradient scenarios and backend metadata for ScoringRules. Used by
`test/ad/runtests.jl`.

Scenarios cover `crps`, the generic (Distributions.jl) forms of `logs`/`dss`,
and the sample (ensemble) forms of `logs`/`dss`, across one closed-form family
per structural class the package supports: a plain continuous family
(Normal, Logistic, Gamma, Exponential, Laplace), a bounded-support variant
(`Truncated`, `Censored`), an extreme-value family (GEV, GPD), a finite
mixture, this package's own two-piece type (`TwoPieceNormal`), and a discrete
family (Poisson).

Six scenarios were once declared broken rather than omitted, for two related
reasons; all six are now fixed and their scenarios below are normal
(non-broken), each a regression test for its issue's fix:

- Gamma, GEV and Poisson `crps` (#11) all routed through
  `SpecialFunctions.gamma_inc` with a *differentiated* shape/rate argument in
  its Dual-incompatible first slot — directly for Gamma and GEV, and via
  `Distributions.cdf(Poisson, ·)` for Poisson (whose `StatsFuns.gammaccdf`
  shares one type parameter across all three arguments, so promoting the
  differentiated rate forces the untouched integer shape to the same Dual
  type by ordinary promotion).
- Student-t, Beta and LogLogistic `crps` (#6) all routed through
  `SpecialFunctions.beta_inc` the same way — directly for Beta and
  LogLogistic, and via `Distributions.cdf(TDist, ·)` for Student-t (which
  shares StatsFuns' type-parameter-promotion pattern through its own
  `tdistcdf -> fdistccdf -> betaccdf` chain).

Both families are fixed by routing through `EpiAwareADTools.cdf_ad_safe`
instead, whose `_gamma_cdf`/`_beta_cdf` type their arguments independently
rather than sharing one promoted type. The Student-t fix additionally needed
an explicit guard at the standardised observation `z == 0` (see
`crps/student.jl`'s `_t_cdf`): there the true derivative is 0, but the naive
composition hits an IEEE `Inf * 0.0 = NaN` rather than that limit.

A scenario declared broken skips the (possibly-erroring) reference gradient
computation unconditionally rather than attempting and catching, so `res1` is
`nothing` and `check_broken` marks it `@test_broken` on every backend by
construction; each stays in the registry as a standing, honest record of its
gap until the fix lands (as all six now have).

All scenarios run across the ForwardDiff / ReverseDiff / Enzyme (forward and
reverse) / Mooncake (forward and reverse) backend matrix declared in
`backends()`. The reference is computed with ForwardDiff and matched by the
other backends to `rtol = 5e-2, atol = 1e-6` (the harness defaults).
"""
module ADFixtures

using ScoringRules
using Distributions: Beta, Exponential, Gamma, GeneralizedExtremeValue,
                     GeneralizedPareto, Laplace, LogLogistic, Logistic,
                     MixtureModel, Normal, Poisson, TDist, censored, truncated
using ADTypes: ADTypes, AutoForwardDiff, AutoReverseDiff, AutoMooncake,
               AutoMooncakeForward, AutoEnzyme
using DifferentiationInterface: DifferentiationInterface, Constant
import DifferentiationInterfaceTest as DIT
import ForwardDiff, ReverseDiff, Mooncake, Enzyme

export scenarios, backends, broken_scenario_names,
       backend_broken_scenarios, backend_skip_scenarios

# ForwardDiff reference gradient for a scenario function.
function _reference(f, θ, contexts)
    return DifferentiationInterface.gradient(
        f, AutoForwardDiff(), θ, contexts...)
end

"""
    backends()

The AD backends to test, as `(; name, backend)` named tuples. Matches every
backend `test/ad/scenarios.jl` emits a testitem for.
"""
function backends()
    return [
        (name = "ForwardDiff", backend = AutoForwardDiff()),
        (name = "ReverseDiff (tape)",
            backend = AutoReverseDiff(compile = false)),
        (name = "Enzyme forward",
            backend = AutoEnzyme(
                mode = Enzyme.set_runtime_activity(Enzyme.Forward))),
        (name = "Enzyme reverse",
            backend = AutoEnzyme(
                mode = Enzyme.set_runtime_activity(Enzyme.Reverse))),
        (name = "Mooncake reverse", backend = AutoMooncake(config = nothing)),
        (name = "Mooncake forward", backend = AutoMooncakeForward())
    ]
end

# Scenario names whose `crps` cannot be reference-differentiated at all: see
# the module docstring. Empty now that #6 and #11 are both fixed; kept as a
# `Ref` (rather than removed) so a future gap has the same standing place to
# declare itself, next to the scenario that would declare it, not duplicated
# here.
const _BROKEN_NAMES = Ref(String[])

"Scenario names broken on every backend. Empty: #6 and #11 (the two classes
of families whose `crps` routed a differentiated shape/rate argument through
`SpecialFunctions.gamma_inc`/`beta_inc`'s Dual-incompatible first slot) are
both fixed; see the module docstring."
broken_scenario_names() = _BROKEN_NAMES[]

"Per-backend broken scenario names (`Dict{String, Set{String}}`). Empty until
a real per-backend run of `test/ad/scenarios.jl` shows a scenario that passes
on ForwardDiff but fails on a specific other backend; see the PR that added
this registry for the run that first populated (or left empty) this dict."
backend_broken_scenarios() = Dict{String, Set{String}}()

"Per-backend scenario names too unstable to run at all (crashes rather than a
wrong answer). Empty for the same reason as `backend_broken_scenarios`."
backend_skip_scenarios() = Dict{String, Set{String}}()

"""
    scenarios(; with_reference = false, category = :marginal)

The AD gradient scenarios. Each is a `DIT.Scenario{:gradient, :out}` whose
`res1` carries a ForwardDiff reference when `with_reference = true`. ScoringRules
does not distinguish scenario groups, so `category` is accepted (the `ADRegistry`
contract requires it) and ignored.
"""
function scenarios(; with_reference::Bool = false, category::Symbol = :marginal)
    out = DIT.Scenario{:gradient, :out}[]
    broken_names = String[]

    # `broken = true` skips the (possibly-erroring) reference computation
    # unconditionally rather than attempting and catching: #6's whole point is
    # that ForwardDiff itself cannot differentiate these, so even computing
    # `res1` under `with_reference = true` would throw before the harness gets
    # a chance to sort the scenario into its broken bucket.
    function _push!(name, f, θ₀, contexts; broken::Bool = false)
        broken && push!(broken_names, name)
        res1 = (with_reference && !broken) ? _reference(f, θ₀, contexts) :
               nothing
        prep_args = (; x = θ₀, contexts = contexts)
        push!(out,
            res1 === nothing ?
            DIT.Scenario{:gradient, :out}(
                f, θ₀, contexts...; prep_args = prep_args, name = name) :
            DIT.Scenario{:gradient, :out}(
                f, θ₀, contexts...;
                res1 = res1, prep_args = prep_args, name = name))
    end

    # --- plain continuous families: one obs vector per scenario, gradient
    # w.r.t. the distribution's own parameters ------------------------------

    _push!("Normal crps",
        (θ, obs) -> sum(y -> crps(Normal(θ[1], θ[2]), y), obs),
        [1.0, 2.0], (Constant([-1.0, 0.5, 2.0, 4.0]),))

    _push!("Logistic crps",
        (θ, obs) -> sum(y -> crps(Logistic(θ[1], θ[2]), y), obs),
        [0.5, 1.5], (Constant([-1.0, 0.5, 2.0, 4.0]),))

    # #11 (fixed): `_crps_gamma` now goes through `cdf_ad_safe` rather than a
    # direct `cdf`/`gamma_inc(shape, ·)` call, which was not dual-safe in
    # `shape`.
    _push!("Gamma crps",
        (θ, obs) -> sum(y -> crps(Gamma(θ[1], θ[2]), y), obs),
        [2.0, 1.5], (Constant([0.5, 1.2, 2.5, 4.0]),))

    _push!("Exponential crps",
        (θ, obs) -> sum(y -> crps(Exponential(θ[1]), y), obs),
        [1.5], (Constant([0.3, 1.0, 2.5]),))

    _push!("Laplace crps",
        (θ, obs) -> sum(y -> crps(Laplace(θ[1], θ[2]), y), obs),
        [0.0, 1.0], (Constant([-2.0, 0.5, 3.0]),))

    # --- bounded-support structural variants: location/scale differentiated,
    # the truncation/censoring bounds held fixed --------------------------

    _push!("Truncated Normal crps",
        (θ, obs) -> sum(
            y -> crps(truncated(Normal(θ[1], θ[2]), 0.0, 5.0), y), obs),
        [1.0, 1.0], (Constant([0.5, 2.0, 4.0]),))

    _push!("Censored Normal crps",
        (θ, obs) -> sum(
            y -> crps(censored(Normal(θ[1], θ[2]), 0.0, 5.0), y), obs),
        [1.0, 1.0], (Constant([0.0, 2.0, 5.0]),))

    # --- extreme-value family: shape held away from 0 (ξ = 0.2 / 0.3) to
    # exercise the general closed form, not the ξ ≈ 0 limiting-case branch --

    # #11 (fixed): `_crps_gev` now goes through `cdf_ad_safe` rather than a
    # direct `gamma_inc(1 - shape, x)` call, which was not dual-safe in the
    # first argument.
    _push!("GEV crps",
        (θ, obs) -> sum(
            y -> crps(GeneralizedExtremeValue(θ[1], θ[2], θ[3]), y), obs),
        [0.0, 1.0, 0.2], (Constant([-0.5, 0.5, 2.0]),))

    # GPD's closed form uses only `^`/`exp`/`max`/`min` (no `gamma_inc`), so
    # unlike GEV above it is not affected by #11 — confirmed differentiable.
    _push!("GPD crps",
        (θ, obs) -> sum(y -> crps(GeneralizedPareto(θ[1], θ[2], θ[3]), y), obs),
        [0.0, 1.0, 0.3], (Constant([0.1, 1.0, 3.0]),))

    # --- finite mixture: two Normal components, the mixing weight itself
    # differentiated (θ[5], with the second weight `1 - θ[5]`) --------------

    _push!("Mixture-of-normals crps",
        (θ, obs) -> sum(
            y -> crps(
                MixtureModel(
                    [Normal(θ[1], θ[2]), Normal(θ[3], θ[4])],
                    [θ[5], 1 - θ[5]]), y),
            obs),
        [-1.0, 1.0, 2.0, 1.5, 0.4], (Constant([-2.0, 0.0, 2.0, 4.0]),))

    # --- two-piece: this package's own type (src/distributions/twopiece.jl),
    # not a Distributions.jl import ------------------------------------------

    # `_crps_2pnorm` splits at `y == location` via `min(yc, 0)`/`max(yc, 0)`
    # (`yc = y - location`), a genuine kink with an ambiguous subgradient
    # there; keep every obs strictly away from `location` (0.0) so the
    # gradient check lands on the smooth part of each arm, not the kink.
    _push!("TwoPieceNormal crps",
        (θ, obs) -> sum(y -> crps(TwoPieceNormal(θ[1], θ[2], θ[3]), y), obs),
        [0.0, 1.0, 2.0], (Constant([-2.0, 0.5, 3.0]),))

    # --- discrete family: λ differentiated, obs real-valued (crps is defined
    # against the discrete step CDF at any real y, not just integers) -------

    # #11 (fixed): `_crps_pois` now goes through `cdf_ad_safe` rather than a
    # direct `cdf(Poisson(lambda), y)` call, which routed through
    # `StatsFuns.gammaccdf`'s shared-type-parameter promotion and was not
    # dual-safe in `lambda` — not the `besseli` calls elsewhere in
    # `_crps_pois`, which are fine.
    _push!("Poisson crps",
        (θ, obs) -> sum(y -> crps(Poisson(θ[1]), y), obs),
        [3.0], (Constant([0.0, 2.0, 3.0, 5.0]),))

    # --- generic (any UnivariateDistribution) logs/dss, via the analytic
    # -logpdf / mean-var forms in src/generics.jl ----------------------------

    _push!("Normal logs",
        (θ, obs) -> sum(y -> logs(Normal(θ[1], θ[2]), y), obs),
        [1.0, 2.0], (Constant([-1.0, 0.5, 2.0]),))

    _push!("Normal dss",
        (θ, obs) -> sum(y -> dss(Normal(θ[1], θ[2]), y), obs),
        [1.0, 2.0], (Constant([-1.0, 0.5, 2.0]),))

    # --- sample (ensemble) logs/dss: gradient w.r.t. the ensemble itself,
    # the observation held fixed. These are the two functions #9 proposes
    # adding member weights to — a regression harness here covers the
    # unweighted path that change must not disturb ---------------------------

    _push!("Sample logs wrt ensemble",
        (dat, y) -> logs(dat, y),
        [0.5, 1.2, -0.3, 2.1, 0.0], (Constant(0.8),))

    _push!("Sample dss wrt ensemble",
        (dat, y) -> dss(dat, y),
        [0.5, 1.2, -0.3, 2.1, 0.0], (Constant(0.8),))

    # --- #6 (fixed): Student-t, Beta and LogLogistic `crps` now go through
    # cdf_ad_safe rather than a direct beta_inc/cdf(TDist) call, which was
    # not dual-safe in the differentiated shape argument -------------------

    _push!("Student-t crps",
        (θ, obs) -> sum(y -> crps(TDist(θ[1]), y), obs),
        [5.0], (Constant([0.5, -0.3, 1.5]),))

    _push!("Beta crps",
        (θ, obs) -> sum(y -> crps(Beta(θ[1], θ[2]), y), obs),
        [2.0, 3.0], (Constant([0.2, 0.5, 0.8]),))

    _push!("LogLogistic crps",
        (θ, obs) -> sum(y -> crps(LogLogistic(θ[1], θ[2]), y), obs),
        [2.0, 3.0], (Constant([0.5, 1.5, 3.0]),))

    _BROKEN_NAMES[] = broken_names
    return out
end

end # module ADFixtures
