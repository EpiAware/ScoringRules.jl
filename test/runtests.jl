# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Main test entry. Discovers `@testitem`s (the managed QA testset under
# `test/package/` plus the package's own unit tests) with TestItemRunner. The
# `:ad`-tagged items live under `test/ad/` with their own environment and run in
# dedicated per-backend CI, so they are excluded here (see test/ad/runtests.jl).
#
# Filters:
#   skip_quality  — skip the QA testset (fast local iteration)
#   quality_only  — run only the QA testset
#   readme_only   — run only `:readme`-tagged items (README/tutorial tests)

using TestItemRunner

# `run_tests` roots discovery at this package's own `test/` tree rather than the
# whole package root, so a nested worktree checked out under the repo (the
# `worktrees/wt-*` convention) is never scanned and cannot inject test items or
# silently shadow a same-named `@testsnippet` (kit #191).
#
# This is TestItemRunner's public API, matching `test/ad/runtests.jl`. The
# equivalent kit helper drives an unexported TestItemRunner internal whose
# signature changed in TestItemRunner 1.2, so it breaks on any resolve that
# picks up a current release (kit #451).

if "skip_quality" in ARGS
    TestItemRunner.run_tests(@__DIR__;
        filter = ti -> !(:quality in ti.tags) && !(:ad in ti.tags))
elseif "quality_only" in ARGS
    TestItemRunner.run_tests(@__DIR__; filter = ti -> :quality in ti.tags)
elseif "readme_only" in ARGS
    TestItemRunner.run_tests(@__DIR__; filter = ti -> :readme in ti.tags)
else
    TestItemRunner.run_tests(@__DIR__; filter = ti -> !(:ad in ti.tags))
end
