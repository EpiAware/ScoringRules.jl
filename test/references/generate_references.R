#!/usr/bin/env Rscript
# Generate reference score values from the R `scoringRules` package.
#
# The Julia test suite compares its output against these committed CSVs so that
# CI needs no R installation, while the values are anchored to the real
# reference implementation. Regenerate after touching a scored family:
#
#     Rscript test/references/generate_references.R
#
# Each family writes one CSV under test/references/data/ whose columns are the
# natural parameters of that family plus the observation `y` and one column per
# scoring rule (crps/logs/dss) that scoringRules provides for it.

suppressMessages(library(scoringRules))

# Locate this script's directory so it can be run from anywhere.
script_path <- sub("^--file=", "",
                   grep("^--file=", commandArgs(FALSE), value = TRUE))
script_dir <- if (length(script_path)) dirname(script_path) else "test/references"
outdir <- file.path(script_dir, "data")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

write_ref <- function(name, df) {
  path <- file.path(outdir, paste0(name, ".csv"))
  write.csv(df, path, row.names = FALSE)
  cat(sprintf("wrote %s (%d rows)\n", path, nrow(df)))
}

# Expand a named list of parameter vectors into a full grid data.frame.
grid <- function(...) expand.grid(..., KEEP.OUT.ATTRS = FALSE,
                                  stringsAsFactors = FALSE)

## ---- normal ---------------------------------------------------------------
g <- grid(mean = c(-3, 0, 2.5), sd = c(0.5, 1, 4),
          y = c(-6, -1, 0, 0.7, 3, 8))
g$crps <- crps_norm(g$y, mean = g$mean, sd = g$sd)
g$logs <- logs_norm(g$y, mean = g$mean, sd = g$sd)
g$dss  <- dss_norm(g$y, mean = g$mean, sd = g$sd)
write_ref("norm", g)

## ---- truncated normal -----------------------------------------------------
g <- grid(location = c(-1, 0, 2), scale = c(0.7, 2),
          lower = c(-Inf, -1, 0), upper = c(Inf, 1, 5),
          y = c(-2, 0, 0.5, 3))
g <- g[g$lower < g$upper, ]
g$crps <- crps_tnorm(g$y, location = g$location, scale = g$scale,
                     lower = g$lower, upper = g$upper)
# clamp y into [lower, upper] for logs (density is 0 outside; logs = Inf there)
g$logs <- logs_tnorm(g$y, location = g$location, scale = g$scale,
                     lower = g$lower, upper = g$upper)
write_ref("tnorm", g)

## ---- censored normal ------------------------------------------------------
g <- grid(location = c(-1, 0, 2), scale = c(0.7, 2),
          lower = c(-Inf, -1, 0), upper = c(Inf, 1, 5),
          y = c(-2, 0, 0.5, 3))
g <- g[g$lower < g$upper, ]
g$crps <- crps_cnorm(g$y, location = g$location, scale = g$scale,
                     lower = g$lower, upper = g$upper)
write_ref("cnorm", g)

## ---- generalised truncated/censored normal --------------------------------
g <- grid(location = c(0, 1.5), scale = c(1, 3),
          lower = c(-Inf, -1, 0), upper = c(Inf, 2),
          lmass = c(0, 0.1), umass = c(0, 0.2),
          y = c(-2, 0, 1, 4))
g <- g[g$lower < g$upper, ]
# point masses only make sense at finite bounds
g <- g[!(is.infinite(g$lower) & g$lmass != 0), ]
g <- g[!(is.infinite(g$upper) & g$umass != 0), ]
g$crps <- crps_gtcnorm(g$y, location = g$location, scale = g$scale,
                       lower = g$lower, upper = g$upper,
                       lmass = g$lmass, umass = g$umass)
write_ref("gtcnorm", g)
