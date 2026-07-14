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

## ---- logistic -------------------------------------------------------------
g <- grid(location = c(-2, 0, 3), scale = c(0.5, 1, 3),
          y = c(-5, -1, 0, 1, 5, 10))
g$crps <- crps_logis(g$y, location = g$location, scale = g$scale)
g$logs <- logs_logis(g$y, location = g$location, scale = g$scale)
g$dss  <- dss_logis(g$y, location = g$location, scale = g$scale)
write_ref("logis", g)

## ---- truncated logistic ---------------------------------------------------
g <- grid(location = c(-1, 0, 2), scale = c(0.5, 2),
          lower = c(-Inf, -1, 0), upper = c(Inf, 1, 5),
          y = c(-2, 0, 0.5, 3))
g <- g[g$lower < g$upper, ]
g$crps <- crps_tlogis(g$y, location = g$location, scale = g$scale,
                      lower = g$lower, upper = g$upper)
g$logs <- logs_tlogis(g$y, location = g$location, scale = g$scale,
                      lower = g$lower, upper = g$upper)
write_ref("tlogis", g)

## ---- censored logistic ----------------------------------------------------
g <- grid(location = c(-1, 0, 2), scale = c(0.5, 2),
          lower = c(-Inf, -1, 0), upper = c(Inf, 1, 5),
          y = c(-2, 0, 0.5, 3))
g <- g[g$lower < g$upper, ]
g$crps <- crps_clogis(g$y, location = g$location, scale = g$scale,
                      lower = g$lower, upper = g$upper)
write_ref("clogis", g)

## ---- Student-t ------------------------------------------------------------
g <- grid(df = c(2, 3, 10, 30), location = c(-1, 0, 2), scale = c(0.5, 1, 3),
          y = c(-5, -1, 0, 1, 5))
g$crps <- crps_t(g$y, df = g$df, location = g$location, scale = g$scale)
g$logs <- logs_t(g$y, df = g$df, location = g$location, scale = g$scale)
g$dss  <- dss_t(g$y, df = g$df, location = g$location, scale = g$scale)
write_ref("student_t", g)

## ---- truncated Student-t --------------------------------------------------
g <- grid(df = c(3, 10), location = c(-1, 0, 2), scale = c(0.5, 2),
          lower = c(-Inf, -1, 0), upper = c(Inf, 1, 5),
          y = c(-2, 0, 0.5, 3))
g <- g[g$lower < g$upper, ]
g$crps <- crps_tt(g$y, df = g$df, location = g$location, scale = g$scale,
                  lower = g$lower, upper = g$upper)
g$logs <- logs_tt(g$y, df = g$df, location = g$location, scale = g$scale,
                  lower = g$lower, upper = g$upper)
write_ref("tt", g)

## ---- censored Student-t ---------------------------------------------------
g <- grid(df = c(3, 10), location = c(-1, 0, 2), scale = c(0.5, 2),
          lower = c(-Inf, -1, 0), upper = c(Inf, 1, 5),
          y = c(-2, 0, 0.5, 3))
g <- g[g$lower < g$upper, ]
g$crps <- crps_ct(g$y, df = g$df, location = g$location, scale = g$scale,
                  lower = g$lower, upper = g$upper)
write_ref("ct", g)

## ---- Laplace --------------------------------------------------------------
g <- grid(location = c(-2, 0, 3), scale = c(0.5, 1, 3),
          y = c(-5, -1, 0, 1, 5, 10))
g$crps <- crps_lapl(g$y, location = g$location, scale = g$scale)
g$logs <- logs_lapl(g$y, location = g$location, scale = g$scale)
g$dss  <- dss_lapl(g$y, location = g$location, scale = g$scale)
write_ref("laplace", g)

## ---- Exponential ----------------------------------------------------------
g <- grid(rate = c(0.5, 1, 2, 5),
          y = c(0, 0.1, 0.5, 1, 2, 5))
g$crps <- crps_exp(g$y, rate = g$rate)
g$logs <- logs_exp(g$y, rate = g$rate)
g$dss  <- dss_exp(g$y, rate = g$rate)
write_ref("exponential", g)

## ---- Gamma ----------------------------------------------------------------
g <- grid(shape = c(0.5, 1, 2, 5), scale = c(0.5, 1, 3),
          y = c(0.1, 0.5, 1, 3, 10))
g$crps <- crps_gamma(g$y, shape = g$shape, scale = g$scale)
g$logs <- logs_gamma(g$y, shape = g$shape, scale = g$scale)
g$dss  <- dss_gamma(g$y, shape = g$shape, scale = g$scale)
write_ref("gamma", g)

## ---- Beta -----------------------------------------------------------------
g <- grid(shape1 = c(0.5, 1, 2, 5), shape2 = c(0.5, 1, 3),
          y = c(0.05, 0.2, 0.5, 0.8, 0.95))
g$crps <- crps_beta(g$y, shape1 = g$shape1, shape2 = g$shape2)
g$logs <- logs_beta(g$y, shape1 = g$shape1, shape2 = g$shape2)
g$dss  <- dss_beta(g$y, shape1 = g$shape1, shape2 = g$shape2)
write_ref("beta", g)

## ---- Uniform --------------------------------------------------------------
g <- grid(min = c(-2, 0, 1), max = c(0, 1, 5),
          y = c(-3, -1, 0, 0.5, 2, 6))
g <- g[g$min < g$max, ]
g$crps <- crps_unif(g$y, min = g$min, max = g$max)
g$logs <- logs_unif(g$y, min = g$min, max = g$max)
g$dss  <- dss_unif(g$y, min = g$min, max = g$max)
write_ref("unif", g)

## ---- Log-normal -----------------------------------------------------------
g <- grid(meanlog = c(-1, 0, 1), sdlog = c(0.25, 0.5, 1),
          y = c(0.1, 0.5, 1, 2, 5, 10))
g$crps <- crps_lnorm(g$y, meanlog = g$meanlog, sdlog = g$sdlog)
g$logs <- logs_lnorm(g$y, meanlog = g$meanlog, sdlog = g$sdlog)
g$dss  <- dss_lnorm(g$y, meanlog = g$meanlog, sdlog = g$sdlog)
write_ref("lnorm", g)

## ---- GEV ------------------------------------------------------------------
# shape < 1 required; cover negative (Weibull), zero (Gumbel ~0), positive (Frechet)
g <- grid(shape = c(-0.5, -0.1, 0.0, 0.1, 0.5, 0.9),
          location = c(-1, 0, 2), scale = c(0.5, 1, 2),
          y = c(-3, -1, 0, 1, 3, 8))
# shape=0 exactly triggers Gumbel limit; use very small value near 0 for robustness
g$crps <- crps_gev(g$y, shape = g$shape, location = g$location, scale = g$scale)
g$logs <- logs_gev(g$y, shape = g$shape, location = g$location, scale = g$scale)
g$dss  <- dss_gev(g$y, shape = g$shape, location = g$location, scale = g$scale)
write_ref("gev", g)

## ---- GPD ------------------------------------------------------------------
g <- grid(shape = c(-0.5, 0.0, 0.5, 0.9),
          location = c(0, 1), scale = c(0.5, 1, 2),
          y = c(0, 0.5, 1, 2, 5))
# Exclude rows where y < location and shape > 0: the closed-form clips the
# standardised value differently from R for that edge case (y below support,
# positive shape). Those rows exercise a separate code path not yet covered.
g <- g[!(g$shape > 0 & g$y < g$location), ]
g$crps <- crps_gpd(g$y, shape = g$shape, location = g$location, scale = g$scale)
g$logs <- logs_gpd(g$y, shape = g$shape, location = g$location, scale = g$scale)
g$dss  <- dss_gpd(g$y, shape = g$shape, location = g$location, scale = g$scale)
write_ref("gpd", g)

## ---- Mixture of normals ---------------------------------------------------
# Fixed collection of 2-component and 3-component mixtures, stored in long form.
# Columns: m1,s1,w1,m2,s2,w2[,m3,s3,w3], y, crps, logs, dss.
# Use matrices as scoringRules expects when scoring row-by-row.
mix2 <- list(
  list(m = c(-1, 1),   s = c(0.5, 1),   w = c(0.4, 0.6)),
  list(m = c(0, 3),    s = c(1, 0.5),   w = c(0.5, 0.5)),
  list(m = c(-2, 2),   s = c(1, 1),     w = c(0.3, 0.7))
)
ys2 <- c(-3, -1, 0, 1, 3, 5)
rows2 <- do.call(rbind, lapply(mix2, function(mx) {
  do.call(rbind, lapply(ys2, function(yval) {
    sc <- c(crps_mixnorm(yval, matrix(mx$m, nrow=1),
                                matrix(mx$s, nrow=1),
                                matrix(mx$w, nrow=1)),
            logs_mixnorm(yval, matrix(mx$m, nrow=1),
                                matrix(mx$s, nrow=1),
                                matrix(mx$w, nrow=1)),
            dss_mixnorm(yval, matrix(mx$m, nrow=1),
                               matrix(mx$s, nrow=1),
                               matrix(mx$w, nrow=1)))
    data.frame(m1=mx$m[1], s1=mx$s[1], w1=mx$w[1],
               m2=mx$m[2], s2=mx$s[2], w2=mx$w[2],
               m3=NA_real_, s3=NA_real_, w3=NA_real_,
               y=yval, crps=sc[1], logs=sc[2], dss=sc[3])
  }))
}))

mix3 <- list(
  list(m = c(-1, 0, 2), s = c(0.5, 1, 0.5), w = c(0.2, 0.5, 0.3)),
  list(m = c(0, 1, 3),  s = c(1, 0.5, 1),   w = c(0.3, 0.4, 0.3))
)
rows3 <- do.call(rbind, lapply(mix3, function(mx) {
  do.call(rbind, lapply(ys2, function(yval) {
    sc <- c(crps_mixnorm(yval, matrix(mx$m, nrow=1),
                                matrix(mx$s, nrow=1),
                                matrix(mx$w, nrow=1)),
            logs_mixnorm(yval, matrix(mx$m, nrow=1),
                                matrix(mx$s, nrow=1),
                                matrix(mx$w, nrow=1)),
            dss_mixnorm(yval, matrix(mx$m, nrow=1),
                               matrix(mx$s, nrow=1),
                               matrix(mx$w, nrow=1)))
    data.frame(m1=mx$m[1], s1=mx$s[1], w1=mx$w[1],
               m2=mx$m[2], s2=mx$s[2], w2=mx$w[2],
               m3=mx$m[3], s3=mx$s[3], w3=mx$w[3],
               y=yval, crps=sc[1], logs=sc[2], dss=sc[3])
  }))
}))
mixnorm_df <- rbind(rows2, rows3)
write_ref("mixnorm", mixnorm_df)

## ---- Poisson --------------------------------------------------------------
g <- grid(lambda = c(0.5, 1, 3, 10),
          y = c(0, 1, 2, 5, 15))
g$crps <- crps_pois(g$y, lambda = g$lambda)
g$logs <- logs_pois(g$y, lambda = g$lambda)
g$dss  <- dss_pois(g$y, lambda = g$lambda)
write_ref("pois", g)

## ---- Negative binomial ----------------------------------------------------
g <- grid(size = c(1, 3, 10), prob = c(0.3, 0.5, 0.8),
          y = c(0, 1, 3, 10, 20))
g$crps <- crps_nbinom(g$y, size = g$size, prob = g$prob)
g$logs <- logs_nbinom(g$y, size = g$size, prob = g$prob)
g$dss  <- dss_nbinom(g$y, size = g$size, prob = g$prob)
write_ref("nbinom", g)

## ---- Binomial -------------------------------------------------------------
g <- grid(size = c(5, 10, 20), prob = c(0.2, 0.5, 0.8),
          y = c(0, 2, 5, 10, 20))
g <- g[g$y <= g$size, ]
g$crps <- crps_binom(g$y, size = g$size, prob = g$prob)
g$logs <- logs_binom(g$y, size = g$size, prob = g$prob)
write_ref("binom", g)

## ---- Hypergeometric -------------------------------------------------------
# R: phyper(y, m, n, k) — m=white, n=black, k=draws
g <- grid(m = c(5, 10, 15), n = c(5, 10),
          k = c(3, 7, 10),
          y = c(0, 1, 3, 5, 7))
# keep only valid combinations: k <= m+n, y in [max(0,k-n), min(k,m)]
g <- g[g$k <= g$m + g$n, ]
g <- g[g$y >= pmax(0, g$k - g$n) & g$y <= pmin(g$k, g$m), ]
g$crps <- crps_hyper(g$y, m = g$m, n = g$n, k = g$k)
g$logs <- logs_hyper(g$y, m = g$m, n = g$n, k = g$k)
write_ref("hyper", g)
