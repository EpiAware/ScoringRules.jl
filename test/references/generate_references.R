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
# Includes observations below the support with positive shape (y < location),
# which the closed form now handles exactly as R does.
g <- grid(shape = c(-0.5, 0.0, 0.5, 0.9),
          location = c(0, 1), scale = c(0.5, 1, 2),
          y = c(-2, -0.5, 0, 0.5, 1, 2, 5))
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

## ============================================================
## SECTION 2: sample-based, multivariate, weighted, quantile/
## interval, RPS, and extra-distribution scores
## ============================================================

# Helper: write a fixed-seed ensemble as a CSV.
write_ens <- function(name, mat) {
  # mat is a numeric matrix (d x m for multivariate, or 1 x m for univariate).
  # Written with row/col indices so the reader knows d and m.
  df <- as.data.frame(mat)
  colnames(df) <- paste0("m", seq_len(ncol(df)))
  rownames(df) <- NULL
  write_ref(name, df)
}

## ---- sample univariate (crps/logs/dss vs R crps_sample / logs_sample / dss_sample) ----
# Four ensembles of size 20 drawn with set.seed().
set.seed(123)
ens_ids <- 1:4
ens_list <- lapply(ens_ids, function(i) rnorm(20, mean = i * 0.5 - 1, sd = i * 0.3 + 0.5))
for (i in ens_ids) write_ens(sprintf("ens_univ_%d", i), t(matrix(ens_list[[i]])))

ys_univ <- c(-2, -0.5, 0, 1, 3)
methods  <- c("edf", "kde")
rows_univ <- do.call(rbind, lapply(ens_ids, function(i) {
  dat <- ens_list[[i]]
  do.call(rbind, lapply(ys_univ, function(yval) {
    data.frame(
      ens_id  = i,
      y       = yval,
      method  = "edf",
      crps    = crps_sample(yval, dat = dat, method = "edf"),
      logs    = logs_sample(yval, dat = dat),
      dss     = dss_sample(yval, dat = dat),
      crps_kde = crps_sample(yval, dat = dat, method = "kde")
    )
  }))
}))
write_ref("sample_univ_scores", rows_univ)

## ---- sample multivariate (es / vs / mmds vs R es_sample / vs_sample / mmds_sample) ----
# Two 3-dimensional ensembles of size 15 each.
set.seed(456)
d_mv <- 3L; m_mv <- 15L
ens_mv1 <- matrix(rnorm(d_mv * m_mv, mean = 0, sd = 1), nrow = d_mv, ncol = m_mv)
ens_mv2 <- matrix(rnorm(d_mv * m_mv, mean = c(1, -1, 0.5), sd = c(1, 2, 0.5)), nrow = d_mv, ncol = m_mv)
write_ens("ens_mv_1", ens_mv1)
write_ens("ens_mv_2", ens_mv2)

ys_mv <- list(c(0, 0, 0), c(1, -1, 0.5), c(-2, 2, -1))
ps_vs  <- c(0.5, 1.0)
rows_mv <- do.call(rbind, lapply(1:2, function(eid) {
  Xmat <- if (eid == 1) ens_mv1 else ens_mv2
  do.call(rbind, lapply(seq_along(ys_mv), function(yi) {
    y <- ys_mv[[yi]]
    do.call(rbind, lapply(ps_vs, function(p) {
      data.frame(
        ens_id = eid,
        y_id   = yi,
        y1     = y[1], y2 = y[2], y3 = y[3],
        p_vs   = p,
        es     = es_sample(y, dat = Xmat),
        vs     = vs_sample(y, dat = Xmat, p = p),
        mmds   = mmds_sample(y, dat = Xmat)
      )
    }))
  }))
}))
write_ref("sample_mv_scores", rows_mv)

## ---- weighted univariate (twcrps / owcrps vs R twcrps_sample / owcrps_sample) ----
# Use univariate ensembles 1 and 2 from above.
ab_pairs <- list(c(-Inf, Inf), c(-1, 1), c(0, 2), c(-0.5, Inf), c(-Inf, 0.5))
rows_wtu <- do.call(rbind, lapply(1:2, function(eid) {
  dat <- ens_list[[eid]]
  do.call(rbind, lapply(ys_univ, function(yval) {
    do.call(rbind, lapply(seq_along(ab_pairs), function(ki) {
      ab <- ab_pairs[[ki]]
      a  <- ab[1]; b <- ab[2]
      tw <- twcrps_sample(yval, dat = dat, a = a, b = b)
      ow <- owcrps_sample(yval, dat = dat, a = a, b = b)
      data.frame(ens_id=eid, y=yval, ab_id=ki, a=a, b=b, twcrps=tw, owcrps=ow)
    }))
  }))
}))
write_ref("sample_weighted_univ", rows_wtu)

## ---- weighted multivariate (twes/owes/twvs/owvs/twmmds/owmmds) ----
# Use multivariate ensembles; scalar [a,b] intervals broadcast to all dims.
ab_pairs_mv <- list(c(-Inf, Inf), c(-1, 1), c(0, 2))
rows_wtmv <- do.call(rbind, lapply(1:2, function(eid) {
  Xmat <- if (eid == 1) ens_mv1 else ens_mv2
  do.call(rbind, lapply(seq_along(ys_mv), function(yi) {
    y <- ys_mv[[yi]]
    do.call(rbind, lapply(seq_along(ab_pairs_mv), function(ki) {
      ab <- ab_pairs_mv[[ki]]
      a  <- ab[1]; b <- ab[2]
      data.frame(
        ens_id = eid, y_id = yi,
        y1 = y[1], y2 = y[2], y3 = y[3],
        ab_id = ki, a = a, b = b,
        p_vs  = 0.5,
        twes  = twes_sample(y, dat = Xmat, a = a, b = b),
        owes  = owes_sample(y, dat = Xmat, a = a, b = b),
        twvs  = twvs_sample(y, dat = Xmat, a = a, b = b, p = 0.5),
        owvs  = owvs_sample(y, dat = Xmat, a = a, b = b, p = 0.5),
        twmmds = twmmds_sample(y, dat = Xmat, a = a, b = b),
        owmmds = owmmds_sample(y, dat = Xmat, a = a, b = b)
      )
    }))
  }))
}))
write_ref("sample_weighted_mv", rows_wtmv)

## ---- quantile / interval scores (qs_quantiles / ints_quantiles / qs_sample / ints_sample) ----
q_levels <- c(0.1, 0.25, 0.5, 0.75, 0.9)
q_forecasts_list <- list(
  c(-1.28, -0.67, 0.0, 0.67, 1.28),   # standard normal quantiles
  c(-2.0,  -0.5,  1.0, 2.5,  4.0),
  c( 0.5,   1.0,  1.5, 2.0,  2.5)
)
ys_qs <- c(-2, 0, 1, 3)
# qs_quantiles takes one (x, alpha) pair at a time; loop over levels.
rows_qs <- do.call(rbind, lapply(seq_along(q_forecasts_list), function(qi) {
  qf <- q_forecasts_list[[qi]]
  do.call(rbind, lapply(ys_qs, function(yval) {
    scores <- mapply(function(x, a) qs_quantiles(yval, x = x, alpha = a),
                     qf, q_levels)
    int80 <- ints_quantiles(yval, x_lower = qf[1], x_upper = qf[5],
                             target_coverage = 0.8)
    int50 <- ints_quantiles(yval, x_lower = qf[2], x_upper = qf[4],
                             target_coverage = 0.5)
    data.frame(
      q_set = qi, y = yval,
      q1 = qf[1], q2 = qf[2], q3 = qf[3], q4 = qf[4], q5 = qf[5],
      qs_a1 = scores[1], qs_a2 = scores[2], qs_a3 = scores[3],
      qs_a4 = scores[4], qs_a5 = scores[5],
      ints_80 = int80, ints_50 = int50
    )
  }))
}))
write_ref("quantile_scores", rows_qs)

# Ensemble quantile / interval scores (qs_sample / ints_sample)
# Use univariate ensemble 1 and 2, alpha values = q_levels.
rows_qs_samp <- do.call(rbind, lapply(1:2, function(eid) {
  dat <- ens_list[[eid]]
  do.call(rbind, lapply(ys_univ, function(yval) {
    do.call(rbind, lapply(q_levels, function(alpha) {
      qs_val   <- qs_sample(yval, dat = dat, alpha = alpha)
      int_val  <- ints_sample(yval, dat = dat, target_coverage = 0.8)
      data.frame(ens_id=eid, y=yval, alpha=alpha, qs=qs_val, ints_80=int_val)
    }))
  }))
}))
write_ref("quantile_sample_scores", rows_qs_samp)

## ---- RPS (rps_probs vs Julia rps) ----
# x (R) = probability vector over K categories; y = observed category (1-indexed).
rps_cases <- list(
  list(p = c(0.3, 0.2, 0.5), K = 3),
  list(p = c(0.1, 0.1, 0.3, 0.5), K = 4),
  list(p = c(0.5, 0.3, 0.2), K = 3),
  list(p = c(0.25, 0.25, 0.25, 0.25), K = 4)
)
rows_rps <- do.call(rbind, lapply(seq_along(rps_cases), function(ci) {
  cas <- rps_cases[[ci]]
  do.call(rbind, lapply(1:cas$K, function(y) {
    data.frame(case_id=ci, K=cas$K, y=y,
               p1=cas$p[1], p2=cas$p[2], p3=cas$p[3],
               p4 = if (cas$K >= 4) cas$p[4] else NA_real_,
               rps = rps_probs(y, x = cas$p))
  }))
}))
write_ref("rps_scores", rows_rps)

## ---- extra distributions: LogLogistic, LogLaplace, TwoPieceNormal, TwoPieceExponential ----

# LogLogistic: R uses (locationlog, scalelog); Julia uses LogLogistic(α, β)
#   with α = exp(locationlog), β = 1/scalelog.
# dss_llogis is NOT compared (known NaN issue in R).
g <- grid(locationlog = c(-1, 0, 1),
          scalelog    = c(0.3, 0.5, 0.7),
          y           = c(0.1, 0.5, 1, 2, 5))
g$crps <- crps_llogis(g$y, locationlog = g$locationlog, scalelog = g$scalelog)
g$logs <- logs_llogis(g$y, locationlog = g$locationlog, scalelog = g$scalelog)
write_ref("llogis", g)

# LogLaplace: R uses (locationlog, scalelog); Julia uses LogLaplace(μ, σ)
#   with μ = locationlog, σ = scalelog (same parameterisation).
# dss_llapl works only when scalelog < 1/2; restrict to scalelog = 0.3.
g <- grid(locationlog = c(-1, 0, 1),
          scalelog    = c(0.3, 0.5, 0.7),
          y           = c(0.1, 0.5, 1, 2, 5))
g$crps <- crps_llapl(g$y, locationlog = g$locationlog, scalelog = g$scalelog)
g$logs <- logs_llapl(g$y, locationlog = g$locationlog, scalelog = g$scalelog)
# dss only when scalelog < 0.5 (finite variance required)
g$dss  <- ifelse(g$scalelog < 0.5,
                 dss_llapl(g$y, locationlog = g$locationlog, scalelog = g$scalelog),
                 NA_real_)
write_ref("llapl", g)

# TwoPieceNormal: R uses (scale1, scale2, location); Julia TwoPieceNormal(location, scale1, scale2).
g <- grid(location = c(-1, 0, 2),
          scale1   = c(0.5, 1, 2),
          scale2   = c(0.5, 1, 2),
          y        = c(-3, -1, 0, 1, 3))
g$crps <- crps_2pnorm(g$y, scale1 = g$scale1, scale2 = g$scale2, location = g$location)
g$logs <- logs_2pnorm(g$y, scale1 = g$scale1, scale2 = g$scale2, location = g$location)
write_ref("twopiecenorm", g)

# TwoPieceExponential: R uses (scale1, scale2, location); Julia TwoPieceExponential(location, scale1, scale2).
g <- grid(location = c(-1, 0, 2),
          scale1   = c(0.5, 1, 2),
          scale2   = c(0.5, 1, 2),
          y        = c(-3, -1, 0, 1, 3))
g$crps <- crps_2pexp(g$y, scale1 = g$scale1, scale2 = g$scale2, location = g$location)
g$logs <- logs_2pexp(g$y, scale1 = g$scale1, scale2 = g$scale2, location = g$location)
write_ref("twopieceexp", g)
