#!/usr/bin/env Rscript
## Reference results for the zero adjusted continuous families ZAGA and
## ZAIG (gamlss.dist R/ZAGA.R, R/ZAIG.R). Outputs tests/reference/za_dpqr.json,
## tests/reference/za_fits.json and simulated data in tests/reference/data/.
## Mixed families randomise the quantile residuals of the zeros, so only the
## residuals of y > 0 are recorded.
suppressMessages({
  library(gamlss)
  library(jsonlite)
})

set.seed(2026)
dir.create("tests/reference/data", recursive = TRUE, showWarnings = FALSE)

num <- function(x) I(unname(as.numeric(x)))
chr <- function(x) I(unname(as.character(x)))

## ---- d/p/q
dpqr <- list()
adddpqr <- function(fam, params, xs, ps = c(0.01, 0.25, 0.5, 0.9, 0.99)) {
  dfun <- get(paste0("d", fam)); pfun <- get(paste0("p", fam))
  qfun <- get(paste0("q", fam))
  out <- list()
  for (i in seq_along(params)) {
    pa <- params[[i]]
    out[[i]] <- list(params = pa, x = num(xs),
                     d = num(do.call(dfun, c(list(x = xs), pa))),
                     dlog = num(do.call(dfun, c(list(x = xs), pa, list(log = TRUE)))),
                     p = num(do.call(pfun, c(list(q = xs), pa))),
                     pupper = num(do.call(pfun, c(list(q = xs), pa, list(lower.tail = FALSE)))),
                     ps = num(ps), q = num(do.call(qfun, c(list(p = ps), pa))))
  }
  dpqr[[fam]] <<- out
}
xs <- c(0, 0.01, 0.4, 1, 2.3, 6.5)
pars <- list(list(mu = 1, sigma = 1, nu = 0.1), list(mu = 2.5, sigma = 0.4, nu = 0.3),
             list(mu = 0.6, sigma = 1.6, nu = 0.75))
adddpqr("ZAGA", pars, xs)
adddpqr("ZAIG", pars, xs)
write_json(dpqr, "tests/reference/za_dpqr.json", digits = I(17), auto_unbox = TRUE)

## ---- simulated data
n <- 600
x1 <- round(runif(n), 4)
simdata <- function(name, df) {
  write.csv(df, file.path("tests/reference/data", paste0(name, ".csv")), row.names = FALSE)
  read.csv(file.path("tests/reference/data", paste0(name, ".csv")))
}
sim_zaga <- simdata("sim_zaga", data.frame(
  y = rZAGA(n, mu = exp(0.4 + 0.7 * x1), sigma = exp(-0.4 + 0.5 * x1), nu = plogis(-0.6 + 1.1 * x1)), x1 = x1))
sim_zaig <- simdata("sim_zaig", data.frame(
  y = rZAIG(n, mu = exp(0.4 + 0.7 * x1), sigma = 0.6, nu = plogis(-0.6 + 1.1 * x1)), x1 = x1))

## ---- fits (same record() fields as gen_reference.R)
fits <- list()
safe_se_vcov <- function(m) {
  v <- try(suppressWarnings(vcov(m, type = "se")), silent = TRUE)
  if (inherits(v, "try-error")) NULL else num(v)
}
se_qr <- function(m, what) {
  Qr <- m[[paste0(what, ".qr")]]
  p1 <- seq_len(m[[paste0(what, ".df")]])
  num(sqrt(diag(chol2inv(Qr$qr[p1, p1, drop = FALSE]))))
}
record <- function(name, m, datname, data, nd) {
  out <- list(family = m$family[1], data = datname, G.deviance = m$G.deviance,
              P.deviance = m$P.deviance, aic = m$aic, sbc = m$sbc,
              df.fit = m$df.fit, df.residual = m$df.residual, noObs = m$noObs,
              N = m$N, iter = m$iter, converged = m$converged,
              parameters = chr(m$parameters))
  for (p in m$parameters) {
    cf <- coef(m, p)
    out[[paste0("coef.", p)]] <- num(cf)
    out[[paste0("coefnames.", p)]] <- chr(names(cf))
    fv <- fitted(m, p)
    out[[paste0("fitted10.", p)]] <- num(head(fv, 10))
    out[[paste0("fittedsum.", p)]] <- sum(fv)
    out[[paste0("se_qr.", p)]] <- se_qr(m, p)
    out[[paste0("pred.", p, ".response")]] <-
      num(predict(m, what = p, newdata = nd, data = data, type = "response"))
  }
  rp <- resid(m)[data$y > 0]
  out$resid_pos10 <- num(head(rp, 10))
  out$resid_possum <- sum(rp)
  out$se_vcov <- safe_se_vcov(m)
  fits[[name]] <<- out
}
ctrl <- gamlss.control(trace = FALSE)
nd <- data.frame(x1 = c(0.05, 0.5, 0.95))

m <- gamlss(y ~ x1, sigma.formula = ~x1, nu.formula = ~x1, family = ZAGA, data = sim_zaga, control = ctrl)
record("zaga_sim", m, "sim_zaga", sim_zaga, nd)
m <- gamlss(y ~ 1, family = ZAGA, data = sim_zaga, control = ctrl)
record("zaga_sim_int", m, "sim_zaga", sim_zaga, nd)
m <- gamlss(y ~ x1, sigma.formula = ~x1, nu.formula = ~x1, family = ZAGA, data = sim_zaga,
            method = CG(), control = ctrl)
record("zaga_sim_cg", m, "sim_zaga", sim_zaga, nd)
m <- gamlss(y ~ x1, nu.formula = ~x1, family = ZAIG, data = sim_zaig, control = ctrl)
record("zaig_sim", m, "sim_zaig", sim_zaig, nd)
m <- gamlss(y ~ x1, sigma.formula = ~x1, nu.formula = ~1, family = ZAIG(nu.link = "probit"),
            data = sim_zaig, control = ctrl)
record("zaig_sim_probit", m, "sim_zaig", sim_zaig, nd)

write_json(list(meta = list(R = R.version.string, gamlss = as.character(packageVersion("gamlss")),
                            gamlss.dist = as.character(packageVersion("gamlss.dist"))),
                fits = fits),
           "tests/reference/za_fits.json", digits = I(17), auto_unbox = TRUE)
cat("za_dpqr.json and za_fits.json written\n")
