#!/usr/bin/env Rscript
## Reference results for the "going to zero" penalised B-spline smoother pbz()
## (Step 6).  pbz() is pb() plus a second, order-1 penalty that switches on
## when the fit collapses to edf <= lim, shrinking the smooth toward a constant.
##
## ONLY ML (default) and a fixed lambda are exercised: R gamlss 5.5-0's
## gamlss.pbz() aborts for df / GAIC / GCV (its inner regpen() is called with a
## lambda argument it does not accept -> "unused argument (lambda)"), so those
## have no reference behaviour.  The Python port rejects them for the same
## reason; see test_pbz.py / CHANGELOG.
##
## Each case embeds its own data so the Python test is hermetic and uses
## byte-identical inputs.  Output: tests/reference/pbz_fits.json
suppressMessages({
  library(gamlss)
  library(jsonlite)
})

num <- function(x) I(unname(as.numeric(x)))
chr <- function(x) I(unname(as.character(x)))

cases <- list()

## per-smoother lambda / edf from the coefSmo list of a fitted parameter
smo_field <- function(m, p, field) {
  cs <- m[[paste0(p, ".coefSmo")]]
  if (is.null(cs) || length(cs) == 0) return(NULL)
  num(sapply(cs, function(z) {
    v <- z[[field]]
    if (is.null(v)) NA_real_ else as.numeric(v)
  }))
}

record_pbz <- function(name, m, data, spec) {
  out <- list(
    spec = spec,
    data = lapply(data, num),
    family = m$family[1],
    parameters = chr(m$parameters),
    df.fit = m$df.fit, df.residual = m$df.residual,
    noObs = m$noObs, N = m$N, iter = m$iter, converged = m$converged,
    G.deviance = m$G.deviance, P.deviance = m$P.deviance,
    aic = m$aic, sbc = m$sbc
  )
  for (p in m$parameters) {
    cf <- coef(m, p)
    out[[paste0("coef.", p)]] <- num(cf)
    out[[paste0("coefnames.", p)]] <- chr(names(cf))
    out[[paste0("fitted.", p)]] <- num(fitted(m, p))
    out[[paste0("df.", p)]] <- m[[paste0(p, ".df")]]
    out[[paste0("nldf.", p)]] <- m[[paste0(p, ".nl.df")]]
    lam <- smo_field(m, p, "lambda")
    if (!is.null(lam)) {
      out[[paste0("lambda.", p)]] <- lam
      out[[paste0("edf.", p)]] <- smo_field(m, p, "edf")
    }
  }
  cases[[name]] <<- out
}

ctrl <- gamlss.control(trace = FALSE, n.cyc = 200)
newx <- c(15.5, 20.0, 25.3, 33.7, 40.0)  # mix of on-grid and between-grid x

add_pred <- function(name, m, data, nd, smo_fun_x = NULL) {
  cases[[name]]$pred.newdata <<- lapply(as.list(nd), num)
  for (p in m$parameters) {
    cases[[name]][[paste0("pred.link.", p)]] <<-
      num(predict(m, what = p, newdata = nd, data = data, type = "link"))
    cases[[name]][[paste0("pred.resp.", p)]] <<-
      num(predict(m, what = p, newdata = nd, data = data, type = "response"))
  }
  if (!is.null(smo_fun_x))
    cases[[name]]$getSmo.fun.mu <<- num(getSmo(m, "mu")$fun(smo_fun_x))
}

## ---- single-predictor real data: abdom (n=610, inter=20 unclamped) ----
data(abdom)
ad <- list(x = abdom$x, y = abdom$y)

## basic ML: edf stays > lim, so this behaves like pb (no order-1 penalty)
m <- gamlss(y ~ pbz(x), family = NO, data = abdom, control = ctrl)
record_pbz("pbz_abdom", m, ad, list(family = "NO", formula = "y ~ pbz(x)"))
add_pred("pbz_abdom", m, abdom, data.frame(x = newx), smo_fun_x = newx)
## extrapolation outside the training range (must be linear, like pb)
exx <- c(8.0, 50.0)
cases[["pbz_abdom"]]$extrap.x <- num(exx)
cases[["pbz_abdom"]]$extrap.fun <- num(getSmo(m, "mu")$fun(exx))
cases[["pbz_abdom"]]$extrap.pred <- num(predict(m, what = "mu",
    newdata = data.frame(x = exx), data = abdom, type = "link"))

## GA (log link), ML
m <- gamlss(y ~ pbz(x), family = GA, data = abdom, control = ctrl)
record_pbz("pbz_abdom_ga", m, ad, list(family = "GA", formula = "y ~ pbz(x)"))

## pbz in mu AND sigma, ML
m <- gamlss(y ~ pbz(x), sigma.formula = ~pbz(x), family = NO, data = abdom,
            control = ctrl)
record_pbz("pbz_abdom_both", m, ad,
           list(family = "NO", formula = "y ~ pbz(x)",
                sigma_formula = "~pbz(x)"))
add_pred("pbz_abdom_both", m, abdom, data.frame(x = newx), smo_fun_x = newx)

## fixed lambda: lambda=100 forces a low edf (~2.38 <= lim) so the order-1
## penalty IS active -- exercises the double-penalty regpen under parity
m <- gamlss(y ~ pbz(x, lambda = 100), family = NO, data = abdom, control = ctrl)
record_pbz("pbz_abdom_fixlam", m, ad,
           list(family = "NO", formula = "y ~ pbz(x, lambda = 100)"))
add_pred("pbz_abdom_fixlam", m, abdom, data.frame(x = newx), smo_fun_x = newx)

## ---- the headline pbz behaviour: a term with no real signal shrinks to ----
## a constant.  Pure noise -> the order-2 penalty rails (lambda -> 1e7), which
## flips the fit to fixed-lambda mode and drops the order-1 penalty, settling
## at edf ~ 2 (R's warm-start quirk; see PBZ docstring).
set.seed(7); nz <- 150; xz <- runif(nz, 0, 1); yz <- rnorm(nz, 5, 1)
dz <- data.frame(x = xz, y = yz)
m <- gamlss(y ~ pbz(x), family = NO, data = dz, control = ctrl)
record_pbz("pbz_noise", m, list(x = xz, y = yz),
           list(family = "NO", formula = "y ~ pbz(x)"))

## a strongly linear relationship -> also collapses toward the line / constant
set.seed(13); nl <- 120; xl <- runif(nl, 0, 1); yl <- 3 + 2 * xl + rnorm(nl, 0, 0.3)
dl <- data.frame(x = xl, y = yl)
m <- gamlss(y ~ pbz(x), family = NO, data = dl, control = ctrl)
record_pbz("pbz_linear", m, list(x = xl, y = yl),
           list(family = "NO", formula = "y ~ pbz(x)"))

## ---- CG and mixed-free deterministic paths ----
## pbz in mu+sigma under CG, ML
m <- gamlss(y ~ pbz(x), sigma.formula = ~pbz(x), family = NO, data = abdom,
            method = CG(), control = ctrl)
record_pbz("pbz_abdom_cg", m, ad,
           list(family = "NO", formula = "y ~ pbz(x)",
                sigma_formula = "~pbz(x)", method = "CG"))
add_pred("pbz_abdom_cg", m, abdom, data.frame(x = newx), smo_fun_x = newx)

## fixed lambda under CG (deterministic -> should be bit-exact)
m <- gamlss(y ~ pbz(x, lambda = 100), family = NO, data = abdom,
            method = CG(), control = ctrl)
record_pbz("pbz_abdom_fixlam_cg", m, ad,
           list(family = "NO", formula = "y ~ pbz(x, lambda = 100)",
                method = "CG"))

## ---- simulated 2-predictor data: two smoothers + parametric mix ----
set.seed(99)
n2 <- 200
x1 <- runif(n2, 0, 1)
x2 <- runif(n2, 0, 1)
z  <- rnorm(n2)
ys <- 1 + 2 * sin(2 * pi * x1) + 2 * (x2 - 0.5)^2 - 0.5 * z + rnorm(n2, 0, 0.4)
sd <- list(x1 = x1, x2 = x2, z = z, y = ys)
sdf <- data.frame(x1 = x1, x2 = x2, z = z, y = ys)

## two pbz smoothers (backfitting over both)
m <- gamlss(y ~ pbz(x1) + pbz(x2), family = NO, data = sdf, control = ctrl)
record_pbz("pbz_sim_two", m, sd,
           list(family = "NO", formula = "y ~ pbz(x1) + pbz(x2)"))
add_pred("pbz_sim_two", m, sdf,
         data.frame(x1 = c(0.15, 0.5, 0.85), x2 = c(0.25, 0.6, 0.9)))

## parametric term + pbz smoother
m <- gamlss(y ~ z + pbz(x1), family = NO, data = sdf, control = ctrl)
record_pbz("pbz_sim_mix", m, sd,
           list(family = "NO", formula = "y ~ z + pbz(x1)"))
add_pred("pbz_sim_mix", m, sdf,
         data.frame(z = c(-1.2, 0.0, 1.1), x1 = c(0.15, 0.5, 0.85)))

## pbz + pb in one predictor (mixed smoother types backfit together)
m <- gamlss(y ~ pbz(x1) + pb(x2), family = NO, data = sdf, control = ctrl)
record_pbz("pbz_plus_pb", m, sd,
           list(family = "NO", formula = "y ~ pbz(x1) + pb(x2)"))
add_pred("pbz_plus_pb", m, sdf,
         data.frame(x1 = c(0.15, 0.5, 0.85), x2 = c(0.25, 0.6, 0.9)))

## ---- the order-1 penalty active at an INTERIOR (non-railed) fixed point ----
## raising lim to 10 makes the real-signal abdom smooth use the order-1 penalty
## too, settling at edf ~ 5.09 with finite lambda -- the two-lambda ML loop
## converging through the engine without railing to a constant
m <- gamlss(y ~ pbz(x, lim = 10), family = NO, data = abdom, control = ctrl)
record_pbz("pbz_abdom_lim10", m, ad,
           list(family = "NO", formula = "y ~ pbz(x, lim = 10)"))
add_pred("pbz_abdom_lim10", m, abdom, data.frame(x = newx), smo_fun_x = newx)

## do1order activation in SIGMA: homoscedastic data -> the sigma smooth has no
## signal and shrinks toward a constant
set.seed(202); nh <- 160; xh <- runif(nh, 0, 1); yh <- 2 + sin(3*xh) + rnorm(nh, 0, 0.4)
hd <- data.frame(x = xh, y = yh)
m <- gamlss(y ~ pbz(x), sigma.formula = ~pbz(x), family = NO, data = hd, control = ctrl)
record_pbz("pbz_sigma_do1", m, list(x = xh, y = yh),
           list(family = "NO", formula = "y ~ pbz(x)", sigma_formula = "~pbz(x)"))

## pbz in mu, sigma AND nu (BCCG, 3 parameters) -- nu collapses to a constant
m <- gamlss(y ~ pbz(x), sigma.formula = ~pbz(x), nu.formula = ~pbz(x),
            family = BCCG, data = abdom, control = ctrl)
record_pbz("pbz_abdom_bccg", m, ad,
           list(family = "BCCG", formula = "y ~ pbz(x)",
                sigma_formula = "~pbz(x)", nu_formula = "~pbz(x)"))

## prior weights including zeros (N = sum(w != 0))
set.seed(404); wv <- as.numeric(rep(c(1, 2, 0, 3, 1), length.out = nrow(abdom)))
m <- gamlss(y ~ pbz(x), weights = wv, family = NO, data = abdom, control = ctrl)
record_pbz("pbz_weighted", m, list(x = abdom$x, y = abdom$y, w = wv),
           list(family = "NO", formula = "y ~ pbz(x)", weights = "w"))

## few distinct x: inter clamps to n_distinct and the basis is rank-deficient;
## ML still fits via the SVD regpen
set.seed(808); xf <- as.numeric(sample(1:8, 200, TRUE)); yf <- 2 + sin(xf) + rnorm(200, 0, 0.3)
m <- gamlss(y ~ pbz(x), family = NO, data = data.frame(x = xf, y = yf), control = ctrl)
record_pbz("pbz_fewdist", m, list(x = xf, y = yf),
           list(family = "NO", formula = "y ~ pbz(x)"))

## deparse label: lambda = 1000000 must be labelled "lambda = 1e+06" (R deparse
## re-renders the numeric literal; scientific when strictly shorter than fixed)
m <- gamlss(y ~ pbz(x, lambda = 1000000), family = NO, data = abdom, control = ctrl)
record_pbz("pbz_abdom_fl1e6", m, ad,
           list(family = "NO", formula = "y ~ pbz(x, lambda = 1000000)"))

## mixed algorithm (RS then CG): the intercept <-> smooth-mean split is
## non-identifiable (as for pb), but fitted/deviance/df/edf/lambda and
## predictions are -- the test checks the identifiable quantities
m <- gamlss(y ~ pbz(x), sigma.formula = ~pbz(x), family = NO, data = abdom,
            method = mixed(2, 20), control = ctrl)
record_pbz("pbz_abdom_mixed", m, ad,
           list(family = "NO", formula = "y ~ pbz(x)",
                sigma_formula = "~pbz(x)", method = "mixed"))
add_pred("pbz_abdom_mixed", m, abdom, data.frame(x = newx))

write_json(cases, "tests/reference/pbz_fits.json", digits = NA,
           auto_unbox = TRUE, na = "string")
cat("WROTE tests/reference/pbz_fits.json with", length(cases), "cases:\n")
for (nm in names(cases))
  cat(sprintf("  %-20s iter=%d  Gdev=%.4f\n", nm, cases[[nm]]$iter,
              cases[[nm]]$G.deviance))
