#!/usr/bin/env Rscript
## Reference results for the penalised B-spline smoothers pb() (Step 2).
## Each case embeds its own data so the Python test is hermetic and uses
## byte-identical inputs.  Output: tests/reference/pb_fits.json
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

record_pb <- function(name, m, data, spec) {
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

## prediction + getSmo references attached to an already-recorded case.
## nd: a newdata data.frame; smo_fun_x: x grid for getSmo(m,"mu")$fun
## (only meaningful when mu has a single smoother).
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

m <- gamlss(y ~ pb(x), family = NO, data = abdom, control = ctrl)
record_pb("pb_abdom", m, ad, list(family = "NO", formula = "y ~ pb(x)"))
add_pred("pb_abdom", m, abdom, data.frame(x = newx), smo_fun_x = newx)
## extrapolation: x below 12.3 and above 42.4 (outside the training range)
exx <- c(8.0, 50.0)
cases[["pb_abdom"]]$extrap.x <- num(exx)
cases[["pb_abdom"]]$extrap.fun <- num(getSmo(m, "mu")$fun(exx))
cases[["pb_abdom"]]$extrap.pred <- num(predict(m, what = "mu",
    newdata = data.frame(x = exx), data = abdom, type = "link"))

m <- gamlss(y ~ pb(x), family = GA, data = abdom, control = ctrl)
record_pb("pb_abdom_ga", m, ad, list(family = "GA", formula = "y ~ pb(x)"))

m <- gamlss(y ~ pb(x), sigma.formula = ~pb(x), family = NO, data = abdom,
            control = ctrl)
record_pb("pb_abdom_both", m, ad,
          list(family = "NO", formula = "y ~ pb(x)",
               sigma_formula = "~pb(x)"))
add_pred("pb_abdom_both", m, abdom, data.frame(x = newx), smo_fun_x = newx)

m <- gamlss(y ~ pb(x, lambda = 100), family = NO, data = abdom, control = ctrl)
record_pb("pb_abdom_fixlam", m, ad,
          list(family = "NO", formula = "y ~ pb(x, lambda=100)"))

## Step 5 selection methods (optimiser-dependent: ~1e-4 parity vs R)
m <- gamlss(y ~ pb(x, df = 5), family = NO, data = abdom, control = ctrl)
record_pb("pb_abdom_df", m, ad,
          list(family = "NO", formula = "y ~ pb(x, df = 5)"))
add_pred("pb_abdom_df", m, abdom, data.frame(x = newx), smo_fun_x = newx)

m <- gamlss(y ~ pb(x, max.df = 5), family = NO, data = abdom, control = ctrl)
record_pb("pb_abdom_maxdf", m, ad,
          list(family = "NO", formula = "y ~ pb(x, max.df = 5)"))

m <- gamlss(y ~ pb(x, method = "GAIC", k = 2), family = NO, data = abdom,
            control = ctrl)
record_pb("pb_abdom_gaic", m, ad,
          list(family = "NO", formula = "y ~ pb(x, method = \"GAIC\", k = 2)"))

m <- gamlss(y ~ pb(x, method = "GCV"), family = NO, data = abdom,
            control = ctrl)
record_pb("pb_abdom_gcv", m, ad,
          list(family = "NO", formula = "y ~ pb(x, method = \"GCV\")"))

## df selection in SIGMA (alongside ML in mu)
m <- gamlss(y ~ pb(x), sigma.formula = ~pb(x, df = 4), family = NO,
            data = abdom, control = ctrl)
record_pb("pb_abdom_df_sigma", m, ad,
          list(family = "NO", formula = "y ~ pb(x)",
               sigma_formula = "~pb(x, df = 4)"))

## a different target df (lower -> more smoothing)
m <- gamlss(y ~ pb(x, df = 3), family = NO, data = abdom, control = ctrl)
record_pb("pb_abdom_df3", m, ad,
          list(family = "NO", formula = "y ~ pb(x, df = 3)"))

## df selection under the CG algorithm (Step 4 x Step 5)
m <- gamlss(y ~ pb(x, df = 5), family = NO, data = abdom, method = CG(),
            control = ctrl)
record_pb("pb_abdom_df_cg", m, ad,
          list(family = "NO", formula = "y ~ pb(x, df = 5)", method = "CG"))

## NOTE: GAIC with other k (e.g. k=4) is intentionally NOT a parity case --
## R's nlminb and scipy can descend to different optima of the flat/multimodal
## GAIC objective (for k=4 R sticks at lambda=1e7/edf=2 while the port finds a
## lower-GAIC interior optimum).  Both are valid GAIC smooths; see CHANGELOG.

## max.df larger than the ML edf -> cap inactive, falls back to ML (bit-exact)
m <- gamlss(y ~ pb(x, max.df = 20), family = NO, data = abdom, control = ctrl)
record_pb("pb_abdom_maxdf_off", m, ad,
          list(family = "NO", formula = "y ~ pb(x, max.df = 20)"))

## ---- simulated 2-predictor data: two smoothers + mixed term ----
set.seed(99)
n2 <- 200
x1 <- runif(n2, 0, 1)
x2 <- runif(n2, 0, 1)
z  <- rnorm(n2)
ys <- 1 + 2 * sin(2 * pi * x1) + 2 * (x2 - 0.5)^2 - 0.5 * z + rnorm(n2, 0, 0.4)
sd <- list(x1 = x1, x2 = x2, z = z, y = ys)
sdf <- data.frame(x1 = x1, x2 = x2, z = z, y = ys)

m <- gamlss(y ~ pb(x1) + pb(x2), family = NO, data = sdf, control = ctrl)
record_pb("pb_sim_two", m, sd,
          list(family = "NO", formula = "y ~ pb(x1) + pb(x2)"))
add_pred("pb_sim_two", m, sdf,
         data.frame(x1 = c(0.15, 0.5, 0.85), x2 = c(0.25, 0.6, 0.9)))

## two smoothers, each with its own target df
m <- gamlss(y ~ pb(x1, df = 4) + pb(x2, df = 6), family = NO, data = sdf,
            control = ctrl)
record_pb("pb_sim_two_df", m, sd,
          list(family = "NO",
               formula = "y ~ pb(x1, df = 4) + pb(x2, df = 6)"))

m <- gamlss(y ~ z + pb(x1), family = NO, data = sdf, control = ctrl)
record_pb("pb_sim_mix", m, sd,
          list(family = "NO", formula = "y ~ z + pb(x1)"))
add_pred("pb_sim_mix", m, sdf,
         data.frame(z = c(-1.2, 0.0, 1.1), x1 = c(0.15, 0.5, 0.85)))

## ---- CG and mixed algorithms with smoothers in mu and sigma ----
m <- gamlss(y ~ pb(x), sigma.formula = ~pb(x), family = NO, data = abdom,
            method = CG(), control = ctrl)
record_pb("pb_abdom_cg", m, ad,
          list(family = "NO", formula = "y ~ pb(x)",
               sigma_formula = "~pb(x)", method = "CG"))
add_pred("pb_abdom_cg", m, abdom, data.frame(x = newx), smo_fun_x = newx)

m <- gamlss(y ~ pb(x), sigma.formula = ~pb(x), family = NO, data = abdom,
            method = mixed(2, 20), control = ctrl)
record_pb("pb_abdom_mixed", m, ad,
          list(family = "NO", formula = "y ~ pb(x)",
               sigma_formula = "~pb(x)", method = "mixed"))
# mixed split is non-identifiable, but the total prediction is not:
add_pred("pb_abdom_mixed", m, abdom, data.frame(x = newx))

## CG with a single smoother (mu) and a parametric sigma -- asymmetric coupling
m <- gamlss(y ~ pb(x), sigma.formula = ~x, family = NO, data = abdom,
            method = CG(), control = ctrl)
record_pb("pb_abdom_mu_cg", m, ad,
          list(family = "NO", formula = "y ~ pb(x)",
               sigma_formula = "~x", method = "CG"))
add_pred("pb_abdom_mu_cg", m, abdom, data.frame(x = newx), smo_fun_x = newx)

## CG with a non-identity (log) link
m <- gamlss(y ~ pb(x), family = GA, data = abdom, method = CG(), control = ctrl)
record_pb("pb_abdom_ga_cg", m, ad,
          list(family = "GA", formula = "y ~ pb(x)", method = "CG"))
add_pred("pb_abdom_ga_cg", m, abdom, data.frame(x = newx), smo_fun_x = newx)

## CG with two smoothers in one parameter (backfitting maxit=1 over both)
m <- gamlss(y ~ pb(x1) + pb(x2), family = NO, data = sdf, method = CG(),
            control = ctrl)
record_pb("pb_sim_two_cg", m, sd,
          list(family = "NO", formula = "y ~ pb(x1) + pb(x2)", method = "CG"))
add_pred("pb_sim_two_cg", m, sdf,
         data.frame(x1 = c(0.15, 0.5, 0.85), x2 = c(0.25, 0.6, 0.9)))

## ---- Step 4 x Step 5: selection methods under CG / mixed ----
## fixed lambda under CG (deterministic -> should be bit-exact)
m <- gamlss(y ~ pb(x, lambda = 100), family = NO, data = abdom,
            method = CG(), control = ctrl)
record_pb("pb_abdom_fixlam_cg", m, ad,
          list(family = "NO", formula = "y ~ pb(x, lambda = 100)",
               method = "CG"))

## NOTE: optimiser-based selection under CG is intentionally NOT a parity
## case.  CG's single-sweep (maxit=1) backfitting amplifies the optimiser
## sensitivity, so GCV under CG converges to within ~1e-4 of R but with a
## different iteration count, and max.df under CG fails to converge in R
## (200 cycles, converged=FALSE).  Use RS for GCV / GAIC / max.df; the
## deterministic selectors (fixed lambda, df) are fine under CG (below).

## df selection under mixed (RS then CG)
m <- gamlss(y ~ pb(x, df = 5), family = NO, data = abdom,
            method = mixed(2, 20), control = ctrl)
record_pb("pb_abdom_df_mixed", m, ad,
          list(family = "NO", formula = "y ~ pb(x, df = 5)",
               method = "mixed"))

## ---- extreme / edge cases ----
## small n (< 99) -> R clamps the knot intervals to inter = 10
set.seed(321)
ns <- 40
xs <- sort(runif(ns, 0, 5))
ysm <- 2 + sin(xs) + rnorm(ns, 0, 0.3)
m <- gamlss(y ~ pb(x), family = NO, data = data.frame(x = xs, y = ysm),
            control = ctrl)
record_pb("pb_small_n", m, list(x = xs, y = ysm),
          list(family = "NO", formula = "y ~ pb(x)"))

## prior weights, including some zero weights (exercises N = sum(w != 0))
wts <- as.numeric(rep(c(1, 2, 0, 3, 1), length.out = nrow(abdom)))
m <- gamlss(y ~ pb(x), weights = wts, family = NO, data = abdom, control = ctrl)
record_pb("pb_weighted", m, list(x = abdom$x, y = abdom$y, w = wts),
          list(family = "NO", formula = "y ~ pb(x)", weights = "w"))

## df target larger than the basis allows -> R caps df to 3 (with a warning)
m <- suppressWarnings(gamlss(y ~ pb(x, df = 50), family = NO, data = abdom,
                             control = ctrl))
record_pb("pb_df_toobig", m, ad,
          list(family = "NO", formula = "y ~ pb(x, df = 50)"))

## few distinct x: inter clamps to n_distinct and the basis is rank-deficient;
## ML still fits (SVD regpen handles the deficiency) -- parity check
set.seed(808)
nf <- 200
xf <- as.numeric(sample(1:8, nf, replace = TRUE))   # 8 distinct values
yf <- 2 + sin(xf) + rnorm(nf, 0, 0.3)
m <- gamlss(y ~ pb(x), family = NO, data = data.frame(x = xf, y = yf),
            control = ctrl)
record_pb("pb_few_distinct", m, list(x = xf, y = yf),
          list(family = "NO", formula = "y ~ pb(x)"))

## R deparse re-renders numeric literals in the term label: lambda = 1000000
## must be labelled "pb(x, lambda = 1e+06)" (scientific when strictly shorter)
m <- gamlss(y ~ pb(x, lambda = 1000000), family = NO, data = abdom, control = ctrl)
record_pb("pb_abdom_fl1e6", m, ad,
          list(family = "NO", formula = "y ~ pb(x, lambda = 1000000)"))

## order-1 difference penalty (random walk) instead of the default order 2
m <- gamlss(y ~ pb(x, order = 1), family = NO, data = abdom, control = ctrl)
record_pb("pb_abdom_order1", m, ad,
          list(family = "NO", formula = "y ~ pb(x, order = 1)"))

write_json(cases, "tests/reference/pb_fits.json", digits = NA,
           auto_unbox = TRUE, na = "string")
cat("WROTE tests/reference/pb_fits.json with", length(cases), "cases:\n")
for (nm in names(cases))
  cat(sprintf("  %-18s iter=%d  Gdev=%.4f\n", nm, cases[[nm]]$iter,
              cases[[nm]]$G.deviance))
