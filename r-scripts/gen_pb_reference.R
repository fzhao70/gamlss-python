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

## ---- single-predictor real data: abdom (n=610, inter=20 unclamped) ----
data(abdom)
ad <- list(x = abdom$x, y = abdom$y)

m <- gamlss(y ~ pb(x), family = NO, data = abdom, control = ctrl)
record_pb("pb_abdom", m, ad, list(family = "NO", formula = "y ~ pb(x)"))

m <- gamlss(y ~ pb(x), family = GA, data = abdom, control = ctrl)
record_pb("pb_abdom_ga", m, ad, list(family = "GA", formula = "y ~ pb(x)"))

m <- gamlss(y ~ pb(x), sigma.formula = ~pb(x), family = NO, data = abdom,
            control = ctrl)
record_pb("pb_abdom_both", m, ad,
          list(family = "NO", formula = "y ~ pb(x)",
               sigma_formula = "~pb(x)"))

m <- gamlss(y ~ pb(x, lambda = 100), family = NO, data = abdom, control = ctrl)
record_pb("pb_abdom_fixlam", m, ad,
          list(family = "NO", formula = "y ~ pb(x, lambda=100)"))

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

m <- gamlss(y ~ z + pb(x1), family = NO, data = sdf, control = ctrl)
record_pb("pb_sim_mix", m, sd,
          list(family = "NO", formula = "y ~ z + pb(x1)"))

write_json(cases, "tests/reference/pb_fits.json", digits = NA,
           auto_unbox = TRUE, na = "string")
cat("WROTE tests/reference/pb_fits.json with", length(cases), "cases:\n")
for (nm in names(cases))
  cat(sprintf("  %-18s iter=%d  Gdev=%.4f\n", nm, cases[[nm]]$iter,
              cases[[nm]]$G.deviance))
