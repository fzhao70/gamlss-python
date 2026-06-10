#!/usr/bin/env Rscript
## Generate reference results from the original R gamlss for verifying
## the Python port. Outputs JSON to tests/reference/.
suppressMessages({
  library(gamlss)
  library(jsonlite)
})

set.seed(2024)
dir.create("tests/reference/data", recursive = TRUE, showWarnings = FALSE)

num <- function(x) I(unname(as.numeric(x)))
chr <- function(x) I(unname(as.character(x)))

## ===================================================================
## Part 1: d/p/q/r reference values
## ===================================================================
dpqr <- list()

adddpqr <- function(fam, params, xs, qs = NULL,
                    ps = c(0.01, 0.25, 0.5, 0.9, 0.99), discrete = FALSE) {
  dfun <- get(paste0("d", fam)); pfun <- get(paste0("p", fam))
  qfun <- get(paste0("q", fam))
  out <- list()
  for (i in seq_along(params)) {
    pa <- params[[i]]
    d  <- do.call(dfun, c(list(x = xs), pa))
    dl <- do.call(dfun, c(list(x = xs), pa, list(log = TRUE)))
    p  <- do.call(pfun, c(list(q = xs), pa))
    pu <- do.call(pfun, c(list(q = xs), pa, list(lower.tail = FALSE)))
    q  <- do.call(qfun, c(list(p = ps), pa))
    out[[i]] <- list(params = pa, x = num(xs), d = num(d), dlog = num(dl),
                     p = num(p), pupper = num(pu), ps = num(ps), q = num(q))
  }
  dpqr[[fam]] <<- out
}

adddpqr("NO",  list(list(mu = 0, sigma = 1), list(mu = -1.5, sigma = 2.5),
                    list(mu = 3, sigma = 0.3)),
        xs = c(-3.2, -1, 0, 0.7, 2.9))
adddpqr("NO2", list(list(mu = 0, sigma = 1), list(mu = 1, sigma = 4)),
        xs = c(-3.2, -1, 0, 0.7, 2.9))
adddpqr("GA",  list(list(mu = 1, sigma = 1), list(mu = 5, sigma = 0.4),
                    list(mu = 0.5, sigma = 2)),
        xs = c(0.05, 0.5, 1, 2.4, 7.3))
adddpqr("IG",  list(list(mu = 1, sigma = 1), list(mu = 2.5, sigma = 0.6)),
        xs = c(0.05, 0.5, 1, 2.4, 7.3))
adddpqr("EXP", list(list(mu = 1), list(mu = 3.3)),
        xs = c(0.05, 0.5, 1, 2.4, 7.3))
adddpqr("LOGNO", list(list(mu = 0, sigma = 1), list(mu = 1.2, sigma = 0.4)),
        xs = c(0.05, 0.5, 1, 2.4, 7.3))
adddpqr("WEI", list(list(mu = 1, sigma = 1), list(mu = 2.5, sigma = 1.7)),
        xs = c(0.05, 0.5, 1, 2.4, 7.3))
adddpqr("WEI3", list(list(mu = 1, sigma = 1), list(mu = 2.5, sigma = 1.7)),
        xs = c(0.05, 0.5, 1, 2.4, 7.3))
adddpqr("GU",  list(list(mu = 0, sigma = 1), list(mu = 2, sigma = 0.7)),
        xs = c(-3.2, -1, 0, 0.7, 2.9))
adddpqr("RG",  list(list(mu = 0, sigma = 1), list(mu = 2, sigma = 0.7)),
        xs = c(-3.2, -1, 0, 0.7, 2.9))
adddpqr("LO",  list(list(mu = 0, sigma = 1), list(mu = -1, sigma = 2.2)),
        xs = c(-3.2, -1, 0, 0.7, 2.9))
adddpqr("TF",  list(list(mu = 0, sigma = 1, nu = 5),
                    list(mu = 1, sigma = 2, nu = 15)),
        xs = c(-3.2, -1, 0, 0.7, 2.9))
adddpqr("PE",  list(list(mu = 0, sigma = 1, nu = 2),
                    list(mu = 0.5, sigma = 1.5, nu = 1.2),
                    list(mu = 0, sigma = 1, nu = 3.5)),
        xs = c(-3.2, -1, 0, 0.7, 2.9))
adddpqr("BCCG", list(list(mu = 5, sigma = 0.1, nu = 1),
                     list(mu = 2, sigma = 0.3, nu = -0.5),
                     list(mu = 2, sigma = 0.3, nu = 0)),
        xs = c(0.5, 1.5, 3, 5.5, 9))
adddpqr("BCT", list(list(mu = 5, sigma = 0.1, nu = 1, tau = 5),
                    list(mu = 2, sigma = 0.3, nu = -0.5, tau = 10)),
        xs = c(0.5, 1.5, 3, 5.5, 9))
adddpqr("BCPE", list(list(mu = 5, sigma = 0.1, nu = 1, tau = 2),
                     list(mu = 2, sigma = 0.3, nu = -0.5, tau = 3.5)),
        xs = c(0.5, 1.5, 3, 5.5, 9))
adddpqr("JSU", list(list(mu = 0, sigma = 1, nu = 0, tau = 1),
                    list(mu = 1, sigma = 2, nu = -0.7, tau = 1.5)),
        xs = c(-3.2, -1, 0, 0.7, 2.9))
adddpqr("JSUo", list(list(mu = 0, sigma = 1, nu = 0, tau = 1),
                     list(mu = 1, sigma = 2, nu = -0.7, tau = 1.5)),
        xs = c(-3.2, -1, 0, 0.7, 2.9))
adddpqr("GG", list(list(mu = 1, sigma = 1, nu = 1),
                   list(mu = 2.5, sigma = 0.5, nu = -0.4)),
        xs = c(0.05, 0.5, 1, 2.4, 7.3))
adddpqr("SHASHo", list(list(mu = 0, sigma = 1, nu = 0, tau = 1),
                       list(mu = 1, sigma = 2, nu = 0.5, tau = 1.4)),
        xs = c(-3.2, -1, 0, 0.7, 2.9))
adddpqr("PO",  list(list(mu = 1), list(mu = 7.5)),
        xs = c(0, 1, 2, 5, 11), discrete = TRUE)
adddpqr("NBI", list(list(mu = 1, sigma = 1), list(mu = 7.5, sigma = 0.3)),
        xs = c(0, 1, 2, 5, 11), discrete = TRUE)
adddpqr("NBII", list(list(mu = 1, sigma = 1), list(mu = 7.5, sigma = 0.3)),
        xs = c(0, 1, 2, 5, 11), discrete = TRUE)
adddpqr("GEOM", list(list(mu = 1), list(mu = 4.2)),
        xs = c(0, 1, 2, 5, 11), discrete = TRUE)
adddpqr("PIG", list(list(mu = 1, sigma = 1), list(mu = 7.5, sigma = 0.3)),
        xs = c(0, 1, 2, 5, 11), discrete = TRUE)
adddpqr("ZIP", list(list(mu = 3, sigma = 0.2), list(mu = 7.5, sigma = 0.05)),
        xs = c(0, 1, 2, 5, 11), discrete = TRUE)
adddpqr("ZIP2", list(list(mu = 3, sigma = 0.2), list(mu = 7.5, sigma = 0.05)),
        xs = c(0, 1, 2, 5, 11), discrete = TRUE)
adddpqr("ZINBI", list(list(mu = 3, sigma = 0.5, nu = 0.2)),
        xs = c(0, 1, 2, 5, 11), discrete = TRUE)
adddpqr("ZANBI", list(list(mu = 3, sigma = 0.5, nu = 0.2)),
        xs = c(0, 1, 2, 5, 11), discrete = TRUE)
adddpqr("BI",  list(list(bd = 10, mu = 0.5), list(bd = 10, mu = 0.23)),
        xs = c(0, 1, 3, 7, 10), discrete = TRUE)
adddpqr("BB",  list(list(bd = 10, mu = 0.5, sigma = 1),
                    list(bd = 10, mu = 0.23, sigma = 0.5)),
        xs = c(0, 1, 3, 7, 10), discrete = TRUE)
adddpqr("ZABI", list(list(bd = 10, mu = 0.5, sigma = 0.2)),
        xs = c(0, 1, 3, 7, 10), discrete = TRUE)
adddpqr("ZIBI", list(list(bd = 10, mu = 0.5, sigma = 0.2)),
        xs = c(0, 1, 3, 7, 10), discrete = TRUE)
adddpqr("BE",  list(list(mu = 0.5, sigma = 0.5), list(mu = 0.2, sigma = 0.3)),
        xs = c(0.05, 0.2, 0.5, 0.7, 0.95))
adddpqr("BEo", list(list(mu = 2, sigma = 3), list(mu = 0.5, sigma = 0.8)),
        xs = c(0.05, 0.2, 0.5, 0.7, 0.95))

write_json(dpqr, "tests/reference/dpqr.json", digits = I(17),
           auto_unbox = TRUE)
cat("dpqr.json written\n")

## ===================================================================
## Part 2: simulated datasets (round to 6 dp for exact CSV round trip)
## ===================================================================
simdata <- function(name, df) {
  df <- as.data.frame(lapply(df, function(c)
    if (is.numeric(c)) round(c, 6) else c))
  write.csv(df, file.path("tests/reference/data", paste0(name, ".csv")),
            row.names = FALSE)
  df
}

n <- 200
x1 <- runif(n); x2 <- rnorm(n)
sim_nbi <- simdata("sim_nbi", data.frame(
  y = rNBI(n, mu = exp(1 + 0.7 * x1), sigma = 0.6), x1 = x1, x2 = x2))
sim_nbii <- simdata("sim_nbii", data.frame(
  y = rNBII(n, mu = exp(1 + 0.7 * x1), sigma = 0.6), x1 = x1, x2 = x2))
sim_geom <- simdata("sim_geom", data.frame(
  y = rGEOM(n, mu = exp(0.5 + x1)), x1 = x1))
sim_pig <- simdata("sim_pig", data.frame(
  y = rPIG(n, mu = exp(1 + 0.7 * x1), sigma = 0.5), x1 = x1))
sim_zip <- simdata("sim_zip", data.frame(
  y = rZIP(n, mu = exp(1.2 + 0.6 * x1), sigma = 0.15), x1 = x1))
sim_zip2 <- simdata("sim_zip2", data.frame(
  y = rZIP2(n, mu = exp(1.2 + 0.6 * x1), sigma = 0.15), x1 = x1))
sim_zinbi <- simdata("sim_zinbi", data.frame(
  y = rZINBI(n, mu = exp(1.4 + 0.5 * x1), sigma = 0.4, nu = 0.15), x1 = x1))
sim_zanbi <- simdata("sim_zanbi", data.frame(
  y = rZANBI(n, mu = exp(1.4 + 0.5 * x1), sigma = 0.4, nu = 0.15), x1 = x1))
bd <- rep(c(8L, 10L, 12L, 20L), length.out = n)
pr <- plogis(-0.4 + 1.1 * x1)
ybi <- rBI(n, bd = bd, mu = pr)
sim_bi <- simdata("sim_bi", data.frame(y = ybi, fail = bd - ybi, bd = bd,
                                       x1 = x1))
ybb <- rBB(n, bd = bd, mu = pr, sigma = 0.4)
sim_bb <- simdata("sim_bb", data.frame(y = ybb, fail = bd - ybb, bd = bd,
                                       x1 = x1))
yzabi <- rZABI(n, bd = bd, mu = pr, sigma = 0.2)
sim_zabi <- simdata("sim_zabi", data.frame(y = yzabi, fail = bd - yzabi,
                                           bd = bd, x1 = x1))
yzibi <- rZIBI(n, bd = bd, mu = pr, sigma = 0.2)
sim_zibi <- simdata("sim_zibi", data.frame(y = yzibi, fail = bd - yzibi,
                                           bd = bd, x1 = x1))
sim_be <- simdata("sim_be", data.frame(
  y = rBE(n, mu = plogis(0.3 + 0.8 * x1), sigma = 0.35), x1 = x1))
sim_beo <- simdata("sim_beo", data.frame(
  y = rBEo(n, mu = exp(0.5 + 0.4 * x1), sigma = exp(0.8)), x1 = x1))
sim_jsu <- simdata("sim_jsu", data.frame(
  y = rJSU(500, mu = 2 + x2[rep(1:n, length.out = 500)], sigma = 1,
           nu = -0.5, tau = 1.5),
  x = x2[rep(1:n, length.out = 500)]))
sim_gg <- simdata("sim_gg", data.frame(
  y = rGG(n, mu = exp(1 + 0.5 * x1), sigma = 0.4, nu = 1.3), x1 = x1))
sim_shasho <- simdata("sim_shasho", data.frame(
  y = rSHASHo(500, mu = 1, sigma = 1.2, nu = 0.3, tau = 1.1),
  x = rep(0, 500)))
wint <- rep(c(1L, 2L, 3L), length.out = 610)
write.csv(data.frame(w = wint), "tests/reference/data/weights_int.csv",
          row.names = FALSE)

## ===================================================================
## Part 3: model fits
## ===================================================================
data(abdom); data(aids); data(usair); data(species); data(fabric)
data(rent)
aids$qrt <- factor(aids$qrt)

fits <- list()

safe_se_vcov <- function(m) {
  v <- try(suppressWarnings(vcov(m, type = "se")), silent = TRUE)
  if (inherits(v, "try-error")) NULL else num(v)
}
se_qr <- function(m, what) {
  Qr <- m[[paste0(what, ".qr")]]
  p1 <- seq_len(m[[paste0(what, ".df")]])
  covm <- chol2inv(Qr$qr[p1, p1, drop = FALSE])
  num(sqrt(diag(covm)))
}

record <- function(name, m, datname, residuals = TRUE, se = TRUE,
                   extra = list()) {
  out <- list(
    family = m$family[1],
    data = datname,
    G.deviance = m$G.deviance,
    P.deviance = m$P.deviance,
    aic = m$aic, sbc = m$sbc,
    df.fit = m$df.fit, df.residual = m$df.residual,
    noObs = m$noObs, N = m$N, iter = m$iter,
    converged = m$converged,
    parameters = chr(m$parameters)
  )
  for (p in m$parameters) {
    cf <- coef(m, p)
    if (!is.null(cf)) {
      out[[paste0("coef.", p)]] <- num(cf)
      out[[paste0("coefnames.", p)]] <- chr(names(cf))
    }
    fv <- fitted(m, p)
    out[[paste0("fitted10.", p)]] <- num(head(fv, 10))
    out[[paste0("fittedsum.", p)]] <- sum(fv)
    if (se) out[[paste0("se_qr.", p)]] <- se_qr(m, p)
  }
  if (residuals) {
    out$resid10 <- num(head(resid(m), 10))
    out$residsum <- sum(resid(m))
  }
  if (se) out$se_vcov <- safe_se_vcov(m)
  fits[[name]] <<- c(out, extra)
}

ctrl <- gamlss.control(trace = FALSE)
ctrl200 <- gamlss.control(trace = FALSE, n.cyc = 200)

## --- continuous on abdom -------------------------------------------
m <- gamlss(y ~ x, family = NO, data = abdom, control = ctrl)
nd <- data.frame(x = c(15.5, 25, 38.2))
record("no_abdom", m, "abdom", extra = list(
  pred.mu.link = num(predict(m, what = "mu", newdata = nd, data = abdom)),
  pred.mu.response = num(predict(m, what = "mu", newdata = nd, data = abdom,
                                 type = "response")),
  pred.sigma.response = num(predict(m, what = "sigma", newdata = nd,
                                    data = abdom, type = "response")),
  se.link.mu = num(lpred(m, what = "mu", se.fit = TRUE)$se.fit[1:10]),
  se.response.mu = num(lpred(m, what = "mu", type = "response",
                             se.fit = TRUE)$se.fit[1:10])))

m <- gamlss(y ~ x, sigma.formula = ~x, family = NO, data = abdom,
            control = ctrl)
record("no_abdom_sx", m, "abdom")

m <- gamlss(y ~ poly(x, 3), sigma.formula = ~x, family = NO, data = abdom,
            control = ctrl)
record("no_abdom_poly", m, "abdom", extra = list(
  pred.mu.link = num(predict(m, what = "mu", newdata = nd, data = abdom)),
  terms.mu.const = attr(predict(m, what = "mu", type = "terms"),
                        "constant"),
  terms.mu.col1 = num(predict(m, what = "mu", type = "terms")[1:10, 1])))

m <- gamlss(y ~ x, sigma.formula = ~x, family = GA, data = abdom,
            control = ctrl)
record("ga_abdom", m, "abdom")

m <- gamlss(y ~ x, family = LOGNO, data = abdom, control = ctrl)
record("logno_abdom", m, "abdom")

m <- gamlss(y ~ x, family = IG, data = abdom, control = ctrl)
record("ig_abdom", m, "abdom")

m <- gamlss(y ~ x, family = GU, data = abdom, control = ctrl)
record("gu_abdom", m, "abdom")

m <- gamlss(y ~ x, family = RG, data = abdom, control = ctrl)
record("rg_abdom", m, "abdom")

m <- gamlss(y ~ x, family = LO, data = abdom, control = ctrl)
record("lo_abdom", m, "abdom")

m <- gamlss(y ~ x, family = WEI, data = abdom, control = ctrl)
record("wei_abdom", m, "abdom")

m <- gamlss(y ~ x, family = WEI3, data = abdom, control = ctrl)
record("wei3_abdom", m, "abdom")

m <- gamlss(y ~ x, sigma.formula = ~x, family = TF, data = abdom,
            control = ctrl)
record("tf_abdom", m, "abdom")

m <- gamlss(y ~ x, family = PE, data = abdom, control = ctrl200)
record("pe_abdom", m, "abdom")

m <- gamlss(y ~ x, sigma.formula = ~x, family = BCCG, data = abdom,
            control = ctrl200)
record("bccg_abdom", m, "abdom")

m <- gamlss(y ~ x, sigma.formula = ~x, family = BCT, data = abdom,
            control = ctrl200)
record("bct_abdom", m, "abdom")

m <- gamlss(y ~ x, sigma.formula = ~x, family = BCPE, data = abdom,
            control = ctrl200)
record("bcpe_abdom", m, "abdom")

m <- gamlss(y ~ x, sigma.formula = ~x, family = BCTo, data = abdom,
            control = ctrl200)
record("bcto_abdom", m, "abdom")

## CG and mixed methods
m <- gamlss(y ~ x, sigma.formula = ~x, family = NO, data = abdom,
            method = CG(), control = ctrl)
record("no_abdom_cg", m, "abdom")

m <- gamlss(y ~ x, sigma.formula = ~x, family = BCT, data = abdom,
            method = mixed(2, 50), control = ctrl200)
record("bct_abdom_mixed", m, "abdom")

## weights (integer frequencies)
m <- gamlss(y ~ x, family = NO, data = abdom, weights = wint,
            control = ctrl)
record("no_abdom_w", m, "abdom", extra = list(weights = "weights_int"))

## --- usair gamma ----------------------------------------------------
m <- gamlss(y ~ x1 + x2 + x3 + x4 + x5 + x6, family = GA, data = usair,
            control = ctrl)
record("ga_usair", m, "usair")

## --- rent (factors) -------------------------------------------------
m <- gamlss(R ~ Fl + A + H + loc, sigma.formula = ~Fl, family = GA,
            data = rent, control = ctrl)
record("ga_rent", m, "rent")

## --- discrete -------------------------------------------------------
m <- gamlss(y ~ x + qrt, family = PO, data = aids, control = ctrl)
ndpo <- data.frame(x = c(10, 46), qrt = factor(c(2, 4), levels = 1:4))
record("po_aids", m, "aids", residuals = FALSE, extra = list(
  pred.mu.response = num(predict(m, what = "mu", newdata = ndpo,
                                 data = aids, type = "response"))))

m <- gamlss(y ~ x + qrt, family = NBI, data = aids, control = ctrl)
record("nbi_aids", m, "aids", residuals = FALSE)

m <- gamlss(y ~ x, family = PO, data = fabric, control = ctrl)
record("po_fabric", m, "fabric", residuals = FALSE)

m <- gamlss(y ~ x1 + x2, sigma.formula = ~x1, family = NBI, data = sim_nbi,
            control = ctrl)
record("nbi_sim", m, "sim_nbi", residuals = FALSE)

m <- gamlss(y ~ x1, family = NBII, data = sim_nbii, control = ctrl)
record("nbii_sim", m, "sim_nbii", residuals = FALSE)

m <- gamlss(y ~ x1, family = GEOM, data = sim_geom, control = ctrl)
record("geom_sim", m, "sim_geom", residuals = FALSE)

m <- gamlss(y ~ x1, family = PIG, data = sim_pig, control = ctrl)
record("pig_sim", m, "sim_pig", residuals = FALSE)

m <- gamlss(y ~ x1, family = ZIP, data = sim_zip, control = ctrl)
record("zip_sim", m, "sim_zip", residuals = FALSE)

m <- gamlss(y ~ x1, family = ZIP2, data = sim_zip2, control = ctrl)
record("zip2_sim", m, "sim_zip2", residuals = FALSE)

m <- gamlss(y ~ x1, family = ZINBI, data = sim_zinbi, control = ctrl200)
record("zinbi_sim", m, "sim_zinbi", residuals = FALSE)

m <- gamlss(y ~ x1, family = ZANBI, data = sim_zanbi, control = ctrl200)
record("zanbi_sim", m, "sim_zanbi", residuals = FALSE)

## --- binomial type ---------------------------------------------------
m <- gamlss(cbind(y, fail) ~ x1, family = BI, data = sim_bi,
            control = ctrl)
record("bi_sim", m, "sim_bi", residuals = FALSE)

m <- gamlss(cbind(y, fail) ~ x1, family = BB, data = sim_bb,
            control = ctrl)
record("bb_sim", m, "sim_bb", residuals = FALSE)

m <- gamlss(cbind(y, fail) ~ x1, family = ZABI, data = sim_zabi,
            control = ctrl200)
record("zabi_sim", m, "sim_zabi", residuals = FALSE)

m <- gamlss(cbind(y, fail) ~ x1, family = ZIBI, data = sim_zibi,
            control = ctrl200)
record("zibi_sim", m, "sim_zibi", residuals = FALSE)

## --- (0,1) -----------------------------------------------------------
m <- gamlss(y ~ x1, family = BE, data = sim_be, control = ctrl)
record("be_sim", m, "sim_be")

m <- gamlss(y ~ x1, family = BEo, data = sim_beo, control = ctrl)
record("beo_sim", m, "sim_beo")

## --- extra continuous -------------------------------------------------
m <- gamlss(y ~ x, family = JSU, data = sim_jsu, control = ctrl200)
record("jsu_sim", m, "sim_jsu")

m <- gamlss(y ~ x1, family = GG, data = sim_gg, control = ctrl200)
record("gg_sim", m, "sim_gg")

m <- gamlss(y ~ 1, family = SHASHo, data = sim_shasho, control = ctrl200)
record("shasho_sim", m, "sim_shasho")

m <- gamlss(R ~ Fl + A, family = EXP, data = rent, control = ctrl)
record("exp_rent", m, "rent")

write_json(fits, "tests/reference/fits.json", digits = I(17),
           auto_unbox = TRUE)
cat("fits.json written:", length(fits), "cases\n")
