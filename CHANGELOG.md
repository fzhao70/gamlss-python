# Changelog

All notable changes to **gamlss-python** are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/); the overriding
goal of every change is to reproduce R `gamlss` numerically (see the
README for the verification methodology).

## [Unreleased]

### Added — penalised B-spline smoothers `pb()` (in progress)

Model formulas may now contain `pb()` terms, e.g.

```python
import gamlss as gl
ab = gl.load_data("abdom")
m = gl.gamlss("y ~ pb(x)", sigma_formula="~pb(x)", family=gl.NO(), data=ab)
```

This is being ported in verified steps, each gated on numerical parity
with the original R package:

- **Step 1 — smoother core** (`gamlss/smooth.py`). Paul Eilers' B-spline
  basis (`bbase`), the difference penalty, and the `PB` smoother class
  with its SVD penalised-least-squares solver (`regpen`) and the ML
  (maximum-likelihood) smoothing-parameter loop. Verified against R's
  standalone `gamlss.pb` to ~1e-13 (λ, edf, coefficients, fitted values);
  the basis matches to ~1e-12 and the penalty exactly.

- **Step 2 — fitting via the RS algorithm.** Backfitting
  (`engine.additive_fit`, a port of R's `additive.fit`) is wired into the
  inner GLIM iteration (`glim.fit`) and the RS outer loop, and the formula
  layer (`ParamFormula.design`) now splits `pb()` terms out — each
  contributes its *linear* column to the parametric design (labelled as in
  R, e.g. `pb(x)`) plus a smoother fitted by backfitting. Supported:
  `pb()` in any or all distribution parameters, ML (default) and
  fixed-`lambda` smoothing. Verified against R on six models — `pb` in
  `mu` and/or `sigma`, identity and log links, single and multiple
  smoothers, and parametric + smoother mixes — with coefficients,
  per-smoother λ/edf, degrees of freedom, deviance/AIC/SBC, fitted values
  and the **exact RS iteration count** all matching to rtol 1e-6
  (`tests/test_pb.py`, reference `r-scripts/gen_pb_reference.R`).

- **Step 3 — prediction and `getSmo()`.** `getSmo(m, "mu")` returns the
  fitted smoother (carrying its natural-spline `fun`, plus `coef`, `lambda`,
  `edf`, `fv`, `knots`); `predict(newdata=...)` and `predictAll()` now add
  each smoother's `fun(xeval)` — a natural cubic spline through the fitted
  values, as in R's `predict.gamlss` — to the linear predictor. Verified
  against R for link/response predictions and `getSmo$fun(xeval)` to
  rtol 1e-6 (`tests/test_pb.py`).

- **Step 4 — CG and mixed algorithms.** `pb()` now also fits under
  `method=CG()` and `method=mixed()` (backfitting with one sweep per inner
  CG step, plus the smooth in CG's step-halving), matching R's `CG`/`mixed`
  fitting paths. Verified against R across CG with one or both parameters
  smoothed, a parametric + smoothed-parameter mix, the GA (log) link, and
  two smoothers in one parameter: `CG()` reproduces coefficients, λ/edf, df,
  deviance and predictions to rtol 1e-6.
  Note: the linear column of `pb(x)` is concurve with the smooth, so for an
  early-stopped `mixed` fit the *split* of the x-effect into a parametric
  coefficient vs the smooth is not identifiable and can differ from R, even
  though the fitted model (fitted values, deviance, df, edf, λ) is identical
  to ~1e-14.

### Not yet supported (planned)

- Term plots / `lpred(type="terms")` for `pb()` terms — currently raises
  `NotImplementedError` (*Step 3 follow-up*).
- `pb()` smoothing-parameter selection by GAIC / GCV / fixed `df`
  (only ML and fixed `lambda` are available so far) — *Step 5*.
- `pbz()` (shrink-to-zero P-splines) and other smoothers
  (`cs`, `ps`, `ri`, `random`, ...) — *Step 6+*.

## [0.1.0]

Initial port of the parametric GAMLSS core: R `gamlss` (5.5-0) and
`gamlss.dist` (6.1-1). The RS / CG / mixed fitting algorithms, all four
distribution parameters, link functions, ~40 families with their
`d`/`p`/`q`/`r` functions, R-compatible formulas (factors, `poly()`,
`cbind()`, offsets), weights, predictions, standard errors (qr and
numerical-Hessian vcov) and quantile residuals — verified against the
original R package across 43 reference models plus dense d/p/q/r grids
for every family.
