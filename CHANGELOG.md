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

### Not yet supported (planned)

- Prediction, `getSmo()` and term plots for `pb()` terms — *Step 3*.
- `pb()` with the CG / mixed algorithms (currently raises
  `NotImplementedError`) — *Step 4*.
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
