# Changelog

All notable changes to **gamlss-python** are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/); the overriding
goal of every change is to reproduce R `gamlss` numerically (see the
README for the verification methodology).

## [Unreleased]

### Added — shrink-to-zero P-spline smoother `pbz()`

Model formulas may now contain `pbz()` terms, e.g.

```python
m = gl.gamlss("y ~ pbz(x)", family=gl.NO(), data=ab)
```

`pbz()` (Durbán's double-penalty P-spline, `pb_goingtozero.R`) is `pb()`
plus a *second*, order-1 difference penalty that activates only when the
fit collapses to `edf ≤ lim`: the usual order-2 penalty shrinks the smooth
toward a straight *line*, and the extra order-1 penalty then shrinks it
toward a *constant* (a zero effect), so a term can drop out — useful for
model selection. It reuses the entire `pb()` machinery (basis, SVD
`regpen`, backfitting, `getSmo`/`predict`); the differences faithful to R
are that `pbz(x)` contributes a column of **zeros** (not the linear `x`) to
the parametric design, its non-linear df is `edf − 1` (not `edf − 2`), and
a warm-started `λ` that hits the `1e7` rail flips the fit to fixed-`λ` mode
and drops the order-1 penalty (R's start-value quirk, `pbz.R:223,237-242`).

Verified against R on ML and fixed-`λ` fits — basic and double-penalty-active
(pure-noise and strongly-linear data that shrink to a constant), `NO`/`GA`
links, `mu` and `sigma`, two smoothers, parametric + smoother mixes, and
under `CG()` — reproducing coefficients (including the aliased `NA`/`NaN`
column), per-smoother λ/edf, df, deviance/AIC/SBC, fitted values, predictions
and the exact iteration count (bit-exact ML/fixed-λ; rtol 1e-6 overall;
`tests/test_pbz.py`, reference `r-scripts/gen_pbz_reference.R`).

**Only ML (default) and a fixed `lambda` are supported.** R gamlss 5.5-0's
`gamlss.pbz()` is broken for `df` / `GAIC` / `GCV`: its inner `regpen()` is
defined as `function(y, X, w)` but those branches call it as
`regpen(y, X, w, lambda)`, so R aborts with `"unused argument (lambda)"`.
There is therefore no reference behaviour to match, and `pbz(df=…)`,
`pbz(method="GAIC")` and `pbz(method="GCV")` raise a clear
`NotImplementedError` (use `pb()` for those selectors).

Coverage was hardened by differential fuzzing — ~40 `pbz`/`pb` configurations
fit in both R and the port on byte-identical data and compared — exercising
the order-1 penalty at an *interior* fixed point (`lim=10`) and railed to a
constant, in `mu`/`sigma`/`nu` (incl. a 3-parameter `BCCG` fit), under
`RS`/`CG`/`mixed`, with weights, few-distinct-`x`, multiple smoothers and
`pbz`+`pb` mixes. Under `mixed`, the intercept↔smooth-mean split is
non-identifiable (as for `pb`) but every identifiable quantity — fitted
values, deviance, df, edf, λ and predictions — matches R.

### Fixed

- `pb()` ML smoothing now persists the **final** converged `λ` as the warm
  start for the next fit, matching R's `assign(startLambdaName, lambda)` after
  the ML loop (`pb.R:35`). The warm-start assignment sat *inside* the ML loop,
  after the convergence `break`, so on convergence the last iterate was
  dropped and the next `pb()` fit warm-started one step stale. The stale start
  nudged the ML trajectory and could add an extra backfitting sweep, drifting
  the RS per-iteration deviance off R's (converged results were unaffected).
  The `GAIC`/`GCV` branches and `pbz()` already persisted after their loops;
  `pb()` ML now matches (reported in #1).

- Smoother term labels now match R's `deparse` of numeric literals: a value
  like `lambda = 1000000` is rendered `lambda = 1e+06` (scientific notation
  when strictly shorter than fixed, 15 significant digits), so `pb()`/`pbz()`
  coefficient names are byte-identical to R for large/small λ.

## [0.2.0] - 2026-06-19

### Added — penalised B-spline smoothers `pb()`

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
  values, as in R's `predict.gamlss` — to the linear predictor. The smooth
  extrapolates *linearly* beyond the data range, matching R's
  `splinefun(method="natural")` (scipy's `CubicSpline` would extend the
  boundary cubic and diverge sharply — e.g. the wrong sign far outside the
  range). Verified against R for link/response predictions and
  `getSmo$fun(xeval)`, including extrapolation, to rtol 1e-6
  (`tests/test_pb.py`). Edge cases also covered: small n (R's `inter→10`
  clamp), prior weights including zeros (`N = sum(w≠0)`), `df` larger than
  the basis (R caps to 3), and too few distinct x — ML still fits the
  rank-deficient basis (SVD regpen), while df/GCV selection on it now raises
  a clear "B-basis is singular" error as R does, instead of silently
  returning a degenerate fit.

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

- **Step 5 — GAIC / GCV / `df` / `max.df` smoothing-parameter selection.**
  Beyond ML and a fixed `lambda`, `pb()` now supports `pb(x, df=5)` (target
  effective df, R `uniroot`), `pb(x, max.df=8)` (cap), `pb(x, method="GAIC",
  k=2)` and `pb(x, method="GCV")`. GAIC mirrors R's `nlminb` as a *local*
  search from the warm-start λ (a global search lands on a different optimum
  of the often-flat GAIC objective). df/max.df and GCV match R to ~1e-6 on
  edf/fitted/deviance (~3e-5 on the flat λ itself) rather than the ~1e-13 of
  ML and fixed-λ. **GAIC parity is not guaranteed:** when the GAIC objective
  is flat/multimodal, scipy's optimiser and R's `nlminb` can descend to
  different optima (e.g. with `k=4` R stops at λ=1e7/edf=2 while the port
  finds a lower-GAIC interior optimum, changing the iteration count). Both
  are valid GAIC-selected smooths; if exact R agreement matters, prefer ML,
  GCV, a fixed `df`, or a fixed `lambda`. Verified against R for selection
  in mu and sigma, under RS and CG, with single and multiple smoothers, and
  the inactive-`max.df` fallback to bit-exact ML; df selection is also
  checked independently (the fitted nl.df equals the requested df). Under
  CG/mixed the deterministic selectors match R (fixed `lambda` under CG is
  bit-exact; `df` under CG and under `mixed` match), but the optimiser-based
  ones are more sensitive there: CG's single-sweep backfitting amplifies the
  optimiser difference (GCV under CG drifts ~1e-4 with a different iteration
  count) and `max.df` under CG may not converge in R, so prefer RS for
  GCV / GAIC / `max.df`.

### Not yet supported (planned)

- Term plots / `lpred(type="terms")` for `pb()` / `pbz()` terms — currently
  raises `NotImplementedError` (*Step 3 follow-up*).
- Other smoothers (`cs`, `ps`, `ri`, `random`, ...).

## [0.1.0]

Initial port of the parametric GAMLSS core: R `gamlss` (5.5-0) and
`gamlss.dist` (6.1-1). The RS / CG / mixed fitting algorithms, all four
distribution parameters, link functions, ~40 families with their
`d`/`p`/`q`/`r` functions, R-compatible formulas (factors, `poly()`,
`cbind()`, offsets), weights, predictions, standard errors (qr and
numerical-Hessian vcov) and quantile residuals — verified against the
original R package across 43 reference models plus dense d/p/q/r grids
for every family.
