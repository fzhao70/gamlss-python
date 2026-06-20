"""Verify penalised B-spline (pb) fits against the original R gamlss.

Step 2 coverage: pb() in mu and/or sigma, identity and log links, single
and multiple smoothers, parametric + smoother mixes, ML and fixed-lambda
smoothing, on real (abdom, n=610 so inter=20 is unclamped) and simulated
data.  Each case carries byte-identical data, so the Python fit must
reproduce R's coefficients, per-smoother lambda/edf, degrees of freedom,
deviance/AIC/SBC, fitted values and the exact RS iteration count.

Reference: r-scripts/gen_pb_reference.R -> tests/reference/pb_fits.json
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

import gamlss as gl

HERE = os.path.dirname(__file__)
REF = os.path.join(HERE, "reference", "pb_fits.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(REF),
    reason="pb reference not generated (run r-scripts/gen_pb_reference.R)")

with open(REF) as fh:
    PB = json.load(fh)

NAMES = list(PB)
_cache = {}

# The linear column of pb(x) and the smooth are concurve (the smooth can
# also represent a linear trend), so the split of the total x-effect into a
# parametric linear coefficient vs the smooth is NOT identifiable.  Fits run
# to full convergence (pure RS / pure CG) pin it reproducibly, but an
# early-stopped fit (mixed: RS(2) then CG(1)) leaves it at a path-dependent
# point that differs from R while the *fitted model* is identical.  For such
# cases we verify the identifiable quantities (fitted, deviance, df, edf,
# lambda) and skip only the raw-coefficient split.
COEF_SPLIT_NONIDENTIFIABLE = {"pb_abdom_mixed"}

# df / max.df select lambda by root-finding (R uniroot) and GAIC / GCV by
# optimisation (R nlminb); scipy's solvers don't reproduce those stopping
# points bit-for-bit.  Measured parity is ~1e-6 on edf/fitted/deviance and
# ~3e-5 on the (flat, least-determined) lambda itself, vs ~1e-13 for ML and
# fixed-lambda.  These cases therefore use a relaxed tolerance.
OPTIMIZER_SELECTION = {"pb_abdom_df", "pb_abdom_maxdf", "pb_abdom_gaic",
                       "pb_abdom_gcv", "pb_abdom_df_sigma", "pb_abdom_df3",
                       "pb_abdom_df_cg", "pb_sim_two_df", "pb_abdom_df_mixed",
                       "pb_df_toobig"}
# Not here (matched to ~1e-13, strict): pb_abdom_maxdf_off (inactive cap ->
# plain ML) and pb_abdom_fixlam_cg (fixed lambda under CG is deterministic).


def get_fit(name):
    if name not in _cache:
        ref = PB[name]
        spec = ref["spec"]
        df = pd.DataFrame({k: np.asarray(v, dtype=float)
                           for k, v in ref["data"].items()})
        fam = getattr(gl.dist, spec["family"])()
        method = {"CG": gl.CG(), "mixed": gl.mixed(2, 20)}.get(
            spec.get("method"))
        kw = {"method": method} if method is not None else {}
        if spec.get("weights"):
            kw["weights"] = df[spec["weights"]].to_numpy()
        _cache[name] = gl.gamlss(
            spec["formula"],
            sigma_formula=spec.get("sigma_formula", "~1"),
            nu_formula=spec.get("nu_formula", "~1"),
            tau_formula=spec.get("tau_formula", "~1"),
            family=fam, data=df, n_cyc=200, trace=False, **kw)
    return _cache[name]


@pytest.mark.parametrize("name", NAMES)
def test_pb_fit_matches_r(name):
    ref = PB[name]
    m = get_fit(name)

    # family / structure
    assert m.family[0] == ref["family"]
    assert m.parameters == list(ref["parameters"])

    # identical iteration path -> identical cycle count and convergence
    assert m.iter == ref["iter"], f"iteration count {m.iter} != R {ref['iter']}"
    assert bool(m.converged) == bool(ref["converged"])

    # ML / fixed-lambda match R to ~1e-13; optimiser-selected lambdas to ~1e-6
    rt = 1e-4 if name in OPTIMIZER_SELECTION else 1e-6
    rt_dev = 1e-4 if name in OPTIMIZER_SELECTION else 1e-9

    # degrees of freedom (non-integer for pb -> allclose, not exact ==)
    np.testing.assert_allclose(m.df_fit, ref["df.fit"], rtol=rt,
                               err_msg="df.fit")
    np.testing.assert_allclose(m.df_residual, ref["df.residual"], rtol=rt,
                               err_msg="df.residual")
    assert m.noObs == ref["noObs"]
    assert m.N == ref["N"]

    # deviance and information criteria
    np.testing.assert_allclose(m.G_deviance, ref["G.deviance"], rtol=rt_dev,
                               err_msg="global deviance")
    np.testing.assert_allclose(m.P_deviance, ref["P.deviance"], rtol=rt_dev,
                               err_msg="penalised deviance")
    np.testing.assert_allclose(m.aic, ref["aic"], rtol=rt_dev, err_msg="aic")
    np.testing.assert_allclose(m.sbc, ref["sbc"], rtol=rt_dev, err_msg="sbc")

    # per-parameter: coefficients (+ names), fitted, df, nl.df, smoother lambda/edf
    for p in m.parameters:
        coef = getattr(m, f"{p}_coefficients")
        assert list(coef.index) == list(ref[f"coefnames.{p}"]), \
            f"{p} coefficient names"
        if name not in COEF_SPLIT_NONIDENTIFIABLE:
            np.testing.assert_allclose(np.asarray(coef, dtype=float),
                                       ref[f"coef.{p}"], rtol=rt, atol=1e-7,
                                       err_msg=f"{p} coefficients")
        np.testing.assert_allclose(getattr(m, f"{p}_fv"), ref[f"fitted.{p}"],
                                   rtol=rt, atol=1e-7,
                                   err_msg=f"{p} fitted values")
        np.testing.assert_allclose(getattr(m, f"{p}_df"), ref[f"df.{p}"],
                                   rtol=rt, err_msg=f"{p} df")
        np.testing.assert_allclose(getattr(m, f"{p}_nl_df"), ref[f"nldf.{p}"],
                                   rtol=rt, atol=1e-7, err_msg=f"{p} nl.df")
        if f"lambda.{p}" in ref:
            cs = getattr(m, f"{p}_coefSmo")
            np.testing.assert_allclose([c["lambda"] for c in cs],
                                       ref[f"lambda.{p}"], rtol=rt,
                                       err_msg=f"{p} smoother lambda")
            np.testing.assert_allclose([c["edf"] for c in cs],
                                       ref[f"edf.{p}"], rtol=rt,
                                       err_msg=f"{p} smoother edf")


@pytest.mark.parametrize("name", [n for n in NAMES if "pred.newdata" in PB[n]])
def test_pb_predict_matches_r(name):
    """getSmo() spline + predict() on new data, vs R predict.gamlss.

    Covers single smoother (mu), smoothers in mu+sigma, two smoothers in
    one parameter, and a parametric + smoother mix.
    """
    ref = PB[name]
    m = get_fit(name)
    nd = pd.DataFrame({k: np.asarray(v, dtype=float)
                       for k, v in ref["pred.newdata"].items()})
    rt = 1e-4 if name in OPTIMIZER_SELECTION else 1e-6

    # getSmo(m, "mu")$fun must reproduce R's natural spline (single-smoother
    # mu cases, where a 1-D x grid is well defined)
    if "getSmo.fun.mu" in ref:
        sm = m.getSmo("mu")
        np.testing.assert_allclose(sm["fun"](nd["x"].to_numpy()),
                                   ref["getSmo.fun.mu"], rtol=rt, atol=1e-6,
                                   err_msg="getSmo$fun(xeval)")

    # full prediction (parametric + every smoother) on new data
    for p in m.parameters:
        np.testing.assert_allclose(
            m.predict(what=p, newdata=nd, type="link"),
            ref[f"pred.link.{p}"], rtol=rt, atol=1e-6,
            err_msg=f"predict link {p}")
        np.testing.assert_allclose(
            m.predict(what=p, newdata=nd, type="response"),
            ref[f"pred.resp.{p}"], rtol=rt, atol=1e-6,
            err_msg=f"predict response {p}")


# requested target df per case (the defining property of df selection),
# verified independently of R
DF_TARGETS = {
    "pb_abdom_df": {"mu": [5]},
    "pb_abdom_df3": {"mu": [3]},
    "pb_abdom_df_cg": {"mu": [5]},
    "pb_abdom_df_sigma": {"sigma": [4]},
    "pb_sim_two_df": {"mu": [4, 6]},
    "pb_df_toobig": {"mu": [3]},   # df=50 > basis -> R caps to df=3
}


@pytest.mark.parametrize("name", list(DF_TARGETS))
def test_pb_df_selection_hits_target(name):
    """pb(x, df=k) must produce a smooth whose nl.df == k (R-independent)."""
    m = get_fit(name)
    for p, want in DF_TARGETS[name].items():
        got = [c["nl_df"] for c in getattr(m, f"{p}_coefSmo")]
        np.testing.assert_allclose(got, want, atol=1e-3,
                                   err_msg=f"{name} {p}: nl.df {got} != df {want}")


def test_pb_maxdf_cap():
    """max.df caps the edf when active, and is a no-op (== ML) when not."""
    # active: ML edf (~5.5) exceeds max.df=5 -> capped at 5
    assert get_fit("pb_abdom_maxdf").mu_coefSmo[0]["edf"] <= 5.0 + 1e-3
    # inactive: max.df=20 > ML edf -> identical to the plain ML pb(x) fit
    e_off = get_fit("pb_abdom_maxdf_off").mu_coefSmo[0]["edf"]
    e_ml = get_fit("pb_abdom").mu_coefSmo[0]["edf"]
    np.testing.assert_allclose(e_off, e_ml, rtol=1e-12)


def test_pb_predict_extrapolation():
    """Smooth extrapolation outside the training range must be LINEAR, as in
    R's splinefun(method="natural") -- not the boundary cubic that scipy's
    CubicSpline extends (which diverges sharply, e.g. wrong sign far out)."""
    ref = PB["pb_abdom"]
    m = get_fit("pb_abdom")
    ex = np.asarray(ref["extrap.x"], dtype=float)  # x below and above the range
    np.testing.assert_allclose(m.getSmo("mu")["fun"](ex), ref["extrap.fun"],
                               rtol=1e-6, atol=1e-6,
                               err_msg="getSmo$fun extrapolation (must be linear)")
    np.testing.assert_allclose(
        m.predict(what="mu", newdata=pd.DataFrame({"x": ex}), type="link"),
        ref["extrap.pred"], rtol=1e-6, atol=1e-6,
        err_msg="predict extrapolation")


def test_pb_singular_basis_errors_like_r():
    """df selection on a rank-deficient basis (too few distinct x) must raise
    a clear 'B-basis is singular' error, as R does (not silently return a
    degenerate edf=2 fit).  ML on the same data still fits (SVD regpen)."""
    rng = np.random.default_rng(11)
    x = rng.integers(1, 9, 200).astype(float)        # 8 distinct values
    y = 2 + np.sin(x) + rng.normal(0, 0.3, 200)
    data = pd.DataFrame({"x": x, "y": y})
    # ML: fits fine despite rank-deficiency
    m = gl.gamlss("y ~ pb(x)", family=gl.NO(), data=data, trace=False)
    assert np.isfinite(m.getSmo("mu")["edf"])
    # df selection: solve(R) is singular -> clear error
    with pytest.raises(ValueError, match="singular"):
        gl.gamlss("y ~ pb(x, df=5)", family=gl.NO(), data=data, trace=False)


def test_pb_getsmo_indexing_and_terms_guard():
    """getSmo() indexing (1-based, which=0 -> list) and the terms guard."""
    m = get_fit("pb_sim_two")          # mu has two smoothers: pb(x1), pb(x2)
    allsmo = m.getSmo("mu", which=0)
    assert isinstance(allsmo, list) and len(allsmo) == 2
    assert m.getSmo("mu", which=1) is allsmo[0]
    assert m.getSmo("mu", which=2) is allsmo[1]
    assert m.getSmo(parameter="mu", which=1) is allsmo[0]
    with pytest.raises(ValueError):    # sigma has no smoother here
        m.getSmo("sigma")
    # type="terms" with smoothers is not supported yet -> explicit error,
    # rather than silently dropping the smooth term
    with pytest.raises(NotImplementedError):
        m.lpred(what="mu", type="terms")
    with pytest.raises(NotImplementedError):
        m.predict(what="mu", newdata=pd.DataFrame({"x1": [0.3], "x2": [0.4]}),
                  type="terms")
