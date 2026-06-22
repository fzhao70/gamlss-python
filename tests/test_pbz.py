"""Verify the "going to zero" penalised B-spline pbz() against R gamlss.

pbz() is pb() with a second, order-1 penalty that activates when the fit
collapses to edf <= lim, shrinking the smooth toward a constant (so a term
can drop out for model selection).  Coverage is restricted to the two
selectors R gamlss 5.5-0 can actually run -- ML (default) and a fixed lambda;
its df / GAIC / GCV branches abort with "unused argument (lambda)" (an
upstream regpen() bug), so the port rejects those (see the dedicated test).

These are all deterministic fixed-point / fixed-lambda fits, so the Python fit
must reproduce R's coefficients, per-smoother lambda/edf, degrees of freedom,
deviance/AIC/SBC, fitted values and the exact iteration count.

Reference: r-scripts/gen_pbz_reference.R -> tests/reference/pbz_fits.json
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

import gamlss as gl

HERE = os.path.dirname(__file__)
REF = os.path.join(HERE, "reference", "pbz_fits.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(REF),
    reason="pbz reference not generated (run r-scripts/gen_pbz_reference.R)")

with open(REF) as fh:
    PBZ = json.load(fh)

NAMES = list(PBZ)
_cache = {}

# mixed (RS then CG) leaves the intercept <-> smooth-mean split at a
# path-dependent point: like pb, the *fitted model* (fitted values, deviance,
# df, edf, lambda, predictions) matches R exactly, but the raw intercept /
# coefficient split is not identifiable, and the RS->CG cycle count is
# BLAS-path dependent.  Assert the identifiable quantities only.
COEF_SPLIT_NONIDENTIFIABLE = {"pbz_abdom_mixed"}
ITER_COUNT_NONPORTABLE = {n for n in NAMES
                          if PBZ[n]["spec"].get("method") == "mixed"}


def _coef(vals):
    """R serialises the aliased pbz(x) coefficient as the string 'NA'
    (na='string' in write_json); the port returns NaN there -- map for an
    equal_nan comparison."""
    return np.array([np.nan if v == "NA" else v for v in vals], dtype=float)


def get_fit(name):
    if name not in _cache:
        ref = PBZ[name]
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
def test_pbz_fit_matches_r(name):
    ref = PBZ[name]
    m = get_fit(name)

    assert m.family[0] == ref["family"]
    assert m.parameters == list(ref["parameters"])

    # ML and fixed-lambda are deterministic -> identical iteration path
    # (mixed's RS->CG cycle count is BLAS-path dependent; see note above)
    if name not in ITER_COUNT_NONPORTABLE:
        assert m.iter == ref["iter"], \
            f"iteration count {m.iter} != R {ref['iter']}"
    assert bool(m.converged) == bool(ref["converged"])

    np.testing.assert_allclose(m.df_fit, ref["df.fit"], rtol=1e-6,
                               err_msg="df.fit")
    np.testing.assert_allclose(m.df_residual, ref["df.residual"], rtol=1e-6,
                               err_msg="df.residual")
    assert m.noObs == ref["noObs"]
    assert m.N == ref["N"]

    np.testing.assert_allclose(m.G_deviance, ref["G.deviance"], rtol=1e-8,
                               err_msg="global deviance")
    np.testing.assert_allclose(m.P_deviance, ref["P.deviance"], rtol=1e-8,
                               err_msg="penalised deviance")
    np.testing.assert_allclose(m.aic, ref["aic"], rtol=1e-8, err_msg="aic")
    np.testing.assert_allclose(m.sbc, ref["sbc"], rtol=1e-8, err_msg="sbc")

    for p in m.parameters:
        coef = getattr(m, f"{p}_coefficients")
        assert list(coef.index) == list(ref[f"coefnames.{p}"]), \
            f"{p} coefficient names"
        # the aliased pbz column is NaN in both R and the port (equal_nan)
        if name not in COEF_SPLIT_NONIDENTIFIABLE:
            np.testing.assert_allclose(np.asarray(coef, dtype=float),
                                       _coef(ref[f"coef.{p}"]), rtol=1e-6,
                                       atol=1e-7, equal_nan=True,
                                       err_msg=f"{p} coefficients")
        np.testing.assert_allclose(getattr(m, f"{p}_fv"), ref[f"fitted.{p}"],
                                   rtol=1e-6, atol=1e-7,
                                   err_msg=f"{p} fitted values")
        np.testing.assert_allclose(getattr(m, f"{p}_df"), ref[f"df.{p}"],
                                   rtol=1e-6, err_msg=f"{p} df")
        np.testing.assert_allclose(getattr(m, f"{p}_nl_df"), ref[f"nldf.{p}"],
                                   rtol=1e-6, atol=1e-7, err_msg=f"{p} nl.df")
        if f"lambda.{p}" in ref:
            cs = getattr(m, f"{p}_coefSmo")
            np.testing.assert_allclose([c["lambda"] for c in cs],
                                       ref[f"lambda.{p}"], rtol=1e-6,
                                       err_msg=f"{p} smoother lambda")
            np.testing.assert_allclose([c["edf"] for c in cs],
                                       ref[f"edf.{p}"], rtol=1e-6,
                                       err_msg=f"{p} smoother edf")


@pytest.mark.parametrize("name", [n for n in NAMES if "pred.newdata" in PBZ[n]])
def test_pbz_predict_matches_r(name):
    """getSmo() natural spline + predict() on new data, vs R predict.gamlss."""
    ref = PBZ[name]
    m = get_fit(name)
    nd = pd.DataFrame({k: np.asarray(v, dtype=float)
                       for k, v in ref["pred.newdata"].items()})

    if "getSmo.fun.mu" in ref:
        sm = m.getSmo("mu")
        np.testing.assert_allclose(sm["fun"](nd["x"].to_numpy()),
                                   ref["getSmo.fun.mu"], rtol=1e-6, atol=1e-6,
                                   err_msg="getSmo$fun(xeval)")

    for p in m.parameters:
        np.testing.assert_allclose(
            m.predict(what=p, newdata=nd, type="link"),
            ref[f"pred.link.{p}"], rtol=1e-6, atol=1e-6,
            err_msg=f"predict link {p}")
        np.testing.assert_allclose(
            m.predict(what=p, newdata=nd, type="response"),
            ref[f"pred.resp.{p}"], rtol=1e-6, atol=1e-6,
            err_msg=f"predict response {p}")


def test_pbz_extrapolation():
    """pbz extrapolation outside the training range must be LINEAR (R's
    splinefun(method='natural')), exactly as for pb."""
    ref = PBZ["pbz_abdom"]
    m = get_fit("pbz_abdom")
    ex = np.asarray(ref["extrap.x"], dtype=float)
    np.testing.assert_allclose(m.getSmo("mu")["fun"](ex), ref["extrap.fun"],
                               rtol=1e-6, atol=1e-6,
                               err_msg="getSmo$fun extrapolation")
    np.testing.assert_allclose(
        m.predict(what="mu", newdata=pd.DataFrame({"x": ex}), type="link"),
        ref["extrap.pred"], rtol=1e-6, atol=1e-6, err_msg="predict extrapolation")


def test_pbz_shrinks_toward_constant():
    """The defining pbz property: a term with no real signal collapses toward
    a constant (edf -> ~1-2 via the order-1 penalty).  Checked R-independently
    across the parameter it smooths -- mu (noise/linear), sigma (homoscedastic
    data) and nu (BCCG)."""
    for name, param in (("pbz_noise", "mu"), ("pbz_linear", "mu"),
                        ("pbz_sigma_do1", "sigma"), ("pbz_abdom_bccg", "nu")):
        edf = get_fit(name).getSmo(param)["edf"]
        assert edf <= 2.01, \
            f"{name}: {param} edf {edf} did not shrink toward a constant"


def test_pbz_order1_penalty_active_interior():
    """pbz(x, lim=10) on a real-signal smooth activates the order-1 penalty at
    an INTERIOR fixed point (the two-lambda ML loop, not railed): its edf must
    sit strictly between a constant and the plain pb edf, and below lim."""
    e_lim10 = get_fit("pbz_abdom_lim10").getSmo("mu")["edf"]
    e_plain = get_fit("pbz_abdom").getSmo("mu")["edf"]   # same data, lim=3 (off)
    assert 2.0 < e_lim10 < e_plain <= 10.0, \
        f"lim10 edf {e_lim10} not an interior do1order fit (plain {e_plain})"


def test_pbz_rejects_broken_selectors():
    """R gamlss 5.5-0's pbz() aborts for df / GAIC / GCV (an upstream regpen()
    argument bug), so the port refuses them with a clear NotImplementedError
    rather than diverging from a reference R cannot produce."""
    from gamlss.data import load_data
    ad = load_data("abdom")
    for call in ("y ~ pbz(x, df=5)",
                 'y ~ pbz(x, method="GAIC")',
                 'y ~ pbz(x, method="GCV")'):
        with pytest.raises(NotImplementedError, match="pbz"):
            gl.gamlss(call, family=gl.NO(), data=ad, trace=False)
