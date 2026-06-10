"""Verify gamlss model fits against the original R gamlss results.

For every reference case the Python fit must reproduce R's
coefficients, deviance, AIC/SBC, degrees of freedom, iteration count,
fitted values, (continuous) quantile residuals, qr-based standard
errors and the numerical-Hessian (vcov) standard errors.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

import gamlss as gl
from cases import CASES, fit_case

HERE = os.path.dirname(__file__)

with open(os.path.join(HERE, "reference", "fits.json")) as fh:
    FITS = json.load(fh)

_fitted_cache = {}


def get_fit(name):
    if name not in _fitted_cache:
        _fitted_cache[name] = fit_case(name)
    return _fitted_cache[name]


NAMES = [n for n in FITS if n in CASES]

# near-flat likelihood directions (e.g. tau -> inf in BCTo) surface fp
# noise in the stationary point at ~1e-5 even though deviance agrees
# to 1e-9; per-case coefficient tolerance overrides.
COEF_RTOL = {"bcto_abdom": 1e-4}


@pytest.mark.parametrize("name", NAMES)
def test_fit_matches_r(name):
    ref = FITS[name]
    m, data = get_fit(name)

    # family and structure
    assert m.family[0] == ref["family"]
    assert m.parameters == list(ref["parameters"])

    # degrees of freedom and observation counts: exact
    assert m.df_fit == ref["df.fit"]
    assert m.df_residual == ref["df.residual"]
    assert m.noObs == ref["noObs"]
    assert m.N == ref["N"]

    # convergence behaviour: identical path -> identical cycle count
    assert m.iter == ref["iter"], f"iteration count {m.iter} != R {ref['iter']}"
    assert bool(m.converged) == bool(ref["converged"])

    # global deviance and information criteria
    np.testing.assert_allclose(m.G_deviance, ref["G.deviance"], rtol=1e-9,
                               err_msg="global deviance")
    np.testing.assert_allclose(m.P_deviance, ref["P.deviance"], rtol=1e-9)
    np.testing.assert_allclose(m.aic, ref["aic"], rtol=1e-9)
    np.testing.assert_allclose(m.sbc, ref["sbc"], rtol=1e-9)

    # per-parameter coefficients and fitted values
    crtol = COEF_RTOL.get(name, 1e-6)
    for p in m.parameters:
        key = f"coef.{p}"
        if key in ref:
            np.testing.assert_allclose(
                np.asarray(m.coef(p), dtype=float), ref[key],
                rtol=crtol, atol=1e-9, err_msg=f"{p} coefficients")
        np.testing.assert_allclose(
            m.fitted(p)[:10], ref[f"fitted10.{p}"], rtol=max(crtol, 1e-6),
            atol=1e-9, err_msg=f"{p} fitted values")
        np.testing.assert_allclose(
            np.sum(m.fitted(p)), ref[f"fittedsum.{p}"],
            rtol=max(crtol, 1e-8), err_msg=f"{p} fitted sum")

    # quantile residuals via resid() (continuous: deterministic;
    # integer frequency weights expand the vector as in R)
    if "resid10" in ref and ref.get("resid10") is not None:
        r = m.get_residuals()
        np.testing.assert_allclose(r[:10], ref["resid10"],
                                   rtol=1e-6, atol=1e-8,
                                   err_msg="quantile residuals")
        np.testing.assert_allclose(np.sum(r), ref["residsum"],
                                   rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", NAMES)
def test_se_qr(name):
    """qr-type standard errors: chol2inv of the working-WLS R matrix."""
    ref = FITS[name]
    m, _ = get_fit(name)
    from scipy.linalg import solve_triangular

    for p in m.parameters:
        key = f"se_qr.{p}"
        if key not in ref:
            continue
        qr = getattr(m, f"{p}_qr")
        rank = qr["rank"]
        R = qr["R"][:rank, :rank]
        Rinv = solve_triangular(R, np.eye(rank))
        se = np.sqrt(np.sum(Rinv**2, axis=1))
        np.testing.assert_allclose(se, ref[key],
                                   rtol=max(COEF_RTOL.get(name, 1e-6), 1e-6),
                                   err_msg=f"{p} qr-type std errors")


@pytest.mark.parametrize("name", [n for n in NAMES
                                  if FITS[n].get("se_vcov") is not None])
def test_se_vcov(name):
    """vcov standard errors via the optimHess numerical Hessian.

    When the optimHess matrix is indefinite, R (and the port) fall
    back to HessianPB, whose tiny quadratic-fit coefficients amplify
    last-bit density differences between R and scipy; tolerance is
    relaxed for those cases.
    """
    ref = FITS[name]
    ref_se = np.array([np.nan if v == "NaN" else v for v in ref["se_vcov"]],
                      dtype=float)
    if np.any(np.isnan(ref_se)):
        pytest.skip("R's vcov is degenerate (NaN se) for this fit; "
                    "R's summary falls back to type='qr' here")
    m, _ = get_fit(name)
    se = np.asarray(m.vcov(type="se"), dtype=float)
    rtol = 1e-2 if m._vcov_used_fallback else 2e-4
    np.testing.assert_allclose(se, ref_se, rtol=rtol, atol=1e-10,
                               err_msg="vcov std errors")


def test_predict_no_abdom():
    ref = FITS["no_abdom"]
    m, data = get_fit("no_abdom")
    nd = pd.DataFrame({"x": [15.5, 25, 38.2]})
    np.testing.assert_allclose(
        m.predict(what="mu", newdata=nd), ref["pred.mu.link"], rtol=1e-8)
    np.testing.assert_allclose(
        m.predict(what="mu", newdata=nd, type="response"),
        ref["pred.mu.response"], rtol=1e-8)
    np.testing.assert_allclose(
        m.predict(what="sigma", newdata=nd, type="response"),
        ref["pred.sigma.response"], rtol=1e-8)
    se_link = m.lpred(what="mu", se_fit=True)["se.fit"][:10]
    np.testing.assert_allclose(se_link, ref["se.link.mu"], rtol=1e-6)
    se_resp = m.lpred(what="mu", type="response", se_fit=True)["se.fit"][:10]
    np.testing.assert_allclose(se_resp, ref["se.response.mu"], rtol=1e-6)


def test_predict_poly_and_terms():
    ref = FITS["no_abdom_poly"]
    m, data = get_fit("no_abdom_poly")
    nd = pd.DataFrame({"x": [15.5, 25, 38.2]})
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = m.predict(what="mu", newdata=nd)
    np.testing.assert_allclose(pred, ref["pred.mu.link"], rtol=1e-7)
    terms = m.lpred(what="mu", type="terms")
    np.testing.assert_allclose(terms.attrs["constant"],
                               ref["terms.mu.const"], rtol=1e-8)
    np.testing.assert_allclose(terms.iloc[:10, 0], ref["terms.mu.col1"],
                               rtol=1e-7, atol=1e-10)


def test_predict_po_aids_factor():
    ref = FITS["po_aids"]
    m, data = get_fit("po_aids")
    nd = pd.DataFrame({"x": [10, 46], "qrt": [2, 4]})
    np.testing.assert_allclose(
        m.predict(what="mu", newdata=nd, type="response"),
        ref["pred.mu.response"], rtol=1e-8)


def test_gaic_loglik():
    ref = FITS["no_abdom"]
    m, _ = get_fit("no_abdom")
    assert np.isclose(gl.GAIC(m, k=2), ref["aic"], rtol=1e-10)
    assert np.isclose(gl.GAIC(m, k=np.log(m.noObs)), ref["sbc"], rtol=1e-10)
    assert np.isclose(gl.logLik(m), -ref["G.deviance"] / 2, rtol=1e-12)
    assert np.isclose(gl.deviance(m), ref["G.deviance"], rtol=1e-12)
