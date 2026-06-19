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


def get_fit(name):
    if name not in _cache:
        ref = PB[name]
        spec = ref["spec"]
        df = pd.DataFrame({k: np.asarray(v, dtype=float)
                           for k, v in ref["data"].items()})
        fam = getattr(gl.dist, spec["family"])()
        _cache[name] = gl.gamlss(
            spec["formula"],
            sigma_formula=spec.get("sigma_formula", "~1"),
            nu_formula=spec.get("nu_formula", "~1"),
            tau_formula=spec.get("tau_formula", "~1"),
            family=fam, data=df, n_cyc=200, trace=False)
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

    # degrees of freedom (non-integer for pb -> allclose, not exact ==)
    np.testing.assert_allclose(m.df_fit, ref["df.fit"], rtol=1e-6,
                               err_msg="df.fit")
    np.testing.assert_allclose(m.df_residual, ref["df.residual"], rtol=1e-6,
                               err_msg="df.residual")
    assert m.noObs == ref["noObs"]
    assert m.N == ref["N"]

    # deviance and information criteria
    np.testing.assert_allclose(m.G_deviance, ref["G.deviance"], rtol=1e-9,
                               err_msg="global deviance")
    np.testing.assert_allclose(m.P_deviance, ref["P.deviance"], rtol=1e-9,
                               err_msg="penalised deviance")
    np.testing.assert_allclose(m.aic, ref["aic"], rtol=1e-9, err_msg="aic")
    np.testing.assert_allclose(m.sbc, ref["sbc"], rtol=1e-9, err_msg="sbc")

    # per-parameter: coefficients (+ names), fitted, df, nl.df, smoother lambda/edf
    for p in m.parameters:
        coef = getattr(m, f"{p}_coefficients")
        assert list(coef.index) == list(ref[f"coefnames.{p}"]), \
            f"{p} coefficient names"
        np.testing.assert_allclose(np.asarray(coef, dtype=float),
                                   ref[f"coef.{p}"], rtol=1e-6, atol=1e-9,
                                   err_msg=f"{p} coefficients")
        np.testing.assert_allclose(getattr(m, f"{p}_fv"), ref[f"fitted.{p}"],
                                   rtol=1e-6, atol=1e-9,
                                   err_msg=f"{p} fitted values")
        np.testing.assert_allclose(getattr(m, f"{p}_df"), ref[f"df.{p}"],
                                   rtol=1e-6, err_msg=f"{p} df")
        np.testing.assert_allclose(getattr(m, f"{p}_nl_df"), ref[f"nldf.{p}"],
                                   rtol=1e-6, atol=1e-9, err_msg=f"{p} nl.df")
        if f"lambda.{p}" in ref:
            cs = getattr(m, f"{p}_coefSmo")
            np.testing.assert_allclose([c["lambda"] for c in cs],
                                       ref[f"lambda.{p}"], rtol=1e-6,
                                       err_msg=f"{p} smoother lambda")
            np.testing.assert_allclose([c["edf"] for c in cs],
                                       ref[f"edf.{p}"], rtol=1e-6,
                                       err_msg=f"{p} smoother edf")
