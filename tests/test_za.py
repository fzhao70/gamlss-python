"""Verify the zero adjusted continuous families ZAGA and ZAIG against R.

Reference: r-scripts/gen_za_reference.R (tests/reference/za_*.json). The
fit checks mirror tests/test_fits.py. ZAGA/ZAIG are "Mixed": the quantile
residuals of the zeros are randomised by construction, so only the
residuals of y > 0 are compared.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

import gamlss as gl
import gamlss.dist as dist

HERE = os.path.dirname(__file__)
REF = os.path.join(HERE, "reference")

with open(os.path.join(REF, "za_dpqr.json")) as fh:
    DPQR = json.load(fh)
with open(os.path.join(REF, "za_fits.json")) as fh:
    FITS = json.load(fh)["fits"]

# qIG is solved by root finding: R's uniroot stops at ~1.2e-4
LOOSE_Q = {"ZAIG": 2e-4}

CASES = {
    "zaga_sim": dict(family=lambda: gl.ZAGA(), data="sim_zaga", sigma="~x1", nu="~x1"),
    "zaga_sim_int": dict(family=lambda: gl.ZAGA(), data="sim_zaga", formula="y ~ 1"),
    "zaga_sim_cg": dict(family=lambda: gl.ZAGA(), data="sim_zaga", sigma="~x1", nu="~x1",
                        method=lambda: gl.CG()),
    "zaig_sim": dict(family=lambda: gl.ZAIG(), data="sim_zaig", nu="~x1"),
    "zaig_sim_probit": dict(family=lambda: gl.ZAIG(nu_link="probit"), data="sim_zaig",
                            sigma="~x1"),
}
NEWDATA = pd.DataFrame({"x1": [0.05, 0.5, 0.95]})
_cache = {}


def get_fit(name):
    if name not in _cache:
        spec = CASES[name]
        data = pd.read_csv(os.path.join(REF, "data", f"{spec['data']}.csv"))
        kw = dict(trace=False)
        if "method" in spec:
            kw["method"] = spec["method"]()
        m = gl.gamlss(spec.get("formula", "y ~ x1"), sigma_formula=spec.get("sigma", "~1"),
                      nu_formula=spec.get("nu", "~1"), family=spec["family"](), data=data, **kw)
        _cache[name] = (m, data)
    return _cache[name]


PARAMS = [(fam, i) for fam, cases in DPQR.items() for i in range(len(cases))]


@pytest.mark.parametrize("fam,i", PARAMS, ids=[f"{f}-{i}" for f, i in PARAMS])
def test_dpqr(fam, i):
    case = DPQR[fam][i]
    params = {k: np.asarray(v, dtype=float) for k, v in case["params"].items()}
    x = np.asarray(case["x"], dtype=float)
    ps = np.asarray(case["ps"], dtype=float)
    dfun, pfun, qfun = (getattr(dist, k + fam) for k in "dpq")
    np.testing.assert_allclose(dfun(x, **params), case["d"], rtol=1e-10, atol=1e-300)
    np.testing.assert_allclose(dfun(x, log=True, **params), case["dlog"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(pfun(x, **params), case["p"], rtol=1e-10, atol=1e-300)
    np.testing.assert_allclose(pfun(x, lower_tail=False, **params), case["pupper"],
                               rtol=1e-7, atol=4e-15)
    q = qfun(ps, **params)
    np.testing.assert_allclose(q, case["q"], rtol=LOOSE_Q.get(fam, 1e-9), atol=1e-12)
    if fam in LOOSE_Q:  # Python's root is more precise than R's (q = 0 is the point mass)
        pos = q > 0
        np.testing.assert_allclose(pfun(q[pos], **params), ps[pos], atol=1e-10)


@pytest.mark.parametrize("name", list(FITS))
def test_fit_matches_r(name):
    ref = FITS[name]
    m, data = get_fit(name)
    assert m.family[0] == ref["family"]
    assert m.parameters == list(ref["parameters"])
    assert m.df_fit == ref["df.fit"]
    assert m.df_residual == ref["df.residual"]
    assert m.noObs == ref["noObs"] and m.N == ref["N"]
    assert m.iter == ref["iter"], f"iteration count {m.iter} != R {ref['iter']}"
    assert bool(m.converged) == bool(ref["converged"])
    for k, attr in (("G.deviance", "G_deviance"), ("P.deviance", "P_deviance"),
                    ("aic", "aic"), ("sbc", "sbc")):
        np.testing.assert_allclose(getattr(m, attr), ref[k], rtol=1e-9, err_msg=k)
    for p in m.parameters:
        np.testing.assert_allclose(np.asarray(m.coef(p), dtype=float), ref[f"coef.{p}"],
                                   rtol=1e-6, atol=1e-9, err_msg=f"{p} coefficients")
        np.testing.assert_allclose(m.fitted(p)[:10], ref[f"fitted10.{p}"], rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(np.sum(m.fitted(p)), ref[f"fittedsum.{p}"], rtol=1e-8)
        got = m.predict(what=p, newdata=NEWDATA, type="response")
        np.testing.assert_allclose(np.asarray(got, dtype=float), ref[f"pred.{p}.response"],
                                   rtol=1e-6, err_msg=f"{p} predictions")
    r = np.asarray(m.get_residuals())[data["y"].to_numpy() > 0]
    np.testing.assert_allclose(r[:10], ref["resid_pos10"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(np.sum(r), ref["resid_possum"], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", list(FITS))
def test_se_qr(name):
    from scipy.linalg import solve_triangular
    ref = FITS[name]
    m, _ = get_fit(name)
    for p in m.parameters:
        qr = getattr(m, f"{p}_qr")
        rank = qr["rank"]
        Rinv = solve_triangular(qr["R"][:rank, :rank], np.eye(rank))
        np.testing.assert_allclose(np.sqrt(np.sum(Rinv**2, axis=1)), ref[f"se_qr.{p}"],
                                   rtol=1e-6, err_msg=f"{p} qr-type std errors")


@pytest.mark.parametrize("name", [n for n in FITS if FITS[n].get("se_vcov") is not None])
def test_se_vcov(name):
    ref = FITS[name]
    m, _ = get_fit(name)
    se = np.asarray(m.vcov(type="se"), dtype=float)
    rtol = 1e-2 if m._vcov_used_fallback else 2e-4
    np.testing.assert_allclose(se, ref["se_vcov"], rtol=rtol, atol=1e-10)


def test_rqres_mixed_zero_randomised():
    """Zeros get uniforms on (0, nu): residuals <= qnorm(nu) there."""
    from scipy.stats import norm
    m, data = get_fit("zaga_sim")
    zero = data["y"].to_numpy() == 0
    r = np.asarray(m.get_residuals())
    assert np.all(r[zero] <= norm.ppf(m.fitted("nu")[zero]) + 1e-12)
