"""API-surface tests: the R-like interface works end to end."""

from __future__ import annotations

import io
import contextlib

import numpy as np
import pandas as pd
import pytest

import gamlss as gl


@pytest.fixture(scope="module")
def abdom():
    return gl.load_data("abdom")


@pytest.fixture(scope="module")
def m_no(abdom):
    return gl.gamlss("y ~ x", sigma_formula="~x", family=gl.NO(),
                     data=abdom, trace=False)


def test_print_repr(m_no):
    out = repr(m_no)
    assert "Family:  ('NO', 'Normal')" in out
    assert "Global Deviance" in out
    assert "Mu Coefficients" in out


def test_summary_prints_both_types(m_no):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tab_v = m_no.summary(type="vcov")
    text = buf.getvalue()
    assert "Mu link function:  identity" in text
    assert "Sigma link function:  log" in text
    assert "Global Deviance" in text
    assert list(tab_v.columns) == ["Estimate", "Std. Error", "t value",
                                   "Pr(>|t|)"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tab_q = m_no.summary(type="qr")
    assert len(tab_q) == len(tab_v) == 4
    # both routes agree on the estimates
    np.testing.assert_allclose(tab_q["Estimate"], tab_v["Estimate"],
                               rtol=1e-12)


def test_functional_generics(m_no):
    assert gl.is_gamlss(m_no)
    np.testing.assert_allclose(gl.fitted(m_no, "mu"), m_no.mu_fv)
    np.testing.assert_allclose(gl.fv(m_no, "sigma"), m_no.sigma_fv)
    assert gl.coef(m_no, "mu").index[0] == "(Intercept)"
    all_ = gl.coefAll(m_no)
    assert set(all_) == {"mu", "sigma"}
    assert gl.deviance(m_no) == m_no.G_deviance
    assert gl.logLik(m_no) == -m_no.G_deviance / 2
    np.testing.assert_allclose(gl.lp(m_no, "mu"), m_no.mu_lp)
    with pytest.raises(ValueError):
        gl.fitted(m_no, "tau")


def test_aic_table(m_no, abdom):
    m2 = gl.gamlss("y ~ poly(x, 3)", sigma_formula="~x", family=gl.NO(),
                   data=abdom, trace=False)
    tab = gl.AIC(m_no, m2)
    assert list(tab.columns) == ["df", "AIC"]
    assert tab.iloc[0]["AIC"] <= tab.iloc[1]["AIC"]
    # AICc
    aicc = gl.AIC(m_no, c=True)
    assert aicc > gl.AIC(m_no)


def test_residual_types(m_no):
    r_simple = m_no.get_residuals(what="mu", type="simple")
    r_weight = m_no.get_residuals(what="mu", type="weighted")
    np.testing.assert_allclose(
        r_weight, np.sqrt(m_no.mu_wt) * r_simple, rtol=1e-12
    )
    z = gl.residuals(m_no)
    assert z.shape == (m_no.N,)


def test_predict_all(m_no, abdom):
    nd = abdom.head(3)[["x"]]
    out = m_no.predictAll(newdata=nd)
    assert set(out) >= {"mu", "sigma"}
    np.testing.assert_allclose(
        out["mu"], m_no.predict(what="mu", newdata=nd, type="response"))
    df = m_no.predictAll(newdata=nd, output="data.frame")
    assert isinstance(df, pd.DataFrame)


def test_no_newdata_predict_identities(m_no):
    np.testing.assert_allclose(m_no.predict(what="mu"), m_no.mu_lp)
    np.testing.assert_allclose(
        m_no.predict(what="mu", type="response"), m_no.mu_fv)


def test_rsq(m_no):
    r2 = gl.Rsq(m_no)
    assert 0 < r2 < 1


def test_start_from(abdom, m_no):
    m2 = gl.gamlss("y ~ x", sigma_formula="~x", family=gl.NO(),
                   data=abdom, start_from=m_no, trace=False)
    np.testing.assert_allclose(m2.G_deviance, m_no.G_deviance, rtol=1e-7)


def test_fix_and_start(abdom):
    m = gl.gamlss("y ~ x", family=gl.NO(), data=abdom,
                  sigma_start=20.0, sigma_fix=True, trace=False)
    assert m.sigma_fix is True
    assert m.sigma_df == 0
    np.testing.assert_allclose(m.sigma_fv, 20.0)
    # vcov treats the fixed parameter like R (fixed sigma branch)
    se = m.vcov(type="se")
    assert "fixed sigma" in se.index


def test_weights_noninteger_warns(abdom):
    w = np.ones(len(abdom))
    w[0] = 1.5
    m = gl.gamlss("y ~ x", family=gl.NO(), data=abdom, weights=w,
                  trace=False)
    assert m.noObs == m.N  # non-integer weights: noObs = N
    with pytest.warns(UserWarning):
        m.get_residuals()


def test_na_data_raises():
    df = pd.DataFrame({"y": [1.0, np.nan, 3.0], "x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="NA"):
        gl.gamlss("y ~ x", family=gl.NO(), data=df, trace=False)


def test_family_lookup_by_string(abdom):
    m = gl.gamlss("y ~ x", family="NO", data=abdom, trace=False)
    assert m.family[0] == "NO"


def test_control_validation_warns():
    with pytest.warns(UserWarning):
        gl.gamlss_control(c_crit=-1)
    with pytest.warns(UserWarning):
        gl.glim_control(cyc=0)


def test_links_roundtrip():
    for name in ["identity", "log", "logit", "probit", "cloglog",
                 "inverse", "sqrt", "1/mu^2", "mu^2", "cauchit"]:
        lk = gl.make_link_gamlss(name)
        mu = np.array([0.2, 0.5, 0.7]) if name in (
            "logit", "probit", "cloglog", "cauchit") else np.array(
            [0.5, 1.0, 2.0])
        eta = lk.linkfun(mu)
        np.testing.assert_allclose(lk.linkinv(eta), mu, rtol=1e-9)


def test_offset_formula(abdom):
    df = abdom.copy()
    df["t"] = np.linspace(1, 2, len(df))
    m = gl.gamlss("y ~ x + offset(log(t))", family=gl.NO(), data=df,
                  trace=False)
    np.testing.assert_allclose(m.mu_offset, np.log(df["t"]))
    assert "offset" not in " ".join(m.mu_coefficients.index)
