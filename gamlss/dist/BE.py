"""Beta distribution (BE). Port of gamlss.dist R/BE.R.

Parameterised so that E(y) = mu and Var(y) = sigma^2 * mu * (1 - mu),
with y on (0, 1).
"""

from __future__ import annotations

import numpy as np
from scipy import special as _sp
from scipy import stats as _st

from ..family import GamlssFamily, checklink


def BE(mu_link="logit", sigma_link="logit"):
    mstats = checklink("mu.link", "Beta", mu_link,
                       ("logit", "probit", "cloglog", "cauchit", "log", "own"))
    dstats = checklink("sigma.link", "Beta", sigma_link,
                       ("logit", "probit", "cloglog", "cauchit", "log", "own"))

    def dldm(y, mu, sigma):
        a = mu * (1 - sigma**2) / (sigma**2)
        b = a * (1 - mu) / mu
        dldm = ((1 - sigma**2) / (sigma**2)) * (
            -_sp.digamma(a) + _sp.digamma(b) + np.log(y) - np.log(1 - y)
        )
        return dldm

    def d2ldm2(mu, sigma):
        a = mu * (1 - sigma**2) / (sigma**2)
        b = a * (1 - mu) / mu
        d2ldm2 = -(((1 - sigma**2) ** 2) / (sigma**4)) * (
            _sp.polygamma(1, a) + _sp.polygamma(1, b)
        )
        return d2ldm2

    def dldd(y, mu, sigma):
        a = mu * (1 - sigma**2) / (sigma**2)
        b = a * (1 - mu) / mu
        dldd = -(2 / (sigma**3)) * (
            mu * (-_sp.digamma(a) + _sp.digamma(a + b) + np.log(y))
            + (1 - mu) * (-_sp.digamma(b) + _sp.digamma(a + b)
                          + np.log(1 - y))
        )
        return dldd

    def d2ldd2(mu, sigma):
        a = mu * (1 - sigma**2) / (sigma**2)
        b = a * (1 - mu) / mu
        d2ldd2 = -(4 / (sigma**6)) * (
            (mu**2) * _sp.polygamma(1, a)
            + ((1 - mu) ** 2) * _sp.polygamma(1, b)
            - _sp.polygamma(1, a + b)
        )
        return d2ldd2

    def d2ldmdd(mu, sigma):
        a = mu * (1 - sigma**2) / (sigma**2)
        b = a * (1 - mu) / mu
        d2ldmdd = (2 * (1 - sigma**2) / (sigma**5)) * (
            mu * _sp.polygamma(1, a) - (1 - mu) * _sp.polygamma(1, b)
        )
        return d2ldmdd

    return GamlssFamily(
        family=("BE", "Beta"),
        parameters={"mu": True, "sigma": True},
        nopar=2,
        type="Continuous",
        links={"mu": mstats, "sigma": dstats},
        derivatives={
            "dldm": dldm,
            "d2ldm2": d2ldm2,
            "dldd": dldd,
            "d2ldd2": d2ldd2,
            "d2ldmdd": d2ldmdd,
        },
        G_dev_incr=lambda y, mu, sigma: -2 * dBE(y, mu, sigma, log=True),
        rqres={"pfun": "pBE", "type": "Continuous"},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.full(len(y), 0.5),
        },
        valid={
            "mu": lambda mu: bool(np.all((mu > 0) & (mu < 1))),
            "sigma": lambda sigma: bool(np.all((sigma > 0) & (sigma < 1))),
        },
        y_valid=lambda y: bool(np.all((y > 0) & (y < 1))),
        mean=lambda mu, sigma: mu,
        variance=lambda mu, sigma: sigma**2 * mu * (1 - mu),
    )


def dBE(x, mu=0.5, sigma=0.2, log=False):
    if np.any(np.asarray(mu) <= 0) or np.any(np.asarray(mu) >= 1):
        raise ValueError("mu must be between 0 and 1")
    if np.any(np.asarray(sigma) <= 0) or np.any(np.asarray(sigma) >= 1):
        raise ValueError("sigma must be between 0 and 1")
    x, mu, sigma = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    a = mu * (1 - sigma**2) / (sigma**2)
    b = a * (1 - mu) / mu
    with np.errstate(divide="ignore", invalid="ignore"):
        fy = _st.beta.logpdf(x, a, b) if log else _st.beta.pdf(x, a, b)
    return fy


def pBE(q, mu=0.5, sigma=0.2, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0) or np.any(np.asarray(mu) >= 1):
        raise ValueError("mu must be between 0 and 1")
    if np.any(np.asarray(sigma) <= 0) or np.any(np.asarray(sigma) >= 1):
        raise ValueError("sigma must be between 0 and 1")
    q = np.asarray(q, float)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    a = mu * (1 - sigma**2) / (sigma**2)
    b = a * (1 - mu) / mu
    if lower_tail:
        cdf = _st.beta.logcdf(q, a, b) if log_p else _st.beta.cdf(q, a, b)
    else:
        cdf = _st.beta.logsf(q, a, b) if log_p else _st.beta.sf(q, a, b)
    return cdf


def qBE(p, mu=0.5, sigma=0.2, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0) or np.any(np.asarray(mu) >= 1):
        raise ValueError("mu must be between 0 and 1")
    if np.any(np.asarray(sigma) <= 0) or np.any(np.asarray(sigma) >= 1):
        raise ValueError("sigma must be between 0 and 1")
    p = np.asarray(p, float)
    if np.any(p <= 0) or np.any(p >= 1):
        raise ValueError("p must be between 0 and 1")
    if log_p:
        p = np.exp(p)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    a = mu * (1 - sigma**2) / (sigma**2)
    b = a * (1 - mu) / mu
    if not lower_tail:
        return _st.beta.isf(p, a, b)
    return _st.beta.ppf(p, a, b)


def rBE(n, mu=0.5, sigma=0.2, rng=None):
    if np.any(np.asarray(mu) <= 0) or np.any(np.asarray(mu) >= 1):
        raise ValueError("mu must be between 0 and 1")
    if np.any(np.asarray(sigma) <= 0) or np.any(np.asarray(sigma) >= 1):
        raise ValueError("sigma must be between 0 and 1")
    if np.any(np.asarray(n) <= 0):
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng() if rng is None else rng
    n = int(np.ceil(n))
    p = rng.uniform(size=n)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    a = mu * (1 - sigma**2) / (sigma**2)
    b = a * (1 - mu) / mu
    r = _st.beta.ppf(p, a, b)
    return r
