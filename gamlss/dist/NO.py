"""Normal distribution (NO, NO2). Port of gamlss.dist R/NO.r."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ..family import GamlssFamily, checklink


def NO(mu_link="identity", sigma_link="log"):
    mstats = checklink("mu.link", "Normal", mu_link,
                       ("inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "Normal", sigma_link,
                       ("inverse", "log", "identity", "own"))
    return GamlssFamily(
        family=("NO", "Normal"),
        parameters={"mu": True, "sigma": True},
        nopar=2,
        type="Continuous",
        links={"mu": mstats, "sigma": dstats},
        derivatives={
            "dldm": lambda y, mu, sigma: (1 / sigma**2) * (y - mu),
            "d2ldm2": lambda sigma: -(1 / sigma**2),
            "dldd": lambda y, mu, sigma: ((y - mu) ** 2 - sigma**2) / (sigma**3),
            "d2ldd2": lambda sigma: -(2 / (sigma**2)),
            "d2ldmdd": lambda y: np.zeros(len(np.asarray(y))),
        },
        G_dev_incr=lambda y, mu, sigma: -2 * dNO(y, mu, sigma, log=True),
        rqres={"pfun": "pNO", "type": "Continuous"},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.full(len(y), np.std(y, ddof=1)),
        },
        valid={
            "mu": lambda mu: True,
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
        },
        y_valid=lambda y: True,
        mean=lambda mu, sigma: mu,
        variance=lambda mu, sigma: sigma**2,
    )


def dNO(x, mu=0, sigma=1, log=False):
    x, mu, sigma = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    return _st.norm.logpdf(x, mu, sigma) if log else _st.norm.pdf(x, mu, sigma)


def pNO(q, mu=0, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    q = np.asarray(q, float)
    if lower_tail:
        cdf = _st.norm.logcdf(q, mu, sigma) if log_p else _st.norm.cdf(q, mu, sigma)
    else:
        cdf = _st.norm.logsf(q, mu, sigma) if log_p else _st.norm.sf(q, mu, sigma)
    return cdf


def qNO(p, mu=0, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be between 0 and 1")
    if not lower_tail:
        p = 1 - p
    return _st.norm.ppf(p, mu, sigma)


def rNO(n, mu=0, sigma=1, rng=None):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    rng = np.random.default_rng() if rng is None else rng
    return rng.normal(mu, sigma, size=n)


# ------------------------------------------------------------------ NO2
# Normal with sigma as the variance
def NO2(mu_link="identity", sigma_link="log"):
    mstats = checklink("mu.link", "Normal", mu_link,
                       ("inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "Normal", sigma_link,
                       ("inverse", "log", "identity", "own"))
    return GamlssFamily(
        family=("NO2", "Normal with variance"),
        parameters={"mu": True, "sigma": True},
        nopar=2,
        type="Continuous",
        links={"mu": mstats, "sigma": dstats},
        derivatives={
            "dldm": lambda y, mu, sigma: (1 / sigma) * (y - mu),
            "d2ldm2": lambda sigma: -(1 / sigma),
            "dldd": lambda y, mu, sigma: 0.5 * ((y - mu) ** 2 - sigma) / (sigma**2),
            "d2ldd2": lambda sigma: -(1 / (2 * sigma**2)),
            "d2ldmdd": lambda y: np.zeros(len(np.asarray(y))),
        },
        G_dev_incr=lambda y, mu, sigma: -2 * dNO2(y, mu, sigma, log=True),
        rqres={"pfun": "pNO2", "type": "Continuous"},
        initial={
            "mu": lambda y: y + 0.00001,
            "sigma": lambda y: np.full(len(y), np.var(y, ddof=1)),
        },
        valid={
            "mu": lambda mu: True,
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
        },
        y_valid=lambda y: True,
        mean=lambda mu, sigma: mu,
        variance=lambda mu, sigma: sigma,
    )


def dNO2(x, mu=0, sigma=1, log=False):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    return dNO(x, mu, np.sqrt(np.asarray(sigma, float)), log=log)


def pNO2(q, mu=0, sigma=1, lower_tail=True, log_p=False):
    return pNO(q, mu, np.sqrt(np.asarray(sigma, float)),
               lower_tail=lower_tail, log_p=log_p)


def qNO2(p, mu=0, sigma=1, lower_tail=True, log_p=False):
    return qNO(p, mu, np.sqrt(np.asarray(sigma, float)),
               lower_tail=lower_tail, log_p=log_p)


def rNO2(n, mu=0, sigma=1, rng=None):
    return rNO(n, mu, np.sqrt(np.asarray(sigma, float)), rng=rng)
