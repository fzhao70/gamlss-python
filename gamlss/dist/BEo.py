"""Beta original distribution (BEo). Port of gamlss.dist R/BEo.R.

The beta distribution in its original parameterisation, with mu and
sigma the two beta shape parameters (shape1 and shape2).
"""

from __future__ import annotations

import numpy as np
from scipy import special as _sp
from scipy import stats as _st

from ..family import GamlssFamily, checklink


def BEo(mu_link="log", sigma_link="log"):
    mstats = checklink("mu.link", "BEo", mu_link,
                       ("inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "BEo", sigma_link,
                       ("inverse", "log", "identity", "own"))
    return GamlssFamily(
        family=("BEo", "Beta original"),
        parameters={"mu": True, "sigma": True},
        nopar=2,
        type="Continuous",
        links={"mu": mstats, "sigma": dstats},
        derivatives={
            "dldm": lambda y, mu, sigma: -_sp.digamma(mu)
            + _sp.digamma(mu + sigma) + np.log(y),
            "d2ldm2": lambda mu, sigma: -_sp.polygamma(1, mu)
            + _sp.polygamma(1, mu + sigma),
            "dldd": lambda y, mu, sigma: -_sp.digamma(sigma)
            + _sp.digamma(mu + sigma) + np.log(1 - y),
            "d2ldd2": lambda mu, sigma: -_sp.polygamma(1, sigma)
            + _sp.polygamma(1, mu + sigma),
            "d2ldmdd": lambda mu, sigma: _sp.polygamma(1, mu + sigma),
        },
        G_dev_incr=lambda y, mu, sigma: -2 * dBEo(y, mu, sigma, log=True),
        rqres={"pfun": "pBEo", "type": "Continuous"},
        initial={
            "mu": lambda y: np.full(len(y), 2.0),
            "sigma": lambda y: np.full(len(y), 2.0),
        },
        valid={
            "mu": lambda mu: bool(np.all((mu > 0) & (mu < 1))),
            "sigma": lambda sigma: bool(np.all((sigma > 0) & (sigma < 1))),
        },
        y_valid=lambda y: bool(np.all((y > 0) & (y < 1))),
        mean=lambda mu, sigma: mu / (mu + sigma),
        variance=lambda mu, sigma: (mu * sigma)
        / ((mu + sigma) ** 2 * (mu + sigma + 1)),
    )


def dBEo(x, mu=0.5, sigma=0.2, log=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    x, mu, sigma = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        fy = _st.beta.logpdf(x, mu, sigma) if log else _st.beta.pdf(x, mu,
                                                                    sigma)
    return fy


def pBEo(q, mu=0.5, sigma=0.2, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    q = np.asarray(q, float)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    if lower_tail:
        cdf = (_st.beta.logcdf(q, mu, sigma) if log_p
               else _st.beta.cdf(q, mu, sigma))
    else:
        cdf = (_st.beta.logsf(q, mu, sigma) if log_p
               else _st.beta.sf(q, mu, sigma))
    return cdf


def qBEo(p, mu=0.5, sigma=0.2, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    p = np.asarray(p, float)
    if np.any(p <= 0) or np.any(p >= 1):
        raise ValueError("p must be between 0 and 1")
    if log_p:
        p = np.exp(p)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    if not lower_tail:
        return _st.beta.isf(p, mu, sigma)
    return _st.beta.ppf(p, mu, sigma)


def rBEo(n, mu=0.5, sigma=0.2, rng=None):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(n) <= 0):
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng() if rng is None else rng
    n = int(np.ceil(n))
    p = rng.uniform(size=n)
    r = _st.beta.ppf(p, np.asarray(mu, float), np.asarray(sigma, float))
    return r
