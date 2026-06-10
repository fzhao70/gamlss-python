"""Gamma distribution (GA). Port of gamlss.dist R/GA.R.

Parameterised so that E(y) = mu and Var(y) = sigma^2 * mu^2.
"""

from __future__ import annotations

import numpy as np
from scipy import special as _sp
from scipy import stats as _st

from ..family import GamlssFamily, checklink


def GA(mu_link="log", sigma_link="log"):
    mstats = checklink("mu.link", "Gamma", mu_link,
                       ("inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "Gamma", sigma_link,
                       ("inverse", "log", "identity", "own"))
    return GamlssFamily(
        family=("GA", "Gamma"),
        parameters={"mu": True, "sigma": True},
        nopar=2,
        type="Continuous",
        links={"mu": mstats, "sigma": dstats},
        derivatives={
            "dldm": lambda y, mu, sigma: (y - mu) / ((sigma**2) * (mu**2)),
            "d2ldm2": lambda mu, sigma: -1 / ((sigma**2) * (mu**2)),
            "dldd": lambda y, mu, sigma: (2 / sigma**3)
            * ((y / mu) - np.log(y) + np.log(mu) + np.log(sigma**2) - 1
               + _sp.digamma(1 / (sigma**2))),
            "d2ldd2": lambda sigma: (4 / sigma**4)
            - (4 / sigma**6) * _sp.polygamma(1, 1 / sigma**2),
            "d2ldmdd": lambda y: np.zeros(len(np.asarray(y))),
        },
        G_dev_incr=lambda y, mu, sigma: -2 * dGA(y, mu, sigma, log=True),
        rqres={"pfun": "pGA", "type": "Continuous"},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.ones(len(y)),
        },
        valid={
            "mu": lambda mu: bool(np.all(mu > 0)),
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
        },
        y_valid=lambda y: bool(np.all(y > 0)),
        mean=lambda mu, sigma: mu,
        variance=lambda mu, sigma: sigma**2 * mu**2,
    )


def dGA(x, mu=1, sigma=1, log=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    x, mu, sigma = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_lik = (
            (1 / sigma**2) * np.log(x / (mu * sigma**2))
            - x / (mu * sigma**2)
            - np.log(x)
            - _sp.gammaln(1 / sigma**2)
        )
    fy = log_lik if log else np.exp(log_lik)
    return np.where(x <= 0, 0.0, fy)


def pGA(q, mu=1, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    q = np.asarray(q, float)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    shape = 1 / sigma**2
    scale = mu * sigma**2
    if lower_tail:
        cdf = (_st.gamma.logcdf(q, shape, scale=scale) if log_p
               else _st.gamma.cdf(q, shape, scale=scale))
    else:
        cdf = (_st.gamma.logsf(q, shape, scale=scale) if log_p
               else _st.gamma.sf(q, shape, scale=scale))
    return cdf


def qGA(p, mu=1, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be between 0 and 1")
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    shape = 1 / sigma**2
    scale = mu * sigma**2
    if not lower_tail:
        return _st.gamma.isf(p, shape, scale=scale)
    return _st.gamma.ppf(p, shape, scale=scale)


def rGA(n, mu=1, sigma=1, rng=None):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    rng = np.random.default_rng() if rng is None else rng
    p = rng.uniform(size=int(np.ceil(n)))
    return qGA(p, mu=mu, sigma=sigma)
