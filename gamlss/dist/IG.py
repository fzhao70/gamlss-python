"""Inverse Gaussian distribution (IG). Port of gamlss.dist R/IG.R."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st
from scipy.optimize import brentq

from ..family import GamlssFamily, checklink


def IG(mu_link="log", sigma_link="log"):
    mstats = checklink("mu.link", "Inverse Gaussian", mu_link,
                       ("1/mu^2", "inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "Inverse Gaussian", sigma_link,
                       ("inverse", "log", "identity", "own"))
    return GamlssFamily(
        family=("IG", "Inverse Gaussian"),
        parameters={"mu": True, "sigma": True},
        nopar=2,
        type="Continuous",
        links={"mu": mstats, "sigma": dstats},
        derivatives={
            "dldm": lambda y, mu, sigma: (y - mu) / ((sigma**2) * (mu**3)),
            "d2ldm2": lambda mu, sigma: -1 / ((mu**3) * (sigma**2)),
            "dldd": lambda y, mu, sigma: (-1 / sigma)
            + ((y - mu) ** 2) / (y * (sigma**3) * (mu**2)),
            "d2ldd2": lambda sigma: -2 / (sigma**2),
            "d2ldmdd": lambda y: np.zeros(len(np.asarray(y))),
        },
        G_dev_incr=lambda y, mu, sigma: -2 * dIG(y, mu, sigma, log=True),
        rqres={"pfun": "pIG", "type": "Continuous"},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.full(
                len(y), np.std(y, ddof=1) / np.mean(y) ** 1.5
            ),
        },
        valid={
            "mu": lambda mu: bool(np.all(mu > 0)),
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
        },
        y_valid=lambda y: bool(np.all(y > 0)),
        mean=lambda mu, sigma: mu,
        variance=lambda mu, sigma: sigma**2 * mu**3,
    )


def dIG(x, mu=1, sigma=1, log=False):
    if np.any(np.asarray(mu) < 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be positive")
    x, mu, sigma = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_lik = (
            -0.5 * np.log(2 * np.pi)
            - np.log(sigma)
            - (3 / 2) * np.log(x)
            - ((x - mu) ** 2) / (2 * sigma**2 * (mu**2) * x)
        )
    fy = log_lik if log else np.exp(log_lik)
    return np.where(x <= 0, 0.0, fy)


def pIG(q, mu=1, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) < 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be positive")
    q, mu, sigma = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        cdf1 = _st.norm.cdf(((q / mu) - 1) / (sigma * np.sqrt(q)))
        lcdf2 = (2 / (mu * sigma**2)) + _st.norm.logcdf(
            (-((q / mu) + 1)) / (sigma * np.sqrt(q))
        )
        cdf = cdf1 + np.exp(lcdf2)
    if not lower_tail:
        cdf = 1 - cdf
    if log_p:
        cdf = np.log(cdf)
    return np.where(q <= 0, 0.0, cdf)


def qIG(p, mu=1, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1 - p
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be between 0 and 1")
    lp = max(p.size, np.asarray(mu).size, np.asarray(sigma).size)
    p = np.resize(p, lp)
    mu = np.resize(np.asarray(mu, float), lp)
    sigma = np.resize(np.asarray(sigma, float), lp)
    q = np.zeros(lp)
    for i in range(lp):
        def h(qq):
            return float(pIG(qq, mu=mu[i], sigma=sigma[i]) - p[i])

        if pIG(mu[i], mu=mu[i], sigma=sigma[i]) < p[i]:
            hi = mu[i] + sigma[i]
            j = 2
            while pIG(hi, mu=mu[i], sigma=sigma[i]) < p[i]:
                hi = mu[i] + j * sigma[i]
                j += 1
            interval = (mu[i], hi)
        else:
            interval = (np.finfo(float).tiny, mu[i])
        q[i] = brentq(h, *interval)
    return q


def rIG(n, mu=1, sigma=1, rng=None):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    rng = np.random.default_rng() if rng is None else rng
    p = rng.uniform(size=int(np.ceil(n)))
    return qIG(p, mu=mu, sigma=sigma)
