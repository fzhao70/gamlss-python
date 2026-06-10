"""Inverse Gamma distribution (IGAMMA). Port of gamlss.dist R/IGA.R.

The R source file is IGA.R but the family it defines is IGAMMA
(``c("IGAMMA", "Inverse Gamma")``) with dIGAMMA/pIGAMMA/qIGAMMA/rIGAMMA.
IGAMMA(alpha, mu/(alpha+1)) where alpha = 1/sigma^2; x>0, alpha>0, mu>0.
"""

from __future__ import annotations

import numpy as np
from scipy import special as _sp
from scipy import stats as _st

from ..family import GamlssFamily, checklink
from .GA import qGA


def IGAMMA(mu_link="log", sigma_link="log"):
    mstats = checklink("mu.link", "Inverse Gamma", mu_link,
                       ("inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "Inverse Gamma", sigma_link,
                       ("inverse", "log", "identity", "own"))

    def dldm(y, mu, sigma):
        alpha = 1 / (sigma**2)
        dldm = (alpha / mu) - ((alpha + 1) / y)
        return dldm

    def d2ldm2(y, mu, sigma):
        d2ldm2 = -(1 / (sigma**2 * mu**2))
        return d2ldm2

    def dldd(y, mu, sigma):
        alpha = 1 / (sigma**2)
        dldd = (-2 / (sigma**3)) * (
            np.log(mu) + (alpha / (alpha + 1)) + np.log(alpha + 1)
            - _sp.digamma(alpha) - np.log(y) - (mu / y)
        )
        return dldd

    def d2ldd2(y, mu, sigma):
        d2ldd2 = -((4 * (-((sigma**2 * (1 + 2 * sigma**2))
                          / ((1 + sigma**2) ** 2))
                         + _sp.polygamma(1, 1 / sigma**2))) / (sigma**6))
        return d2ldd2

    def d2ldmdd(y, mu, sigma):
        d2ldmdd = -(2 / (mu * sigma + mu * sigma**3))
        return d2ldmdd

    return GamlssFamily(
        family=("IGAMMA", "Inverse Gamma"),
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
        G_dev_incr=lambda y, mu, sigma: -2 * dIGAMMA(y, mu, sigma, log=True),
        rqres={"pfun": "pIGAMMA", "type": "Continuous"},
        initial={
            "mu": lambda y: np.full(len(y), np.mean(y)),
            "sigma": lambda y: np.full(
                len(y), ((np.mean(y) ** 2) / np.var(y, ddof=1)) + 2
            ),
        },
        valid={
            "mu": lambda mu: bool(np.all(mu > 0)),
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
        },
        y_valid=lambda y: True,
        mean=lambda mu, sigma: np.where(
            sigma**2 < 1, ((1 + sigma**2) * mu) / (1 - sigma**2), np.inf
        ),
        variance=lambda mu, sigma: np.where(
            sigma**2 < 0.5,
            ((1 + sigma**2) ** 2 * mu**2 * sigma**2)
            / ((1 - sigma**2) ** 2 * (1 - 2 * sigma**2)),
            np.inf,
        ),
    )


def dIGAMMA(x, mu=1, sigma=0.5, log=False):
    if np.any(np.asarray(mu) < 0):
        raise ValueError("mu must be greater than 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    x, mu, sigma = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = 1 / (sigma**2)
        lfy = (alpha * np.log(mu) + alpha * np.log(alpha + 1)
               - _sp.gammaln(alpha) - (alpha + 1) * np.log(x)
               - ((mu * (alpha + 1)) / x))
    fy = lfy if log else np.exp(lfy)
    return np.where(x <= 0, 0.0, fy)


def pIGAMMA(q, mu=1, sigma=0.5, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    q, mu, sigma = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = 1 / (sigma**2)
        lcdf = _st.gamma.logsf(((mu * (alpha + 1)) / q), alpha)
    cdf = np.exp(lcdf) if not log_p else lcdf
    if not lower_tail:
        cdf = 1 - cdf
    return np.where(q <= 0, 0.0, cdf)


def qIGAMMA(p, mu=1, sigma=0.5, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) < 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be positive")
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1 - p
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be between 0 and 1")
    p, mu, sigma = np.broadcast_arrays(
        p, np.asarray(mu, float), np.asarray(sigma, float)
    )
    nu = -1
    p = p if nu > 0 else 1 - p
    z = qGA(p, mu=1, sigma=sigma * np.abs(nu))
    mu2 = mu * (1 + sigma**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = mu2 * z ** (1 / nu)
    return y


def rIGAMMA(n, mu=1, sigma=0.5, rng=None):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    if np.any(np.asarray(n) <= 0):
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng() if rng is None else rng
    p = rng.uniform(size=int(np.ceil(n)))
    return qIGAMMA(p, mu=mu, sigma=sigma)
