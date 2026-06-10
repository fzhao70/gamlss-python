"""Logistic (LO) and Log Normal (LOGNO). Port of gamlss.dist
R/Logistic.R and R/logNO.R."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ..family import GamlssFamily, checklink


def LO(mu_link="identity", sigma_link="log"):
    mstats = checklink("mu.link", "Logistic", mu_link,
                       ("inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "Logistic", sigma_link,
                       ("inverse", "log", "identity", "own"))
    return GamlssFamily(
        family=("LO", "Logistic"),
        parameters={"mu": True, "sigma": True},
        nopar=2,
        type="Continuous",
        links={"mu": mstats, "sigma": dstats},
        derivatives={
            "dldm": lambda y, mu, sigma: (1 / sigma)
            * (np.exp((y - mu) / sigma) - 1) / (1 + np.exp((y - mu) / sigma)),
            "d2ldm2": lambda sigma: -1 / (3 * sigma**2),
            "dldd": lambda y, mu, sigma: -(1 / sigma) - (y - mu) / sigma**2
            + 2 * (((y - mu) / sigma**2) * np.exp((y - mu) / sigma))
            / (1 + np.exp((y - mu) / sigma)),
            "d2ldd2": lambda sigma: -(1 / (3 * sigma**2)) * (1 + (np.pi**2 / 3)),
            "d2ldmdd": lambda y: np.zeros(len(np.asarray(y))),
        },
        G_dev_incr=lambda y, mu, sigma: -2 * dLO(y, mu, sigma, log=True),
        rqres={"pfun": "pLO", "type": "Continuous"},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.full(
                len(y), (np.sqrt(3) * np.std(y, ddof=1)) / np.sqrt(np.pi)
            ),
        },
        valid={
            "mu": lambda mu: True,
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
        },
        y_valid=lambda y: True,
        mean=lambda mu, sigma: mu,
        variance=lambda mu, sigma: (np.pi**2 * sigma**2) / 3,
    )


def dLO(x, mu=0, sigma=1, log=False):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    x, mu, sigma = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    return (_st.logistic.logpdf(x, mu, sigma) if log
            else _st.logistic.pdf(x, mu, sigma))


def pLO(q, mu=0, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    q = np.asarray(q, float)
    if lower_tail:
        return (_st.logistic.logcdf(q, mu, sigma) if log_p
                else _st.logistic.cdf(q, mu, sigma))
    return (_st.logistic.logsf(q, mu, sigma) if log_p
            else _st.logistic.sf(q, mu, sigma))


def qLO(p, mu=0, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be between 0 and 1")
    if not lower_tail:
        p = 1 - p
    return _st.logistic.ppf(p, mu, sigma)


def rLO(n, mu=0, sigma=1, rng=None):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    rng = np.random.default_rng() if rng is None else rng
    return _st.logistic.ppf(rng.uniform(size=n), mu, sigma)


# ------------------------------------------------------------ LOGNO
def LOGNO(mu_link="identity", sigma_link="log"):
    mstats = checklink("mu.link", "Log Normal", mu_link,
                       ("inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "Log Normal", sigma_link,
                       ("inverse", "log", "identity", "own"))
    return GamlssFamily(
        family=("LOGNO", "Log Normal"),
        parameters={"mu": True, "sigma": True},
        nopar=2,
        type="Continuous",
        links={"mu": mstats, "sigma": dstats},
        derivatives={
            "dldm": lambda y, mu, sigma: (np.log(y) - mu) / sigma**2,
            "d2ldm2": lambda sigma: -1 / sigma**2,
            "dldd": lambda y, mu, sigma: (1 / (sigma**3))
            * ((np.log(y) - mu) ** 2 - sigma**2),
            "d2ldd2": lambda sigma: -2 / sigma**2,
            "d2ldmdd": lambda y: np.zeros(len(np.asarray(y))),
        },
        G_dev_incr=lambda y, mu, sigma: -2 * dLOGNO(y, mu=mu, sigma=sigma,
                                                    log=True),
        rqres={"pfun": "pLOGNO", "type": "Continuous"},
        initial={
            "mu": lambda y: (np.log(y) + np.mean(np.log(y))) / 2,
            "sigma": lambda y: np.full(len(y), np.std(np.log(y), ddof=1)),
        },
        valid={
            "mu": lambda mu: bool(np.all(mu > 0)),
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
        },
        y_valid=lambda y: bool(np.all(y > 0)),
        mean=lambda mu, sigma: np.exp(mu + sigma**2 / 2),
        variance=lambda mu, sigma: np.exp(2 * mu + sigma**2)
        * (np.exp(sigma**2) - 1),
    )


def dLOGNO(x, mu=0, sigma=1, log=False):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    x, mu, sigma = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        loglik = (
            -np.log(x) - np.log(sigma) - 0.5 * np.log(2 * np.pi)
            - (np.log(x) - mu) ** 2 / (2 * sigma**2)
        )
    fy = loglik if log else np.exp(loglik)
    return np.where(x <= 0, 0.0, fy)


def pLOGNO(q, mu=0, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    q = np.asarray(q, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (np.log(np.maximum(q, 0)) - mu) / sigma
    z = np.where(q <= 0, -np.inf, z)
    if lower_tail:
        return _st.norm.logcdf(z) if log_p else _st.norm.cdf(z)
    return _st.norm.logsf(z) if log_p else _st.norm.sf(z)


def qLOGNO(p, mu=0, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be between 0 and 1")
    if not lower_tail:
        p = 1 - p
    return np.exp(mu + sigma * _st.norm.ppf(p))


def rLOGNO(n, mu=0, sigma=1, rng=None):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    rng = np.random.default_rng() if rng is None else rng
    return np.exp(rng.normal(mu, sigma, size=n))
