"""Weibull distributions WEI and WEI3. Port of gamlss.dist R/WEI.R,
R/WEI3.R."""

from __future__ import annotations

import numpy as np
from scipy import special as _sp

from ..family import GamlssFamily, checklink


def WEI(mu_link="log", sigma_link="log"):
    mstats = checklink("mu.link", "Weibull", mu_link,
                       ("inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "Weibull", sigma_link,
                       ("inverse", "log", "identity", "own"))
    return GamlssFamily(
        family=("WEI", "Weibull"),
        parameters={"mu": True, "sigma": True},
        nopar=2,
        type="Continuous",
        links={"mu": mstats, "sigma": dstats},
        derivatives={
            "dldm": lambda y, mu, sigma: ((y / mu) ** sigma - 1)
            * (sigma / mu),
            "d2ldm2": lambda mu, sigma: -(sigma**2) / mu**2,
            "dldd": lambda y, mu, sigma: 1 / sigma
            - np.log(y / mu) * ((y / mu) ** sigma - 1),
            "d2ldd2": lambda sigma: -1.82368 / sigma**2,
            "d2ldmdd": lambda mu: 0.422784 / mu,
        },
        G_dev_incr=lambda y, mu, sigma: -2 * dWEI(y, mu, sigma, log=True),
        rqres={"pfun": "pWEI", "type": "Continuous"},
        initial={
            "mu": lambda y: np.exp(
                np.log(y) + 0.5772 / (1.283 / np.sqrt(np.var(np.log(y),
                                                             ddof=1)))
            ),
            "sigma": lambda y: np.full(
                len(y), 1.283 / np.sqrt(np.var(np.log(y), ddof=1))
            ),
        },
        valid={
            "mu": lambda mu: bool(np.all(mu > 0)),
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
        },
        y_valid=lambda y: bool(np.all(y > 0)),
        mean=lambda mu, sigma: mu * _sp.gamma(1 / sigma + 1),
        variance=lambda mu, sigma: mu**2
        * (_sp.gamma(2 / sigma + 1) - _sp.gamma(1 / sigma + 1) ** 2),
    )


def dWEI(x, mu=1, sigma=1, log=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    x, mu, sigma = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        loglik = (np.log(sigma) - np.log(mu)
                  + (sigma - 1) * (np.log(x) - np.log(mu))
                  - (x / mu) ** sigma)
    fy = loglik if log else np.exp(loglik)
    return np.where(x <= 0, 0.0, fy)


def pWEI(q, mu=1, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    q, mu, sigma = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    z = (np.maximum(q, 0) / mu) ** sigma
    cdf = -np.expm1(-z)
    if not lower_tail:
        cdf = 1 - cdf
    if log_p:
        cdf = np.log(cdf)
    return cdf


def qWEI(p, mu=1, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be between 0 and 1")
    if not lower_tail:
        p = 1 - p
    return mu * (-np.log1p(-p)) ** (1 / np.asarray(sigma, float))


def rWEI(n, mu=1, sigma=1, rng=None):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    rng = np.random.default_rng() if rng is None else rng
    return qWEI(rng.uniform(size=int(np.ceil(n))), mu=mu, sigma=sigma)


# -------------------------------------------------------------- WEI3
def WEI3(mu_link="log", sigma_link="log"):
    mstats = checklink("mu.link", "WEI3bull", mu_link,
                       ("inverse", "log", "identity"))
    dstats = checklink("sigma.link", "WEI3bull", sigma_link,
                       ("inverse", "log", "identity"))
    return GamlssFamily(
        family=("WEI3", "Weibull type 3"),
        parameters={"mu": True, "sigma": True},
        nopar=2,
        type="Continuous",
        links={"mu": mstats, "sigma": dstats},
        derivatives={
            "dldm": lambda y, mu, sigma: (
                (y * _sp.gamma((1 / sigma) + 1) / mu) ** sigma - 1
            ) * (sigma / mu),
            "d2ldm2": lambda sigma, mu: -(sigma**2) / mu**2,
            "dldd": lambda y, mu, sigma: 1 / sigma
            - np.log(y * _sp.gamma((1 / sigma) + 1) / mu)
            * ((y * _sp.gamma((1 / sigma) + 1) / mu) ** sigma - 1)
            + (_sp.digamma((1 / sigma) + 1))
            * ((y * _sp.gamma((1 / sigma) + 1) / mu) ** sigma - 1) / sigma,
            "d2ldd2": lambda sigma: -(
                1.644934 + (0.422784 - _sp.digamma((1 / sigma) + 1)) ** 2
            ) / (sigma * sigma),
            "d2ldmdd": lambda mu, sigma: (
                0.422784 - _sp.digamma((1 / sigma) + 1)
            ) / mu,
        },
        G_dev_incr=lambda y, mu, sigma: -2 * dWEI3(y, mu, sigma, log=True),
        rqres={"pfun": "pWEI3", "type": "Continuous"},
        initial={
            "mu": lambda y: y + 0.01,
            "sigma": lambda y: np.full(
                len(y), 1.283 / np.sqrt(np.var(np.log(y), ddof=1))
            ),
        },
        valid={
            "mu": lambda mu: bool(np.all(mu > 0)),
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
        },
        y_valid=lambda y: bool(np.all(y > 0)),
        mean=lambda mu, sigma: mu,
        variance=lambda mu, sigma: (mu / _sp.gamma(1 / sigma + 1)) ** 2
        * (_sp.gamma(2 / sigma + 1) - _sp.gamma(1 / sigma + 1) ** 2),
    )


def dWEI3(x, mu=1, sigma=1, log=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    mu2 = np.asarray(mu, float) / _sp.gamma((1 / np.asarray(sigma, float)) + 1)
    return dWEI(x, mu=mu2, sigma=sigma, log=log)


def pWEI3(q, mu=1, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    mu2 = np.asarray(mu, float) / _sp.gamma((1 / np.asarray(sigma, float)) + 1)
    return pWEI(q, mu=mu2, sigma=sigma, lower_tail=lower_tail, log_p=log_p)


def qWEI3(p, mu=1, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    mu2 = np.asarray(mu, float) / _sp.gamma((1 / np.asarray(sigma, float)) + 1)
    return qWEI(p, mu=mu2, sigma=sigma, lower_tail=lower_tail, log_p=log_p)


def rWEI3(n, mu=1, sigma=1, rng=None):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    mu2 = np.asarray(mu, float) / _sp.gamma((1 / np.asarray(sigma, float)) + 1)
    return rWEI(n, mu=mu2, sigma=sigma, rng=rng)
