"""Zero Inflated Poisson distribution (ZIP). Port of gamlss.dist R/ZIP.R."""

from __future__ import annotations

import numpy as np
from scipy import special as _sp
from scipy import stats as _st

from ..family import GamlssFamily, checklink


def _zip_dldm(y, mu, sigma):
    with np.errstate(over="ignore"):
        dldm0 = -(1 - sigma) * (((1 - sigma) + sigma * np.exp(mu)) ** (-1))
        dldm = np.where(y == 0, dldm0, (y / mu) - 1)
    return dldm


def _zip_dldd(y, mu, sigma):
    dldd0 = (1 - np.exp(-mu)) * ((sigma + (1 - sigma) * np.exp(-mu)) ** (-1))
    dldd = np.where(y == 0, dldd0, -1 / (1 - sigma))
    return dldd


def ZIP(mu_link="log", sigma_link="logit"):
    mstats = checklink("mu.link", "ZIP", mu_link,
                       ("1/mu^2", "log", "identity"))
    dstats = checklink("sigma.link", "ZIP", sigma_link,
                       ("logit", "probit", "cloglog", "cauchit", "log",
                        "own"))

    def d2ldm2(y, mu, sigma):
        dldm = _zip_dldm(y, mu, sigma)
        d2ldm2 = -dldm * dldm
        d2ldm2 = np.where(d2ldm2 < -1e-15, d2ldm2, -1e-15)
        return d2ldm2

    def d2ldd2(y, mu, sigma):
        dldd = _zip_dldd(y, mu, sigma)
        d2ldd2 = -dldd * dldd
        d2ldd2 = np.where(d2ldd2 < -1e-15, d2ldd2, -1e-15)
        return d2ldd2

    def d2ldmdd(y, mu, sigma):
        dldm = _zip_dldm(y, mu, sigma)
        dldd = _zip_dldd(y, mu, sigma)
        d2ldmdd = -dldm * dldd
        return d2ldmdd

    return GamlssFamily(
        family=("ZIP", "Poisson Zero Inflated"),
        parameters={"mu": True, "sigma": True},
        nopar=2,
        type="Discrete",
        links={"mu": mstats, "sigma": dstats},
        derivatives={
            "dldm": _zip_dldm,
            "d2ldm2": d2ldm2,
            "dldd": _zip_dldd,
            "d2ldd2": d2ldd2,
            "d2ldmdd": d2ldmdd,
        },
        G_dev_incr=lambda y, mu, sigma: -2 * dZIP(y, mu, sigma, log=True),
        rqres={"pfun": "pZIP", "type": "Discrete", "ymin": 0},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.full(len(y), 0.1),
        },
        valid={
            "mu": lambda mu: bool(np.all(mu > 0)),
            "sigma": lambda sigma: bool(np.all((sigma > 0) & (sigma < 1))),
        },
        y_valid=lambda y: bool(np.all(y >= 0)),
        mean=lambda mu, sigma: (1 - sigma) * mu,
        variance=lambda mu, sigma: mu * (1 - sigma) * (1 + mu * sigma),
    )


def dZIP(x, mu=5, sigma=0.1, log=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    if np.any(np.asarray(sigma) <= 0) or np.any(np.asarray(sigma) >= 1):
        raise ValueError("sigma must be between 0 and 1")
    x, mu, sigma = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        logfy = np.where(
            x == 0,
            np.log(sigma + (1 - sigma) * np.exp(-mu)),
            np.log(1 - sigma) - mu + x * np.log(mu) - _sp.gammaln(x + 1),
        )
    fy = logfy if log else np.exp(logfy)
    fy = np.where(x < 0, 0, fy)
    return fy


def pZIP(q, mu=5, sigma=0.1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    if np.any(np.asarray(sigma) <= 0) or np.any(np.asarray(sigma) >= 1):
        raise ValueError("sigma must be between 0 and 1")
    q, mu, sigma = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    cdf = _st.poisson.cdf(np.floor(q), mu)
    cdf = sigma + (1 - sigma) * cdf
    if not lower_tail:
        cdf = 1 - cdf
    if log_p:
        with np.errstate(divide="ignore", invalid="ignore"):
            cdf = np.log(cdf)
    cdf = np.where(q < 0, 0, cdf)
    return cdf


def qZIP(p, mu=5, sigma=0.1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if np.any(p <= 0) or np.any(p >= 1):
        raise ValueError("p must be between 0 and 1")
    if not lower_tail:
        p = 1 - p
    p, mu, sigma = np.broadcast_arrays(
        p, np.asarray(mu, float), np.asarray(sigma, float)
    )
    pnew = ((p - sigma) / (1 - sigma)) - (1e-7)  # added Monday, March 15, 2010
    pnew = np.where(pnew > 0, pnew, 0)
    q = _st.poisson.ppf(pnew, mu)
    # R's qpois returns 0 at p == 0; scipy's ppf returns -1
    q = np.where(pnew == 0, 0.0, q)
    return q


def rZIP(n, mu=5, sigma=0.1, rng=None):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must greated than 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must greated than 0")
    if np.any(np.asarray(n) <= 0):
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng() if rng is None else rng
    n = int(np.ceil(n))
    p = rng.uniform(size=n)
    r = qZIP(p, mu=mu, sigma=sigma)
    return r.astype(int)
