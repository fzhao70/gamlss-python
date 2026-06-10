"""Zero Adjusted Binomial distribution (ZABI). Port of gamlss.dist R/ZABI.R."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ..family import GamlssFamily, checklink
from .BI import _BI_dldm, dBI, _dBI0


def ZABI(mu_link="logit", sigma_link="logit"):
    mstats = checklink("mu.link", "ZABI", mu_link,
                       ("logit", "probit", "cloglog", "cauchit", "log", "own"))
    dstats = checklink("sigma.link", "ZABI", sigma_link,
                       ("logit", "probit", "cloglog", "cauchit", "log", "own"))

    def dldm(y, mu, sigma, bd):
        dldm = np.where(
            y == 0, 0,
            _BI_dldm(y, mu, bd)
            + (_dBI0(bd, mu) * _BI_dldm(0, mu, bd)) / (1 - _dBI0(bd, mu)),
        )
        return dldm

    def d2ldm2(y, mu, sigma, bd):
        dldm = np.where(
            y == 0, 0,
            _BI_dldm(y, mu, bd)
            + (_dBI0(bd, mu) * _BI_dldm(0, mu, bd)) / (1 - _dBI0(bd, mu)),
        )
        d2ldm2 = np.where(y == 0, 0, -(dldm) ** 2)
        return d2ldm2

    def dldd(y, mu, sigma, bd):
        dldd = np.where(y == 0, 1 / sigma, -1 / (1 - sigma))
        return dldd

    def d2ldd2(y, mu, sigma, bd):
        d2ldd2 = -1 / (sigma * (1 - sigma))
        return d2ldd2

    def d2ldmdd(y, mu, sigma, bd):
        d2ldmdd = 0
        return d2ldmdd

    return GamlssFamily(
        family=("ZABI", "Zero Adjusted Binomial"),
        parameters={"mu": True, "sigma": True},
        nopar=2,
        type="Discrete",
        links={"mu": mstats, "sigma": dstats},
        derivatives={
            "dldm": dldm,
            "d2ldm2": d2ldm2,
            "dldd": dldd,
            "d2ldd2": d2ldd2,
            "d2ldmdd": d2ldmdd,
        },
        G_dev_incr=lambda y, mu, sigma, bd: -2 * dZABI(y, bd, mu, sigma,
                                                       log=True),
        rqres={"pfun": "pZABI", "type": "Discrete", "ymin": 0},
        initial={
            "mu": lambda y: np.full(len(y), 0.5),
            "sigma": lambda y: np.full(len(y), 0.3),
        },
        valid={
            "mu": lambda mu: bool(np.all((mu > 0) & (mu < 1))),
            "sigma": lambda sigma: bool(np.all((sigma > 0) & (sigma < 1))),
        },
        y_valid=lambda y: bool(np.all(y >= 0)),
        mean=lambda bd, mu, sigma: (1 - sigma) * bd * mu
        / (1 - (1 - mu) ** bd),
        variance=lambda bd, mu, sigma: bd * mu * (1 - sigma)
        * (1 - mu + bd * mu) / (1 - (1 - mu) ** bd)
        - ((1 - sigma) * bd * mu / (1 - (1 - mu) ** bd)) ** 2,
    )


def dZABI(x, bd=1, mu=0.5, sigma=0.1, log=False):
    if np.any(np.asarray(mu) <= 0) or np.any(np.asarray(mu) >= 1):
        raise ValueError("mu must be between 0 and 1")
    if np.any(np.asarray(sigma) <= 0) or np.any(np.asarray(sigma) >= 1):
        raise ValueError("sigma must be between 0 and 1")
    x, sigma, mu, bd = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(sigma, float),
        np.asarray(mu, float), np.asarray(bd, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        logfy = np.where(
            x == 0,
            np.log(sigma),
            np.log(1 - sigma) + dBI(x, bd, mu, log=True)
            - np.log(1 - _dBI0(bd, mu)),
        )
    fy = np.exp(logfy) if log is False else logfy
    fy = np.where(x < 0, 0.0, fy)
    return fy


def pZABI(q, bd=1, mu=0.5, sigma=0.1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0) or np.any(np.asarray(mu) >= 1):
        raise ValueError("mu must be between 0 and 1")
    if np.any(np.asarray(sigma) <= 0) or np.any(np.asarray(sigma) >= 1):
        raise ValueError("sigma must be between 0 and 1")
    q, bd, mu, sigma = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(bd, float),
        np.asarray(mu, float), np.asarray(sigma, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        cdf1 = _st.binom.cdf(q, n=bd, p=mu)
        cdf2 = _st.binom.cdf(0, n=bd, p=mu)
        cdf3 = sigma + ((1 - sigma) * (cdf1 - cdf2) / (1 - cdf2))
        cdf = np.where(q == 0, sigma, cdf3)
        if not lower_tail:
            cdf = 1 - cdf
        if log_p:
            cdf = np.log(cdf)
    cdf = np.where(q < 0, 0.0, cdf)
    return cdf


def qZABI(p, bd=1, mu=0.5, sigma=0.1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0) or np.any(np.asarray(mu) >= 1):
        raise ValueError("mu must be between 0 and 1")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    p = np.asarray(p, float)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be between 0 and 1")
    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1 - p
    p, bd, mu, sigma = np.broadcast_arrays(
        p, np.asarray(bd, float), np.asarray(mu, float),
        np.asarray(sigma, float)
    )
    pnew = (p - sigma) / (1 - sigma) - 1e-10
    pnew2 = _st.binom.cdf(0, n=bd, p=mu) * (1 - pnew) + pnew
    with np.errstate(invalid="ignore"):
        q = np.where(pnew > 0, _st.binom.ppf(pnew2, n=bd, p=mu), 0)
    return q


def rZABI(n, bd=1, mu=0.5, sigma=0.1, rng=None):
    if np.any(np.asarray(mu) <= 0) or np.any(np.asarray(mu) >= 1):
        raise ValueError("mu must be between 0 and 1")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must greated than 0")
    if np.any(np.asarray(n) <= 0):
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng() if rng is None else rng
    n = int(np.ceil(n))
    p = rng.uniform(size=n)
    r = qZABI(p, mu=mu, sigma=sigma, bd=bd)
    return np.asarray(r).astype(int)
