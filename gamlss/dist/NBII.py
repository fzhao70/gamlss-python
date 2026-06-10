"""Negative Binomial type II distribution (NBII). Port of gamlss.dist R/NBII.R."""

from __future__ import annotations

import numpy as np
from scipy import special as _sp
from scipy import stats as _st

from ..family import GamlssFamily, checklink


def _nbii_dldm(y, mu, sigma):
    return (1 / sigma) * (_sp.digamma(y + (mu / sigma))
                          - _sp.digamma(mu / sigma) - np.log(1 + sigma))


def _nbii_dldd(y, mu, sigma):
    return (-(mu / (sigma**2)) * (_sp.digamma(y + (mu / sigma))
                                  - _sp.digamma(mu / sigma)
                                  - np.log(1 + sigma))
            + (y - mu) / (sigma * (1 + sigma)))


def NBII(mu_link="log", sigma_link="log"):
    mstats = checklink("mu.link", "Negative Binomial type II", mu_link,
                       ("inverse", "log", "identity", "sqrt"))
    dstats = checklink("sigma.link", "Negative Binomial type II", sigma_link,
                       ("inverse", "log", "identity", "sqrt"))

    def d2ldm2(y, mu, sigma):
        dldm = _nbii_dldm(y, mu, sigma)
        d2ldm2 = ((1 / sigma) ** 2) * (_sp.polygamma(1, y + (mu / sigma))
                                       - _sp.polygamma(1, mu / sigma))
        # MS Thursday, September 29, 2005
        d2ldm2 = -dldm**2 if np.any(d2ldm2 >= 0) else d2ldm2
        d2ldm2 = np.where(d2ldm2 < -1e-15, d2ldm2, -1e-15)
        return d2ldm2

    def d2ldd2(y, mu, sigma):
        dldd = _nbii_dldd(y, mu, sigma)
        d2ldd2 = -dldd**2
        d2ldd2 = np.where(d2ldd2 < -1e-15, d2ldd2, -1e-15)
        return d2ldd2

    def d2ldmdd(y, mu, sigma):
        dldm = _nbii_dldm(y, mu, sigma)
        dldd = _nbii_dldd(y, mu, sigma)
        d2ldmdd = -dldm * dldd
        return d2ldmdd

    return GamlssFamily(
        family=("NBII", "Negative Binomial type II"),
        parameters={"mu": True, "sigma": True},
        nopar=2,
        type="Discrete",
        links={"mu": mstats, "sigma": dstats},
        derivatives={
            "dldm": _nbii_dldm,
            "d2ldm2": d2ldm2,
            "dldd": _nbii_dldd,
            "d2ldd2": d2ldd2,
            "d2ldmdd": d2ldmdd,
        },
        G_dev_incr=lambda y, mu, sigma: -2 * dNBII(y, mu=mu, sigma=sigma,
                                                   log=True),
        rqres={"pfun": "pNBII", "type": "Discrete", "ymin": 0},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.full(
                len(y), max((np.var(y, ddof=1) / np.mean(y)) - 1, 0.1)
            ),
        },
        valid={
            "mu": lambda mu: bool(np.all(mu > 0)),
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
        },
        y_valid=lambda y: bool(np.all(y >= 0)),
        mean=lambda mu, sigma: mu,
        variance=lambda mu, sigma: mu + sigma * mu,
    )


def dNBII(x, mu=1, sigma=1, log=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    sigma_vector = np.asarray(sigma).size > 1
    x, mu, sigma = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        size = mu / sigma
        prob = size / (size + mu)
        nb = (_st.nbinom.logpmf(x, n=size, p=prob) if log
              else _st.nbinom.pmf(x, n=size, p=prob))
        po = _st.poisson.logpmf(x, mu) if log else _st.poisson.pmf(x, mu)
        if sigma_vector:
            fy = np.where(sigma > 0.0001, nb, po)
        else:
            fy = po if sigma.flat[0] < 0.0001 else nb
    fy = np.where(x < 0, 0, fy)
    return fy


def pNBII(q, mu=1, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    sigma_vector = np.asarray(sigma).size > 1
    q, mu, sigma = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    qf = np.floor(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        size = mu / sigma
        prob = size / (size + mu)
        if lower_tail:
            nb = (_st.nbinom.logcdf(qf, n=size, p=prob) if log_p
                  else _st.nbinom.cdf(qf, n=size, p=prob))
            po = (_st.poisson.logcdf(qf, mu) if log_p
                  else _st.poisson.cdf(qf, mu))
        else:
            nb = (_st.nbinom.logsf(qf, n=size, p=prob) if log_p
                  else _st.nbinom.sf(qf, n=size, p=prob))
            po = (_st.poisson.logsf(qf, mu) if log_p
                  else _st.poisson.sf(qf, mu))
        if sigma_vector:
            cdf = np.where(sigma > 0.0001, nb, po)
        else:
            cdf = po if sigma.flat[0] < 0.0001 else nb
    cdf = np.where(q < 0, 0, cdf)
    return cdf


def qNBII(p, mu=1, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be between 0 and 1")
    if not lower_tail:
        p = 1 - p
    sigma_vector = np.asarray(sigma).size > 1
    p, mu, sigma = np.broadcast_arrays(
        p, np.asarray(mu, float), np.asarray(sigma, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        size = mu / sigma
        prob = size / (size + mu)
        nb = _st.nbinom.ppf(p, n=size, p=prob)
        po = _st.poisson.ppf(p, mu)
        if sigma_vector:
            q = np.where(sigma > 0.0001, nb, po)
        else:
            q = po if sigma.flat[0] < 0.0001 else nb
    # R's qnbinom/qpois return 0 at p == 0; scipy's ppf returns -1
    q = np.where(p == 0, 0.0, q)
    return q


def rNBII(n, mu=1, sigma=1, rng=None):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    if np.any(np.asarray(n) <= 0):
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng() if rng is None else rng
    n = int(np.ceil(n))
    p = rng.uniform(size=n)
    r = qNBII(p, mu=mu, sigma=sigma)
    return r.astype(int)
