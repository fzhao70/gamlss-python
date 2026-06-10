"""Negative Binomial type I distribution (NBI). Port of gamlss.dist R/NBI.R."""

from __future__ import annotations

import numpy as np
from scipy import special as _sp
from scipy import stats as _st

from ..family import GamlssFamily, checklink
from .PO import dPO


def _nbi_dldm(y, mu, sigma):
    """NBI dldm; also used by ZINBI/ZANBI via R's NBI()$dldm."""
    return (y - mu) / (mu * (1 + mu * sigma))


def _nbi_dldd(y, mu, sigma):
    """NBI dldd; also used by ZINBI/ZANBI via R's NBI()$dldd."""
    return -((1 / sigma) ** 2) * (
        _sp.digamma(y + (1 / sigma)) - _sp.digamma(1 / sigma)
        - np.log(1 + mu * sigma) - (y - mu) * sigma / (1 + mu * sigma)
    )


def _dNBI_at0(mu, sigma, log=False):
    """R's dNBI(0, mu, sigma) with vector mu/sigma.

    In R the final ``ifelse(x < 0, 0, fy)`` inside dNBI truncates the
    result to length(x) == 1, so the call returns the value for the
    FIRST observation only, which callers (ZINBI/ZANBI derivatives,
    dZANBI) then recycle over all observations."""
    mu, sigma = np.broadcast_arrays(
        np.asarray(mu, float), np.asarray(sigma, float)
    )
    fy = dNBI(np.zeros(mu.shape), mu=mu, sigma=sigma, log=log)
    return np.asarray(fy).flat[0]


def _pNBI_at0(mu, sigma):
    """R's pNBI(0, mu, sigma) with vector mu/sigma: as in _dNBI_at0,
    the final ``ifelse(q < 0, 0, cdf)`` truncates to length(q) == 1."""
    mu, sigma = np.broadcast_arrays(
        np.asarray(mu, float), np.asarray(sigma, float)
    )
    cdf = pNBI(np.zeros(mu.shape), mu=mu, sigma=sigma)
    return np.asarray(cdf).flat[0]


def NBI(mu_link="log", sigma_link="log"):
    mstats = checklink("mu.link", "Negative Binomial type I", mu_link,
                       ("inverse", "log", "identity", "sqrt"))
    dstats = checklink("sigma.link", "Negative Binomial type I", sigma_link,
                       ("inverse", "log", "identity", "sqrt"))

    def d2ldd2(y, mu, sigma):
        dldd = _nbi_dldd(y, mu, sigma)
        d2ldd2 = -dldd**2
        d2ldd2 = np.where(d2ldd2 < -1e-15, d2ldd2, -1e-15)
        return d2ldd2

    return GamlssFamily(
        family=("NBI", "Negative Binomial type I"),
        parameters={"mu": True, "sigma": True},
        nopar=2,
        type="Discrete",
        links={"mu": mstats, "sigma": dstats},
        derivatives={
            "dldm": _nbi_dldm,
            "d2ldm2": lambda mu, sigma: -1 / (mu * (1 + mu * sigma)),
            "dldd": _nbi_dldd,
            "d2ldd2": d2ldd2,
            "d2ldmdd": lambda y: np.zeros(len(np.asarray(y))),
        },
        G_dev_incr=lambda y, mu, sigma: -2 * dNBI(y, mu=mu, sigma=sigma,
                                                  log=True),
        rqres={"pfun": "pNBI", "type": "Discrete", "ymin": 0},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.full(
                len(y),
                max((np.var(y, ddof=1) - np.mean(y)) / (np.mean(y) ** 2), 0.1),
            ),
        },
        valid={
            "mu": lambda mu: bool(np.all(mu > 0)),
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
        },
        y_valid=lambda y: bool(np.all(y >= 0)),
        mean=lambda mu, sigma: mu,
        variance=lambda mu, sigma: mu + sigma * mu**2,
    )


def dNBI(x, mu=1, sigma=1, log=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    sigma_vector = np.asarray(sigma).size > 1
    x, mu, sigma = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        size = 1 / sigma
        prob = size / (size + mu)
        nb = (_st.nbinom.logpmf(x, n=size, p=prob) if log
              else _st.nbinom.pmf(x, n=size, p=prob))
        if sigma_vector:
            fy = np.where(sigma > 0.0001, nb, dPO(x, mu=mu, log=log))
        else:
            fy = dPO(x, mu=mu, log=log) if sigma.flat[0] < 0.0001 else nb
    fy = np.where(x < 0, 0, fy)
    return fy


def pNBI(q, mu=1, sigma=1, lower_tail=True, log_p=False):
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
        size = 1 / sigma
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


def qNBI(p, mu=1, sigma=1, lower_tail=True, log_p=False):
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
        size = 1 / sigma
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


def rNBI(n, mu=1, sigma=1, rng=None):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    if np.any(np.asarray(n) <= 0):
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng() if rng is None else rng
    n = int(np.ceil(n))
    p = rng.uniform(size=n)
    r = qNBI(p, mu=mu, sigma=sigma)
    return r.astype(int)
