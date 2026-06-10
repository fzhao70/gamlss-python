"""Poisson distribution (PO). Port of gamlss.dist R/po.r."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ..family import GamlssFamily, checklink


def PO(mu_link="log"):
    mstats = checklink("mu.link", "Poisson", mu_link,
                       ("inverse", "log", "sqrt", "identity"))
    return GamlssFamily(
        family=("PO", "Poisson"),
        parameters={"mu": True},
        nopar=1,
        type="Discrete",
        links={"mu": mstats},
        derivatives={
            "dldm": lambda y, mu: (y - mu) / mu,
            "d2ldm2": lambda mu: -1 / mu,
        },
        G_dev_incr=lambda y, mu: -2 * dPO(y, mu=mu, log=True),
        rqres={"pfun": "pPO", "type": "Discrete", "ymin": 0},
        initial={"mu": lambda y: (y + np.mean(y)) / 2},
        valid={"mu": lambda mu: bool(np.all(mu > 0))},
        y_valid=lambda y: bool(np.all(y >= 0)),
        mean=lambda mu: mu,
        variance=lambda mu: mu,
    )


def dPO(x, mu=1, log=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    x, mu = np.broadcast_arrays(np.asarray(x, float), np.asarray(mu, float))
    fy = _st.poisson.logpmf(x, mu) if log else _st.poisson.pmf(x, mu)
    return np.where(x < 0, 0.0, fy)


def pPO(q, mu=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    q, mu = np.broadcast_arrays(np.asarray(q, float), np.asarray(mu, float))
    qf = np.floor(q)
    if lower_tail:
        cdf = _st.poisson.logcdf(qf, mu) if log_p else _st.poisson.cdf(qf, mu)
    else:
        cdf = _st.poisson.logsf(qf, mu) if log_p else _st.poisson.sf(qf, mu)
    cdf = np.where(q < 0, 0.0, cdf)
    return cdf


def qPO(p, mu=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be between 0 and 1")
    if not lower_tail:
        p = 1 - p
    return _st.poisson.ppf(p, np.asarray(mu, float))


def rPO(n, mu=1, rng=None):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    rng = np.random.default_rng() if rng is None else rng
    return rng.poisson(mu, size=n).astype(int)
