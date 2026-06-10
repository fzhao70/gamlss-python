"""Binomial distribution (BI). Port of gamlss.dist R/BI.R."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ..family import GamlssFamily, checklink


def _BI_dldm(y, mu, bd):
    """BI dldm (R: BI()$dldm, reused by ZABI/ZIBI)."""
    return (y - bd * mu) / (mu * (1 - mu))


def _dBI0(bd, mu):
    """R's dBI(0, bd, mu) for vector bd/mu.

    R's dBI ends with ``ifelse(x < 0, 0, fy)`` whose result takes the
    shape of x; with scalar x=0 it silently truncates to
    dbinom(0, bd[1], mu[1]).  ZABI/ZIBI rely on this (the zero-mass
    correction uses the FIRST observation's bd and mu for every row),
    so it is replicated here for exact agreement with R.
    """
    bd0 = float(np.asarray(bd, dtype=float).ravel()[0])
    mu0 = float(np.asarray(mu, dtype=float).ravel()[0])
    return float(_st.binom.pmf(0.0, n=bd0, p=mu0))


def BI(mu_link="logit"):
    mstats = checklink("mu.link", "Binomial", mu_link,
                       ("logit", "probit", "cloglog", "cauchit", "log", "own"))
    return GamlssFamily(
        family=("BI", "Binomial"),
        parameters={"mu": True},
        nopar=1,
        type="Discrete",
        links={"mu": mstats},
        derivatives={
            "dldm": _BI_dldm,
            "d2ldm2": lambda mu, bd: -(bd / (mu * (1 - mu))),
        },
        G_dev_incr=lambda y, mu, bd: -2 * dBI(y, bd=bd, mu=mu, log=True),
        rqres={"pfun": "pBI", "type": "Discrete", "ymin": 0},
        initial={"mu": lambda y, bd: (y + 0.5) / (bd + 1)},
        valid={"mu": lambda mu: bool(np.all(mu > 0)) and bool(np.all(mu < 1))},
        y_valid=lambda y: bool(np.all(y >= 0)),
        mean=lambda bd, mu: bd * mu,
        variance=lambda bd, mu: bd * mu * (1 - mu),
    )


def dBI(x, bd=1, mu=0.5, log=False):
    if np.any(np.asarray(mu) < 0) or np.any(np.asarray(mu) > 1):
        raise ValueError("mu must be between 0 and 1")
    if np.any(np.asarray(bd) < np.asarray(x)):
        raise ValueError("x must be <= than the binomial denominator")
    x, bd, mu = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(bd, float), np.asarray(mu, float)
    )
    fy = (_st.binom.logpmf(x, n=bd, p=mu) if log
          else _st.binom.pmf(x, n=bd, p=mu))
    return np.where(x < 0, 0.0, fy)


def pBI(q, bd=1, mu=0.5, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) < 0) or np.any(np.asarray(mu) > 1):
        raise ValueError("mu must be between 0 and 1")
    if np.any(np.asarray(bd) < np.asarray(q)):
        raise ValueError("q must be <= than the binomial denominator")
    q, bd, mu = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(bd, float), np.asarray(mu, float)
    )
    if lower_tail:
        cdf = (_st.binom.logcdf(q, n=bd, p=mu) if log_p
               else _st.binom.cdf(q, n=bd, p=mu))
    else:
        cdf = (_st.binom.logsf(q, n=bd, p=mu) if log_p
               else _st.binom.sf(q, n=bd, p=mu))
    cdf = np.where(q < 0, 0.0, cdf)
    return cdf


def qBI(p, bd=1, mu=0.5, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) < 0) or np.any(np.asarray(mu) > 1):
        raise ValueError("mu must be between 0 and 1")
    p = np.asarray(p, float)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be between 0 and 1")
    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1 - p
    return _st.binom.ppf(p, n=np.asarray(bd, float), p=np.asarray(mu, float))


def rBI(n, bd=1, mu=0.5, rng=None):
    if np.any(np.asarray(mu) < 0) or np.any(np.asarray(mu) > 1):
        raise ValueError("mu must be between 0 and 1")
    if np.any(np.asarray(n) <= 0):
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng() if rng is None else rng
    n = int(np.ceil(n))
    p = rng.uniform(size=n)
    r = _st.binom.ppf(p, n=np.asarray(bd, float), p=np.asarray(mu, float))
    return r.astype(int)
