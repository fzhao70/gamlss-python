"""Geometric distribution (GEOM). Port of gamlss.dist R/GEOM.R.

R's dgeom counts failures before the first success (support 0, 1, 2, ...),
which maps to scipy.stats.geom with loc=-1.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ..family import GamlssFamily, checklink


def GEOM(mu_link="log"):
    mstats = checklink("mu.link", "Geometric", mu_link,
                       ("log", "probit", "cloglog", "cauchit", "log", "own"))
    return GamlssFamily(
        family=("GEOM", "Geometric"),
        parameters={"mu": True},
        nopar=1,
        type="Discrete",
        links={"mu": mstats},
        derivatives={
            "dldm": lambda y, mu: (y - mu) / (mu + (mu**2)),
            "d2ldm2": lambda mu: -1 / (mu + (mu**2)),
        },
        G_dev_incr=lambda y, mu: -2 * dGEOM(x=y, mu=mu, log=True),
        rqres={"pfun": "pGEOM", "type": "Discrete", "ymin": 0},
        initial={"mu": lambda y: np.full(len(y), np.mean(y))},
        valid={"mu": lambda mu: bool(np.all(mu > 0))},
        y_valid=lambda y: bool(np.all(y >= 0)),
        mean=lambda mu: mu,
        variance=lambda mu: mu + mu**2,
    )


def dGEOM(x, mu=2, log=False):
    if np.any(np.asarray(mu) < 0):
        raise ValueError("mu must be > 0)")
    x, mu = np.broadcast_arrays(np.asarray(x, float), np.asarray(mu, float))
    prob = 1 / (mu + 1)
    fy = (_st.geom.logpmf(x, prob, loc=-1) if log
          else _st.geom.pmf(x, prob, loc=-1))
    fy = np.where(x < 0, 0, fy)
    return fy


def pGEOM(q, mu=2, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) < 0):
        raise ValueError("mu must be > 0")
    q, mu = np.broadcast_arrays(np.asarray(q, float), np.asarray(mu, float))
    prob = 1 / (mu + 1)
    qf = np.floor(q)
    if lower_tail:
        cdf = (_st.geom.logcdf(qf, prob, loc=-1) if log_p
               else _st.geom.cdf(qf, prob, loc=-1))
    else:
        cdf = (_st.geom.logsf(qf, prob, loc=-1) if log_p
               else _st.geom.sf(qf, prob, loc=-1))
    cdf = np.where(q < 0, 0, cdf)
    return cdf


def qGEOM(p, mu=2, lower_tail=True, log_p=False):
    # NOTE: as in the R source, lower.tail/log.p are accepted but the
    # quantiles are computed with lower.tail = TRUE, log.p = FALSE.
    if np.any(np.asarray(mu) < 0):
        raise ValueError("mu must be > 0)")
    p = np.asarray(p, float)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be between 0 and 1")
    p, mu = np.broadcast_arrays(p, np.asarray(mu, float))
    prob = 1 / (mu + 1)
    QQQ = _st.geom.ppf(p, prob, loc=-1)
    # R's qgeom returns 0 at p == 0; scipy's ppf returns -1
    QQQ = np.where(p == 0, 0.0, QQQ)
    return QQQ


def rGEOM(n, mu=2, rng=None):
    if np.any(np.asarray(mu) < 0):
        raise ValueError("mu must be > 0)")
    if np.any(np.asarray(n) <= 0):
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng() if rng is None else rng
    n = int(np.ceil(n))
    p = rng.uniform(size=n)
    r = qGEOM(p, mu=mu)
    return r.astype(int)
