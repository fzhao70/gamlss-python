"""Exponential distribution (EXP). Port of gamlss.dist R/EXP.R."""

from __future__ import annotations

import numpy as np

from ..family import GamlssFamily, checklink


def EXP(mu_link="log"):
    mstats = checklink("mu.link", "Exponential", mu_link,
                       ("inverse", "log", "sqrt", "identity"))
    return GamlssFamily(
        family=("EXP", "Exponential"),
        parameters={"mu": True},
        nopar=1,
        type="Continuous",
        links={"mu": mstats},
        derivatives={
            "dldm": lambda y, mu: (y - mu) / mu**2,
            "d2ldm2": lambda mu: -1 / mu**2,
        },
        G_dev_incr=lambda y, mu: -2 * dEXP(y, mu=mu, log=True),
        rqres={"pfun": "pEXP", "type": "Continuous"},
        initial={"mu": lambda y: (y + np.mean(y)) / 2},
        valid={"mu": lambda mu: bool(np.all(mu > 0))},
        y_valid=lambda y: bool(np.all(y > 0)),
        mean=lambda mu: mu,
        variance=lambda mu: mu**2,
    )


def dEXP(x, mu=1, log=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    x, mu = np.broadcast_arrays(np.asarray(x, float), np.asarray(mu, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        loglik = -np.log(mu) - x / mu
    fy = loglik if log else np.exp(loglik)
    return np.where(x <= 0, 0.0, fy)


def pEXP(q, mu=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    q, mu = np.broadcast_arrays(np.asarray(q, float), np.asarray(mu, float))
    cdf = -np.expm1(-np.maximum(q, 0) / mu)
    if not lower_tail:
        cdf = 1 - cdf
    if log_p:
        cdf = np.log(cdf)
    return cdf


def qEXP(p, mu=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be between 0 and 1")
    if not lower_tail:
        p = 1 - p
    return -np.asarray(mu, float) * np.log1p(-p)


def rEXP(n, mu=1, rng=None):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    rng = np.random.default_rng() if rng is None else rng
    return rng.exponential(scale=np.broadcast_to(mu, (n,)).astype(float))
