"""Beta Binomial distribution (BB). Port of gamlss.dist R/BB.R."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import special as _sp

from ..family import GamlssFamily, checklink
from .BI import dBI, pBI, qBI


def BB(mu_link="logit", sigma_link="log"):
    mstats = checklink("mu.link", "Beta Binomial", mu_link,
                       ("logit", "probit", "cloglog", "cauchit", "log", "own"))
    dstats = checklink("sigma.link", "Beta Binomial", sigma_link,
                       ("inverse", "log", "identity", "sqrt", "own"))

    def dldm(y, mu, sigma, bd):
        return (1 / sigma) * (_sp.digamma(y + (mu / sigma))
                              - _sp.digamma(bd + ((1 - mu) / sigma) - y)
                              - _sp.digamma(mu / sigma)
                              + _sp.digamma((1 - mu) / sigma))

    def d2ldm2(y, mu, sigma, bd):
        return (1 / (sigma) ** 2) * (
            _sp.polygamma(1, y + (mu / sigma))
            + _sp.polygamma(1, bd + ((1 - mu) / sigma) - y)
            - _sp.polygamma(1, mu / sigma)
            - _sp.polygamma(1, (1 - mu) / sigma)
        )

    def dldd(y, mu, sigma, bd):
        k = 1 / sigma
        dldd = -(k**2) * (_sp.digamma(k) + mu * _sp.digamma(y + mu * k)
                          + (1 - mu) * _sp.digamma(bd + (1 - mu) * k - y)
                          - mu * _sp.digamma(mu * k)
                          - (1 - mu) * _sp.digamma((1 - mu) * k)
                          - _sp.digamma(bd + k))
        return dldd

    def d2ldd2(y, mu, sigma, bd):
        k = 1 / sigma
        dldd = -(k**2) * (_sp.digamma(k) + mu * _sp.digamma(y + mu * k)
                          + (1 - mu) * _sp.digamma(bd + (1 - mu) * k - y)
                          - mu * _sp.digamma(mu * k)
                          - (1 - mu) * _sp.digamma((1 - mu) * k)
                          - _sp.digamma(bd + k))
        d2ldd2 = -dldd**2
        return d2ldd2

    return GamlssFamily(
        family=("BB", "Beta Binomial"),
        parameters={"mu": True, "sigma": True},
        nopar=2,
        type="Discrete",
        links={"mu": mstats, "sigma": dstats},
        derivatives={
            "dldm": dldm,
            "d2ldm2": d2ldm2,
            "dldd": dldd,
            "d2ldd2": d2ldd2,
            "d2ldmdd": lambda y: np.zeros(len(np.asarray(y))),
        },
        G_dev_incr=lambda y, mu, sigma, bd: -2 * dBB(y, mu, sigma, bd,
                                                     log=True),
        rqres={"pfun": "pBB", "type": "Discrete", "ymin": 0},
        initial={
            "mu": lambda y, bd: (y + 0.5) / (bd + 1),
            "sigma": lambda y: np.ones(len(y)),
        },
        valid={
            "mu": lambda mu: bool(np.all(mu > 0)) and bool(np.all(mu < 1)),
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
        },
        y_valid=lambda y: bool(np.all(y >= 0)),
        mean=lambda bd, mu, sigma: bd * mu,
        variance=lambda bd, mu, sigma: bd * mu * (1 - mu)
        * (1 + sigma * (bd - 1) / (1 + sigma)),
    )


def dBB(x, mu=0.5, sigma=1, bd=10, log=False):
    if np.any(np.asarray(mu) < 0) or np.any(np.asarray(mu) > 1):
        raise ValueError("mu must be between 0 and 1")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    if np.any(np.asarray(sigma) < 1e-10):
        warnings.warn(" values of sigma in BB less that 1e-10 are set "
                      "to 1e-10")
    sigma = np.where(np.asarray(sigma, float) < 1e-10, 1e-10,
                     np.asarray(sigma, float))
    if np.any(np.asarray(bd) < np.asarray(x)):
        raise ValueError("x must be <= than the binomial denominator")
    x, mu, sigma, bd = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float),
        np.asarray(sigma, float), np.asarray(bd, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        logfy = (_sp.gammaln(bd + 1) - _sp.gammaln(x + 1)
                 - _sp.gammaln(bd - x + 1)
                 + _sp.gammaln((1 / sigma)) + _sp.gammaln(x + mu * (1 / sigma))
                 + _sp.gammaln(bd + ((1 - mu) / sigma) - x)
                 - _sp.gammaln(mu * (1 / sigma))
                 - _sp.gammaln((1 - mu) / sigma)
                 - _sp.gammaln(bd + (1 / sigma)))
        logfy2 = np.where(sigma > 0.0001, logfy,
                          dBI(x, mu=mu, bd=bd, log=True))
    fy = np.exp(logfy2) if log is False else logfy2
    fy = np.where(x < 0, 0.0, fy)
    return fy


def pBB(q, mu=0.5, sigma=1, bd=10, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0) or np.any(np.asarray(mu) >= 1):
        raise ValueError("mu must be between 0 and 1")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    if np.any(np.asarray(bd) < np.asarray(q)):
        raise ValueError("y must be <= than the binomial denominator")
    q, mu, sigma, bd = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(mu, float),
        np.asarray(sigma, float), np.asarray(bd, float)
    )
    qr, mur, sigmar, bdr = (a.ravel() for a in (q, mu, sigma, bd))
    cdf = np.array([
        np.sum(dBB(np.arange(0, np.floor(qi) + 1), mu=mui, sigma=si, bd=bdi))
        for qi, mui, si, bdi in zip(qr, mur, sigmar, bdr)
    ])
    cdf = cdf.reshape(q.shape)
    cdf = cdf if lower_tail is True else 1 - cdf
    with np.errstate(divide="ignore", invalid="ignore"):
        cdf = cdf if log_p is False else np.log(cdf)
    cdf2 = np.where(sigma > 0.0001, cdf,
                    pBI(q, mu=mu, bd=bd, lower_tail=lower_tail, log_p=log_p))
    cdf2 = np.where(q < 0, 0.0, cdf2)
    return cdf2


def qBB(p, mu=0.5, sigma=1, bd=10, lower_tail=True, log_p=False, fast=False):
    if np.any(np.asarray(mu) <= 0) or np.any(np.asarray(mu) >= 1):
        raise ValueError("mu must be between 0 and 1")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    p = np.asarray(p, float)
    if np.any(p < 0) or np.any(p > 1.0001):
        raise ValueError("p must be between 0 and 1")
    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1 - p
    p, mu, sigma, bd = np.broadcast_arrays(
        p, np.asarray(mu, float), np.asarray(sigma, float),
        np.asarray(bd, float)
    )
    QQQ = np.zeros(p.size)
    pr, nmu, nsigma, nbd = (a.ravel() for a in (p, mu, sigma, bd))
    for i in range(p.size):
        cumpro = 0.0
        for j in range(0, int(nbd[i]) + 1):
            if fast is False:
                cumpro = float(pBB(j, mu=nmu[i], sigma=nsigma[i], bd=nbd[i],
                                   log_p=False))
            else:
                cumpro = cumpro + float(dBB(j, mu=nmu[i], sigma=nsigma[i],
                                            bd=nbd[i], log=False))
            QQQ[i] = j
            if pr[i] <= cumpro:
                break
    invcdf = QQQ.reshape(p.shape)
    invcdf2 = np.where(sigma > 0.0001, invcdf,
                       qBI(p, mu=mu, bd=bd, lower_tail=True))
    return invcdf2


def rBB(n, mu=0.5, sigma=1, bd=10, fast=False, rng=None):
    if np.any(np.asarray(mu) <= 0) or np.any(np.asarray(mu) >= 1):
        raise ValueError("mu must be between 0 and 1")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    if np.any(np.asarray(n) <= 0):
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng() if rng is None else rng
    n = int(np.ceil(n))
    p = rng.uniform(size=n)
    r = qBB(p, mu=mu, sigma=sigma, bd=bd, fast=fast)
    return np.asarray(r).astype(int)
