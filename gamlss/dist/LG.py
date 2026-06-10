"""Logarithmic distribution (LG). Port of gamlss.dist R/LG.R."""

from __future__ import annotations

import numpy as np

from ..family import GamlssFamily, checklink


def LG(mu_link="logit"):
    mstats = checklink("mu.link", "LG", mu_link,
                       ("logit", "probit", "cloglog", "cauchit", "log", "own"))

    def d2ldm2(y, mu):
        dldm = (y / mu) + 1 / ((1 - mu) * np.log(1 - mu))
        d2ldm2 = -dldm**2
        return d2ldm2

    return GamlssFamily(
        family=("LG", "Logarithmic"),
        parameters={"mu": True},
        nopar=1,
        type="Discrete",
        links={"mu": mstats},
        derivatives={
            "dldm": lambda y, mu: (y / mu) + 1 / ((1 - mu) * np.log(1 - mu)),
            "d2ldm2": d2ldm2,
        },
        G_dev_incr=lambda y, mu: -2 * dLG(x=y, mu=mu, log=True),
        rqres={"pfun": "pLG", "type": "Discrete", "ymin": 1},
        initial={"mu": lambda: 0.9},
        valid={"mu": lambda mu: bool(np.all((mu > 0) & (mu < 1)))},
        y_valid=lambda y: bool(np.all(y > 0)),
        mean=lambda mu: -(np.log(1 - mu)) ** -1 * mu * (1 - mu) ** -1,
        variance=lambda mu: -(np.log(1 - mu)) ** -1 * mu
        * (1 + (np.log(1 - mu)) ** -1 * mu) * (1 - mu) ** -2,
    )


def dLG(x, mu=0.5, log=False):
    if np.any(np.asarray(mu) <= 0) or np.any(np.asarray(mu) >= 1):
        raise ValueError("mu must be greater than 0 and less than 1")
    x, mu = np.broadcast_arrays(np.asarray(x, float), np.asarray(mu, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        logfy = x * np.log(mu) - np.log(x) - np.log(-np.log(1 - mu))
    fy = np.exp(logfy) if log is False else logfy
    fy = np.where(x < 1, 0.0, fy)
    return fy


def pLG(q, mu=0.5, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0) or np.any(np.asarray(mu) >= 1):
        raise ValueError("mu must be greater than 0 and less than 1")
    q, mu = np.broadcast_arrays(np.asarray(q, float), np.asarray(mu, float))
    qr, mur = q.ravel(), mu.ravel()
    cdf = np.array([
        np.sum(dLG(np.arange(1, np.floor(qi) + 1), mu=mui))
        for qi, mui in zip(qr, mur)
    ])
    cdf = cdf.reshape(q.shape)
    cdf = cdf if lower_tail is True else 1 - cdf
    with np.errstate(divide="ignore", invalid="ignore"):
        cdf = cdf if log_p is False else np.log(cdf)
    cdf = np.where(q < 1, 0.0, cdf)
    return cdf


def qLG(p, mu=0.5, lower_tail=True, log_p=False, max_value=10000):
    if np.any(np.asarray(mu) <= 0) or np.any(np.asarray(mu) >= 1):
        raise ValueError("mu must be greater than 0 and less than 1")
    p = np.asarray(p, float)
    if np.any(p < 0) or np.any(p > 1.0001):
        raise ValueError("p must be between 0 and 1")
    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1 - p
    p, mu = np.broadcast_arrays(p, np.asarray(mu, float))
    pr, nmu = p.ravel(), mu.ravel()
    QQQ = np.zeros(p.size)
    for i in range(p.size):
        cumpro = 0.0
        if pr[i] + 0.000000001 >= 1:
            QQQ[i] = np.inf
        else:
            for j in range(1, int(max_value) + 1):
                cumpro = float(pLG(j, mu=nmu[i], log_p=False))
                QQQ[i] = j
                if pr[i] <= cumpro:
                    break
    return QQQ.reshape(p.shape)


def rLG(n, mu=0.5, rng=None):
    if np.any(np.asarray(mu) <= 0) or np.any(np.asarray(mu) >= 1):
        raise ValueError("mu must be greater than 0 and less than 1")
    if np.any(np.asarray(n) <= 0):
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng() if rng is None else rng
    n = int(np.ceil(n))
    p = rng.uniform(size=n)
    r = qLG(p, mu=mu)
    return np.asarray(r).astype(int)
