"""Poisson Inverse Gaussian distribution (PIG). Port of gamlss.dist R/PIG.R.

The R implementation delegates to the C routines tofyPIG1, tofyPIG2 and
tocdf (src/tofyPIG1.c, src/tofyPIG2.c, src/tocdf.c), which evaluate the
recursion

    t_0 = mu * (1 + 2*sigma*mu)^(-1/2)
    t_j = (sigma*(2*j - 1)/mu + 1/t_{j-1}) * t_0^2

These are ported below as the private helpers _tofy_pig1, _tofy_pig2
and _tocdf_pig (vectorised over observations).
"""

from __future__ import annotations

import numpy as np
from scipy import special as _sp

from ..family import GamlssFamily, checklink


def _tofy_pig1(y, mu, sigma):
    """Port of C tofyPIG1: returns t_y for each observation."""
    y, mu, sigma = np.broadcast_arrays(
        np.asarray(y, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    ky = np.floor(y).astype(int)  # C truncates y + 1 to int
    t0 = mu * (1 + 2 * sigma * mu) ** (-0.5)
    t = t0.copy()
    ans = t0.copy()
    kmax = int(ky.max()) if ky.size else 0
    for j in range(1, kmax + 1):
        t = ((sigma * (2 * j - 1) / mu) + (1 / t)) * t0**2
        ans = np.where(ky >= j, t, ans)
    return ans


def _tofy_pig2(y, mu, sigma):
    """Port of C tofyPIG2: returns sum_{k=0}^{y-1} log(t_k) (0 if y == 0)."""
    y, mu, sigma = np.broadcast_arrays(
        np.asarray(y, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    ky = np.floor(y).astype(int)
    t0 = mu * (1 + 2 * sigma * mu) ** (-0.5)
    t = t0.copy()
    sumT = np.zeros_like(t0)
    kmax = int(ky.max()) if ky.size else 0
    for j in range(1, kmax + 1):
        sumT = np.where(ky >= j, sumT + np.log(t), sumT)
        t = ((sigma * (2 * j - 1) / mu) + (1 / t)) * t0**2
    return sumT


def _tocdf_pig(q, mu, sigma):
    """Port of C tocdf: returns sum_{j=0}^{q} exp(lp_j) with
    lp_0 = (1 - sqrt(1 + 2*sigma*mu))/sigma and
    lp_j = lp_{j-1} + log(t_{j-1}) - log(j)."""
    q, mu, sigma = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    kq = np.floor(q).astype(int)
    t0 = mu * (1 + 2 * sigma * mu) ** (-0.5)
    t = t0.copy()
    lp = (1 - (1 + 2 * sigma * mu) ** 0.5) / sigma
    cdf = np.exp(lp)
    kmax = int(kq.max()) if kq.size else 0
    for j in range(1, kmax + 1):
        lp = lp + np.log(t) - np.log(j)
        t = ((sigma * (2 * j - 1) / mu) + (1 / t)) * t0**2
        cdf = np.where(kq >= j, cdf + np.exp(lp), cdf)
    return cdf


def _pig_dldm(y, mu, sigma):
    ty = _tofy_pig1(y, mu, sigma)
    dldm = (y - ty) / mu
    return dldm


def _pig_dldd(y, mu, sigma):
    ty = _tofy_pig1(y, mu, sigma)
    dldd = ((ty * (1 + sigma * mu) / mu) - (1 + sigma * y)) / (sigma**2)
    return dldd


def PIG(mu_link="log", sigma_link="log"):
    mstats = checklink("mu.link", "Beta Binomial", mu_link,
                       ("inverse", "log", "identity", "sqrt"))
    dstats = checklink("sigma.link", "Beta Binomial", sigma_link,
                       ("inverse", "log", "identity", "sqrt"))

    def d2ldm2(y, mu, sigma):
        dldm = _pig_dldm(y, mu, sigma)
        d2ldm2 = -dldm * dldm
        d2ldm2 = np.where(d2ldm2 < -1e-15, d2ldm2, -1e-15)
        return d2ldm2

    def d2ldd2(y, mu, sigma):
        dldd = _pig_dldd(y, mu, sigma)
        d2ldd2 = -dldd * dldd
        d2ldd2 = np.where(d2ldd2 < -1e-15, d2ldd2, -1e-15)
        return d2ldd2

    def d2ldmdd(y, mu, sigma):
        dldm = _pig_dldm(y, mu, sigma)
        dldd = _pig_dldd(y, mu, sigma)
        d2ldmdd = -dldm * dldd
        return d2ldmdd

    return GamlssFamily(
        family=("PIG", "Poisson.Inverse.Gaussian"),
        parameters={"mu": True, "sigma": True},
        nopar=2,
        type="Discrete",
        links={"mu": mstats, "sigma": dstats},
        derivatives={
            "dldm": _pig_dldm,
            "d2ldm2": d2ldm2,
            "dldd": _pig_dldd,
            "d2ldd2": d2ldd2,
            "d2ldmdd": d2ldmdd,
        },
        G_dev_incr=lambda y, mu, sigma: -2 * dPIG(y, mu, sigma, log=True),
        rqres={"pfun": "pPIG", "type": "Discrete", "ymin": 0},
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


def dPIG(x, mu=1, sigma=1, log=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    x, mu, sigma = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    xx = np.where(x < 0, 0, x)
    sumlty = _tofy_pig2(xx, mu, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        logfy = (-_sp.gammaln(x + 1)
                 + (1 - np.sqrt(1 + 2 * sigma * mu)) / sigma + sumlty)
    fy = logfy if log else np.exp(logfy)
    fy = np.where(x < 0, 0, fy)
    return fy


def pPIG(q, mu=1, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    q, mu, sigma = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    qq = np.where(q < 0, 0, q)
    cdf = _tocdf_pig(qq, mu, sigma)
    if not lower_tail:
        cdf = 1 - cdf
    if log_p:
        with np.errstate(divide="ignore", invalid="ignore"):
            cdf = np.log(cdf)
    cdf = np.where(q < 0, 0, cdf)
    return cdf


def qPIG(p, mu=1, sigma=1, lower_tail=True, log_p=False, max_value=10000):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if np.any(p < 0) or np.any(p > 1.0001):
        raise ValueError("p must be between 0 and 1")
    if not lower_tail:
        p = 1 - p
    p, mu, sigma = np.broadcast_arrays(
        p, np.asarray(mu, float), np.asarray(sigma, float)
    )
    shape = p.shape
    p = np.ravel(p)
    nmu = np.ravel(mu).astype(float)
    nsigma = np.ravel(sigma).astype(float)
    QQQ = np.zeros(len(p))
    inf_mask = (p + 0.000000001) >= 1
    QQQ[inf_mask] = np.inf
    active = ~inf_mask
    if np.any(active):
        # incremental evaluation of pPIG(j, mu, sigma) for j = 0, 1, ...
        t0 = nmu * (1 + 2 * nsigma * nmu) ** (-0.5)
        t = t0.copy()
        lp = (1 - (1 + 2 * nsigma * nmu) ** 0.5) / nsigma
        cumpro = np.exp(lp)
        QQQ = np.where(active, 0.0, QQQ)
        active = active & (p > cumpro)
        for j in range(1, max_value + 1):
            if not np.any(active):
                break
            lp = lp + np.log(t) - np.log(j)
            t = ((nsigma * (2 * j - 1) / nmu) + (1 / t)) * t0**2
            cumpro = cumpro + np.exp(lp)
            QQQ = np.where(active, j, QQQ)
            active = active & (p > cumpro)
    return QQQ.reshape(shape)


def rPIG(n, mu=1, sigma=1, max_value=10000, rng=None):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    if np.any(np.asarray(n) <= 0):
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng() if rng is None else rng
    n = int(np.ceil(n))
    p = rng.uniform(size=n)
    r = qPIG(p, mu=mu, sigma=sigma, max_value=max_value)
    return r
