"""Skew normal type 1 distribution (SN1, Azzalini type 1). Port of
gamlss.dist R/SN1.R."""

from __future__ import annotations

import numpy as np
from scipy import special as _sp
from scipy import stats as _st
from scipy.integrate import quad
from scipy.optimize import brentq

from ..family import GamlssFamily, checklink


def SN1(mu_link="identity", sigma_link="log", nu_link="identity"):
    mstats = checklink("mu.link", "Skew normal type 1 (Azzalini type 1)",
                       mu_link, ("inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "Skew normal type 1 (Azzalini type 1)",
                       sigma_link, ("inverse", "log", "identity", "own"))
    vstats = checklink("nu.link", "Skew normal type 1 (Azzalini type 1)",
                       nu_link, ("inverse", "log", "identity", "own"))

    def _lpdf_lcdf(y, mu, sigma, nu):
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (y - mu) / sigma
            w = nu * z
            s = ((np.abs(w)) ** 2) / 2
            lpdf = ((1 - (1 / 2)) * np.log(2) - s - _sp.gammaln(1 / 2)
                    - np.log(2))
            lcdf = np.log(0.5 * (1 + _st.gamma.cdf(s, 1 / 2, scale=1)
                                 * np.sign(w)))
        return z, w, s, lpdf, lcdf

    def _dldm(y, mu, sigma, nu):
        z, w, s, lpdf, lcdf = _lpdf_lcdf(y, mu, sigma, nu)
        return (-(np.exp(lpdf - lcdf)) * nu / sigma
                + np.sign(z) * (np.abs(z) ** (2 - 1)) / sigma)

    def _dldd(y, mu, sigma, nu):
        z, w, s, lpdf, lcdf = _lpdf_lcdf(y, mu, sigma, nu)
        return (-(np.exp(lpdf - lcdf)) * nu * z / sigma
                + ((np.abs(z) ** 2) - 1) / sigma)

    def _dldv(y, mu, sigma, nu):
        z, w, s, lpdf, lcdf = _lpdf_lcdf(y, mu, sigma, nu)
        with np.errstate(divide="ignore", invalid="ignore"):
            dwdv = w / nu
            dldv = (np.exp(lpdf - lcdf)) * dwdv
        return dldv

    def dldm(y, mu, sigma, nu):
        return _dldm(y, mu, sigma, nu)

    def d2ldm2(y, mu, sigma, nu):
        dldm = _dldm(y, mu, sigma, nu)
        d2ldm2 = -dldm * dldm
        d2ldm2 = np.where(d2ldm2 < -1e-15, d2ldm2, -1e-15)
        return d2ldm2

    def dldd(y, mu, sigma, nu):
        return _dldd(y, mu, sigma, nu)

    def d2ldd2(y, mu, sigma, nu):
        dldd = _dldd(y, mu, sigma, nu)
        d2ldd2 = -dldd * dldd
        d2ldd2 = np.where(d2ldd2 < -1e-15, d2ldd2, -1e-15)
        return d2ldd2

    def dldv(y, mu, sigma, nu):
        return _dldv(y, mu, sigma, nu)

    def d2ldv2(y, mu, sigma, nu):
        dldv = _dldv(y, mu, sigma, nu)
        d2ldv2 = -dldv * dldv
        d2ldv2 = np.where(d2ldv2 < -1e-15, d2ldv2, -1e-15)
        return d2ldv2

    def d2ldmdd(y, mu, sigma, nu):
        return -(_dldm(y, mu, sigma, nu) * _dldd(y, mu, sigma, nu))

    def d2ldmdv(y, mu, sigma, nu):
        return -(_dldm(y, mu, sigma, nu) * _dldv(y, mu, sigma, nu))

    def d2ldddv(y, mu, sigma, nu):
        return -(_dldd(y, mu, sigma, nu) * _dldv(y, mu, sigma, nu))

    return GamlssFamily(
        family=("SN1", "Skew normal type 1 (Azzalini type 1)"),
        parameters={"mu": True, "sigma": True, "nu": True},
        nopar=3,
        type="Continuous",
        links={"mu": mstats, "sigma": dstats, "nu": vstats},
        derivatives={
            "dldm": dldm,
            "d2ldm2": d2ldm2,
            "dldd": dldd,
            "d2ldd2": d2ldd2,
            "dldv": dldv,
            "d2ldv2": d2ldv2,
            "d2ldmdd": d2ldmdd,
            "d2ldmdv": d2ldmdv,
            "d2ldddv": d2ldddv,
        },
        G_dev_incr=lambda y, mu, sigma, nu: -2 * dSN1(y, mu, sigma, nu,
                                                      log=True),
        rqres={"pfun": "pSN1", "type": "Continuous"},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.full(len(y), np.std(y, ddof=1) / 4),
            "nu": lambda y: np.full(len(y), 0.1),
        },
        valid={
            "mu": lambda mu: True,
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
            "nu": lambda nu: True,
        },
        y_valid=lambda y: True,
        mean=lambda mu, sigma, nu: mu + sigma * nu
        * np.sqrt(2 / ((1 + nu**2) * np.pi)),
        variance=lambda mu, sigma, nu: sigma**2
        * (1 - (2 * nu**2) / ((1 + nu**2) * np.pi)),
    )


def dSN1(x, mu=0, sigma=1, nu=0, log=False):
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be positive")
    x, mu, sigma, nu = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float),
        np.asarray(sigma, float), np.asarray(nu, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x - mu) / sigma
        w = nu * z
        sz = ((np.abs(z)) ** 2) / 2
        s = ((np.abs(w)) ** 2) / 2
        lpdf = (1 - (1 / 2)) * np.log(2) - sz - _sp.gammaln(1 / 2) - np.log(2)
        lcdf1 = np.log(0.5 * (1 + _st.gamma.cdf(s, 1 / 2, scale=1)
                              * np.sign(w)))
        cdf2 = 0.5 + w * np.exp((1 - (1 / 2)) * np.log(2)
                                - _sp.gammaln(1 / 2) - np.log(2))
        lcdf2 = np.log(cdf2)
        lcdf = np.where(s == 0, lcdf2, lcdf1)
        loglik = lpdf + lcdf + np.log(2) - np.log(sigma)
    return loglik if log else np.exp(loglik)


def pSN1(q, mu=0, sigma=1, nu=0, lower_tail=True, log_p=False):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    q = np.asarray(q, float)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    nu = np.asarray(nu, float)
    lp = max(q.size, mu.size, sigma.size, nu.size)
    q = np.resize(q, lp)
    sigma = np.resize(sigma, lp)
    mu = np.resize(mu, lp)
    nu = np.resize(nu, lp)
    cdf = np.zeros(lp)
    for i in range(lp):
        cdf[i] = quad(
            lambda x: float(dSN1(x, mu=0, sigma=1, nu=nu[i])),
            -np.inf, (q[i] - mu[i]) / sigma[i],
        )[0]
    if not lower_tail:
        cdf = 1 - cdf
    if log_p:
        cdf = np.log(cdf)
    return cdf


def qSN1(p, mu=0, sigma=1, nu=0, lower_tail=True, log_p=False):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1 - p
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be between 0 and 1")
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    nu = np.asarray(nu, float)
    lp = max(p.size, mu.size, sigma.size, nu.size)
    p = np.resize(p, lp)
    sigma = np.resize(sigma, lp)
    mu = np.resize(mu, lp)
    nu = np.resize(nu, lp)
    q = np.zeros(lp)
    for i in range(lp):
        def h(qq):
            return pSN1(qq, mu=mu[i], sigma=sigma[i], nu=nu[i]).item()

        def h1(qq):
            return h(qq) - p[i]

        if h(mu[i]) < p[i]:
            interval = [mu[i], mu[i] + sigma[i]]
            j = 2
            while h(interval[1]) < p[i]:
                interval[1] = mu[i] + j * sigma[i]
                j += 1
        else:
            interval = [mu[i] - sigma[i], mu[i]]
            j = 2
            while h(interval[0]) > p[i]:
                interval[0] = mu[i] - j * sigma[i]
                j += 1
        q[i] = brentq(h1, interval[0], interval[1])
    return q


def rSN1(n, mu=0, sigma=1, nu=0, rng=None):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    rng = np.random.default_rng() if rng is None else rng
    p = rng.uniform(size=int(np.ceil(n)))
    return qSN1(p, mu=mu, sigma=sigma, nu=nu)
