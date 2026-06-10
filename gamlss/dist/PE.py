"""Power Exponential distribution (PE). Port of gamlss.dist R/PE.R."""

from __future__ import annotations

import numpy as np
from scipy import special as _sp
from scipy import stats as _st

from ..family import GamlssFamily, checklink


def PE(mu_link="identity", sigma_link="log", nu_link="log"):
    mstats = checklink("mu.link", "Power Exponential", mu_link,
                       ("inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "Power Exponential", sigma_link,
                       ("inverse", "log", "identity", "own"))
    vstats = checklink("nu.link", "Power Exponential", nu_link,
                       ("logshiftto1", "log", "identity", "own"))

    def _c(nu):
        log_c = 0.5 * (-(2 / nu) * np.log(2) + _sp.gammaln(1 / nu)
                       - _sp.gammaln(3 / nu))
        return np.exp(log_c)

    def dldm(y, mu, sigma, nu):
        c = _c(nu)
        z = (y - mu) / sigma
        out = (np.sign(z) * nu) / (2 * sigma * np.abs(z))
        return out * ((np.abs(z / c)) ** nu)

    def d2ldm2(y, mu, sigma, nu):
        c = _c(nu)
        z = (y - mu) / sigma
        dldm = (np.sign(z) * nu) / (2 * sigma * np.abs(z))
        dldm = dldm * ((np.abs(z / c)) ** nu)
        out = -(nu * nu * _sp.gamma(2 - (1 / nu)) * _sp.gamma(3 / nu)) / (
            (sigma * _sp.gamma(1 / nu)) ** 2
        )
        return np.where(nu < 1.05, -dldm * dldm, out)

    def dldd(y, mu, sigma, nu):
        c = _c(nu)
        z = (y - mu) / sigma
        return ((nu / 2) * ((np.abs(z / c)) ** nu) - 1) / sigma

    def dldv(y, mu, sigma, nu):
        c = _c(nu)
        z = (y - mu) / sigma
        dlogc_dv = (1 / (2 * nu**2)) * (
            2 * np.log(2) - _sp.digamma(1 / nu) + 3 * _sp.digamma(3 / nu)
        )
        out = (1 / nu) - 0.5 * (np.log(np.abs(z / c))
                                * ((np.abs(z / c)) ** nu))
        out = out + np.log(2) / (nu**2) + _sp.digamma(1 / nu) / (nu**2)
        return out + (-1 + (nu / 2) * ((np.abs(z / c)) ** nu)) * dlogc_dv

    def d2ldv2(y, mu, sigma, nu):
        dlogc_dv = (1 / (2 * nu**2)) * (
            2 * np.log(2) - _sp.digamma(1 / nu) + 3 * _sp.digamma(3 / nu)
        )
        p = (1 + nu) / nu
        part1 = p * _sp.polygamma(1, p) + 2 * _sp.digamma(p) ** 2
        part2 = _sp.digamma(p) * (np.log(2) + 3 - 3 * _sp.digamma(3 / nu) - nu)
        part3 = -3 * _sp.digamma(3 / nu) * (1 + np.log(2))
        part4 = -(nu + np.log(2)) * np.log(2)
        part5 = -nu + (nu**4) * dlogc_dv**2
        out = part1 + part2 + part3 + part4 + part5
        out = -out / nu**3
        return np.where(out < -1e-15, out, -1e-15)

    return GamlssFamily(
        family=("PE", "Power Exponential"),
        parameters={"mu": True, "sigma": True, "nu": True},
        nopar=3,
        type="Continuous",
        links={"mu": mstats, "sigma": dstats, "nu": vstats},
        derivatives={
            "dldm": dldm,
            "d2ldm2": d2ldm2,
            "dldd": dldd,
            "d2ldd2": lambda sigma, nu: -nu / (sigma**2),
            "dldv": dldv,
            "d2ldv2": d2ldv2,
            "d2ldmdd": lambda y: np.zeros(len(np.asarray(y))),
            "d2ldmdv": lambda y: np.zeros(len(np.asarray(y))),
            "d2ldddv": lambda y, mu, sigma, nu: (1 / (2 * sigma)) * (
                (3 / nu) * (_sp.digamma(1 / nu) - _sp.digamma(3 / nu))
                + 2 + (2 / nu)
            ),
        },
        G_dev_incr=lambda y, mu, sigma, nu: -2 * dPE(y, mu, sigma, nu,
                                                     log=True),
        rqres={"pfun": "pPE", "type": "Continuous"},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: (np.abs(y - np.mean(y))
                                + np.std(y, ddof=1)) / 2,
            "nu": lambda y: np.full(len(y), 1.8),
        },
        valid={
            "mu": lambda mu: True,
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
            "nu": lambda nu: bool(np.all(nu > 0)),
        },
        y_valid=lambda y: True,
        mean=lambda mu, sigma, nu: mu,
        variance=lambda mu, sigma, nu: sigma**2,
    )


def dPE(x, mu=0, sigma=1, nu=2, log=False):
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(nu) < 0):
        raise ValueError("nu must be positive")
    x, mu, sigma, nu = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float),
        np.asarray(sigma, float), np.asarray(nu, float)
    )
    log_c = 0.5 * (-(2 / nu) * np.log(2) + _sp.gammaln(1 / nu)
                   - _sp.gammaln(3 / nu))
    c = np.exp(log_c)
    z = (x - mu) / sigma
    log_lik = (-np.log(sigma) + np.log(nu) - log_c
               - (0.5 * (np.abs(z / c) ** nu))
               - (1 + (1 / nu)) * np.log(2) - _sp.gammaln(1 / nu))
    return log_lik if log else np.exp(log_lik)


def pPE(q, mu=0, sigma=1, nu=2, lower_tail=True, log_p=False):
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(nu) < 0):
        raise ValueError("nu must be positive")
    q, mu, sigma, nu = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(mu, float),
        np.asarray(sigma, float), np.asarray(nu, float)
    )
    log_c = 0.5 * (-(2 / nu) * np.log(2) + _sp.gammaln(1 / nu)
                   - _sp.gammaln(3 / nu))
    c = np.exp(log_c)
    z = (q - mu) / sigma
    s = 0.5 * (np.abs(z / c) ** nu)
    cdf = 0.5 * (1 + _st.gamma.cdf(s, 1 / nu, scale=1) * np.sign(z))
    cdf = np.where(
        nu > 10000,
        (q - (mu - np.sqrt(3) * sigma)) / (np.sqrt(12) * sigma),
        cdf,
    )
    if not lower_tail:
        cdf = 1 - cdf
    if log_p:
        cdf = np.log(cdf)
    return cdf


def qPE(p, mu=0, sigma=1, nu=2, lower_tail=True, log_p=False):
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(nu) < 0):
        raise ValueError("nu must be positive")
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1 - p
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be between 0 and 1")
    p, mu, sigma, nu = np.broadcast_arrays(
        p, np.asarray(mu, float), np.asarray(sigma, float),
        np.asarray(nu, float)
    )
    log_c = 0.5 * (-(2 / nu) * np.log(2) + _sp.gammaln(1 / nu)
                   - _sp.gammaln(3 / nu))
    c = np.exp(log_c)
    with np.errstate(invalid="ignore"):
        s = _st.gamma.ppf((2 * p - 1) * np.sign(p - 0.5), 1 / nu, scale=1)
    z = np.sign(p - 0.5) * ((2 * s) ** (1 / nu)) * c
    return mu + sigma * z


def rPE(n, mu=0, sigma=1, nu=2, rng=None):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    rng = np.random.default_rng() if rng is None else rng
    return qPE(rng.uniform(size=int(np.ceil(n))), mu=mu, sigma=sigma, nu=nu)
