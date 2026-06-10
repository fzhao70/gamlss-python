"""Generalised Gamma distribution (GG). Port of gamlss.dist R/GG.R.

Lopatatsidis-Green parameterisation.
"""

from __future__ import annotations

import numpy as np
from scipy import special as _sp

from ..family import GamlssFamily, checklink
from .GA import dGA, pGA, qGA
from .NO import pNO, qNO


def GG(mu_link="log", sigma_link="log", nu_link="identity"):
    mstats = checklink("mu.link", "GG", mu_link,
                       ("1/mu^2", "log", "identity"))
    dstats = checklink("sigma.link", "GG", sigma_link,
                       ("inverse", "log", "identity"))
    vstats = checklink("nu.link", "GG", nu_link,
                       ("1/nu^2", "log", "identity"))

    def dldm(y, mu, sigma, nu):
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (y / mu) ** nu
            theta = 1 / (sigma**2 * np.abs(nu) ** 2)
            dldm = np.where(
                np.abs(nu) > 1e-06,
                (z - 1) * theta * nu / mu,
                (1 / (mu * (sigma**2)) * (np.log(y) - np.log(mu))),
            )
        return dldm

    def d2ldm2(mu, sigma, nu):
        d2ldm2 = np.where(
            np.abs(nu) > 1e-06,
            -1 / ((mu**2) * (sigma**2)),
            -(1 / (mu**2 * sigma**2)),
        )
        return d2ldm2

    def dldd(y, mu, sigma, nu):
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (y / mu) ** nu
            theta = 1 / (sigma**2 * np.abs(nu) ** 2)
            dldd = np.where(
                np.abs(nu) > 1e-06,
                -2 * theta * (np.log(theta) + 1 + np.log(z) - z
                              - _sp.digamma(theta)) / sigma,
                -(1 / sigma) + (1 / sigma**3)
                * (np.log(y) - np.log(mu)) ** 2,
            )
        return dldd

    def d2ldd2(y, mu, sigma, nu):
        with np.errstate(divide="ignore", invalid="ignore"):
            theta = 1 / (sigma**2 * np.abs(nu) ** 2)
            d2ldd2 = np.where(
                np.abs(nu) > 1e-06,
                4 * (theta / (sigma**2)) * (1 - theta * _sp.polygamma(1, theta)),
                -2 / sigma**2,
            )
        return d2ldd2

    def dldv(y, mu, sigma, nu):
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (y / mu) ** nu
            theta = 1 / (sigma**2 * np.abs(nu) ** 2)
            dldv = (1 / nu) * (1 + 2 * theta * (_sp.digamma(theta) + z
                                                - np.log(theta) - 1
                                                - ((z + 1) / 2) * np.log(z)))
        return dldv

    def d2ldv2(y, mu, sigma, nu):
        with np.errstate(divide="ignore", invalid="ignore"):
            theta = 1 / (sigma**2 * np.abs(nu) ** 2)
            d2ldv2 = -(theta / nu**2) * (
                _sp.polygamma(1, theta) * (1 + 4 * theta)
                - (4 + 3 / theta)
                - np.log(theta) * (2 / theta - np.log(theta))
                + _sp.digamma(theta) * (_sp.digamma(theta) + (2 / theta)
                                        - 2 * np.log(theta))
            )
        return d2ldv2

    def d2ldmdv(y, mu, sigma, nu):
        theta = 1 / (sigma**2 * np.abs(nu) ** 2)
        ddd = (theta / mu) * (_sp.digamma(theta) + (1 / theta)
                              - np.log(theta))
        return ddd

    def d2ldddv(y, mu, sigma, nu):
        theta = 1 / (sigma**2 * np.abs(nu) ** 2)
        d2ldddv = -2 * np.sign(nu) * theta ** (3 / 2) * (
            2 * theta * _sp.polygamma(1, theta) - (1 / theta) - 2
        )
        return d2ldddv

    def _mean(mu, sigma, nu):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            out = np.where(
                (nu > 0) | ((nu < 0) & (sigma**2 * np.abs(nu) < 1)),
                (mu * _sp.gamma(1 / (sigma**2 * nu**2) + 1 / nu))
                / ((1 / (sigma**2 * nu**2)) ** (1 / nu)
                   * _sp.gamma(1 / (sigma**2 * nu**2))),
                np.inf,
            )
        return out

    def _variance(mu, sigma, nu):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            out = np.where(
                (nu > 0) | ((nu < 0) & (sigma**2 * np.abs(nu) < 0.5)),
                (mu**2 / ((1 / (sigma**2 * nu**2)) ** (2 / nu)
                          * (_sp.gamma(1 / (sigma**2 * nu**2))) ** 2))
                * (_sp.gamma(1 / (sigma**2 * nu**2) + 2 / nu)
                   * _sp.gamma(1 / (sigma**2 * nu**2))
                   - (_sp.gamma(1 / (sigma**2 * nu**2) + 1 / nu)) ** 2),
                np.inf,
            )
        return out

    return GamlssFamily(
        family=("GG", "generalised Gamma Lopatatsidis-Green"),
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
            "d2ldmdd": lambda y: np.zeros(len(np.asarray(y))),
            "d2ldmdv": d2ldmdv,
            "d2ldddv": d2ldddv,
        },
        G_dev_incr=lambda y, mu, sigma, nu: -2 * dGG(y, mu=mu, sigma=sigma,
                                                     nu=nu, log=True),
        rqres={"pfun": "pGG", "type": "Continuous"},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.ones(len(y)),
            "nu": lambda y: np.ones(len(y)),
        },
        valid={
            "mu": lambda mu: bool(np.all(mu > 0)),
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
            "nu": lambda nu: True,
        },
        y_valid=lambda y: bool(np.all(y > 0)),
        mean=_mean,
        variance=_variance,
    )


def dGG(x, mu=1, sigma=0.5, nu=1, log=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    nu_len1 = np.asarray(nu).size == 1
    x, mu, sigma, nu = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float),
        np.asarray(sigma, float), np.asarray(nu, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x / mu) ** nu
        if not nu_len1:
            loglik = np.where(
                np.abs(nu) > 1e-06,
                dGA(z, mu=1, sigma=sigma * np.abs(nu), log=True)
                + np.log(np.abs(nu) * z / x),
                -np.log(x) - 0.5 * np.log(2 * np.pi) - np.log(sigma)
                - (1 / (2 * sigma**2)) * (np.log(x) - np.log(mu)) ** 2,
            )
        else:
            if np.all(np.abs(nu) > 1e-06):
                loglik = (dGA(z, mu=1, sigma=sigma * np.abs(nu), log=True)
                          + np.log(np.abs(nu) * z / x))
            else:
                loglik = (-np.log(x) - 0.5 * np.log(2 * np.pi)
                          - np.log(sigma)
                          - (1 / (2 * sigma**2))
                          * (np.log(x) - np.log(mu)) ** 2)
    ft = loglik if log else np.exp(loglik)
    return np.where(x <= 0, 0.0, ft)


def pGG(q, mu=1, sigma=0.5, nu=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    nu_len1 = np.asarray(nu).size == 1
    q, mu, sigma, nu = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(mu, float),
        np.asarray(sigma, float), np.asarray(nu, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (q / mu) ** nu
        if not nu_len1:
            # R: pGA(..., lower.tail = (nu<0)-lower.tail, ...); pgamma
            # treats the non-zero values of (nu<0)-lower.tail as TRUE.
            lt = (nu < 0).astype(int) - (1 if lower_tail else 0)
            cdf_ga = np.where(
                lt != 0,
                pGA(z, mu=1, sigma=sigma * np.abs(nu), lower_tail=True,
                    log_p=log_p),
                pGA(z, mu=1, sigma=sigma * np.abs(nu), lower_tail=False,
                    log_p=log_p),
            )
            cdf = np.where(
                np.abs(nu) > 1e-06,
                cdf_ga,
                pNO(np.log(z), mu=np.log(mu), sigma=sigma),
            )
        else:
            if np.all(np.abs(nu) > 1e-06):
                lt = bool(((nu.flat[0] < 0) - (1 if lower_tail else 0)) != 0)
                cdf = pGA(z, mu=1, sigma=sigma * np.abs(nu), lower_tail=lt,
                          log_p=log_p)
            else:
                cdf = pNO(np.log(q), mu=np.log(mu), sigma=sigma)
    return cdf


def qGG(p, mu=1, sigma=0.5, nu=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) < 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be positive")
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1 - p
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be between 0 and 1")
    nu_len1 = np.asarray(nu).size == 1
    p, mu, sigma, nu = np.broadcast_arrays(
        p, np.asarray(mu, float), np.asarray(sigma, float),
        np.asarray(nu, float)
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if not nu_len1:
            p = np.where(nu > 0, p, 1 - p)
            z = np.where(
                np.abs(nu) > 1e-06,
                qGA(p, mu=1, sigma=sigma * np.abs(nu)),
                qNO(p, mu=np.log(mu), sigma=sigma),
            )
            y = np.where(np.abs(nu) > 1e-06, mu * z ** (1 / nu), np.exp(z))
        else:
            if np.all(np.abs(nu) > 1e-06):
                p = p if nu.flat[0] > 0 else 1 - p
                z = qGA(p, mu=1, sigma=sigma * np.abs(nu))
                y = mu * z ** (1 / nu)
            else:
                z = qNO(p, mu=np.log(mu), sigma=sigma)
                y = np.exp(z)
    return y


def rGG(n, mu=1, sigma=0.5, nu=1, rng=None):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    rng = np.random.default_rng() if rng is None else rng
    p = rng.uniform(size=int(np.ceil(n)))
    return qGG(p, mu=mu, sigma=sigma, nu=nu)
