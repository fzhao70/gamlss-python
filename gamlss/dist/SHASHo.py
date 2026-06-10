"""Sinh-Arcsinh original distribution (SHASHo). Port of gamlss.dist
R/SHASHo.R.

The original SHASH distribution from Jones and Pewsey (2009)
Biometrika, page 2 equation (2) (divided by sigma for the density of
the unstandardized variable).
"""

from __future__ import annotations

import numpy as np
from scipy import special as _sp
from scipy import stats as _st

from ..family import GamlssFamily, checklink
from .NO import pNO


def _dldm(y, mu, sigma, nu, tau):
    z = (y - mu) / sigma
    r = (1 / 2) * (np.exp(tau * np.arcsinh(z) - nu)
                   - np.exp(-tau * np.arcsinh(z) + nu))
    c = (1 / 2) * (np.exp(tau * np.arcsinh(z) - nu)
                   + np.exp(-(tau * np.arcsinh(z) - nu)))
    x = tau * np.arcsinh(z) - nu
    dldm = (1 / (sigma * (1 + z**2) ** (1 / 2))) * (
        r * tau * c - ((tau * np.sinh(x)) / c) + z / (1 + z**2) ** (1 / 2)
    )
    return dldm


def _dldd(y, mu, sigma, nu, tau):
    z = (y - mu) / sigma
    r = (1 / 2) * (np.exp(tau * np.arcsinh(z) - nu)
                   - np.exp(-tau * np.arcsinh(z) + nu))
    c = (1 / 2) * (np.exp(tau * np.arcsinh(z) - nu)
                   + np.exp(-(tau * np.arcsinh(z) - nu)))
    x = tau * np.arcsinh(z) - nu
    dldd = ((z) / (sigma * (1 + z**2) ** (1 / 2))) * (
        r * tau * c - ((tau * np.sinh(x)) / c) + z / ((1 + z**2) ** (1 / 2))
    ) - (1 / sigma)
    return dldd


def _dldv(y, mu, sigma, nu, tau):
    z = (y - mu) / sigma
    r = (1 / 2) * (np.exp(tau * np.arcsinh(z) - nu)
                   - np.exp(-tau * np.arcsinh(z) + nu))
    c = (1 / 2) * (np.exp(tau * np.arcsinh(z) - nu)
                   + np.exp(-(tau * np.arcsinh(z) - nu)))
    dldv = (r * np.cosh(tau * np.arcsinh(z) - nu)
            - (1 / c) * np.sinh(tau * np.arcsinh(z) - nu))
    return dldv


def _dldt(y, mu, sigma, nu, tau):
    z = (y - mu) / sigma
    r = (1 / 2) * (np.exp(tau * np.arcsinh(z) - nu)
                   - np.exp(-tau * np.arcsinh(z) + nu))
    c = (1 / 2) * (np.exp(tau * np.arcsinh(z) - nu)
                   + np.exp(-(tau * np.arcsinh(z) - nu)))
    dldt = (-r * np.cosh(tau * np.arcsinh(z) - nu)
            + (1 / c) * np.sinh(tau * np.arcsinh(z) - nu)) * (
        np.arcsinh(z)
    ) + 1 / tau
    return dldt


def SHASHo(mu_link="identity", sigma_link="log", nu_link="identity",
           tau_link="log"):
    mstats = checklink("mu.link", "Sinh-Arcsinh", mu_link,
                       ("inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "Sinh-Arcsinh", sigma_link,
                       ("inverse", "log", "identity", "own"))
    vstats = checklink("nu.link", "Sinh-Arcsinh", nu_link,
                       ("inverse", "log", "identity", "own"))
    tstats = checklink("tau.link", "Sinh-Arcsinh", tau_link,
                       ("inverse", "log", "identity", "own"))

    def dldm(y, mu, sigma, nu, tau):
        return _dldm(y, mu, sigma, nu, tau)

    def d2ldm2(y, mu, sigma, nu, tau):
        dldm = _dldm(y, mu, sigma, nu, tau)
        d2ldm2 = -dldm * dldm
        return d2ldm2

    def dldd(y, mu, sigma, nu, tau):
        return _dldd(y, mu, sigma, nu, tau)

    def d2ldd2(y, mu, sigma, nu, tau):
        dldd = _dldd(y, mu, sigma, nu, tau)
        d2ldd2 = -dldd * dldd
        return d2ldd2

    def dldv(y, mu, sigma, nu, tau):
        return _dldv(y, mu, sigma, nu, tau)

    def d2ldv2(y, mu, sigma, nu, tau):
        dldv = _dldv(y, mu, sigma, nu, tau)
        d2ldv2 = -dldv * dldv
        return d2ldv2

    def dldt(y, mu, sigma, nu, tau):
        return _dldt(y, mu, sigma, nu, tau)

    def d2ldt2(y, mu, sigma, nu, tau):
        dldt = _dldt(y, mu, sigma, nu, tau)
        d2ldt2 = -dldt * dldt
        return d2ldt2

    def d2ldmdd(y, mu, sigma, nu, tau):
        return -(_dldd(y, mu, sigma, nu, tau) * _dldm(y, mu, sigma, nu, tau))

    def d2ldmdv(y, mu, sigma, nu, tau):
        return -(_dldm(y, mu, sigma, nu, tau) * _dldv(y, mu, sigma, nu, tau))

    def d2ldmdt(y, mu, sigma, nu, tau):
        return -(_dldm(y, mu, sigma, nu, tau) * _dldt(y, mu, sigma, nu, tau))

    def d2ldddv(y, mu, sigma, nu, tau):
        return -(_dldd(y, mu, sigma, nu, tau) * _dldv(y, mu, sigma, nu, tau))

    def d2ldddt(y, mu, sigma, nu, tau):
        return -(_dldd(y, mu, sigma, nu, tau) * _dldt(y, mu, sigma, nu, tau))

    def d2ldvdt(y, mu, sigma, nu, tau):
        return -(_dldv(y, mu, sigma, nu, tau) * _dldt(y, mu, sigma, nu, tau))

    def _mean(mu, sigma, nu, tau):
        q = 1 / tau
        K1 = _sp.kv((q + 1) / 2, 0.25)
        K2 = _sp.kv((q - 1) / 2, 0.25)
        P = np.exp(1 / 4) / (8 * np.pi) ** (1 / 2) * (K1 + K2)
        return mu + sigma * np.sinh(nu / tau) * P

    def _variance(mu, sigma, nu, tau):
        q1 = 1 / tau
        K1 = _sp.kv((q1 + 1) / 2, 0.25)
        K2 = _sp.kv((q1 - 1) / 2, 0.25)
        P1 = np.exp(1 / 4) / (8 * np.pi) ** (1 / 2) * (K1 + K2)
        q2 = 2 / tau
        K3 = _sp.kv((q2 + 1) / 2, 0.25)
        K4 = _sp.kv((q2 - 1) / 2, 0.25)
        P2 = np.exp(1 / 4) / (8 * np.pi) ** (1 / 2) * (K3 + K4)
        return (sigma**2 / 2 * (np.cosh(2 * nu / tau) * P2 - 1)
                - sigma**2 * (np.sinh(nu / tau) * P1) ** 2)

    return GamlssFamily(
        family=("SHASHo", "Sinh-Arcsinh"),
        parameters={"mu": True, "sigma": True, "nu": True, "tau": True},
        nopar=4,
        type="Continuous",
        links={"mu": mstats, "sigma": dstats, "nu": vstats, "tau": tstats},
        derivatives={
            "dldm": dldm,
            "d2ldm2": d2ldm2,
            "dldd": dldd,
            "d2ldd2": d2ldd2,
            "dldv": dldv,
            "d2ldv2": d2ldv2,
            "dldt": dldt,
            "d2ldt2": d2ldt2,
            "d2ldmdd": d2ldmdd,
            "d2ldmdv": d2ldmdv,
            "d2ldmdt": d2ldmdt,
            "d2ldddv": d2ldddv,
            "d2ldddt": d2ldddt,
            "d2ldvdt": d2ldvdt,
        },
        G_dev_incr=lambda y, mu, sigma, nu, tau: -2 * dSHASHo(
            y, mu, sigma, nu, tau, log=True
        ),
        rqres={"pfun": "pSHASHo", "type": "Continuous"},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.full(len(y), np.std(y, ddof=1) / 5),
            "nu": lambda y: np.full(len(y), 0.5),
            "tau": lambda y: np.full(len(y), 0.5),
        },
        valid={
            "mu": lambda mu: True,
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
            "nu": lambda nu: True,
            "tau": lambda tau: bool(np.all(tau > 0)),
        },
        y_valid=lambda y: True,
        mean=_mean,
        variance=_variance,
    )


def dSHASHo(x, mu=0, sigma=1, nu=0, tau=1, log=False):
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(tau) < 0):
        raise ValueError("tau must be positive")
    x, mu, sigma, nu, tau = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float),
        np.asarray(sigma, float), np.asarray(nu, float),
        np.asarray(tau, float)
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z = (x - mu) / sigma
        c = np.cosh(tau * np.arcsinh(z) - nu)
        r = np.sinh(tau * np.arcsinh(z) - nu)
        loglik = (-np.log(sigma) + np.log(tau) - np.log(2 * np.pi) / 2
                  - np.log(1 + (z**2)) / 2 + np.log(c) - (r**2) / 2)
    return loglik if log else np.exp(loglik)


def pSHASHo(q, mu=0, sigma=1, nu=0, tau=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(tau) < 0):
        raise ValueError("tau must be positive")
    q, mu, sigma, nu, tau = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(mu, float),
        np.asarray(sigma, float), np.asarray(nu, float),
        np.asarray(tau, float)
    )
    z = (q - mu) / sigma
    r = np.sinh(tau * np.arcsinh(z) - nu)
    p = pNO(r)
    if not lower_tail:
        p = 1 - p
    if log_p:
        p = np.log(p)
    return p


def qSHASHo(p, mu=0, sigma=1, nu=0, tau=1, lower_tail=True, log_p=False):
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if np.any(p <= 0) or np.any(p >= 1):
        raise ValueError("p must be between 0 and 1")
    if not lower_tail:
        p = 1 - p
    p, mu, sigma, nu, tau = np.broadcast_arrays(
        p, np.asarray(mu, float), np.asarray(sigma, float),
        np.asarray(nu, float), np.asarray(tau, float)
    )
    y = mu + sigma * np.sinh((1 / tau) * np.arcsinh(_st.norm.ppf(p))
                             + (nu / tau))
    return y


def rSHASHo(n, mu=0, sigma=1, nu=0, tau=1, rng=None):
    if np.any(np.asarray(n) <= 0):
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng() if rng is None else rng
    p = rng.uniform(size=int(np.ceil(n)))
    return qSHASHo(p, mu=mu, sigma=sigma, nu=nu, tau=tau)
