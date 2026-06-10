"""Johnson SU original distribution (JSUo). Port of gamlss.dist R/JSUo.R."""

from __future__ import annotations

import numpy as np

from ..family import GamlssFamily, checklink
from .NO import pNO, qNO


def _dldm(y, mu, sigma, nu, tau):
    z = (y - mu) / sigma
    r = nu + tau * np.arcsinh(z)
    return (z / (sigma * (z * z + 1))
            + ((r * tau) / (sigma * (z * z + 1) ** 0.5)))


def _dldd(y, mu, sigma, nu, tau):
    z = (y - mu) / sigma
    r = nu + tau * np.arcsinh(z)
    return (-1 / (sigma * (z * z + 1))
            + ((r * tau * z) / (sigma * (z * z + 1) ** 0.5)))


def _dldv(y, mu, sigma, nu, tau):
    z = (y - mu) / sigma
    r = nu + tau * np.arcsinh(z)
    return -r


def _dldt(y, mu, sigma, nu, tau):
    z = (y - mu) / sigma
    r = nu + tau * np.arcsinh(z)
    return (1 + r * nu - r * r) / tau


def JSUo(mu_link="identity", sigma_link="log", nu_link="identity",
         tau_link="log"):
    mstats = checklink("mu.link", "Johnson SU", mu_link,
                       ("inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "Johnson SU", sigma_link,
                       ("inverse", "log", "identity", "own"))
    vstats = checklink("nu.link", "Johnson SU", nu_link,
                       ("inverse", "log", "identity", "own"))
    tstats = checklink("tau.link", "Johnson SU", tau_link,
                       ("inverse", "log", "identity", "own"))

    def dldm(y, mu, sigma, nu, tau):
        return _dldm(y, mu, sigma, nu, tau)

    def d2ldm2(y, mu, sigma, nu, tau):
        dldm = _dldm(y, mu, sigma, nu, tau)
        d2ldm2 = -dldm * dldm
        d2ldm2 = np.where(d2ldm2 < -1e-15, d2ldm2, -1e-15)
        return d2ldm2

    def dldd(y, mu, sigma, nu, tau):
        return _dldd(y, mu, sigma, nu, tau)

    def d2ldd2(y, mu, sigma, nu, tau):
        dldd = _dldd(y, mu, sigma, nu, tau)
        d2ldd2 = -dldd * dldd
        d2ldd2 = np.where(d2ldd2 < -1e-15, d2ldd2, -1e-15)
        return d2ldd2

    def dldv(y, mu, sigma, nu, tau):
        return _dldv(y, mu, sigma, nu, tau)

    def d2ldv2(y, mu, sigma, nu, tau):
        z = (y - mu) / sigma
        r = nu + tau * np.arcsinh(z)
        d2ldv2 = -r**2
        d2ldv2 = np.where(d2ldv2 < -1e-15, d2ldv2, -1e-15)
        return d2ldv2

    def dldt(y, mu, sigma, nu, tau):
        return _dldt(y, mu, sigma, nu, tau)

    def d2ldt2(y, mu, sigma, nu, tau):
        dldt = _dldt(y, mu, sigma, nu, tau)
        d2ldt2 = -dldt * dldt
        d2ldt2 = np.where(d2ldt2 < -1e-15, d2ldt2, -1e-15)
        return d2ldt2

    def d2ldmdd(y, mu, sigma, nu, tau):
        return -(_dldm(y, mu, sigma, nu, tau) * _dldd(y, mu, sigma, nu, tau))

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

    return GamlssFamily(
        family=("JSUo", "Johnson SU original"),
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
        G_dev_incr=lambda y, mu, sigma, nu, tau: -2 * dJSUo(
            y, mu, sigma, nu, tau, log=True
        ),
        rqres={"pfun": "pJSUo", "type": "Continuous"},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.full(len(y), 0.1),
            "nu": lambda y: np.zeros(len(y)),
            "tau": lambda y: np.full(len(y), 0.5),
        },
        valid={
            "mu": lambda mu: True,
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
            "nu": lambda nu: True,
            "tau": lambda tau: bool(np.all(tau > 0)),
        },
        y_valid=lambda y: True,
        mean=lambda mu, sigma, nu, tau: mu - sigma
        * np.sqrt(np.exp(1 / tau**2)) * np.sinh(nu / tau),
        variance=lambda mu, sigma, nu, tau: 0.5 * sigma**2
        * (np.exp(1 / tau**2) - 1)
        * (np.exp(1 / tau**2) * np.cosh(2 * nu / tau) + 1),
    )


def dJSUo(x, mu=0, sigma=1, nu=0, tau=1, log=False):
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(tau) < 0):
        raise ValueError("tau must be positive")
    x, mu, sigma, nu, tau = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float),
        np.asarray(sigma, float), np.asarray(nu, float),
        np.asarray(tau, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x - mu) / sigma
        r = nu + tau * np.arcsinh(z)
        loglik = (-np.log(sigma) + np.log(tau) - 0.5 * np.log(z * z + 1)
                  - 0.5 * np.log(2 * np.pi) - 0.5 * r * r)
    return loglik if log else np.exp(loglik)


def pJSUo(q, mu=0, sigma=1, nu=0, tau=1, lower_tail=True, log_p=False):
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
    r = nu + tau * np.arcsinh(z)
    p = pNO(r, 0, 1)
    if not lower_tail:
        p = 1 - p
    if log_p:
        p = np.log(p)
    return p


def qJSUo(p, mu=0, sigma=1, nu=0, tau=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(tau) < 0):
        raise ValueError("tau must be positive")
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
    r = qNO(p, 0, 1)
    z = np.sinh((r - nu) / tau)
    q = mu + sigma * z
    return q


def rJSUo(n, mu=0, sigma=1, nu=0, tau=1, rng=None):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    rng = np.random.default_rng() if rng is None else rng
    p = rng.uniform(size=int(np.ceil(n)))
    return qJSUo(p, mu=mu, sigma=sigma, nu=nu, tau=tau)
