"""Johnson SU distribution (JSU). Port of gamlss.dist R/JSU.R.

Reparameterised Johnson SU: mu is the mean and sigma the standard
deviation of the distribution.  The first derivative squares are used
for the expected second derivatives (numerical Fisher), as in R.
"""

from __future__ import annotations

import numpy as np

from ..family import GamlssFamily, checklink
from .NO import pNO, qNO


def _consts(y, mu, sigma, nu, tau):
    """The rtau/w/omega/c/z/r helper block computed inside every R
    derivative function of JSU."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rtau = 1 / tau
        w = np.where(rtau < 0.0000001, 1, np.exp(rtau**2))
        omega = -nu * rtau
        c = (0.5 * (w - 1) * (w * np.cosh(2 * omega) + 1)) ** (-0.5)
        z = (y - (mu + c * sigma * w**0.5 * np.sinh(omega))) / (c * sigma)
        r = -nu + np.arcsinh(z) / rtau
    return rtau, w, omega, c, z, r


def _dldm(y, mu, sigma, nu, tau):
    rtau, w, omega, c, z, r = _consts(y, mu, sigma, nu, tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        dldm = (z / (z * z + 1)
                + (r / (rtau * (z * z + 1) ** 0.5))) / (c * sigma)
    return dldm


def _dldd(y, mu, sigma, nu, tau):
    rtau, w, omega, c, z, r = _consts(y, mu, sigma, nu, tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        dldd = (z + w**0.5 * np.sinh(omega)) * (
            z / (z * z + 1) + (r / (rtau * (z * z + 1) ** 0.5))
        )
        dldd = (dldd - 1) / sigma
    return dldd


def _dldv(y, mu, sigma, nu, tau):
    rtau, w, omega, c, z, r = _consts(y, mu, sigma, nu, tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlogcdv = (rtau * w * np.sinh(2 * omega)) / (w * np.cosh(2 * omega) + 1)
        dzdv = (-(z + w**0.5 * np.sinh(omega)) * dlogcdv
                + (rtau * w**0.5 * np.cosh(omega)))
        dldv = (-dlogcdv
                - (z / (z * z + 1) + (r / (rtau * (z * z + 1) ** 0.5))) * dzdv
                + r)
    return dldv


def _dldt(y, mu, sigma, nu, tau):
    rtau, w, omega, c, z, r = _consts(y, mu, sigma, nu, tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlogcdt = -rtau * w * ((1 / (w - 1))
                               + ((np.cosh(2 * omega))
                                  / (w * np.cosh(2 * omega) + 1)))
        dlogcdt = dlogcdt + ((nu * w * np.sinh(2 * omega))
                             / (w * np.cosh(2 * omega) + 1))
        dzdt = (-(z + w**0.5 * np.sinh(omega)) * dlogcdt
                - rtau * w**0.5 * np.sinh(omega)
                + nu * w**0.5 * np.cosh(omega))
        dldt = (-dlogcdt - (1 / rtau)
                - (z / (z * z + 1)
                   + (r / (rtau * (z * z + 1) ** 0.5))) * dzdt
                + (r * (r + nu)) / rtau)
        dldt = -dldt * rtau * rtau
    return dldt


def JSU(mu_link="identity", sigma_link="log", nu_link="identity",
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
        dldv = _dldv(y, mu, sigma, nu, tau)
        d2ldv2 = -dldv * dldv
        d2ldv2 = np.where(d2ldv2 < -1e-4, d2ldv2, -1e-4)
        return d2ldv2

    def dldt(y, mu, sigma, nu, tau):
        return _dldt(y, mu, sigma, nu, tau)

    def d2ldt2(y, mu, sigma, nu, tau):
        dldt = _dldt(y, mu, sigma, nu, tau)
        d2ldt2 = -dldt * dldt
        d2ldt2 = np.where(d2ldt2 < -1e-4, d2ldt2, -1e-4)
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
        family=("JSU", "Johnson SU"),
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
        G_dev_incr=lambda y, mu, sigma, nu, tau: -2 * dJSU(
            y, mu, sigma, nu, tau, log=True
        ),
        rqres={"pfun": "pJSU", "type": "Continuous"},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.full(len(y), np.std(y, ddof=1) / 4),
            "nu": lambda y: np.zeros(len(y)),
            "tau": lambda y: np.ones(len(y)),
        },
        valid={
            "mu": lambda mu: True,
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
            "nu": lambda nu: True,
            "tau": lambda tau: bool(np.all(tau > 0)),
        },
        y_valid=lambda y: True,
        mean=lambda mu, sigma, nu, tau: mu,
        variance=lambda mu, sigma, nu, tau: sigma**2,
    )


def dJSU(x, mu=0, sigma=1, nu=1, tau=1, log=False):
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
        rtau = 1 / tau
        w = np.where(rtau < 0.0000001, 1, np.exp(rtau**2))
        omega = -nu * rtau
        c = (0.5 * (w - 1) * (w * np.cosh(2 * omega) + 1)) ** (-0.5)
        z = (x - (mu + c * sigma * w**0.5 * np.sinh(omega))) / (c * sigma)
        r = -nu + np.arcsinh(z) / rtau
        loglik = (-np.log(sigma) - np.log(c) - np.log(rtau)
                  - 0.5 * np.log(z * z + 1) - 0.5 * np.log(2 * np.pi)
                  - 0.5 * r * r)
    return loglik if log else np.exp(loglik)


def pJSU(q, mu=0, sigma=1, nu=1, tau=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(tau) < 0):
        raise ValueError("tau must be positive")
    q, mu, sigma, nu, tau = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(mu, float),
        np.asarray(sigma, float), np.asarray(nu, float),
        np.asarray(tau, float)
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rtau = 1 / tau
        w = np.where(rtau < 0.0000001, 1, np.exp(rtau**2))
        omega = -nu * rtau
        c = (0.5 * (w - 1) * (w * np.cosh(2 * omega) + 1)) ** (-0.5)
        z = (q - (mu + c * sigma * w**0.5 * np.sinh(omega))) / (c * sigma)
        r = -nu + np.arcsinh(z) / rtau
    p = pNO(r, 0, 1)
    if not lower_tail:
        p = 1 - p
    if log_p:
        p = np.log(p)
    return p


def qJSU(p, mu=0, sigma=1, nu=1, tau=1, lower_tail=True, log_p=False):
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
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rtau = 1 / tau
        r = qNO(p, 0, 1)
        z = np.sinh(rtau * (r + nu))
        w = np.where(rtau < 0.0000001, 1, np.exp(rtau**2))
        omega = -nu * rtau
        c = (0.5 * (w - 1) * (w * np.cosh(2 * omega) + 1)) ** (-0.5)
        q = (mu + c * sigma * w**0.5 * np.sinh(omega)) + c * sigma * z
    return q


def rJSU(n, mu=0, sigma=1, nu=1, tau=1, rng=None):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    rng = np.random.default_rng() if rng is None else rng
    p = rng.uniform(size=int(np.ceil(n)))
    return qJSU(p, mu=mu, sigma=sigma, nu=nu, tau=tau,
                lower_tail=True, log_p=False)
