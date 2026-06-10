"""Box-Cox Power Exponential (BCPE, BCPEo). Port of gamlss.dist
R/BCPE.R."""

from __future__ import annotations

import numpy as np
from scipy import special as _sp
from scipy import stats as _st

from ..family import GamlssFamily, checklink
from .BCCG import _zofy


def _logc(tau):
    return 0.5 * (-(2 / tau) * np.log(2) + _sp.gammaln(1 / tau)
                  - _sp.gammaln(3 / tau))


def _fT(t, tau, log=False):
    log_c = _logc(tau)
    c = np.exp(log_c)
    loglik = (np.log(tau) - log_c - (0.5 * (np.abs(t / c) ** tau))
              - (1 + (1 / tau)) * np.log(2) - _sp.gammaln(1 / tau))
    return loglik if log else np.exp(loglik)


def _FT(t, tau):
    c = np.exp(_logc(tau))
    s = 0.5 * ((np.abs(t / c)) ** tau)
    Fs = _st.gamma.cdf(s, 1 / tau, scale=1)
    return 0.5 * (1 + Fs * np.sign(t))


def _qT(p, tau):
    c = np.exp(_logc(tau))
    with np.errstate(invalid="ignore"):
        s = _st.gamma.ppf((2 * p - 1) * np.sign(p - 0.5), 1 / tau, scale=1)
    return np.sign(p - 0.5) * ((2 * s) ** (1 / tau)) * c


def _bcpe_family(name, fullname, mu_link, sigma_link, nu_link, tau_link):
    mstats = checklink("mu.link", "Box Cox Power Exponential", mu_link,
                       ("inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "Box Cox Power Exponential", sigma_link,
                       ("inverse", "log", "identity", "own"))
    vstats = checklink("nu.link", "Box Cox Power Exponential", nu_link,
                       ("inverse", "log", "identity", "own"))
    tstats = checklink("tau.link", "Box Cox Power Exponential", tau_link,
                       ("logshiftto1", "log", "identity", "own"))

    def dldm(y, mu, sigma, nu, tau):
        z = _zofy(y, mu, sigma, nu)
        c = np.exp(_logc(tau))
        return (tau / (2 * mu * sigma * c**2)) * (z + sigma * nu * z**2) * (
            (np.abs(z / c)) ** (tau - 2)
        ) - (nu / mu)

    def d2ldm2(y, mu, sigma, nu, tau):
        z = _zofy(y, mu, sigma, nu)
        c = np.exp(_logc(tau))
        dldm = (tau / (2 * mu * sigma * c**2)) * (z + sigma * nu * z**2) * (
            (np.abs(z / c)) ** (tau - 2)
        ) - (nu / mu)
        out = -((tau * tau) * _sp.gamma(2 - (1 / tau)) * _sp.gamma(3 / tau)
                ) / (mu**2 * sigma**2 * _sp.gamma(1 / tau) ** 2)
        out = out - (tau * nu**2) / mu**2
        # R: a global (not element-wise) switch
        if np.any(tau < 1.05):
            return -dldm * dldm
        return out

    def dldd(y, mu, sigma, nu, tau):
        z = _zofy(y, mu, sigma, nu)
        h = _fT(1 / (sigma * np.abs(nu)), tau) / _FT(
            1 / (sigma * np.abs(nu)), tau
        )
        return (1 / sigma) * ((tau / 2) * (np.abs(z / np.exp(_logc(tau))))
                              ** tau - 1) + h / (sigma**2 * np.abs(nu))

    def dldv(y, mu, sigma, nu, tau):
        z = _zofy(y, mu, sigma, nu)
        c = np.exp(_logc(tau))
        h = _fT(1 / (sigma * np.abs(nu)), tau) / _FT(
            1 / (sigma * np.abs(nu)), tau
        )
        out = -(tau / (2 * nu * c**2)) * ((np.abs(z / c)) ** (tau - 2)) * z \
            * ((nu * z + 1 / sigma) * np.log(y / mu) - z)
        return out + np.log(y / mu) + np.sign(nu) * h / (sigma * nu**2)

    def dldt(y, mu, sigma, nu, tau):
        z = _zofy(y, mu, sigma, nu)
        c = np.exp(_logc(tau))
        dlogc_dt = (1 / (2 * tau**2)) * (
            2 * np.log(2) - _sp.digamma(1 / tau) + 3 * _sp.digamma(3 / tau)
        )
        j = (np.log(_FT(1 / (sigma * np.abs(nu)), tau + 0.001))
             - np.log(_FT(1 / (sigma * np.abs(nu)), tau))) / 0.001
        out = ((1 / tau) - 0.5 * np.log(np.abs(z / c))
               * (np.abs(z / c)) ** tau
               + (1 / tau**2) * (np.log(2) + _sp.digamma(1 / tau))
               + ((tau / 2) * ((np.abs(z / c)) ** tau) - 1) * dlogc_dt - j)
        return out

    def d2ldt2(y, mu, sigma, nu, tau):
        p = (tau + 1) / tau
        dlogc_dt = (1 / (2 * tau**2)) * (
            2 * np.log(2) - _sp.digamma(1 / tau) + 3 * _sp.digamma(3 / tau)
        )
        part1 = p * _sp.polygamma(1, p) + 2 * _sp.digamma(p) ** 2
        part2 = _sp.digamma(p) * (np.log(2) + 3 - 3 * _sp.digamma(3 / tau)
                                  - tau)
        part3 = -3 * _sp.digamma(3 / tau) * (1 + np.log(2))
        part4 = -(tau + np.log(2)) * np.log(2)
        part5 = -tau + (tau**4) * dlogc_dt**2
        out = part1 + part2 + part3 + part4 + part5
        out = -out / tau**3
        return np.where(out < -1e-15, out, -1e-15)

    return GamlssFamily(
        family=(name, fullname),
        parameters={"mu": True, "sigma": True, "nu": True, "tau": True},
        nopar=4,
        type="Continuous",
        links={"mu": mstats, "sigma": dstats, "nu": vstats, "tau": tstats},
        derivatives={
            "dldm": dldm,
            "d2ldm2": d2ldm2,
            "dldd": dldd,
            "d2ldd2": lambda sigma, tau: -tau / sigma**2,
            "dldv": dldv,
            "d2ldv2": lambda sigma, tau: -(sigma**2) * (3 * tau + 1) / 4,
            "dldt": dldt,
            "d2ldt2": d2ldt2,
            "d2ldmdd": lambda mu, sigma, nu, tau: -(nu * tau) / (mu * sigma),
            "d2ldmdv": lambda mu, sigma, nu, tau: (
                2 * (tau - 1) - (tau + 1) * (sigma**2) * (nu**2)
            ) / (4 * mu),
            "d2ldmdt": lambda mu, sigma, nu, tau: (nu / (mu * tau)) * (
                1 + tau + (3 / 2) * (_sp.digamma(1 / tau)
                                     - _sp.digamma(3 / tau))
            ),
            "d2ldddv": lambda sigma, nu, tau: -(sigma * nu * tau) / 2,
            "d2ldddt": lambda sigma, tau: (1 / (sigma * tau)) * (
                1 + tau + (3 / 2) * (_sp.digamma(1 / tau)
                                     - _sp.digamma(3 / tau))
            ),
            "d2ldvdt": lambda mu, sigma, nu, tau: (
                ((sigma**2) * nu) / (2 * tau)
            ) * (1 + (tau / 3) + 0.5 * (_sp.digamma(1 / tau)
                                        - _sp.digamma(3 / tau))),
        },
        G_dev_incr=lambda y, mu, sigma, nu, tau: -2 * dBCPE(
            y, mu, sigma, nu, tau, log=True
        ),
        rqres={"pfun": "p" + name, "type": "Continuous"},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.full(len(y), 0.1),
            "nu": lambda y: np.full(len(y), 1.0),
            "tau": lambda y: np.full(len(y), 2.0),
        },
        valid={
            "mu": lambda mu: bool(np.all(mu > 0)),
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
            "nu": lambda nu: True,
            "tau": lambda tau: bool(np.all(tau > 0)),
        },
        y_valid=lambda y: bool(np.all(y > 0)),
    )


def BCPE(mu_link="identity", sigma_link="log", nu_link="identity",
         tau_link="log"):
    return _bcpe_family("BCPE", "Box-Cox Power Exponential",
                        mu_link, sigma_link, nu_link, tau_link)


def BCPEo(mu_link="log", sigma_link="log", nu_link="identity",
          tau_link="log"):
    return _bcpe_family("BCPEo", "Box-Cox Power Exponential orig.",
                        mu_link, sigma_link, nu_link, tau_link)


def dBCPE(x, mu=5, sigma=0.1, nu=1, tau=2, log=False):
    if np.any(np.asarray(mu) < 0):
        raise ValueError("mu must be positive")
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
        z = _zofy(x, mu, sigma, nu)
        logfZ = _fT(z, tau, log=True) - np.log(
            _FT(1 / (sigma * np.abs(nu)), tau)
        )
        logder = (nu - 1) * np.log(x) - nu * np.log(mu) - np.log(sigma)
        loglik = logder + logfZ
    ft = loglik if log else np.exp(loglik)
    return np.where(x <= 0, 0.0, ft)


def pBCPE(q, mu=5, sigma=0.1, nu=1, tau=2, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) < 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(tau) < 0):
        raise ValueError("tau must be positive")
    q, mu, sigma, nu, tau = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(mu, float),
        np.asarray(sigma, float), np.asarray(nu, float),
        np.asarray(tau, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        z = _zofy(q, mu, sigma, nu)
    FYy1 = _FT(z, tau)
    FYy2 = np.where(nu > 0, _FT(-1 / (sigma * np.abs(nu)), tau), 0.0)
    FYy3 = _FT(1 / (sigma * np.abs(nu)), tau)
    FYy = (FYy1 - FYy2) / FYy3
    if not lower_tail:
        FYy = 1 - FYy
    if log_p:
        FYy = np.log(FYy)
    return FYy


def qBCPE(p, mu=5, sigma=0.1, nu=1, tau=2, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) < 0):
        raise ValueError("mu must be positive")
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
    za = np.where(
        nu < 0,
        _qT(p * _FT(1 / (sigma * np.abs(nu)), tau), tau),
        np.where(
            nu == 0,
            _qT(p, tau),
            _qT(1 - (1 - p) * _FT(1 / (sigma * np.abs(nu)), tau), tau),
        ),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ya = np.where(
            nu != 0,
            mu * (nu * sigma * za + 1) ** (1 / np.where(nu == 0, 1, nu)),
            mu * np.exp(sigma * za),
        )
    return ya


def rBCPE(n, mu=5, sigma=0.1, nu=1, tau=2, rng=None):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(tau) <= 0):
        raise ValueError("tau must be positive")
    rng = np.random.default_rng() if rng is None else rng
    p = rng.uniform(size=int(np.ceil(n)))
    return qBCPE(p, mu=mu, sigma=sigma, nu=nu, tau=tau)


dBCPEo = dBCPE
pBCPEo = pBCPE
qBCPEo = qBCPE
rBCPEo = rBCPE
