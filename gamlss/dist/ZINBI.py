"""Zero inflated negative binomial type I distribution (ZINBI). Port of
gamlss.dist R/ZINBI.R."""

from __future__ import annotations

import numpy as np

from ..family import GamlssFamily, checklink
from .NBI import _dNBI_at0, _nbi_dldd, _nbi_dldm, dNBI, pNBI, qNBI

# NOTE: R's dNBI(0, mu, sigma) with vector mu/sigma returns a length-1
# result (see _dNBI_at0), so in the derivatives below the dNBI(0,...)
# factor is the first observation's value recycled over all
# observations, exactly as in the R fit.


def _zinbi_dldm(y, mu, sigma, nu):
    dldm0 = ((1 - nu) * ((nu + (1 - nu) * _dNBI_at0(mu, sigma)) ** (-1))
             * _dNBI_at0(mu, sigma) * _nbi_dldm(0, mu, sigma))
    dldm = np.where(y == 0, dldm0, _nbi_dldm(y, mu, sigma))
    return dldm


def _zinbi_dldd(y, mu, sigma, nu):
    dldd0 = ((1 - nu) * ((nu + (1 - nu) * _dNBI_at0(mu, sigma)) ** (-1))
             * _dNBI_at0(mu, sigma) * _nbi_dldd(0, mu, sigma))
    dldd = np.where(y == 0, dldd0, _nbi_dldd(y, mu, sigma))
    return dldd


def _zinbi_dldv(y, mu, sigma, nu):
    dldv0 = (((nu + (1 - nu) * _dNBI_at0(mu, sigma)) ** (-1))
             * (1 - _dNBI_at0(mu, sigma)))
    dldv = np.where(y == 0, dldv0, -1 / (1 - nu))
    return dldv


def ZINBI(mu_link="log", sigma_link="log", nu_link="logit"):
    mstats = checklink("mu.link", "ZINBI", mu_link,
                       ("inverse", "log", "identity"))
    dstats = checklink("sigma.link", "ZINBI", sigma_link,
                       ("inverse", "log", "identity"))
    vstats = checklink("nu.link", "ZINBI", nu_link,
                       ("logit", "probit", "cloglog", "log", "own"))

    def d2ldm2(y, mu, sigma, nu):
        dldm = _zinbi_dldm(y, mu, sigma, nu)
        d2ldm2 = -dldm * dldm
        d2ldm2 = np.where(d2ldm2 < -1e-15, d2ldm2, -1e-15)
        return d2ldm2

    def d2ldd2(y, mu, sigma, nu):
        dldd = _zinbi_dldd(y, mu, sigma, nu)
        d2ldd2 = -dldd**2
        d2ldd2 = np.where(d2ldd2 < -1e-15, d2ldd2, -1e-15)
        return d2ldd2

    def d2ldv2(y, mu, sigma, nu):
        dldv = _zinbi_dldv(y, mu, sigma, nu)
        d2ldv2 = -dldv**2
        d2ldv2 = np.where(d2ldv2 < -1e-15, d2ldv2, -1e-15)
        return d2ldv2

    def d2ldmdd(y, mu, sigma, nu):
        dldm = _zinbi_dldm(y, mu, sigma, nu)
        dldd = _zinbi_dldd(y, mu, sigma, nu)
        d2ldm2 = -dldm * dldd
        return d2ldm2

    def d2ldmdv(y, mu, sigma, nu):
        dldm = _zinbi_dldm(y, mu, sigma, nu)
        dldv = _zinbi_dldv(y, mu, sigma, nu)
        d2ldmdv = -dldm * dldv
        return d2ldmdv

    def d2ldddv(y, mu, sigma, nu):
        dldd = _zinbi_dldd(y, mu, sigma, nu)
        dldv = _zinbi_dldv(y, mu, sigma, nu)
        d2ldddv = -dldd * dldv
        return d2ldddv

    return GamlssFamily(
        family=("ZINBI", "Zero inflated negative binomial type I"),
        parameters={"mu": True, "sigma": True, "nu": True},
        nopar=3,
        type="Discrete",
        links={"mu": mstats, "sigma": dstats, "nu": vstats},
        derivatives={
            "dldm": _zinbi_dldm,
            "d2ldm2": d2ldm2,
            "dldd": _zinbi_dldd,
            "d2ldd2": d2ldd2,
            "dldv": _zinbi_dldv,
            "d2ldv2": d2ldv2,
            "d2ldmdd": d2ldmdd,
            "d2ldmdv": d2ldmdv,
            "d2ldddv": d2ldddv,
        },
        G_dev_incr=lambda y, mu, sigma, nu: -2 * dZINBI(
            y, mu=mu, sigma=sigma, nu=nu, log=True
        ),
        rqres={"pfun": "pZINBI", "type": "Discrete", "ymin": 0},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.full(
                len(y),
                max((np.var(y, ddof=1) - np.mean(y)) / (np.mean(y) ** 2), 0.1),
            ),
            "nu": lambda y: np.full(
                len(y), ((np.sum(y == 0) / len(y)) + 0.01) / 2
            ),
        },
        valid={
            "mu": lambda mu: bool(np.all(mu > 0)),
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
            "nu": lambda nu: bool(np.all((nu > 0) & (nu < 1))),
        },
        y_valid=lambda y: bool(np.all(y >= 0)),
        mean=lambda mu, sigma, nu: (1 - nu) * mu,
        variance=lambda mu, sigma, nu: (mu * (1 - nu)
                                        + mu**2 * (1 - nu) * (sigma + nu)),
    )


def dZINBI(x, mu=1, sigma=1, nu=0.3, log=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    if np.any(np.asarray(nu) <= 0) or np.any(np.asarray(nu) >= 1):
        raise ValueError("nu must be between 0 and 1")
    x, mu, sigma, nu = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float),
        np.asarray(sigma, float), np.asarray(nu, float)
    )
    fy = dNBI(x, mu=mu, sigma=sigma, log=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        logfy = np.where(
            x == 0, np.log(nu + (1 - nu) * np.exp(fy)), np.log(1 - nu) + fy
        )
    fy2 = logfy if log else np.exp(logfy)
    fy2 = np.where(x < 0, 0, fy2)
    return fy2


def pZINBI(q, mu=1, sigma=1, nu=0.3, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    if np.any(np.asarray(nu) <= 0) or np.any(np.asarray(nu) >= 1):
        raise ValueError("nu must be between 0 and 1")
    q, mu, sigma, nu = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(mu, float),
        np.asarray(sigma, float), np.asarray(nu, float)
    )
    cdf = pNBI(q, mu=mu, sigma=sigma)
    cdf = nu + (1 - nu) * cdf
    if not lower_tail:
        cdf = 1 - cdf
    if log_p:
        with np.errstate(divide="ignore", invalid="ignore"):
            cdf = np.log(cdf)
    cdf = np.where(q < 0, 0, cdf)
    return cdf


def qZINBI(p, mu=1, sigma=1, nu=0.3, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    if np.any(np.asarray(nu) <= 0) or np.any(np.asarray(nu) >= 1):
        raise ValueError("nu must be between 0 and 1")
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be between 0 and 1")
    if not lower_tail:
        p = 1 - p
    p, mu, sigma, nu = np.broadcast_arrays(
        p, np.asarray(mu, float), np.asarray(sigma, float),
        np.asarray(nu, float)
    )
    pnew = (p - nu) / (1 - nu) - (1e-7)  # added 28-2-17
    pnew = np.where(pnew > 0, pnew, 0)
    q = qNBI(pnew, mu=mu, sigma=sigma)
    return q


def rZINBI(n, mu=1, sigma=1, nu=0.3, rng=None):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be greater than 0")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be greater than 0")
    if np.any(np.asarray(nu) <= 0) or np.any(np.asarray(nu) >= 1):
        raise ValueError("nu must be between 0 and 1")
    if np.any(np.asarray(n) <= 0):
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng() if rng is None else rng
    n = int(np.ceil(n))
    p = rng.uniform(size=n)
    r = qZINBI(p, mu=mu, sigma=sigma, nu=nu)
    return r.astype(int)
