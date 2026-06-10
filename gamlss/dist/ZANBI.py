"""Zero altered negative binomial type I distribution (ZANBI). Port of
gamlss.dist R/ZANBI.R."""

from __future__ import annotations

import numpy as np

from ..family import GamlssFamily, checklink
from .NBI import (_dNBI_at0, _nbi_dldd, _nbi_dldm, _pNBI_at0, dNBI, pNBI,
                  qNBI)

# NOTE: R's dNBI(0, mu, sigma) / pNBI(0, mu, sigma) with vector
# mu/sigma return a length-1 result (see _dNBI_at0/_pNBI_at0 in NBI),
# so below the dNBI(0,...) and pNBI(0,...) factors are the first
# observation's value recycled over all observations, exactly as in R.


def _zanbi_dldm(y, mu, sigma):
    dldm0 = (_nbi_dldm(y, mu, sigma)
             + _dNBI_at0(mu, sigma) * _nbi_dldm(0, mu, sigma)
             / (1 - _dNBI_at0(mu, sigma)))
    dldm = np.where(y == 0, 0, dldm0)
    return dldm


def _zanbi_dldd(y, mu, sigma):
    sigma = np.where(sigma < 0.000001, 0.000001, sigma)
    dldd0 = (_nbi_dldd(y, mu, sigma)
             + _dNBI_at0(mu, sigma) * _nbi_dldd(0, mu, sigma)
             / (1 - _dNBI_at0(mu, sigma)))
    dldd = np.where(y == 0, 0, dldd0)
    return dldd


def ZANBI(mu_link="log", sigma_link="log", nu_link="logit"):
    mstats = checklink("mu.link", "ZANBI", mu_link,
                       ("inverse", "log", "identity"))
    dstats = checklink("sigma.link", "ZANBI", sigma_link,
                       ("inverse", "log", "identity"))
    vstats = checklink("nu.link", "ZANBI", nu_link,
                       ("logit", "probit", "cloglog", "cauchit", "log",
                        "own"))

    def dldm(y, mu, sigma, nu):
        return _zanbi_dldm(y, mu, sigma)

    def d2ldm2(y, mu, sigma, nu):
        dldm = _zanbi_dldm(y, mu, sigma)
        d2ldm2 = -dldm * dldm
        # as in the R source: the clamped value is assigned to md2ldm2,
        # which is unused, and the unclamped d2ldm2 is returned
        md2ldm2 = np.where(d2ldm2 < -1e-15, d2ldm2, -1e-15)  # noqa: F841
        return d2ldm2

    def dldd(y, mu, sigma, nu):
        return _zanbi_dldd(y, mu, sigma)

    def d2ldd2(y, mu, sigma, nu):
        dldd = _zanbi_dldd(y, mu, sigma)
        d2ldd2 = -dldd**2
        d2ldd2 = np.where(d2ldd2 < -1e-10, d2ldd2, -1e-10)
        return d2ldd2

    def dldv(y, mu, sigma, nu):
        dldv = np.where(y == 0, 1 / nu, -1 / (1 - nu))
        return dldv

    def d2ldv2(y, mu, sigma, nu):
        d2ldv2 = -1 / (nu * (1 - nu))
        d2ldv2 = np.where(d2ldv2 < -1e-15, d2ldv2, -1e-15)
        return d2ldv2

    def d2ldmdd(y, mu, sigma, nu):
        sigma = np.where(sigma < 0.000001, 0.000001, sigma)
        dldm = np.where(
            y == 0, 0,
            _nbi_dldm(y, mu, sigma)
            + _dNBI_at0(mu, sigma) * _nbi_dldm(0, mu, sigma)
            / (1 - _dNBI_at0(mu, sigma)),
        )
        dldd = np.where(
            y == 0, 0,
            _nbi_dldd(y, mu, sigma)
            + _dNBI_at0(mu, sigma) * _nbi_dldd(0, mu, sigma)
            / (1 - _dNBI_at0(mu, sigma)),
        )
        d2ldm2 = -dldm * dldd
        return d2ldm2

    def _mean(mu, sigma, nu):
        c = (1 - nu) / (1 - (1 + mu * sigma) ** (-1 / sigma))
        return mu * c

    def _variance(mu, sigma, nu):
        c = (1 - nu) / (1 - (1 + mu * sigma) ** (-1 / sigma))
        return mu * c + c * mu**2 * (1 + sigma - c)

    return GamlssFamily(
        family=("ZANBI", "Zero Altered Negative binomial type I"),
        parameters={"mu": True, "sigma": True, "nu": True},
        nopar=3,
        type="Discrete",
        links={"mu": mstats, "sigma": dstats, "nu": vstats},
        derivatives={
            "dldm": dldm,
            "d2ldm2": d2ldm2,
            "dldd": dldd,
            "d2ldd2": d2ldd2,
            "dldv": dldv,
            "d2ldv2": d2ldv2,
            "d2ldmdd": d2ldmdd,
            "d2ldmdv": lambda y: 0,
            "d2ldddv": lambda y: 0,
        },
        G_dev_incr=lambda y, mu, sigma, nu: -2 * dZANBI(
            y, mu=mu, sigma=sigma, nu=nu, log=True
        ),
        rqres={"pfun": "pZANBI", "type": "Discrete", "ymin": 0},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.full(
                len(y),
                max((np.var(y, ddof=1) - np.mean(y)) / (np.mean(y) ** 2), 0.1),
            ),
            "nu": lambda y: np.full(
                len(y), max(np.sum(y == 0) / len(y), 0.01)
            ),
        },
        valid={
            "mu": lambda mu: bool(np.all(mu > 0)),
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
            "nu": lambda nu: bool(np.all((nu > 0) & (nu < 1))),
        },
        y_valid=lambda y: bool(np.all(y >= 0)),
        mean=_mean,
        variance=_variance,
    )


def dZANBI(x, mu=1, sigma=1, nu=0.3, log=False):
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
    # R: fy0 <- dNBI(0, mu, sigma, log = T) is truncated to length 1
    # by the ifelse() inside dNBI (first observation, recycled)
    fy0 = _dNBI_at0(mu, sigma, log=True)
    fy = dNBI(x, mu=mu, sigma=sigma, log=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        logfy = np.where(
            x == 0, np.log(nu),
            np.log(1 - nu) + fy - np.log(1 - np.exp(fy0))
        )
    fy2 = logfy if log else np.exp(logfy)
    fy2 = np.where(x < 0, 0, fy2)
    return fy2


def pZANBI(q, mu=1, sigma=1, nu=0.3, lower_tail=True, log_p=False):
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
    # R: cdf0 <- pNBI(0, mu, sigma) is truncated to length 1 by the
    # ifelse() inside pNBI (first observation, recycled)
    cdf0 = _pNBI_at0(mu, sigma)
    cdf1 = pNBI(q, mu=mu, sigma=sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        cdf3 = nu + ((1 - nu) * (cdf1 - cdf0) / (1 - cdf0))
    cdf = np.where(q == 0, nu, cdf3)
    if not lower_tail:
        cdf = 1 - cdf
    if log_p:
        with np.errstate(divide="ignore", invalid="ignore"):
            cdf = np.log(cdf)
    cdf = np.where(q < 0, 0, cdf)
    return cdf


def qZANBI(p, mu=1, sigma=1, nu=0.3, lower_tail=True, log_p=False):
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
    pnew = (p - nu) / (1 - nu) - 1e-10
    # R: pNBI(0, mu, sigma) is truncated to length 1 (first observation)
    cdf0 = _pNBI_at0(mu, sigma)
    pnew2 = cdf0 * (1 - pnew) + pnew
    pnew2 = np.where(pnew2 > 0, pnew2, 0)
    q = qNBI(pnew2, mu=mu, sigma=sigma)
    return q


def rZANBI(n, mu=1, sigma=1, nu=0.3, rng=None):
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
    r = qZANBI(p, mu=mu, sigma=sigma, nu=nu)
    return r.astype(int)
