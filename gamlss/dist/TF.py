"""t Family distribution (TF). Port of gamlss.dist R/TF.R."""

from __future__ import annotations

import numpy as np
from scipy import special as _sp
from scipy import stats as _st

from ..family import GamlssFamily, checklink
from .NO import dNO, pNO, qNO


def TF(mu_link="identity", sigma_link="log", nu_link="log"):
    mstats = checklink("mu.link", "t Family", mu_link,
                       ("inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "t Family", sigma_link,
                       ("inverse", "log", "identity", "own"))
    vstats = checklink("nu.link", "t Family", nu_link,
                       ("inverse", "log", "identity", "own"))

    def dldm(y, mu, sigma, nu):
        s2 = sigma**2
        dsq = ((y - mu) ** 2) / s2
        omega = (nu + 1) / (nu + dsq)
        return (omega * (y - mu)) / s2

    def dldd(y, mu, sigma, nu):
        s2 = sigma**2
        dsq = ((y - mu) ** 2) / s2
        omega = (nu + 1) / (nu + dsq)
        return (omega * dsq - 1) / sigma

    def dldv(y, mu, sigma, nu):
        s2 = sigma**2
        dsq = ((y - mu) ** 2) / s2
        omega = (nu + 1) / (nu + dsq)
        dsq3 = 1 + (dsq / nu)
        v2 = nu / 2
        v3 = (nu + 1) / 2
        out = (-np.log(dsq3) + (omega * dsq - 1) / nu
               + _sp.digamma(v3) - _sp.digamma(v2))
        return out / 2

    def d2ldv2(y, mu, sigma, nu):
        v2 = nu / 2
        v3 = (nu + 1) / 2
        out = (_sp.polygamma(1, v3) - _sp.polygamma(1, v2)
               + (2 * (nu + 5)) / (nu * (nu + 1) * (nu + 3)))
        out = out / 4
        return np.where(out < -1e-15, out, -1e-15)

    return GamlssFamily(
        family=("TF", "t Family"),
        parameters={"mu": True, "sigma": True, "nu": True},
        nopar=3,
        type="Continuous",
        links={"mu": mstats, "sigma": dstats, "nu": vstats},
        derivatives={
            "dldm": dldm,
            "d2ldm2": lambda sigma, nu: -(nu + 1) / ((nu + 3) * (sigma**2)),
            "dldd": dldd,
            "d2ldd2": lambda sigma, nu: -(2 * nu) / ((nu + 3) * sigma**2),
            "dldv": dldv,
            "d2ldv2": d2ldv2,
            "d2ldmdd": lambda y: np.zeros(len(np.asarray(y))),
            "d2ldmdv": lambda y: np.zeros(len(np.asarray(y))),
            "d2ldddv": lambda sigma, nu: 2 / (sigma * (nu + 3) * (nu + 1)),
        },
        G_dev_incr=lambda y, mu, sigma, nu: -2 * dTF(y, mu, sigma, nu,
                                                     log=True),
        rqres={"pfun": "pTF", "type": "Continuous"},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.full(len(y), np.std(y, ddof=1)),
            "nu": lambda y: np.full(len(y), 10.0),
        },
        valid={
            "mu": lambda mu: True,
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
            "nu": lambda nu: bool(np.all(nu > 0)),
        },
        y_valid=lambda y: True,
        mean=lambda mu, sigma, nu: np.where(nu > 1, mu, np.nan),
        variance=lambda mu, sigma, nu: np.where(
            nu > 2, (sigma**2 * nu) / (nu - 2), np.inf
        ),
    )


def dTF(x, mu=0, sigma=1, nu=10, log=False):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(nu) <= 0):
        raise ValueError("nu must be positive")
    x, mu, sigma, nu = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float),
        np.asarray(sigma, float), np.asarray(nu, float)
    )
    lik = np.where(
        nu > 1000000,
        dNO(x, mu=mu, sigma=sigma, log=False),
        (1 / sigma) * _st.t.pdf((x - mu) / sigma, df=nu),
    )
    return np.log(lik) if log else lik


def pTF(q, mu=0, sigma=1, nu=10, lower_tail=True, log_p=False):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(nu) <= 0):
        raise ValueError("nu must be positive")
    q, mu, sigma, nu = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(mu, float),
        np.asarray(sigma, float), np.asarray(nu, float)
    )
    z = (q - mu) / sigma
    if lower_tail:
        tcdf = _st.t.logcdf(z, nu) if log_p else _st.t.cdf(z, nu)
    else:
        tcdf = _st.t.logsf(z, nu) if log_p else _st.t.sf(z, nu)
    return np.where(
        nu > 1000000,
        pNO(q, mu=mu, sigma=sigma, lower_tail=lower_tail, log_p=log_p),
        tcdf,
    )


def qTF(p, mu=0, sigma=1, nu=10, lower_tail=True, log_p=False):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(nu) <= 0):
        raise ValueError("nu must be positive")
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be between 0 and 1")
    pp = p if lower_tail else 1 - p
    p_, mu, sigma, nu = np.broadcast_arrays(
        np.asarray(pp, float), np.asarray(mu, float),
        np.asarray(sigma, float), np.asarray(nu, float)
    )
    return np.where(
        nu > 1000000,
        qNO(p_, mu=mu, sigma=sigma),
        mu + sigma * _st.t.ppf(p_, nu),
    )


def rTF(n, mu=0, sigma=1, nu=10, rng=None):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(nu) <= 0):
        raise ValueError("nu must be positive")
    rng = np.random.default_rng() if rng is None else rng
    p = rng.uniform(size=int(np.ceil(n)))
    return qTF(p, mu=mu, sigma=sigma, nu=nu)
