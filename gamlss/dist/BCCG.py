"""Box-Cox Cole and Green (BCCG, BCCGo). Port of gamlss.dist R/BCCG.R."""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ..family import GamlssFamily, checklink


def _zofy(y, mu, sigma, nu):
    with np.errstate(divide="ignore", invalid="ignore"):
        z_pow = ((y / mu) ** nu - 1) / (nu * sigma)
        z_log = np.log(y / mu) / sigma
    return np.where(nu != 0, z_pow, z_log)


def _bccg_family(name, fullname, mu_link, sigma_link, nu_link):
    mstats = checklink("mu.link", "BC Cole Green", mu_link,
                       ("inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "BC Cole Green", sigma_link,
                       ("inverse", "log", "identity", "own"))
    vstats = checklink("nu.link", "BC Cole Green", nu_link,
                       ("inverse", "log", "identity", "own"))

    def dldm(y, mu, sigma, nu):
        z = _zofy(y, mu, sigma, nu)
        return ((z / sigma) + nu * (z * z - 1)) / mu

    def d2ldm2(y, mu, sigma, nu):
        return -(1 + 2 * nu * nu * sigma * sigma) / (mu * mu * sigma * sigma)

    def dldd(y, mu, sigma, nu):
        z = _zofy(y, mu, sigma, nu)
        h = _st.norm.pdf(1 / (sigma * np.abs(nu))) / _st.norm.cdf(
            1 / (sigma * np.abs(nu))
        )
        return (z**2 - 1) / sigma + h / (sigma**2 * np.abs(nu))

    def dldv(y, mu, sigma, nu):
        z = _zofy(y, mu, sigma, nu)
        h = _st.norm.pdf(1 / (sigma * np.abs(nu))) / _st.norm.cdf(
            1 / (sigma * np.abs(nu))
        )
        l = np.log(y / mu)
        out = (z - (l / sigma)) * (z / nu) - l * (z * z - 1)
        return out + np.sign(nu) * h / (sigma * nu**2)

    return GamlssFamily(
        family=(name, fullname),
        parameters={"mu": True, "sigma": True, "nu": True},
        nopar=3,
        type="Continuous",
        links={"mu": mstats, "sigma": dstats, "nu": vstats},
        derivatives={
            "dldm": dldm,
            "d2ldm2": d2ldm2,
            "dldd": dldd,
            "d2ldd2": lambda sigma: -2 / (sigma**2),
            "dldv": dldv,
            "d2ldv2": lambda sigma: -7 * sigma * sigma / 4,
            "d2ldmdd": lambda mu, sigma, nu: -2 * nu / (mu * sigma),
            "d2ldmdv": lambda mu: 1 / (2 * mu),
            "d2ldddv": lambda sigma, nu: -sigma * nu,
        },
        G_dev_incr=lambda y, mu, sigma, nu: -2 * dBCCG(y, mu=mu, sigma=sigma,
                                                       nu=nu, log=True),
        rqres={"pfun": "p" + name, "type": "Continuous"},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.full(len(y), 0.1),
            "nu": lambda y: np.full(len(y), 0.5),
        },
        valid={
            "mu": lambda mu: True,
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
            "nu": lambda nu: True,
        },
        y_valid=lambda y: bool(np.all(y > 0)),
    )


def BCCG(mu_link="identity", sigma_link="log", nu_link="identity"):
    return _bccg_family("BCCG", "Box-Cox-Cole-Green",
                        mu_link, sigma_link, nu_link)


def BCCGo(mu_link="log", sigma_link="log", nu_link="identity"):
    return _bccg_family("BCCGo", "Box-Cox-Cole-Green-orig.",
                        mu_link, sigma_link, nu_link)


def dBCCG(x, mu=1, sigma=0.1, nu=1, log=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    x, mu, sigma, nu = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float),
        np.asarray(sigma, float), np.asarray(nu, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        z = _zofy(x, mu, sigma, nu)
        loglik = (nu * np.log(x / mu) - np.log(sigma) - (z * z) / 2
                  - np.log(x) - np.log(2 * np.pi) / 2)
        loglik = loglik - np.log(
            _st.norm.cdf(1 / (sigma * np.abs(nu)))
        )
    ft = loglik if log else np.exp(loglik)
    return np.where(x <= 0, 0.0, ft)


def pBCCG(q, mu=1, sigma=0.1, nu=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    q, mu, sigma, nu = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(mu, float),
        np.asarray(sigma, float), np.asarray(nu, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        z = _zofy(q, mu, sigma, nu)
    FYy1 = _st.norm.cdf(z)
    FYy2 = np.where(nu > 0, _st.norm.cdf(-1 / (sigma * np.abs(nu))), 0.0)
    FYy3 = _st.norm.cdf(1 / (sigma * np.abs(nu)))
    FYy = (FYy1 - FYy2) / FYy3
    if not lower_tail:
        FYy = 1 - FYy
    if log_p:
        FYy = np.log(FYy)
    return FYy


def qBCCG(p, mu=1, sigma=0.1, nu=1, lower_tail=True, log_p=False):
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
    p, mu, sigma, nu = np.broadcast_arrays(
        p, np.asarray(mu, float), np.asarray(sigma, float),
        np.asarray(nu, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(
            nu <= 0,
            _st.norm.ppf(p * _st.norm.cdf(1 / (sigma * np.abs(nu)))),
            _st.norm.ppf(1 - (1 - p) * _st.norm.cdf(1 / (sigma * np.abs(nu)))),
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        ya = np.where(
            nu != 0,
            mu * (nu * sigma * z + 1) ** (1 / np.where(nu == 0, 1, nu)),
            mu * np.exp(sigma * z),
        )
    return ya


def rBCCG(n, mu=1, sigma=0.1, nu=1, rng=None):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    rng = np.random.default_rng() if rng is None else rng
    p = rng.uniform(size=int(np.ceil(n)))
    return qBCCG(p, mu=mu, sigma=sigma, nu=nu)


dBCCGo = dBCCG
pBCCGo = pBCCG
qBCCGo = qBCCG
rBCCGo = rBCCG
