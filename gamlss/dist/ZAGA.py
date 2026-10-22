"""Zero adjusted gamma distribution (ZAGA). Port of gamlss.dist R/ZAGA.R.

P(Y = 0) = nu; Y | Y > 0 ~ GA(mu, sigma), so E(y) = (1 - nu) mu and
Var(y) = (1 - nu) mu^2 (sigma^2 + nu). Type "Mixed": point mass at 0.
"""

from __future__ import annotations

import numpy as np
from scipy import special as _sp
from scipy import stats as _st

from ..family import GamlssFamily, checklink


NU_LINKS = ("logit", "probit", "cloglog", "cauchit", "log", "own")


def _zero(y):
    return np.zeros(len(np.asarray(y)))


def _check(mu, sigma, nu, strict=False):
    bad = (lambda a: np.any(np.asarray(a) <= 0)) if strict else (lambda a: np.any(np.asarray(a) < 0))
    if bad(mu):
        raise ValueError("mu must be positive")
    if bad(sigma):
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(nu) < 0) or np.any(np.asarray(nu) > 1):
        raise ValueError("nu must be between 0 and 1")


def ZAGA(mu_link="log", sigma_link="log", nu_link="logit"):
    mstats = checklink("mu.link", "ZAGA", mu_link, ("inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "ZAGA", sigma_link, ("inverse", "log", "identity", "own"))
    vstats = checklink("nu.link", "ZAGA", nu_link, NU_LINKS)

    def dldd(y, mu, sigma):
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (2 / sigma**3) * ((y / mu) - np.log(y) + np.log(mu) + np.log(sigma**2) - 1
                                  + _sp.digamma(1 / (sigma**2)))
        return np.where(y == 0, 0.0, v)

    return GamlssFamily(
        family=("ZAGA", "Zero Adjusted GA"),
        parameters={"mu": True, "sigma": True, "nu": True},
        nopar=3,
        type="Mixed",
        links={"mu": mstats, "sigma": dstats, "nu": vstats},
        derivatives={
            "dldm": lambda y, mu, sigma: np.where(y == 0, 0.0, (y - mu) / ((sigma**2) * (mu**2))),
            "d2ldm2": lambda y, mu, sigma: np.where(y == 0, 0.0, -1 / ((sigma**2) * (mu**2))),
            "dldd": dldd,
            "d2ldd2": lambda y, sigma: np.where(
                y == 0, 0.0, (4 / sigma**4) - (4 / sigma**6) * _sp.polygamma(1, 1 / sigma**2)),
            "dldv": lambda y, nu: np.where(y == 0, 1 / nu, -1 / (1 - nu)),
            "d2ldv2": lambda nu: -1 / (nu * (1 - nu)),
            "d2ldmdd": _zero,
            "d2ldmdv": _zero,
            "d2ldddv": _zero,
        },
        G_dev_incr=lambda y, mu, sigma, nu: -2 * dZAGA(y, mu, sigma, nu, log=True),
        rqres={"pfun": "pZAGA", "type": "Mixed", "mass_p": 0, "prob_mp": "nu"},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.ones(len(y)),
            "nu": lambda y: np.full(len(y), 0.5),
        },
        valid={
            "mu": lambda mu: True,
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
            "nu": lambda nu: bool(np.all(nu > 0) and np.all(nu < 1)),
        },
        y_valid=lambda y: bool(np.all(y >= 0)),
        mean=lambda mu, sigma, nu: (1 - nu) * mu,
        variance=lambda mu, sigma, nu: (1 - nu) * mu**2 * (sigma**2 + nu),
    )


def dZAGA(x, mu=1, sigma=1, nu=0.1, log=False):
    _check(mu, sigma, nu)
    x, mu, sigma, nu = np.broadcast_arrays(*(np.asarray(a, float) for a in (x, mu, sigma, nu)))
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = (np.log(1 - nu) + (1 / sigma**2) * np.log(x / (mu * sigma**2)) - x / (mu * sigma**2)
               - np.log(x) - _sp.gammaln(1 / sigma**2))
        log_lik = np.where(x == 0, np.log(nu), pos)
    fy = log_lik if log else np.exp(log_lik)
    return np.where(x < 0, 0.0, fy)


def pZAGA(q, mu=1, sigma=1, nu=0.1, lower_tail=True, log_p=False):
    _check(mu, sigma, nu)
    q, mu, sigma, nu = np.broadcast_arrays(*(np.asarray(a, float) for a in (q, mu, sigma, nu)))
    with np.errstate(divide="ignore", invalid="ignore"):
        cdf = _st.gamma.cdf(q, 1 / sigma**2, scale=mu * sigma**2)
        cdf = np.where(q == 0, nu, nu + (1 - nu) * cdf)
        if not lower_tail:
            cdf = 1 - cdf
        if log_p:
            cdf = np.log(cdf)
    return np.where(q < 0, 0.0, cdf)


def qZAGA(p, mu=1, sigma=1, nu=0.1, lower_tail=True, log_p=False):
    _check(mu, sigma, nu, strict=True)
    ly = max(np.size(p), np.size(mu), np.size(sigma), np.size(nu))
    p = np.resize(np.asarray(p, float), ly)
    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1 - p
    if np.any(p < 0) or np.any(p >= 1):
        raise ValueError("p must be between 0 and 1")
    nu = np.asarray(nu, float)
    if nu.size not in (1, p.size):
        raise ValueError(f"nu is of length {nu.size}; must be of length 1 or length(p) = {p.size}")
    nu_b = np.resize(nu, ly)
    p = np.where(p <= nu_b, nu_b, p)
    sigma = np.resize(np.asarray(sigma, float), ly)
    mu = np.resize(np.asarray(mu, float), ly)
    return _st.gamma.ppf((p - nu_b) / (1 - nu_b), 1 / sigma**2, scale=mu * sigma**2)


def rZAGA(n, mu=1, sigma=1, nu=0.1, rng=None):
    _check(mu, sigma, nu, strict=True)
    if np.any(np.asarray(n) <= 0):
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng() if rng is None else rng
    return qZAGA(rng.uniform(size=int(np.ceil(n))), mu=mu, sigma=sigma, nu=nu)
