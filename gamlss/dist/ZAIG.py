"""Zero adjusted inverse Gaussian distribution (ZAIG). Port of gamlss.dist
R/ZAIG.R.

P(Y = 0) = nu; Y | Y > 0 ~ IG(mu, sigma). Type "Mixed": point mass at 0.

pZAIG uses pIG, which evaluates exp(2/(mu sigma^2)) * Phi(.) on the log
scale; R computes it directly and falls back to numerical integration of
dIG when that overflows (NaN). Values agree wherever R's is finite.
qZAIG reproduces R, which applies lower.tail/log.p both to p on entry and
to q on exit.
"""

from __future__ import annotations

import numpy as np

from ..family import GamlssFamily, checklink
from .IG import pIG, qIG
from .ZAGA import NU_LINKS, _check, _zero


def ZAIG(mu_link="log", sigma_link="log", nu_link="logit"):
    mstats = checklink("mu.link", "ZAIG", mu_link, ("inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "ZAIG", sigma_link, ("inverse", "log", "identity", "own"))
    vstats = checklink("nu.link", "ZAIG", nu_link, NU_LINKS)

    def dldd(y, mu, sigma):
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (-1 / sigma) + ((y - mu) ** 2) / (y * (sigma**3) * (mu**2))
        return np.where(y == 0, 0.0, v)

    return GamlssFamily(
        family=("ZAIG", "Zero adjusted IG"),
        parameters={"mu": True, "sigma": True, "nu": True},
        nopar=3,
        type="Mixed",
        links={"mu": mstats, "sigma": dstats, "nu": vstats},
        derivatives={
            "dldm": lambda y, mu, sigma: np.where(y == 0, 0.0, (y - mu) / ((sigma**2) * (mu**3))),
            "d2ldm2": lambda y, mu, sigma: np.where(y == 0, 0.0, -1 / ((mu**3) * (sigma**2))),
            "dldd": dldd,
            "d2ldd2": lambda y, sigma: np.where(y == 0, 0.0, -2 / (sigma**2)),
            "dldv": lambda y, nu: np.where(y == 0, 1 / nu, -1 / (1 - nu)),
            "d2ldv2": lambda nu: -1 / (nu * (1 - nu)),
            "d2ldmdd": _zero,
            "d2ldmdv": _zero,
            "d2ldddv": _zero,
        },
        G_dev_incr=lambda y, mu, sigma, nu: -2 * dZAIG(y, mu, sigma, nu, log=True),
        rqres={"pfun": "pZAIG", "type": "Mixed", "mass_p": 0, "prob_mp": "nu"},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.full(len(y), 0.5),
            "nu": lambda y: np.full(len(y), 0.5),
        },
        valid={
            "mu": lambda mu: True,
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
            "nu": lambda nu: bool(np.all(nu > 0) and np.all(nu < 1)),
        },
        y_valid=lambda y: bool(np.all(y >= 0)),
        mean=None,
        variance=None,
    )


def dZAIG(x, mu=1, sigma=1, nu=0.1, log=False):
    if np.any(np.asarray(mu) < 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be positive")
    x, mu, sigma, nu = np.broadcast_arrays(*(np.asarray(a, float) for a in (x, mu, sigma, nu)))
    xx = np.where(x < 0, 0.001, x)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = np.log(1 - nu) + (-0.5 * np.log(2 * np.pi) - np.log(sigma) - (3 / 2) * np.log(xx)
                                - ((xx - mu) ** 2) / (2 * sigma**2 * (mu**2) * xx))
        log_lik = np.where(xx == 0, np.log(nu), pos)
    fy = log_lik if log else np.exp(log_lik)
    return np.where(x < 0, 0.0, fy)


def pZAIG(q, mu=1, sigma=1, nu=0.1, lower_tail=True, log_p=False):
    if np.any(np.asarray(mu) < 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be positive")
    q, mu, sigma, nu = np.broadcast_arrays(*(np.asarray(a, float) for a in (q, mu, sigma, nu)))
    qq = np.where(q < 0, 0.001, q)
    cdf = pIG(qq, mu=mu, sigma=sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        cdf = np.where(q == 0, nu, nu + (1 - nu) * cdf)
        if not lower_tail:
            cdf = 1 - cdf
        if log_p:
            cdf = np.log(cdf)
    return np.where(q < 0, 0.0, cdf)


def qZAIG(p, mu=1, sigma=1, nu=0.1, lower_tail=True, log_p=False):
    _check(mu, sigma, nu, strict=True)
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1 - p
    if np.any(p < 0) or np.any(p >= 1):
        raise ValueError("p must be between 0 and 1")
    lp = max(p.size, np.size(mu), np.size(sigma), np.size(nu))
    p, sigma, mu, nu = (np.resize(np.asarray(a, float), lp) for a in (p, sigma, mu, nu))
    with np.errstate(divide="ignore", invalid="ignore"):
        p_nu = np.where((p - nu) / (1 - nu) <= 0, 0.5, (p - nu) / (1 - nu))
    q = np.where(p > nu, qIG(p_nu, mu=mu, sigma=sigma), 0.0)
    if not lower_tail:  # aplicado de novo na saída, como no R
        q = 1 - q
    if log_p:
        q = np.log(q)
    return q


def rZAIG(n, mu=1, sigma=1, nu=0.1, rng=None):
    _check(mu, sigma, nu, strict=True)
    if np.any(np.asarray(n) <= 0):
        raise ValueError("n must be a positive integer")
    rng = np.random.default_rng() if rng is None else rng
    return qZAIG(rng.uniform(size=int(np.ceil(n))), mu=mu, sigma=sigma, nu=nu)
