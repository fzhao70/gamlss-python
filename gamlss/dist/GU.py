"""Gumbel (GU) and Reverse Gumbel (RG). Port of gamlss.dist R/GU.R, RG.R."""

from __future__ import annotations

import numpy as np

from ..family import GamlssFamily, checklink


def GU(mu_link="identity", sigma_link="log"):
    mstats = checklink("mu.link", "Gumbel", mu_link,
                       ("inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "Gumbel", sigma_link,
                       ("inverse", "log", "identity", "own"))
    return GamlssFamily(
        family=("GU", "Gumbel"),
        parameters={"mu": True, "sigma": True},
        nopar=2,
        type="Continuous",
        links={"mu": mstats, "sigma": dstats},
        derivatives={
            "dldm": lambda y, mu, sigma: (np.exp((y - mu) / sigma) - 1) / sigma,
            "d2ldm2": lambda sigma: -1 / sigma**2,
            "dldd": lambda y, mu, sigma: -(1 / sigma)
            + ((y - mu) / sigma**2) * (np.exp((y - mu) / sigma) - 1),
            "d2ldd2": lambda sigma: -1.82368 / sigma**2,
            "d2ldmdd": lambda sigma: -0.422784 / sigma**2,
        },
        G_dev_incr=lambda y, mu, sigma: -2 * dGU(y, mu, sigma, log=True),
        rqres={"pfun": "pGU", "type": "Continuous"},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.full(
                len(y), (np.sqrt(6) * np.std(y, ddof=1)) / np.pi
            ),
        },
        valid={
            "mu": lambda mu: True,
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
        },
        y_valid=lambda y: True,
        mean=lambda mu, sigma: mu - 0.5772157 * sigma,
        variance=lambda mu, sigma: 1.64493 * sigma**2,
    )


def dGU(x, mu=0, sigma=1, log=False):
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be positive")
    x, mu, sigma = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    log_lik = -np.log(sigma) + ((x - mu) / sigma) - np.exp((x - mu) / sigma)
    return log_lik if log else np.exp(log_lik)


def pGU(q, mu=0, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be positive")
    q, mu, sigma = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    cdf = 1 - np.exp(-np.exp((q - mu) / sigma))
    if not lower_tail:
        cdf = 1 - cdf
    if log_p:
        cdf = np.log(cdf)
    return cdf


def qGU(p, mu=0, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be positive")
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1 - p
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be between 0 and 1")
    return mu + sigma * np.log(-np.log(1 - p))


def rGU(n, mu=0, sigma=1, rng=None):
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("sigma must be positive")
    rng = np.random.default_rng() if rng is None else rng
    p = rng.uniform(size=int(np.ceil(n)))
    return qGU(p, mu=mu, sigma=sigma)


# -------------------------------------------------------------- RG
def RG(mu_link="identity", sigma_link="log"):
    mstats = checklink("mu.link", "Reverse Gumbel", mu_link,
                       ("inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "Reverse Gumbel", sigma_link,
                       ("inverse", "log", "identity", "own"))
    return GamlssFamily(
        family=("RG", "Reverse Gumbel"),
        parameters={"mu": True, "sigma": True},
        nopar=2,
        type="Continuous",
        links={"mu": mstats, "sigma": dstats},
        derivatives={
            "dldm": lambda y, mu, sigma: (1 - np.exp(-((y - mu) / sigma)))
            / sigma,
            "d2ldm2": lambda sigma: -1 / sigma**2,
            "dldd": lambda y, mu, sigma: -(1 / sigma) * (1 - (y - mu) / sigma)
            - ((y - mu) / sigma**2) * np.exp(-(y - mu) / sigma),
            "d2ldd2": lambda sigma: -1.82368 / sigma**2,
            "d2ldmdd": lambda sigma: -0.422784 / sigma**2,
        },
        G_dev_incr=lambda y, mu, sigma: -2 * dRG(y, mu, sigma, log=True),
        rqres={"pfun": "pRG", "type": "Continuous"},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.full(
                len(y), np.sqrt(6) * np.std(y, ddof=1) / np.pi
            ),
        },
        valid={
            "mu": lambda mu: True,
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
        },
        y_valid=lambda y: True,
        mean=lambda mu, sigma: mu + 0.5772157 * sigma,
        variance=lambda mu, sigma: (np.pi**2 * sigma**2) / 6,
    )


def dRG(x, mu=0, sigma=1, log=False):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    x, mu, sigma = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    log_lik = -np.log(sigma) - ((x - mu) / sigma) - np.exp(-(x - mu) / sigma)
    return log_lik if log else np.exp(log_lik)


def pRG(q, mu=0, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    q, mu, sigma = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(mu, float), np.asarray(sigma, float)
    )
    cdf = np.exp(-np.exp(-(q - mu) / sigma))
    if not lower_tail:
        cdf = 1 - cdf
    if log_p:
        cdf = np.log(cdf)
    return cdf


def qRG(p, mu=0, sigma=1, lower_tail=True, log_p=False):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    p = np.asarray(p, float)
    if log_p:
        p = np.exp(p)
    if not lower_tail:
        p = 1 - p
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p must be between 0 and 1")
    return mu - sigma * np.log(-np.log(p))


def rRG(n, mu=0, sigma=1, rng=None):
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    rng = np.random.default_rng() if rng is None else rng
    p = rng.uniform(size=int(np.ceil(n)))
    return qRG(p, mu=mu, sigma=sigma)
