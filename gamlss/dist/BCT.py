"""Box-Cox t distribution (BCT, BCTo). Port of gamlss.dist R/BCT.R."""

from __future__ import annotations

import numpy as np
from scipy import special as _sp
from scipy import stats as _st

from ..family import GamlssFamily, checklink
from .BCCG import dBCCG, _zofy


def _bct_family(name, fullname, mu_link, sigma_link, nu_link, tau_link):
    mstats = checklink("mu.link", "Box Cox t ", mu_link,
                       ("inverse", "log", "identity", "own"))
    dstats = checklink("sigma.link", "Box Cox t ", sigma_link,
                       ("inverse", "log", "identity", "own"))
    vstats = checklink("nu.link", "Box Cox t ", nu_link,
                       ("inverse", "log", "identity", "own"))
    tstats = checklink("tau.link", "Box Cox t ", tau_link,
                       ("inverse", "log", "identity", "own"))

    def dldm(y, mu, sigma, nu, tau):
        z = _zofy(y, mu, sigma, nu)
        w = (tau + 1) / (tau + z**2)
        return (w * z) / (mu * sigma) + (nu / mu) * (w * (z**2) - 1)

    def d2ldm2(mu, sigma, nu, tau):
        out = -(tau + 2 * nu * nu * sigma * sigma * tau + 1) / (tau + 3)
        return out / (mu * mu * sigma * sigma)

    def dldd(y, mu, sigma, nu, tau):
        z = _zofy(y, mu, sigma, nu)
        w = (tau + 1) / (tau + z**2)
        h = _st.t.pdf(1 / (sigma * np.abs(nu)), df=tau) / _st.t.cdf(
            1 / (sigma * np.abs(nu)), df=tau
        )
        return (w * (z**2) - 1) / sigma + h / (sigma**2 * np.abs(nu))

    def dldv(y, mu, sigma, nu, tau):
        z = _zofy(y, mu, sigma, nu)
        w = (tau + 1) / (tau + z**2)
        h = _st.t.pdf(1 / (sigma * np.abs(nu)), df=tau) / _st.t.cdf(
            1 / (sigma * np.abs(nu)), df=tau
        )
        out = ((w * z**2) / nu) - np.log(y / mu) * (
            w * z**2 + ((w * z) / (sigma * nu)) - 1
        )
        return out + np.sign(nu) * h / (sigma * nu**2)

    def dldt(y, mu, sigma, nu, tau):
        z = _zofy(y, mu, sigma, nu)
        w = (tau + 1) / (tau + z**2)
        j = (
            np.log(_st.t.cdf(1 / (sigma * np.abs(nu)), df=tau + 0.01))
            - np.log(_st.t.cdf(1 / (sigma * np.abs(nu)), df=tau))
        ) / 0.01
        out = -0.5 * np.log(1 + (z**2) / tau) + (w * (z**2)) / (2 * tau)
        return out + 0.5 * _sp.digamma((tau + 1) / 2) \
            - 0.5 * _sp.digamma(tau / 2) - 1 / (2 * tau) - j

    def d2ldt2(tau):
        out = (_sp.polygamma(1, (tau + 1) / 2) - _sp.polygamma(1, tau / 2)
               + 2 * (tau + 5) / (tau * (tau + 1) * (tau + 3)))
        out = out / 4
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
            "d2ldd2": lambda sigma, tau: -2 * tau / (sigma**2 * (tau + 3)),
            "dldv": dldv,
            "d2ldv2": lambda sigma: -7 * (sigma**2) / 4,
            "dldt": dldt,
            "d2ldt2": d2ldt2,
            "d2ldmdd": lambda mu, sigma, nu, tau: -(2 * nu * tau)
            / (mu * sigma * (tau + 3)),
            "d2ldmdv": lambda mu, tau: (tau - 3) / (2 * mu * (tau + 3)),
            "d2ldmdt": lambda mu, nu, tau: (2 * nu)
            / (mu * (tau + 1) * (tau + 3)),
            "d2ldddv": lambda sigma, nu, tau: -(sigma * nu * tau) / (tau + 3),
            "d2ldddt": lambda sigma, tau: 2 / (sigma * (tau + 1) * (tau + 3)),
            "d2ldvdt": lambda sigma, nu, tau: (2 * sigma**2 * nu) / (tau**2),
        },
        G_dev_incr=lambda y, mu, sigma, nu, tau: -2 * dBCT(
            y, mu, sigma, nu, tau, log=True
        ),
        rqres={"pfun": "p" + name, "type": "Continuous"},
        initial={
            "mu": lambda y: (y + np.mean(y)) / 2,
            "sigma": lambda y: np.full(len(y), 0.1),
            "nu": lambda y: np.full(len(y), 0.5),
            "tau": lambda y: np.full(len(y), 10.0),
        },
        valid={
            "mu": lambda mu: bool(np.all(mu > 0)),
            "sigma": lambda sigma: bool(np.all(sigma > 0)),
            "nu": lambda nu: True,
            "tau": lambda tau: bool(np.all(tau > 0)),
        },
        y_valid=lambda y: bool(np.all(y > 0)),
    )


def BCT(mu_link="identity", sigma_link="log", nu_link="identity",
        tau_link="log"):
    return _bct_family("BCT", "Box-Cox t", mu_link, sigma_link, nu_link,
                       tau_link)


def BCTo(mu_link="log", sigma_link="log", nu_link="identity",
         tau_link="log"):
    return _bct_family("BCTo", "Box-Cox-t-orig.", mu_link, sigma_link,
                       nu_link, tau_link)


def dBCT(x, mu=5, sigma=0.1, nu=1, tau=2, log=False):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(tau) <= 0):
        raise ValueError("tau must be positive")
    x, mu, sigma, nu, tau = np.broadcast_arrays(
        np.asarray(x, float), np.asarray(mu, float),
        np.asarray(sigma, float), np.asarray(nu, float),
        np.asarray(tau, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        z = _zofy(x, mu, sigma, nu)
        loglik = (nu - 1) * np.log(x) - nu * np.log(mu) - np.log(sigma)
        fTz = (_sp.gammaln((tau + 1) / 2) - _sp.gammaln(tau / 2)
               - 0.5 * np.log(tau) - _sp.gammaln(0.5))
        fTz = fTz - ((tau + 1) / 2) * np.log(1 + (z * z) / tau)
        loglik = loglik + fTz - np.log(
            _st.t.cdf(1 / (sigma * np.abs(nu)), df=tau)
        )
    loglik = np.where(
        tau > 1000000, dBCCG(x, mu, sigma, nu, log=True), loglik
    )
    ft = loglik if log else np.exp(loglik)
    return np.where(x <= 0, 0.0, ft)


def pBCT(q, mu=5, sigma=0.1, nu=1, tau=2, lower_tail=True, log_p=False):
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
    FYy1 = _st.t.cdf(z, tau)
    FYy2 = np.where(
        nu > 0, _st.t.cdf(-1 / (sigma * np.abs(nu)), df=tau), 0.0
    )
    FYy3 = _st.t.cdf(1 / (sigma * np.abs(nu)), df=tau)
    FYy = (FYy1 - FYy2) / FYy3
    if not lower_tail:
        FYy = 1 - FYy
    if log_p:
        FYy = np.log(FYy)
    return np.where(q <= 0, 0.0, FYy)


def qBCT(p, mu=5, sigma=0.1, nu=1, tau=2, lower_tail=True, log_p=False):
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
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(
            nu <= 0,
            _st.t.ppf(p * _st.t.cdf(1 / (sigma * np.abs(nu)), tau), tau),
            _st.t.ppf(1 - (1 - p) * _st.t.cdf(1 / (sigma * np.abs(nu)), tau),
                      tau),
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        ya = np.where(
            nu != 0,
            mu * (nu * sigma * z + 1) ** (1 / np.where(nu == 0, 1, nu)),
            mu * np.exp(sigma * z),
        )
    return ya


def rBCT(n, mu=5, sigma=0.1, nu=1, tau=2, rng=None):
    if np.any(np.asarray(mu) <= 0):
        raise ValueError("mu must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")
    if np.any(np.asarray(tau) <= 0):
        raise ValueError("tau must be positive")
    rng = np.random.default_rng() if rng is None else rng
    p = rng.uniform(size=int(np.ceil(n)))
    return qBCT(p, mu=mu, sigma=sigma, nu=nu, tau=tau)


dBCTo = dBCT
pBCTo = pBCT
qBCTo = qBCT
rBCTo = rBCT
