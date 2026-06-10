"""Link functions for gamlss families.

Port of gamlss.dist R/make-link-gamlss.R (make.link.gamlss).
Each link provides linkfun (mu -> eta), linkinv (eta -> mu),
mu_eta (dmu/deta as a function of eta) and valideta, exactly as in R.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

# .Machine$double.eps in R
DOUBLE_EPS = 2.220446049250313e-16


def _asarray(x):
    return np.asarray(x, dtype=float)


class LinkGamlss:
    """Equivalent of R's "link-gamlss" structure."""

    def __init__(self, name, linkfun, linkinv, mu_eta, valideta):
        self.name = name
        self.linkfun = linkfun
        self.linkinv = linkinv
        self.mu_eta = mu_eta
        self.valideta = valideta

    def __repr__(self):  # pragma: no cover
        return f"<link-gamlss: {self.name}>"


# ---------------------------------------------------------------- logistic
# R uses qLO/pLO/dLO (standard logistic) for the "logit" link
def _qlogis(p):
    p = _asarray(p)
    return np.log(p / (1 - p))


def _plogis(q):
    q = _asarray(q)
    return 1 / (1 + np.exp(-q))


def _dlogis(x):
    x = _asarray(x)
    ex = np.exp(-np.abs(x))
    return ex / (1 + ex) ** 2


def make_link_gamlss(link: str) -> LinkGamlss:
    """Port of make.link.gamlss()."""
    if not isinstance(link, str):
        raise TypeError("link must be a character string")

    if link == "logit":

        def linkfun(mu):
            return _qlogis(mu)

        def linkinv(eta):
            thresh = -_qlogis(DOUBLE_EPS)
            eta = np.minimum(thresh, np.maximum(_asarray(eta), -thresh))
            return _plogis(eta)

        def mu_eta(eta):
            return np.maximum(_dlogis(eta), DOUBLE_EPS)

        def valideta(eta):
            return True

    elif link == "probit":

        def linkfun(mu):
            return _st.norm.ppf(mu)

        def linkinv(eta):
            thresh = -_st.norm.ppf(DOUBLE_EPS)
            eta = np.minimum(thresh, np.maximum(_asarray(eta), -thresh))
            return _st.norm.cdf(eta)

        def mu_eta(eta):
            return np.maximum(_st.norm.pdf(_asarray(eta)), DOUBLE_EPS)

        def valideta(eta):
            return True

    elif link == "cauchit":

        def linkfun(mu):
            return _st.cauchy.ppf(mu)

        def linkinv(eta):
            thresh = -_st.cauchy.ppf(DOUBLE_EPS)
            eta = np.minimum(np.maximum(_asarray(eta), -thresh), thresh)
            return _st.cauchy.cdf(eta)

        def mu_eta(eta):
            return np.maximum(_st.cauchy.pdf(_asarray(eta)), DOUBLE_EPS)

        def valideta(eta):
            return True

    elif link == "cloglog":

        def linkfun(mu):
            mu = _asarray(mu)
            return np.log(-np.log(1 - mu))

        def linkinv(eta):
            eta = _asarray(eta)
            return np.maximum(
                np.minimum(-np.expm1(-np.exp(eta)), 1 - DOUBLE_EPS), DOUBLE_EPS
            )

        def mu_eta(eta):
            eta = np.minimum(_asarray(eta), 700)
            return np.maximum(np.exp(eta) * np.exp(-np.exp(eta)), DOUBLE_EPS)

        def valideta(eta):
            return True

    elif link == "identity":

        def linkfun(mu):
            return _asarray(mu)

        def linkinv(eta):
            return _asarray(eta)

        def mu_eta(eta):
            return np.ones_like(_asarray(eta))

        def valideta(eta):
            return True

    elif link == "log":

        def linkfun(mu):
            return np.log(_asarray(mu))

        def linkinv(eta):
            return np.maximum(np.exp(_asarray(eta)), DOUBLE_EPS)

        def mu_eta(eta):
            return np.maximum(np.exp(_asarray(eta)), DOUBLE_EPS)

        def valideta(eta):
            return True

    elif link == "sqrt":

        def linkfun(mu):
            return _asarray(mu) ** 0.5

        def linkinv(eta):
            return _asarray(eta) ** 2

        def mu_eta(eta):
            return 2 * _asarray(eta)

        def valideta(eta):
            return bool(np.all(_asarray(eta) > 0))

    elif link == "1/mu^2":

        def linkfun(mu):
            return 1 / _asarray(mu) ** 2

        def linkinv(eta):
            return 1 / _asarray(eta) ** 0.5

        def mu_eta(eta):
            return -1 / (2 * _asarray(eta) ** 1.5)

        def valideta(eta):
            return bool(np.all(_asarray(eta) > 0))

    elif link == "mu^2":

        def linkfun(mu):
            return _asarray(mu) ** 2

        def linkinv(eta):
            return _asarray(eta) ** 0.5

        def mu_eta(eta):
            return 0.5 * _asarray(eta) ** -0.5

        def valideta(eta):
            return bool(np.all(_asarray(eta) > 0))

    elif link == "logshiftto1":

        def linkfun(mu):
            return np.log(_asarray(mu) - 1 + 0.00001)

        def linkinv(eta):
            return 1 + np.maximum(DOUBLE_EPS, np.exp(_asarray(eta)))

        def mu_eta(eta):
            return np.maximum(DOUBLE_EPS, np.exp(_asarray(eta)))

        def valideta(eta):
            return True

    elif link == "logshiftto2":

        def linkfun(mu):
            return np.log(_asarray(mu) - 2 + 0.00001)

        def linkinv(eta):
            return 2 + np.maximum(DOUBLE_EPS, np.exp(_asarray(eta)))

        def mu_eta(eta):
            return np.maximum(DOUBLE_EPS, np.exp(_asarray(eta)))

        def valideta(eta):
            return True

    elif link in ("logshiftto0", "Slog"):

        def linkfun(mu):
            return np.log(_asarray(mu) - 1e-05)

        def linkinv(eta):
            return 1e-05 + np.maximum(DOUBLE_EPS, np.exp(_asarray(eta)))

        def mu_eta(eta):
            return np.maximum(DOUBLE_EPS, np.exp(_asarray(eta)))

        def valideta(eta):
            return True

    elif link in ("[-1,1]", "(0,2]", "(0,5]"):
        delta = 1e-10
        if link == "[-1,1]":
            shift = (-1 - delta, 1 + delta)
        elif link == "(0,2]":
            shift = (0.0, 2 + delta)
        else:
            shift = (0.0, 5 + delta)

        def linkfun(mu, shift=shift):
            mu = _asarray(mu)
            return np.log((mu - shift[0]) / (shift[1] - mu))

        def linkinv(eta, shift=shift):
            thresh = -np.log(DOUBLE_EPS)
            eta = np.minimum(thresh, np.maximum(_asarray(eta), -thresh))
            return (shift[1] * np.exp(eta) + shift[0]) / (1 + np.exp(eta))

        def mu_eta(eta, shift=shift):
            thresh = -np.log(DOUBLE_EPS)
            eta = _asarray(eta)
            res = np.full(eta.shape, DOUBLE_EPS)
            ok = np.abs(eta) < thresh
            e = np.exp(eta[ok])
            res[ok] = (shift[1] * e) / (1 + e) - (e * (shift[1] * e + shift[0])) / (
                1 + e
            ) ** 2
            return res

        def valideta(eta):
            return True

    elif link == "inverse":

        def linkfun(mu):
            return 1 / _asarray(mu)

        def linkinv(eta):
            return 1 / _asarray(eta)

        def mu_eta(eta):
            return -1 / _asarray(eta) ** 2

        def valideta(eta):
            return bool(np.all(_asarray(eta) != 0))

    else:
        raise ValueError(f"{link!r} link not recognised")

    return LinkGamlss(link, linkfun, linkinv, mu_eta, valideta)
