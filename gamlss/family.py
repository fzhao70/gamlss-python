"""gamlss.family infrastructure.

Port of gamlss.dist R/gamlss-family.R.

A GamlssFamily mirrors the R "gamlss.family" list: link functions per
parameter, first/expected-second derivative functions of the
log-likelihood, the global deviance increment, initial value
expressions, validity checks and the quantile residual specification.

Derivative functions are written with the same argument names as the R
source (subsets of y, mu, sigma, nu, tau, bd) and are dispatched by
introspection, mirroring how R resolves them lexically.
"""

from __future__ import annotations

import inspect

import numpy as np

from .links import LinkGamlss, make_link_gamlss

PARAMETERS = ("mu", "sigma", "nu", "tau")


def checklink(which_link, which_dist, link, link_list):
    """Port of checklink(): validate the link and build it."""
    if link is None:
        raise ValueError("The link has not been defined.")
    linktemp = link
    if isinstance(linktemp, LinkGamlss):
        return linktemp.name, linktemp
    if not isinstance(linktemp, str):
        raise TypeError(
            f'"{which_link}" link "{linktemp}" not available for "{which_dist}" '
            f"family; available links are {link_list}"
        )
    if linktemp not in link_list:
        # R falls through to make.link.gamlss for any character link
        pass
    stats = make_link_gamlss(linktemp)
    return linktemp, stats


def _argnames(fun):
    return tuple(inspect.signature(fun).parameters)


class _Dispatched:
    """Wrap an R-style derivative function: call with full state, pass
    only the arguments the function declares (mirrors R's lexical
    scoping of y/mu/sigma/nu/tau/bd in the gamlss environment)."""

    __slots__ = ("fun", "args")

    def __init__(self, fun):
        self.fun = fun
        self.args = _argnames(fun)

    def __call__(self, **state):
        return self.fun(*(state[a] for a in self.args))


class GamlssFamily:
    """Python equivalent of the R gamlss.family object."""

    def __init__(
        self,
        family,
        parameters,
        type,
        links,
        derivatives,
        G_dev_incr,
        rqres,
        initial,
        valid,
        y_valid,
        nopar=None,
        mean=None,
        variance=None,
        extra=None,
    ):
        #: c("NO", "Normal") in R
        self.family = tuple(family)
        #: list(mu=TRUE, sigma=TRUE, ...)
        self.parameters = dict(parameters)
        self.nopar = nopar if nopar is not None else len(self.parameters)
        #: "Continuous", "Discrete" or "Mixed"
        self.type = type

        # links: dict param -> (linkname, LinkGamlss)
        for p, (linkname, stats) in links.items():
            setattr(self, f"{p}_link", linkname)
            setattr(self, f"{p}_linkfun", stats.linkfun)
            setattr(self, f"{p}_linkinv", stats.linkinv)
            setattr(self, f"{p}_dr", stats.mu_eta)

        # first and expected second derivatives, R names:
        #   dldm, d2ldm2, dldd, d2ldd2, dldv, d2ldv2, dldt, d2ldt2
        #   d2ldmdd, d2ldmdv, d2ldmdt, d2ldddv, d2ldddt, d2ldvdt
        self._derivs = {k: _Dispatched(f) for k, f in derivatives.items()}

        #: deviance increment function: called with y and parameters
        self.G_dev_incr = _Dispatched(G_dev_incr)

        #: dict(pfun="pNO", type="Continuous", ymin=None)
        self.rqres = dict(rqres)

        #: dict param -> callable(y, w, bd, ...) returning initial fv
        self.initial = {p: _Dispatched(f) for p, f in initial.items()}

        #: dict param -> callable(value) -> bool
        self.valid = dict(valid)
        self.y_valid = y_valid

        self.mean = mean
        self.variance = variance
        # any extra family-specific payload (e.g. bd handling)
        self.extra = extra or {}

    # -- R-style accessors used by the fitting engine ------------------
    _DLDP = {"mu": "dldm", "sigma": "dldd", "nu": "dldv", "tau": "dldt"}
    _D2LDP2 = {"mu": "d2ldm2", "sigma": "d2ldd2", "nu": "d2ldv2", "tau": "d2ldt2"}
    _CROSS = {
        ("mu", "sigma"): "d2ldmdd",
        ("mu", "nu"): "d2ldmdv",
        ("mu", "tau"): "d2ldmdt",
        ("sigma", "nu"): "d2ldddv",
        ("sigma", "tau"): "d2ldddt",
        ("nu", "tau"): "d2ldvdt",
    }

    def dldp(self, what, **state):
        """First derivative of log-lik wrt parameter `what`."""
        return np.asarray(self._derivs[self._DLDP[what]](**state), dtype=float)

    def d2ldp2(self, what, **state):
        """(Expected) second derivative of log-lik wrt parameter `what`."""
        return np.asarray(self._derivs[self._D2LDP2[what]](**state), dtype=float)

    def d2ldpdq(self, p1, p2, **state):
        """Cross derivative between two parameters (for CG)."""
        key = (p1, p2) if (p1, p2) in self._CROSS else (p2, p1)
        name = self._CROSS[key]
        if name not in self._derivs:
            n = len(np.asarray(state["y"]))
            return np.zeros(n)
        return np.asarray(self._derivs[name](**state), dtype=float)

    def g_dev_incr(self, **state):
        return np.asarray(self.G_dev_incr(**state), dtype=float)

    def link(self, what):
        return getattr(self, f"{what}_link")

    def linkfun(self, what):
        return getattr(self, f"{what}_linkfun")

    def linkinv(self, what):
        return getattr(self, f"{what}_linkinv")

    def dr(self, what):
        """dmu/deta as function of eta for parameter `what`."""
        return getattr(self, f"{what}_dr")

    def __repr__(self):
        lines = [f"\nGAMLSS Family: {self.family[0]} {self.family[1]}"]
        for p in PARAMETERS:
            if p in self.parameters:
                lines.append(
                    f"Link function for {p + ':':<7} {getattr(self, p + '_link')}"
                )
        return "\n".join(lines)


def as_gamlss_family(obj):
    """Port of as.gamlss.family() / gamlss.family.default()."""
    if isinstance(obj, GamlssFamily):
        return obj
    if obj is None:
        from .dist import NO

        return NO()
    if callable(obj):
        return as_gamlss_family(obj())
    if isinstance(obj, str):
        from . import dist

        if hasattr(dist, obj):
            return as_gamlss_family(getattr(dist, obj))
        raise ValueError(f"unknown gamlss family: {obj}")
    raise TypeError("The object argument is invalid")
