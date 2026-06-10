"""Normalised (randomised) quantile residuals.

Port of gamlss R/rqres.R. For Continuous families the residuals are
deterministic, qnorm(F(y)).  For Discrete families they are randomised
using uniforms between F(y-1) and F(y), exactly as in R (so they only
match an R run if the same uniforms are used).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st


def rqres(pfun, type="Continuous", censored=None, ymin=None,
          mass_p=None, prob_mp=None, y=None, rng=None, **params):
    """Compute normalised quantile residuals.

    pfun : callable
        The cumulative distribution function (e.g. ``pNO``).
    type : "Continuous" | "Discrete" | "Mixed"
    ymin : lowest possible value for discrete families.
    params : fitted distribution parameters (mu=..., sigma=..., bd=...).
    rng : numpy Generator used for the randomisation (Discrete/Mixed).
    """
    y = np.asarray(y, dtype=float)
    if rng is None:
        rng = np.random.default_rng()

    if type == "Continuous":
        return _st.norm.ppf(pfun(y, **params))

    if type == "Discrete":
        if censored is not None:
            raise NotImplementedError("censored discrete rqres not implemented")
        aval = pfun(np.where(y == ymin, y, y - 1), **params)
        aval = np.where(y == ymin, 0.0, aval)
        bval = pfun(y, **params)
        uval = rng.uniform(aval, bval)
        uval = np.where(uval > 0.999999, uval - 0.1e-15, uval)
        uval = np.where(uval < 0.000001, uval + 0.1e-15, uval)
        return _st.norm.ppf(uval)

    if type == "Mixed":
        if mass_p is None and prob_mp is None:
            raise ValueError(
                "For mixed distributions mass.p and prob.mp arguments "
                "have to be specified"
            )
        mass_p = np.atleast_1d(mass_p)
        prob_mp = np.asarray(prob_mp)
        if len(mass_p) == 1:
            if mass_p[0] == 0:
                uval = np.where(
                    y == mass_p[0], rng.uniform(0, prob_mp), pfun(y, **params)
                )
            elif mass_p[0] == 1:
                uval = np.where(
                    y == mass_p[0], rng.uniform(1 - prob_mp, 1), pfun(y, **params)
                )
            else:
                raise ValueError("mass point is not at zero or one")
        else:
            uval = np.where(
                y == mass_p[0], rng.uniform(0, prob_mp[:, 0]), 0.0
            )
            uval = np.where(
                (y > mass_p[0]) & (y < mass_p[1]), pfun(y, **params), uval
            )
            uval = np.where(y == 1, rng.uniform(1 - prob_mp[:, 1], 1), uval)
        return _st.norm.ppf(uval)

    raise ValueError(f"unknown rqres type {type!r}")
