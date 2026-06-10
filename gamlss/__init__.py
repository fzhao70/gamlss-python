"""gamlss: Generalised Additive Models for Location Scale and Shape.

A Python port of the R packages gamlss and gamlss.dist with an API
that mirrors the R version ('.' in R names becomes '_' in Python):

    from gamlss import gamlss, NO, GA, BCT
    m = gamlss("y ~ x", sigma_formula="~x", family=GA(), data=df)
    m.summary()
    fitted(m, "mu"); coef(m, "sigma"); deviance(m); GAIC(m, k=2)
"""

from .engine import (
    gamlss,
    gamlss_control,
    glim_control,
    RS,
    CG,
    mixed,
)
from .family import GamlssFamily, as_gamlss_family, checklink
from .links import make_link_gamlss
from .model import GamlssResults
from .methods import (
    AIC,
    GAIC,
    IC,
    Rsq,
    coef,
    coefAll,
    deviance,
    fitted,
    fv,
    is_gamlss,
    logLik,
    lp,
    lpred,
    predict,
    predictAll,
    resid,
    residuals,
    summary,
    vcov,
)
from .dist import *  # noqa: F401,F403  families and d/p/q/r functions
from . import dist
from .data import load_data

__version__ = "0.1.0"
