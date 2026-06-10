"""R-style generic functions for gamlss objects.

These mirror the R API: fitted(m), coef(m, "sigma"), deviance(m),
logLik(m), AIC(m1, m2), GAIC(...), vcov(m), predict(m, ...),
residuals(m), summary(m).  All of them also exist as methods on the
fitted model object.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import GamlssResults


def is_gamlss(x):
    return isinstance(x, GamlssResults)


def fitted(object, what="mu", parameter=None):
    return object.fitted(what=what, parameter=parameter)


def fv(obj, what="mu", parameter=None):
    return obj.fitted(what=what, parameter=parameter)


def coef(object, what="mu", parameter=None):
    return object.coef(what=what, parameter=parameter)


def coefAll(obj, deviance=False):
    return obj.coefAll(deviance=deviance)


def deviance(object, what="G"):
    return object.deviance(what=what)


def logLik(object):
    return object.logLik()


def lp(obj, what="mu", parameter=None):
    return obj.lp(what=what, parameter=parameter)


def lpred(obj, what="mu", parameter=None, type="link", terms=None,
          se_fit=False):
    return obj.lpred(what=what, parameter=parameter, type=type, terms=terms,
                     se_fit=se_fit)


def predict(object, what="mu", parameter=None, newdata=None, type="link",
            terms=None, se_fit=False, data=None):
    return object.predict(what=what, parameter=parameter, newdata=newdata,
                          type=type, terms=terms, se_fit=se_fit, data=data)


def predictAll(object, newdata=None, data=None, output="list"):
    return object.predictAll(newdata=newdata, data=data, output=output)


def residuals(object, what="z-scores", type="simple", terms=None):
    return object.get_residuals(what=what, type=type, terms=terms)


resid = residuals


def vcov(object, type="vcov", robust=False, hessian_fun="R"):
    return object.vcov(type=type, robust=robust, hessian_fun=hessian_fun)


def summary(object, type="vcov", robust=False, save=False, hessian_fun="R",
            print_out=True):
    return object.summary(type=type, robust=robust, save=save,
                          hessian_fun=hessian_fun, print_out=print_out)


def IC(object, k=2):
    if not is_gamlss(object):
        raise TypeError("this is not a gamlss object")
    return object.G_deviance + object.df_fit * k


def _gaic_table(objects, names, k, c):
    df = [o.df_fit for o in objects]
    N = [o.N for o in objects]
    if k == 2 and c:
        cor = [(2 * d * (d + 1)) / (n - d - 1) for d, n in zip(df, N)]
    else:
        cor = [0.0] * len(objects)
    aic = [o.G_deviance + o.df_fit * k + cc for o, cc in zip(objects, cor)]
    val = pd.DataFrame({"df": df, "AIC": aic}, index=names)
    return val.sort_values("AIC")


def AIC(object, *args, k=2, c=False):
    if args:
        objects = [object, *args]
        if not all(is_gamlss(o) for o in objects):
            raise TypeError("some of the objects are not gamlss")
        names = [f"model{i + 1}" for i in range(len(objects))]
        return _gaic_table(objects, names, k, c)
    val = object.G_deviance + object.df_fit * k
    if k == 2 and c:
        val += (2 * object.df_fit * (object.df_fit + 1)) / (
            object.N - object.df_fit - 1
        )
    return val


def GAIC(object, *args, k=2, c=False):
    return AIC(object, *args, k=k, c=c)


def Rsq(object, type="Cox Snell"):
    """Port of Rsq(): generalised R-squared."""
    from .engine import gamlss as _gamlss
    from .family import as_gamlss_family

    # null model: intercepts only, same family
    fam = object._family_obj
    data = object.data
    yname = object.mu_formula.split("~")[0].strip()
    m0 = _gamlss(f"{yname} ~ 1", family=fam, data=data, trace=False)
    r2 = 1 - np.exp((2 / object.N) * (m0.logLik() - object.logLik()))
    if type == "Cox Snell":
        return r2
    cragg = r2 / (1 - np.exp((2 / object.N) * m0.logLik()))
    if type == "Cragg Uhler":
        return cragg
    return {"CoxSnell": r2, "CraggUhler": cragg}
