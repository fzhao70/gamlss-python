"""The fitted GAMLSS model object and its methods.

Mirrors the R gamlss object (a list of class "gamlss") and its S3
methods: print, summary, fitted, coef, deviance, logLik, AIC/GAIC,
vcov, predict/lpred, residuals.  R list element names map to
attributes with '.' replaced by '_' (mu.fv -> mu_fv).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats as _st
from scipy.linalg import solve_triangular

from .rqres import rqres as _rqres_fun

PARAM_ORDER = ("mu", "sigma", "nu", "tau")


class GamlssResults:
    """Fitted gamlss model (R class "gamlss")."""

    # ------------------------------------------------------------ basics
    @property
    def df_res(self):
        # R partial matching: object$df.res -> df.residual
        return self.df_residual

    def _params_with(self, suffix):
        return [p for p in self.parameters if hasattr(self, f"{p}_{suffix}")]

    def _compute_rqres(self):
        """out$residuals <- eval(family$rqres)"""
        spec = self.rqres_spec
        if callable(spec):
            return spec(self)
        from . import dist as _dist

        pfun = getattr(_dist, spec["pfun"])
        params = {p: getattr(self, f"{p}_fv", None) for p in self.parameters}
        params = {k: v for k, v in params.items() if v is not None}
        # while fitting, fv lives in engine state; fall back there
        if not params:
            return None
        if hasattr(self, "bd"):
            params["bd"] = self.bd
        kwargs = {k: v for k, v in spec.items() if k not in ("pfun", "type")}
        return _rqres_fun(pfun, type=spec["type"], y=self.y, rng=self.rng,
                          **kwargs, **params)

    # ------------------------------------------------------------ access
    def fitted(self, what="mu", parameter=None):
        what = parameter or what
        if what not in self.parameters:
            raise ValueError(f"{what} is not a parameter in the gamlss object")
        return getattr(self, f"{what}_fv")

    def coef(self, what="mu", parameter=None):
        what = parameter or what
        if what not in self.parameters:
            raise ValueError(f"{what} is not a parameter in the object")
        return getattr(self, f"{what}_coefficients", None)

    def coefAll(self, deviance=False):
        out = {}
        for p in self.parameters:
            c = getattr(self, f"{p}_coefficients", None)
            if c is not None:
                out[p] = c
        if deviance:
            out["deviance"] = self.deviance()
        return out

    def deviance(self, what="G"):
        if what == "G":
            return self.G_deviance
        if what == "P":
            return self.P_deviance
        raise ValueError("put G for Global or P for Penalized deviance")

    def logLik(self):
        return -self.G_deviance / 2

    def AIC(self, k=2, c=False):
        val = self.G_deviance + self.df_fit * k
        if k == 2 and c:
            val += (2 * self.df_fit * (self.df_fit + 1)) / (
                self.N - self.df_fit - 1
            )
        return val

    GAIC = AIC
    IC = AIC

    def lp(self, what="mu", parameter=None):
        what = parameter or what
        return getattr(self, f"{what}_lp")

    def formula(self, what="mu", parameter=None):
        what = parameter or what
        return getattr(self, f"{what}_formula")

    def model_matrix(self, what="mu", parameter=None):
        what = parameter or what
        return getattr(self, f"{what}_x")

    # --------------------------------------------------------- residuals
    def get_residuals(self, what="z-scores", type="simple", terms=None):
        """Port of residuals.gamlss()."""
        w = self.weights
        if what == "z-scores":
            if np.all(w == 1):
                return self.residuals
            if np.all(np.trunc(w) == w):
                if self.type == "Continuous":
                    return np.repeat(self.residuals, w.astype(int))
                # discrete with frequency weights: recompute on expanded data
                from . import dist as _dist

                spec = self.rqres_spec
                pfun = getattr(_dist, spec["pfun"])
                reps = w.astype(int)
                params = {
                    p: np.repeat(self.fitted(p), reps) for p in self.parameters
                }
                if hasattr(self, "bd"):
                    params["bd"] = np.repeat(self.bd, reps)
                kwargs = {k: v for k, v in spec.items()
                          if k not in ("pfun", "type")}
                return _rqres_fun(pfun, type=spec["type"],
                                  y=np.repeat(self.y, reps), rng=self.rng,
                                  **kwargs, **params)
            warnings.warn("weights which are not frequencies are used: "
                          "residuals remain unweighted")
            return self.residuals
        if what not in self.parameters:
            raise ValueError(f"{what} is not a parameter in the object")
        wv = getattr(self, f"{what}_wv")
        l_p = getattr(self, f"{what}_lp")
        wt = getattr(self, f"{what}_wt")
        os = getattr(self, f"{what}_offset")
        if type == "simple":
            return wv - l_p + os
        if type == "weighted":
            return np.sqrt(wt) * (wv - l_p + os)
        return (wv - l_p + os) + self.lpred(what=what, type="terms",
                                            terms=terms)

    resid = get_residuals

    # ------------------------------------------------------------- lpred
    def lpred(self, what="mu", parameter=None, type="link", terms=None,
              se_fit=False):
        """Port of lpred()."""
        what = parameter or what
        if type not in ("link", "response", "terms"):
            raise ValueError("type must be link, response or terms")

        def cal_se():
            X = getattr(self, f"{what}_x")
            qr = getattr(self, f"{what}_qr")
            p = qr["rank"]
            piv = qr["pivot"][:p]
            R = qr["R"][:p, :p]
            Rinv = solve_triangular(R, np.eye(p))
            XRinv = X[:, piv] @ Rinv
            vl = (XRinv**2) @ np.ones(p)
            return np.sqrt(vl)

        if not se_fit:
            if type == "link":
                return getattr(self, f"{what}_lp")
            if type == "response":
                return getattr(self, f"{what}_fv")
            return self._cal_terms(what, se_fit=False, terms=terms)
        if type == "link":
            return {"fit": getattr(self, f"{what}_lp"), "se.fit": cal_se()}
        if type == "response":
            dr = self._family_obj.dr(what)
            dmudeta = np.abs(dr(getattr(self, f"{what}_lp")))
            return {"fit": getattr(self, f"{what}_fv"),
                    "se.fit": cal_se() * dmudeta}
        return self._cal_terms(what, se_fit=True, terms=terms)

    def _cal_terms(self, what, se_fit=False, terms=None):
        """Port of cal.terms inside lpred(): per-term contributions."""
        di = getattr(self, f"{what}_terms")
        X = getattr(self, f"{what}_x")
        qr = getattr(self, f"{what}_qr")
        beta = np.asarray(getattr(self, f"{what}_coefficients"), dtype=float)
        p = qr["rank"]
        piv = qr["pivot"][:p]
        # assignment of columns to terms (patsy term_slices ~ R assign)
        term_names = []
        asgn = {}
        hasintercept = False
        for t, sl in di.term_name_slices.items():
            if t == "Intercept":
                hasintercept = True
                continue
            term_names.append(t)
            asgn[t] = np.arange(sl.start, sl.stop)
        termsconst = 0.0
        avx = X.mean(axis=0)
        if hasintercept:
            termsconst = float(np.sum(avx[piv] * beta[piv]))
            Xc = X - avx
        else:
            Xc = X
        nterms = len(term_names)
        pred = np.zeros((X.shape[0], nterms))
        ip = np.zeros((X.shape[0], nterms)) if se_fit else None
        unpiv = np.zeros(X.shape[1], dtype=int)
        unpiv[piv] = np.arange(1, p + 1)
        if se_fit:
            R = qr["R"][:p, :p]
            Rinv = solve_triangular(R, np.eye(p))
        for i, t in enumerate(term_names):
            iipiv = asgn[t].copy()
            ii = unpiv[iipiv]
            iipiv[ii == 0] = 0
            if np.any(iipiv > 0):
                pred[:, i] = Xc[:, iipiv] @ np.where(
                    np.isnan(beta[iipiv]), 0.0, beta[iipiv]
                )
                if se_fit:
                    ip[:, i] = (
                        (Xc[:, iipiv] @ Rinv[ii - 1, :]) ** 2
                    ) @ np.ones(p)
        cols = term_names
        out = pd.DataFrame(pred, columns=cols)
        if terms is not None:
            keep = [terms] if isinstance(terms, str) else list(terms)
            out = out[keep]
        out.attrs["constant"] = termsconst
        if se_fit:
            se = pd.DataFrame(np.sqrt(ip), columns=cols)
            if terms is not None:
                se = se[keep]
            return {"fit": out, "se.fit": se}
        return out

    # ----------------------------------------------------------- predict
    def predict(self, what="mu", parameter=None, newdata=None, type="link",
                terms=None, se_fit=False, data=None):
        """Port of predict.gamlss()."""
        what = parameter or what
        if newdata is None:
            return self.lpred(what=what, type=type, terms=terms,
                              se_fit=se_fit)
        if se_fit:
            warnings.warn("se.fit = TRUE is not supported for new data "
                          "values at the moment")
        if not isinstance(newdata, pd.DataFrame):
            raise TypeError("newdata must be a data frame")
        data = data if data is not None else self.data
        if data is None:
            raise ValueError("define the original data using the option data")
        # keep only the variables that appear in newdata, then concatenate
        keep = [c for c in newdata.columns if c in data.columns]
        old = data[keep] if keep else data
        combined = pd.concat([old, newdata[keep] if keep else newdata],
                             ignore_index=True)
        onlydata = np.zeros(len(combined), dtype=bool)
        onlydata[: len(old)] = True

        from .formula import ParamFormula
        from .engine import lm_wfit

        pf = ParamFormula(getattr(self, f"{what}_formula"), combined)
        X, di = pf.design(combined)
        offsetVar = pf.offset(combined, len(combined))
        y_work = getattr(self, f"{what}_lp").copy()
        wt = getattr(self, f"{what}_wt")
        if pf.offset_exprs:
            y_work = y_work - offsetVar[onlydata]
        smo = getattr(self, f"{what}_s", None)
        if smo is not None:
            y_work = y_work - smo.sum(axis=1)
        refit = lm_wfit(X[onlydata], y_work, wt)
        coef_new = refit["coefficients"]
        orig_coef = np.asarray(getattr(self, f"{what}_coefficients"), float)
        if (abs(np.nansum(refit["residuals"])) > 1e-001
                or abs(np.nansum(orig_coef - coef_new)) > 1e-005):
            warnings.warn(
                "There is a discrepancy between the original and the re-fit\n"
                " used to achieve 'safe' predictions"
            )
        Xpred = X[~onlydata]
        if type == "terms":
            term_names = []
            asgn = {}
            hasintercept = False
            for t, sl in di.term_name_slices.items():
                if t == "Intercept":
                    hasintercept = True
                    continue
                term_names.append(t)
                asgn[t] = np.arange(sl.start, sl.stop)
            p = refit["qr"]["rank"]
            piv = refit["qr"]["pivot"][:p]
            termsconst = 0.0
            if hasintercept:
                avx = X[onlydata].mean(axis=0)
                termsconst = float(np.sum(avx[piv] * coef_new[piv]))
                Xpred = Xpred - avx
            unpiv = np.zeros(X.shape[1], dtype=int)
            unpiv[piv] = np.arange(1, p + 1)
            pred = np.zeros((Xpred.shape[0], len(term_names)))
            for i, t in enumerate(term_names):
                iipiv = asgn[t].copy()
                ii = unpiv[iipiv]
                iipiv[ii == 0] = 0
                if np.any(iipiv > 0):
                    pred[:, i] = Xpred[:, iipiv] @ np.where(
                        np.isnan(coef_new[iipiv]), 0.0, coef_new[iipiv]
                    )
            out = pd.DataFrame(pred, columns=term_names,
                               index=newdata.index)
            if terms is not None:
                keepc = [terms] if isinstance(terms, str) else list(terms)
                out = out[keepc]
            out.attrs["constant"] = termsconst
            return out
        pred = Xpred @ np.where(np.isnan(coef_new), 0.0, coef_new)
        if pf.offset_exprs:
            pred = pred + offsetVar[~onlydata]
        if type == "response":
            pred = self._family_obj.linkinv(what)(pred)
        return np.asarray(pred)

    def predictAll(self, newdata=None, data=None, output="list"):
        """Port of predictAll(): fitted parameters for new data."""
        out = {}
        for p in self.parameters:
            out[p] = self.predict(what=p, newdata=newdata, type="response",
                                  data=data)
        if newdata is not None:
            # include y if present in newdata (R returns it too)
            from .formula import ParamFormula

            pf = ParamFormula(self.mu_formula, newdata)
            try:
                yv = pf.response(newdata)
                if yv is not None:
                    out["y"] = yv
            except Exception:
                pass
        if output == "data.frame":
            return pd.DataFrame(out)
        return out

    # ------------------------------------------------- likelihood / vcov
    def gen_likelihood(self):
        """Port of gen.likelihood(): minus log-likelihood as a function
        of the stacked beta vector (additive terms treated as fixed)."""
        from . import dist as _dist

        fam = self._family_obj
        dfun = getattr(_dist, "d" + self.family[0])
        y = self.y
        w = self.weights
        bd = getattr(self, "bd", None)
        Xs, links, coefs, offsets, smo = {}, {}, {}, {}, {}
        for p in self.parameters:
            if hasattr(self, f"{p}_fix"):
                linkfun = fam.linkfun(p)
                fixvalue = float(np.asarray(linkfun(self.fitted(p)[:1]))[0])
                coefs[p] = np.array([fixvalue])
                Xs[p] = np.ones((len(y), 1))
                offsets[p] = np.zeros(len(y))
                smo[p] = None
            else:
                cf = np.asarray(getattr(self, f"{p}_coefficients"), float)
                X = getattr(self, f"{p}_x")
                notna = ~np.isnan(cf)
                if np.any(~notna):
                    cf = cf[notna]
                    X = X[:, notna]
                coefs[p] = cf
                Xs[p] = X
                offsets[p] = getattr(self, f"{p}_offset")
                smo[p] = getattr(self, f"{p}_s", None)

        lens = [len(coefs[p]) for p in self.parameters]
        splits = np.cumsum(lens)[:-1]

        def lik_fun(par):
            par = np.asarray(par, dtype=float)
            if par.size != sum(lens):
                raise ValueError("par is not the right length")
            parts = np.split(par, splits)
            kw = {}
            for p, b in zip(self.parameters, parts):
                eta = Xs[p] @ b + offsets[p]
                if smo[p] is not None:
                    eta = eta + smo[p].sum(axis=1)
                kw[p] = fam.linkinv(p)(eta)
            if bd is not None:
                kw["bd"] = bd
            # R's sum() accumulates in long double; use exact summation
            # so the numerical Hessian is not polluted by rounding noise
            return -math.fsum(w * dfun(y, log=True, **kw))

        start = np.concatenate([coefs[p] for p in self.parameters])
        names = []
        for p in self.parameters:
            cf = getattr(self, f"{p}_coefficients", None)
            if hasattr(self, f"{p}_fix") or cf is None:
                names.append(f"fixed {p}")
            else:
                cfv = np.asarray(cf, float)
                names.extend(f"{p}.{ix}" for ix, ok in
                             zip(cf.index, ~np.isnan(cfv)) if ok)
        return lik_fun, start, names

    def vcov(self, type="vcov", robust=False, hessian_fun="R"):
        """Port of vcov.gamlss()."""
        if robust:
            raise NotImplementedError("robust vcov is not implemented yet")
        like_fun, betaCoef, names = self.gen_likelihood()
        if hessian_fun == "R":
            hess = _optim_hess(betaCoef, like_fun)
        else:
            hess = _hessian_pb(betaCoef, like_fun)["Hessian"]
        self._vcov_used_fallback = False
        try:
            varCov = np.linalg.inv(hess)
            failed = np.any(np.diag(varCov) < 0)
        except np.linalg.LinAlgError:
            failed = True
        if failed:
            # R falls back to the HessianPB quadratic-surface Hessian
            self._vcov_used_fallback = True
            try:
                varCov = np.linalg.inv(_hessian_pb(betaCoef, like_fun)["Hessian"])
            except np.linalg.LinAlgError as e:
                raise RuntimeError(
                    "the Hessian matrix is singular; probably the model is "
                    "overparametrised"
                ) from e
        se = np.sqrt(np.diag(varCov))
        with np.errstate(invalid="ignore"):
            corr = varCov / np.outer(se, se)
        if type == "vcov":
            return pd.DataFrame(varCov, index=names, columns=names)
        if type == "cor":
            return pd.DataFrame(corr, index=names, columns=names)
        if type == "se":
            return pd.Series(se, index=names)
        if type == "coef":
            return pd.Series(betaCoef, index=names)
        return {
            "coef": pd.Series(betaCoef, index=names),
            "se": pd.Series(se, index=names),
            "vcov": pd.DataFrame(varCov, index=names, columns=names),
            "cor": pd.DataFrame(corr, index=names, columns=names),
        }

    # ----------------------------------------------------------- summary
    def summary(self, type="vcov", robust=False, save=False,
                hessian_fun="R", digits=4, print_out=True):
        """Port of summary.gamlss(). Returns the coefficient table."""
        covmat = None
        if type == "vcov":
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    covmat = self.vcov(type="all", robust=robust,
                                       hessian_fun=hessian_fun)
                if np.any(np.isnan(covmat["se"])):
                    raise ValueError("NA in se's")
            except Exception:
                warnings.warn("summary: vcov has failed, option qr is used "
                              "instead")
                type = "qr"

        tables = []
        if type == "vcov":
            coef = covmat["coef"]
            se = covmat["se"]
            tvalue = coef / se
            pvalue = 2 * _st.t.sf(np.abs(tvalue), self.df_res)
            table = pd.DataFrame(
                {"Estimate": coef, "Std. Error": se, "t value": tvalue,
                 "Pr(>|t|)": pvalue}
            )
            # split per parameter for printing
            pos = 0
            for p in self.parameters:
                if hasattr(self, f"{p}_fix"):
                    npar = 1
                else:
                    cf = getattr(self, f"{p}_coefficients")
                    npar = int(np.sum(~np.isnan(np.asarray(cf, float))))
                tables.append((p, table.iloc[pos:pos + npar]))
                pos += npar
        else:
            est_disp = self.family[0] not in ("PO", "BI", "EX", "P1")
            for p in self.parameters:
                if getattr(self, f"{p}_df", 0) == 0:
                    continue
                qr = getattr(self, f"{p}_qr")
                rank = qr["rank"]
                p1 = np.arange(rank - int(getattr(self, f"{p}_nl_df", 0)))
                R = qr["R"][np.ix_(p1, p1)]
                Rinv = solve_triangular(R, np.eye(len(p1)))
                covmat_uns = Rinv @ Rinv.T
                cf = getattr(self, f"{p}_coefficients")
                piv = qr["pivot"][p1]
                coef_p = np.asarray(cf, float)[piv]
                cnames = [cf.index[i] for i in piv]
                s_err = np.sqrt(np.diag(covmat_uns))
                tvalue = coef_p / s_err
                df_r = self.noObs - getattr(self, f"{p}_df")
                if not est_disp:
                    pvalue = 2 * _st.norm.sf(np.abs(tvalue))
                    cols = ["Estimate", "Std. Error", "z value", "Pr(>|z|)"]
                elif df_r > 0:
                    pvalue = 2 * _st.t.sf(np.abs(tvalue), df_r)
                    cols = ["Estimate", "Std. Error", "t value", "Pr(>|t|)"]
                else:
                    pvalue = np.full(len(coef_p), np.nan)
                    cols = ["Estimate", "Std. Error", "t value", "Pr(>|t|)"]
                table = pd.DataFrame(
                    dict(zip(cols, [coef_p, s_err, tvalue, pvalue])),
                    index=cnames,
                )
                tables.append((p, table))

        if print_out:
            line = "*" * 66
            print(line)
            print(f"Family:  {self.family}")
            print(f"\nCall:  {self.call}")
            print(f"Fitting method: {self.method} \n")
            for p, tab in tables:
                print("-" * 66)
                if hasattr(self, f"{p}_fix"):
                    print(f"{p.capitalize()} parameter is fixed")
                    fv = getattr(self, f"{p}_fv")
                    if np.all(fv == fv[0]):
                        print(f"{p.capitalize()} = {fv[0]}")
                    continue
                print(f"{p.capitalize()} link function:  "
                      f"{getattr(self, f'{p}_link')}")
                print(f"{p.capitalize()} Coefficients:")
                with pd.option_context("display.float_format",
                                       lambda v: f"{v:.{digits}g}"):
                    print(tab.to_string())
                print()
            print("-" * 66)
            print(f"No. of observations in the fit:  {self.noObs:g}")
            print(f"Degrees of Freedom for the fit:  {self.df_fit:g}")
            print(f"      Residual Deg. of Freedom:  {self.df_residual:g}")
            print(f"                      at cycle:  {self.iter:g}\n")
            print(f"Global Deviance:     {self.G_deviance}")
            print(f"            AIC:     {self.aic}")
            print(f"            SBC:     {self.sbc}")
            print(line)
        full = pd.concat([t for _, t in tables]) if tables else pd.DataFrame()
        return full

    # ------------------------------------------------------------- print
    def __repr__(self):
        out = []
        out.append(f"\nFamily:  {self.family}")
        out.append(f"Fitting method: {self.method}")
        out.append(f"\nCall:  {self.call}\n")
        for p in self.parameters:
            cf = getattr(self, f"{p}_coefficients", None)
            if cf is None:
                continue
            out.append(f"{p.capitalize()} Coefficients:")
            out.append(cf.to_string())
        out.append(
            f"\n Degrees of Freedom for the fit: {self.df_fit} "
            f"Residual Deg. of Freedom   {self.df_residual}"
        )
        out.append(f"Global Deviance:     {self.G_deviance:.6g}")
        out.append(f"            AIC:     {self.aic:.6g}")
        out.append(f"            SBC:     {self.sbc:.6g}\n")
        return "\n".join(out)


# ---------------------------------------------------------------- hessians
def _optim_hess(par, fn, ndeps=1e-3):
    """Port of R's optimHess (stats C optimhess, gr=NULL), ndeps=1e-3."""
    par = np.asarray(par, dtype=float)
    npar = len(par)
    H = np.empty((npar, npar))
    for j in range(npar):
        for i in range(npar):
            ei = np.zeros(npar)
            ej = np.zeros(npar)
            ei[i] = ndeps
            ej[j] = ndeps
            H[i, j] = (
                fn(par + ej + ei) - fn(par + ej - ei)
                - fn(par - ej + ei) + fn(par - ej - ei)
            ) / (4.0 * ndeps * ndeps)
    return 0.5 * (H + H.T)


def _hessian_pb(pars, fun, rel_step=None, min_abs_par=0.0):
    """Port of the HessianPB() local function in vcov.gamlss()."""
    pars = np.asarray(pars, dtype=float)
    npar = len(pars)
    if rel_step is None:
        rel_step = np.finfo(float).eps ** (1 / 3)
    incr = np.where(np.abs(pars) <= min_abs_par, min_abs_par * rel_step,
                    np.abs(pars) * rel_step)
    base = np.eye(npar)
    cols = [np.zeros((npar, 1)), base, -base]
    frac = [1.0] + list(incr) + list(incr**2)
    for i in range(npar - 1):
        block = base[:, [i]] + base[:, i + 1:]
        cols.append(block)
        frac.extend(incr[i] * incr[i + 1:])
    indMat = np.hstack(cols)
    shifted = pars[:, None] + incr[:, None] * indMat
    indMat = indMat.T
    Xcols = [np.ones((indMat.shape[0], 1)), indMat, indMat**2]
    for i in range(npar - 1):
        Xcols.append(indMat[:, [i]] * indMat[:, i + 1:])
    Xmat = np.hstack(Xcols)
    fvals = np.array([fun(shifted[:, k]) for k in range(shifted.shape[1])])
    coefs = np.linalg.solve(Xmat, fvals) / np.asarray(frac)
    Hess = np.diag(coefs[1 + npar: 1 + 2 * npar])
    # fill lower triangle column-major with the cross coefficients
    cross = coefs[1 + 2 * npar:]
    k = 0
    for j in range(npar - 1):
        for i in range(j + 1, npar):
            Hess[i, j] = cross[k]
            k += 1
    Hess = Hess + Hess.T
    return {"mean": coefs[0], "gradient": coefs[1: 1 + npar],
            "Hessian": Hess}
