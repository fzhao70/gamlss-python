"""The gamlss() fitting function with the RS and CG algorithms.

Faithful port of gamlss R/gamlss-5.R: the inner GLIM iteration
(glim.fit), the outer Rigby-Stasinopoulos (RS) and Cole-Green (CG)
algorithms, gamlss.control and glim.control, with identical clamping,
step-halving and convergence rules.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.linalg import qr as _qr, solve_triangular

from .family import as_gamlss_family
from .formula import ParamFormula, r_colnames
from .model import GamlssResults

# binomial denominator families (R: .gamlss.bi.list)
GAMLSS_BI_LIST = (
    "BI", "Binomial", "BB", "Beta Binomial", "ZIBI", "ZIBB",
    "ZABI", "ZABB", "DBI", "BItr", "BBtr", "ZIBItr", "ZIBBtr",
    "ZABItr", "ZABBtr", "DBItr",
)

PARAM_ORDER = ("mu", "sigma", "nu", "tau")


# ----------------------------------------------------------- controls
def gamlss_control(c_crit=0.001, n_cyc=20, mu_step=1, sigma_step=1,
                   nu_step=1, tau_step=1, gd_tol=np.inf, iter=0,
                   trace=True, autostep=True, save=True, **kwargs):
    """Port of gamlss.control()."""
    if c_crit <= 0:
        warnings.warn("the value of c.crit supplied is zero or negative; "
                      "the default value of 0.001 was used instead")
        c_crit = 0.001
    if n_cyc < 1:
        warnings.warn("the value of n.cyc supplied is zero or negative; "
                      "the default value of 20 was used instead")
        n_cyc = 20
    if iter < 0:
        warnings.warn("the value of iter supplied is negative; "
                      "the default value of 0 was used instead")
        iter = 0
    for nm in ("mu_step", "sigma_step", "nu_step", "tau_step"):
        v = locals()[nm]
        if v > 1 or v < 0:
            warnings.warn(f"the value of {nm} supplied is out of [0,1]; "
                          "the default value of 1 was used instead")
            if nm == "mu_step":
                mu_step = 1
            elif nm == "sigma_step":
                sigma_step = 1
            elif nm == "nu_step":
                nu_step = 1
            else:
                tau_step = 1
    if gd_tol < 0:
        warnings.warn("the value of gd.tol supplied is less than zero; "
                      "the default value of Inf was used instead")
        gd_tol = np.inf
    return {
        "c.crit": c_crit, "n.cyc": int(n_cyc), "mu.step": mu_step,
        "sigma.step": sigma_step, "nu.step": nu_step, "tau.step": tau_step,
        "gd.tol": gd_tol, "iter": int(iter), "trace": bool(trace),
        "autostep": bool(autostep), "save": bool(save),
    }


def glim_control(cc=0.001, cyc=50, glm_trace=False, bf_cyc=30,
                 bf_tol=0.001, bf_trace=False, **kwargs):
    """Port of glim.control()."""
    if cc <= 0:
        warnings.warn("the value of cc supplied is zero or negative; "
                      "the default value of 0.001 was used instead")
        cc = 0.001
    if bf_tol <= 0:
        warnings.warn("the value of bf.tol supplied is zero or negative; "
                      "the default value of 0.001 was used instead")
        bf_tol = 0.001
    if cyc < 1:
        warnings.warn("the value of cyc supplied is zero or negative; "
                      "the default value of 20 was used instead")
        cyc = 20
    if bf_cyc < 1:
        warnings.warn("the value of bf.cyc supplied is zero or negative; "
                      "the default value of 30 was used instead")
        bf_cyc = 30
    return {
        "cc": cc, "cyc": int(cyc), "glm.trace": bool(glm_trace),
        "bf.cyc": int(bf_cyc), "bf.tol": bf_tol, "bf.trace": bool(bf_trace),
    }


# ------------------------------------------------------ method markers
class _Method:
    name = None

    def __repr__(self):
        return f"{self.name}()"


class RS(_Method):
    """Rigby and Stasinopoulos algorithm (default)."""

    name = "RS"

    def __init__(self, n_cyc=None):
        self.n_cyc = n_cyc


class CG(_Method):
    """Cole and Green algorithm."""

    name = "CG"

    def __init__(self, n_cyc=None):
        self.n_cyc = n_cyc


class mixed(_Method):
    """RS for n1 cycles followed by CG for n2 cycles."""

    name = "mixed"

    def __init__(self, n1=1, n2=20):
        self.n1 = n1
        self.n2 = n2

    def __repr__(self):
        return f"mixed({self.n1}, {self.n2})"


# --------------------------------------------------------------- WLS
def lm_wfit(x, y, w):
    """Weighted least squares mirroring R's lm.wfit (QR, tol=1e-7).

    Zero-weight observations are excluded from the solve; their fitted
    values are X @ coef, as in R.  Aliased (rank-deficient) columns get
    NaN coefficients, as R returns NA.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    n, p = x.shape
    ok = w > 0
    wts = np.sqrt(w[ok])
    Xw = x[ok] * wts[:, None]
    yw = y[ok] * wts
    # R's dqrdc2 keeps the natural column order unless a column is
    # (near-)singular, in which case it is pivoted to the end.
    Q, R = _qr(Xw, mode="economic")
    d = np.abs(np.diag(R))
    dmax = d.max() if d.size else 0.0
    if d.size and dmax > 0 and np.all(d > dmax * 1e-7):
        piv = np.arange(p)
        rank = p
    else:
        Q, R, piv = _qr(Xw, mode="economic", pivoting=True)
        d = np.abs(np.diag(R))
        if d.size == 0 or d[0] == 0:
            rank = 0
        else:
            rank = int(np.sum(d > d[0] * 1e-7))
    coef_piv = np.full(p, np.nan)
    if rank > 0:
        qty = Q[:, :rank].T @ yw
        coef_piv[:rank] = solve_triangular(R[:rank, :rank], qty)
    coef = np.full(p, np.nan)
    coef[piv] = coef_piv
    fitted = x @ np.where(np.isnan(coef), 0.0, coef)
    resid = y - fitted
    return {
        "coefficients": coef,
        "fitted.values": fitted,
        "residuals": resid,
        "rank": rank,
        "qr": {"R": R, "pivot": piv, "rank": rank},
        "df.residual": int(ok.sum()) - rank,
    }


# --------------------------------------------------------- param object
class _ParamObject:
    """The R `get.object(what)` equivalent: per-parameter view of the
    family with the other parameters resolved from the current state."""

    def __init__(self, family, what, y, bd, state):
        self.family = family
        self.what = what
        self.y = y
        self.bd = bd
        self.state = state  # live dict: mu/sigma/nu/tau current values
        self.link = family.link(what)
        self.linkfun = family.linkfun(what)
        self.linkinv = family.linkinv(what)
        self.dr = family.dr(what)
        self.valid = family.valid[what]

    def _full_state(self, fv):
        s = dict(self.state)
        s[self.what] = fv
        s["y"] = self.y
        if self.bd is not None:
            s["bd"] = self.bd
        return s

    def dldp(self, fv):
        return self.family.dldp(self.what, **self._full_state(fv))

    def d2ldp2(self, fv):
        return self.family.d2ldp2(self.what, **self._full_state(fv))

    def G_di(self, fv):
        return self.family.g_dev_incr(**self._full_state(fv))


# ------------------------------------------------------------ glim fit
def _glim_fit(f, X, y, w, fv, os, step, control, auto, gd_tol, family_type):
    """Port of the inner glim.fit() of the RS algorithm."""
    cc = control["cc"]
    cyc = control["cyc"]
    trace = control["glm.trace"]
    itn = 0
    lp = eta = np.asarray(f.linkfun(fv), dtype=float)
    dr = np.asarray(f.dr(eta), dtype=float)
    dr = 1 / dr
    di = f.G_di(fv)
    dv = np.sum(w * di)
    olddv = dv + 1
    dldp = f.dldp(fv)
    d2ldp2 = f.d2ldp2(fv)
    d2ldp2 = np.where(d2ldp2 < -1e-15, d2ldp2, -1e-15)
    wt = -(d2ldp2 / (dr * dr))
    wt = np.where(wt > 1e10, 1e10, wt)
    wt = np.where(wt < 1e-10, 1e-10, wt)
    wv = (eta - os) + dldp / (dr * wt)
    if family_type == "Mixed":
        wv = np.where(np.isnan(wv), 0.0, wv)
    iterw = False
    fit = None

    while abs(olddv - dv) > cc and itn < cyc:
        itn += 1
        lpold = lp
        if np.any(np.isnan(wt)) or np.any(np.isnan(wv)):
            raise RuntimeError(
                f"NA's in the working vector or weights for parameter {f.what}"
            )
        if np.any(~np.isfinite(wt)) or np.any(~np.isfinite(wv)):
            raise RuntimeError(
                f"Inf values in the working vector or weights for parameter {f.what}"
            )
        fit = lm_wfit(X, wv, wt * w)
        lp = (fit["fitted.values"] if itn == 1
              else step * fit["fitted.values"] + (1 - step) * lpold)
        eta = lp + os
        fv = np.asarray(f.linkinv(eta), dtype=float)
        di = f.G_di(fv)
        olddv = dv
        dv = np.sum(w * di)
        # automatic step halving (MS BR Friday, April 15, 2005)
        if dv > olddv and itn >= 2 and auto:
            for _ in range(5):
                lp = (lp + lpold) / 2
                eta = lp + os
                fv = np.asarray(f.linkinv(eta), dtype=float)
                di = f.G_di(fv)
                dv = np.sum(w * di)
                if (olddv - dv) > cc:
                    break
        if (dv > olddv + gd_tol) and itn >= 2 and not iterw:
            warnings.warn(
                f"The deviance has increased in an inner iteration for {f.what}\n"
                "Increase gd.tol and if persist, try different steps\n"
                "or model maybe inappropriate"
            )
            iterw = True
        dr = np.asarray(f.dr(eta), dtype=float)
        dr = 1 / dr
        dldp = f.dldp(fv)
        d2ldp2 = f.d2ldp2(fv)
        d2ldp2 = np.where(d2ldp2 < -1e-15, d2ldp2, -1e-15)
        wt = -(d2ldp2 / (dr * dr))
        wt = np.where(wt > 1e10, 1e10, wt)
        wt = np.where(wt < 1e-10, 1e-10, wt)
        wv = (eta - os) + dldp / (dr * wt)
        if family_type == "Mixed":
            wv = np.where(np.isnan(wv), 0.0, wv)
        if trace:
            print(f"GLIM iteration {itn} for {f.what}: "
                  f"Global Deviance = {dv:.4f}")

    out = dict(fit if fit is not None else {})
    out.update({"fv": fv, "wv": wv, "wt": wt, "eta": eta, "os": os, "pen": 0.0})
    return out


# ------------------------------------------------------------ RS loop
def _rs_fit(setup, n_cyc=None, no_warn=True):
    """Port of the RS() algorithm closure."""
    control = setup["control"]
    i_control = setup["i_control"]
    family = setup["family"]
    state = setup["state"]
    fits = setup["fits"]
    y = setup["y"]
    w = setup["w"]
    bd = setup["bd"]

    c_crit = control["c.crit"]
    n_cyc = control["n.cyc"] if n_cyc is None else n_cyc
    trace = control["trace"]
    autostep = control["autostep"]
    steps = {p: control[f"{p}.step"] for p in PARAM_ORDER}
    gd_tol = control["gd.tol"]
    it = control["iter"]
    conv = False

    def g_dev():
        s = dict(state)
        s["y"] = y
        if bd is not None:
            s["bd"] = bd
        return np.sum(w * family.g_dev_incr(**s))

    G_dev = g_dev()
    G_dev_old = G_dev + 1

    while abs(G_dev_old - G_dev) > c_crit and it < n_cyc:
        for what in PARAM_ORDER:
            if what not in family.parameters:
                continue
            if family.parameters[what] and not setup["fix"][what]:
                f = _ParamObject(family, what, y, bd, state)
                fit = _glim_fit(
                    f=f, X=setup["X"][what], y=y, w=w, fv=state[what],
                    os=setup["offset"][what], step=steps[what],
                    control=i_control, auto=autostep, gd_tol=gd_tol,
                    family_type=family.type,
                )
                state[what] = fit["fv"]
                fits[what] = fit
        G_dev_old = G_dev
        G_dev = g_dev()
        it += 1
        setup["iter"] = it
        if trace:
            print(f"GAMLSS-RS iteration {it}: Global Deviance = {G_dev:.4f}")
        if G_dev > (G_dev_old + gd_tol) and it > 1:
            raise RuntimeError(
                "The global deviance is increasing\n"
                "Try different steps for the parameters or the model "
                "maybe inappropriate"
            )
    if abs(G_dev_old - G_dev) < c_crit:
        conv = True
    if not conv and no_warn:
        warnings.warn("Algorithm RS has not yet converged")
    return conv


# ------------------------------------------------------------ CG loop
def _cg_fit(setup, n_cyc=None):
    """Port of the CG() algorithm closure."""
    control = setup["control"]
    i_control = setup["i_control"]
    family = setup["family"]
    state = setup["state"]
    fits = setup["fits"]
    y = setup["y"]
    w = setup["w"]
    bd = setup["bd"]
    N = setup["N"]
    params = [p for p in PARAM_ORDER if p in family.parameters]

    c_crit = control["c.crit"]
    n_cyc = control["n.cyc"] if n_cyc is None else n_cyc
    trace = control["trace"]
    steps = {p: control[f"{p}.step"] for p in PARAM_ORDER}
    gd_tol = control["gd.tol"]
    autostep = control["autostep"]
    it = control["iter"]
    conv = False
    i_c_crit = i_control["cc"]
    i_n_cyc = i_control["cyc"]
    i_trace = i_control["glm.trace"]

    def full_state():
        s = dict(state)
        s["y"] = y
        if bd is not None:
            s["bd"] = bd
        return s

    def g_dev():
        return np.sum(w * family.g_dev_incr(**full_state()))

    G_dev = g_dev()
    G_dev_old = G_dev + 1
    first_iter = True

    # cross weights between every parameter pair (zeros initially)
    wcross = {pair: np.zeros(N) for pair in
              [("mu", "sigma"), ("mu", "nu"), ("mu", "tau"),
               ("sigma", "nu"), ("sigma", "tau"), ("nu", "tau")]}
    eta = {p: np.zeros(N) for p in PARAM_ORDER}
    eta_old = {p: np.zeros(N) for p in PARAM_ORDER}
    eta_s = {}
    objs = {p: _ParamObject(family, p, y, bd, state) for p in params}

    while abs(G_dev_old - G_dev) > c_crit and it < n_cyc:
        i_iter = 0
        u = {}
        u2 = {}
        dr = {}
        wpar = {}
        z = {}
        for p in params:
            fobj = objs[p]
            eta[p] = eta_old[p] = np.asarray(family.linkfun(p)(state[p]), float)
            u[p] = fobj.dldp(state[p])
            u2[p] = fobj.d2ldp2(state[p])
            dr[p] = 1 / np.asarray(family.dr(p)(eta[p]), float)
            wpar[p] = -u2[p] / (dr[p] * dr[p])
            z[p] = (eta_old[p] - setup["offset"][p]) + steps[p] * u[p] / (
                dr[p] * wpar[p]
            )
            # cross derivatives with previously set-up parameters
            for q in params:
                if q == p:
                    break
                u2c = family.d2ldpdq(q, p, **full_state())
                wcross[(q, p)] = -u2c / (dr[q] * dr[p])

        G_dev_in = G_dev + 1
        i_G_dev = G_dev
        first_iter = False

        # inner iteration
        while abs(G_dev_in - i_G_dev) > i_c_crit and i_iter < i_n_cyc:
            for p in params:
                if not (family.parameters[p] and not setup["fix"][p]):
                    continue
                adj = np.zeros(N)
                for q in params:
                    if q == p:
                        continue
                    pair = (p, q) if (p, q) in wcross else (q, p)
                    adj = adj + wcross[pair] * (eta[q] - eta_old[q])
                adj = -adj / wpar[p]
                wv = z[p] + adj
                fit = lm_wfit(setup["X"][p], wv, wpar[p] * w)
                eta[p] = fit["fitted.values"] + setup["offset"][p]
                state[p] = np.asarray(family.linkinv(p)(eta[p]), float)
                fit["eta"] = eta[p]
                fit["fv"] = state[p]
                fit["wv"] = wv
                fit["wt"] = wpar[p]
                fit["os"] = setup["offset"][p]
                fit["pen"] = 0.0
                fits[p] = fit
            G_dev_in = i_G_dev
            i_G_dev = g_dev()
            i_iter += 1
            if i_trace:
                print(f"CG inner iteration {it}: Global Deviance = "
                      f"{i_G_dev:.4f}")
            if i_G_dev > (G_dev_in + gd_tol) and it > 1:
                raise RuntimeError(
                    "The global deviance is increasing in the inner CG loop\n"
                    "Try different steps for the parameters or the model "
                    "maybe inappropriate"
                )

        G_dev_old = G_dev
        G_dev = g_dev()
        if G_dev > G_dev_old and it >= 2 and autostep:
            for _ in range(5):
                for p in params:
                    eta[p] = (eta[p] + eta_old[p]) / 2
                    state[p] = np.asarray(family.linkinv(p)(eta[p]), float)
                G_dev = g_dev()
                if G_dev < G_dev_old:
                    break
        it += 1
        setup["iter"] = it
        if trace:
            print(f"GAMLSS-CG iteration {it}: Global Deviance = {G_dev:.4f}")
        if G_dev > (G_dev_old + gd_tol) and it > 1:
            raise RuntimeError(
                "The global deviance is increasing in CG-algorithm\n"
                "Try different steps for the parameters or the model "
                "maybe inappropriate"
            )
    if abs(G_dev_old - G_dev) < c_crit:
        conv = True
    if not conv:
        warnings.warn("Algorithm CG has not yet converged")
    return conv


# ------------------------------------------------------------- gamlss
def gamlss(formula, sigma_formula="~1", nu_formula="~1", tau_formula="~1",
           family=None, data=None, weights=None, contrasts=None,
           method=None, start_from=None,
           mu_start=None, sigma_start=None, nu_start=None, tau_start=None,
           mu_fix=False, sigma_fix=False, nu_fix=False, tau_fix=False,
           control=None, i_control=None, rng=None, **kwargs):
    """Fit a GAMLSS model. Port of R's gamlss().

    Parameters mirror the R arguments with '.' replaced by '_'
    (e.g. ``sigma.formula`` -> ``sigma_formula``).
    """
    call = _build_call_string(formula, sigma_formula, nu_formula, tau_formula,
                              family, method, weights)
    if control is None:
        control = gamlss_control(**kwargs)
    if i_control is None:
        i_control = glim_control(**kwargs)
    if method is None:
        method = RS()
    if rng is None:
        rng = np.random.default_rng()

    # ---- data checks (R: stop if NA's present) -----------------------
    if data is not None:
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        if data.isna().any().any():
            raise ValueError(
                "The data contains NA's, use data = data.dropna()")

    # ---- the family ---------------------------------------------------
    family = as_gamlss_family(family)

    # ---- model frame for mu / response --------------------------------
    formulas = {
        "mu": ParamFormula(formula, data, lhs_required=True),
        "sigma": ParamFormula(sigma_formula, data),
        "nu": ParamFormula(nu_formula, data),
        "tau": ParamFormula(tau_formula, data),
    }
    Y = formulas["mu"].response(data)
    if Y is None:
        raise ValueError("the model formula needs a response variable")
    N = Y.shape[0]

    # ---- binomial denominators ----------------------------------------
    bd = None
    if family.family[0] in GAMLSS_BI_LIST:
        if Y.ndim == 1:
            y = np.asarray(Y, dtype=float)
            bd = np.ones(N)
            if np.any((y < 0) | (y > 1)):
                raise ValueError("y values must be 0 <= y <= 1")
        elif Y.ndim == 2 and Y.shape[1] == 2:
            if np.any(np.abs(Y - np.round(Y)) > 0.001):
                warnings.warn("non-integer counts in a binomial GAMLSS!")
            bd = Y[:, 0] + Y[:, 1]
            y = Y[:, 0].astype(float)
            if np.any((y < 0) | (y > bd)):
                raise ValueError("y values must be 0 <= y <= N")
        else:
            raise ValueError(
                "For the binomial family, Y must be a vector of 0 and 1's or "
                "a 2 column matrix where col 1 is no. successes and col 2 is "
                "no. failures"
            )
    else:
        y = np.asarray(Y, dtype=float)
        if y.ndim != 1:
            raise ValueError("the response must be a vector for this family")

    # ---- permissible y values ------------------------------------------
    if not family.y_valid(y):
        raise ValueError("response variable out of range")

    # ---- start.from -----------------------------------------------------
    if start_from is not None:
        if not isinstance(start_from, GamlssResults):
            raise TypeError("The object in start_from is not a gamlss object")
        if "mu" in start_from.parameters:
            mu_start = start_from.mu_fv
        if "sigma" in start_from.parameters:
            sigma_start = start_from.sigma_fv
        if "nu" in start_from.parameters:
            nu_start = start_from.nu_fv
        if "tau" in start_from.parameters:
            tau_start = start_from.tau_fv

    # ---- weights ---------------------------------------------------------
    if weights is None:
        w = np.ones(N)
    else:
        if isinstance(weights, str):
            w = np.asarray(data[weights], dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
        if np.any(w < 0):
            raise ValueError("negative weights not allowed")

    # ---- per-parameter design matrices, offsets, starting values ---------
    X = {}
    offset = {}
    dinfos = {}
    starts = {"mu": mu_start, "sigma": sigma_start,
              "nu": nu_start, "tau": tau_start}
    fix = {"mu": mu_fix, "sigma": sigma_fix, "nu": nu_fix, "tau": tau_fix}
    for p, fx in fix.items():
        if not isinstance(fx, (bool, np.bool_)):
            raise TypeError(f"{p}_fix should be logical True or False")
    state = {}
    for p in PARAM_ORDER:
        if p not in family.parameters:
            continue
        Xp, di = formulas[p].design(data)
        X[p] = Xp
        dinfos[p] = di
        offset[p] = formulas[p].offset(data, N)
        st = starts[p]
        if st is not None:
            st = np.asarray(st, dtype=float)
            state[p] = (st if st.size > 1 else np.full(N, float(st)))
        else:
            ini = family.initial[p](y=y, w=w, bd=bd)
            ini = np.asarray(ini, dtype=float)
            state[p] = np.full(N, float(ini)) if ini.size == 1 else ini.copy()

    # ---- fit -------------------------------------------------------------
    setup = {
        "control": control, "i_control": i_control, "family": family,
        "state": state, "fits": {}, "y": y, "w": w, "bd": bd, "N": N,
        "X": X, "offset": offset, "fix": fix, "iter": 0,
    }
    if isinstance(method, RS):
        conv = _rs_fit(setup, n_cyc=method.n_cyc)
    elif isinstance(method, CG):
        conv = _cg_fit(setup, n_cyc=method.n_cyc)
    elif isinstance(method, mixed):
        # R: conv <- RS(n.cyc=n1, no.warn=FALSE); conv <- CG(n.cyc=n2)
        # CG restarts its iteration counter from control$iter.
        conv = _rs_fit(setup, n_cyc=method.n1, no_warn=False)
        conv = _cg_fit(setup, n_cyc=method.n2)
    else:
        raise ValueError("Method must be RS(), CG() or mixed()")

    # ---- output object -----------------------------------------------------
    full = dict(state)
    full["y"] = y
    if bd is not None:
        full["bd"] = bd
    G_dev_incr = family.g_dev_incr(**full)
    G_dev = float(np.sum(w * G_dev_incr))

    res = GamlssResults()
    res.family = family.family
    res._family_obj = family
    res.parameters = [p for p in PARAM_ORDER if p in family.parameters]
    res.call = call
    res.y = y
    res.control = control
    res.i_control = i_control
    res.weights = w
    res.G_deviance = G_dev
    res.N = N
    res.rqres_spec = family.rqres
    res.iter = setup["iter"]
    res.type = family.type
    res.method = repr(method)
    res.contrasts = contrasts
    res.converged = conv
    res.rng = rng
    res.data = data
    if bd is not None:
        res.bd = bd
    noObs = float(np.sum(w)) if np.all(np.trunc(w) == w) else N
    res.noObs = noObs

    save = control["save"]
    df_fit = 0.0
    pen = 0.0
    for p in res.parameters:
        fitted_p = setup["fits"].get(p)
        if family.parameters[p] and not fix[p] and fitted_p is not None:
            cn = r_colnames(dinfos[p])
            setattr(res, f"{p}_fv", state[p])
            setattr(res, f"{p}_lp", fitted_p["eta"])
            setattr(res, f"{p}_wv", fitted_p["wv"])
            setattr(res, f"{p}_wt", fitted_p["wt"])
            setattr(res, f"{p}_link", family.link(p))
            setattr(res, f"{p}_terms", dinfos[p])
            setattr(res, f"{p}_x", X[p])
            setattr(res, f"{p}_qr", fitted_p["qr"])
            coefs = pd.Series(fitted_p["coefficients"], index=cn)
            setattr(res, f"{p}_coefficients", coefs)
            setattr(res, f"{p}_offset", fitted_p["os"])
            setattr(res, f"{p}_formula", formulas[p].original)
            setattr(res, f"{p}_df", float(fitted_p["rank"]))
            setattr(res, f"{p}_nl_df", 0.0)
            setattr(res, f"{p}_pen", 0.0)
        else:
            setattr(res, f"{p}_fix", fix[p])
            setattr(res, f"{p}_df", 0.0)
            setattr(res, f"{p}_fv", state[p])
            setattr(res, f"{p}_pen", 0.0)
            setattr(res, f"{p}_formula", formulas[p].original)
        df_fit += getattr(res, f"{p}_df")
        pen += getattr(res, f"{p}_pen", 0.0)

    res.df_fit = df_fit
    res.pen = pen
    res.df_residual = noObs - df_fit
    res.P_deviance = G_dev + pen
    res.aic = G_dev + 2 * df_fit
    res.sbc = G_dev + np.log(noObs) * df_fit
    # R computes out$residuals <- eval(family$rqres) with the fitted
    # parameter vectors in scope; here those live on the result object.
    res.residuals = res._compute_rqres()
    return res


def _build_call_string(formula, sigma_formula, nu_formula, tau_formula,
                       family, method, weights):
    parts = [f"formula = {formula}"]
    if sigma_formula != "~1":
        parts.append(f"sigma.formula = {sigma_formula}")
    if nu_formula != "~1":
        parts.append(f"nu.formula = {nu_formula}")
    if tau_formula != "~1":
        parts.append(f"tau.formula = {tau_formula}")
    if family is not None:
        fam = family
        try:
            fam = as_gamlss_family(family).family[0]
        except Exception:
            pass
        parts.append(f"family = {fam}")
    if weights is not None:
        parts.append("weights = weights")
    return "gamlss(" + ", ".join(parts) + ")"
