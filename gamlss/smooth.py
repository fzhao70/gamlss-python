"""Penalised B-spline smoothers for gamlss (pb / pbz).

Port of the R gamlss smoother machinery:
- ``bbase``      : Paul Eilers' B-spline basis (truncated-power form), pb.R:31-61
- ``_penalty``   : order-d difference penalty,                          pb.R:80
- ``PB``         : the pb() term object + its gamlss.pb() fitting method,
                   including ``regpen`` (SVD penalised least squares) and the
                   ML smoothing-parameter loop,                          pb.R:166-209

The class is deliberately a *pure function of* ``(x, control)`` for construction
and ``(y, w)`` for fitting, so it can be unit-tested in isolation before being
wired into the backfitting/engine machinery.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline

# R's qr/svd rank tolerance: .Machine$double.eps^0.8
_EPS_08 = float(np.finfo(float).eps) ** 0.8


def _natural_spline(x, fv):
    """R's ``splinefun(x, fv, method="natural")`` for pb prediction.

    A natural cubic spline through the fitted smoother values; used by
    predict() to evaluate the smooth at new x.  Ties in x map to identical
    fv (the fit is a function of x), so duplicates are dropped.
    """
    xu, idx = np.unique(np.asarray(x, dtype=float), return_index=True)
    return CubicSpline(xu, np.asarray(fv, dtype=float)[idx],
                       bc_type="natural", extrapolate=True)


# --------------------------------------------------------------- basis
def bbase(x, xl, xr, ndx, deg):
    """B-spline basis on [xl, xr] with ndx intervals, degree ``deg``.

    Paul Eilers' truncated-power construction, identical to pb.R:48-60::

        knots = seq(xl - deg*dx, xr + deg*dx, by = dx)
        P     = outer(x, knots, (x-t)^deg * (x>t))
        D     = diff(diag(nknots), diff = deg+1) / (gamma(deg+1) * dx^deg)
        B     = (-1)^(deg+1) * P %*% t(D)
    """
    x = np.asarray(x, dtype=float)
    dx = (xr - xl) / ndx
    nk = ndx + 2 * deg + 1
    # R seq(a, by=dx): a + (0:n)*dx  (avoid arange/linspace endpoint drift)
    knots = (xl - deg * dx) + np.arange(nk) * dx
    diff_ = x[:, None] - knots[None, :]
    P = np.where(diff_ > 0, diff_ ** deg, 0.0)  # tpower: (x-t)^deg * (x>t)
    Dk = np.diff(np.eye(nk), n=deg + 1, axis=0) / (math.factorial(deg) * dx ** deg)
    B = (-1.0) ** (deg + 1) * P @ Dk.T
    return B, knots


def _penalty(r, order):
    """Order-``order`` difference penalty matrix; pb.R:80."""
    if order == 0:
        return np.eye(r)
    return np.diff(np.eye(r), n=order, axis=0)


# ----------------------------------------------------------- pb object
class PB:
    """A penalised B-spline term (R's ``pb()`` + ``gamlss.pb()``).

    Construction mirrors pb() (pb.R:23-129): build the basis and penalty,
    apply the knot-interval clamps, and remember the starting lambda.
    ``fit(y, w)`` mirrors gamlss.pb() (pb.R:166-209).
    """

    def __init__(self, x, df=None, lambda_=None, max_df=None, *,
                 inter=20, degree=3, order=2, start=10.0, method="ML", k=2):
        x = np.asarray(x, dtype=float)
        self.x = x
        self.name = None  # filled in by the formula layer (the variable name)
        lx = x.size
        n_distinct = np.unique(x).size
        # pb.R:69-70 -- interval clamps (small n / few distinct values)
        if lx < 99:
            inter = 10
        if n_distinct <= inter:
            inter = n_distinct
        self.inter = inter
        self.degree = degree
        self.order = order
        self.method = method
        self.k = k
        self.lambda_start = float(start)
        self.lambda_ = lambda_

        # padded range, pb.R:71-74
        xl, xr = float(x.min()), float(x.max())
        rng = xr - xl
        xmax = xr + 0.01 * rng
        xmin = xl - 0.01 * rng
        self.X, self.knots = bbase(x, xmin, xmax, inter, degree)
        r = self.X.shape[1]
        self.D = _penalty(r, order)

        # df handling, pb.R:82-89 (df -> df + 2, with bounds)
        if df is not None:
            if df > (r - 2):
                df = 3
            df = 2 if df < 0 else df + 2
        self.df = df
        # max.df, pb.R:91-96
        self.max_df = r if max_df is None else min(max_df, r)

    # -- the SVD penalised least squares core (regpen, pb.R:172-191) ------
    @staticmethod
    def _regpen(Rmat, Qy, D, lam, p):
        RD = np.vstack([Rmat, math.sqrt(lam) * D])      # (p + nrowD) x p
        U, d, _Vt = np.linalg.svd(RD, full_matrices=False)
        rank = int(np.sum(d > d.max() * _EPS_08))
        U1 = U[:p, :rank]
        y1 = U1.T @ Qy
        beta = _Vt.T[:, :rank] @ (y1 / d[:rank])
        edf = float(np.sum(U1 * U1))                    # trace(U1 U1^T)
        return beta, edf

    def fit(self, y, w):
        """Fit the smoother to working response ``y`` with weights ``w``.

        Port of gamlss.pb (pb.R:166-209).  Currently implements the default
        ML smoothing-parameter selection and the fixed-lambda case.
        """
        y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)
        X, D = self.X, self.D
        order = self.order
        N = int(np.sum(w != 0))                         # pb.R:256
        p = D.shape[1]

        # QR once (pb.R:259-262); regpen reuses R/Qy across the lambda loop
        sw = np.sqrt(w)
        Q, Rmat = np.linalg.qr(X * sw[:, None])
        Qy = Q.T @ (sw * y)

        tau2 = sig2 = None
        # warm-start clamp, pb.R(263-dump):4-6
        lam = self.lambda_start
        if lam >= 1e7:
            lam = 1e7
        if lam <= 1e-7:
            lam = 1e-7

        if self.df is None and self.lambda_ is None:
            if self.method != "ML":
                raise NotImplementedError(
                    f"pb method {self.method!r} not yet implemented (Step 5)")
            # ML loop, pb.R:20-37
            for _ in range(50):
                beta, edf = self._regpen(Rmat, Qy, D, lam, p)
                gamma = D @ beta
                fv = X @ beta
                sig2 = float(np.sum(w * (y - fv) ** 2) / (N - edf))
                tau2 = float(np.sum(gamma ** 2) / (edf - order))
                if tau2 < 1e-7:
                    tau2 = 1e-7
                lam_old = lam
                lam = sig2 / tau2
                if lam < 1e-7:
                    lam = 1e-7
                if lam > 1e7:
                    lam = 1e7
                if abs(lam - lam_old) < 1e-7 or lam > 1e10:
                    break
                self.lambda_start = lam        # persist (warm start), pb.R:35
        elif self.df is None or self.lambda_ is not None:
            # fixed lambda, pb.R(263-dump):9-12
            lam = float(self.lambda_)
            beta, edf = self._regpen(Rmat, Qy, D, lam, p)
            fv = X @ beta
        else:
            raise NotImplementedError("fixed-df pb not yet implemented (Step 5)")

        return {
            "fitted.values": fv,
            "fv": fv,                     # R coefSmo$fv alias
            "residuals": y - fv,
            "nl_df": edf - 2,
            "lambda": lam,
            "edf": edf,
            "coef": beta,
            "sig2": sig2,
            "tau2": tau2,
            "knots": self.knots,
            "name": self.name,            # smoothed-variable expression
            "x": self.x,                  # training x (for the spline)
            "fun": _natural_spline(self.x, fv),  # predict: fun(xeval)
        }
