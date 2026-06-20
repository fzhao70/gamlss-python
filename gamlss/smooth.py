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
import warnings

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_triangular
from scipy.optimize import brentq, minimize, minimize_scalar

# R's qr/svd rank tolerance: .Machine$double.eps^0.8
_EPS_08 = float(np.finfo(float).eps) ** 0.8


class _NaturalSpline:
    """R's ``splinefun(x, fv, method="natural")`` for pb prediction.

    A natural cubic spline through the fitted smoother values.  Crucially it
    extrapolates *linearly* outside the data range (continuing the boundary
    value and slope), exactly as R's splinefun does -- scipy's CubicSpline
    instead extends the boundary cubic, which diverges sharply beyond the
    range.  Ties in x map to identical fv (the fit is a function of x), so
    duplicates are dropped.
    """

    def __init__(self, x, fv):
        xu, idx = np.unique(np.asarray(x, dtype=float), return_index=True)
        self._cs = CubicSpline(xu, np.asarray(fv, dtype=float)[idx],
                               bc_type="natural")
        self._x0, self._x1 = float(xu[0]), float(xu[-1])
        self._f0, self._f1 = float(self._cs(self._x0)), float(self._cs(self._x1))
        self._d0 = float(self._cs(self._x0, 1))  # boundary slopes
        self._d1 = float(self._cs(self._x1, 1))

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        xa = np.atleast_1d(x)
        out = self._cs(xa)
        below, above = xa < self._x0, xa > self._x1
        out[below] = self._f0 + self._d0 * (xa[below] - self._x0)
        out[above] = self._f1 + self._d1 * (xa[above] - self._x1)
        return float(out[0]) if scalar else out


def _natural_spline(x, fv):
    return _NaturalSpline(x, fv)


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
                warnings.warn("The df's exceed the number of columns of the "
                              "design matrix\n   they are set to 3")
                df = 3
            if df < 0:
                warnings.warn("the extra df's are set to 0")
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

    # -- eigenvalues of Rinv^T (D^T D) Rinv: the basis for edf(lambda) ----
    @staticmethod
    def _eig_RinvSRinv(Rmat, D, name=None, vectors=False):
        """Eigen-decomposition of ``Rinv^T S Rinv`` (S = D^T D), pb.R:85-88.

        ``edf(lambda) = sum 1 / (1 + lambda * values)`` equals the trace of
        the penalised hat matrix, so this drives GCV and the df/max.df
        root-finding.  Like R (``try(solve(R))``, pb.R:98/115), a rank-
        deficient basis (e.g. too few distinct x) raises a clear error rather
        than silently producing a degenerate fit.
        """
        diag = np.abs(np.diag(Rmat))
        if diag.size == 0 or diag.min() <= diag.max() * _EPS_08:
            raise ValueError(
                f"The B-basis for {name} is singular, "
                "transforming the variable may help")
        Rinv = solve_triangular(Rmat, np.eye(Rmat.shape[0]))
        S = D.T @ D
        M = Rinv.T @ S @ Rinv
        if vectors:
            vals, vecs = np.linalg.eigh(M)
            return vals, vecs
        return np.linalg.eigvalsh(M)

    @staticmethod
    def _lambda_for_edf(vals, target):
        """log-lambda solving ``edf(lambda) == target`` via uniroot, pb.R:120.

        edf is monotone decreasing in log-lambda; bracket on [-30, 30] and
        fall back to 30 when the target is unreachable in range (as R does).
        """
        def edf_minus(loglam):
            return float(np.sum(1.0 / (1.0 + math.exp(loglam) * vals))) - target

        if np.sign(edf_minus(-30.0)) == np.sign(edf_minus(30.0)):
            loglam = 30.0
        else:
            loglam = brentq(edf_minus, -30.0, 30.0, xtol=1e-10)
        return math.exp(loglam)

    def fit(self, y, w):
        """Fit the smoother to working response ``y`` with weights ``w``.

        Port of gamlss.pb (pb.R:166-209).  Smoothing-parameter selection:
        ML (default), GAIC and GCV (``method=``), a fixed ``lambda``, a
        target ``df``, or a ``max.df`` cap.  ML and fixed-lambda match R to
        ~1e-13; df/max.df (R ``uniroot``) and GAIC/GCV (R ``nlminb``) are
        optimiser-dependent and match R to ~1e-4.
        """
        y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)
        X, D = self.X, self.D
        order = self.order
        N = int(np.sum(w != 0))                         # pb.R:256
        n = X.shape[0]                                  # pb.R:257 (for GCV)
        p = D.shape[1]
        k = self.k

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
            if self.method == "ML":
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
                    self.lambda_start = lam    # persist (warm start), pb.R:35
            elif self.method == "GAIC":
                # minimise local GAIC = sum w (y-fv)^2 + k*edf, pb.R:76-81.
                # R uses nlminb -- a *local* search from the warm-start lambda
                # (NOT global), so we mirror that with L-BFGS-B from the same
                # start.  NOTE: GAIC parity with R is not guaranteed -- the
                # objective can be flat/multimodal and L-BFGS-B vs PORT-nlminb
                # may descend to different optima (see CHANGELOG); the result
                # is still a valid GAIC-selected smooth.
                def gaic(lam_arr):
                    b, e = self._regpen(Rmat, Qy, D, float(lam_arr[0]), p)
                    return float(np.sum(w * (y - X @ b) ** 2) + k * e)

                lam = float(minimize(gaic, x0=[lam], method="L-BFGS-B",
                                     bounds=[(1e-7, 1e7)]).x[0])
                beta, edf = self._regpen(Rmat, Qy, D, lam, p)
                fv = X @ beta
                self.lambda_start = lam
            elif self.method == "GCV":
                # minimise generalised cross-validation, pb.R:82-93
                vals, vecs = self._eig_RinvSRinv(Rmat, D, self.name,
                                                 vectors=True)
                yy = vecs.T @ Qy
                y_y = float(np.sum((sw * y) ** 2))

                def gcv(loglam):
                    ild = 1.0 + math.exp(loglam) * vals
                    edf_ = np.sum(1.0 / ild)
                    y_hy2 = (y_y - 2 * np.sum(yy ** 2 / ild)
                             + np.sum(yy ** 2 / ild ** 2))
                    return float((n * y_hy2) / (n - k * edf_) ** 2)

                lam = math.exp(minimize_scalar(
                    gcv, bounds=(math.log(1e-7), math.log(1e7)),
                    method="bounded").x)
                beta, edf = self._regpen(Rmat, Qy, D, lam, p)
                fv = X @ beta
                self.lambda_start = lam
            else:
                raise ValueError(f"unknown pb method {self.method!r}")
            # max.df cap, pb.R:96-110
            if edf > self.max_df:
                lam = self._lambda_for_edf(
                    self._eig_RinvSRinv(Rmat, D, self.name), self.max_df)
                beta, edf = self._regpen(Rmat, Qy, D, lam, p)
                fv = X @ beta
                self.lambda_start = lam
        elif self.lambda_ is not None:
            # fixed lambda, pb.R(263-dump):9-12
            lam = float(self.lambda_)
            beta, edf = self._regpen(Rmat, Qy, D, lam, p)
            fv = X @ beta
        else:
            # fixed df: solve lambda so that edf == df, pb.R:113-129
            lam = self._lambda_for_edf(
                self._eig_RinvSRinv(Rmat, D, self.name), self.df)
            beta, edf = self._regpen(Rmat, Qy, D, lam, p)
            fv = X @ beta

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
