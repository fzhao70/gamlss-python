"""R-style model formulas for gamlss, built on patsy.

Provides:
- R-compatible ``poly()`` (orthogonal polynomials, identical algorithm
  to stats::poly, including prediction via the 3-term recurrence),
- ``cbind()`` for binomial responses,
- in-formula ``offset()`` terms (patsy has no native support),
- R-style design-matrix column names ("(Intercept)", "sexF",
  "poly(x, 2)1", ...).

R formulas use ``^`` for crossing and power inside I(); both meanings
map to Python's ``**`` under patsy, so ``^`` is rewritten to ``**``.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import patsy


# ---------------------------------------------------------------- poly
class _Poly:
    """R's stats::poly as a patsy stateful transform."""

    def __init__(self):
        self._x = []
        self.coefs = None
        self.degree = None
        self.raw = False

    def memorize_chunk(self, x, degree=1, raw=False):
        self.degree = int(degree)
        self.raw = bool(raw)
        self._x.append(np.asarray(x, dtype=float))

    def memorize_finish(self):
        if self.raw:
            return
        x = np.concatenate(self._x)
        n = self.degree
        if n < 1:
            raise ValueError("'degree' must be at least 1")
        if n >= len(np.unique(x)):
            raise ValueError("'degree' must be less than number of unique points")
        xbar = x.mean()
        xc = x - xbar
        X = np.vander(xc, N=n + 1, increasing=True)
        Q, R = np.linalg.qr(X)
        # raw.p <- qr.qy(QR, z) with z = diag(diag(R)):  Q[:,j]*R[j,j]
        rawp = Q * np.diag(R)
        norm2 = (rawp**2).sum(axis=0)
        alpha = ((xc[:, None] * rawp**2).sum(axis=0) / norm2 + xbar)[:n]
        self.coefs = {"alpha": alpha, "norm2": np.concatenate([[1.0], norm2])}

    def transform(self, x, degree=1, raw=False):
        x = np.asarray(x, dtype=float)
        n = int(degree)
        if self.raw or raw:
            Z = np.vander(x, N=n + 1, increasing=True)[:, 1:]
            return Z
        alpha = self.coefs["alpha"]
        norm2 = self.coefs["norm2"]  # length n+2, leading 1
        Z = np.empty((len(x), n + 1))
        Z[:, 0] = 1.0
        if n > 0:
            Z[:, 1] = x - alpha[0]
        for i in range(2, n + 1):
            Z[:, i] = (x - alpha[i - 1]) * Z[:, i - 1] - (
                norm2[i] / norm2[i - 1]
            ) * Z[:, i - 2]
        Z = Z / np.sqrt(norm2[1:])
        return Z[:, 1:]


poly = patsy.stateful_transform(_Poly)


def cbind(*cols):
    """R's cbind for two-column binomial responses."""
    return np.column_stack([np.asarray(c, dtype=float) for c in cols])


# base environment available inside formulas (R-ish function names)
def _base_env():
    return {
        "poly": poly,
        "cbind": cbind,
        "log": np.log,
        "log2": np.log2,
        "log10": np.log10,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "floor": np.floor,
        "ceiling": np.ceil,
        "np": np,
        "pd": pd,
    }


# ------------------------------------------------------- formula utils
def _split_terms(rhs):
    """Split an RHS on top-level '+' (keeping other operators intact)."""
    terms, depth, cur = [], 0, ""
    for ch in rhs:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "+" and depth == 0:
            terms.append(cur)
            cur = ""
        else:
            cur += ch
    terms.append(cur)
    return [t.strip() for t in terms if t.strip()]


def _extract_offsets(rhs):
    """Remove top-level offset(...) terms; return (clean_rhs, offset_exprs)."""
    terms = _split_terms(rhs)
    keep, offsets = [], []
    for t in terms:
        m = re.fullmatch(r"offset\((.*)\)", t, flags=re.S)
        if m:
            offsets.append(m.group(1))
        else:
            keep.append(t)
    if not keep:
        keep = ["1"]
    return " + ".join(keep), offsets


def _r_to_patsy(formula):
    """Rewrite R-isms into patsy syntax (currently: ^ -> **)."""
    return formula.replace("^", "**")


_TL = re.compile(r"\[T\.([^\]]+)\]")  # treatment-coded level
_LV = re.compile(r"\[([^\]]+)\]")  # full-rank level or column index


def _r_colname(name):
    """Translate one patsy column name into the R-style equivalent."""
    if name == "Intercept":
        return "(Intercept)"
    # interaction components are ':'-joined in both systems
    parts = name.split(":")
    out = []
    for part in parts:
        p = part
        # C(f)[T.lev] or C(f, ...)[...] -> f...
        m = re.match(r"^C\(([^,()]+)[^)]*\)(\[.*\])?$", p)
        if m:
            p = m.group(1) + (m.group(2) or "")
        p = _TL.sub(lambda m: m.group(1), p)

        def _idx(m):
            s = m.group(1)
            if re.fullmatch(r"\d+", s):
                return str(int(s) + 1)  # 0-based patsy index -> 1-based R
            return s

        p = _LV.sub(_idx, p)
        p = p.replace(" ** ", "^").replace("**", "^")
        out.append(p)
    return ":".join(out)


def r_colnames(design_info):
    return [_r_colname(c) for c in design_info.column_names]


class ParamFormula:
    """One model formula (for mu, sigma, nu or tau) and its design tools."""

    def __init__(self, formula, data, context=None, lhs_required=False):
        self.original = formula if isinstance(formula, str) else str(formula)
        f = self.original.strip()
        if "~" in f:
            lhs, rhs = f.split("~", 1)
        else:
            lhs, rhs = "", f
        self.lhs = lhs.strip()
        rhs = rhs.strip() or "1"
        rhs, offset_exprs = _extract_offsets(rhs)
        self.rhs = rhs
        self.offset_exprs = offset_exprs
        if lhs_required and not self.lhs:
            raise ValueError(f"formula '{formula}' needs a response (lhs)")
        self.env = _base_env()
        if context:
            self.env.update(context)

    # -- evaluation -----------------------------------------------------
    def response(self, data):
        """Evaluate the LHS (returns None for one-sided formulas)."""
        if not self.lhs:
            return None
        expr = _r_to_patsy(self.lhs)
        y = eval(expr, {"__builtins__": {}}, _DataEnv(data, self.env))
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        return np.asarray(y)

    def design(self, data):
        """Build the design matrix; returns (X ndarray, design_info).

        Columns are reordered to R's term ordering (stable sort by
        interaction degree, then order of appearance in the formula);
        patsy sorts categorical terms first, R does not.
        """
        rhs = _r_to_patsy(self.rhs)
        X = patsy.dmatrix(
            rhs, data, eval_env=patsy.EvalEnvironment([self.env]),
            return_type="matrix", NA_action="raise",
        )
        di = X.design_info
        di, perm = _reorder_like_r(di, self.rhs)
        X = np.asarray(X, dtype=float)[:, perm]
        return X, di

    def design_like(self, design_info, data):
        """Design matrix for new data using a memorised design."""
        (X,) = patsy.build_design_matrices(
            [design_info], data, return_type="matrix", NA_action="raise"
        )
        return np.asarray(X, dtype=float)

    def offset(self, data, N):
        """Evaluate (and sum) all offset() terms; zeros if none."""
        if not self.offset_exprs:
            return np.zeros(N)
        total = np.zeros(N)
        for expr in self.offset_exprs:
            expr = _r_to_patsy(expr)
            v = eval(expr, {"__builtins__": {}}, _DataEnv(data, self.env))
            if isinstance(v, pd.Series):
                v = v.to_numpy()
            total = total + np.asarray(v, dtype=float)
        return total

    def __repr__(self):  # pragma: no cover
        return self.original if "~" in self.original else "~" + self.original


def _normalize_code(s):
    return s.replace("^", "**").replace(" ", "")


def _reorder_like_r(design_info, rhs):
    """Permute patsy terms into R's model.matrix order.

    R sorts terms by interaction degree (keep.order=FALSE) and keeps
    the order of appearance in the formula within a degree.  Returns
    (new_design_info, column_permutation).
    """
    from collections import OrderedDict

    chunks = [_normalize_code(c) for c in _split_terms(rhs)]
    terms = list(design_info.terms)

    def sort_key(idx_term):
        _, term = idx_term
        if not term.factors:
            return (-1, -1, -1)
        degree = len(term.factors)
        chunk_idx = len(chunks)
        pos = 10**9
        for f in term.factors:
            code = _normalize_code(f.code)
            for i, ch in enumerate(chunks):
                if ch == code or code in ch:
                    if i < chunk_idx or (i == chunk_idx):
                        p = ch.find(code)
                        if (i, p) < (chunk_idx, pos):
                            chunk_idx, pos = i, p
                    break
        return (degree, chunk_idx, pos)

    order = sorted(enumerate(terms), key=sort_key)
    if [i for i, _ in order] == list(range(len(terms))):
        return design_info, np.arange(len(design_info.column_names))

    new_codings = OrderedDict()
    perm = []
    new_names = []
    for _, term in order:
        sl = design_info.term_slices[term]
        perm.extend(range(sl.start, sl.stop))
        new_names.extend(design_info.column_names[sl])
        new_codings[term] = design_info.term_codings[term]
    new_di = patsy.DesignInfo(new_names, factor_infos=design_info.factor_infos,
                              term_codings=new_codings)
    return new_di, np.asarray(perm)


class _DataEnv(dict):
    """Mapping that resolves names from the data frame first, then env."""

    def __init__(self, data, env):
        super().__init__()
        self._data = data
        self._env = env

    def __missing__(self, key):
        if self._data is not None and key in self._data:
            col = self._data[key]
            return col.to_numpy() if isinstance(col, pd.Series) else col
        if key in self._env:
            return self._env[key]
        raise KeyError(key)
