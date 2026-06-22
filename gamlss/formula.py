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
from collections import OrderedDict

import numpy as np
import pandas as pd
import patsy

from .smooth import PB, PBZ


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


# ----------------------------------------------------------- smoothers
_SMOOTHER_RE = re.compile(r"(pb|pbz)\s*\((.*)\)\s*$", re.S)
# R argument name -> PB constructor keyword
_PB_KW = {"lambda": "lambda_", "max.df": "max_df"}
_KW_ENV = {"TRUE": True, "FALSE": False, "T": True, "F": False, "NULL": None}


def _r_num(x):
    """Format a numeric like R's ``deparse``/``format(x, digits=15)``.

    Smoother coefficient labels come from R's ``deparse(sys.call())``, which
    re-renders numeric literals: 15 significant digits, scientific notation
    only when *strictly* shorter than fixed (scipen = 0, ties -> fixed), with
    a signed two-or-more-digit exponent.  So ``pb(x, lambda = 1000000)`` is
    labelled ``pb(x, lambda = 1e+06)`` and ``lambda = 10000`` stays ``10000``.
    """
    x = float(x)
    if x != x:               # NaN
        return "NaN"
    if x == 0.0:
        return "0"
    neg = x < 0
    mant, exp = ("%.*e" % (14, abs(x))).split("e")
    exp = int(exp)
    mant = mant.rstrip("0").rstrip(".")
    digits = mant.replace(".", "")
    ndig = len(digits)
    sci = "%se%+03d" % (mant, exp)
    if exp >= ndig - 1:                       # integer (with trailing zeros)
        fixed = digits + "0" * (exp - (ndig - 1))
    elif exp >= 0:                            # decimal point inside the digits
        fixed = digits[:exp + 1] + "." + digits[exp + 1:]
    else:                                     # leading zeros
        fixed = "0." + "0" * (-exp - 1) + digits
    chosen = sci if len(sci) < len(fixed) else fixed
    return "-" + chosen if neg else chosen


def _is_smoother(term):
    """True if a formula term is a pb()/pbz() smoother call."""
    return _SMOOTHER_RE.fullmatch(term.strip()) is not None


def _split_args(s):
    """Split a call's argument string on top-level commas."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [a.strip() for a in out if a.strip()]


def _parse_smoother(term, data, env):
    """Parse ``pb(x, ...)``/``pbz(x, ...)`` -> (label, smoother, design column).

    The first positional argument is the smoothed variable (evaluated in the
    data/formula environment); remaining ``key=value`` args are mapped to the
    PB/PBZ constructor (``lambda`` -> ``lambda_``, ``max.df`` -> ``max_df``).
    The third return value is the column that goes into the *parametric*
    design: the linear ``x`` for ``pb()`` (pb.R:116) or a column of zeros for
    ``pbz()`` (pbz.R:89).
    """
    m = _SMOOTHER_RE.fullmatch(term.strip())
    kind, inner = m.group(1), m.group(2)
    args = _split_args(inner)
    if not args:
        raise ValueError(f"smoother {term!r} needs a variable")
    xval = eval(_r_to_patsy(args[0]), {"__builtins__": {}}, _DataEnv(data, env))
    if isinstance(xval, pd.Series):
        xval = xval.to_numpy()
    xval = np.asarray(xval, dtype=float)
    kwargs = {}
    norm_args = [args[0]]  # the smoothed variable, kept verbatim
    for a in args[1:]:
        if "=" not in a:
            raise ValueError(f"unexpected positional argument in {term!r}: {a!r}")
        k, v = a.split("=", 1)
        k, v = k.strip(), v.strip()
        key = _PB_KW.get(k, k.replace(".", "_"))
        kwargs[key] = eval(v, {"__builtins__": {}}, dict(_KW_ENV))
        # R's deparse re-renders numeric literals (1000000 -> 1e+06); keep
        # non-numeric args (strings, TRUE/FALSE, expressions) verbatim.
        try:
            norm_args.append(f"{k} = {_r_num(float(v))}")
        except (ValueError, TypeError):
            norm_args.append(f"{k} = {v}")
    label = f"{kind}({', '.join(norm_args)})"  # R-style term label
    if kind == "pbz":
        pb = PBZ(xval, **kwargs)
        pb.name = args[0].strip()
        # pbz() puts a column of ZEROS in the parametric design (pbz.R:89),
        # not the linear x that pb() uses -- the whole effect lives in the
        # smooth and the intercept absorbs the constant.
        return label, pb, np.zeros_like(xval)
    pb = PB(xval, **kwargs)
    pb.name = args[0].strip()
    return label, pb, xval


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
        """Build the design matrix; returns (X, design_info, smoothers).

        Parametric terms go through patsy and are reordered to R's term
        order.  pb()/pbz() smoother terms are pulled out: each contributes
        its *linear* column (named by the call, e.g. ``"pb(x)"``) appended
        after the parametric columns -- exactly as R's pb() puts ``xvar <- x``
        in the design matrix -- and a PB object that the engine fits by
        backfitting.  ``smoothers`` is an OrderedDict {label: PB} (empty when
        the formula has no smoothers).
        """
        par_terms, smooth_terms = [], []
        for t in _split_terms(self.rhs):
            (smooth_terms if _is_smoother(t) else par_terms).append(t)
        if smooth_terms:
            par_rhs = " + ".join(par_terms) if par_terms else "1"
        else:
            par_rhs = self.rhs  # unchanged path when no smoothers

        rhs = _r_to_patsy(par_rhs)
        X = patsy.dmatrix(
            rhs, data, eval_env=patsy.EvalEnvironment([self.env]),
            return_type="matrix", NA_action="raise",
        )
        di = X.design_info
        di, perm = _reorder_like_r(di, par_rhs)
        Xpar = np.asarray(X, dtype=float)[:, perm]

        smoothers = OrderedDict()
        cols = []
        for t in smooth_terms:
            label, pb, xcol = _parse_smoother(t, data, self.env)
            smoothers[label] = pb
            cols.append(xcol)
        X_full = np.hstack([Xpar, np.column_stack(cols)]) if cols else Xpar
        return X_full, di, smoothers

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
