"""Verify d/p/q functions against values from the original R gamlss.dist."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

import gamlss.dist as dist

HERE = os.path.dirname(__file__)

with open(os.path.join(HERE, "reference", "dpqr.json")) as fh:
    DPQR = json.load(fh)

# families whose q function uses iterative root finding: R's uniroot has
# tol = .Machine$double.eps^0.25 ~ 1.2e-4, so R's own reference values
# carry that error; Python (brentq, xtol=2e-12) is more precise.
LOOSE_Q = {"IG": 2e-4, "PIG": 2e-4, "ZINBI": 1e-8, "ZANBI": 1e-8}

PARAMS = [
    (fam, i) for fam, cases in DPQR.items() for i in range(len(cases))
]


def _ids():
    return [f"{fam}-{i}" for fam, i in PARAMS]


@pytest.mark.parametrize("fam,i", PARAMS, ids=_ids())
def test_dpqr(fam, i):
    case = DPQR[fam][i]
    params = {k: np.asarray(v, dtype=float)
              for k, v in case["params"].items()}
    x = np.asarray(case["x"], dtype=float)
    ps = np.asarray(case["ps"], dtype=float)

    dfun = getattr(dist, "d" + fam, None)
    pfun = getattr(dist, "p" + fam, None)
    qfun = getattr(dist, "q" + fam, None)
    if dfun is None:
        pytest.skip(f"{fam} not implemented")

    d = dfun(x, **params)
    np.testing.assert_allclose(d, case["d"], rtol=1e-10, atol=1e-300,
                               err_msg=f"d{fam} density")
    dlog = dfun(x, log=True, **params)
    np.testing.assert_allclose(dlog, case["dlog"], rtol=1e-10, atol=1e-12,
                               err_msg=f"d{fam} log density")
    p = pfun(x, **params)
    np.testing.assert_allclose(p, case["p"], rtol=1e-10, atol=1e-300,
                               err_msg=f"p{fam} cdf")
    # upper tail: R computes 1-cdf for several families (catastrophic
    # cancellation near 0) and R/scipy incomplete-beta tails differ at
    # ~1e-8 relative in the extreme tail; use a tail-appropriate
    # tolerance.
    pu = pfun(x, lower_tail=False, **params)
    np.testing.assert_allclose(pu, case["pupper"], rtol=1e-7, atol=4e-15,
                               err_msg=f"p{fam} upper cdf")
    q = qfun(ps, **params)
    rtol = LOOSE_Q.get(fam, 1e-9)
    np.testing.assert_allclose(q, case["q"], rtol=rtol, atol=1e-12,
                               err_msg=f"q{fam} quantile")
