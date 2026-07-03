"""Warm-start (lambda persistence) regression tests for pb() / pbz().

R-independent: these assert an internal invariant of the ML smoothing-parameter
loop, so they run without the R reference JSONs.

Regression for issue #1: gamlss.pb() persists the *final* converged lambda as
the warm start for the next fit() (R does assign(startLambdaName, lambda) after
the ML loop, pb.R:35).  PB.fit previously assigned self.lambda_start inside the
loop, positioned after the convergence `break`, so on convergence the final
iterate was never stored -- the next fit() warm-started one step stale, which
desynchronised the RS iteration trajectory from R.  PBZ.fit already persisted
after its loop; this checks both stay correct.
"""

from __future__ import annotations

import numpy as np

from gamlss.smooth import PB, PBZ


def _wiggly_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    y = np.sin(2 * np.pi * x) + rng.normal(0.0, 0.2, n)
    return x, y, np.ones(n)


def test_pb_ml_persists_final_lambda():
    """After a converged ML pb fit, lambda_start == the returned lambda.

    With the pre-fix code lambda_start held the *second-to-last* iterate, which
    differs from the returned (final) lambda -- so this equality fails.
    """
    x, y, w = _wiggly_data()
    sm = PB(x)                                   # ML, cold start 10.0
    r = sm.fit(y, w)
    assert sm.lambda_start == r["lambda"]        # exact: final iterate stored


def test_pb_ml_warm_start_reproduces_fit():
    """Re-fitting the same data from the persisted warm start reproduces the
    converged fit (within the ML loop's 1e-7 tolerance).  The single-call
    invariant lambda_start == returned lambda still holds exactly."""
    x, y, w = _wiggly_data(seed=1)
    sm = PB(x)
    r1 = sm.fit(y, w)                             # cold start -> converged lam
    r2 = sm.fit(y, w)                             # warm start from convergence
    assert sm.lambda_start == r2["lambda"]        # exact single-call invariant
    # warm-starting at (near) the fixed point returns essentially the same
    # lambda and fit -- a shift < the 1e-7 ML tolerance, not a fresh descent
    np.testing.assert_allclose(r2["lambda"], r1["lambda"], atol=1e-6)
    np.testing.assert_allclose(r2["fitted.values"], r1["fitted.values"],
                               atol=1e-6)


def test_pbz_ml_persists_final_lambda():
    """PBZ.fit already persists after its ML loop; guard it stays that way."""
    x, y, w = _wiggly_data(seed=2)
    sm = PBZ(x)
    r = sm.fit(y, w)
    assert sm.lambda_start == r["lambda"]
