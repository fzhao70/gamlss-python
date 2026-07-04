"""Warm-start (lambda persistence) tests for pb() / pbz().

R-independent: these lock the *exact* lambda-persistence behaviour of R gamlss
5.5-0, which pb() and pbz() implement asymmetrically -- so this port must too.

  - gamlss.pb (pb.R:296-297): assign(startLambdaName, lambda) sits INSIDE the
    ML loop, AFTER the convergence `break`.  On convergence the break skips it,
    so R persists the SECOND-TO-LAST lambda as the warm start.
  - gamlss.pbz (pb_goingtozero.R:291-293): assign sits AFTER the loop, so R
    persists the FINAL lambda.

Guard for issue #1: a proposed change hoisted pb's assignment out of the loop
so it persisted the final lambda (matching pbz).  That is mathematically tidier
but DIVERGES from R gamlss 5.5-0, whose bit-for-bit reproduction is this port's
whole purpose.  These tests fail if pb is changed to persist the final lambda,
or if pbz is changed to persist the second-to-last.
"""

from __future__ import annotations

import numpy as np

from gamlss.smooth import PB, PBZ


def _wiggly_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    y = np.sin(2 * np.pi * x) + rng.normal(0.0, 0.2, n)
    return x, y, np.ones(n)


def test_pb_persists_second_to_last_lambda_like_r():
    """pb() persists the second-to-last ML iterate (R pb.R:296-297).

    After a converged multi-iteration ML fit, ``lambda_start`` holds the iterate
    from *before* the breaking iteration -- so it differs from the returned
    (final) lambda, but only by the convergence tolerance.  Hoisting the assign
    out of the loop (persisting the final lambda) makes ``lambda_start`` equal
    the returned lambda and fails this test.
    """
    x, y, w = _wiggly_data()
    sm = PB(x)                                   # ML, cold start 10.0
    r = sm.fit(y, w)
    # the final assign is skipped by `break`, so the persisted warm start is the
    # prior iterate -- distinct from the returned lambda ...
    assert sm.lambda_start != r["lambda"]
    # ... but only by less than the ML convergence tolerance
    assert abs(sm.lambda_start - r["lambda"]) < 1e-7


def test_pbz_persists_final_lambda_like_r():
    """pbz() persists the FINAL ML iterate (R pb_goingtozero.R:291-293)."""
    x, y, w = _wiggly_data(seed=2)
    sm = PBZ(x)
    r = sm.fit(y, w)
    assert sm.lambda_start == r["lambda"]
