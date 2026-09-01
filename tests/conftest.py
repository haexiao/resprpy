"""Shared helpers for comparing resprpy output against R reference CSVs."""
import os

import numpy as np

REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference")


def load_ref_csv(name):
    """Load an R write.csv reference file as a float array (no header)."""
    path = os.path.join(REF_DIR, name)
    return np.genfromtxt(path, delimiter=",", skip_header=1)


def assert_close(actual, expected, rtol=1e-6, atol=1e-10, label=""):
    """Assert resprpy output matches R reference within tolerances.

    rtol/atol follow np.allclose semantics but with a useful failure message
    showing the actual relative deviations.
    """
    a = np.asarray(actual, dtype=float).ravel()
    e = np.asarray(expected, dtype=float).ravel()
    if a.size != e.size:
        raise AssertionError(f"{label}: size {a.size} != R size {e.size}")
    denom = np.maximum(np.abs(e), 1e-300)
    rel = np.abs(a - e) / denom
    tol = rtol + atol / denom
    bad = rel > tol
    if np.any(bad):
        idx = np.where(bad)[0]
        shown = idx[:5]
        msg = (f"{label}: {len(idx)}/{e.size} values out of tolerance "
               f"(rtol={rtol}, atol={atol}).\n"
               f"  py = {a[shown]}\n"
               f"  R  = {e[shown]}\n"
               f"  rel = {rel[shown]}")
        raise AssertionError(msg)
