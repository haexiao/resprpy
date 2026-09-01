"""Compare ported marelac physics against R reference values.

Note: these are marelac (dependency) functions, not respR exports; they are
kept internal to resprpy and tested here to guarantee convert_DO parity.
"""
import numpy as np

from resprpy._marelac import (gas_satconc, gas_solubility, molvol,
                              molweight, sw_dens, vapor)
from conftest import assert_close, load_ref_csv


def test_marelac_anchors():
    ref = load_ref_csv("ref_marelac.csv")
    # R: columns are vapor, gas_solubility, gas_satconc, molvol(O2), sw_dens, molweight
    S = np.array([0.0, 35.0, 35.0])
    t = np.array([20.0, 20.0, 25.0])
    P = np.array([1.013253, 1.013253, 1.0])

    assert_close(vapor(S=S, t=t), ref[:, 0], label="vapor")
    assert_close(gas_solubility(S=S, t=t), ref[:, 1], label="gas_solubility")
    assert_close(gas_satconc(S=S, t=t, P=P), ref[:, 2], label="gas_satconc")
    # molvol: R solves with uniroot (loose tol); allow a little more slack
    assert_close(molvol(t=t, P=P), ref[:, 3], rtol=1e-5, label="molvol")
    assert_close(sw_dens(S=S, t=t, P=P), ref[:, 4], label="sw_dens")
    assert_close([molweight("O2")] * 3, ref[:, 5], label="molweight")


def test_scalar_shapes():
    """Scalar inputs should give scalar (0-d) results like R."""
    assert np.ndim(vapor(S=35, t=20)) == 0
    assert np.ndim(sw_dens(S=35, t=20)) == 0
    assert np.ndim(molvol(t=20, P=1.013253)) == 0
