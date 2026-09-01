"""Compare ported conversion functions against R reference outputs."""
import numpy as np

from resprpy import convert_DO, convert_MR, convert_rate, convert_val
from conftest import REF_DIR, assert_close, load_ref_csv


def _urchin_oxygen():
    # column 15 = oxygen in urchins.rd
    data = np.genfromtxt(f"{REF_DIR}/urchins.csv", delimiter=",", skip_header=1)
    return data[:, 14]


def test_convert_DO_against_R():
    oxy = _urchin_oxygen()
    ref = load_ref_csv("ref_convert_DO.csv")
    S, t, P = 35.0, 20.0, 1.013253

    cols = {
        "umol/L": convert_DO(oxy, "mg/L", "umol/L", S=S, t=t, P=P),
        "kPa": convert_DO(oxy, "mg/L", "kPa", S=S, t=t, P=P),
        "%Air": convert_DO(oxy, "mg/L", "%Air", S=S, t=t, P=P),
        "mL/L": convert_DO(oxy, "mg/L", "mL/L", S=S, t=t, P=P),
        "Torr": convert_DO(oxy, "mg/L", "Torr", S=S, t=t, P=P),
        "mg/kg": convert_DO(oxy, "mg/L", "mg/kg", S=S, t=t, P=P),
        "umol/kg": convert_DO(oxy, "mg/L", "umol/kg", S=S, t=t, P=P),
    }
    for i, (name, val) in enumerate(cols.items()):
        assert_close(val, ref[:, i], label=f"convert_DO mg/L->{name}")


def test_convert_DO_scalar():
    v = convert_DO(8.0, "mg/L", "umol/L", S=35, t=20, P=1.013253)
    assert np.ndim(v) == 0
    assert np.isclose(v, 250.0093753486, rtol=1e-9)


def test_convert_val_against_R():
    ref = load_ref_csv("ref_convert_val.csv")
    assert_close(convert_val([1, 2.5, 0.05], "L", "mL"), ref[:, 0], label="L->mL")
    assert_close(convert_val([0, 25, 100], "C", "K"), ref[:, 1], label="C->K")
    assert_close(convert_val([32, 98.6, 212], "F", "C"), ref[:, 2], label="F->C")
    assert_close(convert_val([100, 2500, 5], "g", "kg"), ref[:, 3], label="g->kg")
    assert_close(convert_val([1, 2.5, 0.01], "m2", "cm2"), ref[:, 4], label="m2->cm2")
    assert_close(convert_val([1, 1.013253, 0.5], "bar", "kPa"), ref[:, 5], label="bar->kPa")


def test_convert_rate_against_R():
    rates = np.array([-0.001, -0.002, -0.0005])
    ref = load_ref_csv("ref_convert_rate.csv")
    S, t, P = 35.0, 20.0, 1.013253

    r1 = convert_rate(rates, oxy_unit="mg/L", time_unit="sec", output_unit="mg/h",
                      volume=0.6, S=S, t=t, P=P)
    r2 = convert_rate(rates, oxy_unit="mg/L", time_unit="sec", output_unit="mg/h/kg",
                      volume=0.6, mass=0.4, S=S, t=t, P=P)
    r3 = convert_rate(rates, oxy_unit="mg/L", time_unit="min", output_unit="umol/min/g",
                      volume=0.6, mass=0.4, S=S, t=t, P=P)
    r4 = convert_rate(rates, oxy_unit="mg/L", time_unit="hour", output_unit="mL/h/m2",
                      volume=0.6, area=0.01, S=S, t=t, P=P)

    assert_close(r1.rate_output, ref[:, 0], label="mg/h")
    assert_close(r2.rate_output, ref[:, 1], label="mg/h/kg")
    assert_close(r3.rate_output, ref[:, 2], label="umol/min/g")
    assert_close(r4.rate_output, ref[:, 3], label="mL/h/m2")

    assert r1.output_unit == "mgO2/hr"
    assert r2.output_unit == "mgO2/hr/kg"
    assert r3.output_unit == "umolO2/min/g"
    assert r4.output_unit == "mLO2/hr/m2"


def test_convert_rate_summary_fields():
    rates = np.array([-0.001, -0.002])
    r = convert_rate(rates, oxy_unit="mg/L", time_unit="sec", output_unit="mg/h/kg",
                     volume=0.6, mass=0.4, S=35.0, t=20.0, P=1.013253)
    s = r.summary
    assert np.allclose(s["rate.abs"], [-2.16, -4.32], rtol=1e-9)
    assert np.allclose(s["rate.m.spec"], [-5.4, -10.8], rtol=1e-9)
    assert np.allclose(s["volume"], 0.6)
    assert np.allclose(s["mass"], 0.4)
    assert np.allclose(s["t"], 20.0)
    assert np.all(s["S"] == 35.0)


def test_convert_MR_against_R():
    ref = load_ref_csv("ref_convert_MR.csv").reshape(1, -1)
    S, t = 35.0, 20.0
    assert_close(convert_MR(0.001, from_="mg/sec", to="umol/min", S=S, t=t),
                 ref[:, 0], label="abs")
    assert_close(convert_MR(0.001, from_="mg/sec/g", to="umol/min/kg", S=S, t=t),
                 ref[:, 1], label="mass.spec")
    assert_close(convert_MR(0.001, from_="mg/sec/m2", to="umol/min/cm2", S=S, t=t),
                 ref[:, 2], label="area.spec")


def test_convert_MR_from_convert_rate_object():
    rates = np.array([-0.001, -0.002])
    r = convert_rate(rates, oxy_unit="mg/L", time_unit="sec", output_unit="mg/h/kg",
                     volume=0.6, mass=0.4, S=35.0, t=20.0, P=1.013253)
    out = convert_MR(r, to="umol/min/kg", S=35.0, t=20.0)
    # object path must agree with the plain-numeric path on the same values
    direct = convert_MR(np.array([-5.4, -10.8]), from_="mg/hr/kg", to="umol/min/kg",
                        S=35.0, t=20.0)
    assert_close(out.rate_output, direct, label="convert_MR(object)==numeric")


def test_unit_args_runs():
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        from resprpy import unit_args
        unit_args()
    text = buf.getvalue()
    assert "mg/L" in text and "umol/min/g" in text
