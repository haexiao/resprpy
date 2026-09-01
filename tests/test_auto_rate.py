"""auto_rate against R reference outputs (urchins dataset)."""
import numpy as np

from resprpy import auto_rate
from conftest import REF_DIR, assert_close, load_ref_csv


def _urchins_t_oxy():
    data = np.genfromtxt(f"{REF_DIR}/urchins.csv", delimiter=",", skip_header=1)
    return data[:, [0, 14]]


def test_auto_rate_linear_against_R():
    df = _urchins_t_oxy()
    ar = auto_rate(df, plot=False)
    ref = np.genfromtxt(f"{REF_DIR}/ref_auto_rate_urchins.csv",
                        delimiter=",", skip_header=1)
    s = ar.summary
    n = ref.shape[0]
    assert s["rank"].size == n, (s["rank"].size, n)
    # ref columns: rank, intercept_b0, slope_b1, rsq, density, row, endrow,
    # time, endtime, oxy, endoxy, rate
    assert_close(s["intercept_b0"], ref[:, 1], rtol=1e-6, label="intercept_b0")
    assert_close(s["slope_b1"], ref[:, 2], rtol=1e-6, label="slope_b1")
    assert_close(s["rsq"], ref[:, 3], rtol=1e-12, label="rsq")
    assert_close(s["density"], ref[:, 4], rtol=1e-3, label="density")
    assert_close(s["row"], ref[:, 5], rtol=1e-12, label="row")
    assert_close(s["endrow"], ref[:, 6], rtol=1e-12, label="endrow")
    assert_close(s["time"], ref[:, 7], rtol=1e-9, label="time")
    assert_close(s["endtime"], ref[:, 8], rtol=1e-9, label="endtime")
    assert_close(s["oxy"], ref[:, 9], rtol=1e-6, label="oxy")
    assert_close(s["endoxy"], ref[:, 10], rtol=1e-6, label="endoxy")
    assert_close(s["rate"], ref[:, 11], rtol=1e-6, label="rate")
    assert np.all(s["rank"] == ref[:, 0])


def test_auto_rate_density_matches_R():
    df = _urchins_t_oxy()
    ar = auto_rate(df, plot=False)
    grid, y, bw = ar.density
    ref = np.genfromtxt(f"{REF_DIR}/ref_density_urchins.csv",
                        delimiter=",", skip_header=1)
    assert_close(grid, ref[:, 0], rtol=1e-12, label="density grid")
    assert_close(y, ref[:, 1], rtol=1e-9, label="density y")
    assert np.isclose(bw, 0.0006027593, rtol=1e-9)


def test_auto_rate_simple_methods_against_R():
    df = _urchins_t_oxy()
    for m in ("max", "min", "highest", "lowest", "interval", "rolling"):
        ar = auto_rate(df, method=m, plot=False)
        ref = np.genfromtxt(f"{REF_DIR}/ref_auto_rate_{m}.csv",
                            delimiter=",", skip_header=1)
        s = ar.summary
        assert s["rank"].size == ref.shape[0], (m, s["rank"].size, ref.shape[0])
        assert_close(s["slope_b1"], ref[:, 1], rtol=1e-8, label=f"{m} slope")
        # R's roll_lm (incremental C) and our lstsq differ by ~1e-16, which can
        # reorder exactly-tied slopes; align by (slope, row) before comparing.
        o_py = np.lexsort((s["row"], s["slope_b1"]))
        o_r = np.lexsort((ref[:, 2], ref[:, 1]))
        assert_close(s["row"][o_py], ref[o_r][:, 2], rtol=1e-12, label=f"{m} row")
        assert_close(s["endrow"][o_py], ref[o_r][:, 3], rtol=1e-12, label=f"{m} endrow")
        assert_close(s["time"][o_py], ref[o_r][:, 4], rtol=1e-9, label=f"{m} time")
        assert_close(s["endtime"][o_py], ref[o_r][:, 5], rtol=1e-9, label=f"{m} endtime")
