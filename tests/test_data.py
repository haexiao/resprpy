"""Tests for data utilities: subsample, subset_data, select, sim_data."""
import numpy as np

from resprpy import select, sim_data, subsample, subset_data
from conftest import REF_DIR, assert_close, load_ref_csv


def _urchins():
    return np.genfromtxt(f"{REF_DIR}/urchins.csv", delimiter=",", skip_header=1)


def test_subsample_against_R():
    urch = _urchins()
    ref_n = load_ref_csv("ref_subsample_n100.csv").ravel()
    ref_l = load_ref_csv("ref_subsample_len50.csv").ravel()
    s1 = subsample(urch, n=100, plot=False)
    s2 = subsample(urch, length_out=50, plot=False)
    assert s1.shape[0] == ref_n.size
    assert s2.shape[0] == ref_l.size
    assert_close(s1[:, 0], ref_n, label="subsample n=100 (time col)")
    assert_close(s2[:, 0], ref_l, label="subsample length.out=50 (time col)")


def test_subsample_random_start():
    urch = _urchins()
    out = subsample(urch, n=50, random_start=True, plot=False)
    assert 1 <= out.shape[0] <= urch.shape[0] // 50 + 1
    assert out.shape[1] == urch.shape[1]


def test_subset_data_against_R():
    urch = _urchins()
    ref_t = np.genfromtxt(f"{REF_DIR}/ref_subset_time.csv", delimiter=",", skip_header=1)
    ref_r = np.genfromtxt(f"{REF_DIR}/ref_subset_row.csv", delimiter=",", skip_header=1)
    d1 = subset_data(urch, from_=10, to=30, by="time")
    d2 = subset_data(urch, from_=100, to=300, by="row")
    assert d1.shape == ref_t.shape, (d1.shape, ref_t.shape)
    assert d2.shape == ref_r.shape, (d2.shape, ref_r.shape)
    assert_close(d1, ref_t, label="subset_data by time")
    assert_close(d2, ref_r, label="subset_data by row")


def test_select():
    arr = np.arange(12).reshape(4, 3)
    out = select(arr, 1, 3)
    assert np.array_equal(out, arr[:, [0, 2]])
    # named columns via dict
    d = {"time": np.array([1, 2, 3]), "oxygen": np.array([8.0, 7.9, 7.8])}
    out2 = select(d, "oxygen")
    assert np.array_equal(out2[:, 0], d["oxygen"])


def test_sim_data_structure():
    np.random.seed(1)
    s = sim_data(len=300, type="default", preview=False)
    df = s["df"]
    assert df.shape[1] == 2
    assert df.shape[0] == 300
    assert np.array_equal(df[:, 0], np.arange(300))  # x = 0..299
    assert np.size(s["seg_index"]) > 0
    assert np.max(s["seg_index"]) <= 300
    for typ in ("corrupted", "segmented"):
        s2 = sim_data(len=300, type=typ, preview=False)
        assert s2["df"].shape[0] == 300
