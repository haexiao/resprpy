"""Smoke tests for the matplotlib plot functions (respR plot method ports).

Plots are visual approximations of the R graphics -- these tests only verify
the pipeline runs and produces a figure file without error.
"""
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import resprpy as p


@pytest.fixture(scope="module")
def demo_data():
    t = np.arange(0, 60, 0.5)
    o = 8 - 0.001 * t + 0.005 * np.sin(t)
    return np.column_stack([t, o])


@pytest.fixture(scope="module")
def objects(demo_data):
    return {
        "calc_rate": p.calc_rate(demo_data, method="linear", by="row",
                                 width=10),
        "auto_rate": p.auto_rate(demo_data, method="linear", by="row",
                                 width=10),
        "inspect": p.inspect(demo_data, time=1, oxygen=2),
        "oxy_bsr": p.oxy_crit(demo_data, method="bsr", plot=False),
        "oxy_seg": p.oxy_crit(demo_data, method="segmented", plot=False),
    }


def _save(fig, tmp, name):
    path = os.path.join(tmp, name)
    fig.savefig(path)
    assert os.path.getsize(path) > 1000, f"{name} too small"
    return path


def test_plot_calc_rate(objects, tmp_path):
    fig = p.plot_calc_rate(objects["calc_rate"], save=None)
    _save(fig, tmp_path, "calc_rate.png")
    fig2 = p.plot_calc_rate(objects["calc_rate"], panel=2, save=None)
    _save(fig2, tmp_path, "calc_rate_p2.png")


def test_plot_auto_rate(objects, tmp_path):
    fig = p.plot_auto_rate(objects["auto_rate"], save=None)
    _save(fig, tmp_path, "auto_rate.png")
    fig2 = p.plot_auto_rate(objects["auto_rate"], panel=1, save=None)
    _save(fig2, tmp_path, "auto_rate_p1.png")


def test_plot_inspect(objects, tmp_path):
    figs = p.plot_inspect(objects["inspect"], save=None)
    assert isinstance(figs, list) and len(figs) == 1
    _save(figs[0], tmp_path, "inspect.png")


def test_plot_oxy_crit(objects, tmp_path):
    fig = p.plot_oxy_crit(objects["oxy_bsr"], save=None)
    _save(fig, tmp_path, "oxy_bsr.png")
    fig2 = p.plot_oxy_crit(objects["oxy_seg"], save=None)
    _save(fig2, tmp_path, "oxy_seg.png")
    fig3 = p.plot_oxy_crit(objects["oxy_bsr"], panel=2, save=None)
    _save(fig3, tmp_path, "oxy_bsr_p2.png")
