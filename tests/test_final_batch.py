"""Final batch: calc_rate.int/bg/ft, auto_rate.int, adjust_rate (6 methods),
select_rate, oxy_crit, inspect(.ft), convert_rate.ft, test_lin -- all
compared value-by-value against R 4.5.3 reference CSVs."""
import numpy as np
import resprpy as p

from conftest import REF_DIR, assert_close


def _load(name):
    return np.genfromtxt(f"{REF_DIR}/{name}", delimiter=",", skip_header=1)


def _load_str(name):
    """Load an R write.csv reference file with csv.reader (handles R's
    double-quote wrapping). Returns rows as lists of unquoted tokens."""
    import csv
    with open(f"{REF_DIR}/{name}", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[1:]  # skip header row


def _load_str_vals(name):
    """R reference with CSV quoting handled; single-column files return a
    flat list of the column values."""
    rows = _load_str(name)
    if len(rows) and len(rows[0]) == 1:
        return [row[0] for row in rows]
    return rows


def _urchins():
    return _load("urchins.csv")


def _cmp(a, e, rtol=1e-6, label=""):
    a = np.asarray(a, dtype=float)
    e = np.asarray(e, dtype=float)
    both_nan = np.isnan(a) & np.isnan(e)
    a2 = np.where(both_nan, 0.0, a)
    e2 = np.where(both_nan, 0.0, e)
    assert_close(a2, e2, rtol=rtol, label=label)


def _summ_cols(summ, names):
    return np.column_stack([np.asarray(summ[k], dtype=float) for k in names])


def _bool_checks(checks):
    """Python checks matrix -> R 'TRUE'/'FALSE'/'NA' strings."""
    out = []
    for row in np.asarray(checks, dtype=object):
        out.append(["NA" if v is None else str(bool(v)).upper() for v in row])
    return out


# ---------------------------------------------------------------------------
# calc_rate.int
# ---------------------------------------------------------------------------
def test_calc_rate_int_row_against_R():
    u = _urchins()
    r = p.calc_rate_int(u, starts=50, wait=10, measure=80, by="row")
    ref = _load("ref_calc_rate_int_row.csv")
    cols = ["rep", "rank", "intercept_b0", "slope_b1", "rsq", "row", "endrow",
            "time", "endtime", "oxy", "endoxy", "rate.2pt", "rate"]
    _cmp(_summ_cols(r.summary, cols), ref, rtol=1e-6, label="calc_rate.int row")


def test_calc_rate_int_time_against_R():
    u = _urchins()
    r = p.calc_rate_int(u, starts=20, wait=2, measure=15, by="time")
    ref = _load("ref_calc_rate_int_time.csv")
    cols = ["rep", "rank", "intercept_b0", "slope_b1", "rsq", "row", "endrow",
            "time", "endtime", "oxy", "endoxy", "rate.2pt", "rate"]
    _cmp(_summ_cols(r.summary, cols), ref, rtol=1e-6, label="calc_rate.int time")


# ---------------------------------------------------------------------------
# calc_rate.bg
# ---------------------------------------------------------------------------
def test_calc_rate_bg_against_R():
    u = _urchins()
    r = p.calc_rate_bg(u[:, :3], time=1, oxygen=[2, 3])
    ref = _load("ref_calc_rate_bg.csv")
    cols = ["rep", "rank", "intercept_b0", "slope_b1", "rsq", "row", "endrow",
            "time", "endtime", "oxy", "endoxy", "rate.bg"]
    _cmp(_summ_cols(r["summary"], cols), ref[:, :12], rtol=1e-6,
         label="calc_rate.bg summary")
    _cmp([r["rate.bg.mean"]], [ref[0, 12]], rtol=1e-8, label="calc_rate.bg mean")


# ---------------------------------------------------------------------------
# calc_rate.ft
# ---------------------------------------------------------------------------
def test_calc_rate_ft_vec_against_R():
    delta = np.atleast_1d(_load("ref_ft_delta.csv")).ravel()
    r = p.calc_rate_ft(delta, flowrate=0.1)
    ref = _load("ref_calc_rate_ft_vec.csv")
    cols = ["rep", "rank", "intercept_b0", "slope_b1", "rsq", "row", "endrow",
            "time", "endtime", "oxy", "endoxy", "delta_mean", "flowrate",
            "rate"]
    _cmp(_summ_cols(r["summary"], cols), ref, rtol=1e-8, label="calc_rate.ft vec")


def test_calc_rate_ft_df_against_R():
    inp = _load("ref_ft_input.csv")
    # R: cbind(outo, ino) is a matrix -> is.numeric TRUE -> 'vec' branch, so
    # delta stays a 2-column matrix; rate = delta * flowrate (2 columns).
    r = p.calc_rate_ft(inp, flowrate=0.05)
    ref = _load("ref_calc_rate_ft_df.csv")
    _cmp(r["summary"]["delta_mean"], ref[:, 11:13], rtol=1e-8,
         label="ft df delta")
    _cmp(r["summary"]["rate"], ref[:, 14:16], rtol=1e-8, label="ft df rate")


def _ft_inp3():
    inp = _load("ref_ft_input.csv")
    return np.column_stack([np.linspace(0, 100, 100), inp])


def test_calc_rate_ft_insp_against_R():
    ift = p.inspect_ft(_ft_inp3(), time=1, out_oxy=2, in_oxy=3)
    r = p.calc_rate_ft(ift, flowrate=0.05, from_=0, to=100, by="time")
    ref = _load("ref_calc_rate_ft_insp.csv")
    cols = ["rep", "rank", "intercept_b0", "slope_b1", "rsq", "row", "endrow",
            "time", "endtime", "oxy", "endoxy", "delta_mean", "flowrate",
            "rate"]
    _cmp(_summ_cols(r["summary"], cols), ref, rtol=1e-6, label="calc_rate.ft insp")


def test_calc_rate_ft_width_against_R():
    ift = p.inspect_ft(_ft_inp3(), time=1, out_oxy=2, in_oxy=3)
    r = p.calc_rate_ft(ift, flowrate=0.05, by="row", width=20)
    ref = _load("ref_calc_rate_ft_width.csv")
    cols = ["rep", "rank", "intercept_b0", "slope_b1", "rsq", "row", "endrow",
            "time", "endtime", "oxy", "endoxy", "delta_mean", "flowrate",
            "rate"]
    _cmp(_summ_cols(r["summary"], cols), ref, rtol=1e-5, label="calc_rate.ft width")


# ---------------------------------------------------------------------------
# auto_rate.int
# ---------------------------------------------------------------------------
def test_auto_rate_int_against_R():
    u = _urchins()
    r = p.auto_rate_int(u, starts=32, wait=5, measure=20, by="row",
                        method="linear", width=8)
    ref = _load("ref_auto_rate_int.csv")
    cols = ["rep", "rank", "intercept_b0", "slope_b1", "rsq", "density",
            "row", "endrow", "time", "endtime", "oxy", "endoxy", "rate"]
    _cmp(_summ_cols(r["summary"], cols), ref, rtol=1e-5, label="auto_rate.int")


# ---------------------------------------------------------------------------
# adjust_rate (all 6 methods)
# ---------------------------------------------------------------------------
def _adjust_setup():
    dat2 = _load("ref_adjust_dat2.csv")
    cr = p.calc_rate(dat2, from_=[0, 100, 200], to=[80, 180, 280], by="time")
    return dat2, cr


def _adj_cols():
    return ["rep", "rank", "intercept_b0", "slope_b1", "rsq", "row", "endrow",
            "time", "endtime", "oxy", "endoxy", "rate.2pt", "rate",
            "adjustment", "rate.adjusted"]


def test_adjust_rate_value_against_R():
    dat2, cr = _adjust_setup()
    adj = p.adjust_rate(cr, by=-0.0002, method="value")
    ref = _load("ref_adjust_rate_value.csv")
    _cmp(_summ_cols(adj["summary"], _adj_cols()), ref, rtol=1e-6,
         label="adjust_rate value")


def test_adjust_rate_mean_against_R():
    dat2, cr = _adjust_setup()
    bgv = p.calc_rate_bg(dat2, time=1, oxygen=[2])
    adj = p.adjust_rate(cr, by=bgv, method="mean")
    ref = _load("ref_adjust_rate_mean.csv")
    _cmp(_summ_cols(adj["summary"], _adj_cols()), ref, rtol=1e-6,
         label="adjust_rate mean")


def test_adjust_rate_paired_against_R():
    dat2, cr = _adjust_setup()
    adj = p.adjust_rate(cr, by=np.full(3, -0.0001), method="paired")
    ref = _load("ref_adjust_rate_paired.csv")
    _cmp(_summ_cols(adj["summary"], _adj_cols()), ref, rtol=1e-6,
         label="adjust_rate paired")


def _bgdfs():
    t1 = np.arange(0, 61, 2.0)
    bg1 = np.column_stack([t1, 7 - 0.001 * t1])
    t2 = np.arange(0, 61, 2.0)
    bg2 = np.column_stack([t2 + 240, 6.7 - 0.002 * t2])
    return bg1, bg2


def test_adjust_rate_linear_against_R():
    dat2, cr = _adjust_setup()
    bg1, bg2 = _bgdfs()
    adj = p.adjust_rate(cr, by=bg1, by2=bg2, method="linear")
    ref = _load("ref_adjust_rate_linear.csv")
    _cmp(_summ_cols(adj["summary"], _adj_cols()), ref, rtol=1e-6,
         label="adjust_rate linear")


def test_adjust_rate_exponential_against_R():
    dat2, cr = _adjust_setup()
    bg1, bg2 = _bgdfs()
    adj = p.adjust_rate(cr, by=bg1, by2=bg2, method="exponential")
    ref = _load("ref_adjust_rate_exp.csv")
    _cmp(_summ_cols(adj["summary"], _adj_cols()), ref, rtol=1e-6,
         label="adjust_rate exponential")


def test_adjust_rate_concurrent_against_R():
    dat2, cr = _adjust_setup()
    adj = p.adjust_rate(cr, by=dat2, method="concurrent")
    ref = _load("ref_adjust_rate_concurrent.csv")
    _cmp(_summ_cols(adj["summary"], _adj_cols()), ref, rtol=1e-6,
         label="adjust_rate concurrent")


# ---------------------------------------------------------------------------
# select_rate
# ---------------------------------------------------------------------------
def _select_setup():
    dat2, cr = _adjust_setup()
    cv = p.convert_rate(cr, oxy_unit="mg/L", time_unit="sec",
                        output_unit="mg/h/kg", volume=0.1, mass=0.01)
    return cv


def _sel_cols():
    return ["rate.output", "rate.abs", "rate.m.spec"]


def test_select_rate_minimum_against_R():
    cv = _select_setup()
    sr = p.select_rate(cv, "minimum", n=2)
    ref = _load("ref_select_rate_min.csv")
    _cmp(_summ_cols(sr.summary, _sel_cols()), ref[:, [28, 24, 25]],
         rtol=1e-5, label="select_rate minimum")


def test_select_rate_maximum_against_R():
    cv = _select_setup()
    sr = p.select_rate(cv, "maximum")
    ref = _load("ref_select_rate_max.csv")
    _cmp(_summ_cols(sr.summary, _sel_cols()), ref[:, [28, 24, 25]],
         rtol=1e-5, label="select_rate maximum")


def test_select_rate_range_against_R():
    cv = _select_setup()
    sr = p.select_rate(cv, "rate", n=[-100, 100])
    ref = _load("ref_select_rate_range.csv")
    _cmp(_summ_cols(sr.summary, _sel_cols()), ref[:, [28, 24, 25]],
         rtol=1e-5, label="select_rate range")


# ---------------------------------------------------------------------------
# oxy_crit (bsr)
# ---------------------------------------------------------------------------
def test_convert_rate_accepts_adjust_rate_dict():
    """convert_rate must accept an adjust_rate result (dict), as R does."""
    dat2, cr = _adjust_setup()
    bgv = p.calc_rate_bg(dat2, plot=False)
    adj = p.adjust_rate(cr, by=bgv, method="mean")
    assert isinstance(adj, dict), "adjust_rate should return a dict"
    cv = p.convert_rate(adj, oxy_unit="mg/L", time_unit="sec",
                        output_unit="mg/h/kg", volume=0.1, mass=0.01)
    assert np.size(cv.rate_output) == np.size(adj["rate"])


def test_oxy_crit_against_R():
    u = _urchins()
    r = p.oxy_crit(u, method="bsr")
    ref_res = _load("ref_oxy_crit_results.csv")
    ref_crit = np.atleast_2d(_load("ref_oxy_crit_crit.csv"))[0]
    _cmp(r["results"], ref_res, rtol=1e-6, label="oxy_crit results")
    _cmp([r["crit"]["crit.intercept"], r["crit"]["crit.midpoint"]],
         ref_crit, rtol=1e-6, label="oxy_crit crit")


# ---------------------------------------------------------------------------
# oxy_crit (segmented, Muggeo 2003 via 'segmented' 2.2-1 port)
# ---------------------------------------------------------------------------
def test_oxy_crit_segmented_against_R():
    u = _urchins()
    r = p.oxy_crit(u, method="segmented", plot=False)
    ref_sum = np.atleast_2d(_load("ref_oxy_crit_seg_summary.csv"))[0]
    _cmp(r["summary"], ref_sum, rtol=1e-8, label="seg summary")
    ref_fit = _load("ref_oxy_crit_seg_fit.csv")
    _cmp(r["results"], ref_fit, rtol=1e-8, label="seg fit")
    assert abs(r["crit"] - ref_sum[5]) < 1e-12, "seg crit mismatch"


# ---------------------------------------------------------------------------
# inspect / inspect.ft
# ---------------------------------------------------------------------------
def test_inspect_checks_against_R():
    u = _urchins()
    r = p.inspect(u[:, :2], time=1, oxygen=2)
    ref = _load_str_vals("ref_inspect_checks.csv")
    got = _bool_checks(r["checks"])
    assert got == ref, f"inspect checks mismatch:\n py={got}\n R ={ref}"


def _parse_r_locs(s):
    """Extract integer positions from R's deparse-style locs string
    ('NA,integer(0),c(1, 2, 3)' -> [1, 2, 3])."""
    import re
    s2 = s.replace("integer(0)", "")
    return [int(n) for n in re.findall(r"\d+", s2)]


def test_inspect_locs_against_R():
    u = _urchins()
    r = p.inspect(u[:, :2], time=1, oxygen=2)
    ref = _load_str_vals("ref_inspect_locs.csv")
    got = []
    for col in r["locs"]:
        parts = []
        for v in col:
            if v is None:
                parts.append("NA")
            else:
                parts.extend(str(int(x)) for x in np.asarray(v).ravel())
        got.append(",".join(parts))
    py_nums = [_parse_r_locs(s) for s in got]
    r_nums = [_parse_r_locs(s) for s in ref]
    assert py_nums == r_nums, \
        f"inspect locs mismatch:\n py={py_nums}\n R ={r_nums}"


def test_inspect_bad_against_R():
    bad = np.array([[1, 8], [2, 7.9], [np.nan, 7.8], [4, np.inf],
                    [4, 7.6], [3, 7.5]], dtype=float)
    r = p.inspect(bad, time=1, oxygen=2)
    ref = _load_str_vals("ref_inspect_bad_checks.csv")
    got = _bool_checks(r["checks"])
    assert got == ref, f"inspect bad checks mismatch:\n py={got}\n R ={ref}"


def test_inspect_ft_against_R():
    r = p.inspect_ft(_ft_inp3(), time=1, out_oxy=2, in_oxy=3)
    ref_checks = _load_str_vals("ref_inspect_ft_checks.csv")
    got = _bool_checks(r["checks"])
    assert got == ref_checks, \
        f"inspect.ft checks mismatch:\n py={got}\n R ={ref_checks}"
    ref_df = _load("ref_inspect_ft_dataframe.csv")
    _cmp(r["dataframe"], ref_df, rtol=1e-8, label="inspect.ft dataframe")


# ---------------------------------------------------------------------------
# convert_rate.ft
# ---------------------------------------------------------------------------
def test_convert_rate_ft_vec_against_R():
    r = p.convert_rate_ft(np.array([0.01, 0.012, 0.008]), oxy_unit="mg/L",
                          flowrate_unit="mL/min", output_unit="mg/h")
    ref = _load("ref_convert_rate_ft_vec.csv")
    _cmp(r["summary"]["rate.abs"], ref[:, 23], rtol=1e-8, label="cft vec abs")
    _cmp(r["summary"]["rate.output"], ref[:, 26], rtol=1e-8, label="cft vec out")


def test_convert_rate_ft_obj_against_R():
    ift = p.inspect_ft(_ft_inp3(), time=1, out_oxy=2, in_oxy=3)
    cft = p.calc_rate_ft(ift, flowrate=0.05, from_=0, to=100, by="time")
    r = p.convert_rate_ft(cft, oxy_unit="mg/L", flowrate_unit="mL/min",
                          output_unit="mg/h/kg", mass=0.05)
    ref = np.atleast_2d(_load("ref_convert_rate_ft_obj.csv"))
    # obj summary has 29 cols (rep,rank,... + adjustment/rate.adjusted + 13):
    # rate.abs = col 25 (0-based 24), rate.m.spec = 26 (0-based 25),
    # rate.output = 29 (0-based 28)
    _cmp(r["summary"]["rate.abs"], ref[:, 24], rtol=1e-8, label="cft obj abs")
    _cmp(r["summary"]["rate.m.spec"], ref[:, 25], rtol=1e-8, label="cft obj m")
    _cmp(r["summary"]["rate.output"], ref[:, 28], rtol=1e-8, label="cft obj out")


def test_convert_rate_ft_area_against_R():
    r = p.convert_rate_ft(np.array([0.01, 0.012]), oxy_unit="mg/L",
                          flowrate_unit="mL/min", output_unit="mg/h/m2",
                          area=0.02)
    ref = _load("ref_convert_rate_ft_area.csv")
    _cmp(r["summary"]["rate.a.spec"], ref[:, 25], rtol=1e-8, label="cft area a")
    _cmp(r["summary"]["rate.output"], ref[:, 26], rtol=1e-8, label="cft area out")


# ---------------------------------------------------------------------------
# import_file (NeoFox csv; Witrox/AutoResp samples are mis-parsed by R's fread
# itself -- whitespace split -- so only the clean CSV is compared)
# ---------------------------------------------------------------------------
def test_import_file_neofox_against_R():
    r = p.import_file("reference/respR-master/vignettes/ACACTB11.csv.utf8")
    ref = _load("ref_import_neofox.csv")
    assert r["data"].shape == ref.shape, \
        f"import_file shape {r['data'].shape} != R {ref.shape}"
    py = np.asarray(r["data"][:, 1:], dtype=float)
    _cmp(py, ref[:, 1:], rtol=1e-10, label="import_file neofox")


# ---------------------------------------------------------------------------
# test_lin (structure; RNG differs R vs numpy)
# ---------------------------------------------------------------------------
def test_test_lin_structure():
    r = p.test_lin(reps=5, seed=99)
    names = ["real", "measured", "length_line", "length_incorrect",
             "length_detected"]
    assert list(r["df"].keys()) == names
    for k in names:
        assert len(r["df"][k]) == 5
    assert r["results"]["coefficients"].size == 2
    ref = _load("ref_test_lin_df.csv")
    assert ref.shape[0] == 5 and ref.shape[1] == 5
