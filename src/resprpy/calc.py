"""calc_rate: calculate metabolic rates from oxygen time-series data.

Ported from respR 2.3.4. Fits linear regressions (oxygen ~ time) over
user-selected data regions and returns the same summary columns as R:
rank, intercept_b0, slope_b1, rsq, row, endrow, time, endtime, oxy,
endoxy, rate.2pt, rate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# helpers (ported respR internals)
# ---------------------------------------------------------------------------
_TIME_RE = re.compile(r"(?i)^\b(time|t)\b$")
_OX_RE = re.compile(r"(?i)^(o|o2|ox|oxy|oxygen)$")
_ROW_RE = re.compile(r"(?i)^\b(row|r)\b$")


def _by_val(by, req=True, default=None, which=("t", "o", "r"), msg=""):
    if req and by is None:
        raise ValueError(f"{msg}: 'by' input is required.")
    if not req and by is None and default is not None:
        by = default
    if "t" in which and _TIME_RE.match(str(by)):
        return "time"
    if "o" in which and _OX_RE.match(str(by)):
        return "oxygen"
    if "r" in which and _ROW_RE.match(str(by)):
        return "row"
    raise ValueError(f"{msg}: 'by' input not valid or not recognised.")


def _truncate_data(x, from_, to, by):
    """Subset the data frame between from/to (by 'time', 'row' or 'oxygen')."""
    dt_col1 = np.asarray(x[:, 0], dtype=float)
    dt_col2 = np.asarray(x[:, 1], dtype=float)
    if by == "time":
        rng = (np.nanmin(dt_col1), np.nanmax(dt_col1))
        if from_ < rng[0]:
            from_ = rng[0]
        if to > rng[1]:
            to = rng[1]
        i_from = int(np.argmin(np.abs(dt_col1 - from_)))
        i_to = int(np.argmin(np.abs(dt_col1 - to)))
        mask = (dt_col1 >= dt_col1[i_from]) & (dt_col1 <= dt_col1[i_to])
        return x[mask]
    if by == "row":
        if to > x.shape[0]:
            to = x.shape[0]
        return x[int(from_) - 1:int(to)]  # R: dt[from:to] rows are 1-indexed inclusive
    # oxygen
    o_range = (np.nanmin(dt_col2), np.nanmax(dt_col2))
    if from_ > o_range[1]:
        from_ = o_range[1]
    elif from_ < o_range[0]:
        from_ = o_range[0]
    if to > o_range[1]:
        to = o_range[1]
    elif to < o_range[0]:
        to = o_range[0]
    lower, upper = sorted((from_, to))
    idx = np.where((dt_col2 >= lower) & (dt_col2 <= upper))[0]
    return x[idx[0]:idx[-1] + 1]


def _linear_fit(dt):
    """lm(oxygen ~ time): intercept b0, slope b1, rsq = signif(r.squared, 3)."""
    t = np.asarray(dt[:, 0], dtype=float)
    o = np.asarray(dt[:, 1], dtype=float)
    X = np.column_stack([np.ones_like(t), t])
    coef, *_ = np.linalg.lstsq(X, o, rcond=None)
    b0, b1 = coef
    pred = X @ coef
    ss_res = float(np.sum((o - pred) ** 2))
    ss_tot = float(np.sum((o - np.mean(o)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # R: signif(r.squared, 3) -- 3 significant digits
    rsq = float(f"{r2:.3g}")
    return b0, b1, rsq


def _extract_indices(x, subsets, n):
    """row/endrow/time/endtime/oxy/endoxy for subset n (mirrors respR)."""
    sub = subsets[n]
    t_full = np.asarray(x[:, 0], dtype=float)
    o_full = np.asarray(x[:, 1], dtype=float)
    first_t, last_t = sub[0, 0], sub[-1, 0]
    first_o, last_o = sub[0, 1], sub[-1, 1]
    # R: row = match(first_t, x[[1]]) -> FIRST match; endrow = tail(which(last_t == x[[1]]), 1)
    row = int(np.where(t_full == first_t)[0][0]) + 1
    endrow = int(np.where(t_full == last_t)[0][-1]) + 1
    return {"row": row, "endrow": endrow, "time": first_t, "endtime": last_t,
            "oxy": first_o, "endoxy": last_o}


# ---------------------------------------------------------------------------
# calc_rate
# ---------------------------------------------------------------------------
# summary column order, identical to respR's calc_rate summary data.table
SUMMARY_COLUMNS = ("rep", "rank", "intercept_b0", "slope_b1", "rsq", "row",
                   "endrow", "time", "endtime", "oxy", "endoxy",
                   "rate.2pt", "rate")


@dataclass
class CalcRate:
    """Result of calc_rate(). Mirrors the respR 'calc_rate' object."""
    call: object = None
    inputs: dict = field(default_factory=dict)
    dataframe: np.ndarray = None
    subsets: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    rate_2pt: np.ndarray = None
    rate: np.ndarray = None

    @property
    def summary_table(self):
        """Full summary as a 2-D array with respR's exact column order
        (13 columns: rep, rank, intercept_b0, slope_b1, rsq, row, endrow,
        time, endtime, oxy, endoxy, rate.2pt, rate)."""
        return np.column_stack([self.summary[k] for k in SUMMARY_COLUMNS])

    @property
    def summary_columns(self):
        """Column names of summary_table (respR order)."""
        return list(SUMMARY_COLUMNS)

    def __repr__(self):
        n = np.size(self.rate)
        return f"<resprpy.CalcRate n={n}>"


def calc_rate(x, from_=None, to=None, by="time", plot=True, **kwargs):
    """Calculate metabolic rates from oxygen vs time data.

    Parameters
    ----------
    x : array-like, shape (n, 2)
        Two columns: time and oxygen (like respR, only the first two
        columns are used; extra columns are ignored).
    from_, to : float or array-like
        Start/end of the region(s) to fit, in 'by' units (time by default).
        Multiple pairs give multiple rates.
    by : str
        'time', 'row' or 'oxygen'.
    plot : bool
        Accepted for respR API compatibility (not implemented yet).

    Returns
    -------
    CalcRate object with .summary (dict of arrays), .rate, .rate_2pt.
    """
    inputs = dict(x=x, from_=from_, to=to, by=by, plot=plot)
    by = _by_val(by, msg="calc_rate:")
    df = np.asarray(x, dtype=float)
    if df.ndim == 1:
        df = df.reshape(-1, 1)
    if df.shape[1] > 2:
        df = df[:, :2]
    if df.shape[0] == 1:
        raise ValueError("calc_rate: Input data contains only 1 row. Please check inputs.")

    t_col = df[:, 0]
    o_col = df[:, 1]

    if from_ is None:
        from_ = np.nanmin(t_col) if by == "time" else (1 if by == "row" else o_col[0])
    if to is None:
        to = np.nanmax(t_col) if by == "time" else (df.shape[0] if by == "row" else o_col[-1])

    from_arr = np.atleast_1d(np.asarray(from_, dtype=float))
    to_arr = np.atleast_1d(np.asarray(to, dtype=float))
    if from_arr.size != to_arr.size:
        raise ValueError("calc_rate: 'from' and 'to' have unequal lengths.")
    if np.any(from_arr == to_arr):
        raise ValueError("calc_rate: some 'from' values are equal to the paired values in 'to'.")

    if by == "time":
        if np.any(from_arr > to_arr):
            raise ValueError("calc_rate: some 'from' time values are later than the paired values in 'to'.")
        t_range = (np.nanmin(t_col), np.nanmax(t_col))
        if np.any(from_arr > t_range[1]):
            raise ValueError("calc_rate: some 'from' time values are higher than the values present in 'x'.")
        if np.any(to_arr < t_range[0]):
            raise ValueError("calc_rate: some 'to' time values are lower than the values present in 'x'.")
    elif by == "row":
        if np.any(from_arr > to_arr):
            raise ValueError("calc_rate: some 'from' row numbers are higher than the paired values in 'to'.")
        if np.any(from_arr > df.shape[0]):
            raise ValueError("calc_rate: some 'from' row numbers are beyond the number of rows present in 'x'.")
    elif by == "oxygen":
        o_range = (np.nanmin(o_col), np.nanmax(o_col))
        both_below = np.array([(a < o_range[0] and b < o_range[0]) for a, b in zip(from_arr, to_arr)])
        both_above = np.array([(a > o_range[1] and b > o_range[1]) for a, b in zip(from_arr, to_arr)])
        if np.any(both_below):
            raise ValueError("calc_rate: some paired 'from' and 'to' values are both below the range of oxygen data in 'x'.")
        if np.any(both_above):
            raise ValueError("calc_rate: some paired 'from' and 'to' values are both above the range of oxygen data in 'x'.")

    n_seg = from_arr.size
    subsets = [_truncate_data(df, from_arr[i], to_arr[i], by) for i in range(n_seg)]
    coefs = [_linear_fit(s) for s in subsets]
    indices = [_extract_indices(df, subsets, i) for i in range(n_seg)]

    rank = np.arange(1, n_seg + 1)
    summary = {
        "rep": np.full(n_seg, np.nan),
        "rank": rank,
        "intercept_b0": np.array([c[0] for c in coefs]),
        "slope_b1": np.array([c[1] for c in coefs]),
        "rsq": np.array([c[2] for c in coefs]),
        "row": np.array([ix["row"] for ix in indices]),
        "endrow": np.array([ix["endrow"] for ix in indices]),
        "time": np.array([ix["time"] for ix in indices]),
        "endtime": np.array([ix["endtime"] for ix in indices]),
        "oxy": np.array([ix["oxy"] for ix in indices]),
        "endoxy": np.array([ix["endoxy"] for ix in indices]),
        "rate.2pt": np.array([(ix["endoxy"] - ix["oxy"]) / (ix["endtime"] - ix["time"])
                              for ix in indices]),
        "rate": np.array([c[1] for c in coefs]),
    }
    rate = np.array([c[1] for c in coefs])
    return CalcRate(call=None, inputs=inputs, dataframe=df, subsets=subsets,
                    summary=summary, rate_2pt=summary["rate.2pt"], rate=rate)
