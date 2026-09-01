"""inspect / inspect.ft: QC diagnostics for respirometry data frames.

Ported from respR 2.3.4. Checks: numeric, Inf/-Inf, NA/NaN, sequential,
duplicated, evenly-spaced. Returns the same checks/locs matrices as R.
"""
from __future__ import annotations

import numpy as np

_RNAMES = ("numeric", "Inf/-Inf", "NA/NaN", "sequential", "duplicated",
           "evenly-spaced  ")


# ---------------------------------------------------------------------------
# individual checks (mirror check_*.R)
# ---------------------------------------------------------------------------
def _check_num(x):
    """!is.numeric(x): check = any non-numeric; highlight = all rows flagged."""
    arr = np.asarray(x)
    test = arr.dtype.kind not in "fiub"
    check = bool(test)
    highlight = np.full(arr.shape[0], check)
    return check, highlight


def _check_inf(x):
    arr = np.asarray(x, dtype=float)
    test = np.isinf(arr)
    return bool(np.any(test)), np.where(test)[0]


def _check_na(x):
    arr = np.asarray(x, dtype=float)
    test = np.isnan(arr)
    return bool(np.any(test)), np.where(test)[0]


def _check_seq(x):
    arr = np.asarray(x, dtype=float)
    test = np.diff(arr) < 0
    test = np.where(np.isnan(test), False, test)
    return bool(np.any(test)), np.where(test)[0]


def _check_dup(x):
    arr = np.asarray(x, dtype=float)
    uniq, counts = np.unique(arr, return_counts=True)
    dup_vals = uniq[counts > 1]
    test = np.isin(arr, dup_vals)
    return bool(np.any(test)), np.where(test)[0]


def _check_evn(x):
    arr = np.asarray(x, dtype=float)
    spacing = np.diff(arr)
    if spacing.size == 0:
        return False, np.array([], dtype=int)
    vals, counts = np.unique(spacing, return_counts=True)
    mod = vals[np.argmax(counts)]
    test = spacing != mod
    test = np.where(np.isnan(test), True, test)
    check = len(np.unique(spacing)) > 1
    return bool(check), np.where(test)[0]


# ---------------------------------------------------------------------------
# check_timeseries
# ---------------------------------------------------------------------------
def _check_timeseries(x_list, type_="time"):
    """x_list: list of 1-D column arrays. Returns (checks, locs)."""
    ncol = len(x_list)
    checks = np.full((6, ncol), False, dtype=object)
    locs = np.full((6, ncol), None, dtype=object)

    for c in range(ncol):
        arr = x_list[c]
        num_check, num_hl = _check_num(arr)
        checks[0, c] = num_check
        locs[0, c] = None
        if type_ == "time":
            if not num_check:
                inf_check, inf_hl = _check_inf(arr)
                checks[1, c] = inf_check
                locs[1, c] = inf_hl + 1  # R 1-based
                nan_check, nan_hl = _check_na(arr)
                checks[2, c] = nan_check
                locs[2, c] = nan_hl + 1
                seq_check, seq_hl = _check_seq(arr)
                checks[3, c] = seq_check
                locs[3, c] = seq_hl + 1
                dup_check, dup_hl = _check_dup(arr)
                checks[4, c] = dup_check
                locs[4, c] = dup_hl + 1
                evn_check, evn_hl = _check_evn(arr)
                checks[5, c] = evn_check
                locs[5, c] = evn_hl + 1
            else:
                locs[1:6, c] = [np.array([], dtype=int)] * 5
        else:  # oxygen: only numeric/Inf/NA; seq/dup/evn are NA (not checked)
            if not num_check:
                inf_check, inf_hl = _check_inf(arr)
                checks[1, c] = inf_check
                locs[1, c] = inf_hl + 1
                nan_check, nan_hl = _check_na(arr)
                checks[2, c] = nan_check
                locs[2, c] = nan_hl + 1
            else:
                locs[1:3, c] = [np.array([], dtype=int)] * 2
            checks[3, c] = None
            checks[4, c] = None
            checks[5, c] = None
    return checks, locs


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------
def _column_id(strings, colnames, msg):
    """Map column-name strings to 1-based indices (R column.id)."""
    names = list(colnames)
    if any(s not in names for s in strings):
        raise ValueError(f"{msg}: One or more column names not found in data "
                         "frame.")
    return [names.index(s) + 1 for s in strings]


def inspect(x, time=None, oxygen=None, width=0.1, plot=True,
            add_data=None, **kwargs):
    df = np.asarray(x, dtype=float)
    if df.ndim == 1:
        df = df.reshape(-1, 1)
    ncol = df.shape[1]
    colnames = [f"V{i + 1}" for i in range(ncol)]
    if hasattr(x, "columns") and x.columns is not None:
        colnames = list(x.columns)

    if time is None:
        time = 1
    if isinstance(time, str):
        time = _column_id([time], colnames, "inspect: ")[0]
    if oxygen is None:
        oxygen = 2
    if isinstance(oxygen, str):
        oxygen = _column_id([oxygen], colnames, "inspect: ")[0]
    if isinstance(oxygen, (list, tuple, np.ndarray)):
        oxygen = [int(o) for o in oxygen]
    if width is None:
        width = 0.1

    time_cols = [time] if np.isscalar(time) else list(time)
    oxy_cols = [oxygen] if np.isscalar(oxygen) else list(oxygen)

    xval = [df[:, t - 1] for t in time_cols]
    yval = [df[:, o - 1] for o in oxy_cols]

    x_results = _check_timeseries(xval, "time")
    y_results = _check_timeseries(yval, "oxygen")

    checks = np.hstack([x_results[0], y_results[0]])
    locs_raw = np.hstack([x_results[1], y_results[1]])
    all_cols = [colnames[t - 1] for t in time_cols] + \
               [colnames[o - 1] for o in oxy_cols]
    locs = [locs_raw[:, i] for i in range(locs_raw.shape[1])]

    dataframe = np.hstack([df[:, t - 1:t] for t in time_cols] +
                          [df[:, o - 1:o] for o in oxy_cols])
    if isinstance(dataframe, np.ndarray) and dataframe.ndim == 1:
        dataframe = dataframe.reshape(-1, 1)

    out = {"call": None, "dataframe": dataframe, "add.data": add_data,
           "inputs": {"x": x, "time": time, "oxygen": oxygen, "width": width,
                      "plot": plot, "add.data": add_data},
           "checks": checks, "locs_raw": locs_raw, "locs": locs,
           "colnames": all_cols}
    return out


# ---------------------------------------------------------------------------
# inspect.ft
# ---------------------------------------------------------------------------
def inspect_ft(x, time=None, out_oxy=None, in_oxy=None, in_oxy_value=None,
               delta_oxy=None, plot=True, add_data=None, **kwargs):
    df = np.asarray(x, dtype=float)
    if df.ndim == 1:
        df = df.reshape(-1, 1)
    ncol = df.shape[1]
    colnames = [f"V{i + 1}" for i in range(ncol)]
    if hasattr(x, "columns") and x.columns is not None:
        colnames = list(x.columns)

    if time is None:
        time = 1
    if isinstance(time, str):
        time = _column_id([time], colnames, "inspect.ft: ")[0]
    if isinstance(out_oxy, str):
        out_oxy = _column_id([out_oxy], colnames, "inspect.ft: ")[0]
    if isinstance(in_oxy, str):
        in_oxy = _column_id([in_oxy], colnames, "inspect.ft: ")[0]
    if isinstance(delta_oxy, str):
        delta_oxy = _column_id([delta_oxy], colnames, "inspect.ft: ")[0]

    if out_oxy is None and delta_oxy is None:
        delta_oxy = [i for i in range(1, ncol + 1) if i != time]

    def to_list(v):
        return [v] if np.isscalar(v) else list(v)

    time_cols = to_list(time)
    out_cols = to_list(out_oxy) if out_oxy is not None else None
    in_cols = to_list(in_oxy) if in_oxy is not None else None
    del_cols = to_list(delta_oxy) if delta_oxy is not None else None

    time_all = [df[:, t - 1] for t in time_cols]
    out_all = [df[:, o - 1] for o in out_cols] if out_cols else None
    in_all = [df[:, i - 1] for i in in_cols] if in_cols else None
    if in_oxy_value is not None:
        in_all = [np.full(df.shape[0], in_oxy_value)]
    del_all = [df[:, d - 1] for d in del_cols] if del_cols else None

    time_results = _check_timeseries(time_all, "time")
    out_results = _check_timeseries(out_all, "oxygen") if out_all else None
    in_results = _check_timeseries(in_all, "oxygen") if in_all else None
    del_results = _check_timeseries(del_all, "oxygen") if del_all else None

    parts = [time_results[0]]
    locparts = [time_results[1]]
    if out_results is not None:
        parts.append(out_results[0])
        locparts.append(out_results[1])
    if in_results is not None:
        parts.append(in_results[0])
        locparts.append(in_results[1])
    if del_results is not None:
        parts.append(del_results[0])
        locparts.append(del_results[1])
    checks = np.hstack(parts)
    locs_raw = np.hstack(locparts)

    if del_all is None and out_all is not None and in_all is not None:
        del_all = [o - i for o, i in zip(out_all, in_all)]
    elif del_all is None:
        src = out_all if out_all is not None else (in_all if in_all is not None
                                                   else None)
        if src is None:
            del_all = [np.full(df.shape[0], np.nan)]
        else:
            del_all = [np.full(o.shape[0], np.nan) for o in src]

    # dataframe: time + out + in + delta columns
    cols = []
    for t in time_all:
        cols.append(t.reshape(-1, 1))
    for o in (out_all or []):
        cols.append(o.reshape(-1, 1))
    for i in (in_all or []):
        cols.append(i.reshape(-1, 1))
    for d in del_all:
        cols.append(d.reshape(-1, 1))
    dataframe = np.hstack(cols) if cols else df

    all_colnames = ([colnames[t - 1] for t in time_cols] +
                    ([colnames[o - 1] for o in out_cols] if out_cols else []) +
                    ([colnames[i - 1] for i in in_cols] if in_cols else []) +
                    ([colnames[d - 1] for d in del_cols] if del_cols else []))
    locs = [locs_raw[:, i] for i in range(locs_raw.shape[1])]

    out = {"call": None, "dataframe": dataframe, "add.data": add_data,
           "inputs": {"df": x, "time": time, "out.oxy": out_oxy,
                      "in.oxy": in_oxy, "in.oxy.value": in_oxy_value,
                      "delta.oxy": delta_oxy, "plot": plot,
                      "add.data": add_data},
           "data": {"time": time_all, "out.oxy": out_all, "in.oxy": in_all,
                    "delta.oxy": del_all},
           "checks": checks, "locs_raw": locs_raw, "locs": locs,
           "colnames": all_colnames}
    return out
