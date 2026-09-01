"""Intermittent-flow and flow-through respirometry: calc_rate.int, calc_rate.bg,
calc_rate.ft, calc_rate.rep, auto_rate.int, auto_rate.rep.

Ported from respR 2.3.4, numerical behaviour identical to R.
"""
from __future__ import annotations

import numpy as np

from .calc import _by_val, _extract_indices, _truncate_data, calc_rate
from .auto import auto_rate, calc_win, rolling_reg_row

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _static_roll(df, win):
    """roll::roll_lm over a fixed window, dropping the first win-1 rows."""
    n = df.shape[0]
    out = []
    for i in range(win - 1, n):
        sub = df[i - win + 1:i + 1]
        t = sub[:, 0].astype(float)
        o = sub[:, 1].astype(float)
        X = np.column_stack([np.ones_like(t), t])
        coef, *_ = np.linalg.lstsq(X, o, rcond=None)
        b0, b1 = coef
        pred = X @ coef
        ss_res = float(np.sum((o - pred) ** 2))
        ss_tot = float(np.sum((o - np.mean(o)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        out.append([b0, b1, r2])
    return np.array(out)


def _time_lm(df, start, end):
    """lm(y ~ x) over rows with x in [start, end]; full-precision rsq (no signif)."""
    x = df[:, 0].astype(float)
    y = df[:, 1].astype(float)
    mask = (x >= start) & (x <= end)
    sx, sy = x[mask], y[mask]
    X = np.column_stack([np.ones_like(sx), sx])
    coef, *_ = np.linalg.lstsq(X, sy, rcond=None)
    b0, b1 = coef
    pred = X @ coef
    ss_res = float(np.sum((sy - pred) ** 2))
    ss_tot = float(np.sum((sy - np.mean(sy)) ** 2))
    rsq = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return b0, b1, rsq


def _seq_to(n, by):
    """R: seq(1, n, by)"""
    return np.arange(1, n + 1, by, dtype=float)


def _midpt(p):
    return (float(np.min(p)) + float(np.max(p))) / 2.0


# ---------------------------------------------------------------------------
# calc_rate.rep
# ---------------------------------------------------------------------------
def calc_rate_rep(x, from_=None, to=None, by="time", plot=True, rep=1, **kwargs):
    out = calc_rate(x, from_=from_, to=to, by=by, plot=plot, **kwargs)
    out.summary["rep"] = np.full(np.size(out.rate), float(rep))
    return out


# ---------------------------------------------------------------------------
# calc_rate.int
# ---------------------------------------------------------------------------
def calc_rate_int(x, starts=None, wait=None, measure=None, by="row", plot=True,
                  **kwargs):
    df = np.asarray(x, dtype=float)
    if df.ndim == 1:
        df = df.reshape(-1, 1)
    if df.shape[1] > 2:
        df = df[:, :2]
    nrow = df.shape[0]
    by = _by_val(by, req=True, default="row", which=("t", "r"), msg="calc_rate.int")

    starts = np.atleast_1d(np.asarray(starts, dtype=float))
    if starts.size == 1 and by == "row":
        starts = _seq_to(nrow, starts[0])
    if starts.size == 1 and by == "time":
        # R: seq(df[[1]][1], tail(df[[1]],1), starts) -- closed end
        starts = np.arange(df[0, 0], df[-1, 0] + 1e-9, starts[0], dtype=float)
    if by == "row":
        ends = np.concatenate([starts[1:] - 1, [nrow]])
    else:  # time: map to nearest row indices first
        starts = np.array([int(np.argmin(np.abs(df[:, 0] - z))) + 1 for z in starts])
        ends = np.concatenate([starts[1:] - 1, [nrow]])

    if wait is None:
        wait = 0
    if measure is None:
        measure = nrow if by == "row" else float(np.max(df[:, 0]))
    if by == "row":
        from_ = wait + 1
        to = wait + measure
    else:
        from_ = wait
        to = wait + measure

    reps = [_truncate_data(df, starts[i], ends[i], "row") for i in range(len(starts))]
    res = []
    for q, p in enumerate(reps, start=1):
        if by == "row":
            r = calc_rate_rep(p, from_=from_, to=to, by="row", plot=False, rep=q)
        else:
            from_rel = from_ + p[0, 0]
            to_rel = to + p[0, 0]
            r = calc_rate_rep(p, from_=from_rel, to=to_rel, by="time", plot=False, rep=q)
        res.append(r)

    # merge summaries
    cols = list(res[0].summary.keys())
    summ = {c: np.concatenate([r.summary[c] for r in res]) for c in cols}
    row_width = summ["endrow"] - summ["row"]
    if by == "row":
        summ["row"] = starts - 1 + summ["row"]
        summ["endrow"] = summ["row"] + row_width
    else:
        cum = np.concatenate([[0.0], np.cumsum([p.shape[0] for p in reps])])[:-1]
        summ["row"] = cum + summ["row"]
        summ["endrow"] = summ["row"] + row_width

    rate = summ["rate"]
    from .calc import CalcRate
    out = CalcRate(call=None, inputs=dict(x=x, starts=starts, wait=wait,
                                          measure=measure, by=by, plot=plot),
                   dataframe=df, subsets=reps, summary=summ, rate=rate,
                   rate_2pt=summ.get("rate.2pt"))
    return out


# ---------------------------------------------------------------------------
# calc_rate.bg
# ---------------------------------------------------------------------------
def calc_rate_bg(x, time=1, oxygen=None, plot=True, **kwargs):
    if isinstance(x, dict) and "dataframe" in x:  # inspect object
        df = np.asarray(x["dataframe"], dtype=float)
    else:
        df = np.asarray(x, dtype=float)
    if df.ndim == 1:
        df = df.reshape(-1, 1)
    if time is None:
        time = 1
    if oxygen is None:
        oxygen = [i for i in range(1, df.shape[1] + 1) if i != time]
    xval = df[:, time - 1]
    nres = len(oxygen)
    b0s, b1s, r2s = [], [], []
    for o in oxygen:
        y = df[:, o - 1]
        X = np.column_stack([np.ones_like(xval), xval])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        b0, b1 = coef
        pred = X @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        b0s.append(b0)
        b1s.append(b1)
        r2s.append(r2)
    b0s = np.array(b0s)
    b1s = np.array(b1s)
    r2s = np.array(r2s)
    n = df.shape[0]
    oxy_first = np.array([df[0, o - 1] for o in oxygen])
    oxy_last = np.array([df[n - 1, o - 1] for o in oxygen])
    summary = {
        "rep": np.full(nres, np.nan),
        "rank": np.arange(1, nres + 1, dtype=float),
        "intercept_b0": b0s,
        "slope_b1": b1s,
        "rsq": r2s,
        "row": np.ones(nres),
        "endrow": np.full(nres, float(n)),
        "time": np.full(nres, float(df[0, 0])),
        "endtime": np.full(nres, float(df[n - 1, 0])),
        "oxy": oxy_first,
        "endoxy": oxy_last,
        "rate.bg": b1s,
    }
    bg = b1s
    out = {
        "call": None,
        "inputs": dict(x=x, time=time, oxygen=oxygen, plot=plot),
        "dataframe": df,
        "lm": None,
        "summary": summary,
        "rate.bg": bg,
        "rate.bg.mean": float(np.mean(bg)),
    }
    return out


# ---------------------------------------------------------------------------
# calc_rate.ft
# ---------------------------------------------------------------------------
def _na_like(a):
    if a.ndim == 1:
        return np.full(a.shape[0], np.nan)
    return np.full(a.shape, np.nan)


def calc_rate_ft(x=None, flowrate=None, from_=None, to=None, by=None,
                 width=None, plot=True, **kwargs):
    # x: numeric vector/matrix, or inspect.ft dict. R's is.numeric(matrix) is
    # TRUE, so numpy 2-D arrays take the 'vec' branch exactly like R matrices.
    if flowrate is None:
        raise ValueError("calc_rate.ft: 'flowrate' input is required.")
    if isinstance(x, dict) and "data" in x:
        xtype = "insp"
    elif np.isscalar(x) or isinstance(x, (list, np.ndarray)):
        xtype = "vec"
    else:
        raise ValueError("calc_rate.ft: 'x' must be an inspect.ft object, a "
                         "numeric value or vector, or 2-column data.frame.")

    if xtype == "vec":
        delta = np.asarray(x, dtype=float)
        if delta.ndim == 0:
            delta = delta.reshape(1)
        data = delta
        summary = {
            "intercept_b0": _na_like(delta),
            "slope_b1": _na_like(delta),
            "rsq": _na_like(delta),
            "row": _na_like(delta),
            "endrow": _na_like(delta),
            "time": _na_like(delta),
            "endtime": _na_like(delta),
            "oxy": _na_like(delta),
            "endoxy": _na_like(delta),
            "delta_mean": delta,
        }
    else:  # insp
        data = x["data"]
        time_arr = np.asarray(data["time"][0], dtype=float)
        delta_arr = np.asarray(data["delta.oxy"][0], dtype=float)
        dt = np.column_stack([time_arr, delta_arr])
        t_range = (np.nanmin(time_arr), np.nanmax(time_arr))
        r_range = (1, time_arr.size)
        if by is None:
            by = "time"
        by = _by_val(by, which=("t", "r"), msg="calc_rate.ft")
        if width is None:
            if by == "time":
                if from_ is None and to is None:
                    from_, to = t_range[0], t_range[1]
                if from_ is None:
                    from_ = t_range[0]
                if to is None:
                    to = t_range[1]
            else:  # row
                if from_ is None and to is None:
                    from_, to = float(r_range[0]), float(r_range[1])
                if from_ is None:
                    from_ = float(r_range[0])
                if to is None:
                    to = float(r_range[1])
        if by == "time":
            if np.any(np.atleast_1d(np.asarray(from_, dtype=float)) > t_range[1]):
                raise ValueError("calc_rate.ft: Some 'from' time values are higher "
                                 "than the values present in 'x'.")
            if np.any(np.atleast_1d(np.asarray(to, dtype=float)) < t_range[0]):
                raise ValueError("calc_rate.ft: Some 'to' time values are lower "
                                 "than the values present in 'x'.")
        if by == "row":
            if np.any(np.atleast_1d(np.asarray(from_, dtype=float)) > r_range[1]):
                raise ValueError("calc_rate.ft: Some 'from' row numbers are beyond "
                                 "the number of rows present in 'x'.")
        from_arr = np.atleast_1d(np.asarray(from_, dtype=float))
        to_arr = np.atleast_1d(np.asarray(to, dtype=float))
        if from_arr.size != to_arr.size:
            raise ValueError("calc_rate.ft: 'from' and 'to' have unequal lengths.")
        if np.any(np.array([p > q for p, q in zip(from_arr, to_arr)])):
            raise ValueError("calc_rate.ft: Some 'from' values are greater than "
                             "the paired values in 'to'.")
        if width is not None:
            if by == "time":
                raise ValueError("calc_rate.ft: 'width' can only be used with "
                                 "'by = row'.")
            win = calc_win(dt, width, by, "calc_rate.ft")
            roll = rolling_reg_row(dt, width=win)
            # R rolling_reg_row cols: row,endrow,time,endtime,oxy,endoxy,b0,b1,rsq
            # names(summary)[8] <- "slope_b1"; delta_mean appended (col 10);
            # summary <- summary[, c(7:9, 1:6, 10)] -> final order:
            #   b0, slope_b1, rsq, row, endrow, time, endtime, oxy, endoxy, delta_mean
            dmean = np.array([np.mean(delta_arr[int(roll[i, 0]) - 1:int(roll[i, 1])])
                              for i in range(roll.shape[0])])
            s = roll[:, [6, 7, 8, 0, 1, 2, 3, 4, 5]]
            summary = {"intercept_b0": s[:, 0], "slope_b1": s[:, 1],
                       "rsq": s[:, 2], "row": s[:, 3], "endrow": s[:, 4],
                       "time": s[:, 5], "endtime": s[:, 6], "oxy": s[:, 7],
                       "endoxy": s[:, 8], "delta_mean": dmean}
            delta = dmean
        else:
            dfs = [_truncate_data(dt, from_arr[z], to_arr[z], by)
                   for z in range(len(from_arr))]
            indices = [_extract_indices(dt, dfs, z) for z in range(len(dfs))]
            rows = []
            for z in range(len(dfs)):
                ix = indices[z]
                b0, b1, r2 = _time_lm(dfs[z], ix["time"], ix["endtime"])
                rows.append([b0, b1, r2, ix["row"], ix["endrow"], ix["time"],
                             ix["endtime"], ix["oxy"], ix["endoxy"]])
            arr = np.array(rows)
            dmean = np.array([np.mean(delta_arr[int(ix["row"]) - 1:int(ix["endrow"])])
                              for ix in indices])
            summary = {"intercept_b0": arr[:, 0], "slope_b1": arr[:, 1],
                       "rsq": arr[:, 2], "row": arr[:, 3], "endrow": arr[:, 4],
                       "time": arr[:, 5], "endtime": arr[:, 6],
                       "oxy": arr[:, 7], "endoxy": arr[:, 8],
                       "delta_mean": dmean}
            delta = dmean

    rate = delta * flowrate
    n = rate.shape[0] if rate.ndim > 1 else rate.size
    summ_tbl = {**summary, "rep": np.full(n, np.nan),
                "rank": np.arange(1, n + 1, dtype=float),
                "flowrate": np.full(n, float(flowrate)), "rate": rate}
    if xtype == "insp":
        dframe = dt
    elif delta.ndim == 1:
        dframe = np.column_stack([np.full(delta.size, np.nan), delta])
    else:
        dframe = np.column_stack([np.full(delta.shape[0], np.nan), delta])
    out = {"call": None,
           "inputs": {"x": x, "flowrate": flowrate, "from_": from_, "to": to,
                      "by": by, "width": width, "plot": plot},
           "dataframe": dframe, "data": data, "subsets": None,
           "delta.oxy": delta, "input_type": xtype, "summary": summ_tbl,
           "rate": rate}
    return out


# ---------------------------------------------------------------------------
# auto_rate.rep / auto_rate.int
# ---------------------------------------------------------------------------
def auto_rate_rep(x, method="linear", width=None, by="row", plot=True, rep=1,
                  n=1, rep_row=None, meas_row=None, meas_endrow=None,
                  rep_data=None, **kwargs):
    out = auto_rate(x, method=method, width=width, by=by, plot=plot, **kwargs)
    nrates = np.size(out.rate)
    out.summary["rep"] = np.full(nrates, float(rep))
    if n > nrates:
        n = nrates
    for k in out.summary:
        out.summary[k] = out.summary[k][:n]
    out.rate = out.rate[:n]
    if "metadata" not in out.metadata:
        out.metadata = dict(out.metadata)
    out.metadata["subset_regs"] = n
    if out.peaks is not None and np.size(out.peaks):
        out.peaks = out.peaks[:n]
    out.rep_data = rep_data
    out.summary["rep_row"] = np.full(n, np.nan if rep_row is None else float(rep_row))
    out.summary["meas_row"] = np.full(n, np.nan if meas_row is None else float(meas_row))
    out.summary["meas_endrow"] = np.full(n, np.nan if meas_endrow is None else float(meas_endrow))
    return out


def auto_rate_int(x, starts=None, wait=None, measure=None, by="row",
                  method="linear", width=None, n=1, plot=True, **kwargs):
    df = np.asarray(x, dtype=float)
    if df.ndim == 1:
        df = df.reshape(-1, 1)
    if df.shape[1] > 2:
        df = df[:, :2]
    nrow = df.shape[0]
    by = _by_val(by, req=True, default="row", which=("t", "r"), msg="auto_rate.int")
    if width is None:
        raise ValueError("auto_rate.int: Please enter a 'width'.")

    starts = np.atleast_1d(np.asarray(starts, dtype=float))
    if starts.size == 1 and by == "row":
        starts = _seq_to(nrow, starts[0])
    if starts.size == 1 and by == "time":
        # R: seq(df[[1]][1], tail(df[[1]],1), starts) -- closed end
        starts = np.arange(df[0, 0], df[-1, 0] + 1e-9, starts[0], dtype=float)
    if by == "row":
        ends = np.concatenate([starts[1:] - 1, [nrow]])
    else:
        starts = np.array([int(np.argmin(np.abs(df[:, 0] - z))) + 1 for z in starts])
        ends = np.concatenate([starts[1:] - 1, [nrow]])

    if wait is None:
        wait = 0
    if measure is None:
        measure = nrow if by == "row" else float(np.max(df[:, 0]))
    if by == "row":
        from_ = wait + 1
        to = wait + measure
    else:
        from_ = wait
        to = wait + measure

    reps = [_truncate_data(df, starts[i], ends[i], "row") for i in range(len(starts))]
    offsets = np.sort(np.concatenate([[0.0],
                                      np.cumsum([p.shape[0] for p in reps])[:-1]]))
    res = []
    for q, p in enumerate(reps, start=1):
        t_off = float(offsets[q - 1])
        if by == "row":
            sub = _truncate_data(p, from_=from_, to=to, by="row")
            meas_row = float(from_)
            meas_endrow = float(to)
        else:
            from_rel = from_ + p[0, 0]
            to_rel = to + p[0, 0]
            sub = _truncate_data(p, from_=from_rel, to=to_rel, by="time")
            meas_row = float(np.argmin(np.abs(p[:, 0] - from_rel)))
            meas_endrow = float(np.argmin(np.abs(p[:, 0] - to_rel)))
        r = auto_rate_rep(sub, method=method, width=width, by=by, plot=False,
                          rep=q, n=n, rep_row=t_off, meas_row=meas_row,
                          meas_endrow=meas_endrow, rep_data=p)
        res.append(r)

    cols = list(res[0].summary.keys())
    summ = {c: np.concatenate([r.summary[c] for r in res]) for c in cols}
    row_width = summ["endrow"] - summ["row"]
    if by == "row":
        summ["row"] = summ["row"] + summ["rep_row"] + summ["meas_row"] - 1
        summ["endrow"] = summ["row"] + row_width
    else:
        summ["row"] = summ["row"] + summ["rep_row"] + summ["meas_row"]
        summ["endrow"] = summ["row"] + row_width
    for c in ("rep_row", "meas_row", "meas_endrow"):
        summ.pop(c, None)
    for r in res:
        for c in ("rep_row", "meas_row", "meas_endrow"):
            r.summary.pop(c, None)
    rate = summ["rate"]
    out = {"call": None,
           "inputs": dict(x=x, starts=starts, wait=wait, measure=measure, by=by,
                          method=method, width=width, n=n, plot=plot),
           "dataframe": df, "subsets": reps, "results": res, "summary": summ,
           "rate": rate}
    return out
