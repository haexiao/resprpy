"""adjust_rate / adjust_rate.ft: subtract background rates from measured rates.

Ported from respR 2.3.4. Methods: value, mean, paired, concurrent, linear,
exponential.
"""
from __future__ import annotations

import numpy as np

from .calc import _by_val
from .intflow import _midpt, calc_rate_bg
from .calc import calc_rate
from .data import subset_data


def _rate_obj_parts(x):
    """Return (rate_array, summary_dict) for calc_rate*/auto_rate* objects."""
    if isinstance(x, dict):
        if "rate" in x and "summary" in x:
            return (np.atleast_1d(np.asarray(x["rate"], dtype=float)),
                    x["summary"])
        return None, None
    if hasattr(x, "rate") and hasattr(x, "summary"):
        return np.atleast_1d(np.asarray(x.rate, dtype=float)), x.summary
    return None, None


def _bg_values(by):
    """Extract background rate values from calc_rate.bg / calc_rate / numeric."""
    if isinstance(by, dict):
        if "rate.bg" in by:
            return np.atleast_1d(np.asarray(by["rate.bg"], dtype=float))
        if "rate" in by and "summary" in by:
            return np.atleast_1d(np.asarray(by["rate"], dtype=float))
    if hasattr(by, "rate") and hasattr(by, "summary"):
        return np.atleast_1d(np.asarray(by.rate, dtype=float))
    return np.atleast_1d(np.asarray(by, dtype=float))


def _bg_df(by):
    """Get the background dataframe from inspect / calc_rate.bg / calc_rate."""
    if isinstance(by, dict) and "dataframe" in by:
        return np.asarray(by["dataframe"], dtype=float)
    if hasattr(by, "dataframe") and by.dataframe is not None:
        return np.asarray(by.dataframe, dtype=float)
    return np.asarray(by, dtype=float)


def adjust_rate(x, by, method=None, by2=None, time_x=None, time_by=None,
                time_by2=None):
    if method is None:
        method = "mean"
    if method not in ("value", "mean", "paired", "concurrent", "linear",
                      "exponential"):
        raise ValueError("adjust_rate: 'method' input not recognised.")
    dynamic = method in ("linear", "exponential")

    rate_obj, x_summary = _rate_obj_parts(x)
    if rate_obj is not None:
        rate = rate_obj
    else:
        rate = np.atleast_1d(np.asarray(x, dtype=float))

    if method == "value":
        bg1 = _bg_values(by)
        adjustment = np.asarray(bg1, dtype=float).ravel()
        out_model = None
        rate_adjusted = rate - adjustment
    elif method == "mean":
        bg1 = _bg_values(by)
        if bg1.size > 1:
            pass  # message in R; mean used
        adjustment = np.asarray([float(np.mean(bg1))])
        out_model = None
        rate_adjusted = rate - adjustment
    elif method == "paired":
        bg1 = _bg_values(by)
        if rate.size != bg1.size:
            raise ValueError("adjust_rate: for method = 'paired' the 'x' and "
                             "'by' inputs should have the same number of rates.")
        adjustment = bg1
        out_model = None
        rate_adjusted = rate - adjustment
    elif method == "concurrent":
        rate_obj_x, x_sum = _rate_obj_parts(x)
        if rate_obj_x is None:
            raise ValueError("adjust_rate: For method = 'concurrent' the 'x' "
                             "input must be a rate object.")
        starts = np.atleast_1d(np.asarray(x_sum["time"], dtype=float))
        ends = np.atleast_1d(np.asarray(x_sum["endtime"], dtype=float))
        bg_df = _bg_df(by)
        adjustments = []
        for p, q in zip(starts, ends):
            sub = subset_data(bg_df, from_=p, to=q, by="time")
            bgo = calc_rate_bg(sub, time=1,
                               oxygen=[i + 1 for i in range(1, sub.shape[1])],
                               plot=False)
            adjustments.append(float(np.mean(bgo["rate.bg"])))
        adjustment = np.array(adjustments)
        out_model = None
        if rate.size != adjustment.size:
            raise ValueError("adjust_rate: Error applying adjustment. Input "
                             "rates and adjustment are different lengths.")
        rate_adjusted = rate - adjustment
    elif dynamic:
        if rate_obj is not None and x_summary is not None:
            t_rate = (np.atleast_1d(np.asarray(x_summary["time"], dtype=float)) +
                      np.atleast_1d(np.asarray(x_summary["endtime"], dtype=float))) / 2.0
        else:
            t_rate = np.atleast_1d(np.asarray(time_x, dtype=float))
            if t_rate.size != rate.size:
                raise ValueError("adjust_rate: For a numeric 'x' input, 'time_x' "
                                 "must be of the same length.")

        def bg_point(b, tb):
            if isinstance(b, dict) and "dataframe" in b:
                df = np.asarray(b["dataframe"], dtype=float)
                bgv = np.mean(calc_rate_bg(
                    df, time=1, oxygen=[i + 1 for i in range(1, df.shape[1])],
                    plot=False)["rate.bg"])
                return bgv, _midpt(df[:, 0])
            if isinstance(b, np.ndarray) and b.ndim == 2:
                bgv = np.mean(calc_rate_bg(
                    b, time=1, oxygen=[i + 1 for i in range(1, b.shape[1])],
                    plot=False)["rate.bg"])
                return bgv, _midpt(b[:, 0])
            return float(np.asarray(b, dtype=float).ravel()[0]), float(tb)

        if by2 is None:
            raise ValueError("adjust_rate: For dynamic methods 'by2' input is "
                             "required.")
        bg1, t_bg1 = bg_point(by, time_by)
        bg2, t_bg2 = bg_point(by2, time_by2)
        if t_bg2 < t_bg1:
            raise ValueError("adjust_rate: Timestamp for 'by2' is before "
                             "timestamp for 'by'.")
        if method == "linear":
            X = np.column_stack([np.ones(2), [t_bg1, t_bg2]])
            coef, *_ = np.linalg.lstsq(X, [bg1, bg2], rcond=None)
            bg_int, bg_slp = coef
            if np.isnan(bg_slp):
                bg_slp = 0.0
            adjustment = t_rate * bg_slp + bg_int
            out_model = None
        else:  # exponential
            if bg1 == 0 or bg2 == 0:
                raise ValueError("adjust_rate: method = 'exponential' cannot be "
                                 "used when a 'by' or 'by2' background rate is "
                                 "zero.")
            if np.sign(bg1) != np.sign(bg2):
                raise ValueError("adjust_rate: method = 'exponential' cannot be "
                                 "used when 'by' and 'by2' background rates are "
                                 "not the same sign.")
            both_neg = bg1 < 0 and bg2 < 0
            b1, b2 = (-bg1, -bg2) if both_neg else (bg1, bg2)
            X = np.column_stack([np.ones(2), [t_bg1, t_bg2]])
            coef, *_ = np.linalg.lstsq(X, np.log([b1, b2]), rcond=None)
            exp_int, exp_slp = np.exp(coef[0]), np.exp(coef[1])
            adjustment = exp_int * exp_slp ** t_rate
            if both_neg:
                adjustment = -adjustment
            out_model = None
        if rate.size != adjustment.size:
            raise ValueError("adjust_rate: Error applying adjustment. Input "
                             "rates and adjustment are different lengths.")
        rate_adjusted = rate - adjustment
    else:
        raise ValueError("adjust_rate: 'method' input not recognised.")

    if adjustment.size == 1:
        adjustment = np.full(rate_adjusted.size, adjustment.ravel()[0])

    if rate_obj is not None and x_summary is not None:
        summary = dict(x_summary)
        summary["adjustment"] = adjustment
        summary["rate.adjusted"] = rate_adjusted
        if isinstance(x, dict) and "dataframe" in x:
            df = np.asarray(x["dataframe"], dtype=float)
        elif hasattr(x, "dataframe") and x.dataframe is not None:
            df = np.asarray(x.dataframe, dtype=float)
        else:
            df = None
    else:
        summary = {"rank": np.arange(1, rate_adjusted.size + 1, dtype=float),
                   "rate": rate, "adjustment": adjustment,
                   "rate.adjusted": rate_adjusted}
        df = None

    out = {"call": None,
           "inputs": dict(x=x, by=by, by2=by2, time_x=time_x,
                          time_by=time_by, time_by2=time_by2),
           "dataframe": df, "summary": summary,
           "adjustment.method": method, "adjustment.model": out_model,
           "rate": rate, "adjustment": adjustment,
           "rate.adjusted": rate_adjusted}
    return out


def adjust_rate_ft(x, by):
    """Flow-through background adjustment: rate.adjusted = rate - adjustment."""
    if isinstance(x, dict) and "rate" in x:
        rate = np.atleast_1d(np.asarray(x["rate"], dtype=float))
    else:
        rate = np.atleast_1d(np.asarray(x, dtype=float))
    if isinstance(by, dict) and "rate" in by:
        adjustment = np.atleast_1d(np.asarray(by["rate"], dtype=float))
    else:
        adjustment = np.atleast_1d(np.asarray(by, dtype=float))
    if adjustment.size > 1:
        adjustment = np.array([float(np.mean(adjustment))])
    rate_adjusted = rate - adjustment
    if adjustment.size == 1:
        adj_full = np.full(rate_adjusted.size, adjustment.ravel()[0])
    else:
        adj_full = adjustment
    if isinstance(x, dict) and "rate" in x:
        summary = dict(x["summary"])
        summary["adjustment"] = adj_full
        summary["rate.adjusted"] = rate_adjusted
        out = {"call": None,
               "inputs": dict(x=x, by=by, from_=x.get("from"), to=x.get("to"),
                              by_=x.get("by"), width=x.get("width"),
                              flowrate=x.get("flowrate")),
               "dataframe": x.get("dataframe"), "data": x.get("data"),
               "subsets": x.get("subsets"), "delta.oxy": x.get("delta.oxy"),
               "input_type": x.get("input_type"), "summary": summary,
               "rate": rate, "adjustment": adj_full,
               "rate.adjusted": rate_adjusted}
    else:
        summary = {"rank": np.arange(1, rate_adjusted.size + 1, dtype=float),
                   "rate": rate, "adjustment": adj_full,
                   "rate.adjusted": rate_adjusted}
        out = {"call": None, "inputs": dict(x=x, by=by), "dataframe": None,
               "data": None, "subsets": None, "delta.oxy": None,
               "input_type": None, "summary": summary, "rate": rate,
               "adjustment": adj_full, "rate.adjusted": rate_adjusted}
    return out
