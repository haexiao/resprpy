"""select_rate / select_rate.ft / test_lin.

Ported from respR 2.3.4. select_rate supports all 27 methods; output mirrors
the R object (summary, rate.output, original, select_calls).
"""
from __future__ import annotations

import numpy as np

from .calc import calc_rate
from .auto import auto_rate
from .data import sim_data


def _get(x, key, default=None):
    if hasattr(x, key):
        return getattr(x, key)
    if isinstance(x, dict):
        return x.get(key, default)
    return default


def _summ_rows(summ, keep):
    """Index every summary column by keep (1-based R indices -> 0-based)."""
    keep0 = np.asarray(keep, dtype=int) - 1
    return {k: np.asarray(v)[keep0] for k, v in summ.items()}


def _arrange(summ, key, desc=False):
    """R arrange(): stable sort of summary row order by column key."""
    vals = np.asarray(summ[key], dtype=float)
    order = np.argsort(vals, kind="stable")
    if desc:
        order = order[::-1]
    return order


def _quantile(a, q):
    return float(np.quantile(np.asarray(a, dtype=float), q))


def _between(vals, lo, hi):
    vals = np.asarray(vals, dtype=float)
    return (vals >= lo) & (vals <= hi)


_METHODS = ("overlap", "duration", "density", "manual", "manual_omit", "time",
            "time_omit", "row", "row_omit", "oxygen", "oxygen_omit", "rsq",
            "intercept", "slope", "rep", "rep_omit", "rank", "rank_omit",
            "rate", "rate.output", "maximum_percentile", "minimum_percentile",
            "maximum", "minimum", "lowest_percentile", "highest_percentile",
            "lowest", "highest", "zero", "nonzero", "negative", "positive",
            "rolling", "linear")


def select_rate(x, method=None, n=None):
    if method is None:
        raise ValueError("select_rate: Please specify a 'method'")
    if method not in _METHODS:
        raise ValueError("select_rate: 'method' input not recognised")

    summary = dict(_get(x, "summary"))
    if hasattr(x, "rate_output"):
        rate_output = np.atleast_1d(np.asarray(x.rate_output, dtype=float))
    else:
        rate_output = np.atleast_1d(
            np.asarray(_get(x, "rate.output"), dtype=float))
    raw_df = _get(x, "dataframe")

    needs_df = ("overlap", "duration", "density", "time", "time_omit", "row",
                "row_omit", "oxygen", "oxygen_omit", "rsq", "intercept",
                "slope", "rolling", "linear")
    if raw_df is None and method in needs_df:
        raise ValueError(
            f"select_rate: The '{method}' method is not accepted for "
            "convert_rate objects which have been created using numeric inputs.")

    nrow_summ = len(rate_output)

    # R: 'density'/'linear' methods only accepted for auto_rate linear objects
    # (summary$density not all NA), unless the summary is empty.
    if method in ("density", "linear"):
        dens = summary.get("density")
        if dens is None:
            is_linear = nrow_summ == 0
        else:
            d = np.asarray(dens, dtype=float)
            is_linear = (not np.all(np.isnan(d))) or nrow_summ == 0
        if not is_linear:
            raise ValueError(
                f"select_rate: The '{method}' method is only accepted for "
                "rates determined in 'auto_rate' via the 'linear' method.")

    reordered = False
    keep = None
    summ = None

    # ---------------- reordering methods ----------------
    # R: 'rolling' reorders by row; 'n' input ignored
    if method == "rolling":
        summ = dict(summary)
        keep = np.argsort(np.asarray(summ["row"], dtype=float), kind="stable") + 1
        reordered = True
    if method == "row" and n is None:
        summ = dict(summary)
        keep = np.argsort(np.asarray(summ["row"], dtype=float), kind="stable") + 1
        reordered = True
    if method == "time" and n is None:
        summ = dict(summary)
        keep = np.argsort(np.asarray(summ["time"], dtype=float), kind="stable") + 1
        reordered = True
    if method == "intercept" and n is None:
        summ = dict(summary)
        keep = np.argsort(np.asarray(summ["intercept_b0"], dtype=float),
                          kind="stable") + 1
        reordered = True
    if method == "slope" and n is None:
        summ = dict(summary)
        keep = np.argsort(np.asarray(summ["slope_b1"], dtype=float),
                          kind="stable") + 1
        reordered = True
    if method == "linear":
        summ = dict(summary)
        keep = np.argsort(-np.asarray(summ["density"], dtype=float),
                          kind="stable") + 1
        reordered = True
    if method == "density" and n is None:
        summ = dict(summary)
        keep = np.argsort(-np.asarray(summ["density"], dtype=float),
                          kind="stable") + 1
        reordered = True
    if method in ("rep", "rank") and n is None:
        summ = dict(summary)
        rep = np.asarray(summ["rep"], dtype=float)
        rnk = np.asarray(summ["rank"], dtype=float)
        keep = np.lexsort((rnk, rep)) + 1
        reordered = True
    if method == "rsq" and n is None:
        summ = dict(summary)
        keep = np.argsort(-np.asarray(summ["rsq"], dtype=float), kind="stable") + 1
        reordered = True
    if method == "lowest" and n is None:
        summ = dict(summary)
        ro = np.asarray(summ["rate.output"], dtype=float)
        if np.all(ro <= 0):
            keep = np.argsort(-ro, kind="stable") + 1
        else:
            keep = np.argsort(ro, kind="stable") + 1
        reordered = True
    if method == "highest" and n is None:
        summ = dict(summary)
        ro = np.asarray(summ["rate.output"], dtype=float)
        if np.all(ro <= 0):
            keep = np.argsort(ro, kind="stable") + 1
        else:
            keep = np.argsort(-ro, kind="stable") + 1
        reordered = True
    if method == "minimum" and n is None:
        summ = dict(summary)
        keep = np.argsort(np.asarray(summ["rate.output"], dtype=float),
                          kind="stable") + 1
        reordered = True
    if method == "maximum" and n is None:
        summ = dict(summary)
        keep = np.argsort(-np.asarray(summ["rate.output"], dtype=float),
                          kind="stable") + 1
        reordered = True

    # ---------------- simple selection methods ----------------
    if method == "positive":
        keep = np.sort(np.where(rate_output > 0)[0] + 1)
        summ = dict(summary)
        reordered = False
    if method == "negative":
        keep = np.sort(np.where(rate_output < 0)[0] + 1)
        summ = dict(summary)
        reordered = False
    if method == "nonzero":
        keep = np.sort(np.where(rate_output != 0)[0] + 1)
        summ = dict(summary)
        reordered = False
    if method == "zero":
        keep = np.sort(np.where(rate_output == 0)[0] + 1)
        summ = dict(summary)
        reordered = False

    # ---------------- n-based selection methods ----------------
    if n is not None and method == "lowest":
        if int(n) != n or n < 0:
            raise ValueError("select_rate: For 'lowest' method 'n' must contain "
                             "only positive integers.")
        if np.all(rate_output <= 0):
            keep = np.sort(np.argsort(rate_output)[-int(n):] + 1)
        elif np.all(rate_output >= 0):
            keep = np.sort(np.argsort(rate_output)[:int(n)] + 1)
        keep = np.sort(keep)
        summ = dict(summary)
        reordered = False
    if n is not None and method == "highest":
        if int(n) != n or n < 0:
            raise ValueError("select_rate: For 'highest' method 'n' must contain "
                             "only positive integers.")
        if np.all(rate_output <= 0):
            keep = np.sort(np.argsort(rate_output)[:int(n)] + 1)
        elif np.all(rate_output >= 0):
            keep = np.sort(np.argsort(rate_output)[-int(n):] + 1)
        keep = np.sort(keep)
        summ = dict(summary)
        reordered = False
    if method == "lowest_percentile":
        if n is None or n <= 0 or n >= 1:
            raise ValueError("select_rate: For 'percentile' methods 'n' must be "
                             "between 0 and 1.")
        if np.all(rate_output <= 0):
            cutoff = _quantile(rate_output, 1 - n)
            keep = np.sort(np.where(rate_output >= cutoff)[0] + 1)
        elif np.all(rate_output >= 0):
            cutoff = _quantile(rate_output, n)
            keep = np.sort(np.where(rate_output <= cutoff)[0] + 1)
        keep = np.sort(keep)
        summ = dict(summary)
        reordered = False
    if method == "highest_percentile":
        if n is None or n <= 0 or n >= 1:
            raise ValueError("select_rate: For 'percentile' methods 'n' must be "
                             "between 0 and 1.")
        if np.all(rate_output <= 0):
            cutoff = _quantile(rate_output, n)
            keep = np.sort(np.where(rate_output <= cutoff)[0] + 1)
        elif np.all(rate_output >= 0):
            cutoff = _quantile(rate_output, 1 - n)
            keep = np.sort(np.where(rate_output >= cutoff)[0] + 1)
        keep = np.sort(keep)
        summ = dict(summary)
        reordered = False
    if n is not None and method == "minimum":
        if int(n) != n or n < 0:
            raise ValueError("select_rate: For 'minimum' method 'n' must contain "
                             "only positive integers.")
        keep = np.sort(np.argsort(rate_output)[:int(n)] + 1)
        keep = np.sort(keep)
        summ = dict(summary)
        reordered = False
    if n is not None and method == "maximum":
        if int(n) != n or n < 0:
            raise ValueError("select_rate: For 'maximum' method 'n' must contain "
                             "only positive integers.")
        keep = np.sort(np.argsort(rate_output)[-int(n):] + 1)
        keep = np.sort(keep)
        summ = dict(summary)
        reordered = False
    if method == "minimum_percentile":
        if n is None or n <= 0 or n >= 1:
            raise ValueError("select_rate: For 'percentile' methods 'n' must be "
                             "between 0 and 1.")
        cutoff = _quantile(rate_output, n)
        keep = np.sort(np.where(rate_output <= cutoff)[0] + 1)
        keep = np.sort(keep)
        summ = dict(summary)
        reordered = False
    if method == "maximum_percentile":
        if n is None or n <= 0 or n >= 1:
            raise ValueError("select_rate: For 'percentile' methods 'n' must be "
                             "between 0 and 1.")
        cutoff = _quantile(rate_output, 1 - n)
        keep = np.sort(np.where(rate_output >= cutoff)[0] + 1)
        keep = np.sort(keep)
        summ = dict(summary)
        reordered = False
    if method in ("rate", "rate.output"):
        if n is None or np.size(n) != 2:
            raise ValueError("select_rate: For 'rate' method 'n' must be a "
                             "vector of two values.")
        lo, hi = sorted(n)
        keep = np.sort(np.where(_between(rate_output, lo, hi))[0] + 1)
        keep = np.sort(keep)
        summ = dict(summary)
        reordered = False
    if n is not None and method == "rsq":
        if np.size(n) != 2:
            raise ValueError("select_rate: For 'rsq' method 'n' must be a vector "
                             "of two values.")
        lo, hi = sorted(n)
        keep = np.sort(np.where(_between(summary["rsq"], lo, hi))[0] + 1)
        keep = np.sort(keep)
        summ = dict(summary)
        reordered = False
    if n is not None and method == "intercept":
        if np.size(n) != 2:
            raise ValueError("select_rate: For 'intercept' method 'n' must be a "
                             "vector of two values.")
        lo, hi = sorted(n)
        keep = np.sort(np.where(_between(summary["intercept_b0"], lo, hi))[0] + 1)
        keep = np.sort(keep)
        summ = dict(summary)
        reordered = False
    if n is not None and method == "slope":
        if np.size(n) != 2:
            raise ValueError("select_rate: For 'slope' method 'n' must be a "
                             "vector of two values.")
        lo, hi = sorted(n)
        keep = np.sort(np.where(_between(summary["slope_b1"], lo, hi))[0] + 1)
        keep = np.sort(keep)
        summ = dict(summary)
        reordered = False
    if n is not None and method == "rep":
        rep = np.asarray(summary["rep"], dtype=float)
        if np.all(np.isnan(rep)):
            raise ValueError("select_rate: All 'rep' are NA so nothing to select!")
        n_arr = np.atleast_1d(np.asarray(n, dtype=float))
        keep = np.sort(np.unique(np.concatenate(
            [np.where(rep == z)[0] + 1 for z in sorted(n_arr)])))
        summ = dict(summary)
        reordered = False
    if n is not None and method == "rep_omit":
        rep = np.asarray(summary["rep"], dtype=float)
        if np.all(np.isnan(rep)):
            raise ValueError("select_rate: All 'rep' are NA so nothing to select!")
        n_arr = np.atleast_1d(np.asarray(n, dtype=float))
        remove = np.sort(np.unique(np.concatenate(
            [np.where(rep == z)[0] + 1 for z in sorted(n_arr)])))
        keep = np.setdiff1d(np.arange(1, nrow_summ + 1), remove)
        summ = dict(summary)
        reordered = False
    if n is not None and method == "rank":
        rnk = np.asarray(summary["rank"], dtype=float)
        n_arr = np.atleast_1d(np.asarray(n, dtype=float))
        keep = np.sort(np.unique(np.concatenate(
            [np.where(rnk == z)[0] + 1 for z in sorted(n_arr)])))
        summ = dict(summary)
        reordered = False
    if n is not None and method == "rank_omit":
        rnk = np.asarray(summary["rank"], dtype=float)
        n_arr = np.atleast_1d(np.asarray(n, dtype=float))
        remove = np.sort(np.unique(np.concatenate(
            [np.where(rnk == z)[0] + 1 for z in sorted(n_arr)])))
        keep = np.setdiff1d(np.arange(1, nrow_summ + 1), remove)
        summ = dict(summary)
        reordered = False
    if n is not None and method == "row":
        if np.size(n) != 2:
            raise ValueError("select_rate: For 'row' method 'n' must be a vector "
                             "of two values.")
        if np.any(np.asarray(n) > raw_df.shape[0]):
            raise ValueError("select_rate: Input for 'n': row inputs out of data "
                             "frame range.")
        lo, hi = sorted(n)
        k1 = np.where(np.asarray(summary["row"]) >= lo)[0] + 1
        k2 = np.where(np.asarray(summary["endrow"]) <= hi)[0] + 1
        keep = np.sort(np.array([k for k in k1 if k in k2]))
        summ = dict(summary)
        reordered = False
    if method == "row_omit":
        if n is None or not np.all(np.asarray(n) == np.floor(np.asarray(n))):
            raise ValueError("select_rate: For 'row_omit' method 'n' must "
                             "contain only positive integers.")
        if np.any(np.asarray(n) > raw_df.shape[0]):
            raise ValueError("select_rate: Input for 'n': row inputs out of data "
                             "frame range.")
        n_order = np.sort(np.asarray(n, dtype=float))
        if n_order.size == 1:
            n_order = np.array([n_order[0], n_order[0]])
        if n_order.size == 2:
            k1 = np.where(np.asarray(summary["row"]) > n_order[1])[0] + 1
            k2 = np.where(np.asarray(summary["endrow"]) < n_order[0])[0] + 1
            keep = np.concatenate([k1, k2])
        else:
            remove = set()
            rows = np.asarray(summary["row"], dtype=int)
            endrows = np.asarray(summary["endrow"], dtype=int)
            for i in n_order:
                for j, (r, e) in enumerate(zip(rows, endrows), start=1):
                    if r <= i <= e:
                        remove.add(j)
            keep = np.array([j for j in range(1, nrow_summ + 1)
                             if j not in remove])
        keep = np.sort(keep)
        summ = dict(summary)
        reordered = False
    if n is not None and method == "time":
        if np.size(n) != 2:
            raise ValueError("select_rate: For 'time' method 'n' must be a "
                             "vector of two values.")
        tmin = np.nanmin(raw_df[:, 0])
        tmax = np.nanmax(raw_df[:, 0])
        if np.any(np.asarray(n) < tmin) or np.any(np.asarray(n) > tmax):
            raise ValueError("select_rate: Input for 'n': time inputs out of "
                             "time data range.")
        lo, hi = sorted(n)
        k1 = np.where(np.asarray(summary["time"]) >= lo)[0] + 1
        k2 = np.where(np.asarray(summary["endtime"]) <= hi)[0] + 1
        keep = np.sort(np.array([k for k in k1 if k in k2]))
        summ = dict(summary)
        reordered = False
    if method == "time_omit":
        if n is None or not np.all(np.asarray(n, dtype=float) ==
                                   np.asarray(n, dtype=float)):
            pass
        tmin = np.nanmin(raw_df[:, 0])
        tmax = np.nanmax(raw_df[:, 0])
        if np.any(np.asarray(n) < tmin) or np.any(np.asarray(n) > tmax):
            raise ValueError("select_rate: Input for 'n': time inputs out of "
                             "time data range.")
        n_order = np.sort(np.asarray(n, dtype=float))
        if n_order.size == 1:
            n_order = np.array([n_order[0], n_order[0]])
        if n_order.size == 2:
            k1 = np.where(np.asarray(summary["time"]) > n_order[1])[0] + 1
            k2 = np.where(np.asarray(summary["endtime"]) < n_order[0])[0] + 1
            keep = np.concatenate([k1, k2])
        else:
            remove = set()
            times = np.asarray(summary["time"], dtype=float)
            endtimes = np.asarray(summary["endtime"], dtype=float)
            for i in n_order:
                for j, (t1, t2) in enumerate(zip(times, endtimes), start=1):
                    if t1 <= i <= t2:
                        remove.add(j)
            keep = np.array([j for j in range(1, nrow_summ + 1)
                             if j not in remove])
        keep = np.sort(keep)
        summ = dict(summary)
        reordered = False
    if method == "oxygen":
        if n is None or np.size(n) != 2:
            raise ValueError("select_rate: For 'oxygen' method 'n' must be a "
                             "vector of two values.")
        lo, hi = sorted(n)
        remove = set()
        rows = np.asarray(summary["row"], dtype=int)
        endrows = np.asarray(summary["endrow"], dtype=int)
        for j, (r, e) in enumerate(zip(rows, endrows), start=1):
            if np.any(raw_df[r - 1:e, 1] <= lo):
                remove.add(j)
            if np.any(raw_df[r - 1:e, 1] >= hi):
                remove.add(j)
        keep = np.array([j for j in range(1, nrow_summ + 1) if j not in remove])
        keep = np.sort(keep)
        summ = dict(summary)
        reordered = False
    if method == "oxygen_omit":
        if n is None:
            raise ValueError("select_rate: For 'oxygen_omit' method 'n' must "
                             "contain only numeric values of oxygen.")
        n_order = np.sort(np.asarray(n, dtype=float))
        remove = set()
        rows = np.asarray(summary["row"], dtype=int)
        endrows = np.asarray(summary["endrow"], dtype=int)
        for z in n_order:
            for j, (r, e) in enumerate(zip(rows, endrows), start=1):
                if np.any(raw_df[r - 1:e, 1] == z):
                    remove.add(j)
        keep = np.array([j for j in range(1, nrow_summ + 1) if j not in remove])
        keep = np.sort(keep)
        summ = dict(summary)
        reordered = False
    if method == "manual":
        if n is None or not np.all(np.isin(n, np.arange(1, nrow_summ + 1))):
            raise ValueError("select_rate: For 'manual' method: 'n' values are "
                             "out of range of $summary data.frame rows...")
        keep = np.sort(np.atleast_1d(np.asarray(n, dtype=int)))
        summ = dict(summary)
        reordered = False
    if method == "manual_omit":
        if n is None or not np.all(np.isin(n, np.arange(1, nrow_summ + 1))):
            raise ValueError("select_rate: For 'manual' method: 'n' values are "
                             "out of range of $summary data.frame rows...")
        keep = np.sort(np.array([j for j in range(1, nrow_summ + 1)
                                 if j not in np.atleast_1d(
                                     np.asarray(n, dtype=int))]))
        summ = dict(summary)
        reordered = False
    if n is not None and method == "density":
        if np.size(n) != 2:
            raise ValueError("select_rate: For 'density' method 'n' must be a "
                             "vector of two values.")
        lo, hi = sorted(n)
        keep = np.sort(np.where(_between(summary["density"], lo, hi))[0] + 1)
        keep = np.sort(keep)
        summ = dict(summary)
        reordered = False
    if method == "duration":
        if n is None or np.size(n) != 2:
            raise ValueError("select_rate: For 'duration' method 'n' must be a "
                             "vector of two values.")
        lo, hi = sorted(n)
        durations = np.asarray(summary["endtime"]) - np.asarray(summary["time"])
        k1 = np.where(durations >= lo)[0] + 1
        k2 = np.where(durations <= hi)[0] + 1
        keep = np.sort(np.array([k for k in k1 if k in k2]))
        summ = dict(summary)
        reordered = False
    if method == "overlap":
        if n is None:
            n = 0
        if n < 0 or n > 1:
            raise ValueError("select_rate: For 'overlap' method 'n' must be "
                             "between 0 and 1 inclusive.")
        rows = list(np.asarray(summary["row"], dtype=float))
        endrows = list(np.asarray(summary["endrow"], dtype=float))
        ranks = list(np.asarray(summary["rank"], dtype=float))
        width = [e - r for r, e in zip(rows, endrows)]
        overlaps = True
        while overlaps:
            ov_list = [False] * len(rows)
            for i in range(len(rows) - 1, -1, -1):
                start, end, w = rows[i], endrows[i], width[i]
                ov = int(round(w * n))
                ov_list[i] = any(
                    rows[j] + ov <= end and endrows[j] - ov >= start
                    for j in range(len(rows)) if j != i)
            overlaps = any(ov_list)
            if overlaps:
                remove = max(j for j, v in enumerate(ov_list) if v)
                del rows[remove]
                del endrows[remove]
                del width[remove]
                del ranks[remove]
        orig_rank = np.asarray(summary["rank"], dtype=float)
        keep = np.sort(np.array([j + 1 for j in range(nrow_summ)
                                 if orig_rank[j] in ranks]))
        summ = dict(summary)
        reordered = False

    if keep is None:
        raise ValueError(f"select_rate: method '{method}' not applied.")

    # ---------------- assemble output ----------------
    # R semantics: return a copy with summary/rate.output replaced; the
    # input object is never modified (select_rate keeps 'original').
    import copy
    out = copy.copy(x) if hasattr(x, "summary") else dict(x)
    if hasattr(out, "summary"):
        out.summary = _summ_rows(summ, keep)
        out.rate_output = np.atleast_1d(
            np.asarray(summ["rate.output"])[np.asarray(keep, dtype=int) - 1])
        if getattr(out, "original", None) is None:
            out.original = x
        out.select_calls = getattr(out, "select_calls", None) or []
        out.select_calls = out.select_calls + [None]
    else:
        out["summary"] = _summ_rows(summ, keep)
        out["rate.output"] = np.atleast_1d(
            np.asarray(summ["rate.output"])[np.asarray(keep, dtype=int) - 1])
        if out.get("original") is None:
            out["original"] = x
        out["select_calls"] = out.get("select_calls") or []
        out["select_calls"] = out["select_calls"] + [None]
    return out


def select_rate_ft(x, method=None, n=None):
    return select_rate(x, method, n)


# ---------------------------------------------------------------------------
# test_lin
# ---------------------------------------------------------------------------
def test_lin(reps=1, len_=300, sd=0.05, type="default", preview=False,
             plot=False, seed=None):
    if seed is not None:
        np.random.seed(seed)

    def run_once():
        dt = sim_data(len=len_, type=type, preview=preview)
        coef_sim = dt["coef"]
        df = dt["df"]
        len_main = dt["len_main"]
        seg_index = dt["seg_index"]
        autorate = auto_rate(df, plot=False)
        coef_meas = np.atleast_1d(autorate.rate)[0]
        sttrow = int(autorate.summary["row"][0])
        endrow = int(autorate.summary["endrow"][0])
        seg_detec = np.arange(sttrow, endrow + 1)
        ins = int(np.sum(np.isin(seg_detec, seg_index)))
        outs = int(np.sum(~np.isin(seg_detec, seg_index)))
        return [coef_sim, coef_meas, len_main, outs, ins]

    runs = np.array([run_once() for _ in range(reps)])
    names = ["real", "measured", "length_line", "length_incorrect",
             "length_detected"]
    df_out = {names[i]: runs[:, i] for i in range(5)}
    # lm(real ~ measured)
    X = np.column_stack([np.ones(reps), runs[:, 1]])
    coef, *_ = np.linalg.lstsq(X, runs[:, 0], rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((runs[:, 0] - pred) ** 2))
    ss_tot = float(np.sum((runs[:, 0] - np.mean(runs[:, 0])) ** 2))
    rsq = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    test = {"coefficients": coef, "rsq": rsq}
    return {"df": df_out, "results": test}
