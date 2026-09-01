"""format_time: add a numeric time column (seconds) to a data frame.

Ported from respR 2.3.4. Parses a datetime column (lubridate-style formats)
and returns the elapsed seconds since the first row, plus `start`
(respR default start = 1, so the first row is time 1 -- this is why the
user's rmr outputs start at time 781 for row 157: 780 s elapsed + 1).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

# fmt strings tried in order (lubridate "ymdHMS" accepts these shapes)
_CANDIDATES = [
    "%y/%m/%d %H:%M:%S",   # after respR's format(dtm, "%y/%m/%d %H:%M:%S")
    "%Y-%m-%d %H:%M:%S",   # raw OXY-10 xlsx timestamps
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
]


def _parse_dt(value):
    if isinstance(value, datetime):
        # respR round-trips POSIXct -> character (which truncates fractional
        # seconds) -> parse, so mimic by truncating to whole seconds.
        return value.replace(microsecond=0)
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[us]").astype(datetime).replace(microsecond=0)
    s = str(value).strip()
    for fmt in _CANDIDATES:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"format_time: failed to parse datetime {value!r}")


def format_time(x, time=1, format="ymdHMS", start=1):
    """Add a numeric time column (seconds since first row + start).

    Parameters
    ----------
    x : array-like, shape (n, k)
        Data with a datetime column (strings or datetime objects).
    time : int or list of int
        1-based column index(es) of the datetime column(s); multiple
        columns are pasted together.
    format : str
        Accepted for respR compatibility; only lubridate 'ymdHMS'-style
        formats are supported (auto-detected).
    start : float
        Value of the first time point (respR default 1).

    Returns
    -------
    (x, time_num) : the original data plus a time_num column.
    """
    x = np.asarray(x, dtype=object) if not hasattr(x, "columns") else x
    if isinstance(time, (list, tuple)):
        cols = [int(i) - 1 for i in time]
        ts = [" ".join(str(x[r, c]) for c in cols) for r in range(x.shape[0])]
    else:
        c = int(time) - 1
        ts = [x[r, c] for r in range(x.shape[0])]

    dtm = [_parse_dt(t) for t in ts]
    if all(d is None for d in dtm):
        raise ValueError("format_time: failed to parse data. Please check 'format' input.")

    t0 = dtm[0]
    intervals = np.array([(d - t0).total_seconds() for d in dtm])

    # times crossing midnight (time-only data going backwards)
    if np.any(np.diff(intervals) < 0):
        indx = [0] + list(np.where(np.diff(intervals) < 0)[0] + 1) + [len(intervals)]
        new_dtm = []
        for i in range(len(indx) - 1):
            for d in dtm[indx[i]:indx[i + 1]]:
                new_dtm.append(d + timedelta(days=i))
        intervals = np.array([(d - new_dtm[0]).total_seconds() for d in new_dtm])

    if np.any(np.diff(intervals) < 0):
        raise ValueError(
            "format_time: Parsing of time-only data unsuccessful:\n"
            "    Non-sequential numeric time values found.")

    intervals = intervals + start
    return x, intervals
