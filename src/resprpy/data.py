"""Data utilities ported from respR: subsample, subset_data, select, sim_data.

- subsample: thin a dataset by taking every n-th row or a fixed number of rows
- subset_data: subset data between from/to by time, row or oxygen
- select: column selection (respR re-exports dplyr::select)
- sim_data: generate simulated respirometry data with a known linear section
"""
from __future__ import annotations

import numpy as np

from .calc import _by_val, _truncate_data


# ---------------------------------------------------------------------------
# subsample
# ---------------------------------------------------------------------------
def subsample(x, n=None, length_out=None, random_start=False, plot=True):
    """Take a subset of a dataset by every n-th row, or a fixed number of rows.

    Mirrors respR::subsample. Returns a new array (data is not modified).

    Parameters
    ----------
    x : array-like, shape (n, k) or 1-D
        The dataset to subsample.
    n : int, optional
        Take every n-th row starting from row 1 (or a random row if
        random_start=True).
    length_out : int, optional
        Subsample to this many rows (evenly spaced, endpoints included).
    random_start : bool
        With n: start at a random row 1..n (R: sample(1:n, 1)).
    plot : bool
        Accepted for respR API compatibility (no-op).
    """
    if n is None and length_out is None:
        raise ValueError("subsample: One of 'n' or 'length.out' is required.")
    if n is not None and length_out is not None:
        raise ValueError("subsample: Only one of 'n' or 'length.out' should be entered.")
    x = np.asarray(x)
    nrows = x.shape[0] if x.ndim == 2 else len(x)
    index = np.arange(1, nrows + 1)
    if n is not None and random_start:
        start = int(np.random.randint(1, n + 1))  # R: sample(1:n, 1)
    else:
        start = 1
    end = nrows
    if n is not None:
        idx = index[start - 1:end:n]
    else:
        idx = np.linspace(start, end, int(length_out)).astype(int)
    if x.ndim == 2:
        return x[idx - 1, :]
    return x[idx - 1]


# ---------------------------------------------------------------------------
# subset_data
# ---------------------------------------------------------------------------
def subset_data(x, from_=None, to=None, by="time", quiet=True):
    """Subset a dataset between from/to, by time, row or oxygen.

    Mirrors respR::subset_data (respR also accepts inspect objects; here x
    is a (n, 2) time/oxygen array, or an inspect-like object with .dataframe).
    """
    if hasattr(x, "dataframe"):
        dt = np.asarray(x.dataframe, dtype=float)
        is_inspect = True
    else:
        dt = np.asarray(x, dtype=float)
        if dt.ndim == 1:
            dt = dt.reshape(-1, 1)
        is_inspect = False
    by = _by_val(by, msg="subset_data")

    t = dt[:, 0]
    o = dt[:, 1]
    if from_ is None:
        from_ = np.nanmin(t) if by == "time" else (1 if by == "row" else o[0])
    if to is None:
        to = np.nanmax(t) if by == "time" else (dt.shape[0] if by == "row" else o[-1])
    from_ = float(from_)
    to = float(to)

    if by == "row":
        if from_ < 1 or from_ > dt.shape[0]:
            raise ValueError(f"subset_data: 'from' must be between 1 and {dt.shape[0]}")
        if to < from_ + 1:
            raise ValueError("subset_data: 'to' must be greater than 'from'")

    out = _truncate_data(dt, from_, to, by)
    if out.shape[0] == 0:
        import warnings
        warnings.warn("subset_data: subsetting criteria result in empty dataset!")
    if is_inspect:
        x.dataframe = out
        return x
    return out


# ---------------------------------------------------------------------------
# select (respR re-exports dplyr::select)
# ---------------------------------------------------------------------------
def select(x, *cols):
    """Select columns by name or 1-based position (dplyr::select equivalent).

    x may be a 2-D array or a dict-like object with named columns
    (keys -> columns). Returns a new array with the selected columns.
    """
    if hasattr(x, "columns") and hasattr(x, "values"):
        # pandas-like
        return x[list(cols)]
    if isinstance(x, dict):
        keys = list(x.keys())
        arr = np.column_stack([np.asarray(x[k]) for k in keys])
    else:
        arr = np.asarray(x)
        keys = None
    sel = []
    for c in cols:
        if isinstance(c, str):
            if keys is None:
                raise ValueError(f"select: column '{c}' requested but input has no column names")
            sel.append(keys.index(c))
        else:
            sel.append(int(c) - 1)
    if arr.ndim == 1:
        return arr
    return arr[:, sel]


# ---------------------------------------------------------------------------
# sim_data
# ---------------------------------------------------------------------------
def sim_data(len=300, type="default", sd=0.05, preview=True):
    """Generate simulated respirometry data with a known linear section.

    Mirrors respR::sim_data. Random (numpy RNG); the structure matches R:
    returns dict(df, coef, len_main, seg_index). 'type' is one of
    'default', 'corrupted' or 'segmented'.
    """
    n = int(np.floor(abs(np.random.normal(0.25 * len, 0.05 * len))))
    ampli = np.random.normal(0.8, 0.05)
    dip = ampli * np.cos(np.linspace(0, np.pi / 2, n)) - 1 + ampli
    ris = ampli * np.cos(np.linspace(np.pi, np.pi / 2, n)) - ampli
    ris = ris - np.max(ris)
    cor = np.cos(np.linspace(0, 2 * np.pi,
                             int(np.random.normal(0.25 * len, 0.05 * len))))
    len_x = len - n
    if type == "corrupted":
        len_z = int(np.floor(np.random.normal(0.25 * len_x, 0.02 * len_x)))
    if type == "segmented":
        len_z = int(np.floor(np.random.normal(0.35 * len_x, 0.02 * len_x)))
    if type != "default":
        len_x = len_x - len_z
    x = np.arange(1, len_x + 1)
    b_1 = np.random.normal(0, 0.02)
    y = b_1 * x
    if type == "corrupted":
        cor = np.cos(np.linspace(0, 2 * np.pi, len_z))
        poke = int(np.random.randint(1, len_x + 1))  # R sample(1:len_x, 1)
        y = np.insert(y, poke, cor + y[poke - 1] - 1)  # append after=poke
    elif type == "segmented":
        x_2 = np.arange(1, len_z + 1)
        b_s = b_1 * np.random.uniform(0.5, 0.6)
        y_2 = b_s * x_2
        y = np.concatenate([y, y_2 + y[-1]])
    if b_1 >= 0:
        joined = np.concatenate([ris, y])
    else:
        joined = np.concatenate([dip, y + dip[-1]])
    noise = np.random.normal(0, sd, np.size(joined))
    dat = joined + noise + 9
    df = np.column_stack([np.arange(0, np.size(dat), 1), dat])

    if type == "default":
        linseg = np.arange(n, n + len_x) + 1   # max(seq(1,n)) + seq(1,len_x)
    elif type == "corrupted":
        seg2 = np.arange(n + 1, poke + n + 1)
        seg3 = np.arange(max(seg2) + 1 + len_z, len + 1)
        linseg = seg2 if np.size(seg2) > np.size(seg3) else seg3
    else:  # segmented
        linseg = np.arange(n + 1, len_x + n + 1)

    lindf = df[linseg - 1, :]
    coefi = np.polyfit(lindf[:, 0], lindf[:, 1], 1)[0]
    return {"df": df, "coef": coefi, "len_main": len_x, "seg_index": linseg}
