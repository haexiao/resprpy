"""auto_rate: automatically determine linear regions of respirometry data.

Ported from respR 2.3.4. The 'linear' method computes rolling regressions,
builds a kernel density estimate of the slopes (R's density() with
bw="SJ-ste", adjust=0.95), finds density peaks and returns the most linear
regions. Other methods (max/min/highest/lowest/interval/rolling) select
rolling-regression windows directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.optimize import brentq

from .calc import _by_val, _truncate_data, calc_rate

K2 = 0.5
K4 = 0.375


# ---------------------------------------------------------------------------
# R stats::density() port (Gaussian kernel, bw="SJ-ste"), faithful to
# R 4.5.3: bw.SJ() pair-counts + C_bw_phi4/phi6, BinDist linear binning,
# 2n FFT cross-correlation, linear interpolation onto the output grid.
# ---------------------------------------------------------------------------
DELMAX = 1000.0


def _bw_nrd0(x):
    """R bw.nrd0: 0.9 * min(sd, IQR/1.34) * n^(-1/5)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 data points")
    hi = x.std(ddof=1)
    q = np.percentile(x, [25.0, 75.0])
    lo = min(hi, (q[1] - q[0]) / 1.34)
    if not lo:
        lo = hi or abs(x[0]) or 1.0
    return 0.9 * lo * n ** (-0.2)


def _bw_den_binned(xxx):
    """C_bw_den_binned: distance-bin counts from bin counts (R 4.5.3)."""
    nb = len(xxx)
    cnt = np.zeros(nb)
    for ii in range(nb):
        w = float(xxx[ii])
        cnt[0] += w * (w - 1.0)          # don't count distances to self
        for jj in range(ii):
            cnt[ii - jj] += w * xxx[jj]
    cnt[0] *= 0.5                        # same-bin pairs were double-counted
    return cnt


def _bw_den(nb, x):
    """C_bw_den: pairwise distance-bin counts for un-binned data."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite x in bandwidth calculation")
    rang = (x.max() - x.min()) * 1.01
    if rang == 0.0:
        raise ValueError("data are constant in bandwidth calculation")
    dd = rang / nb
    cnt = np.zeros(nb)
    for i in range(1, n):
        ii = int(x[i] / dd)              # C cast truncates toward zero
        for j in range(i):
            jj = int(x[j] / dd)
            cnt[abs(ii - jj)] += 1.0
    return dd, cnt


def _bw_pair_cnts(x, nb=1000):
    """R bw_pair_cnts: bin (n > nb/2) or not, returns (d, cnt)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n > nb / 2.0:
        r_min, r_max = x.min(), x.max()
        d = (r_max - r_min) * 1.01 / nb
        xx = np.trunc(np.abs(x) / d) * np.sign(x)
        xx = xx - np.min(xx) + 1
        xxx = np.bincount(xx.astype(int) - 1, minlength=nb)[:nb]
        return d, _bw_den_binned(xxx)
    return _bw_den(nb, x)


def _bw_phi4(n, d, cnt, h):
    """C_bw_phi4: E[phi^(4)] estimate via distance-bin counts."""
    nbin = len(cnt)
    s = 0.0
    for i in range(nbin):
        delta = i * d / h
        delta *= delta
        if delta >= DELMAX:
            break
        s += np.exp(-delta / 2.0) * (delta * delta - 6.0 * delta + 3.0) * cnt[i]
    s = 2.0 * s + n * 3.0                # add in diagonal
    return s / (n * (n - 1) * h ** 5.0) / np.sqrt(2.0 * np.pi)


def _bw_phi6(n, d, cnt, h):
    """C_bw_phi6: E[phi^(6)] estimate via distance-bin counts."""
    nbin = len(cnt)
    s = 0.0
    for i in range(nbin):
        delta = i * d / h
        delta *= delta
        if delta >= DELMAX:
            break
        s += np.exp(-delta / 2.0) * (
            delta ** 3 - 15.0 * delta * delta + 45.0 * delta - 15.0) * cnt[i]
    s = 2.0 * s - 15.0 * n               # add in diagonal
    return s / (n * (n - 1) * h ** 7.0) / np.sqrt(2.0 * np.pi)


def _zeroin2(ax, bx, fa, fb, f, tol, maxit=1000):
    """R_zeroin2 (R 4.5.3 src/library/stats/src/zeroin.c): Brent root
    finding with the exact same stopping rule (tol_act = 2eps|b| + tol/2),
    so results match R's uniroot() bit-for-bit along the same path."""
    eps = np.finfo(float).eps
    a, b = ax, bx
    c, fc = a, fa
    maxit = maxit + 1
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    while maxit:
        maxit -= 1
        prev_step = b - a
        if abs(fc) < abs(fb):
            # C: a = b; b = c; c = a;  -- sequential, so c gets the NEW a
            # (i.e. the old b), NOT the old a. Faithful translation required.
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa
        tol_act = 2 * eps * abs(b) + tol / 2
        new_step = (c - b) / 2
        if abs(new_step) <= tol_act or fb == 0.0:
            return b
        if abs(prev_step) >= tol_act and abs(fa) > abs(fb):
            cb = c - b
            if a == c:
                t1 = fb / fa
                p = cb * t1
                q = 1.0 - t1
            else:
                q = fa / fc
                t1 = fb / fc
                t2 = fb / fa
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)
            if p > 0:
                q = -q
            else:
                p = -p
            if p < 0.75 * cb * q - abs(tol_act * q) / 2 and p < abs(prev_step * q / 2):
                new_step = p / q
        if abs(new_step) < tol_act:
            new_step = tol_act if new_step > 0 else -tol_act
        a, fa = b, fb
        b += new_step
        fb = f(b)
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c, fc = a, fa
    return b


def bw_sj(x, nb=1000, method="ste"):
    """R bw.SJ (stats, R 4.5.3): Sheather-Jones bandwidth, 'ste' or 'dpi'."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        raise ValueError("need at least 2 data points")
    d, cnt = _bw_pair_cnts(x, nb)
    scale = min(x.std(ddof=1), (np.percentile(x, 75) - np.percentile(x, 25)) / 1.349)
    a = 1.24 * scale * n ** (-1 / 7)
    b = 1.23 * scale * n ** (-1 / 9)
    c1 = 1.0 / (2.0 * np.sqrt(np.pi) * n)
    SDh = lambda h: _bw_phi4(n, d, cnt, h)
    TDh = lambda h: _bw_phi6(n, d, cnt, h)
    TD = -TDh(b)
    if not np.isfinite(TD) or TD <= 0:
        raise ValueError("sample is too sparse to find TD")
    if method == "dpi":
        return (c1 / SDh((2.394 / (n * TD)) ** (1 / 7))) ** (1 / 5)
    hmax = 1.144 * scale * n ** (-1 / 5)
    alph2 = 1.357 * (SDh(a) / TD) ** (1 / 7)
    if not np.isfinite(alph2):
        raise ValueError("sample is too sparse to find alph2")
    fSD = lambda h: (c1 / SDh(alph2 * h ** (5 / 7))) ** (1 / 5) - h
    lower, upper = 0.1 * hmax, hmax
    itry = 1
    while fSD(lower) * fSD(upper) > 0:
        if itry > 99:
            raise ValueError("no solution in the specified range of bandwidths")
        if itry % 2:
            upper *= 1.2
        else:
            lower /= 1.2
        itry += 1
    return _zeroin2(lower, upper, fSD(lower), fSD(upper), fSD, 0.1 * lower)


def _bin_dist(x, weights, lo, up, n):
    """C_BinDist (R 4.5.3 massdist.c): linear-assignment binning."""
    xdelta = (up - lo) / (n - 1)
    y = np.zeros(2 * n)
    ixmax = n - 2
    for xi, wi in zip(x, weights):
        if not np.isfinite(xi):
            continue
        xpos = (xi - lo) / xdelta
        ix = int(np.floor(xpos))
        fx = xpos - ix
        if 0 <= ix <= ixmax:
            y[ix] += (1 - fx) * wi
            y[ix + 1] += fx * wi
        elif ix == -1:
            y[0] += fx * wi
        elif ix == ixmax + 1:
            y[ix] += (1 - fx) * wi
    return y


def r_density(x, bw=None, adjust=0.95, n=512, cut=3.0, ext=4.0):
    """R stats::density(x, bw='SJ-ste', adjust=..., kernel='gaussian', n=512,
    cut=3, ext=4). Returns (x, y, bw) exactly as R."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    nx = len(x)
    n_user = n
    n = max(n, 512)
    if n > 512:
        n = 2 ** int(np.ceil(np.log2(n)))
    if bw is None:
        bw = bw_sj(x, method="ste")
    bw = float(adjust) * bw
    if bw <= 0:
        raise ValueError("'bw' is not positive.")
    from_ = x.min() - cut * bw
    to = x.max() + cut * bw
    lo = from_ - ext * bw
    up = to + ext * bw
    weights = np.full(nx, 1.0 / nx)
    y = _bin_dist(x, weights, lo, up, n) * 1.0   # totMass = 1

    kmax = (2 * n - 1) / (n - 1) * (up - lo)
    kords = np.linspace(0.0, kmax, 2 * n)
    kords[n + 1:] = -kords[n - 1:0:-1]
    # gaussian kernel
    kords = np.exp(-0.5 * (kords / bw) ** 2) / (bw * np.sqrt(2.0 * np.pi))

    conv = np.fft.ifft(np.fft.fft(y) * np.conj(np.fft.fft(kords))).real
    # R: fft(..., inverse=TRUE) is NOT scaled by 1/N (unlike numpy's ifft),
    # then divided by length(y) = 2n -- so the two factors cancel exactly:
    # R y = Re(ifft_numpy(conv))[1:n] / (2n) * (2n) = ifft_numpy[:n]
    conv = np.maximum(0.0, conv[:n])
    xords = np.linspace(lo, up, n)
    xg = np.linspace(from_, to, n_user)
    yg = np.interp(xg, xords, conv)
    return xg, yg, bw


# ---------------------------------------------------------------------------
# rolling regressions
# ---------------------------------------------------------------------------
def _ols(x, y):
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return coef[0], coef[1], r2


def rolling_reg_row(df, width):
    """Rolling regressions over windows of `width` rows (respR
    rolling_reg_row). Returns array with columns row, endrow, time, endtime,
    oxy, endoxy, intercept_b0, slope_b1, rsq."""
    x = df[:, 0]
    y = df[:, 1]
    n = len(x)
    out = []
    for i in range(0, n - width + 1):
        b0, b1, r2 = _ols(x[i:i + width], y[i:i + width])
        out.append([i + 1, i + width, x[i], x[i + width - 1],
                    y[i], y[i + width - 1], b0, b1, r2])
    return np.array(out)


def rolling_reg_time(df, width):
    """Rolling regressions over time windows of span `width` (respR
    rolling_reg_time; b1 = cov/var). Columns same as rolling_reg_row."""
    x = df[:, 0]
    y = df[:, 1]
    n = len(x)
    out = []
    for i in range(n):
        start_t = x[i] - width
        mask = (x >= start_t) & (x <= x[i])
        if np.sum(mask) < 2:
            continue
        xs, ys = x[mask], y[mask]
        b1 = np.cov(xs, ys, ddof=1)[0, 1] / np.var(xs, ddof=1)
        b0 = np.mean(ys) - b1 * np.mean(xs)
        with np.errstate(invalid="ignore"):
            r2 = float(np.corrcoef(xs, ys)[0, 1] ** 2)
        out.append([x[i], x[i], b0, b1, r2])
    out = np.array(out)
    # R: results <- results[time >= df[[1]][1]]  (already satisfied), then
    # row = seq_len(.N); endrow from matching endtime in original data
    t_full = x
    endrows = []
    for et in out[:, 1]:
        idx = np.where(t_full == et)[0]
        endrows.append(int(idx[-1]) + 1 if len(idx) else 1)
    endrows = np.sort(np.unique(endrows))
    n2 = out.shape[0]
    rows = np.arange(1, n2 + 1)
    oxy = y[rows - 1]
    endoxy = y[endrows - 1]
    return np.column_stack([rows, endrows, out[:, 0], out[:, 1],
                            oxy, endoxy, out[:, 2], out[:, 3], out[:, 4]])


# ---------------------------------------------------------------------------
# calc_win / kernel_method
# ---------------------------------------------------------------------------
def calc_win(dt, width, by, msg):
    if by == "row":
        if width is None:
            width = 0.2
        if width > 1 and width > dt.shape[0]:
            raise ValueError(f"{msg}: 'width' exceeds length of dataset")
        if width > 1 and not float(width).is_integer():
            raise ValueError(f"{msg}: 'width' should be an integer of 2 or higher")
        if width == 1:
            raise ValueError(f"{msg}: 'width' cannot be 1 row")
        if width == dt.shape[0]:
            raise ValueError(f"{msg}: 'width' cannot be the total number of rows in the input data")
        if width < 1:
            win = int(np.floor(width * dt.shape[0]))
        else:
            win = int(width)
    else:  # time
        t = dt[:, 0]
        trange = np.nanmax(t) - np.nanmin(t)
        if width is None:
            width = 0.2 * trange
        if width >= trange:
            raise ValueError(f"{msg}: 'width' cannot exceed or equal total time data range")
        win = width
    return win


def _rank(x):
    """R rank() with default ties.method='average'."""
    return stats.rankdata(x)


def kernel_method(dt, width, top_only=False):
    """Rolling regression + KDE of slopes + peak detection (respR
    kernel_method). Returns dict(rollreg, subsets, peaks, density)."""
    rollreg = rolling_reg_row(dt, width)
    slopes = rollreg[:, 7]
    grid, y, bw = r_density(slopes, adjust=0.95)
    # peaks: local maxima of the density (R: which(diff(sign(diff(y))) == -2) + 1)
    signs = np.sign(np.diff(y))
    peak_pos = np.where(np.diff(signs) == -2)[0] + 1  # 0-based positions in y
    peak_idx = peak_pos + 1  # 1-based index into density grid
    index = np.column_stack([peak_idx, grid[peak_pos], y[peak_pos]])
    # order by density (R: order(-rank(density)))
    if index.shape[0] == 0:
        return {"rollreg": rollreg, "subsets": [], "peaks": np.empty((0, 3)),
                "density": (grid, y, bw)}
    order = np.argsort(-_rank(index[:, 2]), kind="stable")
    if top_only:
        order = order[:1]
    ranked = index[order]

    # fragments: rollreg rows whose slope is within bw*0.95 of each peak
    tol = bw * 0.95
    frags = []
    keep = []
    for k, pk in enumerate(ranked[:, 1]):
        mask = (rollreg[:, 7] >= pk - tol) & (rollreg[:, 7] <= pk + tol)
        fr = rollreg[mask]
        if fr.shape[0] == 0:
            continue
        # split into contiguous runs (gap in 'row' > width)
        row_gaps = np.diff(fr[:, 0]) > width
        splits = np.where(row_gaps)[0] + 1
        runs = np.split(fr, splits)
        best = max(runs, key=lambda r: r.shape[0])
        frags.append(best)
        keep.append(k)
    ranked = ranked[keep]

    subsets = []
    for fr in frags:
        r0 = int(np.min(fr[:, 0]))       # row
        r1 = int(np.max(fr[:, 1]))       # endrow (column 1 of the roll table)
        subsets.append(_truncate_data(dt, r0, r1, "row"))
    return {"rollreg": rollreg, "subsets": subsets, "peaks": ranked,
            "density": (grid, y, bw)}


# ---------------------------------------------------------------------------
# auto_rate methods
# ---------------------------------------------------------------------------
def _auto_rate_simple(dt, width, by, mode):
    if by == "row":
        rollreg = rolling_reg_row(dt, width)
    else:
        rollreg = rolling_reg_time(dt, width)
    slopes = rollreg[:, 7]
    if mode == "max":
        results = rollreg[np.argsort(-_rank(slopes), kind="stable")]
    elif mode == "min":
        results = rollreg[np.argsort(_rank(slopes), kind="stable")]
    elif mode == "highest":
        if np.any(slopes > 0) and np.any(slopes < 0):
            raise ValueError("auto_rate: Analysis produces both negative and positive rates. "
                             "The 'highest' method is intended to order by the lowest *absolute* rate "
                             "amongst rates all having the same sign.")
        results = rollreg[np.argsort(-_rank(np.abs(slopes)), kind="stable")]
    elif mode == "lowest":
        if np.any(slopes > 0) and np.any(slopes < 0):
            raise ValueError("auto_rate: Analysis produces both negative and positive rates. "
                             "The 'lowest' method is intended to order by the lowest *absolute* rate "
                             "amongst rates all having the same sign.")
        results = rollreg[np.argsort(_rank(np.abs(slopes)), kind="stable")]
    elif mode == "interval":
        if by == "row":
            sequence = np.arange(width, dt.shape[0] + 1, width)
            results = rollreg[sequence - width]
        else:
            sequence = np.arange(np.min(dt[:, 0]), np.max(dt[:, 0]) + width, width)
            mask = np.isin(rollreg[:, 2], sequence)
            results = rollreg[mask]
    else:  # rolling
        results = rollreg
    return rollreg, results


def _method_linear(dt, width, by):
    kde = kernel_method(dt, width)
    subsets = kde["subsets"]
    if len(subsets):
        # verify step: re-run kernel method on each subset with 0.85*width
        testwin = int(np.floor(width * 0.85))
        new_subsets = []
        for s in subsets:
            k2 = kernel_method(s, testwin, top_only=True)
            new_subsets.extend(k2["subsets"])
        subsets = new_subsets
    rates = []
    for s in subsets:
        # respR runs calc_rate on the FULL dataset with from/to from the
        # subset's first/last time, so row/endrow refer to the original data
        cr = calc_rate(dt, from_=s[0, 0], to=s[-1, 0], by="time", plot=False)
        sm = cr.summary
        rates.append([sm["intercept_b0"][0], sm["slope_b1"][0], sm["rsq"][0],
                      sm["row"][0], sm["endrow"][0], sm["time"][0],
                      sm["endtime"][0], sm["oxy"][0], sm["endoxy"][0]])
    results = np.array(rates) if rates else np.empty((0, 9))
    return kde, results


# ---------------------------------------------------------------------------
# auto_rate
# ---------------------------------------------------------------------------
@dataclass
class AutoRate:
    """Result of auto_rate(). Mirrors the respR 'auto_rate' object."""
    call: object = None
    inputs: dict = field(default_factory=dict)
    dataframe: np.ndarray = None
    width: object = None
    by: str = None
    method: str = None
    roll: np.ndarray = None
    density: object = None
    peaks: np.ndarray = None
    bandwidth: float = None
    metadata: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    rate: np.ndarray = None

    @property
    def summary_table(self):
        cols = ["rep", "rank", "intercept_b0", "slope_b1", "rsq", "density",
                "row", "endrow", "time", "endtime", "oxy", "endoxy", "rate"]
        return np.column_stack([self.summary[k] for k in cols])

    @property
    def summary_columns(self):
        return ["rep", "rank", "intercept_b0", "slope_b1", "rsq", "density",
                "row", "endrow", "time", "endtime", "oxy", "endoxy", "rate"]

    def __repr__(self):
        n = np.size(self.rate)
        return f"<resprpy.AutoRate n={n} method={self.method!r}>"


def auto_rate(x, method="linear", width=None, by="row", plot=True, **kwargs):
    """Automatically determine the most linear regions of respirometry data.

    Parameters
    ----------
    x : array-like, shape (n, 2)
        time and oxygen columns.
    method : str
        'linear' (default; KDE-based), 'max', 'min', 'highest', 'lowest',
        'interval', 'rolling', 'maximum', 'minimum'.
    width : float or int
        Window width as a fraction (0..1) or number of rows (by='row'), or
        time span (by='time'). Default 0.2.
    by : str
        'row' (default) or 'time'.
    plot : bool
        Accepted for respR API compatibility (not implemented yet).

    Returns
    -------
    AutoRate object with .summary (dict), .summary_table, .rate, .roll, .peaks.
    """
    inputs = dict(x=x, method=method, width=width, by=by, plot=plot)
    by = _by_val(by, which=("t", "r"), msg="auto_rate")
    if method not in ("linear", "max", "min", "interval", "rolling",
                      "highest", "lowest", "maximum", "minimum"):
        raise ValueError("auto_rate: 'method' input not recognised")
    dt = np.asarray(x, dtype=float)
    if dt.ndim == 1:
        dt = dt.reshape(-1, 1)
    if dt.shape[1] > 2:
        dt = dt[:, :2]
    if dt.shape[0] == 1:
        raise ValueError("auto_rate: Input data contains only 1 row. Please check inputs.")
    if by == "row":
        dt = dt[np.argsort(dt[:, 0], kind="stable")] if not np.all(np.diff(dt[:, 0]) >= 0) else dt

    win = calc_win(dt, width, by, "auto_rate")

    if method == "maximum":
        rollreg, results = _auto_rate_simple(dt, win, by, "max")
        density, peaks, bandwidth = None, None, None
    elif method == "minimum":
        rollreg, results = _auto_rate_simple(dt, win, by, "min")
        density, peaks, bandwidth = None, None, None
    elif method == "max" or method == "min":
        import warnings
        warnings.warn("auto_rate: The 'min' and 'max' methods have been deprecated...")
        # respR quirk (kept for compatibility): method="max" actually runs the
        # min ordering and vice versa (auto_rate.R: max -> auto_rate_min,
        # min -> auto_rate_max). Faithfully reproduced.
        mode = "min" if method == "max" else "max"
        rollreg, results = _auto_rate_simple(dt, win, by, mode)
        density, peaks, bandwidth = None, None, None
    elif method == "highest":
        rollreg, results = _auto_rate_simple(dt, win, by, "highest")
        density, peaks, bandwidth = None, None, None
    elif method == "lowest":
        rollreg, results = _auto_rate_simple(dt, win, by, "lowest")
        density, peaks, bandwidth = None, None, None
    elif method == "interval":
        rollreg, results = _auto_rate_simple(dt, win, by, "interval")
        density, peaks, bandwidth = None, None, None
    elif method == "rolling":
        rollreg, results = _auto_rate_simple(dt, win, by, "rolling")
        density, peaks, bandwidth = None, None, None
    else:  # linear
        kde, results = _method_linear(dt, win, by)
        rollreg = kde["rollreg"]
        density = kde["density"]
        peaks = kde["peaks"]
        bandwidth = density[2]

    n_res = results.shape[0]
    if method == "linear":
        # summary: rep, rank, intercept_b0, slope_b1, rsq, density, row,
        # endrow, time, endtime, oxy, endoxy, rate
        dens_vals = peaks[:, 2] if peaks.shape[0] == n_res else np.full(n_res, np.nan)
        summary = {
            "rep": np.full(n_res, np.nan),
            "rank": np.arange(1, n_res + 1),
            "intercept_b0": results[:, 0] if n_res else np.array([]),
            "slope_b1": results[:, 1] if n_res else np.array([]),
            "rsq": results[:, 2] if n_res else np.array([]),
            "density": dens_vals,
            "row": results[:, 3] if n_res else np.array([]),
            "endrow": results[:, 4] if n_res else np.array([]),
            "time": results[:, 5] if n_res else np.array([]),
            "endtime": results[:, 6] if n_res else np.array([]),
            "oxy": results[:, 7] if n_res else np.array([]),
            "endoxy": results[:, 8] if n_res else np.array([]),
            "rate": results[:, 1] if n_res else np.array([]),
        }
        # remove duplicated rows (R: which(!duplicated(summary[,1:7])) where
        # summary = results + density column; so the key is the first 7 cols
        # of results: intercept_b0, slope_b1, rsq, row, endrow, time, endtime)
        if n_res:
            key = results[:, :7]
            keep = _not_duplicated_rows(key)
            for k in summary:
                summary[k] = summary[k][keep]
            if peaks.shape[0]:
                peaks = peaks[keep]
            n_res = int(np.sum(keep))
            summary["rank"] = np.arange(1, n_res + 1)
    else:
        summary = {
            "rep": np.full(n_res, np.nan),
            "rank": np.arange(1, n_res + 1),
            "intercept_b0": results[:, 6],
            "slope_b1": results[:, 7],
            "rsq": results[:, 8],
            "density": np.full(n_res, np.nan),
            "row": results[:, 0],
            "endrow": results[:, 1],
            "time": results[:, 2],
            "endtime": results[:, 3],
            "oxy": results[:, 4],
            "endoxy": results[:, 5],
            "rate": results[:, 7],
        }

    rate = np.array(summary["slope_b1"])
    metadata = {"width": win, "by": by, "method": method,
                "total_regs": rollreg.shape[0] if rollreg.size else 0}
    if method == "linear":
        metadata["total_peaks"] = peaks.shape[0] if peaks.size else 0
        metadata["kde_bw"] = bandwidth
    out = AutoRate(call=None, inputs=inputs, dataframe=dt, width=win, by=by,
                   method=method, roll=rollreg, density=density, peaks=peaks,
                   bandwidth=bandwidth, metadata=metadata, summary=summary,
                   rate=rate)
    return out


def _not_duplicated_rows(arr):
    """First occurrence of each unique row (R duplicated() semantics, exact
    comparison like R's data.table)."""
    seen = set()
    keep = []
    for r in arr:
        key = tuple(float(v) for v in r)
        if key not in seen:
            seen.add(key)
            keep.append(True)
        else:
            keep.append(False)
    return np.array(keep, dtype=bool)
