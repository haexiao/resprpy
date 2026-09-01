"""matplotlib 复刻 respR 的 plot 方法 (plot.inspect / plot.calc_rate /
plot.auto_rate / plot.oxy_crit).

配色与布局对齐 R 的 util_plots.R: r1=black 主散点, r2=goldenrod 高亮,
c1=lightgreen (oxy_crit), 时间轴蓝色, 行号轴红色, 灰色网格.
图形内容与 R 对应面板一致 (数据、拟合线、断点/边界线、图例、方程文本);
渲染风格为 matplotlib 近似, 不追求逐像素一致.

用法::
    import resprpy as p
    res = p.calc_rate(df, method="linear", by="row", width=10)
    p.plot_calc_rate(res)          # 显示窗口
    p.plot_calc_rate(res, save="out.png")
"""
from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "resprpy plotting requires matplotlib: "
        "pip install 'resprpy[plot]'") from e

# ---- R util_plots.R colours ----
_R1 = "black"        # primary points
_R2 = "goldenrod"    # highlighted region points
_R3 = "0.85"         # plot background / grid
_C1 = "lightgreen"   # oxy_crit points
_AX_TIME = "blue"
_AX_ROW = "red"
_LINE_BSR = ["red", "orange"]
_LINE_SEG = ["steelblue"]
_LINE_TYPES_BSR = ["--", "-."]
_LINE_TYPES_SEG = [":"]
_LINE_WT = 3


def _extendrange(v, f=0.05):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (0.0, 1.0)
    lo, hi = np.min(v), np.max(v)
    rng = hi - lo
    if rng <= 0:
        rng = 1.0
    return (lo - f * rng, hi + f * rng)


def _grid(ax):
    ax.grid(True, lw=0.7, color=_R3)


def _row_axis(ax, rownums, vals):
    """R 的红色行号顶轴 (axis(side=3, col='red'))."""
    axr = ax.twiny()
    axr.scatter(rownums, vals, s=0)
    axr.set_xlim(rownums[0], rownums[-1])
    axr.tick_params(axis="x", colors=_AX_ROW, labelsize=8)
    return axr


def _time_axis(ax):
    ax.tick_params(axis="x", colors=_AX_TIME, labelsize=8)


def _axis_legend(ax, axr, yv, rownums, legend):
    """R 的纯文本图例: Time (蓝, 左下) / Row (红, 右上)."""
    if not legend:
        return
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    ax.text(xlo + 0.01 * (xhi - xlo), ylo + 0.02 * (yhi - ylo), "Time",
            color=_AX_TIME, fontsize=8,
            bbox=dict(facecolor="0.9", edgecolor="none", alpha=0.8))
    axr.text(xhi - 0.02 * (xhi - xlo), yhi - 0.06 * (yhi - ylo), "Row",
             color=_AX_ROW, fontsize=8,
             bbox=dict(facecolor="0.9", edgecolor="none", alpha=0.8))


def _lm_line(ax, xs, ys, **kw):
    b = np.polyfit(xs, ys, 1)
    xx = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(xx, np.polyval(b, xx), **kw)


# ======================================================================
# plot.inspect
# ======================================================================
def plot_inspect(x, width=None, pos=None, legend=True, rate_rev=True,
                 save=None, **kw):
    """plot.inspect: 每个氧列一个图 (时间序列 + 行号顶轴, 可选滑动回归线).

    x: inspect 对象 (dict, 含 dataframe/inputs/checks).
    """
    df = np.asarray(x["dataframe"], dtype=float)
    nres = df.shape[1] - 1
    if pos is None:
        pos = list(range(1, nres + 1))
    if any(p > nres for p in pos):
        raise ValueError(f"plot.inspect: Invalid 'pos' rank: only {nres} "
                         "oxygen columns found.")
    if width is None:
        width = x.get("inputs", {}).get("width", 0.1)

    figs = []
    for p in pos:
        yv = df[:, p]  # 0-based: col 0 = time, col p = oxygen p
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.scatter(df[:, 0], yv, s=6, c=_R1, zorder=2)
        _grid(ax)
        ax.set_ylim(_extendrange(yv))
        _time_axis(ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        # 顶部行号轴 (R: axis 3)
        rownums = np.arange(1, len(yv) + 1)
        axr = _row_axis(ax, rownums, yv)
        _axis_legend(ax, axr, yv, rownums, legend)
        # add.data (滑动回归线, R: rollreg2.p)
        win = int(np.floor(width * len(yv)))
        if win >= 2 and len(yv) >= win:
            rates = _roll_slope(df[:, 0], yv, win)
            tmid = (df[:-1, 0][win - 1:] + df[win - 1:, 0]) / 2.0 if False else None
            axr2 = ax.twinx()
            axr2.plot(np.arange(win, len(yv) + 1), rates, c=_R2, lw=1)
            axr2.set_ylabel("")
            axr2.tick_params(labelsize=8)
        ax.set_title(f"Inspect: Oxygen Column {p} (Time vs Oxygen)",
                     fontsize=10, fontweight="bold")
        fig.tight_layout()
        figs.append(fig)
        if save:
            fig.savefig(save if len(pos) == 1 else f"{save}_{p}.png")
    if not save:
        plt.show()
    return figs


def _roll_slope(t, o, width):
    n = len(t)
    out = np.full(n, np.nan)
    for i in range(width - 1, n):
        tt = t[i - width + 1:i + 1]
        oo = o[i - width + 1:i + 1]
        X = np.column_stack([np.ones_like(tt), tt])
        out[i] = np.linalg.solve(X.T @ X, X.T @ oo)[1]
    return out[width - 1:]


# ======================================================================
# plot.calc_rate
# ======================================================================
def _std_residuals(X, resid, n, p):
    """R rstandard() 近似: r_i / (sigma * sqrt(1 - h_ii))."""
    XtX_inv = np.linalg.inv(X.T @ X)
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    sigma = np.sqrt(np.sum(resid ** 2) / (n - p))
    with np.errstate(divide="ignore", invalid="ignore"):
        return resid / (sigma * np.sqrt(1.0 - h))


def _qq_vals(vals):
    """R qqnorm 绘图位置 (i - 0.5)/n + qqline 用 Q1/Q3 拟合."""
    from scipy import stats
    n = len(vals)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = stats.norm.ppf(probs)
    qq_x = np.sort(vals)
    q1, q3 = np.quantile(vals, [0.25, 0.75])
    t1, t3 = stats.norm.ppf([0.25, 0.75])
    slope = (q3 - q1) / (t3 - t1)
    intercept = q1 - slope * t1
    return theo, qq_x, slope, intercept


def plot_calc_rate(x, pos=1, panel=None, legend=True, save=None, **kw):
    """plot.calc_rate: 2x2 面板 (Full Timeseries / Close-up / Residuals / QQ).

    x: CalcRate 对象 (dataframe, subsets, summary, rate).
    panel: None = 全部 1..4, 或单值/列表.
    """
    nres = len(x.rate)
    if pos is None:
        pos = 1
    if pos > nres or pos < 1:
        raise ValueError(f"plot.calc_rate: Invalid 'pos' rank: only {nres} "
                         "rates found.")
    if panel is None:
        panel = [1, 2, 3, 4]
    elif isinstance(panel, int):
        panel = [panel]
    if any(p > 4 for p in panel):
        raise ValueError("plot.calc_rate: 'panel' input should be 1 to 4 or "
                         "'NULL' for all.")

    df = np.asarray(x.dataframe, dtype=float)
    sdf = np.asarray(x.subsets[pos - 1], dtype=float)
    summary = x.summary
    row0 = int(summary["row"][pos - 1])
    endrow = int(summary["endrow"][pos - 1])
    rownums = np.arange(row0, endrow + 1)
    b = np.polyfit(sdf[:, 0], sdf[:, 1], 1)
    fit_vals = np.polyval(b, sdf[:, 0])
    resid = sdf[:, 1] - fit_vals
    n = len(sdf)
    Xs = np.column_stack([np.ones(n), sdf[:, 0]])
    rsq = float(summary["rsq"][pos - 1])

    npanels = len(panel)
    fig, axes = plt.subplots(
        (npanels + 1) // 2 if npanels > 1 else 1,
        2 if npanels > 1 else 1,
        figsize=(9, 6 if npanels > 1 else 3))
    axes = np.atleast_1d(axes).ravel()

    pi = 0
    if 1 in panel:  # full timeseries
        ax = axes[pi]; pi += 1
        ax.scatter(df[:, 0], df[:, 1], s=6, c=_R1, zorder=2)
        ax.scatter(sdf[:, 0], sdf[:, 1], s=8, c=_R2, zorder=3)
        _lm_line(ax, sdf[:, 0], sdf[:, 1], ls=":", lw=1.2, c=_R2)
        _grid(ax)
        _time_axis(ax)
        axr = _row_axis(ax, np.arange(1, len(df) + 1), df[:, 1])
        ax.set_title("Full Timeseries", fontsize=10, fontweight="bold")
        _axis_legend(ax, axr, df[:, 1],
                     np.arange(1, len(df) + 1), legend)
    if 2 in panel:  # close-up region
        ax = axes[pi]; pi += 1
        ax.scatter(sdf[:, 0], sdf[:, 1], s=8, c=_R2)
        _lm_line(ax, sdf[:, 0], sdf[:, 1], ls="--", lw=1.5, c=_R2)
        _grid(ax)
        _time_axis(ax)
        _row_axis(ax, rownums, sdf[:, 1])
        cf = b  # polyfit -> [slope, intercept]; R coef = (intercept, slope)
        eq = (f"y = {cf[1]:.3g} {'+' if cf[0] >= 0 else '-'} "
              f"{abs(cf[0]):.3g} x")
        ax.set_title("Close-up Region", fontsize=10, fontweight="bold")
        ax.text(0.05, 0.90, eq, transform=ax.transAxes, fontsize=8)
        ax.text(0.05, 0.80, f"r2 = {rsq:.3g}", transform=ax.transAxes,
                fontsize=8)
    if 3 in panel:  # residuals vs fitted
        ax = axes[pi]; pi += 1
        ax.scatter(fit_vals, resid, s=8, c=_R2)
        ax.axhline(0, ls=":", lw=1.5, c="black")
        _grid(ax)
        m = np.max(np.abs(resid)) if len(resid) else 1
        ax.set_ylim(m, -m)  # R: ylim = c(max, -max) -> reversed axis
        ax.set_title("Std. Residuals \nvs Fitted Values", fontsize=10,
                     fontweight="bold")
    if 4 in panel:  # q-q plot
        ax = axes[pi]; pi += 1
        if n > 2:
            vals = _std_residuals(Xs, resid, n, 2)
        else:
            vals = fit_vals
        theo, qq, slope, intercept = _qq_vals(vals)
        ax.scatter(theo, qq, s=8, c=_R2)
        tt = np.array([theo[0], theo[-1]])
        ax.plot(tt, intercept + slope * tt, ls=":", lw=1.5, c="black")
        _grid(ax)
        ax.set_title("Theoretical Q. \nvs Std. Residuals", fontsize=10,
                     fontweight="bold")

    fig.suptitle(f"calc.rate: Rank {pos} of {nres} Total Rates",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save:
        fig.savefig(save)
    else:
        plt.show()
    return fig


# ======================================================================
# plot.auto_rate
# ======================================================================
def plot_auto_rate(x, pos=1, panel=False, legend=True, rate_rev=True,
                   save=None, **kw):
    """plot.auto_rate: 按 method 布局 (multi/sub/rollreg/residual/qq/density).

    x: AutoRate 对象.
    """
    nres = len(x.rate)
    if nres == 0:
        print("plot.auto_rate: Nothing to plot! No rates found.")
        return None
    if pos is None:
        pos = 1
    if pos > nres:
        raise ValueError(f"plot.auto_rate: Invalid 'pos' rank: only {nres} "
                         "rates found.")

    df = np.asarray(x.dataframe, dtype=float)
    summary = x.summary
    start = int(summary["row"][pos - 1])
    end = int(summary["endrow"][pos - 1])
    rownums_sub = np.arange(start, end + 1)
    rownums = np.arange(1, len(df) + 1)
    sdt = df[start - 1:end]
    roll = np.asarray(x.roll, dtype=float)
    rolldt_x = (roll[:, 3] + roll[:, 2]) / 2.0  # (endtime+time)/2
    rolldt_y = roll[:, 7]                        # slope_b1
    rate = float(summary["slope_b1"][pos - 1])
    rsq = float(summary["rsq"][pos - 1])
    b = np.polyfit(sdt[:, 0], sdt[:, 1], 1)
    fit_vals = np.polyval(b, sdt[:, 0])
    resid = sdt[:, 1] - fit_vals
    n = len(sdt)
    Xs = np.column_stack([np.ones(n), sdt[:, 0]])

    method = getattr(x, "method", "linear") or "linear"
    if method in ("max", "min", "maximum", "minimum", "highest", "lowest",
                  "rolling", "interval"):
        # layout mat: 2x6 -> 上排 1,1,1,2,2,2 下排 3,3,4,4,5,5
        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(2, 6)
        ax1 = fig.add_subplot(gs[0, 0:3])
        ax2 = fig.add_subplot(gs[0, 3:6])
        ax3 = fig.add_subplot(gs[1, 0:2])
        ax4 = fig.add_subplot(gs[1, 2:4])
        ax5 = fig.add_subplot(gs[1, 4:6])
    else:  # linear -> 2x3
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        axes = axes.ravel()
        ax1, ax2, ax3, ax4, ax5 = axes[:5]
    # panel 1: full timeseries
    ax1.scatter(df[:, 0], df[:, 1], s=6, c=_R1, zorder=2)
    ax1.scatter(sdt[:, 0], sdt[:, 1], s=8, c=_R2, zorder=3)
    _lm_line(ax1, sdt[:, 0], sdt[:, 1], ls=":", lw=1.2, c=_R2)
    _grid(ax1)
    _time_axis(ax1)
    axr1 = _row_axis(ax1, rownums, df[:, 1])
    ax1.set_title("Full Timeseries", fontsize=10, fontweight="bold")
    _axis_legend(ax1, axr1, df[:, 1], rownums, legend)
    # panel 2: close-up
    ax2.scatter(sdt[:, 0], sdt[:, 1], s=8, c=_R2)
    _lm_line(ax2, sdt[:, 0], sdt[:, 1], ls="--", lw=1.5, c=_R2)
    _grid(ax2)
    _time_axis(ax2)
    _row_axis(ax2, rownums_sub, sdt[:, 1])
    cf = b  # polyfit -> [slope, intercept]; R coef = (intercept, slope)
    eq = (f"y = {cf[1]:.3g} {'+' if cf[0] >= 0 else '-'} "
          f"{abs(cf[0]):.3g} x")
    ax2.set_title("Close-up Region", fontsize=10, fontweight="bold")
    ax2.text(0.05, 0.90, eq, transform=ax2.transAxes, fontsize=8)
    ax2.text(0.05, 0.80, f"r2 = {rsq:.3g}", transform=ax2.transAxes,
             fontsize=8)
    # panel 3: rolling rate
    ylim = _extendrange(rolldt_y)
    if rate_rev:
        ylim = (ylim[1], ylim[0])
    ax3.scatter(rolldt_x, rolldt_y, s=6, c=_R2)
    ax3.set_ylim(ylim)
    ax3.axhline(rate, ls="--", lw=1, c="black")
    _grid(ax3)
    _time_axis(ax3)
    _row_axis(ax3, np.arange(1, len(rolldt_x) + 1), rolldt_y)
    ax3.set_title("Rolling Rate", fontsize=10, fontweight="bold")
    # panel 4: residuals
    ax4.scatter(fit_vals, resid, s=8, c=_R2)
    ax4.axhline(0, ls=":", lw=1.5, c="black")
    _grid(ax4)
    m = np.max(np.abs(resid)) if len(resid) else 1
    ax4.set_ylim(m, -m)
    ax4.set_title("Std. Residuals \nvs Fitted Values", fontsize=10,
                  fontweight="bold")
    # panel 5: q-q
    if n > 2:
        vals = _std_residuals(Xs, resid, n, 2)
    else:
        vals = fit_vals
    theo, qq, slope_q, intercept_q = _qq_vals(vals)
    ax5.scatter(theo, qq, s=8, c=_R2)
    tt = np.array([theo[0], theo[-1]])
    ax5.plot(tt, intercept_q + slope_q * tt, ls=":", lw=1.5, c="black")
    _grid(ax5)
    ax5.set_title("Theoretical Q. \nvs Std. Residuals", fontsize=10,
                  fontweight="bold")
    # panel 6 (linear only): density
    if method not in ("max", "min", "maximum", "minimum", "highest",
                      "lowest", "rolling", "interval"):
        dens = x.density
        ax6 = axes[5]
        if isinstance(dens, tuple) and len(dens) >= 2:
            dx, dy = dens[0], dens[1]
            ax6.plot(dx, dy, c=_R2)
            ax6.fill_between(dx, dy, color=_R2, alpha=0.5)
            peaks = np.asarray(x.peaks, dtype=float)
            if peaks.ndim == 2 and peaks.shape[0] >= pos:
                ax6.axvline(peaks[pos - 1, 1], ls="--", c="black")
            ax6.set_title("Density of Rolling beta1", fontsize=10,
                          fontweight="bold")
    fig.suptitle(f"auto.rate: Rank {pos} of {nres} Total Rates",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save:
        fig.savefig(save)
    else:
        plt.show()
    return fig


# ======================================================================
# plot.oxy_crit
# ======================================================================
def plot_oxy_crit(x, legend=True, panel=None, rate_rev=True, save=None,
                  **kw):
    """plot.oxy_crit: 面板 1 时间序列+断点横线, 面板 2 Rate~Oxygen 结果.

    x: oxy_crit 对象 (dict: dataframe, df_rate_oxygen, method, crit,
    summary, convert).
    """
    if panel is None:
        panel = [1, 2]
    elif isinstance(panel, int):
        panel = [panel]
    if any(p > 2 for p in panel):
        raise ValueError("plot.oxy_crit: 'panel' input should be 1 to 2 or "
                         "'NULL' for both.")
    convert = x.get("convert", True)
    method = x.get("method", "bsr")
    df = np.asarray(x["dataframe"], dtype=float)
    dro = np.asarray(x["df_rate_oxygen"], dtype=float)

    npanels = len(panel)
    fig, axes = plt.subplots(npanels, 1, figsize=(7, 4 * npanels))
    if npanels == 1:
        axes = [axes]
    pi = 0
    if 1 in panel:
        ax = axes[pi]; pi += 1
        ax.scatter(df[:, 0], df[:, 1], s=6, c=_C1)
        _grid(ax)
        if convert:
            if method == "bsr":
                ci = float(x["crit"]["crit.intercept"])
                cm = float(x["crit"]["crit.midpoint"])
                ax.axhline(ci, color=_LINE_BSR[0], lw=_LINE_WT,
                           ls=_LINE_TYPES_BSR[0])
                ax.axhline(cm, color=_LINE_BSR[1], lw=_LINE_WT,
                           ls=_LINE_TYPES_BSR[1])
                if legend:
                    ax.legend([f"Intercept (BSR): {ci:.4g}",
                               f"Midpoint (BSR): {cm:.4g}"],
                              loc="upper right", fontsize=8, frameon=False)
            else:
                cb = float(x["crit"])
                ax.axhline(cb, color=_LINE_SEG[0], lw=_LINE_WT,
                           ls=_LINE_TYPES_SEG[0])
                if legend:
                    ax.legend([f"Breakpoint (Seg): {cb:.4g}"],
                              loc="upper right", fontsize=8, frameon=False)
            ax.set_title("Oxygen~Time Timeseries", fontsize=11,
                         fontweight="bold")
        else:
            ylim = _extendrange(dro[:, 1])
            if rate_rev:
                ylim = (ylim[1], ylim[0])
            ax.set_ylim(ylim)
            ax.set_title("Rate~Oxygen Timeseries", fontsize=11,
                         fontweight="bold")
    if 2 in panel and method == "bsr":
        ax = axes[pi]; pi += 1
        ylim = _extendrange(dro[:, 1])
        if rate_rev:
            ylim = (ylim[1], ylim[0])
        ax.set_ylim(ylim)
        ax.scatter(dro[:, 0], dro[:, 1], s=6, c=_C1)
        _grid(ax)
        cutoff = float(x["summary"][0])
        seg1 = dro[dro[:, 0] <= cutoff]
        seg2 = dro[dro[:, 0] > cutoff]
        if len(seg1) >= 2:
            _lm_line(ax, seg1[:, 0], seg1[:, 1], ls=":", lw=1, c="black")
        if len(seg2) >= 2:
            _lm_line(ax, seg2[:, 0], seg2[:, 1], ls=":", lw=1, c="black")
        ci = float(x["crit"]["crit.intercept"])
        cm = float(x["crit"]["crit.midpoint"])
        ax.axvline(ci, color=_LINE_BSR[0], lw=_LINE_WT, ls=_LINE_TYPES_BSR[0])
        ax.axvline(cm, color=_LINE_BSR[1], lw=_LINE_WT, ls=_LINE_TYPES_BSR[1])
        if legend:
            ax.legend([f"Intercept (BSR): {ci:.4g}",
                       f"Midpoint (BSR): {cm:.4g}"],
                      loc="lower right", fontsize=8, frameon=False)
        ax.set_title("Broken-Stick Result (Rate~Oxygen)", fontsize=11,
                     fontweight="bold")
    elif 2 in panel and method == "segmented":
        ax = axes[pi]; pi += 1
        ylim = _extendrange(dro[:, 1])
        if rate_rev:
            ylim = (ylim[1], ylim[0])
        ax.set_ylim(ylim)
        ax.scatter(dro[:, 0], dro[:, 1], s=6, c=_C1)
        _grid(ax)
        cb = float(x["crit"])
        ax.axvline(cb, color=_LINE_SEG[0], lw=_LINE_WT, ls=_LINE_TYPES_SEG[0])
        # segmented fit line (results = seg_fit x/y)
        sf = np.asarray(x["results"], dtype=float)
        if sf.ndim == 2 and sf.shape[1] >= 2:
            ax.plot(sf[:, 0], sf[:, 1], ls=":", lw=1, c="black")
        if legend:
            ax.legend([f"Breakpoint (Seg): {cb:.4g}"], loc="lower right",
                      fontsize=8, frameon=False)
        ax.set_title("Segmented Result (Rate~Oxygen)", fontsize=11,
                     fontweight="bold")

    fig.tight_layout()
    if save:
        fig.savefig(save)
    else:
        plt.show()
    return fig
