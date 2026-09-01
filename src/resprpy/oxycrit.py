"""oxy_crit: critical oxygen tension (Pcrit/O2crit) analysis via
Broken-Stick regression (Yeager & Ultsch 1989) or Segmented regression
(Muggeo 2003, ported from the R 'segmented' package 2.2-1).

Ported from respR 2.3.4.
"""
from __future__ import annotations

import re

import numpy as np

from .data import subsample
from .auto import rolling_reg_row
from ._rng import RMersenneTwister


def _roll_mean(y, width):
    """roll::roll_mean(y, width): trailing-window mean, first width-1 NA."""
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    out = np.full(n, np.nan)
    for i in range(width - 1, n):
        out[i] = np.mean(y[i - width + 1:i + 1])
    return out


def _static_roll_slope(df, width):
    """static_roll(df, win)$slope_b1: rolling regression slope per window.

    Matches roll::roll_lm numerics: b = solve(X'X, X'y) per window.
    """
    n = df.shape[0]
    out = np.full(n, np.nan)
    for i in range(width - 1, n):
        sub = df[i - width + 1:i + 1]
        t = sub[:, 0].astype(float)
        o = sub[:, 1].astype(float)
        X = np.column_stack([np.ones_like(t), t])
        b = np.linalg.solve(X.T @ X, X.T @ o)
        out[i] = b[1]
    return out


def generate_mrdf(dt, width):
    """Build rate~oxygen data: x = rolling mean of oxygen, y = rolling slope."""
    dt = np.asarray(dt, dtype=float)
    y = dt[:, 1]
    rollx = _roll_mean(y, width)
    rolly = _static_roll_slope(dt, width)
    mask = ~(np.isnan(rollx) | np.isnan(rolly))
    return np.column_stack([rollx[mask], rolly[mask]])


def broken_stick(dt, n):
    """Two-segment OLS fit split at row n; returns splitpoint, sumRSS,
    intersection intercept and midpoint."""
    x = dt[:, 0]
    y = dt[:, 1]

    def lm_fit(sub):
        X = np.column_stack([np.ones(len(sub)), sub[:, 0]])
        b = np.linalg.lstsq(X, sub[:, 1], rcond=None)[0]
        return b, np.sum((sub[:, 1] - X @ b) ** 2)

    s1 = slice(0, n)  # R: dt[1:n] -- rows 1..n (n rows, 1-based)
    s2 = slice(n, len(x))  # R: dt[(n + 1):nrow(dt)]
    b1, rss1 = lm_fit(dt[s1])
    b2, rss2 = lm_fit(dt[s2])
    sum_rss = rss1 + rss2
    intercept = (b2[0] - b1[0]) / (b1[1] - b2[1]) if b1[1] != b2[1] else np.nan
    midpoint = np.mean([x[n], x[n - 1]]) if n < len(x) else np.nan
    return {"splitpoint": float(x[n - 1]), "sumRSS": float(sum_rss),
            "pcrit.intercept": float(intercept), "pcrit.midpoint": float(midpoint),
            "l1_coef.b0": float(b1[0]), "l1_coef.b1": float(b1[1]),
            "l2_coef.b0": float(b2[0]), "l2_coef.b1": float(b2[1])}


def _calc_mode(x):
    """R mode.default-ish: first most frequent value (ties -> smallest)."""
    x = np.asarray(x)
    vals, counts = np.unique(x, return_counts=True)
    return vals[np.argmax(counts)]


def _brent_fmin(f, ax, bx, tol):
    """R's Brent_fmin (optimize.c), verbatim port: golden-section +
    parabolic-interpolation one-dimensional minimisation."""
    c = (3.0 - np.sqrt(5.0)) * 0.5
    eps = np.finfo(float).eps
    tol1 = eps + 1.0
    eps = np.sqrt(eps)
    a, b = ax, bx
    v = a + c * (b - a)
    w = x = v
    d = e = 0.0
    fx = f(x)
    fv = fw = fx
    tol3 = tol / 3.0
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
        p = q = r = 0.0
        if abs(e) > tol1:  # fit parabola
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):
            # golden-section step
            e = (b - x) if x < xm else (a - x)
            d = c * e
        else:  # parabolic-interpolation step
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1
                if x >= xm:
                    d = -d
        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1
        fu = f(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x


# ======================================================================
# Segmented regression (Muggeo 2003) -- verbatim port of 'segmented' 2.2-1
# seg.lm.fit / seg.lm.fit.boot / segmented.default (oxy_crit calling path)
# ======================================================================

def _adj_psi(psii, lim):
    """segmented::adj.psi: pmin(pmax(LIM[1,], psii), LIM[2,])."""
    return np.minimum(np.maximum(lim[0], psii), lim[1])


def _in_psi_ids(LIM, PSI):
    """segmented::in.psi(ret.id=TRUE): per-psi in-range flags."""
    a = PSI < LIM[0]
    b = PSI > LIM[1]
    return ~a & ~b


def _far_psi(Z, PSI, id_psi_group, ret_id=True, fc=0.93):
    """segmented::far.psi: per-group bin-count check with factor."""
    n_seg = len(np.unique(id_psi_group))
    id_far_ok = []
    ff = []
    for g in np.unique(id_psi_group):
        mask = id_psi_group == g
        zz = Z[:, mask]
        pp = PSI[:, mask]
        npsij = int(np.sum(mask))
        rs = np.sum((zz > pp).astype(int), axis=1) + 1
        # R tabulate: bins start at 1, length = max(bin) -- no bin 0
        nj = np.bincount(rs, minlength=npsij + 2)[1:]
        if len(nj) != npsij + 1:
            rs = np.sum((zz >= pp).astype(int), axis=1) + 1
            nj = np.bincount(rs, minlength=npsij + 2)[1:]
        id_ok = nj >= 2
        id_far_ok.append(id_ok[:-1])
        ff.append(np.where(np.diff(nj) > 0, 1.0 / fc, fc))
    id_far_ok = np.concatenate(id_far_ok)
    ff = np.concatenate(ff)
    if not ret_id:
        return bool(np.all(id_far_ok))
    return id_far_ok, ff


def _mylm(Xm, y, w=None, offs=None):
    """segmented mylm / mylmWO: b = solve(crossprod(X), crossprod(X, y))."""
    if w is None or (np.var(w) <= 0 and np.var(offs) <= 0):
        b = np.linalg.solve(Xm.T @ Xm, Xm.T @ y)
        fit = Xm @ b
        r = y - fit
        return {"coefficients": b, "fitted.values": fit, "residuals": r,
                "L0": float(r @ r), "df.residual": len(y) - len(b)}
    sw = np.sqrt(w)
    x1 = Xm * sw[:, None]
    y1 = (y - offs) * sw
    b = np.linalg.solve(x1.T @ x1, x1.T @ y1)
    fit = Xm @ b
    r = y - fit
    return {"coefficients": b, "fitted.values": fit, "residuals": r,
            "L0": float(np.sum(w * r ** 2)), "df.residual": len(y) - len(b)}


def _seg_lm_fit(y, XREG, Z, PSI, w, offs, opz, return_all_sol=False):
    """segmented::seg.lm.fit (single fit; iterated Muggeo refinement).

    Returns dict with keys obj/it/psi/psi.values/idU/U/V/rangeZ/epsilon/
    nomiOK/SumSquares.no.gap/id.psi.group/id.warn. On the fix.npsi failure
    path with return_all_sol=TRUE returns (dev_values, psi_values) tuple.
    """
    n = len(y)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    PSI = np.asarray(PSI, dtype=float)
    if PSI.ndim == 1:
        PSI = PSI.reshape(1, -1)
    alpha = opz["alpha"]
    rangeZ = opz.get("rangeZ")
    if rangeZ is None:
        rangeZ = np.array([np.min(Z, axis=0), np.max(Z, axis=0)])
    limZ = opz.get("limZ")
    if limZ is None:
        limZ = np.array([np.quantile(Z, alpha[0], axis=0),
                         np.quantile(Z, alpha[1], axis=0)])

    psi = PSI[0].copy()
    psi = _adj_psi(psi, limZ)
    PSI = np.tile(psi, (n, 1))
    id_psi_group = np.asarray(opz["id_psi_group"])
    hh = opz["h"]
    digits = opz.get("digits")
    nomiOK = opz["nomiOK"]
    toll = opz["toll"]
    fix_npsi = opz.get("stop.if.error", opz.get("fix.npsi", True))
    visual = opz.get("visual", False)
    it_max = opz["it.max"]
    fc = opz["fc"]
    it = 0
    epsilon = 10.0
    k_values = []
    dev_values = []
    psi_values = []
    invXtX = opz.get("invXtX")
    Xty = opz.get("Xty")
    id_w_offs = (np.var(offs) <= 0) and (np.var(w) <= 0)
    if id_w_offs:
        def fitter(Xm):
            b, *_ = np.linalg.lstsq(Xm, y, rcond=None)
            return b
    else:
        def fitter(Xm):
            sw = np.sqrt(w)
            b, *_ = np.linalg.lstsq(Xm * sw[:, None], (y - offs) * sw,
                                    rcond=None)
            return b

    is_ok = _in_psi_ids(limZ, PSI)
    if not (bool(np.all(is_ok)) and not np.any(np.isnan(is_ok))):
        raise ValueError("starting psi out of the range.. see 'alpha' in "
                         "seg.control.")
    if not _far_psi(Z, PSI, id_psi_group, False):
        raise ValueError("psi starting values too close each other or at the "
                         "boundaries. Please change them (e.g. set "
                         "'quant=TRUE' in seg.control()), or decrease their "
                         "number.")

    n_psi = Z.shape[1]
    V = (Z > PSI).astype(float)
    U = (Z - PSI) * V
    V = -V

    # ---- id.changeCoef: shift segmented columns of XREG by their min ----
    id_change_coef = False
    minZ = None
    xreg_names = opz.get("xreg_names")
    nomi_seg = opz.get("nomiSeg")
    if xreg_names is not None and nomi_seg:
        hit = [i for i, nm in enumerate(xreg_names) if nm in nomi_seg]
        if hit:
            id_change_coef = True
            for i in hit:
                minZ = float(np.min(XREG[:, i]))
                XREG = XREG.copy()
                XREG[:, i] = XREG[:, i] - minZ

    if it_max == 0:
        obj = _mylm(np.column_stack([XREG, U]), y, w, offs)
        L1 = obj["L0"]
        obj["coefficients"] = np.concatenate(
            [obj["coefficients"], np.zeros(V.shape[1])])
        obj["epsilon"] = epsilon
        obj["it"] = it
        return {"obj": obj, "it": it, "psi": psi, "psi.values": psi_values,
                "idU": XREG.shape[1] + np.arange(1, len(psi) + 1),
                "U": U, "V": V, "rangeZ": rangeZ, "epsilon": epsilon,
                "nomiOK": nomiOK, "SumSquares.no.gap": L1,
                "id.psi.group": id_psi_group, "id.warn": True}

    if not opz.get("usesegreg", False):
        dev_values.append(opz["dev0"])
        psi_values.append(None)  # NA

    if opz.get("fit.psi0") is None:
        obj0 = _mylm(np.column_stack([XREG, U]), y, w, offs)
        L0 = obj0["L0"]
    else:
        L0 = opz["fit.psi0"]["L0"]

    n_int_dev0 = len(str(abs(int(L0))))
    dev_values.append(L0)
    psi_values.append(psi.copy())
    id_warn = False
    id_psi_changed = np.zeros(it_max, dtype=bool)
    tol_op = (np.linspace(0.001, np.finfo(float).eps ** 0.25, it_max)
              if opz.get("tol.opt") is None
              else np.repeat(opz["tol.opt"], it_max))
    idU = XREG.shape[1] + np.arange(1, n_psi + 1)
    idV = np.arange(1, n_psi + 1) + int(np.max(idU))

    while abs(epsilon) > toll:
        it += 1
        X = np.column_stack([XREG, U, V])
        coef = fitter(X)
        beta_c = coef[idU - 1]
        gamma_c = coef[idV - 1]
        if np.any(beta_c == 0.0) or np.any(gamma_c == 0.0):
            if fix_npsi:
                if return_all_sol:
                    return dev_values, psi_values
                raise ValueError("breakpoint estimate too close or at the "
                                 "boundary causing NA estimates.. too many "
                                 "breakpoints being estimated?")
            else:
                id_ok = gamma_c != 0
                psi = psi[id_ok]
                if len(psi) <= 0:
                    return 0
                gamma_c = gamma_c[id_ok]
                beta_c = beta_c[id_ok]
                Z = Z[:, id_ok]
                rangeZ = rangeZ[:, id_ok]
                limZ = limZ[:, id_ok]
                nomiOK = np.asarray(nomiOK)[id_ok]
                id_psi_group = id_psi_group[id_ok]
        psi_old = psi.copy()
        psi = psi_old + hh * gamma_c / beta_c
        psi = _adj_psi(psi, limZ)
        # tapply sort within groups (single group: no-op)
        psi = _sort_by_group(psi, id_psi_group)

        def search_min(hh_):
            psi_k = psi * hh_ + psi_old * (1.0 - hh_)
            PSI_k = np.tile(psi_k, (n, 1))
            U1 = (Z - PSI_k) * (Z > PSI_k)
            obj1 = _mylm(np.column_stack([XREG, U1]), y, w, offs)
            return obj1["L0"]

        use_k = _brent_fmin(search_min, 0.0, 1.0, tol_op[it - 1])
        L1 = search_min(use_k)
        k_values.append(use_k)
        psi = psi * use_k + psi_old * (1.0 - use_k)
        psi = _adj_psi(psi, limZ)
        if digits is not None:
            psi = np.round(psi, digits)
        PSI = np.tile(psi, (n, 1))
        V = (Z > PSI).astype(float)
        U = (Z - PSI) * V
        V = -V
        epsilon = (L0 - L1) / (abs(L0) + 0.1)
        L0 = L1
        k_values.append(use_k)
        psi_values.append(psi.copy())
        dev_values.append(L0)
        id_psi_far, ff = _far_psi(Z, PSI, id_psi_group, True, fc=fc)
        id_psi_in = _in_psi_ids(limZ, PSI)
        id_psi_ok = id_psi_in & id_psi_far
        if not bool(np.all(id_psi_ok)):
            if fix_npsi:
                psi = psi * np.where(id_psi_far, 1.0, ff)
                PSI = np.tile(psi, (n, 1))
                id_psi_changed[it - 1] = True
            else:
                Z = Z[:, id_psi_ok]
                PSI = PSI[:, id_psi_ok]
                rangeZ = rangeZ[:, id_psi_ok]
                limZ = limZ[:, id_psi_ok]
                nomiOK = np.asarray(nomiOK)[id_psi_ok]
                id_psi_group = id_psi_group[id_psi_ok]
                psi_old = psi_old[id_psi_ok]
                psi = psi[id_psi_ok]
                if PSI.shape[1] <= 0:
                    return 0
        if it >= it_max:
            id_warn = True
            break

    psi = _sort_by_group(psi, id_psi_group)
    PSI = np.tile(psi, (n, 1))
    V = (Z > PSI).astype(float)
    U = (Z - PSI) * V
    V = -V
    obj = _mylm(np.column_stack([XREG, U]), y, w, offs)
    L1 = obj["L0"]
    # id.changeCoef: fix intercept
    if id_change_coef:
        id_int = 0  # "(Intercept)" is first column in XREG
        obj["coefficients"][id_int] -= np.sum(
            obj["coefficients"][1:1 + len(nomi_seg)] * minZ)
    obj["coefficients"] = np.concatenate(
        [obj["coefficients"], np.zeros(V.shape[1])])
    obj["epsilon"] = epsilon
    obj["it"] = it
    return {"obj": obj, "it": it, "psi": psi, "psi.values": psi_values,
            "idU": idU, "U": U, "V": V, "rangeZ": rangeZ,
            "epsilon": epsilon, "nomiOK": nomiOK,
            "SumSquares.no.gap": L1, "id.psi.group": id_psi_group,
            "id.warn": id_warn}


def _sort_by_group(psi, id_psi_group):
    """unlist(tapply(psi, id.psi.group, sort)): sort within each group."""
    out = np.empty_like(psi)
    pos = 0
    for g in np.unique(id_psi_group):
        m = id_psi_group == g
        grp = np.sort(psi[m])
        out[pos:pos + len(grp)] = grp
        pos += len(grp)
    return out


def _extract_psi(dev_values, psi_values):
    """seg.lm.fit.boot extract.psi: best (min dev) psi from all.sol path."""
    dev_values = list(dev_values)
    psi_values = list(psi_values)
    if len(psi_values) and psi_values[0] is None:
        dev_values = dev_values[1:]
        psi_values = psi_values[1:]
    dev_arr = np.array(dev_values, dtype=float)
    id_ok = int(np.argmin(dev_arr))
    psi_ok = np.atleast_1d(psi_values[id_ok])
    return {"SumSquares.no.gap": float(dev_arr[id_ok]), "psi": psi_ok}


def _seg_seed_from_mean(mY):
    """seg.lm.fit.boot seed derivation from mean(y) (R string handling)."""
    sep_dec = r"\."
    s = f"{mY:.15g}" if not isinstance(mY, str) else mY
    parts = re.split(sep_dec, s)
    joined = "".join(parts)
    vv = [c for c in joined if c != "0"]
    vv = vv[:5]
    seed_str = "".join(vv)
    seed = int(seed_str) if seed_str else 1
    return seed


def _seg_lm_fit_boot(y, XREG, Z, PSI, w, offs, opz, n_boot=10, size_boot=None,
                     jt=False, non_param=True, random=True, break_boot=5):
    """segmented::seg.lm.fit.boot: bootstrap-restart search."""
    n = len(y)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    PSI = np.asarray(PSI, dtype=float)
    if PSI.ndim == 1:
        PSI = np.tile(PSI, (n, 1))
    w = np.asarray(w, dtype=float)
    offs = np.asarray(offs, dtype=float)
    id_psi_group = np.asarray(opz["id_psi_group"])

    # ---- seed ----
    if opz.get("seed") is None:
        mY = float(np.mean(y))
        seed = _seg_seed_from_mean(mY)
    elif np.isnan(opz["seed"]):
        rng0 = RMersenneTwister(1)
        seed = int("".join(str(x) for x in rng0.sample(0, 6, replace=True)))
        seed = int("".join(str(v) for v in rng0.sample(0, 6, replace=True)))
    else:
        seed = int(opz["seed"])
    rng = RMersenneTwister(seed)

    visual_boot = opz.get("visualBoot", False)
    opz1 = dict(opz)
    opz1["it.max"] = 0
    rangeZ = opz.get("rangeZ")
    if rangeZ is None:
        rangeZ = np.array([np.min(Z, axis=0), np.max(Z, axis=0)])
    alpha = opz["alpha"]
    limZ = opz.get("limZ")
    if limZ is None:
        limZ = np.array([np.quantile(Z, alpha[0], axis=0),
                         np.quantile(Z, alpha[1], axis=0)])

    # ---- o0 initial fit ----
    try:
        o0 = _seg_lm_fit(y, XREG, Z, PSI, w, offs, opz,
                         return_all_sol=False)
    except Exception:
        o0 = None
    if not isinstance(o0, dict):
        try:
            o0 = _seg_lm_fit(y, XREG, Z, opz["PSI1"], w, offs, opz,
                             return_all_sol=False)
        except Exception:
            o0 = None
    if not isinstance(o0, dict):
        res = _seg_lm_fit(y, XREG, Z, PSI, w, offs, opz,
                          return_all_sol=True)
        o0 = _extract_psi(*res)
        if not non_param:
            non_param = True
    if isinstance(o0, dict):
        est_psi00 = est_psi0 = o0["psi"]
        ss00 = o0["SumSquares.no.gap"]
    else:
        if random:
            est_psi00 = est_psi0 = np.array(
                [rng.unif() * (limZ[1, i] - limZ[0, i]) + limZ[0, i]
                 for i in range(Z.shape[1])])
            PSI1 = np.tile(est_psi0, (n, 1))
            o0 = _seg_lm_fit(y, XREG, Z, PSI1, w, offs, opz1)
            ss00 = o0["SumSquares.no.gap"]
        else:
            est_psi00 = est_psi0 = np.mean(PSI, axis=0)
            ss00 = opz["dev0"]

    k_psi = len(est_psi0)
    all_est_psi_boot = np.full((n_boot, k_psi), np.nan)
    all_selected_psi = np.full((n_boot, k_psi), np.nan)
    all_est_psi = np.full((n_boot, k_psi), np.nan)
    all_ss = np.full(n_boot, np.nan)
    all_selected_ss = np.full(n_boot, np.nan)
    if size_boot is None:
        size_boot = n
    Z_orig = Z
    count_random = 0
    k_psi_change = 1
    alpha_boot = 0.1
    n_boot_rev = 3

    for k in range(1, n_boot + 1):
        sel_ss = all_selected_ss[~np.isnan(all_selected_ss)]
        diff_sel = np.diff(sel_ss)[::-1]
        if (len(diff_sel) >= n_boot_rev - 1 and
                np.all(np.round(diff_sel[:n_boot_rev - 1], 6) == 0)):
            qpsi = np.array([np.mean(est_psi0[i] >= Z[:, i])
                             for i in range(Z.shape[1])])
            qpsi = np.where(np.abs(qpsi - 0.5) < 0.1, alpha_boot, qpsi)
            alpha_boot = 1 - alpha_boot
            est_psi0 = np.array([np.quantile(Z[:, i], 1 - qpsi[i])
                                 for i in range(Z.shape[1])])
        PSI = np.tile(est_psi0, (n, 1))
        if jt:
            Z = Z_orig + np.random.normal(0, 1e-4, Z_orig.shape)
        if non_param:
            id_s = rng.sample_int(n, size=size_boot, replace=True)
            id0 = np.array(id_s) - 1
            try:
                o_boot = _seg_lm_fit(y[id0], XREG[id0], Z[id0], PSI[id0],
                                     w[id0], offs[id0], opz,
                                     return_all_sol=False)
            except Exception:
                o_boot = None
        else:
            yy = fitted_ok + np.array(rng.sample(residuals_o0,
                                                 size=n, replace=True))
            o_boot = _seg_lm_fit(yy, XREG, Z_orig, PSI, w, offs, opz)
        if isinstance(o_boot, dict):
            est_psi_boot = o_boot["psi"]
        else:
            est_psi_boot = np.array(
                [rng.unif() * (limZ[1, i] - limZ[0, i]) + limZ[0, i]
                 for i in range(Z.shape[1])])
            est_psi_boot = _sort_by_group(est_psi_boot, id_psi_group)

        PSI = np.tile(est_psi_boot, (n, 1))
        try:
            o = _seg_lm_fit(y, XREG, Z_orig, PSI, w, offs, opz,
                            return_all_sol=True)
        except Exception:
            o = None

        if not isinstance(o, dict) and random:
            est_psi0 = np.array(
                [rng.unif() * (limZ[1, i] - limZ[0, i]) + limZ[0, i]
                 for i in range(Z.shape[1])])
            PSI1 = np.tile(est_psi0, (n, 1))
            o = _seg_lm_fit(y, XREG, Z, PSI1, w, offs, opz1)
            count_random += 1

        if isinstance(o, dict):
            if "coefficients" not in o["obj"]:
                o = _extract_psi(o.get("dev.values", []),
                                 o.get("psi.values", []))
            all_est_psi[k - 1] = o["psi"]
            all_ss[k - 1] = o["SumSquares.no.gap"]
            if o["SumSquares.no.gap"] <= o0["SumSquares.no.gap"]:
                o0 = o
                k_psi_change = k
            est_psi0 = o0["psi"]
            all_selected_psi[k - 1] = est_psi0
            all_selected_ss[k - 1] = o0["SumSquares.no.gap"]
        est_psi0 = _sort_by_group(est_psi0, id_psi_group)

        asss = all_selected_ss[~np.isnan(all_selected_ss)]
        if len(asss) > break_boot:
            if np.all(np.flip(np.round(np.diff(asss), 6))
                      [:break_boot - 1] == 0):
                break

    all_selected_psi = np.vstack([np.atleast_1d(est_psi00), all_selected_psi])
    all_selected_ss = np.concatenate([[ss00], all_selected_ss])
    ris = {"all.selected.psi": all_selected_psi,
           "all.selected.ss": all_selected_ss,
           "all.psi": all_est_psi, "all.ss": all_ss}

    if not isinstance(o0.get("obj"), dict):
        # psi too close: outdistance logic (seg.lm.fit.boot tail)
        min_n = opz.get("min.n", 2) - 1
        npsi = np.bincount(id_psi_group)[1:]
        new_psi = []
        for j, g in enumerate(np.unique(id_psi_group), start=1):
            mask = id_psi_group == g
            psi_j = np.sort(est_psi0[mask])
            Z_ok = np.unique(Z[:, mask][:, 0])
            m_j = np.min(limZ[0, mask])
            M_j = np.max(limZ[1, mask])
            for kk in range(len(psi_j)):
                edges = np.concatenate([[m_j - 1e8], psi_j, [M_j + 1e8]])
                id_group = np.searchsorted(edges, Z_ok, side="right") - 1
                id_group = np.clip(id_group, 0, len(edges) - 2)
                n_j = np.bincount(id_group, minlength=len(edges) - 1)
                if min_n > 1:
                    def min1(x):
                        for _ in range(min_n - 1):
                            x = x[x != np.min(x)]
                        return np.min(x) if len(x) else 0.0

                    def max1(x):
                        for _ in range(min_n - 1):
                            x = x[x != np.max(x)]
                        return np.max(x) if len(x) else 0.0
                else:
                    min1, max1 = np.min, np.max
                M_j_k = (max1(Z_ok[id_group == kk + 1]) - 1e6 * (n_j[kk + 1] <= min_n)
                         if n_j[kk + 1] > 0 else -1e6 * (n_j[kk] <= min_n))
                m_j_k = (min1(Z_ok[id_group == kk + 2]) + 1e6 * (n_j[kk + 2] <= min_n)
                         if n_j[kk + 2] > 0 else 1e6 * (n_j[kk] <= min_n))
                psi_j[kk] = psi_j[kk] + np.where(
                    abs(M_j_k - psi_j[kk]) < abs(m_j_k - psi_j[kk]),
                    M_j_k - psi_j[kk] - 0.0001,
                    m_j_k - psi_j[kk] + 0.0001)
            new_psi.append(psi_j)
        est_psi0 = np.concatenate(new_psi)
        PSI1 = np.tile(est_psi0, (n, 1))
        o0 = _seg_lm_fit(y, XREG, Z, PSI1, w, offs, opz1)

    if not isinstance(o0, dict):
        return 0
    o0["boot.restart"] = ris
    o0["seed"] = seed
    return o0


# ======================================================================
# oxy_crit
# ======================================================================

def _oxy_segmented(dt_mr):
    """oxy_crit method='segmented' full chain (lm -> segmented -> objF)."""
    x = dt_mr[:, 0]
    y = dt_mr[:, 1]
    n = len(y)
    # ---- lmfit <- lm(y ~ x, dt_mr) ----
    X_lm = np.column_stack([np.ones(n), x])
    b_lm = np.linalg.solve(X_lm.T @ X_lm, X_lm.T @ y)
    fit_lm = X_lm @ b_lm
    res_lm = y - fit_lm
    dev0 = float(res_lm @ res_lm)

    # ---- segmented.default setup ----
    seg_control = dict(toll=1e-5, it_max=30, visual=False, K=10, h=1.25,
                       n_boot=10, size_boot=None, gap=False, jt=False,
                       break_boot=5, non_param=True, random=True, pow_=[1, 1],
                       seed=None, quant=False, digits=None, conv_psi=False,
                       alpha=None, fix_npsi=True, fc=0.95, check_next=True,
                       tol_opt=None, fit_psi0=None, min_n=2)
    Z = x.reshape(-1, 1)
    # quant=FALSE -> psi init = psiE (equally spaced), PSI1 = psiQ (median)
    psiE = np.array([np.min(Z) + np.ptp(Z) / 2.0])
    psiQ = np.array([np.quantile(Z, 0.5)])
    initial_psi = psiE
    PSI1 = np.tile(psiQ, (n, 1))
    id_psi_group = np.array([1])
    k_psi = 1
    PSI = np.tile(psiE, (n, 1))
    # admissible-range check
    c1 = np.all(Z <= PSI, axis=0)
    c2 = np.all(Z >= PSI, axis=0)
    if np.sum(c1 + c2) != 0 or np.isnan(np.sum(c1 + c2)):
        raise ValueError("starting psi out of the admissible range")
    alpha = max(0.05, 1.0 / n)
    alpha = np.array([alpha, 1.0 - alpha])
    XREG = np.column_stack([np.ones(n), x])
    opz = {"toll": seg_control["toll"], "h": seg_control["h"],
           "stop.if.error": seg_control["fix_npsi"], "dev0": dev0,
           "visual": seg_control["visual"], "it.max": seg_control["it_max"],
           "usesegreg": False, "tol.opt": seg_control["tol_opt"],
           "nomiOK": ["U1.x"], "id.psi.group": id_psi_group,
           "gap": seg_control["gap"],
           "visualBoot": seg_control["visual"], "pow": seg_control["pow_"],
           "digits": seg_control["digits"], "invXtX": None, "Xty": None,
           "alpha": alpha, "fix.npsi": seg_control["fix_npsi"],
           "fc": seg_control["fc"], "seed": seg_control["seed"],
           "fit.psi0": seg_control["fit_psi0"], "min.n": seg_control["min_n"],
           "limZ": None, "rangeZ": None, "nomiSeg": ["x"],
           "PSI1": PSI1, "xreg_names": ["(Intercept)", "x"]}
    w = np.ones(n)
    offs = np.zeros(n)
    opz["id_psi_group"] = id_psi_group

    n_boot = seg_control["n_boot"]
    if n_boot <= 0:
        obj = _seg_lm_fit(y, XREG, Z, PSI, w, offs, opz)
    else:
        obj = _seg_lm_fit_boot(y, XREG, Z, PSI, w, offs, opz,
                               n_boot=n_boot,
                               size_boot=seg_control["size_boot"],
                               random=seg_control["random"],
                               break_boot=seg_control["break_boot"])
    if not isinstance(obj, dict):
        # return lm fit (obj0)
        return {"coefficients": b_lm, "psi": np.full((1, 3), np.nan),
                "it": 0, "fitted": fit_lm, "failed": True}
    seed = obj["seed"]
    it = obj["it"]
    psi = obj["psi"]
    U = obj["U"]
    V = obj["V"]
    rangeZ = obj["rangeZ"]
    idU = obj["idU"]
    nomiOK = obj["nomiOK"]
    obj_inner = obj["obj"]  # mylm result dict

    # ---- objF: lm(y ~ x + U1.x + psi1.x) ----
    idU0 = int(np.asarray(idU).ravel()[0]) - 1  # 0-based index of U column
    beta_c = obj_inner["coefficients"][idU0]
    Vxb = V * beta_c  # n x 1
    X_F = np.column_stack([XREG, U, Vxb])
    b_F = np.linalg.solve(X_F.T @ X_F, X_F.T @ y)
    fit_F = X_F @ b_F
    res_F = y - fit_F
    objF = {"coefficients": b_F, "fitted.values": fit_F, "residuals": res_F,
            "df.residual": n - X_F.shape[1]}
    # coefficient replacement (segmented.lm.r)
    objF["coefficients"][0:2] = obj_inner["coefficients"][0:2]
    objF["coefficients"][2] = obj_inner["coefficients"][idU0]
    objF["coefficients"][3] = 0.0
    objF["fitted.values"] = obj_inner["fitted.values"]
    objF["residuals"] = obj_inner["residuals"]
    # vcov (summary.lm): sigma^2 * chol2inv(qr.R)
    sigma2 = float(objF["residuals"] @ objF["residuals"]) / objF["df.residual"]
    cov = sigma2 * np.linalg.inv(X_F.T @ X_F)
    vv = float(cov[3, 3])
    # ris.psi matrix: Initial / Est. / St.Err
    ris_psi = np.full((k_psi, 3), np.nan)
    ris_psi[:, 1] = psi
    ris_psi[:, 2] = np.sqrt(vv)
    initial = np.atleast_1d(initial_psi[0])
    ris_psi[:, 0] = initial

    coefficients = objF["coefficients"]
    return {"coefficients": coefficients, "psi": ris_psi, "it": it,
            "fitted": objF["fitted.values"], "obj": obj, "seed": seed,
            "failed": False}


def oxy_crit(x, method="bsr", time=None, oxygen=None, rate=None,
             width=0.1, parallel=False, thin=5000, plot=True, **kwargs):
    if isinstance(x, dict) and "dataframe" in x:
        df = np.asarray(x["dataframe"], dtype=float)
    else:
        df = np.asarray(x, dtype=float)
    if df.ndim == 1:
        df = df.reshape(-1, 1)
    if not (0.001 <= width <= 0.999):
        raise ValueError("oxy_crit: 'width' input should be between 0.001 to "
                         "0.999, representing a proportion of the total data "
                         "length.")
    if method not in ("bsr", "segmented"):
        raise ValueError("oxy_crit: 'method' input not recognised.")

    if time is None and oxygen is None and rate is None:
        time, oxygen = 1, 2
        col1, col2 = time, oxygen
        convert = True
    elif time is not None and oxygen is not None and rate is None:
        col1, col2 = time, oxygen
        convert = True
    elif time is None and oxygen is not None and rate is not None:
        col1, col2 = oxygen, rate
        convert = False
    else:
        raise ValueError("oxy_crit: Inputs should be 'time' and 'oxygen' "
                         "columns, or 'oxygen' and 'rate' columns.")

    dt = df[:, [col1 - 1, col2 - 1]].astype(float)
    if convert:
        win = int(np.floor(width * dt.shape[0]))
        dt_mr = generate_mrdf(dt, win)
    else:
        dt_mr = dt
    # setorder by x (col 1)
    dt_mr = dt_mr[np.argsort(dt_mr[:, 0], kind="stable")]

    if method == "bsr":
        if thin is not None and dt_mr.shape[0] > thin:
            sdt = subsample(dt_mr, length_out=thin, plot=False)
        else:
            sdt = dt_mr
        lseq = np.arange(3, sdt.shape[0] - 2 + 1)
        results = [broken_stick(sdt, z) for z in lseq]
        # rbindlist + setorder(sumRSS)
        keys = ["splitpoint", "sumRSS", "pcrit.intercept", "pcrit.midpoint",
                "l1_coef.b0", "l1_coef.b1", "l2_coef.b0", "l2_coef.b1"]
        mat = np.array([[r[k] for k in keys] for r in results])
        order = np.argsort(mat[:, 1], kind="stable")
        mat = mat[order]
        # results[, c(1, 2, 5:8, 3, 4)] -> splitpoint, sumRSS, l1.b0, l1.b1,
        # l2.b0, l2.b1, pcrit.intercept, pcrit.midpoint
        mat = mat[:, [0, 1, 4, 5, 6, 7, 2, 3]]
        rcols = ["splitpoint", "sumRSS", "l1_coef.b0", "l1_coef.b1",
                 "l2_coef.b0", "l2_coef.b1", "crit.intercept",
                 "crit.midpoint"]
        summary = mat[0]
        crit = {"crit.intercept": float(summary[6]),
                "crit.midpoint": float(summary[7])}
    else:  # segmented (Muggeo 2003)
        fit = _oxy_segmented(dt_mr)
        seg_fit = np.column_stack([dt_mr[:, 0], fit["fitted"]])
        crit = float(fit["psi"][0, 1])  # psi[2] = Est.
        summary = np.concatenate([fit["coefficients"],
                                  [fit["psi"][0, 2], crit]])
        mat = seg_fit

    if convert:
        dt_out = np.column_stack([dt[:, 0], dt[:, 1]])
    else:
        dt_out = dt
    out = {"call": None,
           "inputs": dict(x=x, method=method, time=time, oxygen=oxygen,
                          rate=rate, width=width, plot=plot, thin=thin,
                          parallel=parallel),
           "dataframe": dt_out, "df_rate_oxygen": dt_mr, "width": width,
           "convert": convert, "method": method, "results": mat,
           "summary": summary, "crit": crit}
    return out
