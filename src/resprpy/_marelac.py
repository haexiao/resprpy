"""Port of the marelac R package (v2.1.11) functions used by respR.

Only the pieces respR uses are ported:
- vapor():      water vapour pressure over seawater (bar)
- gas_solubility() / gas_satconc(): O2 solubility (Benson & Krause 1984)
- molweight() / molvol(): O2 molar weight / molar volume (van der Waals)
- sw_dens() / sw_gibbs(): seawater density via the UNESCO Gibbs potential
- atmComp():   dry-air mole fraction of O2

All vectorised with numpy; scalar input gives 0-d output like R scalars.
Verified against marelac 2.1.11 on R 4.5.3 (see tests/test_marelac.py).
"""
from __future__ import annotations

import numpy as np

from ._gibbs_coeffs import CARR, GJK

# ---------------------------------------------------------------------------
# physical constants (marelac::.marelac and atomicweight)
# ---------------------------------------------------------------------------
MOLW_WEIGHT = {"O2": 31.9988}      # g/mol (marelac::molweight("O2"))
ATM_COMP = {"O2": 0.20946}         # dry-air mole fraction
WAALS_A = {"O2": 1.378}            # L^2 bar / mol^2
WAALS_B = {"O2": 0.03183}          # L / mol
R_GAS = 0.0831447215               # L bar / (mol K)

# marelac::.marelac$SolubCoeff["O2",] -- Benson & Krause (1984), type 1
SOLUB_COEFF_O2 = {
    "A1": -58.3877, "A2": 85.8079, "A3": 23.8439, "A4": 0.0,
    "B1": -0.034892, "B2": 0.015568, "B3": -0.0019387, "type": 1,
}

MOLAR_VOL_STP = 22.4136            # L/mol, used by type-1 solubility


def vapor(S=35.0, t=25.0):
    """Water vapour pressure (bar) over seawater of salinity S at t degC."""
    S = np.asarray(S, dtype=float)
    t = np.asarray(t, dtype=float)
    K = 273.15 + t
    return np.exp(24.4543 - 67.4509 * (100.0 / K)
                  - 4.8489 * np.log(K / 100.0) - 0.000544 * S)


def _solub_o2(S, t):
    """Benson-Krause solubility of O2 in water (umol/L at 1 atm-ish)."""
    S = np.asarray(S, dtype=float)
    t = np.asarray(t, dtype=float)
    K = t + 273.15
    c = SOLUB_COEFF_O2
    bet = (c["A1"] + c["A2"] * (100.0 / K) + c["A3"] * np.log(K / 100.0)
           + c["A4"] * (K / 100.0) ** 2
           + S * (c["B1"] + c["B2"] * (K / 100.0) + c["B3"] * (K / 100.0) ** 2))
    if c["type"] == 1:
        return np.exp(bet) / MOLAR_VOL_STP * 1e6 / 1.013253
    raise NotImplementedError("only solubility type 1 (O2) is ported")


def gas_solubility(S=35.0, t=25.0, species="O2"):
    """marelac::gas_solubility for O2."""
    if species != "O2":
        raise NotImplementedError("only O2 is ported")
    if np.any(np.asarray(S, dtype=float) < 0):
        raise ValueError("Salinity should be >= 0")
    return _solub_o2(S, t)


def gas_satconc(S=35.0, t=25.0, P=1.013253, species="O2"):
    """marelac::gas_satconc: O2 saturation concentration (umol/L)."""
    if species != "O2":
        raise NotImplementedError("only O2 is ported")
    S = np.asarray(S, dtype=float)
    t = np.asarray(t, dtype=float)
    P = np.asarray(P, dtype=float)
    if np.any(S < 0):
        raise ValueError("Salinity should be >= 0")
    Vapor = vapor(t=t, S=S)
    return _solub_o2(S, t) * P * ATM_COMP["O2"] * (1.0 - Vapor)


def molweight(species="O2"):
    """marelac::molweight: molar weight in g/mol (only O2 ported)."""
    if species != "O2":
        raise NotImplementedError("only O2 is ported")
    return MOLW_WEIGHT["O2"]


def _vdw_root(P, T, a, b, qty=1.0):
    """Van der Waals molar volume: solve (P + qty^2 a/V^2)(V/qty - b) = R T.

    Mirrors marelac::molvol's uniroot over [-10, 1e6]; scipy brentq is used
    which is far more accurate than R's uniroot tolerance, so results agree
    with R to well beyond any practical precision.
    """
    from scipy.optimize import brentq

    def f(V):
        return (P + qty * qty * a / (V * V)) * (V / qty - b) - R_GAS * T

    return brentq(f, -10.0, 1e6)


def molvol(t=25.0, P=1.013253, species="O2", quantity=1.0):
    """marelac::molvol: molar volume of O2 (L/mol) at temperature t (degC)
    and pressure P (bar), via the van der Waals equation."""
    if species != "O2":
        raise NotImplementedError("only O2 is ported")
    t = np.asarray(t, dtype=float)
    P = np.asarray(P, dtype=float)
    quantity = np.asarray(quantity, dtype=float)
    t, P, quantity = np.broadcast_arrays(t, P, quantity)
    out = np.empty(t.shape, dtype=float)
    a = WAALS_A["O2"]
    b = WAALS_B["O2"]
    for idx, (tt, PP, qq) in enumerate(zip(t.ravel(), P.ravel(), quantity.ravel())):
        TK = 273.15 + tt
        if a == 0.0 and b == 0.0:
            out.ravel()[idx] = qq * R_GAS * TK / PP
        else:
            out.ravel()[idx] = _vdw_root(PP, TK, a, b, qq)
    return out


# ---------------------------------------------------------------------------
# UNESCO seawater Gibbs potential (marelac::sw_gibbs / sw_dens, "Gibbs" method)
# ---------------------------------------------------------------------------
_SU = 40.188617
_TU = 40.0
_PU = 1e8


def sw_gibbs(S=35.0, t=25.0, p=0.0, dS=0, dt=0, dp=0):
    """marelac::sw_gibbs. p is gauge pressure in bar (p = P - 1.013253).
    Returns the Gibbs potential (J/kg); derivatives selected by dS/dt/dp."""
    S = np.asarray(S, dtype=float)
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    S, t, p = np.broadcast_arrays(S, t, p)
    shape = S.shape
    S = S.ravel().copy()
    t = t.ravel()
    p = p.ravel()
    n = S.size

    isna = np.isnan(S)
    S[isna] = 0.0
    if np.any(S < 0):
        raise ValueError("Salinity should be >= 0")

    x2 = S / _SU
    x = np.sqrt(x2)
    y = t / _TU
    z = p * 1e5 / _PU

    mfac = 1.0
    if dt == 1:
        mfac = mfac / _TU
    if dt == 2:
        mfac = mfac / _TU / _TU
    if dp == 1:
        mfac = mfac * 1e-8
    if dp == 2:
        mfac = mfac * 1e-16

    # ---- pure water part ----
    gjk = np.array(GJK, dtype=float)
    nr, nc = 8, 7
    if dt >= 1:
        nr -= 1
        gjk = gjk[1:, :]
        for j in range(1, nr):
            gjk[j, :] *= (j + 1)
    if dt == 2:
        nr -= 1
        gjk = gjk[1:, :]
        for j in range(1, nr):
            gjk[j, :] *= (j + 1)
    if dp >= 1:
        nc -= 1
        gjk = gjk[:, 1:]
        for k in range(1, nc):
            gjk[:, k] *= (k + 1)
    if dp == 2:
        nc -= 1
        gjk = gjk[:, 1:]
        for k in range(1, nc):
            gjk[:, k] *= (k + 1)

    yp = y[:, None] ** np.arange(nr)      # y^(j-1),  (n, nr)
    zp = z[:, None] ** np.arange(nc)      # z^(k-1),  (n, nc)
    Gpure = np.sum((yp @ gjk) * zp, axis=1)

    # ---- seawater part ----
    Gsea = np.zeros(n)
    if np.any(S > 0):
        gijk = np.zeros((7, 7, 6))
        for i, j, k, coef in CARR:
            # R: Carr[,2:3] += 1 (j and k), then gijk[cbind(i,j,k)] = coef
            gijk[int(i) - 1, int(j), int(k)] = coef
        nj, nk, ni = 7, 6, 7
        if dt >= 1:
            nj -= 1
            gijk = gijk[:, 1:, :]
            for j in range(1, nj):
                gijk[:, j, :] *= (j + 1)
        if dt == 2:
            nj -= 1
            gijk = gijk[:, 1:, :]
            for j in range(1, nj):
                gijk[:, j, :] *= (j + 1)
        if dp >= 1:
            nk -= 1
            gijk = gijk[:, :, 1:]
            for k in range(1, nk):
                gijk[:, :, k] *= (k + 1)
        if dp == 2:
            nk -= 1
            gijk = gijk[:, :, 1:]
            for k in range(1, nk):
                gijk[:, :, k] *= (k + 1)

        yp2 = y[:, None] ** np.arange(nj)  # (n, nj)
        zp2 = z[:, None] ** np.arange(nk)  # (n, nk)

        if dS == 0 and dt == 0:
            # term1: gijk[0,j,k] * (x2 log x) * y^(j-1) z^(k-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                s1 = np.sum((yp2 @ gijk[0, :, :]) * zp2, axis=1)   # (n,)
                term1 = (x2 * np.log(x)) * s1
                # term2: sum_{i=2..7} gijk[i-1,j,k] x^i, then over j,k
                xp = x[:, None] ** np.arange(2, 8)                 # x^i, i=2..7, (n, 6)
                inner = np.einsum("ni,ijk->njk", xp, gijk[1:, :, :])
                term2 = np.sum(inner * yp2[:, :, None] * zp2[:, None, :], axis=(1, 2))
            Gsea = term1 + term2
        elif dS == 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                xp = x[:, None] ** np.arange(2, 8)
                inner = np.einsum("ni,ijk->njk", xp, gijk[1:, :, :])
                Gsea = np.sum(inner * yp2[:, :, None] * zp2[:, None, :], axis=(1, 2))
                if dt == 1:
                    Gsea = Gsea + gijk[0, 0, 0] * x2 * np.log(x)
                if dp == 1:
                    Gsea = Gsea + gijk[0, 1, 0] * x2 * np.log(x)
        else:
            raise NotImplementedError("dS != 0 derivative not ported")

        Gsea[np.isnan(Gsea)] = 0.0

    Gibbs = Gsea + Gpure
    Gibbs[isna] = np.nan
    return (Gibbs * mfac).reshape(shape)


def sw_dens(S=35.0, t=25.0, P=1.013253):
    """marelac::sw_dens (method="Gibbs"): seawater density in kg/m3."""
    S = np.asarray(S, dtype=float)
    t = np.asarray(t, dtype=float)
    P = np.asarray(P, dtype=float)
    if np.any(S < 0):
        raise ValueError("Salinity should be >= 0")
    S, t, P = np.broadcast_arrays(S, t, P)
    p = np.maximum(0.0, P - 1.013253)
    g = sw_gibbs(S, t, p, dS=0, dt=0, dp=1)
    return 1.0 / g
