"""Unit string matching and scaling machinery, ported from respR.

respR identifies units by string-matching against named regex tables
(REGEX in _unit_regexes.py, auto-generated from the R source).  Canonical
unit keys carry a type suffix, e.g. "mg/L.o2", "hr.time", "kg.mass",
"m2.area", "pmol.o2", "kPa.o2p", "L.vol", "C.temp", "kPa.p".
"""
from __future__ import annotations

import re

import numpy as np

from ._unit_regexes import REGEX

# separator used to split rate unit strings like "mg/h/kg"
UNIT_SEP = re.compile(r"(?:-1|[_/.\s]|per)+")

# adjust_scale tables (respR adjust_scale.R)
_PREFIX = ["p", "n", "u", "m", "", "k", "sec", "min", "hr", "day"]
_MULTIP = [1e-12, 1e-9, 1e-6, 0.001, 1.0, 1000.0, 3600.0, 60.0, 1.0, 1.0 / 24.0]
_SCALE_RE = re.compile(r"^(p|n|u|m||k|sec|min|hr|day)?(mol|g|L|l|)$")

# adjust_scale_area tables
_AREA_PREFIX = ["m", "c", "", "k"]
_AREA_MULTIP = [1e-6, 1e-4, 1.0, 1e6]
_AREA_RE = re.compile(r"^(m|c||k)?(m2)$")


def _match_any(patterns, s):
    return any(re.search(p, s) for p in patterns)


def units_val(unit, is_="o2", msg="units.val"):
    """respR::units.val: canonical key(s) for a unit string of a given type."""
    all_units = _tables(is_)
    if is_ == "o2" and isinstance(unit, str) and unit.lower() in ("%", "perc", "percent", "percentage"):
        raise ValueError(
            f'{msg}: unit "%" has been deprecated. Please use "%Air" or "%Oxy" instead. See unit_args().')
    string = re.sub(r" ", "", re.sub(r"\s+", " ", str(unit)).strip())
    for key, patterns in all_units:
        if _match_any(patterns, string):
            return key
    raise ValueError(
        f"{msg}: unit '{unit}' not recognised. Check it is valid for the input or output type.\n"
        "Output rate unit strings should be in correct order: O2/Time or O2/Time/Mass or O2/Time/Area.\n"
        "See unit_args() for details.")


def _tables(is_):
    R = REGEX
    if is_ == "time":
        return [("min.time", R["min.time.rgx"]), ("sec.time", R["sec.time.rgx"]),
                ("hr.time", R["hr.time.rgx"]), ("day.time", R["day.time.rgx"])]
    if is_ == "o2":
        return [
            ("%Air.o2", R["PercAir.o2.rgx"]), ("%Oxy.o2", R["PercOxy.o2.rgx"]),
            ("mg/L.o2", R["mgperL.o2.rgx"]), ("ug/L.o2", R["ugperL.o2.rgx"]),
            ("mol/L.o2", R["molperL.o2.rgx"]), ("mmol/L.o2", R["mmolperL.o2.rgx"]),
            ("umol/L.o2", R["umolperL.o2.rgx"]), ("nmol/L.o2", R["nmolperL.o2.rgx"]),
            ("pmol/L.o2", R["pmolperL.o2.rgx"]), ("mL/L.o2", R["mLperL.o2.rgx"]),
            ("uL/L.o2", R["uLperL.o2.rgx"]), ("cm3/L.o2", R["cm3perL.o2.rgx"]),
            ("mm3/L.o2", R["mm3perL.o2.rgx"]), ("mg/kg.o2", R["mgperkg.o2.rgx"]),
            ("mg/kg.o2", R["ppm.o2.rgx"]), ("ug/kg.o2", R["ugperkg.o2.rgx"]),
            ("mL/kg.o2", R["mLperkg.o2.rgx"]), ("uL/kg.o2", R["uLperkg.o2.rgx"]),
            ("mol/kg.o2", R["molperkg.o2.rgx"]), ("mmol/kg.o2", R["mmolperkg.o2.rgx"]),
            ("umol/kg.o2", R["umolperkg.o2.rgx"]), ("nmol/kg.o2", R["nmolperkg.o2.rgx"]),
            ("pmol/kg.o2", R["pmolperkg.o2.rgx"]), ("cm3/kg.o2", R["cm3perkg.o2.rgx"]),
            ("mm3/kg.o2", R["mm3perkg.o2.rgx"]), ("Torr.o2p", R["Torr.o2p.rgx"]),
            ("hPa.o2p", R["hPa.o2p.rgx"]), ("kPa.o2p", R["kPa.o2p.rgx"]),
            ("mmHg.o2p", R["mmHg.o2p.rgx"]), ("inHg.o2p", R["inHg.o2p.rgx"]),
        ]
    if is_ == "vol":
        return [("L.vol", R["L.vol.rgx"]), ("mL.vol", R["mL.vol.rgx"]), ("uL.vol", R["uL.vol.rgx"])]
    if is_ == "mass":
        return [("kg.mass", R["kg.mass.rgx"]), ("g.mass", R["g.mass.rgx"]),
                ("mg.mass", R["mg.mass.rgx"]), ("ug.mass", R["ug.mass.rgx"])]
    if is_ == "area":
        return [("km2.area", R["km2.area.rgx"]), ("m2.area", R["m2.area.rgx"]),
                ("cm2.area", R["cm2.area.rgx"]), ("mm2.area", R["mm2.area.rgx"])]
    if is_ == "o1":
        return [("mg.o2", R["mg.o2.rgx"]), ("ug.o2", R["ug.o2.rgx"]),
                ("mol.o2", R["mol.o2.rgx"]), ("mmol.o2", R["mmol.o2.rgx"]),
                ("umol.o2", R["umol.o2.rgx"]), ("nmol.o2", R["nmol.o2.rgx"]),
                ("pmol.o2", R["pmol.o2.rgx"]), ("mL.o2", R["mL.o2.rgx"]),
                ("uL.o2", R["uL.o2.rgx"]), ("cm3.o2", R["cm3.o2.rgx"]),
                ("mm3.o2", R["mm3.o2.rgx"])]
    if is_ == "flow":
        # R units.val: `uL/sec.flow` = uLpersec.flow.rgx (regex object names are
        # slash-free camelCase)
        return [(k, R[v]) for k, v in (
            ("uL/sec", "uLpersec.flow.rgx"), ("mL/sec", "mLpersec.flow.rgx"),
            ("L/sec", "Lpersec.flow.rgx"), ("uL/min", "uLpermin.flow.rgx"),
            ("mL/min", "mLpermin.flow.rgx"), ("L/min", "Lpermin.flow.rgx"),
            ("uL/hr", "uLperhr.flow.rgx"), ("mL/hr", "mLperhr.flow.rgx"),
            ("L/hr", "Lperhr.flow.rgx"), ("uL/day", "uLperday.flow.rgx"),
            ("mL/day", "mLperday.flow.rgx"), ("L/day", "Lperday.flow.rgx"))]
    if is_ == "pressure":
        return [("kPa.p", R["kPa.p.rgx"]), ("hPa.p", R["hPa.p.rgx"]),
                ("Pa.p", R["Pa.p.rgx"]), ("uBar.p", R["uBar.p.rgx"]),
                ("mBar.p", R["mBar.p.rgx"]), ("Bar.p", R["Bar.p.rgx"]),
                ("atm.p", R["atm.p.rgx"]), ("Torr.p", R["Torr.p.rgx"]),
                ("mmHg.p", R["mmHg.p.rgx"]), ("inHg.p", R["inHg.p.rgx"])]
    if is_ == "temperature":
        return [("C.temp", R["C.temp.rgx"]), ("K.temp", R["K.temp.rgx"]),
                ("F.temp", R["F.temp.rgx"])]
    raise ValueError(f"unknown unit type '{is_}'")


def unit_type(unit, msg=""):
    """respR::unit_type: type of a unit string (time/vol/mass/area/pressure/temperature)."""
    string = re.sub(r" ", "", re.sub(r"\s+", " ", str(unit)).strip())
    for typ in ("time", "vol", "mass", "area", "pressure", "temperature"):
        for key, patterns in _tables(typ):
            if _match_any(patterns, string):
                return typ
    raise ValueError(f"{msg}: '{unit}' unit not recognised.")


def unit_type_o1(unit, msg=""):
    """respR::unit_type_o1: check a unit is an oxygen-amount unit (for rates)."""
    string = re.sub(r" ", "", re.sub(r"\s+", " ", str(unit)).strip())
    for key, patterns in _tables("o1"):
        if _match_any(patterns, string):
            return "o1"
    raise ValueError(
        f"{msg}: '{unit}' unit not recognised as an oxygen unit that can be used for rates or concentrations.")


def units_clean(unit, is_):
    """respR::units.clean: strip the type suffix from a canonical key."""
    s = str(unit)
    if is_ == "o2":
        s = re.sub(r"\.o2p", ".o2", s)
        s = re.sub(r"\.o2", "", s)
    elif is_ == "time":
        s = re.sub(r"\.time", "", s)
    elif is_ == "vol":
        s = re.sub(r"\.vol", "", s)
    elif is_ == "mass":
        s = re.sub(r"\.mass", "", s)
    elif is_ == "area":
        s = re.sub(r"\.area", "", s)
    elif is_ == "o1":
        s = re.sub(r"\.o1", "", s)
    elif is_ == "flow":
        s = re.sub(r"\.flow", "", s)
    elif is_ == "pressure":
        s = re.sub(r"\.p$", "", s)
    elif is_ == "temperature":
        s = re.sub(r"\.temp", "", s)
    return s


def adjust_scale(x, input_, output):
    """respR::adjust_scale: rescale values between canonical keys of the
    same base unit (e.g. "mmol.o2" -> "pmol.o2", "sec.time" -> "hr.time")."""
    x = np.asarray(x, dtype=float)
    bef = re.sub(r"\..*", "", str(input_))
    aft = re.sub(r"\..*", "", str(output))
    bm = _SCALE_RE.match(bef)
    am = _SCALE_RE.match(aft)
    if bm is None or am is None or bm.group(2) != am.group(2):
        raise ValueError("adjust_scale: Units do not match and cannot be converted.")
    a = _MULTIP[_PREFIX.index(bm.group(1))] if bm.group(1) else 1.0
    b = _MULTIP[_PREFIX.index(am.group(1))] if am.group(1) else 1.0
    return x * (a / b)


def adjust_scale_area(x, input_, output):
    """respR::adjust_scale_area: rescale between area canonical keys
    (e.g. "m2.area" -> "cm2.area")."""
    x = np.asarray(x, dtype=float)
    bef = re.sub(r"\..*", "", str(input_))
    aft = re.sub(r"\..*", "", str(output))
    bm = _AREA_RE.match(bef)
    am = _AREA_RE.match(aft)
    if bm is None or am is None or bm.group(2) != am.group(2):
        raise ValueError("adjust_scale_area: Units do not match and cannot be converted.")
    a = _AREA_MULTIP[_AREA_PREFIX.index(bm.group(1))] if bm.group(1) else 1.0
    b = _AREA_MULTIP[_AREA_PREFIX.index(am.group(1))] if am.group(1) else 1.0
    return x * (a / b)


def stp_check(unit, type_):
    """respR::StP.check: does this unit require S/t/P inputs?"""
    if type_ == "oxy":
        return _match_any(REGEX["oxy.StP.req.rgx"], unit)
    if type_ == "mr":
        return _match_any(REGEX["mr.StP.req.rgx"], unit)
    return False


def stp_val(unit, type_, S, t, P, P_chk=True, msg=""):
    """respR::StP.val: validate/complete S, t, P for a unit; returns P
    (possibly defaulted to 1.013253 bar)."""
    if P_chk and P is not None and not all(v is None for v in (P if isinstance(P, (list, tuple, np.ndarray)) else [P])):
        Pv = np.asarray(P, dtype=float)
        if np.any((Pv < 0.9) | (Pv > 1.08)):
            import warnings
            warnings.warn(
                f"{msg}: One or more of the Atmospheric Pressure inputs 'P' are outside the normal "
                "realistic range.\nP values should not be outside the typical natural range of 0.9 to 1.1 "
                "except for special applications.\nPlease make sure it is entered in 'bar' units. "
                "Conversion performed regardless.")
    if S is None:
        if stp_check(unit, type_):
            raise ValueError(f"{msg}: Input or output units require Salinity input (i.e. S = ??)")
    elif t is None:
        if stp_check(unit, type_):
            raise ValueError(f"{msg}: Input or output units require Temperature input (i.e. t = ??)")
    elif P is None:
        if stp_check(unit, type_):
            import warnings
            warnings.warn(
                f"{msg}: Input or output units require Atmospheric Pressure input (i.e. P = ??).\n"
                "Default value of P = 1.013253 bar has been used.")
            P = 1.013253
    return P


def split_unit_string(s):
    """Split a rate unit string like 'mg/h/kg' into parts ['mg','h','kg'],
    mirroring respR's read.table(text=gsub(unit.sep.rgx, ' ', unit))."""
    parts = [p for p in UNIT_SEP.split(str(s)) if p != ""]
    return parts


def unit_args():
    """Print the accepted unit strings (respR::unit_args)."""
    oxyunit = ["mg/L", "ug/L", "mol/L", "mmol/L", "umol/L", "nmol/L", "pmol/L"]
    oxyunit_tsp = ["uL/L", "mL/L", "mm3/L", "cm3/L", "cc/L", "mg/kg", "ug/kg",
                   "ppm", "mol/kg", "mmol/kg", "umol/kg", "nmol/kg", "pmol/kg",
                   "uL/kg", "mL/kg", "%Air", "%Oxy", "Torr", "hPa", "kPa",
                   "mmHg", "inHg"]
    oxyunit_out = ["ug", "mg", "pmol", "nmol", "umol", "mmol", "mol",
                   "uL", "mL", "mm3", "cm3"]
    timeunit = ["sec", "min", "hour", "day"]
    massunit = ["ug", "mg", "g", "kg"]
    areaunit = ["mm2", "cm2", "m2", "km2"]
    flowunit = ["uL", "mL", "L"]
    print("Note: A string-matching algorithm is used to identify units.")
    print("Example 1: These are recognised as the same: 'mg/L', 'mg/l', 'mg L-1', 'mg per litre', 'mg.L-1'")
    print("Example 2: These are recognised as the same: 'Hour', 'hr', 'h'")
    print("\n# Input Units # --------------------------------------")
    print("Oxygen concentration units should use SI units (`L` or `kg`) for the denominator.\n")
    print("Oxygen Concentration or Pressure Units - Do not require t, S and P")
    print(oxyunit)
    print("Oxygen Concentration or Pressure Units - Require t, S and P")
    print(oxyunit_tsp)
    print("\nVolume units for use in flow rates in calc_rate.ft and convert_rate.ft")
    print("(e.g. as in 'ml/min', 'L/s', etc.)")
    print(flowunit)
    print("\nTime units (for 'time.unit' or as part of 'flowrate.unit')")
    print(timeunit)
    print("\nMass units")
    print(massunit)
    print("\nArea units")
    print(areaunit)
    print("\n# Metabolic Rate Units # -----------------------------")
    print("\nFor use in 'convert_rate', 'convert_rate.ft', 'convert_MR'")
    print("\nMust be in correct order:")
    print("Absolute rates:        Oxygen/Time       e.g. 'mg/sec',     'umol/min',     'mL/h'")
    print("Mass-specific rates:   Oxygen/Time/Mass  e.g. 'mg/sec/ug',  'umol/min/g',   'mL/h/kg'")
    print("Area-specific rates:   Oxygen/Time/Area  e.g. 'mg/sec/mm2', 'umol/min/cm2', 'mL/h/m2'")
    print("\nOutput Oxygen amount units")
    print(oxyunit_out)
    print("\nOutput Time units")
    print(timeunit)
    print("\nOutput Mass units for mass-specific rates")
    print(massunit)
    print("\nOutput Area units for surface area-specific rates")
    print(areaunit)
