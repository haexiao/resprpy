"""convert_rate.ft: convert flow-through metabolic rates between units.

Ported from respR 2.3.4. Converts a rate (delta oxygen * flowrate, or an
existing calc_rate.ft / adjust_rate.ft object) to an output unit, optionally
mass- or area-specific.
"""
from __future__ import annotations

import numpy as np

from ._units import UNIT_SEP, units_val, units_clean, adjust_scale, \
    adjust_scale_area, stp_val
from .convert import convert_DO


def flow_unit_parse(unit, which):
    if which == "vol":
        if unit in ("uL/sec", "uL/min", "uL/hr", "uL/day"):
            return "uL.vol"
        if unit in ("mL/sec", "mL/min", "mL/hr", "mL/day"):
            return "mL.vol"
        if unit in ("L/sec", "L/min", "L/hr", "L/day"):
            return "L.vol"
    if which == "time":
        if unit in ("uL/sec", "mL/sec", "L/sec"):
            return "sec.time"
        if unit in ("uL/min", "mL/min", "L/min"):
            return "min.time"
        if unit in ("uL/hr", "mL/hr", "L/hr"):
            return "hr.time"
        if unit in ("uL/day", "mL/day", "L/day"):
            return "day.time"
    return None


def convert_rate_ft(x, oxy_unit=None, flowrate_unit=None, output_unit=None,
                    mass=None, area=None, S=None, t=None, P=1.013253,
                    plot=False, **kwargs):
    summ_ext_cols = ["rank", "intercept_b0", "slope_b1", "rsq", "row",
                     "endrow", "time", "endtime", "oxy", "endoxy",
                     "delta_mean", "flowrate", "rate", "adjustment",
                     "rate.adjusted"]

    is_obj = isinstance(x, dict) and "rate" in x and "summary" in x
    if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0) or \
            (isinstance(x, (list, np.ndarray)) and not is_obj):
        rate = np.atleast_1d(np.asarray(x, dtype=float))
        input_type = "vec"
        n = rate.size
        summ_ext = {c: np.full(n, np.nan) for c in summ_ext_cols}
        summ_ext["rank"] = np.arange(1, n + 1, dtype=float)
        summ_ext["rate"] = rate
    elif is_obj:
        rate = np.atleast_1d(np.asarray(x["rate"], dtype=float))
        input_type = x.get("input_type", "vec")
        summ_ext = dict(x["summary"])
        summ_ext["adjustment"] = np.full(rate.size, np.nan)
        summ_ext["rate.adjusted"] = np.full(rate.size, np.nan)
    else:
        raise ValueError("convert_rate.ft: 'x' must be an calc_rate.ft or "
                         "adjust_rate.ft object, or a numeric value or vector.")

    if oxy_unit is None:
        raise ValueError("convert_rate.ft: 'oxy.unit' input is required.")
    if flowrate_unit is None:
        raise ValueError("convert_rate.ft: 'flowrate.unit' input is required.")

    if output_unit is None and mass is None and area is None:
        output_unit = "mg/h"
    if output_unit is None and mass is not None and area is None:
        output_unit = "mg/h/kg"
    if output_unit is None and mass is None and area is not None:
        output_unit = "mg/h/m2"
    if mass is not None and area is not None:
        raise ValueError("convert_rate.ft: Cannot have inputs for both 'mass' "
                         "and 'area'.")

    oxy = units_val(oxy_unit, "o2")
    flow = units_val(flowrate_unit, "flow")
    out_tokens = [p for p in UNIT_SEP.split(str(output_unit)) if p != ""]
    is_spec = len(out_tokens) == 3
    if is_spec:
        if mass is not None and area is None:
            is_mass_spec, is_area_spec = True, False
        elif area is not None and mass is None:
            is_mass_spec, is_area_spec = False, True
        else:
            raise ValueError("convert_rate.ft: 'output.unit' requires a value "
                             "for 'mass' or 'area'")
    else:
        is_mass_spec = is_area_spec = False

    A = units_val(out_tokens[0], "o1")
    B = units_val(out_tokens[1], "time")
    if is_spec:
        C = units_val(out_tokens[2], "mass" if is_mass_spec else "area")

    if not is_mass_spec and mass is not None:
        raise ValueError("convert_rate.ft: a 'mass' has been entered, but a "
                         "mass-specific unit has not been specified in "
                         "'output.unit'.")
    if not is_area_spec and area is not None:
        raise ValueError("convert_rate.ft: an 'area' has been entered, but an "
                         "area-specific unit has not been specified in "
                         "'output.unit'.")

    oxy_clean = units_clean(oxy, "o2")
    flow_clean = units_clean(flow, "flow")
    out_clean = [str(u).split(".")[0] for u in out_tokens]
    out_clean[0] = out_clean[0] + "O2"
    output_clean = "/".join(out_clean)

    P = stp_val(oxy_clean, "oxy", S, t, P, P_chk=False, msg="convert_rate.ft")
    P = stp_val(output_clean, "mr", S, t, P, P_chk=True, msg="convert_rate.ft")

    # convert oxygen delta (concentration) to output amount unit
    if A in ("pmol.o2", "nmol.o2", "umol.o2", "mmol.o2", "mol.o2"):
        RO2 = convert_DO(rate, oxy, "mmol/L", S, t, P)
        RO2 = adjust_scale(RO2, "mmol.o2", A)
    elif A in ("mg.o2", "ug.o2"):
        RO2 = convert_DO(rate, oxy, "mg/L", S, t, P)
        RO2 = adjust_scale(RO2, "mg.o2", A)
    elif A in ("mL.o2", "uL.o2"):
        RO2 = convert_DO(rate, oxy, "mL/L", S, t, P)
        RO2 = adjust_scale(RO2, "mL.o2", A)
    elif A == "cm3.o2":
        RO2 = convert_DO(rate, oxy, "cm3/L", S, t, P)
    elif A == "mm3.o2":
        RO2 = convert_DO(rate, oxy, "cm3/L", S, t, P) * 1000.0
    else:
        raise ValueError(f"convert_rate.ft: output amount unit '{A}' not "
                         "recognised.")

    time_component = flow_unit_parse(flow_clean, "time")
    RO2 = adjust_scale(RO2, time_component, B)
    vol_component = flow_unit_parse(flow_clean, "vol")
    volume = adjust_scale(1.0, vol_component, "L.vol")
    VO2 = RO2 * volume

    if is_mass_spec:
        multm = adjust_scale(mass, "kg.mass", C)
        rate_m_spec = VO2 / multm
        rate_a_spec = np.full(rate.size, np.nan)
        VO2_out = rate_m_spec
    elif is_area_spec:
        multm = adjust_scale_area(area, "m2.area", C)
        rate_m_spec = np.full(rate.size, np.nan)
        rate_a_spec = VO2 / multm
        VO2_out = rate_a_spec
    else:
        rate_m_spec = np.full(rate.size, np.nan)
        rate_a_spec = np.full(rate.size, np.nan)
        VO2_out = VO2

    n = rate.size
    summary = {"rate.input": rate,
               "oxy.unit": np.array([oxy_clean] * n),
               "flowrate.unit": np.array([flow_clean] * n),
               "mass": np.full(n, np.nan if mass is None else float(mass)),
               "area": np.full(n, np.nan if area is None else float(area)),
               "S": np.full(n, np.nan if S is None else float(S)),
               "t": np.full(n, np.nan if t is None else float(t)),
               "P": np.full(n, np.nan if P is None else float(P)),
               "rate.abs": VO2, "rate.m.spec": rate_m_spec,
               "rate.a.spec": rate_a_spec,
               "output.unit": np.array([output_clean] * n),
               "rate.output": VO2_out}
    full = dict(summ_ext)
    full.update(summary)

    df = x.get("dataframe") if is_obj else None
    out = {"call": None,
           "inputs": {"x": x, "oxy.unit": oxy_clean,
                      "flowrate.unit": flow_clean,
                      "output.unit": output_clean, "mass": mass, "area": area,
                      "S": S, "t": t, "P": P},
           "dataframe": df, "data": x.get("data") if is_obj else None,
           "subsets": x.get("subsets") if is_obj else None,
           "delta.oxy": x.get("delta.oxy") if is_obj else None,
           "input_type": input_type, "summary": full,
           "rate.input": rate, "output.unit": output_clean,
           "rate.output": VO2_out}
    return out
