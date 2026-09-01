"""resprpy -- Python port of the respR R package.

Processing and analysis of respirometry data: inspect data, calculate
oxygen uptake rates (manually or automatically), correct for background
respiration, convert units, and estimate critical oxygen tensions.

Ported from respR 2.3.4 (Harianto & Carey, Methods in Ecology and
Evolution 2019, doi:10.1111/2041-210X.13162). License: GPL-3.

The public API mirrors respR's flat namespace: only respR 2.3.4's own
exported functions are available directly from resprpy (e.g. resprpy.calc_rate,
resprpy.format_time). Functions from respR's dependencies (e.g. the marelac
physics helpers used by convert_DO) stay internal (resprpy._marelac).
"""
from ._units import unit_args
from .adjust import adjust_rate, adjust_rate_ft
from .auto import AutoRate, auto_rate
from .calc import CalcRate, calc_rate
from .convert import (ConvertDO, ConvertRate, convert_DO, convert_MR,
                      convert_rate, convert_val)
from .convertft import convert_rate_ft
from .data import select, sim_data, subsample, subset_data
from .importers import import_file
from .inspectmod import inspect, inspect_ft
from .intflow import auto_rate_int, calc_rate_bg, calc_rate_ft, calc_rate_int
from .oxycrit import oxy_crit

try:  # optional plotting
    from .plots import (plot_inspect, plot_calc_rate, plot_auto_rate,
                        plot_oxy_crit)
    _PLOT_AVAILABLE = True
except ImportError:  # matplotlib not installed
    plot_inspect = plot_calc_rate = plot_auto_rate = plot_oxy_crit = None
    _PLOT_AVAILABLE = False
from .selectrate import select_rate, select_rate_ft, test_lin
from .timefmt import format_time

__version__ = "0.1.0"

__all__ = [
    "adjust_rate", "adjust_rate_ft",
    "auto_rate", "auto_rate_int", "AutoRate",
    "calc_rate", "calc_rate_bg", "calc_rate_ft", "calc_rate_int", "CalcRate",
    "convert_DO", "convert_MR", "convert_rate", "convert_rate_ft",
    "convert_val", "ConvertDO", "ConvertRate",
    "format_time",
    "import_file",
    "inspect", "inspect_ft",
    "oxy_crit",
    "plot_auto_rate", "plot_calc_rate", "plot_inspect", "plot_oxy_crit",
    "select", "select_rate", "select_rate_ft",
    "sim_data", "subsample", "subset_data",
    "test_lin",
    "unit_args",
]
