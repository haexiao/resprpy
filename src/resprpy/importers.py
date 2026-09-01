"""import_file: import respirometry instrument files.

Ported from respR 2.3.4 (deprecated in respR; kept for API completeness).
Includes all instrument parsers: Pyro Workbench, Firesting-Pyro, MiniDOT,
AutoResp/Witrox, Vernier (csv/txt/qmbl raw), PreSens OXY10/OXY4/OxyView/
Generic/Datamanager, NeoFox.
"""
from __future__ import annotations

import csv
import html
import io
import os
import re

import numpy as np


# ---------------------------------------------------------------------------
# minimal fread replacement (comma/tab/semicolon, fill=TRUE semantics)
# ---------------------------------------------------------------------------
def _detect_delim(sample_lines):
    best, best_n = None, -1
    for d in (",", "\t", ";"):
        n = max((l.count(d) for l in sample_lines[:50]), default=0)
        if n > best_n:
            best, best_n = d, n
    return best or ","


def _fread(path, skip=0, nrows=None, header="auto", fill=True, sep=None,
           dec=".", na_strings=("", "NA"), col_classes=None):
    """Read a table like data.table::fread (subset of behaviours)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    if sep is None:
        sep = _detect_delim(lines)
    if isinstance(skip, str):
        for i, l in enumerate(lines):
            if skip in l:
                skip = i
                break
        else:
            skip = 0
    rows = lines[skip:]
    # apply nrows to NON-BLANK lines (R fread blank.lines.skip applies after
    # skip, so nrows=1 reads the first non-blank line)
    if nrows is not None:
        kept = []
        for l in rows:
            if l.strip() == "":
                continue
            kept.append(l)
            if len(kept) >= nrows:
                break
        rows = kept
    parsed = []
    maxc = 0
    for l in rows:
        l = l.rstrip("\r\n")
        # R fread fill=TRUE keeps blank lines as NA rows but blank.lines.skip
        # still skips truly empty lines; skipping them keeps row numbering
        # aligned with what respR's parsers expect.
        if l.strip() == "":
            continue
        row = l.split(sep) if sep else re.split(r"[,;\t]", l)
        if dec == ",":
            row = [c.replace(",", ".") if _looks_num(c.replace(",", "."))
                   else c for c in row]
        parsed.append(row)
        maxc = max(maxc, len(row))
    if fill:
        parsed = [r + [""] * (maxc - len(r)) for r in parsed]
    if not parsed:
        return np.zeros((0, 0)), []
    arr = np.array(parsed, dtype=object)
    ncol = arr.shape[1]
    if header == "auto":
        header = True
        first = arr[0]
        # fread auto-header: first row all character, not all blank
        if all(str(c).strip() == "" for c in first):
            header = False
        else:
            try:
                floats = [float(str(c).replace(",", ".")) for c in first]
                if all(np.isfinite(floats)):
                    header = False
            except ValueError:
                header = True
    if header:
        names = [str(c).strip() for c in arr[0]]
        arr = arr[1:]
    else:
        names = [f"V{i + 1}" for i in range(ncol)]
    # type conversion: numeric columns -> float, otherwise keep raw strings
    out = np.zeros((arr.shape[0], ncol), dtype=object)
    for c in range(ncol):
        if col_classes and f"V{c + 1}" in col_classes and \
                col_classes[f"V{c + 1}"] == "character":
            out[:, c] = arr[:, c]
            continue
        col = arr[:, c]
        nums = []
        ok = True
        for v in col:
            s = str(v).strip()
            if s in na_strings or s == "":
                nums.append(np.nan)
                continue
            try:
                nums.append(float(s.replace(",", ".")))
            except ValueError:
                ok = False
                break
        if ok:
            out[:, c] = nums
        else:
            out[:, c] = col
    return out, names


def _looks_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _get_dec(path):
    """Detect decimal separator ('.' or ',') from the last rows."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    tail = lines[-10:]
    def count_numeric(d):
        n = 0
        for l in tail:
            for c in re.split(r"[,;\t]", l.rstrip("\r\n")):
                if _looks_num(c.replace(d, ".")):
                    n += 1
        return n
    pnt = count_numeric(".")
    com = count_numeric(",")
    return "." if pnt > com else ","


def _clean_nms(nms, keep_pct=False):
    """R gsub chains used by parsers to clean column names."""
    out = []
    for n in nms:
        s = str(n)
        s = s.replace("%", "perc")
        s = re.sub(r"[^A-Za-z0-9///' ]", "", s)
        s = s.replace("/", " ")
        s = s.strip()
        s = re.sub(r"  +", " ", s)
        out.append(s)
    return out


def _make_unique(names):
    seen = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}.{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


# ---------------------------------------------------------------------------
# instrument parsers
# ---------------------------------------------------------------------------
def _parse_oxy10(path, dec="."):
    data, names = _fread(path, skip=37, header=True, dec=dec)
    names = [re.sub(r"[^A-Za-z0-9///' ]", "", n.replace("%", "perc")).replace(
        "/", " ") for n in names]
    return data, names


def _parse_oxy4(path, dec="."):
    return _parse_oxy10(path, dec)


def _parse_presens(path, dec="."):
    data0, _ = _fread(path, header=False, dec=dec)
    rowstart = None
    for i in range(data0.shape[0]):
        if re.match(r"^Date/", str(data0[i, 0])):
            rowstart = i + 1
            break
    if rowstart is None:
        raise ValueError("import_file: could not locate data start.")
    data, _ = _fread(path, skip=rowstart, dec=dec)
    nms, _ = _fread(path, skip=rowstart - 1, nrows=1, header=False, dec=dec)
    names = _clean_nms(nms[0])
    names = [re.sub(r"  +", " ", n.strip()).replace(" ", "_") for n in names]
    if len(names) < data.shape[1]:
        names = names + [f"V{i + 1}" for i in range(len(names), data.shape[1])]
    if "NA" in names:
        idx = names.index("NA")
        names = names[:idx] + names[idx + 1:]
        data = np.delete(data, idx, axis=1)
    return data, names[:data.shape[1]]


def _parse_oxyview_txt(path, dec="."):
    data0, _ = _fread(path, dec=dec, header=False)
    rowstart = None
    for i in range(data0.shape[0]):
        if re.search(r"date\(", str(data0[i, 0])):
            rowstart = i + 1
            break
    if rowstart is None:
        raise ValueError("import_file: could not locate data start.")
    data, _ = _fread(path, skip=rowstart - 1, dec=dec)
    nms = [str(c) for c in data0[rowstart - 1, :]]
    names = _clean_nms(nms)
    names = [re.sub(r"  +", "", n.strip()).replace(" ", "_") for n in names]
    if "NA" in names:
        idx = names.index("NA")
        names = names[:idx] + names[idx + 1:]
        data = np.delete(data, idx, axis=1)
    return data, names[:data.shape[1]]


def _parse_oxyview_csv(path, dec="."):
    data0, _ = _fread(path, dec=dec, header=False)
    rowstart = None
    for i in range(data0.shape[0]):
        if re.search(r"date\(", str(data0[i, 0])):
            rowstart = i + 1
            break
    if rowstart is None:
        raise ValueError("import_file: could not locate data start.")
    data, _ = _fread(path, skip=rowstart - 1, dec=dec)
    nms = [str(c) for c in data0[rowstart - 1, :]]
    names = _clean_nms(nms)
    names = [re.sub(r"  +", "", n.strip()).replace(" ", "_") for n in names]
    if "NA" in names:
        idx = names.index("NA")
        names = names[:idx] + names[idx + 1:]
        data = np.delete(data, idx, axis=1)
    return data, names[:data.shape[1]]


def _parse_autoresp_witrox(path, dec="."):
    data0, _ = _fread(path, dec=dec, header=False)
    rowstart = None
    for i in range(data0.shape[0]):
        if re.match(r"Date", str(data0[i, 0])):
            rowstart = i + 1
    if rowstart is None:
        raise ValueError("import_file: could not locate data start.")
    data, _ = _fread(path, skip=rowstart, dec=dec,
                     col_classes={"V2": "character"})
    nms, _ = _fread(path, skip=rowstart - 1, nrows=1, header=False, dec=dec)
    names = _clean_nms([str(c) for c in nms[0]])
    names = [re.sub(r"  +", "", n.strip()).replace(" ", "_") for n in names]
    names = _make_unique(names)
    return data, names[:data.shape[1]]


def _parse_minidot(path, dec="."):
    data0, _ = _fread(path, dec=dec, header=False)
    rowstart = None
    for i in range(data0.shape[0]):
        if "Unix" in str(data0[i, 0]):
            rowstart = i + 1
            break
    if rowstart is None:
        raise ValueError("import_file: could not locate data start.")
    data, _ = _fread(path, skip=rowstart, dec=dec)
    nms3, _ = _fread(path, skip=rowstart - 1, nrows=3, header=False, dec=dec)
    nms1 = [str(c) for c in nms3[1, :]]
    nms2 = [str(c) for c in nms3[0, :]]
    names = [f"{a} {b}".strip() for a, b in zip(nms1, nms2)]
    return data, names[:data.shape[1]]


def _parse_datamanager(path, dec="."):
    data, names = _fread(path, skip=1, dec=dec)
    return data, names


def _parse_vernier_csv(path, dec="."):
    data, names = _fread(path, header=True, dec=dec)
    return data, names


def _parse_neofox(path, dec="."):
    data, names = _fread(path, dec=dec)
    return data, names


def _parse_workbench(path, dec="."):
    data, _ = _fread(path, skip=72, header=False, dec=dec)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    header_line = lines[71].rstrip("\r\n") if len(lines) >= 72 else ""
    names = header_line.split("\t")
    return data, names[:data.shape[1]]


def _parse_pyro(path, dec="."):
    data0, _ = _fread(path, dec=dec, header=False,
                      na_strings=("", "NA", "---"))
    settings = [i for i in range(data0.shape[0])
                if "Settings" in str(data0[i, 0])]
    if len(settings) > 1:
        raise ValueError("import_file: Data file appears to have multiple "
                         "datasets. Import stopped.")
    raw_sub = data0[:50]
    rowstart = None
    for i in range(raw_sub.shape[0]):
        row = [str(v) for v in raw_sub[i, :]]
        if any(re.search(r"Ch ?\d", v) for v in row):
            rowstart = i + 1
            break
    if rowstart is None:
        raise ValueError("import_file: could not locate channel header.")
    r0 = raw_sub[rowstart - 2] if rowstart >= 2 else raw_sub[0]
    r1 = raw_sub[rowstart - 1]
    ch1_locs = [j for j in range(len(r1)) if str(r1[j]) == "Ch1"]
    ch1_locs += [j for j in range(len(r1)) if str(r1[j]) == "Ch 1"]
    ch1_locs = sorted(set(ch1_locs))
    col_nms = [str(v) for v in r0]
    for j in ch1_locs:
        seg = [str(v) for v in r0[j:j + 4]]
        seg = [v for v in seg if v != "" and v != "nan"]
        ch_type = " ".join(sorted(set(seg)))
        for k in range(j, min(j + 4, len(col_nms))):
            col_nms[k] = ch_type
    names = _clean_nms(col_nms)
    names = [n.replace('c("', "").replace("c(NA, ", "").replace("xb0", "")
             for n in names]
    names = [n.replace("'", " ").strip() for n in names]
    names = [re.sub(r"  +", " ", n).replace(" ", "_") for n in names]
    names = [n[:19] if n.startswith("Advanced_") else n for n in names]
    data, _ = _fread(path, skip=rowstart, header=False, dec=dec,
                     na_strings=("", "NA", "---"))
    return data, _make_unique(names)[:data.shape[1]]


def _parse_vernier_txt(path, dec="."):
    data, _ = _fread(path, skip=7, header=False, dec=dec)
    if data.shape[1] > 0 and np.all(np.isnan(data[:, -1])):
        data = data[:, :-1]
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        all_lines = f.readlines()
    meta_index = [i for i, l in enumerate(all_lines) if "Vernier" in l]
    runs = []
    for mi in meta_index:
        chunk = all_lines[mi:mi + 8]
        runs.append(chunk[2].strip() if len(chunk) > 2 else "")
    cols, _ = _fread(path, nrows=7, header=False, dec=dec)
    if cols.shape[1] > 0 and np.all(np.isnan(cols[:, -1])):
        cols = cols[:, :-1]
    col_nms = []
    for i in range(cols.shape[1]):
        col_nms.append(f"{cols[0, i]} ({cols[2, i]})")
    col_nms = [str(c) for c in col_nms]
    # assemble multi-run blocks
    if any("Vernier" in l for l in all_lines):
        seq = [i + 1 for i, l in enumerate(all_lines) if "Vernier" in l]
        seq = sorted([s - 1 for s in seq] + [s + 7 for s in seq])
        seq = [1] + seq + [len(all_lines)]
        pairs = [[seq[i], seq[i + 1]] for i in range(0, len(seq) - 1, 2)]
        nrows = max(p[1] - p[0] + 1 for p in pairs)
        blocks = []
        for p in pairs:
            sub = data[p[0] - 1:p[1]]
            if sub.shape[0] < nrows:
                pad = np.full((nrows - sub.shape[0], sub.shape[1]), np.nan)
                sub = np.vstack([sub, pad])
            blocks.append(sub)
        data = np.hstack(blocks)
    n_blocks = data.shape[1] // len(col_nms) if col_nms else 1
    all_col_nms = col_nms * max(1, n_blocks)
    tagged = []
    for i in range(len(all_col_nms)):
        run_i = i // len(col_nms) if col_nms else 0
        run = runs[run_i] if run_i < len(runs) else ""
        tagged.append(f"{run}: {all_col_nms[i]}")
    return data, tagged


def _parse_vernier_raw(path, dec="."):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    lines = text.splitlines()
    if len(lines) > 1 and any("\t" in l or "," in l for l in lines):
        data, names = _fread(path, dec=dec)
        return data, names
    # qmbl XML-ish: collapse rows
    data_l = [l.replace(",", "") for l in lines]
    data_l = [re.sub(r"NA NA", "", l) for l in data_l]
    # locate text blocks
    str_i = [i for i, l in enumerate(data_l) if "<TextText>" in l]
    enr_i = [i for i, l in enumerate(data_l) if "</TextText>" in l]
    if str_i and enr_i:
        s, e = str_i[0], enr_i[0]
        for i in range(s, e + 1):
            data_l[i] = "<tmp>"
    meta_index = [i for i, l in enumerate(data_l) if "<" in l]
    data_index = [i for i in range(len(data_l)) if i not in meta_index]
    # numeric data blocks
    blocks = []
    cur = []
    for i in data_index:
        try:
            v = float(data_l[i].strip())
            cur.append(v)
        except ValueError:
            if cur:
                blocks.append(cur)
                cur = []
    if cur:
        blocks.append(cur)
    runs = [re.sub(r"</?DataSetName>", "", l).rstrip() for l in data_l
            if "<DataSetName>" in l]
    meta = [l for l in data_l if "<MBLChannelIndex>" in l]
    channels = [re.sub(r"</?MBLChannelIndex>", "", l).rstrip() for l in meta]
    channels = [None] + channels
    n_runs = max(1, len(runs))
    n_channels = len(channels)
    n_per = max(1, len(blocks) // max(1, n_runs))
    if n_channels != n_per:
        channels = [None] * len(blocks)
    def col_nm(idx):
        obj = [l for l in data_l if "<DataObjectName>" in l]
        un = [l for l in data_l if "<ColumnUnits>" in l]
        if idx < len(obj) and idx < len(un):
            o = re.sub(r"</?DataObjectName>", "", obj[idx]).replace(" ", "")
            u = re.sub(r"</?ColumnUnits>", "", un[idx]).replace(" ", "")
            u = html.unescape(u)
            return f"{o} ({u})"
        return f"col{idx + 1}"
    col_nms = [col_nm(i) for i in range(n_per)]
    names = []
    for r in range(n_runs):
        for c in range(n_per):
            run = runs[r] if r < len(runs) else f"run{r + 1}"
            ch = channels[c] if c < len(channels) else None
            names.append(f"{run}: Ch.{ch}: {col_nms[c]}")
    nrf = max((len(b) for b in blocks), default=0)
    mat = np.full((nrf, len(blocks)), np.nan)
    for j, b in enumerate(blocks):
        mat[:len(b), j] = b
    return mat, names


# ---------------------------------------------------------------------------
# import_file
# ---------------------------------------------------------------------------
def import_file(path, export=False):
    if not os.path.exists(path):
        raise FileNotFoundError("import_file: File not found - please check "
                                "file path.")
    if ".xls" in path:
        raise ValueError("import_file: Excel file detected. Excel support has "
                         "been removed.")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.readlines()
    dec = _get_dec(path)
    ext = path[-4:]

    def head(n=20):
        return [l.lower() for l in raw[:n]]

    if any("workbench" in l for l in head()):
        data, names = _parse_workbench(path, dec)
    elif any("pyro" in l for l in head()):
        data, names = _parse_pyro(path, dec)
    elif any("minidot" in l for l in head()):
        data, names = _parse_minidot(path, dec)
    elif any("calibration data" in l for l in head()):
        data, names = _parse_autoresp_witrox(path, dec)
    elif ": time (" in raw[0].lower() if raw else False:
        data, names = _parse_vernier_csv(path, dec)
    elif any("vernier format" in l for l in head()):
        data, names = _parse_vernier_txt(path, dec)
    elif any("qmbl" in l for l in head()):
        data, names = _parse_vernier_raw(path, dec)
    elif any("oxy10" in l for l in head()):
        data, names = _parse_oxy10(path, dec)
    elif any("oxyview" in l for l in [x.lower() for x in raw[:100]]) and \
            ext == ".txt":
        data, names = _parse_oxyview_txt(path, dec)
    elif any("oxyview" in l for l in [x.lower() for x in raw[:100]]) and \
            ext == ".csv":
        data, names = _parse_oxyview_csv(path, dec)
    elif any("oxy4" in l for l in head(100)):
        data, names = _parse_oxy4(path, dec)
    elif any("mux channel" in l for l in head(80)) and \
            any("parameters" in l for l in head(80)) and \
            any("firmware" in l for l in head(80)):
        data, names = _parse_presens(path, dec)
    elif "tau - phase method" in raw[0].lower() if raw else False:
        data, names = _parse_neofox(path, dec)
    elif any("fractional error" in l for l in head()):
        raise ValueError("import_file: AutoResp metadata file detected. "
                         "Currently these files are unsupported in respR.")
    elif any("presens datamanager" in l for l in head()):
        data, names = _parse_datamanager(path, dec)
    else:
        raise ValueError("import_file: Source file cannot be identified.")

    if data.shape[1] > len(names):
        names = names + [f"V{i + 1}" for i in range(len(names),
                                                    data.shape[1])]
    out = {"data": data, "colnames": names[:data.shape[1]]}
    if export:
        out_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                                "parsed-" + os.path.splitext(
                                    os.path.basename(path))[0] + ".csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(names[:data.shape[1]])
            for i in range(data.shape[0]):
                w.writerow(["" if np.isnan(v) else f"{v:.10g}"
                            for v in data[i]])
        out["exported"] = out_path
    return out
