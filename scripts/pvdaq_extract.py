"""Check and extract one PVDAQ system: metadata and measured AC power, ready for the bench.

    python scripts/pvdaq_extract.py 1422                  # all years on OEDI
    python scripts/pvdaq_extract.py 1422 --years 2016 2017

Metadata checks (from the OEDI metadata JSON, as in pvdaq_screen.py)
  fixed array, one orientation, tilt and azimuth recorded, module and inverter recorded,
  not a net meter; DC rating = module count x wattage in the module model name (else System.power)
Data checks (from the OEDI time series)
  AC power column(s) found; sampling interval; samples per year; duplicate timestamps;
  negative values (count and energy); night offset; units (W or kW, against the DC rating);
  DST signature (duplicated or missing local hours); timestamp offset against clear-sky GHI,
  separately for winter and summer, for a fixed standard-time zone; outage days and scorable
  daylight hours per year, with the bench's own rules (bench.clean_days, bench.best_lag)

Writes to data/validation/PVDAQ/<sid>/
  <sid>_metadata.json       raw metadata JSON
  <sid>_raw_<year>.parquet  extracted time column and AC column(s), per year (cache)
  <sid>_hourly.parquet      timestamp (hour start, local logger time), pv_power_hourly_kW,
                            raw_samples_in_hour, system_id
  <sid>_report.md           PASS / WARN / FAIL per check, per-year table, and a draft
                            [location], [array], [data] and [measured] block for config/site/

Nothing here uses a model output: the timestamp check uses clear-sky GHI only, as the bench
does. Needs network access to OEDI, s3fs, pyarrow, pvlib and the solarmodel package.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solarmodel import paths  # noqa: E402

ROOT = paths.PROJECT_ROOT                  # the root bench resolves configs and measured files against
_spec = importlib.util.spec_from_file_location("inv", Path(__file__).parent / "pvdaq_sensor_inventory.py")
inv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inv)

TIME_COLS = ("measured_on", "timestamp", "time", "datetime", "date_time", "Timestamp")
STATUS = {True: "PASS", False: "FAIL", None: "WARN"}
_ov = paths.CONFIG_DIR / "pvdaq_overrides.json"
OVERRIDES = {k: v for k, v in json.loads(_ov.read_text()).items() if not k.startswith("_")} if _ov.exists() else {}


def check_metadata(sid: int, name: str, meta: dict) -> tuple[dict, list[tuple]]:
    mounts = list((meta.get("Mount") or {}).values())
    mods = list((meta.get("Modules") or {}).values())
    invs = list((meta.get("Inverters") or {}).values())
    site = inv.site_of(meta)
    planes = sorted({(inv.num(m.get("tilt")), inv.num(m.get("azimuth"))) for m in mounts}, key=str)
    tracks = [inv.tracking(m.get("tracking")) for m in mounts]

    # DC rating: quantity x module power. PVDAQ module records carry no power field, so the
    # wattage is read from the model name (sw245, spr-315e, opt270-60, nu-u240f1, 280W); a name
    # with no single number in 50..700 W, or a quantity of 1, falls back to System.power.
    dc_w, dc_ok = 0.0, bool(mods)
    for m in mods:
        p = next((inv.num(m.get(k)) for k in ("nameplate_power", "power", "rated_power", "pmax", "stc_power",
                                              "module_power", "watts") if inv.num(m.get(k))), None)
        model = str(m.get("model", ""))
        if p is None and "/" not in model:
            w = [int(x) for x in re.findall(r"\d+", model) if 50 <= int(x) <= 700]
            p = w[0] if w else None
        q = inv.num(m.get("quantity"))
        if p and q and q > 1:
            dc_w += p * q
        else:
            dc_ok = False
    sys_kw = inv.num((meta.get("System") or {}).get("power"))
    src = "module count x model-name wattage"
    if dc_ok and dc_w and sys_kw and abs(dc_w / 1000 / sys_kw - 1) > 0.03:
        src += f" (System.power says {sys_kw} kW)"
    if not (dc_ok and dc_w):
        dc_w, dc_ok, src = (1000 * sys_kw, True, "System.power (no module wattage)") if sys_kw else (0.0, False, "")
    info = dict(
        system_id=sid, name=name,
        lat=inv.num(site.get("latitude")), lon=inv.num(site.get("longitude")), elevation_m=inv.num(site.get("elevation")),
        planes=planes, tracking=tracks,
        modules=[{k: m.get(k) for k in m} for m in mods], inverters=[{k: i.get(k) for k in i} for i in invs],
        dc_kw=round(dc_w / 1000, 3) if dc_ok and dc_w else None, dc_source=src,
    )
    ov = OVERRIDES.get(str(sid), {})
    if ov.get("dc_kw"):
        info.update(dc_kw=float(ov["dc_kw"]), dc_source=f"override: {ov.get('source', 'config/pvdaq_overrides.json')}")
    if info["dc_kw"] is None and inv.num((meta.get("System") or {}).get("power")):
        info.update(dc_kw=inv.num(meta["System"]["power"]), dc_source="System.power (check against module count x datasheet)")
    mod_known = inv.module_known(mods)
    inv_known = any(str(i.get("manufacturer", "")).strip() not in ("", "Unknown", "None", "nan") for i in invs)
    checks = [
        ("metadata", "fixed array", all(t is False for t in tracks) if tracks else None, f"tracking={tracks}"),
        ("metadata", "one orientation", len(planes) == 1 if planes else None, f"(tilt, azimuth)={planes}"),
        ("metadata", "tilt and azimuth recorded", all(p[0] is not None and p[1] is not None for p in planes) if planes else False, ""),
        ("metadata", "module recorded (class A, else B)", mod_known or None, info["modules"][:1]),
        ("metadata", "inverter recorded", inv_known or None, [f"{i.get('manufacturer')} {i.get('model')}" for i in invs]),
        ("metadata", "not a net meter", "net meter" not in name.lower(), name),
        ("metadata", "DC rating", True if info["dc_source"] == "module count x model-name wattage" or info["dc_source"].startswith("override") else None, f"{info['dc_kw']} kW from {info['dc_source']}"),
        ("metadata", "coordinates", None not in (info["lat"], info["lon"]), f"{info['lat']}, {info['lon']}"),
    ]
    return info, checks


PARQUET = "{base}/parquet/pvdata/system_id={sid}"          # long: time, metric_id, value; day files
CSV = "{base}/csv/pvdata/system_id={sid}"                  # wide: one column per sensor, <name>__<metric_id>
PRIZE = "{base}/2023-solar-data-prize/{sid}_OEDI/data"     # wide, one folder


def _ls(path: str) -> list[str]:
    try:
        return inv.fs.ls(path)
    except (FileNotFoundError, OSError):
        return []


FIRST_YEAR = 1990   # PVDAQ starts in 1994; earlier year folders on OEDI hold bad timestamps (50 has year=1822)


def years_on_oedi(sid: int) -> dict[int, dict]:
    """{year: {"parquet": dir or None, "csv": dir or None}}; year 0 = Data Prize folder."""
    out: dict[int, dict] = {}
    for kind, pat in (("parquet", PARQUET), ("csv", CSV)):
        for d in _ls(pat.format(base=inv.BASE, sid=sid)):
            if (m := re.search(r"year=(\d{4})", d)) and FIRST_YEAR <= int(m.group(1)) <= pd.Timestamp.now().year:
                out.setdefault(int(m.group(1)), {"parquet": None, "csv": None})[kind] = d
    prize = PRIZE.format(base=inv.BASE, sid=sid)
    if not out and _ls(prize):
        out[0] = {"parquet": None, "csv": prize}
    return out


def ac_metrics(sid: int, ys: dict) -> dict[int, str]:
    """AC power metric ids of the system -> column name <sensor_name>__<metric_id>.
    From the PVDAQ metrics table; if that fails, from the header of one CSV day file."""
    out = {}
    try:
        m = inv.table("metrics")
        m = m[m.system_id == sid]
        for _, r in m.iterrows():
            label = f"{r.get('sensor_name', '')} {r.get('common_name', '')} {r.get('standard_name', '')}"
            if inv.AC_COL.search(str(r.get("sensor_name", ""))) or re.search(r"\bAC power\b", label, re.I):
                name = re.sub(r"\s*\(.*\)$", "", str(r.get("sensor_name", "ac_power"))).strip() or "ac_power"
                out[int(r["metric_id"])] = f"{name}__{int(r['metric_id'])}"
    except Exception as e:
        print(f"  metrics table not read: {e}", file=sys.stderr)
    if out:
        return out
    for y in sorted(ys):
        d = ys[y]["csv"]
        if not d:
            continue
        files = [f for f in inv.fs.find(d) if f.endswith(".csv")]
        if files:
            with inv.fs.open(files[len(files) // 2], "r") as fh:
                for c in fh.readline().strip().split(","):
                    c = c.strip().strip('"')
                    if inv.AC_COL.search(c) and (m_ := re.search(r"__(\d+)$", c)):
                        out[int(m_.group(1))] = c
            if out:
                return out
    return out


def _time_col(cols) -> str | None:
    return next((c for c in TIME_COLS if c in cols), None)


def _read_parquet_year(d: str, ids: dict[int, str]) -> pd.DataFrame:
    """One year of the long parquet table, filtered to the AC metrics, pivoted wide."""
    df = pd.read_parquet(d, filesystem=inv.fs, filters=[("metric_id", "in", list(ids))])
    if df.empty:
        return df
    tcol = _time_col(df.columns)
    if tcol is None or "value" not in df.columns:
        raise ValueError(f"unexpected parquet columns {list(df.columns)}")
    df["metric_id"] = pd.to_numeric(df["metric_id"], errors="coerce").astype("Int64")
    w = df.pivot_table(index=tcol, columns="metric_id", values="value", aggfunc="first")
    w.columns = [ids.get(int(c), f"ac_power__{int(c)}") for c in w.columns]
    return w.rename_axis("time").reset_index()


def _read_csv_dir(d: str) -> pd.DataFrame:
    """Daily wide CSV files of one year (or the Data Prize folder): time and AC columns only."""
    files = sorted(f for f in inv.fs.find(d) if f.endswith(".csv"))
    parts = []
    for i, f in enumerate(files):
        with inv.fs.open(f, "r") as fh:
            x = pd.read_csv(fh, low_memory=False)
        tcol = _time_col(x.columns)
        acs = [c for c in x.columns if inv.AC_COL.search(str(c))]
        if tcol and acs:
            parts.append(x[[tcol] + acs].rename(columns={tcol: "time"}))
        if i % 60 == 0:
            print(f"    {i + 1}/{len(files)} files", flush=True)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def extract_year(sid: int, year: int, dirs: dict, ids: dict[int, str], cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_parquet(cache)
    out = pd.DataFrame()
    if dirs["parquet"] and ids:
        try:
            out = _read_parquet_year(dirs["parquet"], ids)
        except Exception as e:
            print(f"    parquet read failed ({e}); trying CSV", file=sys.stderr)
    if out.empty and dirs["csv"]:
        out = _read_csv_dir(dirs["csv"])
    if out.empty:
        return out
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out.dropna(subset=["time"])
    for c in out.columns.drop("time"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out.to_parquet(cache)
    return out


NOT_P = re.compile(r"kvar|kva(?![a-z])|kwh|wh_|energy|pf(_|$)|_pf|phase.?rev|dc_|_max$|volt|curr", re.I)
PER_INV = re.compile(r"^inv\d+(_|$)|^ac_power_\d+(_[a-z]+)?$", re.I)     # not ac_power_metered_1_2 (a total)
INV_AC = re.compile(r"inv\w*?pac|inv\w*?ac_p", re.I)


def _segments(L: pd.Series) -> list[pd.DatetimeIndex]:
    """Cut the daily log-level series at lasting steps of >= 0.8 decades (45-day medians either
    side); segments under 150 valid days or with under 30 valid Apr-Sep days are merged into a
    neighbour, so snow cover and outages never form a segment of their own."""
    before = L.rolling(45, min_periods=15).median().shift(1)
    after = L[::-1].rolling(45, min_periods=15).median()[::-1]
    jump = (after - before).abs()
    cand = jump >= 0.8
    cuts = []
    for _, g in jump[cand].groupby((~cand).cumsum()[cand]):
        cuts.append(g.idxmax())
    edges = [L.index[0]] + cuts + [L.index[-1] + pd.Timedelta(days=1)]
    segs = [L.index[(L.index >= e0) & (L.index < e1)] for e0, e1 in zip(edges[:-1], edges[1:])]
    ok = lambda ix: L[ix].notna().sum() >= 150 and L[ix][L[ix].index.month.isin(range(4, 10))].notna().sum() >= 30
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, ix in enumerate(segs):
            if not ok(ix):
                j = i - 1 if i > 0 else i + 1
                lo, hi = sorted((i, j))
                segs[lo:hi + 1] = [segs[lo].append(segs[hi])]
                changed = True
                break
    return segs


def harmonise(v: pd.Series, t: pd.Series, ref_kw: float | None) -> tuple[pd.Series, list[str]]:
    """One power column -> kW. Splits the record at lasting level steps (logger unit changes),
    scales each segment by the power of ten that puts its 99th-percentile daily peak at 0.9 x the
    expected peak (ref_kw), and sets a segment without a solar daily shape to missing. A factor
    that differs between segments, or that is not W -> kW, is reported."""
    s = pd.Series(v.to_numpy(float), index=pd.DatetimeIndex(t))
    notes = []
    # days whose night (00-03 h) sits near the day's peak are not AC power (stuck registers, other quantities)
    dk = s.index.floor("D")
    dmx = s.groupby(dk).max()
    ngt = s[s.index.hour < 3].groupby(dk[s.index.hour < 3]).median().reindex(dmx.index)
    lvl99 = float(dmx[dmx > 0].quantile(0.9)) if (dmx > 0).any() else 0.0
    bad = dmx.index[(ngt > 0) & (ngt > 0.3 * dmx) & (ngt > 0.05 * lvl99)]
    if len(bad):
        s = s.where(~dk.isin(bad))
        notes.append(f"{len(bad)} days with a night level near the day's peak (not AC power), set missing, "
                     f"{bad[0].date()} to {bad[-1].date()}")
    pos = s.where(s > 0)
    day = pos.groupby(pos.index.floor("D")).max()
    if day.notna().sum() < 30:
        return s, notes
    day = day.asfreq("D")
    L = np.log10(day.rolling(15, center=True, min_periods=5).median())
    segs = _segments(L)
    dkey = s.index.floor("D")
    # boundaries: each day within 20 days of a cut goes to the neighbour whose level it is nearer
    lv = np.log10(day)
    med = [float(lv[ix].median()) for ix in segs]
    lab = pd.Series(0, index=day.index)
    for i, ix in enumerate(segs):
        lab[ix] = i
    for i in range(len(segs) - 1):
        if abs(med[i] - med[i + 1]) < 0.8:
            continue
        c = segs[i + 1][0]
        w = lv[(lv.index >= c - pd.Timedelta(days=20)) & (lv.index < c + pd.Timedelta(days=20))].dropna()
        lab[w.index] = np.where((w - med[i]).abs() <= (w - med[i + 1]).abs(), i, i + 1)
    segs = [lab.index[lab == i] for i in range(len(segs))]
    # daily shape: mean over days of each day's profile (negatives as 0) normalised by its own peak,
    # so every day has equal weight and night noise cannot dominate
    c0 = s.clip(lower=0)
    m0 = c0.groupby(dkey).transform("max")
    dn = c0 / m0.where(m0 > 0)
    hour = s.index.hour
    prof = lambda ix: dn[dkey.isin(ix)].groupby(hour[dkey.isin(ix)]).mean()
    main = max(segs, key=lambda ix: L[ix].notna().sum())
    p0 = prof(main)
    fac = pd.Series(1.0, index=day.index)
    ks = []
    for ix in segs:
        if not len(ix):
            continue
        span = f"{ix[0].date()} to {ix[-1].date()}"
        q = float(day[ix].quantile(0.75))
        pr = prof(ix)
        r = 1.0 if ix is main else pr.corr(p0)
        contrast = float(pr.max() - pr.min()) if len(pr) else 0.0
        if not q > 0 or not (r >= 0.6) or contrast < 0.15:
            fac[ix] = np.nan
            notes.append(f"{span}: no solar daily shape (r={r:.2f}, day/night contrast {contrast:.2f}), set missing")
            continue
        seg = s[dkey.isin(ix)]
        night = seg[(seg.index.hour >= 23) | (seg.index.hour < 3)]
        nf = abs(float(night.median())) / q if len(night) > 100 else 0.0
        if nf > 0.1:
            fac[ix] = np.nan
            notes.append(f"{span}: night level {nf:.2f} x the daytime peak, not AC power, set missing")
            continue
        k = int(np.round(np.log10(0.8 * ref_kw / q))) if ref_kw else (-3 if q > 2000 else 0)
        fac[ix] = 10.0 ** k
        ks.append((span, k))
    if ks:
        uniq = {k for _, k in ks}
        if len(uniq) > 1:
            runs = []
            for span, k in sorted(ks):
                if runs and runs[-1][2] == k:
                    runs[-1][1] = span.split(" to ")[1]
                else:
                    runs.append([span.split(" to ")[0], span.split(" to ")[1], k])
            notes.extend(f"{a_} to {b_}: x{10.0 ** k:g} (unit change)" for a_, b_, k in runs)
        elif ks[0][1]:
            notes.append(f"x{10.0 ** ks[0][1]:g} to kW" + ("" if ks[0][1] == -3 else " (not W: check)"))
    return s * fac.reindex(dkey).to_numpy(), notes


GRID = "5min"
METER = re.compile(r"(^|_)meter_?\d+(_|$)", re.I)


def pick_ac(df: pd.DataFrame, dc: float | None = None) -> tuple[pd.Series, str, list[str]]:
    """One whole-system AC series in kW on a 5-minute grid.

    Every column is converted to kW on its own (harmonise) and averaged onto the grid, so
    loggers that write at different timestamps line up. Candidates are each total column, the
    sum of the inverter columns and the sum of the meter columns (sums only where every member
    has a value). Month by month, a candidate is whole-system if its peak level is at least
    0.9 x the highest candidate's; the most complete whole-system candidate is used and the others
    fill its gaps. Apr-Oct months below 0.6 x the usual level for the calendar month (part of the
    array down) are set missing; Nov-Mar months are only reported (snow cover looks the same)."""
    base = {c: re.sub(r"__\d+$", "", c) for c in df.columns if c != "time"}
    df = pd.concat([df[["time"]]] + [df[[c for c in base if base[c] == b_]].bfill(axis=1).iloc[:, 0].rename(b_)
                                      for b_ in dict.fromkeys(base.values())], axis=1)
    acs = [c for c in df.columns if c != "time" and not NOT_P.search(c)] or [c for c in df.columns if c != "time"]
    inv = [c for c in acs if PER_INV.search(c)]
    met = [c for c in acs if METER.search(c) and c not in inv]
    tot = [c for c in acs if c not in inv and c not in met]
    if len(inv) == 1:
        tot, inv = tot + inv, []
    if len(met) == 1:
        tot, met = tot + met, []
    t0 = pd.DatetimeIndex(df["time"])
    if t0.tz is not None:
        t0 = t0.tz_localize(None)
    notes = []

    def grid(c, share):
        v = pd.Series(pd.to_numeric(df[c], errors="coerce").to_numpy(float), index=t0).dropna()
        v = v.groupby(v.index.floor(GRID)).mean()                       # onto the grid first: less memory
        s, n = harmonise(v, pd.Series(v.index), dc * share if dc else None)
        notes.extend(f"{c}: {x}" for x in n)
        return s.dropna()

    df = df.drop(columns=[c for c in df.columns if c != "time" and c not in acs])
    cands = {c: grid(c, 1.0) for c in tot}
    for group in (inv, met):
        if len(group) >= 2:
            g = pd.concat([grid(c, 1 / len(group)) for c in group], axis=1)
            cands["sum(" + ", ".join(group) + ")"] = g.sum(axis=1, min_count=len(group))
    cands = {k: v.dropna() for k, v in cands.items() if v.notna().sum() > 0}
    if not cands:
        return pd.Series(dtype=float), "none", notes
    # month by month: a candidate is whole-system in a month if its peak level (90th percentile
    # of daily maxima) is at least 0.9 x the highest candidate's that month; the most complete
    # whole-system candidate is used, the others fill its gaps
    prio = sorted(cands, key=lambda k: -len(cands[k]))
    mon = {k: v.index.to_period("M") for k, v in cands.items()}
    lev = {k: v.groupby(v.index.floor("D")).max().pipe(lambda d: d.groupby(d.index.to_period("M")).quantile(0.9))
           for k, v in cands.items()}
    months = sorted(set().union(*[set(l.index) for l in lev.values()]))
    pieces, used, dropped, disagree = [], {k: 0 for k in prio}, {k: 0 for k in prio}, 0
    for m in months:
        lv = {k: float(lev[k].get(m, np.nan)) for k in prio}
        top_m = np.nanmax(list(lv.values()))
        if not top_m > 0:
            continue
        ok_k = [k for k in prio if lv[k] >= 0.9 * top_m]
        for k in prio:
            if pd.notna(lv[k]) and k not in ok_k:
                dropped[k] += 1
        if len(ok_k) >= 2 and min(lv[k] for k in ok_k) < 0.95 * top_m:
            disagree += 1
        sm = None
        for k in ok_k:
            part = cands[k][mon[k] == m]
            sm = part if sm is None else sm.combine_first(part)
            used[k] += 1
        pieces.append(sm)
    if not pieces:
        return pd.Series(dtype=float), "none", notes
    out = pd.concat(pieces).sort_index()
    # whole system down in part: month level below 0.6 x the median for that calendar month
    oc = out.where(out <= 1.3 * dc) if dc else out                     # spikes must not hide a low month
    ml = oc.groupby(oc.index.floor("D")).max().pipe(lambda d: d.groupby(d.index.to_period("M")).quantile(0.9))
    typical = ml.groupby(ml.index.month).transform("median")
    lowm = ml.index[ml < 0.6 * typical]
    low = [m for m in lowm if 4 <= m.month <= 10]                       # snow-free months only
    snow = [m for m in lowm if not 4 <= m.month <= 10]
    if low:
        out = out[~out.index.to_period("M").isin(low)]
        notes.append(f"{len(low)} Apr-Oct months below 0.6 x the usual level for the calendar month (part of the array down), "
                     f"set missing: " + ", ".join(str(m) for m in low[:24]) + (" ..." if len(low) > 24 else ""))
    if snow:
        notes.append(f"{len(snow)} Nov-Mar months below 0.6 x the usual level, kept (snow cover or outage; the bench's "
                     f"outage rule decides): " + ", ".join(str(m) for m in snow[:24]) + (" ..." if len(snow) > 24 else ""))
    desc = "per month, the most complete whole-system candidate of: " + "; ".join(
        f"{k} ({used[k]} months used" + (f", {dropped[k]} months partial" if dropped[k] else "") + ")" for k in prio)
    if disagree:
        notes.append(f"{disagree} months where whole-system candidates differ by 5-10 %")
    return out.sort_index(), desc, notes


def check_data(raw: pd.DataFrame, info: dict, sid: int) -> tuple[pd.DataFrame, list[tuple], pd.DataFrame, dict]:
    from solarmodel import bench
    import pvlib

    checks = []
    dc = info.get("dc_kw")
    p, col, unit_notes = pick_ac(raw, dc)
    s = p.astype(float)
    rf = info.get("_raw") or {}
    rt = pd.DatetimeIndex(raw["time"])
    raw_counts = rf.get("counts", pd.Series(1, index=rt).resample("h").count())
    raw_step = rf.get("step", rt.sort_values().to_series().diff().median())
    raw_dup = rf.get("dup", int(rt.duplicated().sum()))
    # DC: module count x wattage, unless the data rule it out (hourly p99.9 > 1.05 x) and System.power fits
    alt = re.search(r"System.power says ([\d.]+) kW", info.get("dc_source", ""))
    if dc and alt and not info["dc_source"].startswith("override"):
        pk = float(s[~s.index.duplicated()].resample("h").mean().quantile(0.999))
        sysk = float(alt.group(1))
        if pk > 1.05 * dc and pk <= 1.05 * sysk:
            info.update(dc_kw=sysk, dc_source=f"System.power (module records give {dc} kW, below the measured hourly peak {pk:.1f} kW)")
            dc = sysk
    for a_, b_, why in OVERRIDES.get(str(sid), {}).get("exclude", []):
        m = (s.index >= pd.Timestamp(a_)) & (s.index < pd.Timestamp(b_) + pd.Timedelta(days=1))
        s = s.where(~m)
        checks.append(("data", f"excluded {a_} to {b_} (override)", True, f"{int(m.sum())} samples: {why}"))
    dmax = s.groupby(s.index.floor("D")).max()
    zero = (dmax <= (1e-3 * dc if dc else 0)).astype(int)            # daily peak below 0.1 % of DC: no reading
    run_id = (zero.diff() != 0).cumsum()
    rl = zero.groupby(run_id).transform("sum")
    dead_days = dmax.index[(zero == 1) & (rl >= 30)]
    if len(dead_days):
        s = s.where(~s.index.floor("D").isin(dead_days))
        checks.append(("data", "dead logger (>= 30 days with a daily peak below 0.1 % of DC)", None,
                       f"{len(dead_days)} days set missing: {dead_days[0].date()} to {dead_days[-1].date()}"))
    step = s.index.to_series().diff().median()                      # grid step of the series used below
    checks.append(("data", "AC power column", True, col))
    checks.append(("data", "duplicate timestamps", raw_dup == 0 or None, f"{raw_dup} in the raw data (averaged onto the 5-min grid)"))
    checks.append(("data", "sampling interval", True, f"{raw_step} raw, averaged onto {GRID} bins"))

    # units: per column and period, onto the DC rating (pick_ac -> harmonise)
    kw = s
    unit = "kW (harmonised per column and period)"
    checks.append(("data", "units", True if dc and not any("check" in n or "unit change" in n or "left as is" in n
                                                         for n in unit_notes) else None,
                   ("; ".join(unit_notes) or "no conversion needed") + f"; max {kw.max():.2f} kW against DC {dc} kW"))

    # corrections: impossible samples -> NaN (then caught by the outage rule), standby negatives -> 0
    raw_kw = kw.copy()
    lim = 1.3 * dc if dc else None
    spike = (kw > lim) if lim else pd.Series(False, index=kw.index)
    deep = (kw < -0.05 * dc) if dc else pd.Series(False, index=kw.index)
    kw = kw.mask(spike | deep)
    standby = kw < 0
    kw = kw.mask(standby, 0.0)
    checks.append(("data", "corrections", True if not (spike.any() or deep.any()) else None,
                   f"{int(spike.sum())} samples above 1.3 x DC and {int(deep.sum())} below -5 % of DC set to NaN; "
                   f"{int(standby.sum())} standby negatives set to 0"))
    night = raw_kw[(raw_kw.index.hour < 3) | (raw_kw.index.hour >= 23)]
    neg = raw_kw[raw_kw < 0]
    off = float(night.median()) if len(night) else np.nan
    checks.append(("data", "night offset (raw; dark hours are set to 0 below)", True if pd.notna(off) and abs(off) < 0.002 * (dc or kw.max()) else None,
                   f"median night value {off:.4f} kW ({100 * off / (dc or kw.max()):.2f} % of DC)"))
    e_neg = float(-neg.sum() * (step / pd.Timedelta(hours=1))) if len(neg) else 0.0
    checks.append(("data", "negative values (net export?)", len(neg) == 0 or None,
                   f"{len(neg)} raw samples, {e_neg:.2f} kWh (before corrections)"))

    # DST signature: in local time with DST, one spring hour is missing and one autumn hour doubled
    cnt = raw_counts[raw_counts > 0]
    per_h = cnt.groupby(cnt.index.floor("D")).transform("median")      # the day's usual count (sampling rate varies by era)
    doubled = int((cnt > 1.5 * per_h).sum())
    checks.append(("data", "no DST signature (doubled local hours)", doubled == 0 or None,
                   f"{doubled} hours with about twice the samples; if > 0, the logger keeps DST"))

    # hourly means, hour-start labels, local logger time
    hourly = pd.DataFrame({"pv_power_hourly_kW": kw.resample("h").mean(),
                           "raw_samples_in_hour": raw_counts})
    hourly["raw_samples_in_hour"] = hourly["raw_samples_in_hour"].fillna(0).astype(int)
    hourly.loc[hourly["pv_power_hourly_kW"].isna(), "raw_samples_in_hour"] = 0
    hourly["system_id"] = sid

    # timestamp offset against clear-sky GHI for a fixed standard time (no DST), winter and summer
    tz = inv.standard_tz(info["lon"])
    h = hourly["pv_power_hourly_kW"].where(hourly["raw_samples_in_hour"] > 0)
    idx = h.index.tz_localize(tz) + pd.Timedelta(minutes=30)                 # hour centres
    cs = pvlib.location.Location(info["lat"], info["lon"], altitude=info.get("elevation_m") or 0).get_clearsky(idx)
    ghi = pd.Series(cs["ghi"].to_numpy(), index=h.index)
    # labels shifted against clear sky are moved onto fixed standard time per month and DST status
    # (below); lags() is the winter (Nov-Feb) / summer (May-Aug) check used afterwards
    iana = {f"Etc/GMT+{k}": v for k, v in ((5, "America/New_York"), (6, "America/Chicago"), (7, "America/Denver"),
                                            (8, "America/Los_Angeles"))}.get(tz)

    def lags(hh, sel_year=None):
        out = {}
        for season, months in inv.SEASONS:
            sel = hh.index.month.isin(months) & (hh.index.year == sel_year if sel_year else True)
            out[season] = bench.best_lag(hh[sel], ghi[sel], 4) if hh[sel].notna().sum() > 300 else None
        return out

    # shift per calendar month and DST status (US calendar of the site's zone); a block's own
    # estimate is used only if it beats the consensus of its neighbours by >= 0.01 in correlation
    loc = h.index.tz_localize(tz).tz_convert(iana).tz_localize(None) if iana else h.index
    dst = np.asarray((loc - h.index) == pd.Timedelta(hours=1))
    blk = pd.Series(h.index.to_period("M").astype(str), index=h.index).str.cat(pd.Series(np.where(dst, "D", "S"), index=h.index))

    def corr_tab(sel):
        hh, gg = h[sel], ghi[sel]
        sun = gg > 0
        if int((hh.notna() & sun).sum()) < 100:
            return None
        return {L: hh.shift(L)[sun].corr(gg[sun]) for L in range(-4, 5)}

    tabs = {k: corr_tab((blk == k).to_numpy()) for k in blk.unique()}
    order = sorted(tabs, key=lambda k: (k[-1], k[:-1]))                  # by status, then month
    est = {}
    for status in ("S", "D"):
        ks = [k for k in order if k.endswith(status)]
        best = {k: max(tabs[k], key=lambda L: -np.inf if pd.isna(tabs[k][L]) else tabs[k][L]) for k in ks if tabs[k]}
        kk = list(best)
        for i, k in enumerate(kk):
            nb = [best[x] for x in kk[max(0, i - 2):i] + kk[i + 1:i + 3]]
            cons = int(np.median(nb)) if nb else best[k]
            gain = tabs[k][best[k]] - (tabs[k][cons] if pd.notna(tabs[k][cons]) else -np.inf)
            est[k] = best[k] if best[k] == cons or gain >= 0.01 else cons
        for i, k in enumerate(ks):                                         # blocks without enough sun hours
            if k not in est:
                known = [x for x in kk]
                if known:
                    near = min(known, key=lambda x: abs(pd.Period(x[:-1]).ordinal - pd.Period(k[:-1]).ordinal))
                    est[k] = est[near]
                else:
                    est[k] = 0
    shift = blk.map(est).fillna(0).astype(int).to_numpy()
    if shift.any():
        hourly = hourly.set_axis(hourly.index + pd.to_timedelta(shift, unit="h"))
        hourly = hourly[~hourly.index.duplicated(keep="first")].sort_index()
        hourly = hourly.reindex(pd.date_range(hourly.index[0], hourly.index[-1], freq="h", name=hourly.index.name))
        hourly["raw_samples_in_hour"] = hourly["raw_samples_in_hour"].fillna(0).astype(int)
        hourly["system_id"] = sid
        runs = []
        for status, name in (("S", "standard time"), ("D", "DST")):
            ks = [k for k in order if k.endswith(status)]
            seq = []
            for k in ks:
                if seq and seq[-1][2] == est[k]:
                    seq[-1][1] = k[:-1]
                else:
                    seq.append([k[:-1], k[:-1], est[k]])
            if seq:
                runs.append(name + ": " + ", ".join(f"{a_}..{b_} {L:+d} h" for a_, b_, L in seq))
        checks.append(("data", "timestamp shift corrected (labels moved onto fixed standard time, per month)", True, "; ".join(runs)))
        h = hourly["pv_power_hourly_kW"].where(hourly["raw_samples_in_hour"] > 0)
        idx = h.index.tz_localize(tz) + pd.Timedelta(minutes=30)
        cs = pvlib.location.Location(info["lat"], info["lon"], altitude=info.get("elevation_m") or 0).get_clearsky(idx)
        ghi = pd.Series(cs["ghi"].to_numpy(), index=h.index)
    # months still off after the correction cannot be aligned: set missing; a winter or summer of a
    # year that still fails after that is set missing whole
    mper = h.index.to_period("M")
    cap_ = dc or float(h.quantile(0.999))
    darkbig = (ghi == 0) & (h > 0.02 * cap_)                          # production while the sun is down
    m_e = h.clip(lower=0).groupby(mper).sum()
    d_e = h.where(darkbig).groupby(mper).sum()
    dropm = [m for m in m_e.index if m_e[m] > 0 and d_e.get(m, 0) > 0.01 * m_e[m]]

    # dark hours (clear-sky GHI 0 at the hour centre) produce nothing: sensor offsets there are set to 0
    pw = hourly["pv_power_hourly_kW"]
    big = (ghi == 0) & (pw > 0.02 * (dc or float(pw.quantile(0.999))))
    dark = (ghi == 0) & pw.notna() & (pw != 0) & ~big
    e_dark = float(pw[dark].sum())
    hourly.loc[dark, "pv_power_hourly_kW"] = 0.0
    hourly.loc[big, "pv_power_hourly_kW"] = np.nan
    hourly.loc[big, "raw_samples_in_hour"] = 0
    h = hourly["pv_power_hourly_kW"].where(hourly["raw_samples_in_hour"] > 0)
    checks.append(("data", "dark hours: offsets set to 0, production set missing", True if not big.any() else None,
                   f"{int(dark.sum())} hours under 2 % of DC set to 0 ({e_dark:.1f} kWh net); {int(big.sum())} hours above it set missing"))

    mper = h.index.to_period("M")
    for m in sorted(set(mper)):
        tab = corr_tab(mper == m)
        if tab:
            bl = max(tab, key=lambda L: -np.inf if pd.isna(tab[L]) else tab[L])
            if bl != 0 and tab[bl] - (tab[0] if pd.notna(tab[0]) else -np.inf) >= 0.01:
                dropm.append(m)
    for y in sorted(set(h.index.year)):
        for season, months in inv.SEASONS:
            sel = (h.index.year == y) & h.index.month.isin(months) & ~mper.isin(dropm)
            if h[sel].notna().sum() > 300 and bench.best_lag(h[sel], ghi[sel], 4) != 0:
                dropm += [m for m in sorted(set(mper[sel])) if not corr_tab(mper == m) or m not in dropm]
    dropm = sorted(set(dropm))
    if dropm:
        gone = mper.isin(dropm)
        hourly.loc[gone, "pv_power_hourly_kW"] = np.nan
        hourly.loc[gone, "raw_samples_in_hour"] = 0
        h = hourly["pv_power_hourly_kW"].where(hourly["raw_samples_in_hour"] > 0)
        checks.append(("data", "months whose timestamps cannot be aligned, set missing", None, ", ".join(str(m) for m in dropm)))
    lag = lags(h)
    ly_all = {y: lags(h, y) for y in sorted(set(h.index.year))}
    bad_y = {y: ly for y, ly in ly_all.items() if any(v not in (0, None) for v in ly.values())}
    ok = lag["winter"] == 0 and lag["summer"] == 0 and not bad_y
    checks.append(("data", f"timestamp offset vs clear sky ({tz}, hour-start labels), every year", ok if None not in lag.values() else None,
                   f"best lag winter {lag['winter']} h, summer {lag['summer']} h"
                   + ("; years still off: " + ", ".join(f"{y} (winter {d_['winter']}, summer {d_['summer']})" for y, d_ in bad_y.items()) if bad_y else "; 0 h in every year")))
    if lag["winter"] in (-4, 4) or lag["summer"] in (-4, 4):
        checks.append(("data", "lag inside the search range", False, "optimum at the +-4 h bound: exclude, as Nevada 1367"))

    # per year: coverage and scorable daylight hours with the bench's outage rule
    cap = dc or float(h.quantile(0.999))
    hz, gz = h.copy(), ghi.copy()
    hz.index = gz.index = h.index.tz_localize(tz)                          # clean_days wants tz-aware
    clean = pd.Series(bench.clean_days(hz, gz, cap, tz).to_numpy(), index=h.index)
    day = (ghi > bench.DAYLIGHT_WM2)
    years = []
    for y, g in h.groupby(h.index.year):
        d = day[g.index]
        years.append(dict(year=y, samples=int(raw_counts[raw_counts.index.year == y].sum()),
                          daylight_hours=int(d.sum()),
                          with_value=int((g.notna() & d).sum()),
                          scorable=int((g.notna() & d & clean[g.index]).sum()),
                          outage_days=int((~clean[g.index]).groupby(g.index.date).any().sum()),
                          energy_kwh=round(float(g.sum()), 1),
                          max_kw=round(float(raw_kw[raw_kw.index.year == y].max()), 2),
                          negative_kwh=round(float(-raw_kw[(raw_kw.index.year == y) & (raw_kw < 0)].sum() * (step / pd.Timedelta(hours=1))), 1),
                          corrected_samples=int((spike | deep)[raw_kw.index.year == y].sum())))
    yt = pd.DataFrame(years)
    if len(yt):
        yt["scorable_pct"] = (100 * yt.scorable / yt.daylight_hours.where(yt.daylight_hours > 0)).round(1)
        best = yt.loc[yt.scorable_pct.idxmax()]
        if dc:
            bad = yt[(yt.max_kw > 1.3 * dc) | (yt.negative_kwh > 0.01 * yt.energy_kwh.clip(lower=1))]
            checks.append(("data", "years without spikes above 1.3 x DC or large negative energy (raw)", bad.empty or None,
                           "suspect years: " + ", ".join(f"{int(r.year)} (max {r.max_kw} kW, neg {r.negative_kwh} kWh)"
                                                         for r in bad.itertuples()) if len(bad) else ""))
        pk = h.groupby(h.index.year).quantile(0.999) / dc if dc else None
        if pk is not None:
            yt["p999_over_dc"] = yt.year.map(pk).round(2)
            off_ = yt[(yt.with_value > 500) & ((yt.p999_over_dc < 0.6) | (yt.p999_over_dc > 1.05))]
            checks.append(("data", "hourly peak (p99.9) between 0.6 and 1.05 x DC in every year", off_.empty or None,
                           ", ".join(f"{int(r.year)}: {r.p999_over_dc}" for r in off_.itertuples())))
        checks.append(("data", "a year with at least 80 % scorable daylight hours", bool(best.scorable_pct >= 80) or None,
                       f"best {int(best.year)}: {best.scorable_pct} %"))
    hourly = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq="h", name=hourly.index.name))
    hourly["raw_samples_in_hour"] = hourly["raw_samples_in_hour"].fillna(0).astype(int)
    hourly["system_id"] = sid
    return hourly, checks, yt, dict(tz=tz, unit=unit, ac_col=col, step=str(step))


def report(sid, info, checks, yt, d, out: Path) -> str:
    lines = [f"# PVDAQ {sid}: {info['name']}", "", "| part | check | result | detail |", "|---|---|---|---|"]
    lines += [f"| {a} | {b} | {STATUS[c]} | {str(e).replace('|', '/')} |" for a, b, c, e in checks]
    fails = [b for a, b, c, e in checks if c is False]
    warns = [b for a, b, c, e in checks if c is None]
    verdict = "FAIL" if fails else ("WARN" if warns else "PASS")
    lines += ["", f"**Verdict: {verdict}**" + (f" (fails: {', '.join(fails)})" if fails else "")
              + (f" (check: {', '.join(warns)})" if warns else ""), ""]
    if len(yt):
        lines += ["## Years", "", yt.to_markdown(index=False) if hasattr(yt, "to_markdown") else yt.to_string(index=False), ""]
        y = int(yt.loc[yt.scorable_pct.idxmax(), "year"])
    else:
        y = None
    tilt, az = (info["planes"][0] if info["planes"] else (None, None))
    lines += ["## Draft site file blocks (check every value; add [module], [thermal], [losses], [inverter])", "", "```toml",
              "[location]", f'name      = "PVDAQ {sid} - {info["name"]}"', f"lat       = {info['lat']}",
              f"lon       = {info['lon']}", f"altitude  = {info['elevation_m']}", "",
              "[array]", "tracking = false", f"tilt     = {tilt}", f"azimuth  = {az}", "albedo   = 0.20", "",
              "[system]", f"rated_capacity_kw = {info['dc_kw']}", "",
              "[data]", f'sa_start = "{y}-01-01"' if y else "# no usable year", f'sa_end   = "{y}-12-31"' if y else "", "",
              "[measured]", f'file             = "data/validation/PVDAQ/{sid}/{sid}_hourly.parquet"', 'time_col         = "timestamp"',
              'power_col        = "pv_power_hourly_kW"', 'power_unit       = "kW"', f'raw_power_col    = "{d["ac_col"]}"',
              f'tz               = "{d["tz"]}"   # fixed standard time; see the timestamp check',
              'timestamp_label  = "start"', "interval_minutes = 60", "manual_shift_hours = 0",
              "auto_lag_correction = true", "max_lag_hours    = 4", "```", ""]
    text = "\n".join(lines)
    (out / f"{sid}_report.md").write_text(text)
    return verdict


def write_hourly(hourly: pd.DataFrame, out: Path, sid: int) -> str:
    """Corrected hourly series: CSV, then parquet; the CSV is deleted only if the parquet
    reads back identical (index, columns, values)."""
    h = hourly.rename_axis("timestamp")
    csv, pq = out / f"{sid}_hourly.csv", out / f"{sid}_hourly.parquet"
    h.to_csv(csv)
    h.to_parquet(pq)
    back = pd.read_parquet(pq)
    from_csv = pd.read_csv(csv, index_col="timestamp", parse_dates=True)
    # pandas 3 parses CSV datetimes at a different resolution (us vs ns): compare on one unit
    for f in (back, from_csv):
        f.index = pd.DatetimeIndex(f.index).as_unit("ns")
    h = h.set_axis(pd.DatetimeIndex(h.index).as_unit("ns"))
    try:
        pd.testing.assert_frame_equal(back, h, check_freq=False)
        pd.testing.assert_frame_equal(back, from_csv.astype(back.dtypes.to_dict()), check_freq=False,
                                      check_exact=False, rtol=1e-9, atol=1e-12)
    except AssertionError as e:
        return f"parquet differs from CSV, CSV kept: {str(e).splitlines()[0]}"
    csv.unlink()
    return "parquet verified, CSV deleted"


def process(sid: int, years: list[int] | None = None, drop_raw: bool = False, from_cache: bool = False) -> dict:
    """Check, extract and correct one system; returns a summary row."""
    if inv.fs is None and not from_cache:
        raise RuntimeError("s3fs is needed: pip install s3fs")
    systems = pd.read_csv(ROOT / "systems.csv").drop_duplicates("system_id").set_index("system_id")
    name = str(systems.system_public_name.get(sid, f"system {sid}"))
    out = ROOT / "data" / "validation" / "PVDAQ" / str(sid)
    out.mkdir(parents=True, exist_ok=True)
    row = dict(system_id=sid, name=name)

    mfile = out / f"{sid}_metadata.json"
    meta = json.loads(mfile.read_text()) if from_cache and mfile.exists() else inv.metadata_json(sid)
    if meta is None:
        return {**row, "verdict": "FAIL", "note": "no metadata JSON"}
    (out / f"{sid}_metadata.json").write_text(json.dumps(meta, indent=1))
    info, checks = check_metadata(sid, name, meta)
    if info["lat"] is None and sid in systems.index:
        info.update(lat=inv.num(systems.site_latitude[sid]), lon=inv.num(systems.site_longitude[sid]),
                    elevation_m=inv.num(systems.site_elevation[sid]))
    print(f"{sid} {name}: metadata read", flush=True)

    ys = ({int(re.search(r"_raw_(\d+)", f.name).group(1)): {"parquet": None, "csv": None}
           for f in out.glob(f"{sid}_raw_*.parquet")} if from_cache else years_on_oedi(sid))
    if years:
        ys = {y: d for y, d in ys.items() if y in years or y == 0}
    if not ys:
        return {**row, "verdict": "FAIL", "note": "no time series on OEDI"}
    ids = {} if from_cache else ac_metrics(sid, ys)
    print(f"  years on OEDI: {sorted(ys)}; AC metrics: {ids or 'none found'}", flush=True)
    frames = []
    facts = {"counts": [], "dup": 0, "steps": []}
    for y, dirs in sorted(ys.items()):
        print(f"  extracting {y or 'all'}", flush=True)
        df = extract_year(sid, y, dirs, ids, out / f"{sid}_raw_{y or 'all'}.parquet")
        if len(df):
            tt = pd.DatetimeIndex(df["time"])
            if tt.tz is not None:
                tt = tt.tz_localize(None)
            facts["counts"].append(pd.Series(1, index=tt).resample("h").count())
            facts["dup"] += int(tt.duplicated().sum())
            facts["steps"].append(tt.sort_values().to_series().diff().median())
            df = df.drop(columns="time").set_axis(tt.floor(GRID), axis=0)
            df = df.apply(pd.to_numeric, errors="coerce").groupby(level=0).mean()
            frames.append(df.rename_axis("time").reset_index())
    if not frames:
        checks.append(("data", "AC power column", False, "no file with a time and an AC power column"))
        return {**row, "verdict": report(sid, info, checks, pd.DataFrame(), dict(ac_col="", tz=""), out),
                "note": "no AC data"}
    raw = pd.concat(frames, ignore_index=True)
    if years:
        raw = raw[raw["time"].dt.year.isin(years)]

    cnt = pd.concat(facts["counts"]).groupby(level=0).sum()
    info["_raw"] = dict(counts=cnt, dup=facts["dup"], step=pd.Series(facts["steps"]).median())
    hourly, dchecks, yt, d = check_data(raw, info, sid)
    file_note = write_hourly(hourly, out, sid)
    if drop_raw and file_note.startswith("parquet verified"):
        for f in out.glob(f"{sid}_raw_*.parquet"):
            f.unlink()
        file_note += "; raw cache deleted"
    verdict = report(sid, info, checks + dchecks, yt, d, out)
    best = yt.loc[yt.scorable_pct.idxmax()] if len(yt) and yt.scorable_pct.notna().any() else None
    lagc = next((e for a_, b_, c_, e in dchecks if b_.startswith("timestamp offset")), "")
    print(f"  verdict: {verdict}; {file_note}", flush=True)
    return {**row, "verdict": verdict, "lat": info["lat"], "lon": info["lon"], "dc_kw": info["dc_kw"],
            "dc_source": info["dc_source"], "tilt_az": info["planes"], "tz": d["tz"], "timestamp_check": lagc,
            "best_year": int(best.year) if best is not None else None,
            "best_scorable_pct": float(best.scorable_pct) if best is not None else None,
            "years_80pct": ",".join(str(int(y)) for y in yt.loc[yt.scorable_pct >= 80, "year"]) if len(yt) else "",
            "warn_fail": "; ".join(f"{STATUS[c_]}: {b_}" for a_, b_, c_, e in checks + dchecks if c_ is not True),
            "file": file_note}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("system_id", type=int)
    ap.add_argument("--years", nargs="+", type=int, help="years to extract (default: all on OEDI)")
    ap.add_argument("--drop-raw", action="store_true", help="delete the raw cache once the hourly parquet is verified")
    ap.add_argument("--from-cache", action="store_true", help="rebuild from the cached raw files, no OEDI access")
    a = ap.parse_args(argv)
    r = process(a.system_id, a.years, a.drop_raw, a.from_cache)
    print(f"verdict: {r['verdict']} -> {ROOT / 'data' / 'validation' / 'PVDAQ' / str(a.system_id)}")


if __name__ == "__main__":
    main()
