"""Check that each PVDAQ hourly parquet is usable on its own, before its raw cache is deleted.

    python scripts/pvdaq_verify_parquet.py              # every system with an hourly parquet
    python scripts/pvdaq_verify_parquet.py 50 1203

Reads only data/validation/PVDAQ/<sid>/<sid>_hourly.parquet, the system's metadata JSON and
config/pvdaq_overrides.json (never the raw cache or the extractor). Per system:
  schema       index 'timestamp' (naive datetime), columns pv_power_hourly_kW (float),
               raw_samples_in_hour (int), system_id (= sid)
  index        unique, increasing, exactly 1 h apart
  power        no negative or infinite values; max <= 1.3 x DC; p99.9 <= 1.05 x DC in every year
  consistency  power present only where raw_samples_in_hour > 0
  dark hours   0 where clear-sky GHI is 0 at the hour centre
  timestamps   best clear-sky lag 0 h, winter and summer, in every year with data
  bench        bench.load_measured reads it (fixed standard time, hour-start labels) and returns
               every valid hour, in UTC, without NaT
Writes outputs/pvdaq_inventory/parquet_check.csv.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solarmodel import paths  # noqa: E402

ROOT = paths.PROJECT_ROOT                  # the root bench resolves configs and measured files against
PVDAQ = ROOT / "data" / "validation" / "PVDAQ"
OUT = ROOT / "outputs" / "pvdaq_inventory" / "parquet_check.csv"
_spec = importlib.util.spec_from_file_location("inv", Path(__file__).parent / "pvdaq_sensor_inventory.py")
inv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inv)


def _dc(sid: int) -> float | None:
    ov = paths.CONFIG_DIR / "pvdaq_overrides.json"
    o = json.loads(ov.read_text()).get(str(sid), {}) if ov.exists() else {}
    if o.get("dc_kw"):
        return float(o["dc_kw"])
    s = ROOT / "outputs" / "pvdaq_inventory" / "extract_summary.csv"
    if s.exists():
        d = pd.read_csv(s).drop_duplicates("system_id", keep="last").set_index("system_id")
        if sid in d.index and pd.notna(d.dc_kw[sid]):
            return float(d.dc_kw[sid])
    return None


def check(sid: int) -> dict:
    from solarmodel import bench
    import pvlib

    f = PVDAQ / str(sid) / f"{sid}_hourly.parquet"
    meta = json.loads((PVDAQ / str(sid) / f"{sid}_metadata.json").read_text())
    site = inv.site_of(meta)
    lat, lon = float(site["latitude"]), float(site["longitude"])
    tz = inv.standard_tz(lon)
    dc = _dc(sid)
    res, fails = {"system_id": sid, "dc_kw": dc, "tz": tz}, []

    def ok(name, cond, detail=""):
        res[name] = "PASS" if cond else f"FAIL {detail}".strip()
        if not cond:
            fails.append(name)

    d = pd.read_parquet(f)
    p = d.get("pv_power_hourly_kW")
    ok("schema", d.index.name == "timestamp" and isinstance(d.index, pd.DatetimeIndex) and d.index.tz is None
       and list(d.columns) == ["pv_power_hourly_kW", "raw_samples_in_hour", "system_id"]
       and pd.api.types.is_float_dtype(p) and pd.api.types.is_integer_dtype(d.raw_samples_in_hour)
       and set(d.system_id.unique()) == {sid}, f"{d.index.name} {d.index.dtype} {list(d.columns)} {d.dtypes.to_dict()}")
    diffs = d.index.to_series().diff().dropna()
    ok("index", d.index.is_unique and d.index.is_monotonic_increasing and (diffs == pd.Timedelta(hours=1)).all(),
       f"unique {d.index.is_unique}, increasing {d.index.is_monotonic_increasing}, steps {diffs.value_counts().head(3).to_dict()}")
    v = p.dropna()
    q = (p.groupby(p.index.year).quantile(0.999) / dc)[p.groupby(p.index.year).count() > 1000] if dc else pd.Series(dtype=float)
    ok("power", bool(np.isfinite(v).all()) and (v >= 0).all() and (dc is None or (v.max() <= 1.3 * dc and (q <= 1.05).all())),
       f"min {v.min():.3f}, max {v.max():.2f} kW, DC {dc}, p99.9/DC above 1.05: {q[q > 1.05].round(2).to_dict()}")
    ok("consistency", not (p.notna() & (d.raw_samples_in_hour == 0)).any(),
       f"{int((p.notna() & (d.raw_samples_in_hour == 0)).sum())} hours with power but no samples")
    idx = d.index.tz_localize(tz) + pd.Timedelta(minutes=30)
    ghi = pd.Series(pvlib.location.Location(lat, lon, altitude=float(site.get("elevation") or 0)).get_clearsky(idx)["ghi"].to_numpy(), index=d.index)
    nz = int(((ghi == 0) & (p.fillna(0) != 0)).sum())
    ok("dark_hours", nz == 0, f"{nz} dark hours not 0")
    bad = []
    for y in sorted(set(p.index.year)):
        for season, months in inv.SEASONS:
            sel = (p.index.year == y) & p.index.month.isin(months)
            if p[sel].notna().sum() > 300:
                lag = bench.best_lag(p[sel], ghi[sel], 4)
                if lag:
                    bad.append(f"{y} {season} {lag:+d} h")
    ok("timestamps", not bad, "; ".join(bad))
    cfg = {"measured": {"file": str(f.relative_to(ROOT)), "time_col": "timestamp", "power_col": "pv_power_hourly_kW",
                        "power_unit": "kW", "tz": tz, "interval_minutes": 60, "timestamp_label": "start",
                        "manual_shift_hours": 0}}
    s0, s1 = p.first_valid_index(), p.last_valid_index()
    site_obj = bench.Site(name=str(sid), config=cfg, system=None, start=f"{s0:%Y-%m-%d}", end=f"{s1:%Y-%m-%d}",
                          capacity_kw=dc or 0.0)
    try:
        m = bench.load_measured(site_obj)
        n_valid = int(p.notna().sum())
        ok("bench", str(m.index.tz) == "UTC" and m.index.notna().all() and int(m.notna().sum()) == n_valid,
           f"{int(m.notna().sum())} of {n_valid} valid hours, tz {m.index.tz}")
    except Exception as e:
        ok("bench", False, f"{type(e).__name__}: {e}")
    res.update(first=f"{s0:%Y-%m-%d}", last=f"{s1:%Y-%m-%d}", valid_hours=int(p.notna().sum()),
               energy_mwh=round(float(v.sum()) / 1000, 2), verdict="PASS" if not fails else "FAIL: " + ", ".join(fails))
    return res


def main(argv=None):
    ids = [int(x) for x in (argv if argv is not None else sys.argv[1:])] or sorted(
        int(x.name) for x in PVDAQ.iterdir() if x.is_dir() and x.name.isdigit() and (x / f"{x.name}_hourly.parquet").exists())
    rows = []
    for sid in ids:
        try:
            r = check(sid)
        except Exception as e:
            r = {"system_id": sid, "verdict": f"ERROR {type(e).__name__}: {e}"}
        rows.append(r)
        print(f"{sid:>6}  {r['verdict']}" + "".join(f"\n        {k}: {v}" for k, v in r.items()
                                                       if isinstance(v, str) and v.startswith("FAIL") and k != "verdict"), flush=True)
    t = pd.DataFrame(rows)
    if OUT.exists() and len(t) and set(t.system_id) != set(pd.read_csv(OUT).system_id):   # a subset: merge
        old = pd.read_csv(OUT)
        t = pd.concat([old[~old.system_id.isin(t.system_id)], t]).sort_values("system_id")
    t.to_csv(OUT, index=False)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
