"""Sensor inventory of the US validation sites from the PVDAQ metadata on OEDI.

For each system: which on-site sensors exist (POA, GHI/DNI/DHI, module and ambient
temperature, wind), the recorded module, inverter and mount, and which years of
time series are on OEDI. Decides which sites can get a U-value sensor fit
(studies/LasVegas_34/fit_thermal.py) and a measured-POA weather input.

Anonymous S3 read of the per-system metadata JSON (parquet tables as fallback);
needs: pip install pandas pyarrow s3fs (tabulate optional, for summary.md)

    python scripts/pvdaq_sensor_inventory.py
    -> outputs/pvdaq_inventory/{sensors.csv, summary.md}
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
try:
    import s3fs
except ImportError:
    s3fs = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solarmodel import paths  # noqa: E402

ROOT = paths.PROJECT_ROOT                  # the root bench resolves configs and measured files against
OUT = ROOT / "outputs" / "pvdaq_inventory"
BASE = "oedi-data-lake/pvdaq"
SITES = {  # config/site name -> PVDAQ system_id
    "LasVegas_34": 34, "Henderson_1423": 1423, "Arbuckle_2107": 2107, "Golden_4": 4,
    "Boulder_1431": 1431, "SimonSolar_9069": 9069, "SAS_SF1_14200": 14200,
    "Carolina_14201": 14201, "Nevada_1367": 1367,
}
# PVDAQ common_name -> column in the summary
GROUPS = {
    "Irradiance POA": "POA", "Irradiance GHI": "GHI", "Irradiance DNI": "DNI", "Irradiance DHI": "DHI",
    "Temperature module": "T_module", "Temperature ambient": "T_ambient", "Wind speed": "wind",
    "AC power": "AC_power",
}

fs = s3fs.S3FileSystem(anon=True) if s3fs else None
JSON_DIRS = [f"{BASE}/csv/system_metadata", f"{BASE}/csv/systems_metadata", f"{BASE}/csv/metadata",
             f"{BASE}/system_metadata", f"{BASE}/csv"]
PARQUET_BASES = [f"{BASE}/parquet", BASE]


def show_layout() -> None:
    """Print the top of the bucket so wrong paths are easy to fix."""
    if fs is None:
        return
    for p in (BASE, f"{BASE}/csv", f"{BASE}/parquet"):
        try:
            names = [x.rsplit("/", 1)[-1] for x in fs.ls(p)]
            print(f"  {p}/: {names[:40]}", file=sys.stderr)
        except Exception as e:
            print(f"  {p}/: {e}", file=sys.stderr)


HTTP_META = [  # regular PVDAQ systems, then the 2023 Solar Data Prize systems (e.g. 2107, 9069)
    "https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/system_metadata/{sid}_system_metadata.json",
    "https://oedi-data-lake.s3.amazonaws.com/pvdaq/2023-solar-data-prize/{sid}_OEDI/metadata/{sid}_system_metadata.json",
]


def metadata_json(sid: int) -> dict | None:
    """Per-system metadata JSON (Site, Mount, Inverters, Modules, Metrics), over plain HTTPS."""
    import json
    import urllib.error
    import urllib.request
    for local in (paths.CONFIG_DIR / f"{sid}_system_metadata.json", paths.CONFIG_DIR / "site" / f"{sid}_system_metadata.json"):
        if local.exists():                                  # copies added by hand take precedence
            return json.loads(local.read_text())
    for pattern in HTTP_META:
        url = pattern.format(sid=sid)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            print(f"  {sid}: HTTP {e.code} for {url}", file=sys.stderr)
        except Exception as e:
            print(f"  {sid}: {e} for {url}", file=sys.stderr)
    return None


def table(name: str) -> pd.DataFrame:
    if fs is None:
        return pd.DataFrame(columns=["system_id"])
    for base in PARQUET_BASES:
        files = fs.glob(f"{base}/{name}/*.parquet") or fs.glob(f"{base}/{name}/**/*.parquet")
        if files:
            df = pd.concat((pd.read_parquet(fs.open(f, "rb")) for f in files), ignore_index=True)
            if "system_id" in df:
                df["system_id"] = pd.to_numeric(df["system_id"], errors="coerce").astype("Int64")
            return df
    return pd.DataFrame(columns=["system_id"])


def years(sid: int) -> str:
    if fs is None:
        return "s3fs not installed"
    try:                                                   # Solar Data Prize systems keep their data in one folder
        prize = fs.ls(f"{BASE}/2023-solar-data-prize/{sid}_OEDI/data")
        if prize:
            return f"data prize folder ({len(prize)} files)"
    except (FileNotFoundError, OSError):
        pass
    for base in (f"{BASE}/parquet/pvdata", f"{BASE}/csv/pvdata", f"{BASE}/pvdata", "oedi-data-lake/pvdata"):
        try:
            ys = sorted(p.rsplit("=", 1)[-1] for p in fs.ls(f"{base}/system_id={sid}") if "year=" in p)
        except (FileNotFoundError, OSError):
            continue
        if ys:
            return f"{ys[0]}-{ys[-1]} ({len(ys)})"
    return "not found"


AC_COL = re.compile(r"ac_power|ac_pwr|inv\w*_ac_p", re.I)
SEASONS = (("winter", (11, 12, 1, 2)), ("summer", (5, 6, 7, 8)))     # timestamp checks against clear sky


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def tracking(v) -> bool | None:
    s = str(v).strip().lower()
    if s in ("true", "t", "1", "yes", "y"):
        return True
    if s in ("false", "f", "0", "no", "n"):
        return False
    return None


def module_known(mods) -> bool:
    """A module record with a manufacturer and a model."""
    blank = ("", "Unknown", "None", "nan")
    return any(str(m.get("manufacturer", "")).strip() not in blank and str(m.get("model", "")).strip() not in blank
               for m in mods)


def site_of(meta: dict) -> dict:
    """The Site block of a metadata JSON; some JSONs nest it: {"Site 0": {...}}."""
    site = meta.get("Site") if isinstance(meta.get("Site"), dict) else {}
    return site if "latitude" in site else next((v for v in site.values() if isinstance(v, dict)), {})


def standard_tz(lon: float) -> str:
    """Fixed local standard time (no DST) at a longitude; Etc/GMT+8 is UTC-8."""
    std = -int(round(lon / 15.0))
    return f"Etc/GMT+{std}" if std >= 0 else f"Etc/GMT-{-std}"


def join(vals) -> str:
    return "; ".join(sorted({str(v).strip() for v in vals if v is not None and str(v).strip() not in ("", "nan", "None")}))


def from_json(sid: int, meta: dict) -> tuple[list[dict], dict]:
    sensors = [dict(system_id=sid, **{k: v.get(k) for k in ("metric_id", "common_name", "sensor_name", "units")})
               for v in (meta.get("Metrics") or {}).values()]
    mods = (meta.get("Modules") or {}).values()
    invs = (meta.get("Inverters") or {}).values()
    mnts = list((meta.get("Mount") or {}).values())
    info = dict(module=join(f"{m.get('manufacturer', '')} {m.get('model', '')} ({m.get('type', '')}, n={m.get('quantity', '')})" for m in mods),
                inverter=join(f"{i.get('manufacturer', '')} {i.get('model', '')} x{i.get('quantity', '')}" for i in invs),
                mount=join(f"tracking={m.get('tracking')} tilt={m.get('tilt')} az={m.get('azimuth')} {m.get('type', '')}" for m in mnts))
    return sensors, info


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, sensors, missing = [], [], []
    for site, sid in SITES.items():
        meta = metadata_json(sid)
        if meta is None:
            missing.append((site, sid))
            continue
        s, info = from_json(sid, meta)
        sensors += s
        rows.append(dict(site=site, system_id=sid, source="json", **info))

    if missing:                                    # fall back to the parquet metadata tables
        print(f"no metadata JSON for {[m[0] for m in missing]}; trying parquet tables", flush=True)
        metrics, modules, inverters, mount = (table(n) for n in ("metrics", "modules", "inverters", "mount"))
        if metrics.empty and not rows:
            print("found neither metadata JSON nor parquet tables. Bucket layout:", file=sys.stderr)
            show_layout()
            sys.exit(1)
        for site, sid in missing:
            mm = metrics[metrics.system_id == sid]
            sensors += mm.assign(system_id=sid).to_dict("records")
            pick = lambda df, cols: df.loc[df.system_id == sid, [c for c in cols if c in df]].astype(str).agg(" ".join, axis=1) if not df.empty else []
            rows.append(dict(site=site, system_id=sid, source="parquet" if len(mm) else "not found",
                             module=join(pick(modules, ["manufacturer", "model", "type"])),
                             inverter=join(pick(inverters, ["manufacturer", "model"])),
                             mount=join(pick(mount, ["tracking", "tilt", "azimuth"]))))

    sens = pd.DataFrame(sensors)
    sens.to_csv(OUT / "sensors.csv", index=False)
    for r in rows:
        ss = sens[sens.system_id == r["system_id"]] if not sens.empty else sens
        for cname, col in GROUPS.items():
            r[col] = int((ss.get("common_name", pd.Series(dtype=str)) == cname).sum())
        r["years_on_oedi"] = years(r["system_id"])
        print(f"  {r['site']}: {r['POA']} POA, {r['GHI']} GHI, {r['DNI']} DNI, {r['T_module']} module T, "
              f"{r['wind']} wind, years {r['years_on_oedi']} [{r['source']}]", flush=True)
    cols = ["site", "system_id", "POA", "GHI", "DNI", "DHI", "T_module", "T_ambient", "wind", "AC_power",
            "years_on_oedi", "module", "inverter", "mount", "source"]
    s = pd.DataFrame(rows)[cols]
    s.to_csv(OUT / "summary.csv", index=False)
    try:
        (OUT / "summary.md").write_text(s.to_markdown(index=False))
    except ImportError:
        (OUT / "summary.md").write_text(s.to_string(index=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
