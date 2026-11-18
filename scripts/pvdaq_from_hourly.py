"""Bring an hourly measured series prepared earlier onto the PVDAQ extraction protocol.

    python scripts/pvdaq_from_hourly.py 34 1423 2107 4

For each system: reads the hourly series named in its site configuration ([measured]), puts
the stamps on fixed local standard time (hour start), and passes it through the same checks
and corrections as scripts/pvdaq_extract.py (units against the DC rating, dead loggers,
negative and impossible values, timestamp shift per month against clear sky, dark hours).
Writes data/validation/PVDAQ/<sid>/<sid>_hourly.parquet, <sid>_report.md and
<sid>_metadata.json (the PVDAQ JSON where one is stored, else location and dates from the
site configuration). The site configuration itself is not changed here.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solarmodel import paths  # noqa: E402

ROOT = paths.PROJECT_ROOT                  # the root bench resolves configs and measured files against
_spec = importlib.util.spec_from_file_location("ext", Path(__file__).parent / "pvdaq_extract.py")
ext = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ext)
from solarmodel import bench  # noqa: E402

SITES = {34: "LasVegas_34", 1423: "Henderson_1423", 2107: "Arbuckle_2107", 4: "Golden_4"}
META = {34: paths.CONFIG_DIR / "site" / "34_system_metadata.json", 2107: paths.CONFIG_DIR / "2107_system_metadata.json"}


def _meta(sid: int, cfg: dict, first: pd.Timestamp) -> dict:
    if sid in META and META[sid].exists():
        return json.loads(META[sid].read_text())
    loc, arr = cfg["location"], cfg["array"]
    return {"System": {"system_id": sid, "public_name": loc["name"], "power": str(cfg["system"]["rated_capacity_kw"]),
                       "started_on": f"{first:%Y-%m-%d %H:%M:%S}", "comments": "built from the site configuration"},
            "Site": {"public_name": loc["name"], "latitude": str(loc["lat"]), "longitude": str(loc["lon"]),
                     "elevation": str(loc["altitude"])},
            "Mount": {"Mount 0": {"tilt": str(arr["tilt"]), "azimuth": str(arr["azimuth"]),
                                  "tracking": "t" if arr.get("tracking") else "f"}},
            "Modules": {}, "Inverters": {}}


def convert(sid: int) -> str:
    name = SITES[sid]
    f = paths.CONFIG_DIR / "site" / f"{name}.toml"
    cfg = tomllib.loads(f.read_text())
    m = cfg["measured"]
    s = bench.read_measured(m)                                       # negatives kept: check_data corrects them
    half = {"start": 0, "end": -60, "center": -30}[m.get("timestamp_label", "start")]
    s.index = s.index + pd.Timedelta(minutes=half) + pd.Timedelta(hours=m.get("manual_shift_hours", 0))
    tz = ext.inv.standard_tz(cfg["location"]["lon"])
    s.index = s.index.tz_convert(tz).tz_localize(None)               # fixed local standard time, hour start
    first = s.first_valid_index()
    meta = _meta(sid, cfg, first)
    out = ROOT / "data" / "validation" / "PVDAQ" / str(sid)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{sid}_metadata.json").write_text(json.dumps(meta, indent=1))
    info, checks = ext.check_metadata(sid, name, meta)
    info.update(lat=cfg["location"]["lat"], lon=cfg["location"]["lon"], elevation_m=cfg["location"]["altitude"],
                dc_kw=float(cfg["system"]["rated_capacity_kw"]), dc_source="site configuration (rated_capacity_kw)")
    checks = [c for c in checks if c[1] != "DC rating"] + [("metadata", "DC rating", True, f"{info['dc_kw']} kW from the site configuration")]
    frame = pd.DataFrame({"time": s.index, "ac_power": s.to_numpy()})
    info["_raw"] = dict(counts=pd.Series(1, index=s.dropna().index).resample("h").count(), dup=0, step=pd.Timedelta("1h"))
    hourly, dchecks, yt, d = ext.check_data(frame, info, sid)
    note = ext.write_hourly(hourly, out, sid)
    verdict = ext.report(sid, info, checks + dchecks, yt, d, out)
    return f"{name}: {verdict}; {note}; tz {tz}"


if __name__ == "__main__":
    for sid in [int(x) for x in sys.argv[1:]] or SITES:
        print(convert(sid), flush=True)
