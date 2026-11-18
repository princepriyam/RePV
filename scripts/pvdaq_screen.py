"""Screen every public PVDAQ system for the validation set, from its OEDI metadata.

Reads the system list (systems.csv), fetches each system's metadata JSON with the
functions of pvdaq_sensor_inventory.py, reads the header of one of its data files, and
sorts it into a tier. Every rule uses metadata or the data header, none uses model error.

  tier A   fixed, one orientation, tilt and azimuth recorded, module recorded, AC power column
  tier B   as A with a surrogate module (module not recorded)
  tier M   as B but several orientations: usable if each orientation has its own AC column
  tier G   fixed, AC power column, but tilt or azimuth missing: geometry would have to be
           inferred from the measured output (evidence class C, not independent validation)
  excluded tracker, net meter, no data on OEDI, or no AC power column in the data

The AC check reads the data header, because the metadata JSON of the regular PVDAQ
systems has no metrics list (only the 2023 Data Prize systems have one).

    python scripts/pvdaq_screen.py                 # all systems in systems.csv
    python scripts/pvdaq_screen.py 4901 4902 1422  # selected systems
    -> outputs/pvdaq_inventory/screen.csv, screen.md

Needs network access to oedi-data-lake.s3.amazonaws.com, and s3fs for the year count.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solarmodel import paths  # noqa: E402

ROOT = paths.PROJECT_ROOT                  # the root bench resolves configs and measured files against
OUT = ROOT / "outputs" / "pvdaq_inventory"
_spec = importlib.util.spec_from_file_location("inv", Path(__file__).parent / "pvdaq_sensor_inventory.py")
inv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inv)

USED = {34: "LasVegas_34", 1423: "Henderson_1423", 2107: "Arbuckle_2107", 4: "Golden_4", 9069: "SimonSolar_9069"}
EXCLUDED = {1367: "timestamp lag at search bound", 1431: "tracker", 14200: "tracker", 14201: "tracker"}


def _n_years(s: str) -> int:
    m = re.search(r"\((\d+)\)", s)
    if m:
        return int(m.group(1))
    return 1 if s.startswith("data prize") else 0


def data_header(sid: int) -> list[str] | None:
    """Column names of the first data file of a system on OEDI (csv or parquet), or None."""
    fs = inv.fs
    if fs is None:
        return None
    roots = [f"{inv.BASE}/2023-solar-data-prize/{sid}_OEDI/data"] + \
            [f"{b}/system_id={sid}" for b in (f"{inv.BASE}/csv/pvdata", f"{inv.BASE}/parquet/pvdata", f"{inv.BASE}/pvdata")]

    def first_file(path, depth=0):
        if depth > 6:
            return None
        items = fs.ls(path, detail=True)
        files = sorted(i["name"] for i in items if i.get("type") == "file" and i["name"].endswith((".csv", ".parquet")))
        if files:
            return files[0]
        for d in sorted(i["name"] for i in items if i.get("type") == "directory"):
            f = first_file(d, depth + 1)
            if f:
                return f
        return None

    for root in roots:
        try:
            f = first_file(root)
        except (FileNotFoundError, OSError):
            continue
        if not f:
            continue
        try:
            if f.endswith(".parquet"):
                import pyarrow.parquet as pq
                with fs.open(f, "rb") as fh:
                    return list(pq.read_schema(fh).names)
            with fs.open(f, "r") as fh:
                return [c.strip().strip('"') for c in fh.readline().split(",")]
        except Exception as e:
            print(f"  {sid}: cannot read header of {f}: {e}", file=sys.stderr)
    return None


def screen(sid: int, name: str, lat, lon, elev) -> dict:
    row = dict(system_id=sid, name=name, lat=lat, lon=lon, elevation_m=elev)
    if sid in USED:
        row["status"] = f"in assessment ({USED[sid]})"
    meta = inv.metadata_json(sid)
    if meta is None:
        return {**row, "status": row.get("status", "excluded: no metadata JSON")}

    mounts = list((meta.get("Mount") or {}).values())
    mods = list((meta.get("Modules") or {}).values())
    invs = list((meta.get("Inverters") or {}).values())
    metrics = [m.get("common_name") for m in (meta.get("Metrics") or {}).values()]
    tracks = [inv.tracking(m.get("tracking")) for m in mounts]
    planes = {(inv.num(m.get("tilt")), inv.num(m.get("azimuth"))) for m in mounts}
    years = inv.years(sid)
    mod_known = inv.module_known(mods)
    row.update(
        n_mounts=len(mounts), tracking=";".join(str(t) for t in tracks),
        tilt=";".join(str(p[0]) for p in planes), azimuth=";".join(str(p[1]) for p in planes),
        module=inv.join(f"{m.get('manufacturer', '')} {m.get('model', '')} ({m.get('type', '')}, n={m.get('quantity', '')})" for m in mods),
        inverter=inv.join(f"{i.get('manufacturer', '')} {i.get('model', '')} x{i.get('quantity', '')}" for i in invs),
        poa=metrics.count("Irradiance POA"), t_module=metrics.count("Temperature module"), years_on_oedi=years,
    )
    header = data_header(sid)
    ac_cols = [c for c in (header or []) if inv.AC_COL.search(c)]
    row["ac_columns"] = ";".join(ac_cols) if header is not None else "header not read"

    hard, soft = [], []
    if any(t is True for t in tracks):
        hard.append("tracker")
    if "net meter" in name.lower():
        hard.append("net meter")
    if _n_years(years) < 1:
        hard.append("no data years found")
    if header is not None and not ac_cols:
        hard.append("no AC power column")
    if sid in EXCLUDED:
        hard.append(EXCLUDED[sid])
    if not mounts or any(t is None for t in tracks):
        soft.append("tracking not recorded")
    geom_missing = (not mounts) or any(p[0] is None or p[1] is None for p in planes)
    if geom_missing:
        soft.append("tilt or azimuth missing")
    if len(planes) > 1:
        soft.append(f"{len(planes)} orientations")
    if header is None:
        soft.append("AC column unverified")

    if "status" not in row:
        if hard:
            row["status"] = "excluded: " + ", ".join(hard)
        elif geom_missing:
            row["status"] = "tier G: geometry missing (class C if inferred)"
        elif len(planes) > 1:
            row["status"] = f"tier M: {len(planes)} orientations (needs one AC column per orientation)"
        else:
            row["status"] = "tier A: module recorded" if mod_known else "tier B: surrogate module"
    reasons = hard + soft
    row["checks"] = ", ".join(reasons)
    return row


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    systems = pd.read_csv(ROOT / "systems.csv").drop_duplicates("system_id")
    if argv:
        systems = systems[systems.system_id.isin([int(a) for a in argv])]
    rows = []
    for _, s in systems.iterrows():
        r = screen(int(s.system_id), str(s.system_public_name), s.site_latitude, s.site_longitude, s.site_elevation)
        rows.append(r)
        print(f"  {r['system_id']:>6} {r['name'][:55]:<55} {r['status']}", flush=True)
    df = pd.DataFrame(rows)
    order = {"in": 0, "tier A": 1, "tier B": 2, "tier M": 3, "tier G": 4}
    df["_k"] = df.status.map(lambda x: next((v for k, v in order.items() if x.startswith(k)), 5))
    df = df.sort_values(["_k", "system_id"]).drop(columns="_k")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "screen.csv", index=False)
    try:
        (OUT / "screen.md").write_text(df.to_markdown(index=False))
    except ImportError:
        (OUT / "screen.md").write_text(df.to_string(index=False))
    print(df.status.str.extract(r"^(in assessment|tier \w|excluded)")[0].value_counts().to_string())
    print(f"-> {OUT / 'screen.csv'}")


if __name__ == "__main__":
    main()
